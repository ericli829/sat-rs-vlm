param(
    [Parameter(Mandatory = $true)][string]$RunDirectory,
    [string]$Python = 'C:\Users\Ericoneabc\AppData\Local\Microsoft\WindowsApps\python.exe',
    [switch]$ValidateOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$run = (Resolve-Path -LiteralPath $RunDirectory).Path
$planPath = Join-Path $run 'test-plan.json'
$plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
$source = Get-Content -LiteralPath $plan.source_state -Raw | ConvertFrom-Json
$utf8 = New-Object System.Text.UTF8Encoding($false)
$sourceStopped = $false
$restoredPid = $null
$failure = $null
$restoreFailure = $null
$worker = $null
$env:PYTHONIOENCODING = 'utf-8'

function Write-State([string]$Stage, [hashtable]$Extra = @{}) {
    $record = @{stage = $Stage; updated_at = [DateTime]::UtcNow.ToString('o'); pid = $PID;
        progress_file = (Join-Path $plan.output 'status.json'); report = (Join-Path $plan.output 'report.html')}
    foreach ($key in $Extra.Keys) { $record[$key] = $Extra[$key] }
    $path = Join-Path $run 'orchestration.status.json'
    $temporary = $path + '.tmp'
    [System.IO.File]::WriteAllText($temporary, ($record | ConvertTo-Json -Depth 10), $utf8)
    for ($attempt = 0; $attempt -lt 6; $attempt++) {
        try {
            if (Test-Path -LiteralPath $path) { [System.IO.File]::Replace($temporary, $path, ($path + '.previous')) }
            else { [System.IO.File]::Move($temporary, $path) }
            break
        }
        catch { if ($attempt -eq 5) { throw }; Start-Sleep -Milliseconds (50 * [Math]::Pow(2, $attempt)) }
    }
    [Console]::WriteLine(($record | ConvertTo-Json -Depth 10 -Compress))
}

function Assert-OriginalService {
    $taskProcess = Get-Process -Id $source.pid -ErrorAction Stop
    $expected = ([DateTimeOffset]$source.started_at).UtcDateTime
    if ([Math]::Abs(($taskProcess.StartTime.ToUniversalTime() - $expected).TotalSeconds) -gt 10) {
        throw 'Original service PID identity changed'
    }
    $config = Get-Content -LiteralPath $source.config_file -Raw | ConvertFrom-Json
    if (-not $config.model.adapter_path -or $config.model.adapter_path -ne $plan.model.adapter_path -or $config.model.model_id -ne $plan.model.model_id) {
        throw 'Experiment must use the original base model and LoRA adapter'
    }
    $endpoint = [Uri]$source.endpoint
    $listener = Get-NetTCPConnection -LocalPort $endpoint.Port -State Listen -ErrorAction Stop
    if (@($listener | Where-Object { $_.OwningProcess -ne $source.pid }).Count -ne 0) {
        throw 'Original port has an unexpected owner'
    }
}

function Restore-Service {
    $endpoint = [Uri]$source.endpoint
    if (Get-NetTCPConnection -LocalPort $endpoint.Port -State Listen -ErrorAction SilentlyContinue) {
        throw 'Original port is occupied; refusing to replace another listener'
    }
    $oldKey = $env:MODEL_API_KEY
    $oldConfig = $env:COMPETITION_LOCAL_CONFIG
    $oldPythonPath = $env:PYTHONPATH
    $oldEncoding = $env:PYTHONIOENCODING
    try {
        $env:MODEL_API_KEY = (Get-Content -LiteralPath $source.key_file -Raw).Trim()
        $env:COMPETITION_LOCAL_CONFIG = $source.config_file
        $env:PYTHONPATH = Join-Path $repo 'src'
        if ($oldPythonPath) { $env:PYTHONPATH += ";$oldPythonPath" }
        $env:PYTHONIOENCODING = 'utf-8'
        $stdout = Join-Path $run 'restore.stdout.log'
        $stderr = Join-Path $run 'restore.stderr.log'
        $arguments = @('-u', '-m', 'uvicorn', 'sat_rs_vlm.competition.protocol_v2.local_server:create_local_app',
            '--factory', '--host', '127.0.0.1', '--port', $endpoint.Port, '--workers', '1', '--no-access-log')
        $taskProcess = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $repo `
            -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
        $restored = $source.PSObject.Copy()
        $restored.pid = $taskProcess.Id
        $restored.started_at = $taskProcess.StartTime.ToUniversalTime().ToString('o')
        $restored.stdout = $stdout
        $restored.stderr = $stderr
        [System.IO.File]::WriteAllText($plan.source_state, ($restored | ConvertTo-Json -Depth 10), $utf8)
        $directory = Split-Path -Parent $plan.source_state
        [System.IO.File]::WriteAllText((Join-Path $directory 'server.pid'), [string]$taskProcess.Id, $utf8)
        Write-State 'restoring_original_service' @{restored_service_pid = $taskProcess.Id}
        $ready = $false
        for ($attempt = 0; $attempt -lt 120; $attempt++) {
            try {
                $health = Invoke-RestMethod -Uri ($source.endpoint + '/healthz') `
                    -Headers @{Authorization = ('Bearer ' + $env:MODEL_API_KEY)} -TimeoutSec 2
                if ($health.status -eq 'ready') { $ready = $true; break }
            }
            catch { }
            Start-Sleep -Seconds 1
        }
        if (-not $ready) { throw 'Original service did not become ready before the deadline' }
        $preparationPath = Join-Path $directory 'preparation.status.json'
        if (Test-Path -LiteralPath $preparationPath) {
            $preparation = Get-Content -LiteralPath $preparationPath -Raw | ConvertFrom-Json
            $preparation | Add-Member -NotePropertyName server_pid -NotePropertyValue $taskProcess.Id -Force
            $preparation | Add-Member -NotePropertyName updated_at -NotePropertyValue ([DateTime]::UtcNow.ToString('o')) -Force
            [System.IO.File]::WriteAllText($preparationPath, ($preparation | ConvertTo-Json -Depth 10), $utf8)
        }
        return $taskProcess.Id
    }
    finally {
        $env:MODEL_API_KEY = $oldKey
        $env:COMPETITION_LOCAL_CONFIG = $oldConfig
        $env:PYTHONPATH = $oldPythonPath
        $env:PYTHONIOENCODING = $oldEncoding
    }
}

if ($ValidateOnly) {
    Assert-OriginalService
    if (-not (Test-Path -LiteralPath $plan.worker_script -PathType Leaf)) { throw 'Experiment worker missing' }
    @{validated = $true; original_service_pid = $source.pid; model_variant = 'with_lora'; planned_calls = $plan.planned_calls} | ConvertTo-Json -Depth 5
    exit 0
}

try {
    if (Test-Path -LiteralPath $plan.cancel_file) { throw 'Cancellation requested' }
    Assert-OriginalService
    Write-State 'pausing_original_service' @{original_service_pid = $source.pid}
    $taskProcess = Get-Process -Id $source.pid -ErrorAction Stop
    Stop-Process -Id $source.pid -ErrorAction Stop
    $sourceStopped = $true
    if (-not $taskProcess.WaitForExit(10000)) { throw 'Original model did not exit' }
    $workerArguments = @('-u', ('"{0}"' -f $plan.worker_script), '--plan', ('"{0}"' -f $planPath))
    $worker = Start-Process -FilePath $Python -ArgumentList $workerArguments -WorkingDirectory $repo `
        -WindowStyle Hidden -RedirectStandardOutput (Join-Path $run 'worker.stdout.log') `
        -RedirectStandardError (Join-Path $run 'worker.stderr.log') -PassThru
    # Windows PowerShell 5.1 can lose the child exit code if the native handle
    # was never opened before waiting for the process to terminate.
    $workerHandle = $worker.Handle
    # Wait only inside this detached supervisor. Native Python stderr is kept in
    # its own log rather than interpreted as PowerShell terminating errors.
    $worker.Id | Set-Content -LiteralPath (Join-Path $run 'worker.pid') -Encoding ascii
    Write-State 'testing_caption_prompt_then_injected_constraints' @{planned_calls = $plan.planned_calls; worker_pid = $worker.Id}
    $worker.WaitForExit()
    $worker.Refresh()
    $workerExitCode = $worker.ExitCode
    if ($null -eq $workerExitCode) { throw 'Could not read experiment exit code; verify its completion artifacts' }
    if ($workerExitCode -ne 0) { throw "Experiment exited with code $workerExitCode; inspect worker.stderr.log" }
}
catch { $failure = $_.Exception.Message; [Console]::WriteLine($failure) }
finally {
    # A supervisor/logging failure must not restore another GPU model while
    # the experiment child is still alive. Ask it to finish cooperatively first.
    if ($worker) {
        $worker.Refresh()
        if (-not $worker.HasExited) {
            [System.IO.File]::WriteAllText($plan.cancel_file, 'Supervisor interrupted', $utf8)
            $worker.WaitForExit()
        }
    }
    if ($sourceStopped) {
        try { $restoredPid = Restore-Service }
        catch { $restoreFailure = $_.Exception.Message }
    }
}
if ($restoreFailure) { Write-State 'restore_failed' @{error = $restoreFailure; test_error = $failure}; throw $restoreFailure }
if (Test-Path -LiteralPath $plan.cancel_file) { Write-State 'cancelled_original_service_restored' @{restored_service_pid = $restoredPid}; exit 0 }
if ($failure) { Write-State 'failed' @{error = $failure; restored_service_pid = $restoredPid}; exit 1 }
Write-State 'complete_original_service_restored' @{restored_service_pid = $restoredPid}
