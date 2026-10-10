param(
    [Parameter(Mandatory = $true)][string]$RunDirectory,
    [string]$Python = 'C:\Users\Ericoneabc\AppData\Local\Microsoft\WindowsApps\python.exe',
    [switch]$ValidateOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$run = (Resolve-Path -LiteralPath $RunDirectory).Path
$plan = Get-Content -LiteralPath (Join-Path $run 'test-plan.json') -Raw | ConvertFrom-Json
$sourceStatePath = $plan.source_state
$sourceState = Get-Content -LiteralPath $sourceStatePath -Raw | ConvertFrom-Json
$cancelFile = Join-Path $run 'cancel.flag'
$comparison = Join-Path $run 'comparison'
$baseState = $null
$sourceStopped = $false
$failure = $null
$restoreFailure = $null
$restoredPid = $null
$utf8 = New-Object System.Text.UTF8Encoding($false)

function Write-TaskState([string]$Stage, [hashtable]$Extra = @{}) {
    $record = @{
        stage = $Stage
        updated_at = [DateTime]::UtcNow.ToString('o')
        pid = $PID
        model_variant = 'base_only'
        progress_file = (Join-Path $comparison 'status.json')
        report = (Join-Path $comparison 'report.html')
    }
    foreach ($key in $Extra.Keys) { $record[$key] = $Extra[$key] }
    $path = Join-Path $run 'orchestration.status.json'
    $temporary = $path + '.tmp'
    [System.IO.File]::WriteAllText($temporary, ($record | ConvertTo-Json -Depth 10), $utf8)
    for ($attempt = 0; $attempt -lt 6; $attempt++) {
        try {
            if (Test-Path -LiteralPath $path) {
                # Windows PowerShell 5.1 converts a null string argument to an
                # empty path. Use a real backup path for the previous state.
                [System.IO.File]::Replace($temporary, $path, ($path + '.previous'))
            }
            else { [System.IO.File]::Move($temporary, $path) }
            break
        }
        catch {
            if ($attempt -eq 5) { throw }
            Start-Sleep -Milliseconds (50 * [Math]::Pow(2, $attempt))
        }
    }
    [Console]::WriteLine(($record | ConvertTo-Json -Depth 10 -Compress))
}

function Stop-OwnedService($State, [bool]$IsOriginal = $false) {
    $taskProcess = Get-Process -Id $State.pid -ErrorAction SilentlyContinue
    if (-not $taskProcess) { return }
    $expected = ([DateTimeOffset]$State.started_at).UtcDateTime
    $actual = $taskProcess.StartTime.ToUniversalTime()
    if ([Math]::Abs(($actual - $expected).TotalSeconds) -gt 10) {
        throw 'Service PID identity changed; refusing to stop an unrelated process'
    }
    Stop-Process -Id $State.pid -ErrorAction Stop
    if ($IsOriginal) { $script:sourceStopped = $true }
    if (-not $taskProcess.WaitForExit(10000)) { throw 'Service did not exit after stopping' }
}

function Wait-ServiceReady($State, [bool]$HonorCancel) {
    $secret = (Get-Content -LiteralPath $State.key_file -Raw).Trim()
    for ($attempt = 0; $attempt -lt 120; $attempt++) {
        if ($HonorCancel -and (Test-Path -LiteralPath $cancelFile)) { throw 'Cancellation requested' }
        try {
            $health = Invoke-RestMethod -Uri ($State.endpoint + '/healthz') `
                -Headers @{Authorization = "Bearer $secret"} -TimeoutSec 2
            if ($health.status -eq 'ready') { return }
        }
        catch { }
        Start-Sleep -Seconds 1
    }
    throw 'Service did not become ready before the startup deadline'
}

function Stop-SupersededPreparation {
    $directory = Split-Path -Parent $sourceStatePath
    $jobFile = Join-Path $directory 'job.json'
    $preparationFile = Join-Path $directory 'preparation.status.json'
    if (-not (Test-Path -LiteralPath $jobFile) -or -not (Test-Path -LiteralPath $preparationFile)) { return }
    $legacy = Get-Content -LiteralPath $jobFile -Raw | ConvertFrom-Json
    $latest = Get-Content -LiteralPath $preparationFile -Raw | ConvertFrom-Json
    if ($latest.job_pid -eq $legacy.pid -or $latest.stage -ne 'ready') { return }
    $taskProcess = Get-Process -Id $legacy.pid -ErrorAction SilentlyContinue
    if (-not $taskProcess) { return }
    $expected = ([DateTimeOffset]$legacy.started_at).UtcDateTime
    if ([Math]::Abs(($taskProcess.StartTime.ToUniversalTime() - $expected).TotalSeconds) -gt 10) {
        throw 'Superseded preparation PID identity changed'
    }
    $details = Get-CimInstance Win32_Process -Filter ("ProcessId=" + $legacy.pid)
    if (-not $details.CommandLine.Contains('prepare_public_protocol_v2.py') -or `
        -not $details.CommandLine.Contains($directory)) {
        throw 'Superseded preparation command does not match the original service'
    }
    Write-TaskState 'stopping_superseded_preparation' @{preparation_pid = $legacy.pid}
    Stop-Process -Id $legacy.pid -ErrorAction Stop
}

function Restore-OriginalService {
    $originalConfig = Get-Content -LiteralPath $sourceState.config_file -Raw | ConvertFrom-Json
    if (-not $originalConfig.model.adapter_path) { throw 'Original LoRA configuration is missing' }
    $endpoint = [Uri]$sourceState.endpoint
    if (Get-NetTCPConnection -LocalPort $endpoint.Port -State Listen -ErrorAction SilentlyContinue) {
        throw 'Original service port is occupied; refusing to replace its listener'
    }
    $oldKey = $env:MODEL_API_KEY
    $oldConfig = $env:COMPETITION_LOCAL_CONFIG
    $oldPythonPath = $env:PYTHONPATH
    $oldEncoding = $env:PYTHONIOENCODING
    try {
        $env:MODEL_API_KEY = (Get-Content -LiteralPath $sourceState.key_file -Raw).Trim()
        $env:COMPETITION_LOCAL_CONFIG = $sourceState.config_file
        $env:PYTHONPATH = Join-Path $repo 'src'
        if ($oldPythonPath) { $env:PYTHONPATH += ";$oldPythonPath" }
        $env:PYTHONIOENCODING = 'utf-8'
        $stdout = Join-Path $run 'restore.stdout.log'
        $stderr = Join-Path $run 'restore.stderr.log'
        $arguments = @(
            '-u', '-m', 'uvicorn',
            'sat_rs_vlm.competition.protocol_v2.local_server:create_local_app',
            '--factory', '--host', '127.0.0.1', '--port', $endpoint.Port,
            '--workers', '1', '--no-access-log'
        )
        $taskProcess = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $repo `
            -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
        $restored = $sourceState.PSObject.Copy()
        $restored.pid = $taskProcess.Id
        $restored.started_at = $taskProcess.StartTime.ToUniversalTime().ToString('o')
        $restored.stdout = $stdout
        $restored.stderr = $stderr
        [System.IO.File]::WriteAllText($sourceStatePath, ($restored | ConvertTo-Json -Depth 10), $utf8)
        [System.IO.File]::WriteAllText((Join-Path (Split-Path -Parent $sourceStatePath) 'server.pid'), [string]$taskProcess.Id, $utf8)
        Write-TaskState 'restoring_original_service' @{restored_service_pid = $taskProcess.Id}
        Wait-ServiceReady $restored $false
        $preparationFile = Join-Path (Split-Path -Parent $sourceStatePath) 'preparation.status.json'
        if (Test-Path -LiteralPath $preparationFile) {
            $preparation = Get-Content -LiteralPath $preparationFile -Raw | ConvertFrom-Json
            $preparation | Add-Member -NotePropertyName server_pid -NotePropertyValue $taskProcess.Id -Force
            $preparation | Add-Member -NotePropertyName updated_at -NotePropertyValue ([DateTime]::UtcNow.ToString('o')) -Force
            [System.IO.File]::WriteAllText($preparationFile, ($preparation | ConvertTo-Json -Depth 10), $utf8)
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
    $originalConfig = Get-Content -LiteralPath $sourceState.config_file -Raw | ConvertFrom-Json
    if (-not $originalConfig.model.adapter_path) { throw 'Expected an original LoRA service' }
    if (-not (Test-Path -LiteralPath $plan.human_review -PathType Leaf)) { throw 'Human review missing' }
    if (-not (Test-Path -LiteralPath $plan.source_capture -PathType Leaf)) { throw 'Source capture missing' }
    if (-not (Test-Path -LiteralPath $plan.baseline_results -PathType Leaf)) { throw 'Baseline missing' }
    if (-not $plan.source_run -or -not $plan.model_package) { throw 'Source run or package missing' }
    if ($plan.model.model_id -ne $originalConfig.model.model_id) { throw 'Base model must match original model' }
    if ($null -ne $plan.model.adapter_path) { throw 'Base experiment must have a null adapter' }
    $taskProcess = Get-Process -Id $sourceState.pid -ErrorAction Stop
    $expected = ([DateTimeOffset]$sourceState.started_at).UtcDateTime
    if ([Math]::Abs(($taskProcess.StartTime.ToUniversalTime() - $expected).TotalSeconds) -gt 10) {
        throw 'Original service identity changed'
    }
    @{validated = $true; original_service_pid = $sourceState.pid; model_variant = 'base_only'; planned_calls = 250} |
        ConvertTo-Json -Depth 10
    exit 0
}

try {
    if (Test-Path -LiteralPath $cancelFile) { throw 'Cancellation requested' }
    if (Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue) {
        throw 'Base test port 8001 is already occupied'
    }
    Stop-SupersededPreparation
    Write-TaskState 'pausing_original_service' @{original_service_pid = $sourceState.pid}
    Stop-OwnedService $sourceState $true
    $sourceStopped = $true
    Write-TaskState 'starting_base_service'
    $baseDirectory = Join-Path $run 'base-server'
    $baseJson = & (Join-Path $PSScriptRoot 'start_local_protocol_v2.ps1') `
        -Mode qwen2b -NoAdapter -Port 8001 -RunDirectory $baseDirectory `
        -ImagePackage $plan.model_package -ModelPath $plan.model.model_id -Python $Python
    $baseState = ($baseJson -join "`n") | ConvertFrom-Json
    $baseConfig = Get-Content -LiteralPath $baseState.config_file -Raw | ConvertFrom-Json
    if ($null -ne $baseConfig.model.adapter_path) { throw 'Base test must not load an adapter' }
    Write-TaskState 'loading_base_model' @{base_service_pid = $baseState.pid}
    Wait-ServiceReady $baseState $true
    Write-TaskState 'testing_base_model' @{base_service_pid = $baseState.pid; planned_calls = 250}
    $testScript = Join-Path $PSScriptRoot 'repeat_change_region.py'
    & $Python -u $testScript --state (Join-Path $baseDirectory 'state.json') `
        --capture $plan.source_capture --source-run $plan.source_run `
        --score $plan.source_score --package $plan.model_package `
        --human-review $plan.human_review --baseline-results $plan.baseline_results `
        --rounds 3 --collect-baseline --cancel-file $cancelFile --output $comparison
    if ($LASTEXITCODE -ne 0) { throw "Base comparison failed with exit code $LASTEXITCODE" }
}
catch {
    $failure = $_.Exception.Message
    Write-Output ("Comparison interrupted: " + $failure)
}
finally {
    try {
        if ($baseState) {
            Write-TaskState 'stopping_base_service'
            Stop-OwnedService $baseState
        }
        if ($sourceStopped) { $restoredPid = Restore-OriginalService }
    }
    catch {
        $restoreFailure = $_.Exception.Message
    }
}
if ($restoreFailure) {
    Write-TaskState 'restore_failed' @{error = $restoreFailure; test_error = $failure}
    throw $restoreFailure
}
if (Test-Path -LiteralPath $cancelFile) {
    Write-TaskState 'cancelled_original_service_restored' @{restored_service_pid = $restoredPid}
    exit 0
}
if ($failure) {
    Write-TaskState 'test_failed_original_service_restored' @{error = $failure; restored_service_pid = $restoredPid}
    exit 1
}
Write-TaskState 'complete_original_service_restored' @{restored_service_pid = $restoredPid}
