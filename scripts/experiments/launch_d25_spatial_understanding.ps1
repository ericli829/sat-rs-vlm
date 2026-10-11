param([switch]$PrepareOnly)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$RepoDir = 'D:\Desktop\tzb-2026\sat-rs-vlm'
$PythonExe = 'C:\Users\Ericoneabc\AppData\Local\Microsoft\WindowsApps\python.exe'
$ExperimentScript = Join-Path $RepoDir 'scripts\experiments\run_d25_spatial_understanding_qwen.py'
$CaseDir = Join-Path $RepoDir 'reports\training\changechat_grid_ab_20261011\competition_dev_d25_20261011'
$OutputDir = Join-Path $RepoDir 'reports\experiments\d25_spatial_understanding_20261011\run_v1'
$BaseDir = 'D:\Desktop\tzb-2026\Qwen3-VL-2B-Instruct'
$AdapterFile = Join-Path $RepoDir 'reports\training\changechat_grid_ab_20261011\downloads\route_b_stage2_batch8\adapter_model.safetensors'
$AdapterConfig = Join-Path $RepoDir 'reports\training\changechat_grid_ab_20261011\bundle\adapters\route_a\adapter_config.json'
$RequestsFile = Join-Path $CaseDir 'requests_d25.jsonl'
$ReviewFile = Join-Path $CaseDir 'human_review_reference.json'

foreach ($Required in @($PythonExe, $ExperimentScript, $AdapterFile, $AdapterConfig, $RequestsFile, $ReviewFile, (Join-Path $BaseDir 'config.json'))) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) { throw "Missing required file: $Required" }
}

$ScriptArgs = @(
    $ExperimentScript, '--repo', $RepoDir, '--requests', $RequestsFile,
    '--image-root', $CaseDir, '--human-review', $ReviewFile,
    '--route-b-adapter', $AdapterFile, '--route-a-adapter-config', $AdapterConfig,
    '--base-model-dir', $BaseDir, '--output-dir', $OutputDir,
    '--cpu-threads', '8', '--min-free-gpu-mb', '5200'
)
if ($PrepareOnly) {
    & $PythonExe @ScriptArgs --prepare-only
    if ($LASTEXITCODE -ne 0) { throw "Preflight failed with exit code $LASTEXITCODE" }
    exit 0
}

$PreflightFile = Join-Path $OutputDir 'preflight.json'
if (-not (Test-Path -LiteralPath $PreflightFile -PathType Leaf)) { throw 'Run -PrepareOnly before launching inference.' }
$Preflight = Get-Content -LiteralPath $PreflightFile -Raw | ConvertFrom-Json
if (-not $Preflight.passed) { throw 'Preflight did not pass.' }
$CurrentHash = (Get-FileHash -LiteralPath $ExperimentScript -Algorithm SHA256).Hash.ToLowerInvariant()
if ($CurrentHash -ne $Preflight.script_sha256) { throw 'Experiment script changed after preflight; rerun -PrepareOnly.' }
if ((Test-Path -LiteralPath (Join-Path $OutputDir 'predictions.json'))) { throw 'Completed predictions already exist; refusing to overwrite.' }

$LogDir = Join-Path $OutputDir 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$PidFile = Join-Path $LogDir 'spatial_experiment.pid'
if (Test-Path -LiteralPath $PidFile) {
    $PreviousId = [int](Get-Content -LiteralPath $PidFile -Raw)
    $PreviousProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $PreviousId"
    if ($PreviousProcess -and $PreviousProcess.CommandLine -like '*run_d25_spatial_understanding_qwen.py*') {
        throw "Experiment is already running as PID $PreviousId"
    }
}
$StdoutFile = Join-Path $LogDir 'spatial_experiment.stdout.log'
$StderrFile = Join-Path $LogDir 'spatial_experiment.stderr.log'
$ScriptArgs += '--run-model'
$QuotedArgs = foreach ($Argument in $ScriptArgs) {
    if ($Argument.Contains('"')) { throw 'Embedded quotes are not supported in launcher arguments.' }
    '"' + $Argument + '"'
}
$Started = Start-Process -FilePath $PythonExe -ArgumentList ($QuotedArgs -join ' ') -WorkingDirectory $RepoDir -WindowStyle Hidden -PassThru -RedirectStandardOutput $StdoutFile -RedirectStandardError $StderrFile
$ProcessId = $Started.Id
Set-Content -LiteralPath $PidFile -Value $ProcessId -NoNewline
$LaunchRecord = @{pid=$ProcessId; script=$ExperimentScript; started_at=(Get-Date).ToString('o'); stdout=$StdoutFile; stderr=$StderrFile; output=$OutputDir}
$LaunchRecord | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $LogDir 'launch.json') -Encoding utf8
if ($Started.WaitForExit(1500)) {
    Get-Content -LiteralPath $StderrFile -Tail 20
    throw "Experiment exited during startup with code $($Started.ExitCode)"
}
Write-Output "STARTED PID=$ProcessId"
Write-Output "STDOUT=$StdoutFile"
Write-Output "STDERR=$StderrFile"
Write-Output "PIDFILE=$PidFile"
