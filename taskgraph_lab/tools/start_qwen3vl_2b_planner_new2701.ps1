param(
    [string]$PythonPath = "C:\Users\Ericoneabc\AppData\Local\Microsoft\WindowsApps\python.exe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runName = "qwen3vl_2b_planner_lora_new2701_$stamp"
$outputDir = Join-Path $repoRoot "taskgraph_lab\outputs\training\$runName"
$logDir = Join-Path $repoRoot "taskgraph_lab\outputs\logs"
$stdoutPath = Join-Path $logDir "$runName.stdout.log"
$stderrPath = Join-Path $logDir "$runName.stderr.log"
$pidPath = Join-Path $logDir "$runName.pid"
$configPath = Join-Path $repoRoot "taskgraph_lab\configs\qwen3vl_2b_planner_lora_new2701_local.yaml"

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$arguments = @(
    "-u",
    "-m",
    "taskgraph_lab.tools.train_qwen3vl_planner",
    "--config",
    $configPath,
    "--output-dir",
    $outputDir
)
$process = Start-Process `
    -FilePath $PythonPath `
    -ArgumentList $arguments `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru

Set-Content -LiteralPath $pidPath -Value $process.Id -Encoding ascii
Write-Output "[OK] PID=$($process.Id)"
Write-Output "[INFO] RUN=$outputDir"
Write-Output "[INFO] STDOUT=$stdoutPath"
Write-Output "[INFO] STDERR=$stderrPath"
Write-Output "[INFO] PIDFILE=$pidPath"
