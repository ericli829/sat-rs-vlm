param(
    [Parameter(Mandatory = $true)]
    [string]$PythonPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$runner = Join-Path $repoRoot "taskgraph_lab\tools\run_stratified_10.ps1"
$logRoot = Join-Path $repoRoot "taskgraph_lab\outputs\logs"
$stdoutLog = Join-Path $logRoot "deepseek_v4_flash_stratified_10.stdout.log"
$stderrLog = Join-Path $logRoot "deepseek_v4_flash_stratified_10.stderr.log"
$pidPath = Join-Path $logRoot "deepseek_v4_flash_stratified_10.pid"
$key = Read-Host "DeepSeek API key" -MaskInput

if ([string]::IsNullOrWhiteSpace($key)) {
    throw "API key must not be empty"
}

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$env:TASKGRAPH_TEACHER_API_KEY = $key
try {
    $shellPath = (Get-Process -Id $PID).Path
    $arguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $runner,
        "-PythonPath", $PythonPath
    )
    $process = Start-Process `
        -FilePath $shellPath `
        -ArgumentList $arguments `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -PassThru
    [System.IO.File]::WriteAllText($pidPath, [string]$process.Id)
    Write-Output "[OK] PID=$($process.Id)"
    Write-Output "[INFO] STDOUT=$stdoutLog"
    Write-Output "[INFO] STDERR=$stderrLog"
}
finally {
    Remove-Item Env:TASKGRAPH_TEACHER_API_KEY -ErrorAction SilentlyContinue
    $key = $null
}
