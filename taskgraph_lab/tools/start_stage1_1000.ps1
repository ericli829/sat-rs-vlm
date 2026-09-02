param(
    [Parameter(Mandatory = $true)]
    [string]$PythonPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$runner = Join-Path $repoRoot "taskgraph_lab\tools\run_stage1_1000.ps1"
$logRoot = Join-Path $repoRoot "taskgraph_lab\outputs\logs"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runRoot = Join-Path $repoRoot "taskgraph_lab\outputs\stage1\deepseek_v4_flash_batch4_1000_$stamp"
$stdoutLog = Join-Path $logRoot "stage1_batch4_1000_$stamp.stdout.log"
$stderrLog = Join-Path $logRoot "stage1_batch4_1000_$stamp.stderr.log"
$pidPath = Join-Path $logRoot "stage1_batch4_1000_$stamp.pid"
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
        "-PythonPath", $PythonPath,
        "-OutputDirectory", $runRoot
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
    Write-Output "[INFO] RUN=$runRoot"
    Write-Output "[INFO] STDOUT=$stdoutLog"
    Write-Output "[INFO] STDERR=$stderrLog"
}
finally {
    Remove-Item Env:TASKGRAPH_TEACHER_API_KEY -ErrorAction SilentlyContinue
    $key = $null
}

