param([Parameter(Mandatory = $true)][string]$RunDirectory)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$job = Get-Content -LiteralPath (Join-Path $RunDirectory 'job.json') -Raw | ConvertFrom-Json
$taskProcess = Get-Process -Id $job.pid -ErrorAction SilentlyContinue
if (-not $taskProcess) {
    Write-Output 'Replay process has already exited'
    exit 0
}
$expectedStart = ([DateTimeOffset]$job.started_at).UtcDateTime
$actualStart = $taskProcess.StartTime.ToUniversalTime()
if ([Math]::Abs(($actualStart - $expectedStart).TotalSeconds) -gt 1) {
    throw 'PID was reused; refusing to stop an unrelated process'
}
Stop-Process -Id $job.pid -ErrorAction Stop
Write-Output 'Replay client stopped; the model service and tunnel remain running'
