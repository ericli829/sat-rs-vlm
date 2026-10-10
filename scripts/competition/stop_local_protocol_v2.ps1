param([Parameter(Mandatory = $true)][string]$RunDirectory)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
# Stop preparation first so it cannot launch another child during cleanup.
foreach ($name in @('recovery.job.json', 'job.json', 'state.json', 'tunnel.state.json')) {
    $path = Join-Path $RunDirectory $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { continue }
    $state = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    $process = Get-Process -Id $state.pid -ErrorAction SilentlyContinue
    if (-not $process) { continue }
    $expected = [DateTimeOffset]::Parse($state.started_at).UtcDateTime
    $difference = [Math]::Abs(($process.StartTime.ToUniversalTime() - $expected).TotalSeconds)
    if ($difference -gt 10) {
        Write-Warning 'PID start time differs; skipping a reused PID'
        continue
    }
    Stop-Process -Id $state.pid
    Write-Output "Stopped PID $($state.pid)"
}
