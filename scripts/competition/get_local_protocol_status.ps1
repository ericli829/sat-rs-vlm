param([Parameter(Mandatory = $true)][string]$RunDirectory)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$preparation = Join-Path $RunDirectory 'preparation.status.json'
if (Test-Path -LiteralPath $preparation -PathType Leaf) {
    Get-Content -LiteralPath $preparation -Raw
}
$serverState = Join-Path $RunDirectory 'state.json'
if (-not (Test-Path -LiteralPath $serverState -PathType Leaf)) {
    Write-Output 'Server launcher has not created its state file yet'
    exit 0
}
$state = Get-Content -LiteralPath $serverState -Raw | ConvertFrom-Json
$process = Get-Process -Id $state.pid -ErrorAction SilentlyContinue
if (-not $process) {
    Write-Output 'Server process has exited'
    Get-Content -LiteralPath $state.stderr -Tail 20
    exit 1
}
Write-Output "PID: $($state.pid); mode: $($state.mode); endpoint: $($state.endpoint)"
$secret = (Get-Content -LiteralPath $state.key_file -Raw).Trim()
try {
    Invoke-RestMethod -Uri ($state.endpoint + '/healthz') `
        -Headers @{ Authorization = "Bearer $secret" } -TimeoutSec 2 | ConvertTo-Json -Depth 10
}
catch {
    Write-Output 'Health check unavailable: model may still be loading; inspect startup log'
}
Get-Content -LiteralPath $state.stderr -Tail 12
