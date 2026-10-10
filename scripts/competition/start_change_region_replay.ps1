param(
    [Parameter(Mandatory = $true)][string]$StateFile,
    [Parameter(Mandatory = $true)][string]$SourceRun,
    [Parameter(Mandatory = $true)][string]$ScoreFile,
    [Parameter(Mandatory = $true)][string]$RunDirectory,
    [ValidateRange(0, 10)][int]$RepeatRounds = 0,
    [string]$HumanReview,
    [string]$BaselineResults
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$python = 'C:\Users\Ericoneabc\AppData\Local\Microsoft\WindowsApps\python.exe'
$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$statePath = (Resolve-Path -LiteralPath $StateFile).Path
$scorePath = (Resolve-Path -LiteralPath $ScoreFile).Path
$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
$config = Get-Content -LiteralPath $state.config_file -Raw | ConvertFrom-Json
$taskRunPath = [System.IO.Path]::GetFullPath($RunDirectory)
New-Item -ItemType Directory -Path $taskRunPath -Force | Out-Null
if (Test-Path -LiteralPath (Join-Path $taskRunPath 'job.json')) {
    throw 'Use a fresh run directory; a task was already launched here'
}
$stdout = Join-Path $taskRunPath 'stdout.log'
$stderr = Join-Path $taskRunPath 'stderr.log'
$script = Join-Path $PSScriptRoot 'replay_change_region.py'
if ($RepeatRounds -gt 0) {
    if (-not $HumanReview -or -not $BaselineResults) {
        throw 'Repeat mode requires HumanReview and BaselineResults'
    }
    $humanReviewPath = (Resolve-Path -LiteralPath $HumanReview).Path
    $baselineResultsPath = (Resolve-Path -LiteralPath $BaselineResults).Path
    $script = Join-Path $PSScriptRoot 'repeat_change_region.py'
}
$arguments = @(
    '-u', ('"{0}"' -f $script),
    '--state', ('"{0}"' -f $statePath),
    '--capture', ('"{0}"' -f $state.question_file),
    '--source-run', ('"{0}"' -f $SourceRun),
    '--score', ('"{0}"' -f $scorePath),
    '--package', ('"{0}"' -f $config.package),
    '--output', ('"{0}"' -f $taskRunPath)
)
if ($RepeatRounds -gt 0) {
    $arguments += @(
        '--rounds', $RepeatRounds,
        '--human-review', ('"{0}"' -f $humanReviewPath),
        '--baseline-results', ('"{0}"' -f $baselineResultsPath)
    )
}
$taskProcess = Start-Process -FilePath $python -ArgumentList $arguments `
    -WorkingDirectory $repositoryRoot -WindowStyle Hidden `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$job = @{
    pid = $taskProcess.Id
    started_at = $taskProcess.StartTime.ToUniversalTime().ToString('o')
    stdout = $stdout
    stderr = $stderr
    status = (Join-Path $taskRunPath 'status.json')
    report = (Join-Path $taskRunPath 'report.html')
    source_run = $SourceRun
    service_pid = $state.pid
    repeat_rounds = $RepeatRounds
}
$job | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $taskRunPath 'job.json') -Encoding UTF8
$taskProcess.Id | Set-Content -LiteralPath (Join-Path $taskRunPath 'job.pid') -Encoding ASCII
$taskProcess.Refresh()
if ($taskProcess.HasExited) {
    Get-Content -LiteralPath $stderr -Tail 20
    throw 'Replay process exited during startup'
}
$job | ConvertTo-Json -Depth 10
