param(
    [Parameter(Mandatory = $true)]
    [string]$PythonPath,

    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$inputPath = Join-Path $repoRoot "taskgraph_lab\data\seeds\generated_v1\batch_benchmark_24_v1.jsonl"
$configPath = Join-Path $repoRoot "taskgraph_lab\configs\deepseek_v4_flash_batch.yaml"

if ([string]::IsNullOrWhiteSpace($env:TASKGRAPH_TEACHER_API_KEY)) {
    throw "TASKGRAPH_TEACHER_API_KEY is required"
}

try {
    Set-Location -LiteralPath $repoRoot
    & $PythonPath -m taskgraph_lab.tools.benchmark_teacher_batches `
        --input $inputPath `
        --config $configPath `
        --output-dir $OutputDirectory `
        --batch-sizes 1 2 4 8
    if ($LASTEXITCODE -ne 0) {
        throw "batch benchmark exited with code $LASTEXITCODE"
    }
    Write-Output "[OK] Batch benchmark completed"
}
finally {
    Remove-Item Env:TASKGRAPH_TEACHER_API_KEY -ErrorAction SilentlyContinue
}
