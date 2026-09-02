param(
    [Parameter(Mandatory = $true)]
    [string]$PythonPath,

    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,

    [Parameter(Mandatory = $true)]
    [string]$RecheckDirectory,

    [Parameter(Mandatory = $true)]
    [string]$DatasetDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$inputPath = Join-Path $repoRoot "taskgraph_lab\data\seeds\hard_remaining_v1\seed.jsonl"
$configPath = Join-Path $repoRoot "taskgraph_lab\configs\deepseek_v4_flash_stage1_batch4.yaml"
$xlrsPath = "D:\Desktop\tzb-2026\xlrs_questions_answers.json"
$mmePath = "D:\Desktop\tzb-2026\MME_RealWorld.json"
$trainInput1 = Join-Path $repoRoot "taskgraph_lab\outputs\rechecks\stage1_batch4_1000_20260831_212902\accepted.jsonl"
$trainInput2 = Join-Path $repoRoot "taskgraph_lab\outputs\rechecks\stage1_batch4_1750_20260831_235837\accepted.jsonl"
$testInput = Join-Path $repoRoot "taskgraph_lab\outputs\rechecks\choice_cardinality_v2_20260830_235226\combined_revalidation\accepted.jsonl"

if ([string]::IsNullOrWhiteSpace($env:TASKGRAPH_TEACHER_API_KEY)) {
    throw "TASKGRAPH_TEACHER_API_KEY is required"
}

try {
    Set-Location -LiteralPath $repoRoot
    & $PythonPath -u -m taskgraph_lab.tools.generate_teacher_batches `
        --input $inputPath `
        --config $configPath `
        --output-dir $OutputDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "hard-remaining Stage1 generation exited with code $LASTEXITCODE"
    }
    Write-Output "[OK] Hard-remaining Stage1 generation completed"

    & $PythonPath -u -m taskgraph_lab.tools.revalidate_stage1 `
        --base-run-dir $OutputDirectory `
        --xlrs-json $xlrsPath `
        --mme-json $mmePath `
        --output-dir $RecheckDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "hard-remaining revalidation exited with code $LASTEXITCODE"
    }
    Write-Output "[OK] Hard-remaining revalidation completed"

    $newAccepted = Join-Path $RecheckDirectory "accepted.jsonl"
    & $PythonPath -u -m taskgraph_lab.tools.prepare_planner_holdout_sft `
        --train-input $trainInput1 `
        --train-input $trainInput2 `
        --train-input $newAccepted `
        --test-input $testInput `
        --output-dir $DatasetDirectory `
        --target-format dsl
    if ($LASTEXITCODE -ne 0) {
        throw "hard-augmented Planner dataset build exited with code $LASTEXITCODE"
    }
    Write-Output "[OK] Hard-augmented Planner dataset completed"
    Write-Output "[INFO] DATASET=$DatasetDirectory"
}
finally {
    Remove-Item Env:TASKGRAPH_TEACHER_API_KEY -ErrorAction SilentlyContinue
}
