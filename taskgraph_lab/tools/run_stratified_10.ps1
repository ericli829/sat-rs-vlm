param(
    [Parameter(Mandatory = $true)]
    [string]$PythonPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$inputPath = Join-Path $repoRoot "taskgraph_lab\data\seeds\generated_v1\stratified_10_v1.jsonl"
$rawPath = Join-Path $repoRoot "taskgraph_lab\outputs\raw\deepseek_v4_flash_stratified_10_thinking_low.jsonl"
$outputRoot = Join-Path $repoRoot "taskgraph_lab\outputs"
$reportRoot = Join-Path $outputRoot "reports\deepseek_v4_flash_stratified_10_thinking_low"
$validPath = Join-Path $outputRoot "valid\deepseek_v4_flash_stratified_10_thinking_low.jsonl"
$repairedPath = Join-Path $outputRoot "repaired\deepseek_v4_flash_stratified_10_thinking_low.jsonl"
$rejectedPath = Join-Path $outputRoot "rejected\deepseek_v4_flash_stratified_10_thinking_low.jsonl"
$reviewsPath = Join-Path $outputRoot "reviews\deepseek_v4_flash_stratified_10_thinking_low.jsonl"
$sftPath = Join-Path $outputRoot "sft\deepseek_v4_flash_stratified_10_thinking_low.jsonl"

if ([string]::IsNullOrWhiteSpace($env:TASKGRAPH_TEACHER_API_KEY)) {
    throw "TASKGRAPH_TEACHER_API_KEY is required"
}

try {
    Set-Location -LiteralPath $repoRoot
    & $PythonPath -m taskgraph_lab.generation.generate `
        --input $inputPath `
        --config "taskgraph_lab\configs\deepseek_v4_flash.yaml" `
        --output $rawPath `
        --few-shot-file "taskgraph_lab\prompts\few_shot_final_choice.txt"
    if ($LASTEXITCODE -ne 0) {
        throw "generation exited with code $LASTEXITCODE"
    }

    & $PythonPath -m taskgraph_lab.tools.summarize `
        --raw $rawPath `
        --valid $validPath `
        --repaired $repairedPath `
        --rejected $rejectedPath `
        --reviews $reviewsPath `
        --output-dir $reportRoot
    if ($LASTEXITCODE -ne 0) {
        throw "summary exited with code $LASTEXITCODE"
    }

    & $PythonPath -m taskgraph_lab.tools.build_prompt_review `
        --prompt "taskgraph_lab\prompts\system_prompt.txt" `
        --run "thinking_low=$rawPath" `
        --output-json (Join-Path $reportRoot "combined_output.json") `
        --output-md (Join-Path $reportRoot "combined_output.md")
    if ($LASTEXITCODE -ne 0) {
        throw "combined output exited with code $LASTEXITCODE"
    }

    $sftArguments = @(
        "-m", "taskgraph_lab.tools.export_sft",
        "--output", $sftPath
    )
    if (Test-Path -LiteralPath $validPath) {
        $sftArguments += @("--input", $validPath)
    }
    if (Test-Path -LiteralPath $repairedPath) {
        $sftArguments += @("--input", $repairedPath)
    }
    if ($sftArguments -contains "--input") {
        & $PythonPath @sftArguments
        if ($LASTEXITCODE -ne 0) {
            throw "SFT export exited with code $LASTEXITCODE"
        }
    }

    Write-Output "[OK] Stratified generation and reports completed"
}
finally {
    Remove-Item Env:TASKGRAPH_TEACHER_API_KEY -ErrorAction SilentlyContinue
}
