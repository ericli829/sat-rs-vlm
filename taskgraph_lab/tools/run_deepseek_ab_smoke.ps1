param(
    [Parameter(Mandatory = $true)]
    [string]$PythonPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$inputPath = Join-Path $repoRoot "taskgraph_lab\data\seeds\generated_v1\smoke_5_diverse.jsonl"
$key = [Environment]::GetEnvironmentVariable("TASKGRAPH_TEACHER_API_KEY")

if ([string]::IsNullOrWhiteSpace($key)) {
    throw "TASKGRAPH_TEACHER_API_KEY is required"
}

$runs = @(
    @{
        Name = "thinking_low"
        Config = "taskgraph_lab\configs\deepseek_v4_flash.yaml"
        Output = "taskgraph_lab\outputs\raw\deepseek_v4_flash_final_choice_thinking_low_smoke_5.jsonl"
    },
    @{
        Name = "thinking_disabled"
        Config = "taskgraph_lab\configs\deepseek_v4_flash_thinking_disabled.yaml"
        Output = "taskgraph_lab\outputs\raw\deepseek_v4_flash_final_choice_thinking_disabled_smoke_5.jsonl"
    }
)

try {
    Set-Location -LiteralPath $repoRoot
    foreach ($run in $runs) {
        $name = $run.Name
        Write-Output "[INFO] Starting $name"
        & $PythonPath -m taskgraph_lab.generation.generate `
            --input $inputPath `
            --config $run.Config `
            --output $run.Output `
            --few-shot-file "taskgraph_lab\prompts\few_shot_final_choice.txt"
        if ($LASTEXITCODE -ne 0) {
            throw "$name exited with code $LASTEXITCODE"
        }
        Write-Output "[OK] Finished $name"
    }

    $reviewRoot = "taskgraph_lab\outputs\reports\deepseek_v4_flash_final_choice_smoke_ab"
    & $PythonPath -m taskgraph_lab.tools.build_prompt_review `
        --prompt "taskgraph_lab\prompts\system_prompt.txt" `
        --run "thinking_low=$($runs[0].Output)" `
        --run "thinking_disabled=$($runs[1].Output)" `
        --output-json "$reviewRoot\prompt_review.json" `
        --output-md "$reviewRoot\prompt_review.md"
    if ($LASTEXITCODE -ne 0) {
        throw "prompt review exited with code $LASTEXITCODE"
    }
}
finally {
    Remove-Item Env:TASKGRAPH_TEACHER_API_KEY -ErrorAction SilentlyContinue
}
