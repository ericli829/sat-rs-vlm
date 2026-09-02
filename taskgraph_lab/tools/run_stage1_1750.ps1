param(
    [Parameter(Mandatory = $true)]
    [string]$PythonPath,

    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$inputPath = Join-Path $repoRoot "taskgraph_lab\data\seeds\generated_v3_1750\seed.jsonl"
$configPath = Join-Path $repoRoot "taskgraph_lab\configs\deepseek_v4_flash_stage1_batch4.yaml"

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
        throw "stage1 batch generation exited with code $LASTEXITCODE"
    }
    Write-Output "[OK] Stage1 1750-sample generation completed"
}
finally {
    Remove-Item Env:TASKGRAPH_TEACHER_API_KEY -ErrorAction SilentlyContinue
}

