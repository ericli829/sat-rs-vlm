param(
    [Parameter(Mandatory = $true)]
    [string]$PythonPath,

    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$inputPath = Join-Path $repoRoot "taskgraph_lab\data\seeds\generated_v1\seed.jsonl"
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

    $validPath = Join-Path $OutputDirectory "valid.jsonl"
    $repairedPath = Join-Path $OutputDirectory "repaired.jsonl"
    $sftPath = Join-Path $OutputDirectory "stage1_sft.jsonl"
    $sftArguments = @("-m", "taskgraph_lab.tools.export_sft", "--output", $sftPath)
    if (Test-Path -LiteralPath $validPath) {
        $sftArguments += @("--input", $validPath)
    }
    if (Test-Path -LiteralPath $repairedPath) {
        $sftArguments += @("--input", $repairedPath)
    }
    if ($sftArguments -contains "--input") {
        & $PythonPath @sftArguments
        if ($LASTEXITCODE -ne 0) {
            throw "stage1 SFT export exited with code $LASTEXITCODE"
        }
    }
    Write-Output "[OK] Stage1 batch generation completed"
}
finally {
    Remove-Item Env:TASKGRAPH_TEACHER_API_KEY -ErrorAction SilentlyContinue
}
