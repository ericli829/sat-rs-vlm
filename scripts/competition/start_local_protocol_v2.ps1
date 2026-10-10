param(
    [ValidateSet('offline', 'qwen2b')][string]$Mode = 'offline',
    [int]$Port = 8000,
    [string]$ImagePackage = 'D:\Desktop\tzb-2026\dev100-images',
    [string]$ModelPath = 'D:\Desktop\tzb-2026\Qwen3-VL-2B-Instruct',
    [string]$AdapterPath = 'D:\Desktop\tzb-2026\results\levir_train\vrsbench_levircc_replay_formal\round_2_adapter\checkpoint-3406',
    [switch]$NoAdapter,
    [string]$RunDirectory,
    [string]$Python = 'C:\Users\Ericoneabc\AppData\Local\Microsoft\WindowsApps\python.exe'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
if ($Port -lt 1 -or $Port -gt 65535) { throw 'Invalid port' }
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw 'Port already in use; the existing service will not be stopped'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Python executable missing' }
if (-not (Test-Path -LiteralPath (Join-Path $ImagePackage 'manifest.json') -PathType Leaf)) {
    throw 'Image package manifest missing'
}
if ($Mode -eq 'qwen2b') {
    if (-not (Test-Path -LiteralPath (Join-Path $ModelPath 'config.json') -PathType Leaf)) {
        throw 'Local base model config missing'
    }
    if (-not $NoAdapter -and -not (Test-Path -LiteralPath (Join-Path $AdapterPath 'adapter_model.safetensors') -PathType Leaf)) {
        throw 'LoRA weights missing'
    }
}
if (-not $RunDirectory) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $RunDirectory = Join-Path $repo ".tmp\local-protocol-$Mode-$stamp"
}
$run = [System.IO.Path]::GetFullPath($RunDirectory)
if (Test-Path -LiteralPath (Join-Path $run 'state.json')) { throw 'Run directory already used' }
New-Item -ItemType Directory -Path $run -Force | Out-Null
$utf8 = New-Object System.Text.UTF8Encoding($false)
$keyFile = Join-Path $run 'api-key.txt'
$secret = & $Python -c 'import secrets; print(secrets.token_urlsafe(32))'
if ($LASTEXITCODE -ne 0 -or -not $secret) { throw 'Could not generate API key' }
[System.IO.File]::WriteAllText($keyFile, $secret.Trim(), $utf8)
$questions = Join-Path $run 'questions.jsonl'
$configFile = Join-Path $run 'config.json'
$configuredAdapter = $null
if (-not $NoAdapter) { $configuredAdapter = [System.IO.Path]::GetFullPath($AdapterPath) }
$config = @{
    mode = $Mode
    package = [System.IO.Path]::GetFullPath($ImagePackage)
    preload = $true
    question_capture_path = $questions
    model = @{
        backend = 'huggingface'
        model_id = [System.IO.Path]::GetFullPath($ModelPath)
        adapter_path = $configuredAdapter
        device = 'cuda'
        dtype = 'bfloat16'
        local_files_only = $true
        max_new_tokens = 96
        max_pixels = 262144
    }
}
[System.IO.File]::WriteAllText($configFile, ($config | ConvertTo-Json -Depth 10), $utf8)
$stdout = Join-Path $run 'stdout.log'
$stderr = Join-Path $run 'stderr.log'
$oldKey = $env:MODEL_API_KEY
$oldConfig = $env:COMPETITION_LOCAL_CONFIG
$oldPythonPath = $env:PYTHONPATH
$oldEncoding = $env:PYTHONIOENCODING
try {
    $env:MODEL_API_KEY = $secret.Trim()
    $env:COMPETITION_LOCAL_CONFIG = $configFile
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONPATH = Join-Path $repo 'src'
    if ($oldPythonPath) { $env:PYTHONPATH += ";$oldPythonPath" }
    $arguments = @(
        '-u', '-m', 'uvicorn',
        'sat_rs_vlm.competition.protocol_v2.local_server:create_local_app',
        '--factory', '--host', '127.0.0.1', '--port', $Port,
        '--workers', '1', '--no-access-log'
    )
    $process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $repo `
        -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    $state = @{
        pid = $process.Id
        mode = $Mode
        endpoint = "http://127.0.0.1:$Port"
        started_at = (Get-Date).ToUniversalTime().ToString('o')
        key_file = $keyFile
        question_file = $questions
        stdout = $stdout
        stderr = $stderr
        config_file = $configFile
        model_variant = $(if ($NoAdapter) { 'base_only' } else { 'with_lora' })
    }
    [System.IO.File]::WriteAllText((Join-Path $run 'state.json'), ($state | ConvertTo-Json -Depth 10), $utf8)
    [System.IO.File]::WriteAllText((Join-Path $run 'server.pid'), [string]$process.Id, $utf8)
    $process.Refresh()
    if ($process.HasExited) { throw "Server exited during launch; inspect $stderr" }
    $state | ConvertTo-Json -Depth 10
}
finally {
    $env:MODEL_API_KEY = $oldKey
    $env:COMPETITION_LOCAL_CONFIG = $oldConfig
    $env:PYTHONPATH = $oldPythonPath
    $env:PYTHONIOENCODING = $oldEncoding
}
