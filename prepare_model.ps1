param(
    [string]$PythonExe = ".\env\python.exe",
    [string]$Config = "Release",
    [string]$OutType = "f16",
    [string]$QuantType = "Q4_K_M"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$llamaCppDir = Join-Path $root "llama.cpp"
$buildDir = Join-Path $llamaCppDir "build"
$modelDir = Join-Path $root "qwen3.5-9b"
$modelsDir = Join-Path $root "models"
$cacheDir = Join-Path $root ".cache"
$hfHome = Join-Path $cacheDir "huggingface"
$tmpDir = Join-Path $cacheDir "tmp"

New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null
New-Item -ItemType Directory -Path $hfHome -Force | Out-Null
New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null

$env:HF_HOME = $hfHome
$env:HUGGINGFACE_HUB_CACHE = Join-Path $hfHome "hub"
$env:TRANSFORMERS_CACHE = Join-Path $hfHome "transformers"
$env:TMP = $tmpDir
$env:TEMP = $tmpDir

function Resolve-Exe {
    param(
        [string]$SearchRoot,
        [string]$Filter
    )

    $candidate = Get-ChildItem -Path $SearchRoot -Recurse -File -Filter $Filter |
        Sort-Object FullName |
        Select-Object -First 1

    if (-not $candidate) {
        throw "未找到可执行文件: $Filter"
    }

    return $candidate.FullName
}

Write-Host "[1/4] 配置 llama.cpp"
cmake -S $llamaCppDir -B $buildDir -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON

Write-Host "[2/4] 编译 llama-quantize / llama-cli"
cmake --build $buildDir --config $Config --target llama-quantize llama-cli

$ggufPath = Join-Path $modelsDir ("qwen3.5-9b-" + $OutType + ".gguf")
$quantPath = Join-Path $modelsDir ("qwen3.5-9b-" + $QuantType + ".gguf")

Write-Host "[3/4] 转换 HF 权重到 GGUF"
& $PythonExe (Join-Path $llamaCppDir "convert_hf_to_gguf.py") $modelDir --outfile $ggufPath --outtype $OutType

$quantizeExe = Resolve-Exe -SearchRoot $buildDir -Filter "llama-quantize*.exe"

Write-Host "[4/4] 量化到 $QuantType"
& $quantizeExe $ggufPath $quantPath $QuantType

Write-Host "完成:"
Write-Host "  GGUF:  $ggufPath"
Write-Host "  Quant: $quantPath"
