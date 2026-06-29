param(
    [string]$PythonExe = "",
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
$env:PYTHONNOUSERSITE = "1"

function Resolve-PythonExe {
    param(
        [string]$RequestedPythonExe,
        [string]$ProjectRoot
    )

    if ($RequestedPythonExe) {
        if ([System.IO.Path]::IsPathRooted($RequestedPythonExe)) {
            if (Test-Path $RequestedPythonExe) {
                return $RequestedPythonExe
            }
        } else {
            $resolvedRequested = Join-Path $ProjectRoot $RequestedPythonExe
            if (Test-Path $resolvedRequested) {
                return $resolvedRequested
            }
        }

        throw "Specified Python was not found: $RequestedPythonExe"
    }

    $pythonCandidates = @(
        (Join-Path $ProjectRoot "env\Scripts\python.exe"),
        (Join-Path $ProjectRoot ".venv\Scripts\python.exe")
    )

    foreach ($candidate in $pythonCandidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    try {
        $condaPython = (& conda run -n news python -c "import sys; print(sys.executable)" 2>$null | Select-Object -Last 1).Trim()
        if ($LASTEXITCODE -eq 0 -and $condaPython -and (Test-Path $condaPython)) {
            return $condaPython
        }
    } catch {
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        return $pythonCommand.Source
    }

    throw "No usable Python was found. Prepare env/.venv, install conda env 'news', or pass -PythonExe explicitly."
}

function Resolve-Exe {
    param(
        [string]$SearchRoot,
        [string]$Filter
    )

    $candidate = Get-ChildItem -Path $SearchRoot -Recurse -File -Filter $Filter |
        Sort-Object FullName |
        Select-Object -First 1

    if (-not $candidate) {
        throw "Executable not found: $Filter"
    }

    return $candidate.FullName
}

function Test-PythonRuntime {
    param(
        [string]$PythonPath
    )

    $pythonReport = & $PythonPath -c "import sys; import numpy; print(sys.executable); print(sys.version.split()[0]); print(numpy.__version__)" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $pythonReport -or $pythonReport.Count -lt 3) {
        throw "Python runtime check failed. Ensure numpy is installed in the selected Python."
    }

    $resolvedPython = $pythonReport[0].Trim()
    $pythonVersion = $pythonReport[1].Trim()
    $numpyVersion = $pythonReport[2].Trim()

    Write-Host "Resolved Python: $resolvedPython"
    Write-Host "Python Version: $pythonVersion"
    Write-Host "NumPy Version: $numpyVersion"

    $numpyMajor = 0
    [void][int]::TryParse(($numpyVersion -split '\.')[0], [ref]$numpyMajor)
    if ($numpyMajor -ge 2) {
        throw "NumPy 2.x is not supported by the current scipy/sklearn stack during GGUF conversion. Run: `"$resolvedPython`" -m pip install `"numpy<2`" and retry."
    }
}

$PythonExe = Resolve-PythonExe -RequestedPythonExe $PythonExe -ProjectRoot $root
Write-Host "Using Python: $PythonExe"
Test-PythonRuntime -PythonPath $PythonExe

Write-Host "[1/4] Configure llama.cpp"
cmake -S $llamaCppDir -B $buildDir -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON

Write-Host "[2/4] Build llama-quantize / llama-cli"
cmake --build $buildDir --config $Config --target llama-quantize llama-cli

$ggufPath = Join-Path $modelsDir ("qwen3.5-9b-" + $OutType + ".gguf")
$quantPath = Join-Path $modelsDir ("qwen3.5-9b-" + $QuantType + ".gguf")

Write-Host "[3/4] Convert HF weights to GGUF"
& $PythonExe (Join-Path $llamaCppDir "convert_hf_to_gguf.py") $modelDir --outfile $ggufPath --outtype $OutType

$quantizeExe = Resolve-Exe -SearchRoot $buildDir -Filter "llama-quantize*.exe"

Write-Host "[4/4] Quantize to $QuantType"
& $quantizeExe $ggufPath $quantPath $QuantType

Write-Host "Done:"
Write-Host "  GGUF:  $ggufPath"
Write-Host "  Quant: $quantPath"
