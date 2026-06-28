param(
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Test-PythonDeps {
    param([string]$PythonPath)

    try {
        & $PythonPath -c "import chromadb, llama_cpp; print('OK')" 2>$null | Out-Null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

$pythonCandidates = @(
    (Join-Path $projectRoot "env\\Scripts\\python.exe"),
    (Join-Path $projectRoot ".venv\\Scripts\\python.exe"),
    "python"
)

$selectedPython = $null
$useCondaNews = $false

try {
    $condaCheck = & conda run -n news python -c "import chromadb, llama_cpp; print('OK')" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $useCondaNews = $true
    }
} catch {
    $useCondaNews = $false
}

if (-not $useCondaNews) {
    foreach ($candidate in $pythonCandidates) {
        if ($candidate -ne "python" -and -not (Test-Path $candidate)) {
            continue
        }
        if (Test-PythonDeps -PythonPath $candidate) {
            $selectedPython = $candidate
            break
        }
    }
}

if (-not $useCondaNews -and -not $selectedPython) {
    Write-Host "No usable Python runtime was found, or chromadb / llama_cpp is missing." -ForegroundColor Red
    Write-Host "Please prepare the project environment first:" -ForegroundColor Yellow
    Write-Host "  python -m pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

if ($useCondaNews) {
    Write-Host "Using conda environment: news" -ForegroundColor Cyan
    & conda run -n news python (Join-Path $projectRoot "demo_server.py") --port $Port
} else {
    Write-Host "Using Python: $selectedPython" -ForegroundColor Cyan
    & $selectedPython (Join-Path $projectRoot "demo_server.py") --port $Port
}
