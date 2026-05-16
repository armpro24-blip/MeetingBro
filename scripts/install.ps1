$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")

Write-Host "==> Installing MeetingBro..."

# --- Find Python 3.12+ ---
$systemPython = $null
$systemPythonArgs = @()

# Try py -3.12 first
$null = cmd /c "py -3.12 -c ""import sys; assert sys.version_info >= (3, 12)"" >nul 2>nul" 2>$null
if ($LASTEXITCODE -eq 0) {
    $systemPython = "py"
    $systemPythonArgs = @("-3.12")
}

# Fallback to python
if (-not $systemPython) {
    $null = cmd /c "python -c ""import sys; assert sys.version_info >= (3, 12)"" >nul 2>nul" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $systemPython = "python"
    }
}

if (-not $systemPython) {
    Write-Host "Error: Python 3.12+ is required but not found."
    Write-Host "Please install Python 3.12 or later and try again."
    exit 1
}

Write-Host "Using Python interpreter: $systemPython $systemPythonArgs"

# Check npm
try {
    $null = Get-Command npm -ErrorAction Stop
} catch {
    Write-Host "Error: npm not found. Please install Node.js 20+ and try again."
    exit 1
}

# --- Backend ---
Write-Host "==> Setting up backend..."
$backendDir = Join-Path $repoRoot "app/backend"
Set-Location $backendDir

if (-not (Test-Path ".venv")) {
    & $systemPython @systemPythonArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}

$venvPython = Join-Path $backendDir ".venv\Scripts\python.exe"
$venvPip    = Join-Path $backendDir ".venv\Scripts\pip.exe"

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

& $venvPip install -e .
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

if (-not (Test-Path ".env")) {
    $envExample = Join-Path $repoRoot ".env.example"
    if (Test-Path $envExample) {
        Copy-Item $envExample ".env"
    } else {
        New-Item ".env" -ItemType File | Out-Null
    }
}

# --- Frontend ---
Write-Host "==> Setting up frontend..."
$frontendDir = Join-Path $repoRoot "app/frontend"
Set-Location $frontendDir
npm install
if ($LASTEXITCODE -ne 0) { throw "npm install failed" }

Write-Host ""
Write-Host "============================================"
Write-Host "  Installation complete!"
Write-Host "============================================"
Write-Host ""
Write-Host "Start MeetingBro with:"
Write-Host "  .\scripts\start.ps1"
Write-Host ""
