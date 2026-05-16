$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")

$BackendPort = if ($env:BACKEND_PORT) { $env:BACKEND_PORT } else { 8765 }

$venvPython = Join-Path $repoRoot "app\backend\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Error: backend virtual environment not found."
    Write-Host "Please run .\scripts\install.ps1 first."
    exit 1
}

$backendDir  = Join-Path $repoRoot "app\backend"
$frontendDir = Join-Path $repoRoot "app\frontend"

# Kill a process and all of its descendants using taskkill.
# /F = force | /T = terminate tree | /PID = target process.
# This only affects the tree rooted at the given PID, so it will not
# touch other unrelated npm/node/electron processes on the machine.
function Stop-ProcessTree {
    param([int]$Id)
    $null = & taskkill /F /T /PID $Id 2>$null
}

# Start backend
Write-Host "==> Starting backend on port ${BackendPort}..."
$backendProcess = Start-Process -FilePath $venvPython `
    -ArgumentList "-m", "meetingbro.main" `
    -WorkingDirectory $backendDir `
    -PassThru

Write-Host "Backend started (PID: $($backendProcess.Id))"

# Wait for backend to be ready
$waited = 0
$ready = $false
while ($waited -lt 30) {
    if ($backendProcess.HasExited) {
        Write-Host "Error: Backend process exited unexpectedly."
        exit 1
    }

    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:${BackendPort}/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            $ready = $true
            break
        }
    } catch {
        # Ignore errors
    }

    Start-Sleep -Seconds 1
    $waited++
}

if (-not $ready) {
    Write-Host "Warning: Backend health check did not respond within 30 seconds. Continuing anyway..."
} else {
    Write-Host "Backend is ready."
}

# Start frontend
Write-Host "==> Starting frontend..."
$frontendProcess = Start-Process -FilePath "npm" `
    -ArgumentList "run", "dev" `
    -WorkingDirectory $frontendDir `
    -PassThru

Write-Host "Frontend started (PID: $($frontendProcess.Id))"

# The try/finally block guarantees cleanup runs even when the user
# presses Ctrl+C, because Ctrl+C raises a break signal that PowerShell
# routes through the finally block before terminating the script.
try {
    while ($true) {
        if ($backendProcess.HasExited) {
            Write-Host "Backend process exited."
            if (-not $frontendProcess.HasExited) {
                Stop-ProcessTree -Id $frontendProcess.Id
            }
            exit $backendProcess.ExitCode
        }

        if ($frontendProcess.HasExited) {
            Write-Host "Frontend process exited."
            if (-not $backendProcess.HasExited) {
                Stop-ProcessTree -Id $backendProcess.Id
            }
            exit $frontendProcess.ExitCode
        }

        Start-Sleep -Seconds 1
    }
} finally {
    Write-Host ""
    Write-Host "Shutting down..."
    if (-not $backendProcess.HasExited) {
        Stop-ProcessTree -Id $backendProcess.Id
    }
    if (-not $frontendProcess.HasExited) {
        Stop-ProcessTree -Id $frontendProcess.Id
    }
}
