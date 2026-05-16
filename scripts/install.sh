#!/usr/bin/env bash
set -e

# Resolve repository root relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "==> Installing MeetingBro..."

# --- Find Python 3.12+ ---
PYTHON_CMD=""

if command -v python3 >/dev/null 2>&1; then
    if python3 -c 'import sys; assert sys.version_info >= (3, 12)' 2>/dev/null; then
        PYTHON_CMD="python3"
    fi
fi

if [[ -z "$PYTHON_CMD" ]] && command -v python3.12 >/dev/null 2>&1; then
    if python3.12 -c 'import sys; assert sys.version_info >= (3, 12)' 2>/dev/null; then
        PYTHON_CMD="python3.12"
    fi
fi

if [[ -z "$PYTHON_CMD" ]]; then
    echo "Error: Python 3.12+ is required but not found."
    echo "Please install Python 3.12 or later and try again."
    exit 1
fi

echo "Using Python interpreter: $PYTHON_CMD"

# Check npm
if ! command -v npm >/dev/null 2>&1; then
    echo "Error: npm not found. Please install Node.js 20+ and try again."
    exit 1
fi

# --- Backend ---
echo "==> Setting up backend..."
cd "${REPO_ROOT}/app/backend"

if [ ! -d ".venv" ]; then
    "$PYTHON_CMD" -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .

if [ ! -f ".env" ]; then
    if [ -f "${REPO_ROOT}/.env.example" ]; then
        cp "${REPO_ROOT}/.env.example" .env
    else
        touch .env
    fi
fi

# --- Frontend ---
echo "==> Setting up frontend..."
cd "${REPO_ROOT}/app/frontend"
npm install

echo ""
echo "============================================"
echo "  Installation complete!"
echo "============================================"
echo ""
echo "Start MeetingBro with:"
echo "  ./scripts/start.sh"
echo ""
