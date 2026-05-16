#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BACKEND_PORT="${BACKEND_PORT:-8765}"

if [ ! -d "${REPO_ROOT}/app/backend/.venv" ]; then
    echo "Error: backend virtual environment not found."
    echo "Please run ./scripts/install.sh first."
    exit 1
fi

# Recursively get child PIDs of a given PID (cross-platform)
get_children() {
    local parent="$1"
    local children=""
    if command -v pgrep >/dev/null 2>&1; then
        children=$(pgrep -P "$parent" 2>/dev/null || true)
    else
        # macOS / BSD fallback
        children=$(ps -o pid= -ppid "$parent" 2>/dev/null | sed 's/ //g' | grep -v '^$' || true)
    fi
    echo "$children"
}

# Recursively kill a process and all its descendants, then reap it.
# This only touches the tree rooted at the given PID and will not
# affect other unrelated npm/node/electron processes on the machine.
kill_tree() {
    local pid="$1"
    local child
    while IFS= read -r child; do
        if [ -n "$child" ]; then
            kill_tree "$child"
        fi
    done < <(get_children "$pid")
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

cleanup() {
    trap - INT TERM EXIT
    echo ""
    echo "Shutting down..."

    if [ -n "${FRONTEND_PID:-}" ]; then
        kill_tree "$FRONTEND_PID"
    fi

    if [ -n "${BACKEND_PID:-}" ]; then
        kill_tree "$BACKEND_PID"
    fi
}

trap 'cleanup; exit 0' INT TERM

# Start backend
echo "==> Starting backend on port ${BACKEND_PORT}..."
cd "${REPO_ROOT}/app/backend"
source .venv/bin/activate
python -m meetingbro.main &
BACKEND_PID=$!

echo "Backend started (PID: ${BACKEND_PID})"

# Wait for backend to be ready
waited=0
ready=0
while [ "${waited}" -lt 30 ]; do
    if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
        echo "Error: Backend process exited unexpectedly."
        wait "${BACKEND_PID}" || true
        exit 1
    fi

    if curl -sf "http://127.0.0.1:${BACKEND_PORT}/health" >/dev/null 2>&1; then
        ready=1
        break
    fi

    sleep 1
    waited=$((waited + 1))
done

if [ "${ready}" -eq 0 ]; then
    echo "Warning: Backend health check did not respond within 30 seconds. Continuing anyway..."
else
    echo "Backend is ready."
fi

# Start frontend
echo "==> Starting frontend..."
cd "${REPO_ROOT}/app/frontend"
npm run dev &
FRONTEND_PID=$!

echo "Frontend started (PID: ${FRONTEND_PID})"

# Keep running until one of the processes exits
while true; do
    if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
        echo "Backend process exited."
        wait "${BACKEND_PID}" || true
        cleanup
        exit 1
    fi

    if ! kill -0 "${FRONTEND_PID}" 2>/dev/null; then
        echo "Frontend process exited."
        wait "${FRONTEND_PID}" || true
        cleanup
        exit 1
    fi

    sleep 1
done
