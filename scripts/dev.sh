#!/usr/bin/env bash
# Одновременный запуск backend и frontend с корректной остановкой по Ctrl+C.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -d "backend/.venv" ]; then
  # shellcheck disable=SC1091
  source backend/.venv/bin/activate
fi

PYTHON="${PYTHON:-python3}"
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-5173}"

pids=()
cleanup() {
  echo
  echo "==> Останавливаю процессы"
  for pid in "${pids[@]:-}"; do
    [ -n "${pid:-}" ] && kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> Backend  http://localhost:$API_PORT/api/docs"
(cd backend && PYTHONPATH=. "$PYTHON" -m uvicorn app.main:app --reload --host 0.0.0.0 --port "$API_PORT") &
pids+=($!)

if command -v npm >/dev/null 2>&1; then
  echo "==> Frontend http://localhost:$WEB_PORT"
  (cd frontend && npm run dev -- --host --port "$WEB_PORT") &
  pids+=($!)
else
  echo "[!] npm не найден: фронтенд не запущен (установите Node.js 20+)"
fi

wait -n 2>/dev/null || wait
