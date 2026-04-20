#!/usr/bin/env bash
# Full local run: venv → deps → LSTM weights (if missing) → .env stub → uvicorn.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv/bin/python"
PIP="${ROOT}/.venv/bin/pip"
UVICORN="${ROOT}/.venv/bin/uvicorn"

if [[ ! -x "$PY" ]]; then
  echo "Creating virtualenv at .venv ..."
  python3 -m venv "$ROOT/.venv"
fi

echo "Installing dependencies ..."
"$PIP" install -q -r "$ROOT/requirements.txt"

if [[ ! -f "$ROOT/.env" && -f "$ROOT/.env.example" ]]; then
  echo "Creating .env from .env.example (edit and add GOOGLE_MAPS_API_KEY for live traffic)"
  cp "$ROOT/.env.example" "$ROOT/.env"
fi

ARTIFACT="$ROOT/artifacts/lstm_forecaster.pt"
if [[ ! -f "$ARTIFACT" ]]; then
  echo "Training LSTM (first time) → artifacts/lstm_forecaster.pt ..."
  mkdir -p "$ROOT/artifacts"
  "$PY" "$ROOT/scripts/train_lstm.py"
fi

# Prefer PORT env; else first free port in list
PORT="${PORT:-}"
if [[ -z "$PORT" ]]; then
  for p in 8000 8001 8002 8080 8765; do
    if ! lsof -iTCP:"$p" -sTCP:LISTEN -t >/dev/null 2>&1; then
      PORT=$p
      break
    fi
  done
  PORT="${PORT:-8765}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Dashboard:  http://127.0.0.1:${PORT}"
echo "  API docs:   http://127.0.0.1:${PORT}/docs"
echo "  (YOLO weights download on first vision request if needed)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

exec "$UVICORN" app.main:app --reload --host 0.0.0.0 --port "$PORT"
