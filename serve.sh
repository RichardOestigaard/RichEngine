#!/bin/bash
# RichEngine serve script — LFM3 chat + d1-2B decision residents.
#
# Usage:
#   ./serve.sh           # start (kills an existing :8000 listener first)
#   ./serve.sh stop      # stop the server
#   ./serve.sh status    # residents + pool state
#
# Env overrides: PORT, CHAT_MODEL, DECISION_MODEL, MODELS_DIR, ENGINE_BIN.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8000}"
CHAT_MODEL="${CHAT_MODEL:-LiquidAI/LFM3-1.7B-MLX-4bit}"
DECISION_MODEL="${DECISION_MODEL:-LiquidAI/d1-2B-GGUF:Q8_0}"
MODELS_DIR="${MODELS_DIR:-$ROOT/install/models}"
ENGINE_BIN="${ENGINE_BIN:-$ROOT/build/richengine}"
PYTHON="$ROOT/.venv/bin/python"
LOG="/tmp/richengine.log"

running_pid() { lsof -ti ":$PORT" 2>/dev/null || true; }

stop_server() {
	local pid
	pid="$(running_pid)"
	if [ -n "$pid" ]; then
		kill "$pid" 2>/dev/null || true
		sleep 2
	fi
}

wait_ready() {
	for _ in $(seq 1 60); do
		if curl -sf --max-time 2 "http://127.0.0.1:$PORT/status" >/dev/null 2>&1; then
			return 0
		fi
		sleep 1
	done
	echo "server did not become ready — last log lines:" >&2
	tail -20 "$LOG" >&2
	return 1
}

load_model() {
	curl -sf --max-time 120 -X POST "http://127.0.0.1:$PORT/v1/models/load" \
		-H 'Content-Type: application/json' \
		-d "{\"model\":\"$1\"}"
	echo
}

case "${1:-start}" in
stop)
	stop_server
	;;
status)
	curl -s "http://127.0.0.1:$PORT/status" |
		python3 -c "import json,sys; d=json.load(sys.stdin); print('ready:', d.get('ready')); print('residents:', list(d.get('residents', {}))); print('pool:', d.get('pool'))"
	;;
start)
	stop_server
	# max-residents bounds the pool: chat + decision = 2. Without it the
	# second load evicts the first.
	PYTHONPATH="$ROOT" nohup "$PYTHON" -u -P -m server.server \
		--models-dir "$MODELS_DIR" \
		--binary "$ENGINE_BIN" \
		--port "$PORT" \
		--max-residents 2 \
		>"$LOG" 2>&1 &
	echo "richengine pid $!"
	wait_ready
	load_model "$CHAT_MODEL"
	load_model "$DECISION_MODEL"
	curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -m json.tool
	;;
*)
	echo "usage: $0 [start|stop|status]" >&2
	exit 64
	;;
esac
