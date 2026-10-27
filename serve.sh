#!/usr/bin/env bash
# Start the web app as a managed background process.
#
# Detached from this terminal on purpose: closing the window must not kill a
# transcription in progress. Logs go to .run/app.log, the PID to .run/app.pid.
#
#   ./serve.sh start        # start on the default port (8765)
#   ./serve.sh start 9100   # start on a specific port
#   ./serve.sh status
#   ./serve.sh stop
#   ./serve.sh logs
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

RUN_DIR="$ROOT/.run"
PID_FILE="$RUN_DIR/app.pid"
LOG_FILE="$RUN_DIR/app.log"
DEFAULT_PORT=8765

mkdir -p "$RUN_DIR"

python_bin() {
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    printf '%s' "$ROOT/.venv/bin/python"
  else
    printf 'python3'
  fi
}

running_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  printf '%s' "$pid"
}

start() {
  local port="${1:-${MYTRANSCRIBE_PORT:-$DEFAULT_PORT}}"
  if pid="$(running_pid)"; then
    echo "[=] Already running (pid $pid). Use './serve.sh status' or './serve.sh stop'."
    return 0
  fi

  # A detached child in its own session: closing the terminal does not stop it.
  setsid nohup "$(python_bin)" -m mytranscribe --port "$port" \
    >>"$LOG_FILE" 2>&1 &
  local pid=$!
  echo "$pid" >"$PID_FILE"

  local waited=0
  while (( waited < 100 )); do
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "[+] Listening on http://127.0.0.1:$port  (pid $pid)"
      echo "    Model is still loading if /health says 'degraded'; watch with './serve.sh logs'."
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[x] The app exited during startup. Last lines of $LOG_FILE:" >&2
      tail -n 20 "$LOG_FILE" >&2
      rm -f "$PID_FILE"
      return 1
    fi
    sleep 0.5
    waited=$(( waited + 1 ))
  done

  echo "[!] Still starting after $(( waited / 2 ))s (the model takes ~24s to load)."
  echo "    pid $pid is alive; watch it with './serve.sh logs'."
  return 0
}

stop() {
  if ! pid="$(running_pid)"; then
    echo "[=] Not running."
    rm -f "$PID_FILE"
    return 0
  fi
  echo "[*] Stopping pid $pid\u2026"
  kill "$pid" 2>/dev/null || true
  local waited=0
  while kill -0 "$pid" 2>/dev/null && (( waited < 30 )); do
    sleep 0.5
    waited=$(( waited + 1 ))
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "[!] Did not stop in 15s; sending SIGKILL."
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  echo "[+] Stopped."
}

status() {
  if pid="$(running_pid)"; then
    echo "[+] Running (pid $pid)"
    echo -n "    health: "
    for port in ${MYTRANSCRIBE_PORT:-} $DEFAULT_PORT; do
      if curl -fsS "http://127.0.0.1:$port/health" 2>/dev/null; then
        echo
        return 0
      fi
    done
    echo "not answering yet"
  else
    echo "[=] Not running."
  fi
}

case "${1:-start}" in
  start)  start "${2:-}" ;;
  stop)   stop ;;
  restart) stop; start "${2:-}" ;;
  status) status ;;
  logs)   tail -n "${2:-40}" -f "$LOG_FILE" ;;
  *)
    echo "usage: $0 {start [port]|stop|restart [port]|status|logs [lines]}" >&2
    exit 2
    ;;
esac