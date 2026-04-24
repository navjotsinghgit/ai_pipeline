
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDS_FILE="$SCRIPT_DIR/.service_pids"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR" "$SCRIPT_DIR/outputs"

stop_services() {
  echo "stopping services…"
  if [[ -f "$PIDS_FILE" ]]; then
    while IFS= read -r pid; do
      kill "$pid" 2>/dev/null && echo "  killed PID $pid" || true
    done < "$PIDS_FILE"
    rm -f "$PIDS_FILE"
  fi
  echo "done."
}

if [[ "${1-}" == "stop" ]]; then stop_services; exit 0; fi
if [[ -f "$SCRIPT_DIR/venv/bin/activate" ]]; then
  source "$SCRIPT_DIR/venv/bin/activate"
fi

echo "═══════════════════════════════════════════════"
echo "  RetailVision AI Pipeline"
echo "═══════════════════════════════════════════════"

> "$PIDS_FILE"
echo "▶ starting Detection Service on :5001"
(cd "$SCRIPT_DIR/detection_service" && python3 app.py) \
  > "$LOG_DIR/detection.log" 2>&1 &
echo $! >> "$PIDS_FILE"

sleep 1

echo "▶ starting Grouping Service on :5002"
(cd "$SCRIPT_DIR/grouping_service" && python3 app.py) \
  > "$LOG_DIR/grouping.log" 2>&1 &
echo $! >> "$PIDS_FILE"

sleep 1

echo "▶ starting Gateway on :5000"
(cd "$SCRIPT_DIR/gateway" && python3 app.py) \
  > "$LOG_DIR/gateway.log" 2>&1 &
echo $! >> "$PIDS_FILE"

sleep 2

echo ""
echo "   all services started!"
echo ""
echo "   Web UI  →  http://localhost:5000"
echo "   Detect  →  http://localhost:5001/health"
echo "   Group   →  http://localhost:5002/health"
echo ""
echo "   Logs: $LOG_DIR/"
echo "   Stop: bash run_all.sh stop"
echo "═══════════════════════════════════════════════"
