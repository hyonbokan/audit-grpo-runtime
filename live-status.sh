#!/usr/bin/env bash
# Live resources and the newest training run's accepted checkpoint; starts no work.
set -euo pipefail
runtime_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
date -u '+%Y-%m-%d %H:%M:%S UTC'
printf '\nRunning containers\n'
docker ps --format '{{.Names}}: {{.Status}}'
printf '\nGPU memory and utilization\n'
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

# Each controller run keeps its progress in results/<run>/state.json.
state=$(ls -t "$runtime_root"/results/*/state.json 2>/dev/null | head -n 1 || true)
if [[ -n "$state" ]]; then
  run=$(dirname -- "$state")
  printf '\nNewest training run: %s\n' "$(basename -- "$run")"
  jq '{current_step, current_checkpoint, current_adapter_sha256, completion_gate,
       visits: (.visits | length), failed_visits: ([.visits[] | select(.decision == "failed")] | length)}' \
    "$state"
  if [[ -f "$run/training-exit.txt" ]]; then
    printf 'Training finished with exit %s\n' "$(cat "$run/training-exit.txt")"
  fi
  if [[ -f "$run/last-phase.txt" ]]; then
    printf 'Last phase: %s\n' "$(cat "$run/last-phase.txt")"
  fi
else
  printf '\nNo training run state under results/\n'
fi
# launch-lib.sh records each detached deadline watchdog as "<pid> <fire epoch> <containers>".
printf '\nDeadline timers\n'
for record in "$runtime_root"/results/deadline-timers/*.pid; do
  [[ -f "$record" ]] || continue
  read -r pid fires containers < "$record"
  [[ -d "/proc/$pid" ]] && state=waiting || state='not running'
  printf '%s: %s, fires %s, stops %s\n' "$(basename -- "$record" .pid)" "$state" \
    "$(date -u -d "@$fires" '+%Y-%m-%d %H:%M:%S UTC')" "$containers"
done
printf '\nCurrent instructions: %s/private/README.md\n' "$runtime_root"
