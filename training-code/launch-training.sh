#!/usr/bin/env bash
# Multi-visit training under one reviewed schedule, then backup, merge and final evaluation.
set -euo pipefail

code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runtime_root=$(dirname -- "$code_root")
usage="Usage: launch-training.sh RESULT_DIRECTORY CONFIG_JSON [--resume]"
run_root=$(realpath -s -- "${1:?$usage}")
config=$(realpath -s -- "${2:?$usage}")
resume=0
if [[ "${3:-}" == --resume ]]; then resume=1; elif [[ -n "${3:-}" ]]; then echo "$usage" >&2; exit 2; fi
[[ "$run_root" == "$runtime_root/runs/"* && "$config" == "$run_root/"* ]] || {
  echo "The run must be under $runtime_root/runs and its config inside the run directory" >&2
  exit 2
}
if ((resume)); then
  # Resume training from state.json: accepted visits are kept and an unrecorded
  # pending visit continues in place. Finished training is not re-entered.
  [[ -f "$run_root/state.json" ]] || { echo "Resume requires state.json" >&2; exit 2; }
  [[ ! -e "$run_root/training-exit.txt" ]] || { echo "Training already finished" >&2; exit 2; }
  previous=("$run_root"/controller-attempt-*)
  [[ -e "${previous[0]}" ]] || previous=()
  attempt=$(( ${#previous[@]} + 1 ))
  mkdir -p "$run_root/controller-attempt-$attempt"
  for name in launcher-exit.txt last-phase.txt; do
    [[ ! -e "$run_root/$name" ]] || mv "$run_root/$name" "$run_root/controller-attempt-$attempt/"
  done
else
  [[ ! -e "$run_root/state.json" && ! -e "$run_root/launcher-exit.txt" ]] || {
    echo "Run already started; continue it with --resume" >&2
    exit 2
  }
fi
cd -- "$code_root"
sha256sum --check "$run_root/SOURCE_SHA256SUMS" > "$run_root/source-verification.log"
source "$code_root/launch-lib.sh"
container_out="/results/$(basename -- "$run_root")"
pythonpath=/training-code:/training-code/inference:/workspace/code/backend
# The controller writes result directories under the host runtime path, so it mounts it read-write.
controller_mounts=(-v "$code_root:/training-code:ro" -v "$runtime_root:$runtime_root"
  -v "$runtime_root/runs:/results" -v "$FROZEN_WORKSPACE:/workspace:ro"
  -v "$CACHE:$CACHE" -v "$CACHE:/workspace/cache"
  -v "$config:/training-config.json:ro" -v "$runtime_root/diagnostic-tools:/diagnostic-tools:ro")
cpu() {
  docker run --rm --network none "${controller_mounts[@]}" -e "PYTHONPATH=$pythonpath" \
    -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=1 \
    "$CPU_IMAGE" /opt/pilot/gpu/bin/python -m controller "$@" \
    --config /training-config.json --out "$run_root"
}
# Read one key path from a host JSON file without a container.
json_value() {
  python3 -c 'import json, sys
value = json.load(open(sys.argv[1]))
for key in sys.argv[2:]:
    value = value[int(key) if isinstance(value, list) else key]
print(value)' "$@"
}
phase=prepare
cleanup() {
  local status=$?
  trap - EXIT
  printf '%s\n' "$status" > "$run_root/launcher-exit.txt"
  printf '%s\n' "$phase" > "$run_root/last-phase.txt"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
event() {
  phase=$1
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$phase"
  printf '%s\n' "$phase" > "$run_root/last-phase.txt"
}
backup() {
  # Every step returns its own status: errexit is disabled inside a function that is
  # called as `backup ... || handler`, so a failed prepare must not fall through to upload.
  local label=$1
  event "backup-$label"
  timeout --foreground -k 30 3600 docker run --rm --network none "${controller_mounts[@]}" \
    -e "PYTHONPATH=$pythonpath" -e PYTHONDONTWRITEBYTECODE=1 \
    "$CPU_IMAGE" /opt/pilot/gpu/bin/python -m hf_backup \
    --phase prepare --config /training-config.json --run "$container_out" \
    --source /training-code --workspace /workspace --label "$label" \
    > "$run_root/backup-$label-prepare.log" 2>&1 || return $?
  sync_live_code || return $?
  timeout --foreground -k 30 3600 docker exec -w /tmp \
    -e PYTHONPATH="$LIVE_CODE" -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=0 "$LIVE_CONTAINER" \
    /opt/pilot/backup/bin/python -m hf_backup --phase upload \
    --config "$container_out/config.json" --run "$container_out" \
    --source "$LIVE_CODE" --workspace /workspace --label "$label" \
    > "$run_root/backup-$label-upload.log" 2>&1 || return $?
}
visit_backup() {
  # The accepted checkpoint stays on local disk, so one failed upload (network, quota)
  # is reported and does not end unattended training. Initial and final backups stay strict.
  local label=$1 status=0
  backup "$label" || status=$?
  if ((status)); then
    event "backup-failed-$label"
    printf '%s %s %s\n' "$(date -u +%FT%TZ)" "$label" "$status" >> "$run_root/backup-failures.txt"
    # Unuploaded staging would otherwise accumulate one bundle per failed visit.
    docker run --rm --network none "${controller_mounts[@]}" -e "PYTHONPATH=$pythonpath" \
      -e PYTHONDONTWRITEBYTECODE=1 "$CPU_IMAGE" /opt/pilot/gpu/bin/python -m hf_backup \
      --phase discard --config /training-config.json --run "$container_out" \
      --source /training-code --workspace /workspace --label "$label" \
      >> "$run_root/backup-$label-prepare.log" 2>&1 || event "backup-discard-failed-$label"
  fi
  return 0
}
# Free resume state and adapter copies that no later phase reads (storage policy).
prune() {
  event prune
  cpu prune >> "$run_root/retention.log"
}
merge() {
  local adapter=$1 output=$2
  # Use only a complete matching merge; partial merge directories fail closed.
  if [[ -d "$CACHE/${output#/workspace/cache/}" ]]; then
    cpu verify-final-merge
    return
  fi
  event merge
  local base revision
  base=$(json_value "$config" cycle model_path)
  revision=$(json_value "$config" cycle model_revision)
  timeout --foreground -k 30 3600 docker run --rm --network none --cpus 8 --memory 128g \
    "${controller_mounts[@]}" -e OMP_NUM_THREADS=8 "$CPU_IMAGE" \
    /opt/pilot/gpu/bin/python /diagnostic-tools/merge_checkpoint.py \
    --base "$base" --adapter "$adapter" --output "$output" --revision "$revision" \
    >> "$run_root/merge.log" 2>&1
}

if ((resume)); then
  event resume
else
  cpu prepare
  backup initial
fi
mapfile -t policy < <(cpu failure-policy)
visit_attempts=${policy[0]}
skip_failed_visits=${policy[1]}
training_status=0
while true; do
  event training-prepare
  folder=$(cpu next)
  [[ "$folder" != complete ]] || break
  event training
  attempts=0
  cycle_status=0
  while true; do
    attempts=$((attempts + 1))
    resume_flag=()
    [[ ! -f "$folder/prepared.json" ]] || resume_flag=(--resume)
    cycle_status=0
    bash "$code_root/launch-cycle.sh" "$folder" "$folder/config.json" "${resume_flag[@]}" \
      >> "$folder/launcher.log" 2>&1 || cycle_status=$?
    # Success, a time limit (124) or an operator stop (130/143) ends the attempts.
    if ((cycle_status == 0 || cycle_status == 124 || cycle_status == 130 || cycle_status == 143)); then
      break
    fi
    ((attempts < visit_attempts)) || break
    event "training-retry-$attempts"
  done
  if ((cycle_status != 0)); then
    event "training-failed-$cycle_status"
    # A visit time limit (124) is an isolated failure: `next` ends training once the
    # window is exhausted. Only an operator stop, or a policy of stopping on failure, ends it.
    if ((cycle_status == 130 || cycle_status == 143 || !skip_failed_visits)); then
      training_status=1
      break
    fi
    cpu record-failure --attempts "$attempts" >> "$run_root/training-record.log"
  else
    cpu record >> "$run_root/training-record.log"
  fi
  # Thin backups cover every visit since the last verified one, so skipping some loses none.
  due=$(cpu backup-due)
  [[ "$due" != true ]] || visit_backup "visit-$(basename -- "$folder")"
  prune
done
printf '%s\n' "$training_status" > "$run_root/training-exit.txt"
event training-finished
backup training-final
if ((training_status)); then
  event training-unaccepted
  exit 1
fi
settings=$(cpu identity)
mapfile -t values <<< "$settings"
if [[ "${values[3]:-false}" != true ]]; then
  event training-unaccepted
  exit 1
fi
merge "${values[0]}" "${values[1]}"
event evaluation-prepare
suite=$(cpu evaluation)
event evaluation
bash "$code_root/launch-benchmark-suite.sh" "$suite" "$suite/config.json" > "$suite/launcher.log" 2>&1
event release-final-merge
cpu release-final-merge >> "$run_root/retention.log"
backup evaluated-final
event complete
