#!/usr/bin/env bash
# One corrected GRPO cycle: rollout, judge grading, native update, acceptance.
# The next visit's rollout verifies vLLM's copy of the new adapter before it samples.
set -euo pipefail

code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runtime_root=$(dirname -- "$code_root")
usage="Usage: launch-cycle.sh EXISTING_RESULT_DIRECTORY CONFIG_JSON [--resume]"
run_root=$(realpath -s -- "${1:?$usage}")
config_host=$(realpath -s -- "${2:?$usage}")
resume=0
if [[ "${3:-}" == --resume ]]; then resume=1; elif [[ -n "${3:-}" ]]; then echo "$usage" >&2; exit 2; fi
[[ -f "$config_host" ]] || { echo "No config at $config_host" >&2; exit 2; }
[[ "$run_root" == "$runtime_root/runs/"* && -d "$run_root" ]] || {
  echo "The cycle directory must exist under $runtime_root/runs" >&2
  exit 2
}
if ((resume)); then
  # Archive the failed attempt's logs and incomplete phases; completed phases are reused.
  [[ -f "$run_root/prepared.json" ]] || { echo "Resume requires a prepared cycle" >&2; exit 2; }
  python3 "$code_root/cycle_recovery.py" archive "$run_root" "$config_host"
else
  [[ ! -e "$run_root/prepared.json" ]] || {
    echo "Cycle already prepared; continue it with --resume" >&2
    exit 2
  }
  if [[ -e "$run_root/launcher-exit.txt" ]]; then
    # An earlier attempt stopped before preparation; keep its exit record as evidence.
    archive="$run_root/attempts/$(date -u +%Y%m%dT%H%M%SZ)-unprepared"
    mkdir -p "$archive"
    mv "$run_root/launcher-exit.txt" "$run_root/last-phase.txt" "$archive/" 2>/dev/null || true
  fi
fi
cd -- "$code_root"
sha256sum --check "$run_root/SOURCE_SHA256SUMS" > "$run_root/source-verification.log"
source "$code_root/launch-lib.sh"
config=/run-config.json
container_out="/results/$(basename -- "$run_root")"
mounts+=(-v "$config_host:$config:ro")
environment+=(-e UNSLOTH_COMPILE_LOCATION=/workspace/cache/compiled-corrected-cycle
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)

settings=$(docker run --rm --network none "${mounts[@]}" "${environment[@]}" "$CPU_IMAGE" \
  /opt/pilot/gpu/bin/python -m cycle_contract settings --config "$config")
mapfile -t values <<< "$settings"
native_image=${values[0]}
rollout_image=${values[1]}
seconds=${values[2]}  # already capped at the absolute deadline
judge_seconds=${values[3]}
skip_uninformative=${values[4]}
update_gpus=${values[5]}
rollout_gpus=${values[6]}
if ((seconds <= 0)); then echo "Absolute deadline has passed" >&2; exit 124; fi
if ((!resume)); then
  docker run --rm --network none "${mounts[@]}" "${environment[@]}" "$native_image" \
    /opt/pilot/gpu/bin/python -m cycle_contract prepare --config "$config" --out "$container_out"
fi

containers=()
phase=rollout
prefix="audit-grpo-$(basename -- "$run_root")"
timer="$prefix-deadline"
started=$(date +%s)
end=$((started + seconds))
cleanup() {
  local status=$?
  trap - EXIT
  record_containers "$run_root"
  printf '%s\n' "$status" > "$run_root/launcher-exit.txt"
  printf '%s\n' "$phase" > "$run_root/last-phase.txt"
  printf '%s\n' "$(( $(date +%s) - started ))" > "$run_root/cycle-seconds.txt"
  stop_deadline_timer "$timer"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
start_deadline_timer "$timer" "$seconds" "$prefix-rollout" "$prefix-update"

gpu_phase() {
  local image=$1 memory=$2 gpus=$3 container="$prefix-$phase"
  local phase_started
  phase_started=$(date +%s)
  shift 3
  containers+=("$container")
  docker rm "$container" >/dev/null 2>&1 || true
  timeout --foreground -k 10 "$(remaining)s" docker run --name "$container" \
    --network none --gpus "$gpus" --cpus 8 --memory "$memory" --shm-size 16g \
    "${mounts[@]}" "${environment[@]}" "$image" /opt/pilot/gpu/bin/python "$@" \
    > "$run_root/$phase.log" 2>&1
  printf '%s\n' "$(( $(date +%s) - phase_started ))" > "$run_root/$phase-seconds.txt"
}
phase_complete() {
  python3 "$code_root/cycle_recovery.py" complete "$run_root" "$1"
}

phase=rollout
phase_complete rollout || gpu_phase "$rollout_image" 64g "$rollout_gpus" /training-code/inference/rollout.py \
  rollout --config "$config" --out "$container_out/rollout"

phase=grading
if [[ ! -f "$run_root/grading/rewards.json" ]]; then
  left=$(remaining)
  if ((judge_seconds > left)); then judge_seconds=$left; fi
  sync_live_code
  live_exec timeout -k 10 "${judge_seconds}s" /opt/pilot/judge/bin/python \
    -m grpo.judge_worker \
    --samples "$container_out/rollout/samples.json" --split train \
    --out "$container_out/grading" --plan "$container_out/judge-plan.json" --run-paid \
    --cache "$container_out/judge-cache" > "$run_root/grading.log" 2>&1
fi
if [[ "$skip_uninformative" == true ]]; then
  decision=$(docker run --rm --network none "${mounts[@]}" "${environment[@]}" "$native_image" \
    /opt/pilot/gpu/bin/python -m progress group-decision --config "$config" --out "$container_out")
  if [[ "$decision" == skip_no_reward_contrast ]]; then
    phase=skipped_no_reward_contrast
    exit 0
  fi
fi

phase=update
phase_complete update || gpu_phase "$native_image" 128g "$update_gpus" -m grpo.corrected_update \
  --config "$config" --out "$container_out/update"

phase=acceptance
timeout --foreground "$(remaining)s" docker run --rm --network none \
  "${mounts[@]}" "${environment[@]}" "$native_image" \
  /opt/pilot/gpu/bin/python -m cycle_contract finish --config "$config" --out "$container_out" \
  > "$run_root/acceptance.log" 2>&1
phase=complete
