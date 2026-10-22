#!/usr/bin/env bash
# Final evaluation: every project's trained arm samples in one vLLM engine, so the weights load
# once; each project is then graded and compared against its frozen base, and the suite reported.
set -euo pipefail

code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runtime_root=$(dirname -- "$code_root")
run_root=$(realpath -s -- "${1:?Suite result directory required}")
config_path=$(realpath -s -- "${2:?Suite configuration required}")
[[ "$run_root" == "$runtime_root/results/"* && "$config_path" == "$run_root/"* ]]
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
cd -- "$code_root"
sha256sum --check "$run_root/SOURCE_SHA256SUMS" > "$run_root/source-verification.log"
source "$code_root/launch-lib.sh"
container_out="/results/$(basename -- "$run_root")"
mounts+=(-v "$config_path:/suite-config.json:ro")
suite() {
  docker run --rm --network none "${mounts[@]}" "${environment[@]}" "$CPU_IMAGE" \
    /opt/pilot/gpu/bin/python /training-code/inference/benchmark_suite.py "$1" \
    --config /suite-config.json --out "$container_out"
}

containers=()
phase=settings
prefix="audit-grpo-$(basename -- "$run_root")"
timer="$prefix-deadline"
cleanup() {
  local status=$?
  trap - EXIT
  record_containers "$run_root"
  stop_deadline_timer "$timer"
  suite report >> "$run_root/report.log" 2>&1 || true
  printf '%s\n' "$status" > "$run_root/suite-exit.txt"
  printf '%s\n' "$phase" > "$run_root/last-phase.txt"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

settings=$(suite settings)
mapfile -t values <<< "$settings"
native_image=${values[0]}
rollout_image=${values[1]}
seconds=${values[2]}
judge_seconds=${values[3]}
label=${values[4]}
read -r -a projects <<< "${values[5]:-}"
if ((${#projects[@]} == 0)); then
  phase=complete
  exit 0
fi
if ((seconds <= 0)); then echo "Suite deadline has passed" >&2; exit 124; fi
started=$(date +%s)
end=$((started + seconds))
start_deadline_timer "$timer" "$seconds" "$prefix-rollout"

phase=prepare
for project in "${projects[@]}"; do
  timeout --foreground "$(remaining)s" docker run --rm --network none \
    "${mounts[@]}" "${environment[@]}" "$native_image" /opt/pilot/gpu/bin/python \
    /training-code/inference/benchmark_evaluation.py prepare \
    --config "/results/$project/config.json" --out "/results/$project" \
    > "$runtime_root/results/$project/prepare.log" 2>&1
done

phase=rollout
manifest=$(suite manifest)
containers+=("$prefix-rollout")
timeout --foreground -k 10 "$(remaining)s" docker run --name "$prefix-rollout" \
  --network none --gpus all --cpus 8 --memory 64g --shm-size 16g \
  "${mounts[@]}" "${environment[@]}" "$rollout_image" \
  /opt/pilot/gpu/bin/python /training-code/inference/rollout.py evaluate-suite \
  --manifest "$manifest" > "$run_root/rollout.log" 2>&1

# A project whose grading or comparison fails is recorded; the others still finish.
phase=grading
sync_live_code
for project in "${projects[@]}"; do
  host="$runtime_root/results/$project"
  child="/results/$project"
  left=$(remaining)
  limit=$judge_seconds
  if ((limit > left)); then limit=$left; fi
  status=0
  live_exec timeout -k 10 "${limit}s" /opt/pilot/judge/bin/python \
    "$LIVE_CODE/inference/benchmark_judge.py" \
    --config "$child/prepared.json" --samples "$child/$label/samples.json" \
    --out "$child/$label-grading" --run-paid > "$host/$label-grading.log" 2>&1 || status=$?
  if ((status == 0)); then
    left=$(remaining)
    timeout --foreground "${left}s" docker run --rm --network none \
      "${mounts[@]}" "${environment[@]}" "$native_image" /opt/pilot/gpu/bin/python \
      /training-code/inference/benchmark_evaluation.py finish \
      --config "$child/config.json" --out "$child" > "$host/comparison.log" 2>&1 || status=$?
  fi
  printf '%s\n' "$status" > "$host/launcher-exit.txt"
  printf '%s\t%s\n' "$project" "$status" >> "$run_root/project-exits.tsv"
done
phase=complete
