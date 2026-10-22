#!/usr/bin/env bash
# One free-text benchmark evaluation of one project: generate each arm, grade it in the live
# container, then compare (e.g. a frozen base run).
set -euo pipefail

code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runtime_root=$(dirname -- "$code_root")
run_root=$(realpath -s -- "${1:?Usage: launch-evaluation.sh RESULT_DIRECTORY CONFIG_JSON}")
config_path=$(realpath -s -- "${2:?Explicit reviewed evaluation config required}")
[[ "$run_root" == "$runtime_root/results/"* && -d "$run_root" ]] || {
  echo "The evaluation directory must exist under $runtime_root/results" >&2
  exit 2
}
[[ ! -e "$run_root/prepared.json" && ! -e "$run_root/launcher-exit.txt" ]] || {
  echo "Evaluation already started in $run_root" >&2
  exit 2
}
[[ -f "$config_path" && "$config_path" == "$run_root/"* ]] || {
  echo "The config must be a file inside the evaluation directory" >&2
  exit 2
}
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || {
  echo "A process is already using the GPUs" >&2
  exit 2
}
cd -- "$code_root"
sha256sum --check "$run_root/SOURCE_SHA256SUMS" > "$run_root/source-verification.log"
source "$code_root/launch-lib.sh"
config=/run-config.json
container_out="/results/$(basename -- "$run_root")"
mounts+=(-v "$config_path:$config:ro")

settings=$(docker run --rm --network none "${mounts[@]}" "${environment[@]}" "$CPU_IMAGE" \
  /opt/pilot/gpu/bin/python /training-code/inference/benchmark_evaluation.py settings \
  --config "$config")
mapfile -t values <<< "$settings"
native_image=${values[0]}
rollout_image=${values[1]}
seconds=${values[2]}
judge_seconds=${values[3]}
read -r -a labels <<< "${values[4]}"
grading_mode=${values[5]}
docker run --rm --network none "${mounts[@]}" "${environment[@]}" "$native_image" \
  /opt/pilot/gpu/bin/python /training-code/inference/benchmark_evaluation.py prepare \
  --config "$config" --out "$container_out"

containers=()
grading_jobs=()
phase=prepared
prefix="audit-grpo-$(basename -- "$run_root")"
timer="$prefix-deadline"
started=$(date +%s)
end=$((started + seconds))
cleanup() {
  local status=$?
  trap - EXIT
  for job in "${grading_jobs[@]}"; do kill "$job" >/dev/null 2>&1 || true; done
  record_containers "$run_root"
  printf '%s\n' "$status" > "$run_root/launcher-exit.txt"
  printf '%s\n' "$phase" > "$run_root/last-phase.txt"
  printf '%s\n' "$(( $(date +%s) - started ))" > "$run_root/evaluation-seconds.txt"
  stop_deadline_timer "$timer"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
deadline_containers=()
for label in "${labels[@]}"; do deadline_containers+=("$prefix-$label"); done
start_deadline_timer "$timer" "$seconds" "${deadline_containers[@]}"

grade_arm() {
  local label=$1 limit=$2
  live_exec timeout -k 10 "${limit}s" /opt/pilot/judge/bin/python \
    "$LIVE_CODE/inference/benchmark_judge.py" \
    --config "$container_out/prepared.json" --samples "$container_out/$label/samples.json" \
    --out "$container_out/$label-grading" --run-paid > "$run_root/$label-grading.log" 2>&1
}

for label in "${labels[@]}"; do
  phase="$label"
  container="$prefix-$label"
  containers+=("$container")
  timeout --foreground -k 10 "$(remaining)s" docker run --name "$container" \
    --network none --gpus all --cpus 8 --memory 64g --shm-size 16g \
    "${mounts[@]}" "${environment[@]}" "$rollout_image" \
    /opt/pilot/gpu/bin/python /training-code/inference/rollout.py evaluate \
    --config "$container_out/$label-config.json" --out "$container_out/$label" \
    > "$run_root/$label.log" 2>&1
  phase="$label-grading"
  sync_live_code
  left=$(remaining)
  limit=$judge_seconds
  if ((limit > left)); then limit=$left; fi
  if [[ "$grading_mode" == overlap ]]; then
    # Grade this arm while the next arm generates.
    grade_arm "$label" "$limit" &
    grading_jobs+=("$!")
  else
    grade_arm "$label" "$limit"
  fi
done
phase=grading-join
for job in "${grading_jobs[@]}"; do wait "$job"; done
grading_jobs=()

phase=comparison
timeout --foreground "$(remaining)s" docker run --rm --network none \
  "${mounts[@]}" "${environment[@]}" "$native_image" /opt/pilot/gpu/bin/python \
  /training-code/inference/benchmark_evaluation.py finish --config "$config" --out "$container_out" \
  > "$run_root/comparison.log" 2>&1
phase=complete
