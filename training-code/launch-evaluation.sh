#!/usr/bin/env bash
# One matched evaluation: generate each arm, grade it in the live container, then compare.
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
  /opt/pilot/gpu/bin/python /training-code/inference/evaluation.py settings --config "$config")
mapfile -t values <<< "$settings"
native_image=${values[0]}
rollout_image=${values[1]}
seconds=${values[2]}
judge_seconds=${values[3]}
evaluation_module=${values[4]}
read -r -a labels <<< "${values[5]}"
content_panel=${values[6]}
judge_split=${values[7]}
grading_mode=${values[8]}
docker run --rm --network none "${mounts[@]}" "${environment[@]}" "$native_image" \
  /opt/pilot/gpu/bin/python "/training-code/inference/$evaluation_module.py" prepare \
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

grade_native_arm() {
  local label=$1 samples_path=$2 limit=$3
  live_exec timeout -k 10 "${limit}s" /opt/pilot/judge/bin/python \
    "$LIVE_CODE/inference/benchmark_judge.py" \
    --config "$container_out/prepared.json" --samples "$samples_path" \
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
  if [[ "$evaluation_module" == benchmark_evaluation ]]; then
    samples_path="$container_out/$label/samples.json"
    if [[ "$content_panel" == conditional ]]; then
      timeout --foreground "$(remaining)s" docker run --rm --network none \
        "${mounts[@]}" "${environment[@]}" "$native_image" /opt/pilot/gpu/bin/python \
        /training-code/inference/benchmark_evaluation.py normalize \
        --config "$config" --out "$container_out" --label "$label"
      samples_path="$container_out/conditional/$label/samples.json"
    fi
    if [[ "$grading_mode" == overlap ]]; then
      # Grade this arm while the next arm generates.
      grade_native_arm "$label" "$samples_path" "$limit" &
      grading_jobs+=("$!")
    else
      grade_native_arm "$label" "$samples_path" "$limit"
    fi
    continue
  fi
  live_exec timeout -k 10 "${limit}s" /opt/pilot/judge/bin/python \
    -m grpo.judge_worker --workspace /workspace \
    --samples "$container_out/$label/samples.json" --split "$judge_split" \
    --out "$container_out/$label-grading" --plan "$container_out/judge-plan.json" --run-paid \
    > "$run_root/$label-grading.log" 2>&1
done
phase=grading-join
for job in "${grading_jobs[@]}"; do wait "$job"; done
grading_jobs=()

phase=comparison
timeout --foreground "$(remaining)s" docker run --rm --network none \
  "${mounts[@]}" "${environment[@]}" "$native_image" /opt/pilot/gpu/bin/python \
  "/training-code/inference/$evaluation_module.py" finish --config "$config" --out "$container_out" \
  > "$run_root/comparison.log" 2>&1
phase=complete
