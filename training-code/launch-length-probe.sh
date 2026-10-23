#!/usr/bin/env bash
# Medium-effort length probe of the base model on training rows: generation only, no judge.
set -euo pipefail

code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runtime_root=$(dirname -- "$code_root")
run_root=$(realpath -s -- "${1:?Usage: launch-length-probe.sh RESULT_DIRECTORY CONFIG_JSON}")
config_path=$(realpath -s -- "${2:?Explicit reviewed probe config required}")
[[ "$run_root" == "$runtime_root/runs/"* && -d "$run_root" ]] || {
  echo "The probe directory must exist under $runtime_root/runs" >&2
  exit 2
}
[[ ! -e "$run_root/manifest.json" && ! -e "$run_root/launcher-exit.txt" ]] || {
  echo "Probe already started in $run_root" >&2
  exit 2
}
[[ -f "$config_path" && "$config_path" == "$run_root/"* ]] || {
  echo "The config must be a file inside the probe directory" >&2
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
probe=/training-code/inference/length_probe.py

settings=$(docker run --rm --network none "${mounts[@]}" "${environment[@]}" "$CPU_IMAGE" \
  /opt/pilot/gpu/bin/python "$probe" settings --config "$config")
mapfile -t values <<< "$settings"
native_image=${values[0]}
rollout_image=${values[1]}
seconds=${values[2]}

containers=()
phase=started
container="audit-grpo-$(basename -- "$run_root")"
timer="$container-deadline"
started=$(date +%s)
end=$((started + seconds))
cleanup() {
  local status=$?
  trap - EXIT
  record_containers "$run_root"
  printf '%s\n' "$status" > "$run_root/launcher-exit.txt"
  printf '%s\n' "$phase" > "$run_root/last-phase.txt"
  printf '%s\n' "$(( $(date +%s) - started ))" > "$run_root/probe-seconds.txt"
  stop_deadline_timer "$timer"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
start_deadline_timer "$timer" "$seconds" "$container"

phase=prepare
timeout --foreground "$(remaining)s" docker run --rm --network none \
  "${mounts[@]}" "${environment[@]}" "$native_image" /opt/pilot/gpu/bin/python \
  "$probe" prepare --config "$config" --out "$container_out" > "$run_root/prepare.log" 2>&1

phase=generation
containers+=("$container")
timeout --foreground -k 10 "$(remaining)s" docker run --name "$container" \
  --network none --gpus all --cpus 8 --memory 64g --shm-size 16g \
  "${mounts[@]}" "${environment[@]}" "$rollout_image" \
  /opt/pilot/gpu/bin/python "$probe" run --config "$config" --out "$container_out" \
  > "$run_root/generation.log" 2>&1

phase=report
timeout --foreground "$(remaining)s" docker run --rm --network none \
  "${mounts[@]}" "${environment[@]}" "$native_image" /opt/pilot/gpu/bin/python \
  "$probe" report --config "$config" --out "$container_out" > "$run_root/report.log" 2>&1
phase=complete
