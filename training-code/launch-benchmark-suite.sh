#!/usr/bin/env bash
# Held-out benchmark suite: one launch-evaluation.sh per project, then the suite report.
set -euo pipefail

code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runtime_root=$(dirname -- "$code_root")
run_root=$(realpath -s -- "${1:?Suite result directory required}")
config_path=$(realpath -s -- "${2:?Suite configuration required}")
[[ "$run_root" == "$runtime_root/results/"* && "$config_path" == "$run_root/"* ]]
cd -- "$code_root"
sha256sum --check "$run_root/SOURCE_SHA256SUMS" > "$run_root/source-verification.log"
source "$code_root/launch-lib.sh"
container_out="/results/$(basename -- "$run_root")"
cpu() {
  docker run --rm --network none \
    -v "$code_root:/training-code:ro" -v "$runtime_root/results:/results" \
    -v "$CACHE:$CACHE" \
    -v "$config_path:/suite-config.json:ro" \
    -e PYTHONPATH=/training-code/inference -e PYTHONDONTWRITEBYTECODE=1 \
    "$CPU_IMAGE" /opt/pilot/gpu/bin/python \
    /training-code/inference/benchmark_suite.py "$1" \
    --config /suite-config.json --out "$container_out"
}
cleanup() {
  local status=$?
  trap - EXIT
  cpu report >> "$run_root/report.log" 2>&1 || true
  printf '%s\n' "$status" > "$run_root/suite-exit.txt"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cpu report > "$run_root/report.log" 2>&1
while true; do
  child=$(cpu next)
  [[ "$child" != complete ]] || break
  status=0
  bash "$code_root/launch-evaluation.sh" "$child" "$child/config.json" \
    > "$child/launcher.log" 2>&1 || status=$?
  printf '%s\t%s\n' "$(basename -- "$child")" "$status" >> "$run_root/project-exits.tsv"
  cpu report >> "$run_root/report.log" 2>&1
done
