#!/usr/bin/env bash
# Publish one model's finished free-text project runs to results/<name>/results.json and rebuild
# results/README.md. A published model is never overwritten.
# Usage: bash training-code/publish-results.sh NAME ARM RUN_DIRECTORY...
#   e.g. publish-results.sh qwen3.8-27b-base-xhigh base runs/base-free-text-xhigh-20261021-*
set -euo pipefail

code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runtime_root=$(dirname -- "$code_root")
name=${1:?Usage: publish-results.sh NAME ARM RUN_DIRECTORY...}
arm=${2:?The arm to publish, e.g. base or checkpoint-N}
shift 2
(($# > 0)) || {
  echo "Name at least one finished project run under $runtime_root/runs" >&2
  exit 2
}
runs=()
for run in "$@"; do
  run=$(realpath -s -- "$run")
  [[ "$run" == "$runtime_root/runs/"* && -f "$run/comparison.json" ]] || {
    echo "$run is not a finished project run under $runtime_root/runs" >&2
    exit 2
  }
  runs+=("/results/$(basename -- "$run")")
done
cd -- "$code_root"
sha256sum --check --quiet SOURCE_SHA256SUMS
source "$code_root/launch-lib.sh"
mkdir -p "$runtime_root/results"
docker run --rm --network none --user "$(id -u):$(id -g)" "${mounts[@]}" "${environment[@]}" \
  -v "$runtime_root/results:/published" "$CPU_IMAGE" \
  /opt/pilot/gpu/bin/python /training-code/inference/published_results.py publish \
  --name "$name" --arm "$arm" --out /published "${runs[@]}"
echo "Published $runtime_root/results/$name/results.json"
