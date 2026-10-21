#!/usr/bin/env bash
# Build the pinned deps and vLLM images and record their identity; starts no GPU work.
# Usage: bash images/build.sh WHEEL_DIRECTORY [TAG_SUFFIX]   (default suffix cu129-v1)
# WHEEL_DIRECTORY is filled (or completed) by fetch_wheels.py; keep it outside the repository.
set -euo pipefail

images_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
usage="Usage: build.sh WHEEL_DIRECTORY [TAG_SUFFIX]"
wheels=$(realpath -m -- "${1:?$usage}")
suffix=${2:-cu129-v1}
# Build evidence is host-specific, so it goes to the git-ignored results directory.
out="$(dirname -- "$images_root")/results/images-$suffix"
mkdir -p "$out"
record="$out/build.json"
trap 'status=$?; ((status == 0)) || echo "Image build failed with status $status" >&2' EXIT

# HOST_PYTHON: an interpreter with working SSL, when the first python3 on PATH lacks it.
"${HOST_PYTHON:-python3}" "$images_root/fetch_wheels.py" "$wheels"
declare -A tags=([deps]="audit-grpo-deps:$suffix" [vllm]="audit-grpo-vllm:$suffix")
for target in deps vllm; do
  docker build --build-context "wheels=$wheels" --target "$target" --tag "${tags[$target]}" \
    "$images_root"
done

# Image IDs and environment listings identify the build; results cite them.
{
  printf '{\n'
  for target in deps vllm; do
    id=$(docker image inspect --format '{{.Id}}' "${tags[$target]}")
    printf '  "%s": {"tag": "%s", "id": "%s"},\n' "$target" "${tags[$target]}" "$id"
  done
  printf '  "built_at": "%s"\n}\n' "$(date -u +%FT%TZ)"
} > "$record"
for target in deps vllm; do
  for environment in gpu judge; do
    docker run --rm --network none "${tags[$target]}" \
      uv pip freeze --python "/opt/pilot/$environment/bin/python" > "$out/$target-$environment.txt"
  done
done
cat "$record"
