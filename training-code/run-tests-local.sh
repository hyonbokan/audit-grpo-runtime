#!/usr/bin/env bash
# run-tests.sh on a workstation without the rental: the local amd64 deps image under
# emulation, this repository's backend in place of the frozen one, and a cached pytest.
# Usage: bash training-code/run-tests-local.sh [native|judge|all] [extra pytest args]
# vLLM tests do not run here: the local image has no vLLM. Run them with run-tests.sh.
set -euo pipefail

code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
backend=${FROZEN_BACKEND:-$(cd -- "$code_root/../../../../.." && pwd)}
[[ -d "$backend/llm_router" && -d "$backend/scripts/finetuning/tools" ]] || {
  echo "No backend with llm_router and scripts/finetuning/tools at $backend; set FROZEN_BACKEND" >&2
  exit 2
}
image=${LOCAL_IMAGE:-audit-grpo-deps:cu129-v1}
[[ -n "$(docker image ls -q "$image")" ]] || { echo "Docker image $image not found" >&2; exit 2; }

# pytest is not in the pinned images; install it once into a cache directory.
tools=${TEST_TOOLS:-${XDG_CACHE_HOME:-$HOME/.cache}/audit-grpo-test-tools}
if [[ ! -d "$tools/pytest_asyncio" ]]; then
  command -v uv > /dev/null || { echo "uv is required to install pytest into $tools" >&2; exit 2; }
  uv pip install --quiet --target "$tools" --python-version 3.12 \
    --python-platform x86_64-manylinux_2_28 pytest pytest-asyncio
fi

NATIVE_IMAGE=$image JUDGE_IMAGE=$image DOCKER_PLATFORM=linux/amd64 \
  TEST_TOOLS=$tools FROZEN_BACKEND=$backend \
  UNAVAILABLE_TESTS="inference/test_adapter_compat.py inference/test_adapter_mapping.py \
    inference/test_rollout.py inference/test_sampling_support.py inference/test_sampler_compat.py" \
  exec bash "$code_root/run-tests.sh" "$@"
