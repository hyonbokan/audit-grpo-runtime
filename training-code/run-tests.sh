#!/usr/bin/env bash
# CPU test suites in the pinned images; no GPU, network or paid calls.
# Usage: bash training-code/run-tests.sh [native|judge|shell|all] [extra pytest args]
# Exits non-zero when any selected suite fails. Environment overrides, used by
# run-tests-local.sh: NATIVE_IMAGE, JUDGE_IMAGE, DOCKER_PLATFORM, TEST_TOOLS,
# FROZEN_BACKEND, and UNAVAILABLE_TESTS (files the image cannot import, reported as not run).
set -uo pipefail

code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runtime_root=$(dirname -- "$code_root")
which=${1:-all}
shift || true
native_image=${NATIVE_IMAGE:-audit-grpo-vllm:cu129-v1}
judge_image=${JUDGE_IMAGE:-audit-grpo-deps:cu129-v1}
test_tools=${TEST_TOOLS:-$runtime_root/test-tools}
frozen_backend=${FROZEN_BACKEND:-$runtime_root/workspace/code/backend}
# Whitespace-separated paths; word splitting is intended.
# shellcheck disable=SC2206
unavailable=(${UNAVAILABLE_TESTS:-})
# Python 3.14 judge environment: router, reward, budget and panel scoring code.
judge_tests=(
  grpo/test_judge_worker.py grpo/test_reward.py grpo/test_judge_environment.py
  grpo/test_budget.py grpo/test_judge_cache.py
  inference/test_benchmark_judge.py
)

# Tests write beside their modules, so run on a copy.
copy=$(mktemp -d)
trap 'rm -rf "$copy"' EXIT
rsync -a --exclude __pycache__ --exclude .pytest_cache --exclude .ruff_cache "$code_root/" "$copy/"
mounts=(-v "$copy:/training-code" -v "$test_tools:/test-tools:ro"
  -v "$frozen_backend:/workspace/code/backend:ro")
# The pinned tokenizer test reads the model cache when this host's host.env names one.
if [[ -f "$runtime_root/host.env" ]]; then
  # shellcheck source=/dev/null
  source "$runtime_root/host.env"
  [[ -d "${AUDIT_GRPO_CACHE:-}" ]] && mounts+=(-v "$AUDIT_GRPO_CACHE:/workspace/cache:ro")
fi
platform=()
[[ -n "${DOCKER_PLATFORM:-}" ]] && platform=(--platform "$DOCKER_PLATFORM")
ignored=()
for test in "${unavailable[@]}"; do ignored+=(--ignore="$test"); done

suite() {
  local image=$1 python=$2
  shift 2
  docker run --rm --network none --entrypoint "" --user "$(id -u):$(id -g)" "${platform[@]}" \
    "${mounts[@]}" \
    -e PYTHONPATH=/training-code:/training-code/inference:/workspace/code/backend:/test-tools \
    -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 -w /training-code \
    "$image" "$python" -m pytest -q -p no:cacheprovider -rfE "${ignored[@]}" "$@"
}

status=0
report() {
  local name=$1 code=$2 output=$3
  printf '%-7s %s\n' "$name" "$(tail -n 1 <<< "$output")"
  grep '^FAILED\|^ERROR' <<< "$output" | sed 's/ - .*//; s/^/  /'
  ((code == 0)) || status=1
}

if [[ "$which" == all || "$which" == judge ]]; then
  output=$(suite "$judge_image" /opt/pilot/judge/bin/python "${judge_tests[@]}" "$@" 2>&1)
  report judge $? "$output"
fi
if [[ "$which" == all || "$which" == native ]]; then
  # The vLLM image also carries the native training dependencies used by CPU tests.
  skip_judge=()
  for test in "${judge_tests[@]}"; do skip_judge+=(--ignore="$test"); done
  output=$(suite "$native_image" /opt/pilot/gpu/bin/python "${skip_judge[@]}" "$@" 2>&1)
  report native $? "$output"
fi
if [[ "$which" == all || "$which" == shell ]]; then
  # Offline checks of the host scripts; they stub docker and nvidia-smi themselves.
  for test in test_live_status.sh test_preflight.sh; do
    output=$(bash "$code_root/$test" 2>&1)
    code=$?
    ((code == 0)) && output="passed" || output+=$'\n'"FAILED $test"
    report shell $code "$output"
  done
fi
if ((${#unavailable[@]})); then
  printf 'not run (image lacks their dependencies): %s\n' "${unavailable[*]}"
fi
exit "$status"
