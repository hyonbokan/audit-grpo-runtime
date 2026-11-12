#!/usr/bin/env bash
# Start a reviewed training schedule: freeze the source, write the run config (and the step-0
# adapter on a fresh start), then run launch-training.sh detached. The run id is the schedule id.
# Usage: bash training-code/launch-segment.sh SCHEDULE_ID
set -euo pipefail

code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runtime_root=$(dirname -- "$code_root")
schedule=${1:?Usage: launch-segment.sh SCHEDULE_ID}
run_root="$runtime_root/runs/$schedule"
[[ ! -e "$run_root" ]] || {
  echo "$run_root exists; continue it with launch-training.sh RUN CONFIG --resume" >&2
  exit 2
}
cd -- "$code_root"
# The run freezes today's source: regenerate the identity, then verify it.
python3 provenance.py
sha256sum --check --quiet SOURCE_SHA256SUMS
source "$code_root/launch-lib.sh"
[[ -n "${AUDIT_GRPO_HF_REPO:-}" ]] || {
  echo "host.env must set AUDIT_GRPO_HF_REPO, the private repository for run backups" >&2
  exit 2
}
mkdir "$run_root"
# Host-owned outputs, so this host can add SOURCE_SHA256SUMS and logs beside them.
docker run --rm --network none --user "$(id -u):$(id -g)" "${mounts[@]}" "${environment[@]}" \
  "$CPU_IMAGE" /opt/pilot/gpu/bin/python -m training_config --schedule "$schedule" \
  --runtime-host "$runtime_root" --artifact-host "$CACHE/runs" \
  --hf-repo "$AUDIT_GRPO_HF_REPO" --judge-env-file "$JUDGE_ENV"
cp SOURCE_SHA256SUMS "$run_root/"
# The hashes say what ran; the commit and any uncommitted change say where it came from.
if commit=$(git -C "$runtime_root" rev-parse HEAD 2>/dev/null); then
  printf '%s\n' "$commit" > "$run_root/source-commit.txt"
  { git -C "$runtime_root" status --porcelain=v1 --ignored -- training-code
    git -C "$runtime_root" diff HEAD -- training-code
    # A hashed file git does not track runs too: record its content, unless it is ignored
    # (possibly private), which the status above lists by name only.
    while read -r _ name; do
      git ls-files --error-unmatch -- "$name" > /dev/null 2>&1 ||
        git check-ignore -q -- "$name" ||
        git diff --no-index -- /dev/null "$name" || true
    done < SOURCE_SHA256SUMS; } > "$run_root/source-uncommitted.diff"
else
  echo "not a git checkout" > "$run_root/source-commit.txt"
fi
setsid nohup bash "$code_root/launch-training.sh" "$run_root" "$run_root/config.json" \
  > "$run_root/launcher.out" 2>&1 < /dev/null &
echo "Started $schedule (launcher pid $!); progress: $run_root/launcher.out, ./live-status.sh"
