#!/usr/bin/env bash
# live-status.sh reports the newest run's state while Docker/GPU/systemd stay offline.
set -euo pipefail
runtime_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_root="$(mktemp -d)"
trap 'rm -rf -- "$fixture_root"' EXIT
mkdir -p "$fixture_root/bin" "$fixture_root/results/old-run" "$fixture_root/results/new-run"
cp "$runtime_root/live-status.sh" "$fixture_root/live-status.sh"
cat > "$fixture_root/bin/docker" <<'SH'
#!/usr/bin/env bash
if [[ "$1" == ps ]]; then
  printf 'audit-grpo-live: Up 1 hour\n'
fi
SH
cat > "$fixture_root/bin/nvidia-smi" <<'SH'
#!/usr/bin/env bash
printf '0 MiB, 81559 MiB, 0 %%\n'
SH
cat > "$fixture_root/bin/systemctl" <<'SH'
#!/usr/bin/env bash
printf 'audit-grpo-run-deadline.timer\n'
SH
chmod +x "$fixture_root/bin/"*
cat > "$fixture_root/results/old-run/state.json" <<'JSON'
{"current_step":2,"current_checkpoint":"/results/old/checkpoint-2","current_adapter_sha256":"a","completion_gate":true,"visits":[]}
JSON
sleep 1
cat > "$fixture_root/results/new-run/state.json" <<'JSON'
{"current_step":13,"current_checkpoint":"/results/new/checkpoint-13","current_adapter_sha256":"b","completion_gate":true,"visits":[{"decision":"update"},{"decision":"failed"}]}
JSON
printf '0\n' > "$fixture_root/results/new-run/training-exit.txt"
output="$(PATH="$fixture_root/bin:$PATH" bash "$fixture_root/live-status.sh")"
[[ "$output" == *'Newest training run: new-run'* && "$output" == *'"current_step": 13'* ]] || {
  printf 'FAIL: newest run state is missing\n%s\n' "$output" >&2
  exit 1
}
[[ "$output" == *'"failed_visits": 1'* && "$output" == *'Training finished with exit 0'* ]] || {
  printf 'FAIL: visit outcome or training exit is missing\n%s\n' "$output" >&2
  exit 1
}
[[ "$output" != *'checkpoint-2'* ]] || {
  printf 'FAIL: an older run was reported\n' >&2
  exit 1
}
printf 'PASS: newest run state reported\n'
