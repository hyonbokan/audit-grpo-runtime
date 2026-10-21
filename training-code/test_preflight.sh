#!/usr/bin/env bash
# preflight.sh compares the driver with the CUDA version of the committed image locks.
set -euo pipefail
runtime_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_root="$(mktemp -d)"
trap 'rm -rf -- "$fixture_root"' EXIT
mkdir -p "$fixture_root/bin"
printf '#!/usr/bin/env bash\nexit 1\n' > "$fixture_root/bin/docker"
check() {
  printf '#!/usr/bin/env bash\n[[ "$*" == *driver_version* ]] && echo %s\n' "$1" \
    > "$fixture_root/bin/nvidia-smi"
  chmod +x "$fixture_root/bin/"*
  PATH="$fixture_root/bin:$PATH" bash "$runtime_root/preflight.sh" | grep -F driver || true
}
old="$(check 570.86.10)"
[[ "$old" == 'FAIL  driver 570.86.10 is older than 575, which CUDA 12.9 images need;'* ]] || {
  printf 'FAIL: an old driver was not rejected\n%s\n' "$old" >&2
  exit 1
}
new="$(check 580.65.06)"
[[ "$new" == 'PASS  driver 580.65.06 runs the CUDA 12.9 images (needs >= 575)' ]] || {
  printf 'FAIL: a newer driver was not accepted\n%s\n' "$new" >&2
  exit 1
}
# Peer copies: a fixture tree with two fake GPUs and a docker that reports the probe result.
tree="$fixture_root/tree"
mkdir -p "$tree/training-code/fixtures" "$tree/training-code/peer_access" "$tree/images" "$tree/bin"
cp "$runtime_root/preflight.sh" "$runtime_root/.env.example" "$tree/"
cp "$runtime_root/training-code/launch-lib.sh" "$tree/training-code/"
cp "$runtime_root/training-code/fixtures/cycle.json" "$tree/training-code/fixtures/"
cp "$runtime_root"/images/gpu-requirements-*.lock "$tree/images/"
printf '#!/usr/bin/env bash\n[[ "$*" == *driver_version* ]] && echo 580.65.06\n[[ "$*" == *index* ]] && printf "0\\n1\\n"\nexit 0\n' \
  > "$tree/bin/nvidia-smi"
printf '#!/usr/bin/env bash\n[[ "$1" == run ]] && echo "corrupt $CORRUPT"\nexit 0\n' > "$tree/bin/docker"
chmod +x "$tree/bin/"*
peer() {
  printf 'AUDIT_GRPO_CACHE=%s\nAUDIT_GRPO_PEER_ACCESS=%s\n' "$fixture_root" "$1" > "$tree/host.env"
  CORRUPT=$2 PATH="$tree/bin:$PATH" bash "$tree/preflight.sh" | grep -i peer || true
}
expect() {
  [[ "$1" == *"$2"* ]] || { printf 'FAIL: expected %s\n%s\n' "$2" "$1" >&2; exit 1; }
}
expect "$(peer enabled 12)" 'FAIL  peer copies corrupt (12 elements); set AUDIT_GRPO_PEER_ACCESS=disabled'
expect "$(peer enabled 0)" 'PASS  peer copies intact (AUDIT_GRPO_PEER_ACCESS=enabled)'
expect "$(peer disabled 0)" 'PASS  peer copies intact (AUDIT_GRPO_PEER_ACCESS=disabled)'
expect "$(peer disabled 3)" 'FAIL  peer copies corrupt (3 elements) even through the no-peer-access shim'
expect "$(peer maybe 0)" 'FAIL  host.env must set AUDIT_GRPO_PEER_ACCESS to enabled or disabled'
printf 'PASS: driver checked against the image locks; peer copies checked against host.env\n'

