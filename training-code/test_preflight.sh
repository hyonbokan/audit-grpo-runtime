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
printf 'PASS: driver checked against the image locks\n'
