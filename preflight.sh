#!/usr/bin/env bash
# Checks this host against HOST_SETUP.md step 0 and reports what is still missing; starts no
# work and changes nothing. Exit 0 when nothing FAILs; WARN lines need a decision, not a fix.
# Usage: bash preflight.sh
set -uo pipefail
runtime_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
code_root="$runtime_root/training-code"
failures=0
pass() { printf 'PASS  %s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*"; failures=$((failures + 1)); }

# Lowest driver for each CUDA minor the image locks can target (NVIDIA's compatibility table).
declare -A MIN_DRIVER=([12.8]=570 [12.9]=575 [13.0]=580)
MIN_RAM_GIB=160
MIN_FREE_GIB=250
MIN_LINK_MBPS=1000
HOST_TOOLS=(docker jq uv python3 setsid)

# --- Images and driver: the locks bundle CUDA, so the host needs only a new enough driver.
image=$(sed -n 's/^CPU_IMAGE=//p' "$code_root/launch-lib.sh")
suffix=${image#*:}
cuda_tag=${suffix%%-*}
lock="$runtime_root/images/gpu-requirements-$cuda_tag.lock"
cuda=$(sed -n 's/^nvidia-cuda-runtime-cu1[0-9]==\([0-9]*\.[0-9]*\).*/\1/p' "$lock" 2>/dev/null)
need=${MIN_DRIVER[$cuda]:-}
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1)
if [[ -z "$cuda" || -z "$need" ]]; then
  fail "no CUDA runtime with a known driver minimum in ${lock#"$runtime_root/"} (known: ${!MIN_DRIVER[*]})"
elif [[ -z "$driver" ]]; then
  fail "nvidia-smi found no driver"
elif ((${driver%%.*} < need)); then
  fail "driver $driver is older than $need, which CUDA $cuda images need; re-lock for an older CUDA (HOST_SETUP.md step 5) rather than changing the driver"
else
  pass "driver $driver runs the CUDA $cuda images (needs >= $need)"
fi
while IFS=, read -r index name memory; do
  pass "GPU$index:$name,$memory"
done < <(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null)

# --- Host tools; a stale directory early on PATH (seen: ~/bin from an older OS) shadows them.
for tool in "${HOST_TOOLS[@]}"; do
  found=$(command -v "$tool") || { fail "$tool is not installed"; continue; }
  system=$(PATH=/usr/sbin:/usr/bin:/sbin:/bin command -v "$tool") || continue
  [[ "$found" == "$system" ]] || warn "$tool resolves to $found, not the system $system; check it works"
done
if python3 -c 'import ssl, sys; assert sys.version_info >= (3, 10)' 2>/dev/null; then
  pass "python3 on PATH is >= 3.10 with SSL"
else
  fail "python3 on PATH ($(command -v python3)) lacks SSL or is older than 3.10; pass a working one as HOST_PYTHON to images/build.sh"
fi
if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
  pass "docker runs without sudo and has the nvidia runtime"
else
  fail "docker is unreachable without sudo or lacks the nvidia runtime"
fi

# --- Memory, link speed.
ram=$(awk '/MemTotal/ {print int($2 / 1048576)}' /proc/meminfo)
((ram >= MIN_RAM_GIB)) && pass "RAM ${ram} GiB" || fail "RAM ${ram} GiB < ${MIN_RAM_GIB} GiB (update offloads frozen layers to host memory)"
interface=$(ip route show default 2>/dev/null | awk '{print $5; exit}')
speed=$(cat "/sys/class/net/$interface/speed" 2>/dev/null || echo unknown)
if [[ "$speed" =~ ^[0-9]+$ ]] && ((speed >= MIN_LINK_MBPS)); then
  pass "network $interface at $speed Mb/s"
else
  warn "network $interface at $speed Mb/s; ~70 GB of downloads take hours below $MIN_LINK_MBPS Mb/s (load saved images instead of rebuilding)"
fi

# --- Runtime settings: host.env, credentials (names only, never values).
if [[ -f "$runtime_root/host.env" ]] && cache=$(sed -n 's/^AUDIT_GRPO_CACHE=//p' "$runtime_root/host.env") && [[ -n "$cache" ]]; then
  pass "host.env names the cache $cache"
  free=$(df -BG --output=avail "$cache" 2>/dev/null | tail -n 1 | tr -dc 0-9)
  ((${free:-0} >= MIN_FREE_GIB)) && pass "cache volume ${free} GiB free" || warn "cache volume ${free:-?} GiB free < ${MIN_FREE_GIB} GiB"
else
  fail "host.env is missing or does not set AUDIT_GRPO_CACHE (copy host.env.example)"
  cache=
fi
env_file="$runtime_root/.env"
if [[ -f "$env_file" ]]; then
  mode=$(stat -c %a "$env_file")
  [[ "$mode" == 600 ]] && pass ".env mode 600" || fail ".env mode $mode; run chmod 600 .env"
  for key in $(sed -n 's/^\([A-Z_]*\)=.*/\1/p' "$runtime_root/.env.example"); do
    grep -qE "^$key=.+" "$env_file" || fail ".env has no value for $key"
  done
else
  fail ".env is missing (copy .env.example, fill it, chmod 600)"
fi

# --- Images and base model (presence only; HOST_SETUP.md steps 4-5 verify hashes).
for tag in "$image" "audit-grpo-vllm:$suffix"; do
  docker image inspect "$tag" > /dev/null 2>&1 && pass "image $tag present" || fail "image $tag missing (HOST_SETUP.md step 5)"
done
model=$(jq -r .model_id "$code_root/fixtures/cycle.json")
revision=$(jq -r .model_revision "$code_root/fixtures/cycle.json")
snapshot="$cache/huggingface/hub/models--${model//\//--}/snapshots/$revision"
if [[ -n "$cache" && -f "$snapshot/model.safetensors.index.json" ]]; then
  missing=$(jq -r '.weight_map[]' "$snapshot/model.safetensors.index.json" | sort -u |
    while read -r shard; do [[ -e "$snapshot/$shard" ]] || echo "$shard"; done | wc -l)
  ((missing == 0)) && pass "base model $model@${revision:0:8} has every shard" || fail "base model is missing $missing shards"
else
  fail "base model $model@${revision:0:8} is not in the cache (HOST_SETUP.md step 4)"
fi

printf '\n%d failure(s)\n' "$failures"
((failures == 0))
