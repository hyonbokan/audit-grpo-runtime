# Shared launcher pieces. Source after setting code_root and runtime_root.

CPU_IMAGE=audit-grpo-deps:cu129-v1
# Long-running judge/backup container that holds the credentials; code is copied in per phase.
LIVE_CONTAINER=audit-grpo-live
LIVE_CODE=/opt/audit-grpo-training-code
# Frozen backend dependencies, configs and data mounted as /workspace.
FROZEN_WORKSPACE="$runtime_root/workspace"
# The cache is an absolute host path, so it lives in the git-ignored host.env.
[[ -f "$runtime_root/host.env" ]] || {
  echo "Missing $runtime_root/host.env: copy host.env.example and set this host's values" >&2
  exit 2
}
source "$runtime_root/host.env"
CACHE=${AUDIT_GRPO_CACHE:?host.env must set AUDIT_GRPO_CACHE to an absolute cache directory}
# The cache mounts inside the read-only workspace, where Docker cannot create the mountpoint.
mkdir -p "$FROZEN_WORKSPACE/cache"

# Worker mounts and environment; a launcher appends its config mount.
mounts=(-v "$code_root:/training-code:ro" -v "$runtime_root:/runtime:ro"
  -v "$FROZEN_WORKSPACE:/workspace:ro"
  -v "$CACHE:/workspace/cache" -v "$CACHE:$CACHE"
  -v "$runtime_root/results:/results")
environment=(-e PYTHONPATH=/training-code:/training-code/inference:/workspace/code/backend
  -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1
  -e HF_HUB_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1
  -e HF_HOME=/workspace/cache/huggingface -e VLLM_USE_FLASHINFER_SAMPLER=0
  # Compiled graphs and kernels outlive each container, so an engine start reuses them.
  -e VLLM_CACHE_ROOT=/workspace/cache/vllm -e TRITON_CACHE_DIR=/workspace/cache/triton)

# Some hosts report GPU peer access but corrupt peer copies (IOMMU translation between root
# ports); preflight.sh checks the data. "disabled" preloads a shim that reports no peer
# access, so CUDA stages copies through host memory. Compiled once per shim source.
PEER_ACCESS_SETTINGS=(enabled disabled)
case "${AUDIT_GRPO_PEER_ACCESS:-}" in
enabled) ;;
disabled)
  shim_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/peer_access"
  shim_hash=$(sha256sum "$shim_root/no_peer_access.c") || exit 2
  shim="no-peer-access-${shim_hash:0:12}.so"
  if [[ ! -f "$CACHE/peer-access/$shim" ]]; then
    mkdir -p "$CACHE/peer-access"
    docker run --rm --network none -v "$shim_root:/src:ro" \
      -v "$CACHE/peer-access:/out" "$CPU_IMAGE" \
      gcc -shared -fPIC -O2 -o "/out/$shim" /src/no_peer_access.c
  fi
  environment+=(-e "LD_PRELOAD=/workspace/cache/peer-access/$shim")
  ;;
*)
  printf -v allowed '%s, ' "${PEER_ACCESS_SETTINGS[@]}"
  echo "host.env must set AUDIT_GRPO_PEER_ACCESS to one of: ${allowed%, }" \
    "(got '${AUDIT_GRPO_PEER_ACCESS:-}'); run preflight.sh to choose" >&2
  exit 2
  ;;
esac

# Seconds left before $end; fails with the deadline status once it has passed.
remaining() {
  local left=$((end - $(date +%s)))
  ((left > 0)) || return 124
  printf '%s\n' "$left"
}

# Deadline timers: a detached watchdog per run stops the named containers at the deadline,
# even if the launcher shell dies. It needs no root or systemd, so it works on any host.
DEADLINE_TIMERS="$runtime_root/results/deadline-timers"

# The record holds the watchdog's pid, the epoch second it fires and the containers.
start_deadline_timer() {
  local timer=$1 seconds=$2 record="$DEADLINE_TIMERS/$1.pid" waited
  shift 2
  stop_deadline_timer "$timer"
  mkdir -p "$DEADLINE_TIMERS"
  # $0 names the timer, so stop_deadline_timer can tell a reused pid from the watchdog.
  setsid bash -c 'record=$1 seconds=$2; shift 2
    printf "%s %s %s\n" "$$" "$(( $(date +%s) + seconds ))" "$*" > "$record.new"
    mv "$record.new" "$record"
    sleep "$seconds"
    docker stop -t 0 "$@"
    rm -f "$record"' "deadline-timer:$timer" "$record" "$seconds" "$@" \
    < /dev/null > /dev/null 2>&1 &
  for ((waited = 0; waited < 50; waited++)); do
    [[ -s "$record" ]] && return 0
    sleep 0.1
  done
  echo "Deadline timer $timer did not start" >&2
  return 1
}

stop_deadline_timer() {
  local record="$DEADLINE_TIMERS/$1.pid" pid
  [[ -f "$record" ]] || return 0
  read -r pid _ < "$record"
  if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -qF -- "deadline-timer:$1 "; then
    kill -TERM -- "-$pid" 2>/dev/null || true
  fi
  rm -f "$record"
}

# Stop every started container and keep its log and final state under $1.
record_containers() {
  local out=$1 container
  for container in "${containers[@]}"; do
    docker stop -t 0 "$container" >/dev/null 2>&1 || true
    docker logs "$container" > "$out/$container.log" 2>&1 || true
    docker inspect "$container" --format '{{json .State}}' > "$out/$container-state.json" 2>&1 || true
  done
}

# Copy the current code into the live container; run before live_exec, never concurrently.
sync_live_code() {
  docker exec "$LIVE_CONTAINER" mkdir -p "$LIVE_CODE"
  docker cp "$code_root/." "$LIVE_CONTAINER:$LIVE_CODE"
}

live_exec() {
  docker exec -w /tmp \
    -e PYTHONPATH="$LIVE_CODE:$LIVE_CODE/inference:/workspace/code/backend" \
    -e PYTHONDONTWRITEBYTECODE=1 "$LIVE_CONTAINER" "$@"
}
