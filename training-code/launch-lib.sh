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

# Worker mounts and environment; a launcher appends its config mount.
mounts=(-v "$code_root:/training-code:ro" -v "$runtime_root:/runtime:ro"
  -v "$FROZEN_WORKSPACE:/workspace:ro"
  -v "$CACHE:/workspace/cache" -v "$CACHE:$CACHE"
  -v "$runtime_root/results:/results")
environment=(-e PYTHONPATH=/training-code:/training-code/inference:/workspace/code/backend
  -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1
  -e HF_HUB_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1
  -e HF_HOME=/workspace/cache/huggingface -e VLLM_USE_FLASHINFER_SAMPLER=0)

# Seconds left before $end; fails with the deadline status once it has passed.
remaining() {
  local left=$((end - $(date +%s)))
  ((left > 0)) || return 124
  printf '%s\n' "$left"
}

# A transient systemd timer stops the named containers at the deadline, even if this shell dies.
start_deadline_timer() {
  local timer=$1 seconds=$2
  shift 2
  # A previous attempt's transient timer would block reusing the unit name.
  stop_deadline_timer "$timer"
  sudo systemctl reset-failed "$timer.timer" "$timer.service" >/dev/null 2>&1 || true
  sudo systemd-run --quiet --unit "$timer" --on-active="${seconds}s" /usr/bin/docker stop -t 0 "$@"
  sudo systemctl is-active --quiet "$timer.timer"
}

stop_deadline_timer() {
  sudo systemctl stop "$1.timer" >/dev/null 2>&1 || true
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
