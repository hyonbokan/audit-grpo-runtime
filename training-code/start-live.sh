#!/usr/bin/env bash
# Start the long-running judge container that holds the credentials (launchers docker exec
# into it); reuse it if it exists. It needs the network to reach the judge gateway.
# Usage: bash training-code/start-live.sh
set -euo pipefail

code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runtime_root=$(dirname -- "$code_root")
credentials="$runtime_root/.env"
# Run configs name this path as judge_env_file.
JUDGE_ENV=/opt/audit-grpo-secrets/judge.env
[[ -f "$credentials" ]] || {
  echo "Missing $credentials: copy .env.example, fill it and chmod 600 it" >&2
  exit 2
}
source "$code_root/launch-lib.sh"

case "$(docker inspect --format '{{.State.Status}}' "$LIVE_CONTAINER" 2>/dev/null || echo missing)" in
running) ;;
missing)
  docker run -d --name "$LIVE_CONTAINER" --restart unless-stopped \
    "${mounts[@]}" -v "$credentials:$JUDGE_ENV:ro" "${environment[@]}" \
    "$CPU_IMAGE" sleep infinity > /dev/null
  ;;
*) docker start "$LIVE_CONTAINER" > /dev/null ;;
esac
echo "$LIVE_CONTAINER is running"
