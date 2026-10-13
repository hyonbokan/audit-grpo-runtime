#!/usr/bin/env bash
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runtime_root=$(dirname -- "$code_root")
run_root=$(realpath -- "${1:?Existing result directory required}")
config=$(realpath -- "${2:?Reviewed experiment config required}")
[[ "$run_root" == "$runtime_root/results/"* && -f "$config" ]]
[[ ! -e "$run_root/state.json" && ! -e "$run_root/launcher-exit.txt" ]]
cd -- "$code_root"
sha256sum --check "$run_root/SOURCE_SHA256SUMS" > "$run_root/source-verification.log"
phase=prepare
cleanup() {
  local status=$?
  trap - EXIT
  printf '%s\n' "$status" > "$run_root/launcher-exit.txt"
  printf '%s\n' "$phase" > "$run_root/last-phase.txt"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cpu() {
  docker run --rm --user "$(id -u):$(id -g)" --network none \
    -v "$runtime_root:$runtime_root" -v /ephemeral/audit-grpo-cache:/ephemeral/audit-grpo-cache \
    -v "$runtime_root/results:/results" \
    -e PYTHONPATH="$code_root:$code_root/inference:$runtime_root/workspace/code/backend" \
    -e PYTHONDONTWRITEBYTECODE=1 audit-grpo-deps:pilot-v4-conv \
    /opt/pilot/gpu/bin/python -m experiment "$@"
}
panel() {
  local role=$1 label=$2 folders folder
  phase="$role-$label"
  folders=$(cpu panel --config "$config" --out "$run_root" --role "$role" --label "$label")
  while IFS= read -r folder; do
    bash "$code_root/launch-evaluation.sh" "$folder" "$folder/config.json" > "$folder/launcher.log" 2>&1
  done <<< "$folders"
  cpu record-panel --config "$config" --out "$run_root" --role "$role" --label "$label" >> "$run_root/panels.log"
}
cpu prepare --config "$config" --out "$run_root"
validate_due() {
  local actions label
  actions=$(cpu validation-actions --config "$config" --out "$run_root" "$@")
  while IFS= read -r label; do
    [[ -z "$label" ]] || panel validation "$label"
  done <<< "$actions"
}
validate_due
while true; do
  phase=training-prepare
  folder=$(cpu next --config "$config" --out "$run_root")
  [[ "$folder" != complete ]] || break
  phase=training
  bash "$code_root/launch-cycle.sh" "$folder" "$folder/config.json" > "$folder/launcher.log" 2>&1
  cpu record --config "$config" --out "$run_root" >> "$run_root/training-record.log"
  # Move completed immutable checkpoint data to the mounted ephemeral disk. Receipts stay in results.
  if [[ -d "$folder/update" ]]; then
    destination="/ephemeral/audit-grpo-cache/experiment-checkpoints/$(basename -- "$folder")"
    mkdir -p "$(dirname -- "$destination")"
    [[ ! -e "$destination" ]]
    sudo mv -- "$folder/update" "$destination"
    sudo ln -s -- "$destination" "$folder/update"
  fi
  validate_due
done
phase=final-validation
validate_due --finished
phase=selection
cpu lock --config "$config" --out "$run_root" > "$run_root/selection.log"
panel test checkpoint-2
selected_is_initial=$(docker run --rm --network none -v "$run_root:/experiment:ro" audit-grpo-deps:pilot-v4-conv \
  /opt/pilot/gpu/bin/python -c 'import json; print(json.load(open("/experiment/selection-lock.json"))["step"] == 2)')
if [[ "$selected_is_initial" != True ]]; then panel test selected; fi
panel test base
phase=report
cpu report --config "$config" --out "$run_root" > "$run_root/report.log"
phase=complete
