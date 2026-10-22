"""Archive a failed cycle attempt so `launch-cycle.sh --resume` reuses completed phases."""

# Runs with the host interpreter, so it imports only the stdlib.

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

PHASES = ("rollout", "grading", "update")
COMPLETE_STATUS = {"rollout": "versioned_rollout_saved", "update": "corrected_update_saved"}
# Recomputed from phase outputs on every attempt.
ATTEMPT_FILES = ("launcher-exit.txt", "last-phase.txt", "group-decision.json", "cycle-result.json")


def completed_phases(root: Path) -> dict[str, bool]:
    """A phase is reusable when its receipt is complete and every earlier phase is reused."""
    complete = {}
    for phase in PHASES:
        if phase == "grading":
            done = (root / "grading/rewards.json").is_file()
        else:
            receipt = root / phase / "receipt.json"
            done = receipt.is_file() and (
                json.loads(receipt.read_text()).get("status") == COMPLETE_STATUS[phase]
            )
        complete[phase] = done and all(complete.values())
    return complete


def archive_failed_attempt(root: Path, config: dict) -> Path:
    """Move a failed attempt's logs and incomplete phase outputs under attempts/."""
    if not (root / "prepared.json").is_file():
        raise ValueError("Resume requires a prepared cycle")
    if json.loads((root / "prepared.json").read_text()) != config:
        raise ValueError("Resume configuration differs from prepared.json")
    if config.get("schedule"):
        raise ValueError("Resume supports single-step cycles only")
    complete = completed_phases(root)
    attempts = root / "attempts"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = attempts / f"{len(list(attempts.glob('*'))) + 1:02d}-{stamp}"
    archive.mkdir(parents=True)
    # Logs of reused phases stay beside the outputs they describe.
    reused = tuple(
        f"{phase}{suffix}"
        for phase, done in complete.items()
        if done
        for suffix in (".log", "-seconds.txt", "-state.json")
    )
    moved = [phase for phase, done in complete.items() if not done and (root / phase).exists()]
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.name.endswith(reused):
            continue
        attempt_log = path.suffix in (".log", ".txt") and path.name not in (
            "launcher.log",
            "source-verification.log",
        )
        if path.name in ATTEMPT_FILES or attempt_log or path.name.endswith("-state.json"):
            moved.append(path.name)
    for name in moved:
        shutil.move(root / name, archive / name)
    (archive / "resume.json").write_text(
        json.dumps({"reused": complete, "archived": moved}, indent=2) + "\n"
    )
    return archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("archive",))
    parser.add_argument("root", type=Path)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    archive = archive_failed_attempt(args.root, json.loads(args.config.read_text()))
    reused = json.loads((archive / "resume.json").read_text())["reused"]
    print(
        f"Resuming {args.root.name}: reusing {[p for p, d in reused.items() if d]}; "
        f"previous attempt archived in {archive}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
