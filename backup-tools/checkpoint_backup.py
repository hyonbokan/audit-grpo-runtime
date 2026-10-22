"""Off-VM backups independent of the frozen GPU training implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

REQUIRED = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def prepare_snapshot(runtime: Path, state: dict, out: Path) -> dict:
    step, expected = state["current_step"], state["current_adapter_sha256"]
    accepted = [
        v
        for v in state["visits"]
        if v.get("decision") == "update"
        and v["update"]["global_step"] == step
        and v["update"]["adapter_sha256"] == expected
    ]
    if not accepted:
        raise ValueError("No accepted update matches the current checkpoint")
    name = state["current_checkpoint"]
    if not name.startswith("/results/") or ".." in Path(name).parts:
        raise ValueError("Checkpoint path is outside experiment results")
    source = runtime / "runs" / Path(name).relative_to("/results")
    for name in REQUIRED:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    if digest(source / "adapter_model.safetensors") != expected:
        raise ValueError("Checkpoint adapter hash differs")
    if json.loads((source / "trainer_state.json").read_text())["global_step"] != step:
        raise ValueError("Saved trainer step differs from accepted progress")
    out.mkdir(parents=True, exist_ok=False)
    target = out / "checkpoint"
    target.mkdir()
    for name in REQUIRED:
        shutil.copyfile(source / name, target / name)
    for name in ("continuation_schedule.json", "README.md"):
        if (source / name).is_file():
            shutil.copyfile(source / name, target / name)
    save(out / "state.json", state)
    manifest = {
        "step": step,
        "adapter_sha256": expected,
        "files": {
            str(p.relative_to(out)): digest(p)
            for p in sorted(out.rglob("*"))
            if p.is_file()
        },
    }
    save(out / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--backups", type=Path, required=True)
    parser.add_argument("--dotenv", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--deadline", required=True)
    args = parser.parse_args()
    if datetime.now(timezone.utc) >= datetime.fromisoformat(args.deadline):
        print("Backup deadline reached")
        return 0
    state = json.loads((args.experiment / "state.json").read_text())
    previous = args.backups / "latest-checkpoint-backup.json"
    initial = json.loads((args.backups / "hf-backup-receipt.json").read_text())
    saved_step = (
        json.loads(previous.read_text())["step"]
        if previous.exists()
        else initial["completed_optimizer_step"]
    )
    if state["current_step"] <= saved_step:
        print(json.dumps({"status": "no_new_accepted_checkpoint", "step": saved_step}))
        return 0
    snapshot = (
        args.backups
        / f"accepted-checkpoint-{state['current_step']}-{state['current_adapter_sha256'][:12]}"
    )
    if snapshot.exists():
        manifest = json.loads((snapshot / "manifest.json").read_text())
        for name, expected in manifest["files"].items():
            if digest(snapshot / name) != expected:
                raise ValueError("Prepared backup snapshot changed")
    else:
        manifest = prepare_snapshot(args.runtime, state, snapshot)
    plan = json.loads((args.experiment / "source-plan.json").read_text())
    budget = Path(plan["access"]["local_judge_budget"]["path"])
    if not budget.is_file():
        raise FileNotFoundError("Shared judge ledger unavailable for backup")
    shutil.copyfile(budget, snapshot / "shared-judge-budget.json")
    manifest["files"]["shared-judge-budget.json"] = digest(
        snapshot / "shared-judge-budget.json"
    )
    save(snapshot / "manifest.json", manifest)
    from dotenv import load_dotenv
    from huggingface_hub import HfApi

    load_dotenv(args.dotenv, override=True)
    api = HfApi()
    if not api.repo_info(repo_id=args.repo, repo_type="model").private:
        raise ValueError("Backup destination must remain private")
    prefix = f"checkpoints/checkpoint-{manifest['step']}"
    commit = api.upload_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=str(snapshot),
        path_in_repo=prefix,
        commit_message=f"Back up accepted checkpoint{manifest['step']}",
        ignore_patterns=[".cache/**"],
    )
    info = api.repo_info(repo_id=args.repo, repo_type="model", files_metadata=True)
    if not info.private:
        raise ValueError("Remote privacy check failed")
    files = {f.rfilename: f for f in info.siblings}
    for name in ("checkpoint/adapter_model.safetensors", "checkpoint/optimizer.pt"):
        remote = files[prefix + "/" + name]
        lfs = remote.lfs
        actual = (
            lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        )
        if (
            actual != manifest["files"][name]
            or remote.size != (snapshot / name).stat().st_size
        ):
            raise ValueError("Remote checkpoint hash/size differs")
    receipt = {
        "status": "private_remote_checkpoint_verified",
        "step": manifest["step"],
        "adapter_sha256": manifest["adapter_sha256"],
        "repo_id": args.repo,
        "path": prefix,
        "commit": commit.oid,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    save(previous, receipt)
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
