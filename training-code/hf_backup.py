"""Versioned complete recovery archives with immutable HF download verification."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from runtime_io import read_json
from runtime_io import save_json as save
from runtime_io import sha256_file as digest

RESULTS_ROOT = Path("/results")
# Per-visit weights and optimizer state are held once, in the checkpoint directory.
HEAVY_NAMES = {"adapter_model.safetensors", "optimizer.pt"}
RESUME_ONLY_NAMES = {"optimizer.pt", "scheduler.pt", "rng_state.pth"}
VERIFIED_STATUS = "complete_resumable_archive_download_verified"


def backup_kind(label: str) -> str:
    """Visit labels are thin per-visit records; initial and final labels are complete."""
    return "thin" if label.startswith("visit-") else "full"


def backed_up_visits(run: Path) -> int:
    """How many of the run's visits its newest verified backup covers."""
    receipts = (read_json(path) for path in run.glob("backup-*-receipt.json"))
    return max((r["visits"] for r in receipts if r["status"] == VERIFIED_STATUS), default=0)


def copy_complete_tree(source: Path, target: Path, omit: frozenset | set = frozenset()) -> None:
    """Resolve recovery links and omit credential files, caches and unfinished writes."""

    def ignored(_directory, names):
        return [
            name
            for name in names
            if (
                name in omit
                or name in {".git", ".cache", "__pycache__", ".pytest_cache", "node_modules"}
                or name.endswith((".tmp", ".partial", ".pyc"))
                or (name.startswith("backup-") and name.endswith(".log"))
                or ".env" in name
                or name in {"token", "judge.env", "credentials.json"}
            )
        ]

    shutil.copytree(source, target, symlinks=False, ignore=ignored)


def file_manifest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "FILE_SHA256.json"
    }


def verify_files(root: Path, files: dict[str, str]) -> None:
    for relative, expected in files.items():
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError(f"Recovery file verification failed: {relative}")


def verify_resumable(checkpoint: Path, step: int, learning_rate: float) -> dict:
    import torch
    from safetensors import safe_open

    required = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "continuation_schedule.json",
    )
    for name in required:
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    if json.loads((checkpoint / "trainer_state.json").read_text())["global_step"] != step:
        raise ValueError("Recovery optimizer step differs")
    optimizer = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    # Only the step-0 checkpoint has an empty optimizer state.
    if {int(v["step"]) for v in optimizer["state"].values()} != ({step} if step else set()):
        raise ValueError("Recovery optimizer counters differ")
    scheduler = torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True)
    if any(lr != learning_rate for lr in scheduler["_last_lr"]):
        raise ValueError("Recovery learning rate differs")
    rng = torch.load(checkpoint / "rng_state.pth", map_location="cpu", weights_only=False)
    if not {"python", "numpy", "cpu", "cuda"}.issubset(rng):
        raise ValueError("Recovery RNG state is incomplete")
    with safe_open(checkpoint / "adapter_model.safetensors", framework="pt") as tensors:
        count = len(list(tensors.keys()))
    return {
        "state_deserialization": "passed",
        "optimizer_step": step,
        "adapter_tensors": count,
        "independent_machine_restore_tested": False,
    }


def create_bundle(
    config: dict, run: Path, out: Path, source: Path, workspace: Path, kind: str = "full"
) -> dict:
    """A thin bundle holds the visits since the last verified backup and the adapter."""
    thin = kind == "thin"
    state = json.loads((run / "state.json").read_text())
    checkpoint = Path(state["current_checkpoint"])
    if digest(checkpoint / "adapter_model.safetensors") != state["current_adapter_sha256"]:
        raise ValueError("Latest accepted adapter changed before backup")
    out.mkdir(parents=True, exist_ok=False)
    copy_complete_tree(checkpoint, out / "checkpoint", RESUME_ONLY_NAMES if thin else set())
    copy_complete_tree(run, out / "experiment")
    for child in sorted(run.parent.glob(run.name + "-evaluation*")):
        if not thin:
            copy_complete_tree(child, out / "evaluation" / child.name)
    visits = state["visits"]
    first = backed_up_visits(run) if thin else 0
    for index in range(first, len(visits)):
        path = RESULTS_ROOT / Path(visits[index]["root"]).name
        copy_complete_tree(path, out / "visits" / f"{index + 1:03d}", HEAVY_NAMES)
    pending_file = run / "pending-training.json"
    if pending_file.exists():
        pending = json.loads(pending_file.read_text())
        visited = {Path(v["root"]).name for v in state["visits"]}
        name = Path(pending["root"]).name
        if name not in visited and (RESULTS_ROOT / name).exists():
            copy_complete_tree(RESULTS_ROOT / name, out / "unaccepted-pending-visit", HEAVY_NAMES)
    if not thin:
        copy_complete_tree(Path(config["dataset_container"]), out / "frozen-corpus")
        copy_complete_tree(source, out / "training-code")
        copy_complete_tree(workspace / "code/backend", out / "frozen-backend")
        copy_complete_tree(workspace / "configs", out / "workspace-configs")
    shutil.copyfile(Path(config["judge_ledger"]), out / "judge-budget.json")
    for label, ledger in config.get("judge_ledgers", {}).items():
        shutil.copyfile(Path(ledger), out / f"judge-budget-{label}.json")
    if thin:
        evidence = {"kind": "thin", "optimizer_state_included": False}
    else:
        evidence = verify_resumable(
            out / "checkpoint", state["current_step"], config["cycle"]["learning_rate"]
        )
    save(out / "RESTORE_CHECK.json", evidence)
    files = file_manifest(out)
    save(out / "FILE_SHA256.json", files)
    verify_files(out, files)
    return {
        "kind": kind,
        "step": state["current_step"],
        "visits": len(visits),
        "adapter_sha256": state["current_adapter_sha256"],
        "files": files,
        "restore_check": evidence,
    }


def remove_staging(bundle: Path) -> None:
    """Drop the upload's local copies; the metadata file stays as the record."""
    shutil.rmtree(bundle, ignore_errors=True)
    shutil.rmtree(bundle.with_name(bundle.name + "-download-verified"), ignore_errors=True)
    bundle.with_suffix(".tar.gz").unlink(missing_ok=True)
    shutil.rmtree(bundle.parent / "hf-download-cache", ignore_errors=True)


def upload_verified(bundle: Path, metadata: dict, config: dict, receipt_path: Path) -> dict:
    from dotenv import load_dotenv
    from huggingface_hub import HfApi, hf_hub_download

    load_dotenv(config["judge_env_file"], override=True)
    load_dotenv(config.get("hf_env_file", "/workspace/code/backend/.env"), override=False)
    archive = bundle.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz", compresslevel=1, dereference=True) as tar:
        tar.add(bundle, arcname="recovery")
    archive_sha = digest(archive)
    api = HfApi()
    repo = config["hf_repo"]
    if not api.repo_info(repo_id=repo, repo_type="model").private:
        raise ValueError("HF recovery destination must remain private")
    prefix = f"weekend/{config['run_id']}/{bundle.name}"
    commit = api.upload_file(
        repo_id=repo,
        repo_type="model",
        path_or_fileobj=str(archive),
        path_in_repo=prefix + "/recovery.tar.gz",
        commit_message=f"Verified weekend recovery step {metadata['step']}",
    )
    downloaded = Path(
        hf_hub_download(
            repo_id=repo,
            repo_type="model",
            revision=commit.oid,
            filename=prefix + "/recovery.tar.gz",
            cache_dir=str(bundle.parent / "hf-download-cache"),
        )
    )
    if digest(downloaded) != archive_sha:
        raise ValueError("Downloaded immutable recovery archive hash differs")
    restored = bundle.with_name(bundle.name + "-download-verified")
    restored.mkdir(exist_ok=False)
    with tarfile.open(downloaded) as tar:
        tar.extractall(restored, filter="data")
    verify_files(restored / "recovery", metadata["files"])
    if not api.repo_info(repo_id=repo, repo_type="model", revision=commit.oid).private:
        raise ValueError("Immutable recovery revision is not private")
    receipt = {
        "status": VERIFIED_STATUS,
        **metadata,
        "archive_sha256": archive_sha,
        "bytes": archive.stat().st_size,
        "repo_id": repo,
        "commit": commit.oid,
        "path": prefix,
        "utc": datetime.now(UTC).isoformat(),
    }
    save(receipt_path, receipt)
    # The immutable archive remains; remove the duplicate extraction after verification.
    shutil.rmtree(restored)
    remove_staging(bundle)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--phase", choices=("prepare", "upload", "discard"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    out = Path(config["backup_container"]) / args.label
    if args.phase == "prepare":
        metadata = create_bundle(
            config, args.run, out, args.source, args.workspace, backup_kind(args.label)
        )
        save(out.with_suffix(".metadata.json"), metadata)
        print(json.dumps({"snapshot_prepared": args.label, "step": metadata["step"]}))
    elif args.phase == "discard":
        # A failed visit upload leaves its staging behind; the run's own files are untouched.
        remove_staging(out)
        print(json.dumps({"staging_discarded": args.label}))
    else:
        metadata = json.loads(out.with_suffix(".metadata.json").read_text())
        receipt = upload_verified(
            out, metadata, config, args.run / f"backup-{args.label}-receipt.json"
        )
        print(json.dumps({key: receipt[key] for key in ("status", "step", "commit", "bytes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
