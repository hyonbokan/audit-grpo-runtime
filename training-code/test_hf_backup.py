"""Complete recovery bundles resolve links, exclude unfinished writes and verify hashes."""

import hashlib
import json

import pytest

import hf_backup


def test_bundle_resolves_checkpoint_files_and_detects_any_tampering(tmp_path):
    module = hf_backup
    source = tmp_path / "source"
    source.mkdir()
    real = tmp_path / "real"
    real.mkdir()
    (real / "optimizer.pt").write_bytes(b"optimizer state")
    (source / "optimizer.pt").symlink_to(real / "optimizer.pt")
    (source / "unfinished.tmp").write_text("partial")
    (source / ".env").write_text("secret")
    (source / "backup-live-upload.log").write_text("unfinished upload")
    bundle = tmp_path / "bundle"
    module.copy_complete_tree(source, bundle)
    assert (bundle / "optimizer.pt").is_file()
    assert not (bundle / "optimizer.pt").is_symlink()
    assert not (bundle / "unfinished.tmp").exists()
    assert not (bundle / ".env").exists()
    assert not (bundle / "backup-live-upload.log").exists()
    manifest = module.file_manifest(bundle)
    module.verify_files(bundle, manifest)
    (bundle / "optimizer.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError):
        module.verify_files(bundle, manifest)


def test_final_bundle_contains_completed_evaluation_evidence(tmp_path, monkeypatch):
    module = hf_backup
    # State deserialization is exercised against the real pinned checkpoint at launch.
    monkeypatch.setattr(module, "verify_resumable", lambda *_args: {"checked": True})
    run = tmp_path / "results/weekend"
    run.mkdir(parents=True)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    state = {
        "current_step": 2,
        "current_checkpoint": str(checkpoint),
        "visits": [],
        "current_adapter_sha256": hashlib.sha256(b"adapter").hexdigest(),
    }
    (run / "state.json").write_text(json.dumps(state))
    child = run.parent / "weekend-evaluation-phi"
    child.mkdir()
    (child / "comparison.json").write_text('{"status":"graded"}')
    for directory in ["corpus", "source", "workspace/code/backend", "workspace/configs"]:
        (tmp_path / directory).mkdir(parents=True)
    ledger = tmp_path / "ledger.json"
    ledger.write_text("{}")
    config = {
        "dataset_container": str(tmp_path / "corpus"),
        "judge_ledger": str(ledger),
        "cycle": {"learning_rate": 5e-06},
    }
    target = tmp_path / "bundle"
    module.create_bundle(config, run, target, tmp_path / "source", tmp_path / "workspace")
    assert (
        target / "evaluation/weekend-evaluation-phi/comparison.json"
    ).read_text() == '{"status":"graded"}'


def thin_fixture(tmp_path, monkeypatch, visits=3):
    module = hf_backup
    monkeypatch.setattr(module, "verify_resumable", lambda *_args: {"checked": True})
    monkeypatch.setattr(module, "RESULTS_ROOT", tmp_path / "results")
    run = tmp_path / "results/weekend"
    run.mkdir(parents=True)
    (run / "config.json").write_text("{}")
    entries = []
    for index in range(1, visits + 1):
        visit = tmp_path / f"results/weekend-visit-{index:03d}"
        (visit / "grading").mkdir(parents=True)
        (visit / "grading/rewards.json").write_text(f"[{index}]")
        for heavy in (
            f"update/checkpoint-{index + 1}/optimizer.pt",
            f"update/checkpoint-{index + 1}/adapter_model.safetensors",
            "rollout/mapped-adapter/adapter_model.safetensors",
        ):
            (visit / heavy).parent.mkdir(parents=True, exist_ok=True)
            (visit / heavy).write_bytes(b"heavy" * 100)
        entries.append({"root": f"/results/{visit.name}"})
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name, data in {
        "adapter_model.safetensors": b"adapter",
        "adapter_config.json": b"{}",
        "optimizer.pt": b"opt",
        "scheduler.pt": b"sched",
        "rng_state.pth": b"rng",
        "trainer_state.json": b"{}",
    }.items():
        (checkpoint / name).write_bytes(data)
    state = {
        "current_step": visits + 1,
        "current_checkpoint": str(checkpoint),
        "visits": entries,
        "current_adapter_sha256": hashlib.sha256(b"adapter").hexdigest(),
    }
    (run / "state.json").write_text(json.dumps(state))
    for directory in ["corpus", "source", "workspace/code/backend", "workspace/configs"]:
        (tmp_path / directory).mkdir(parents=True)
        (tmp_path / directory / "payload.txt").write_text(directory)
    ledger = tmp_path / "ledger.json"
    ledger.write_text("{}")
    config = {
        "dataset_container": str(tmp_path / "corpus"),
        "judge_ledger": str(ledger),
        "cycle": {"learning_rate": 5e-06},
    }

    def build(kind):
        target = tmp_path / f"bundle-{kind}"
        module.create_bundle(
            config, run, target, tmp_path / "source", tmp_path / "workspace", kind=kind
        )
        return {str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()}

    return module, build


def test_visit_labels_are_thin_and_everything_else_is_full(tmp_path):
    module = hf_backup
    assert module.backup_kind("visit-medium-day-20261010-visit-004") == "thin"
    for label in ("initial", "training-final", "evaluated-final"):
        assert module.backup_kind(label) == "full"


def test_thin_visit_backup_holds_one_adapter_and_only_the_newest_visit_record(
    tmp_path, monkeypatch
):
    _module, build = thin_fixture(tmp_path, monkeypatch)
    files = build("thin")
    assert (
        tmp_path / "bundle-thin/checkpoint/adapter_model.safetensors"
    ).read_bytes() == b"adapter"
    assert "visits/003/grading/rewards.json" in files
    assert not {name for name in files if name.startswith(("visits/001", "visits/002"))}
    assert not {n for n in files if n.endswith(("optimizer.pt", "scheduler.pt", "rng_state.pth"))}
    assert sorted(n for n in files if n.endswith("adapter_model.safetensors")) == [
        "checkpoint/adapter_model.safetensors"
    ]
    assert not {n for n in files if n.startswith(("frozen-", "training-code", "workspace-"))}


def test_full_backup_is_resumable_and_keeps_all_visit_records_without_repeating_weights(
    tmp_path, monkeypatch
):
    _module, build = thin_fixture(tmp_path, monkeypatch)
    files = build("full")
    assert {
        "checkpoint/optimizer.pt",
        "checkpoint/rng_state.pth",
        "frozen-corpus/payload.txt",
    } <= files
    for number in ("001", "002", "003"):
        assert f"visits/{number}/grading/rewards.json" in files
    assert sorted(
        n for n in files if n.endswith(("adapter_model.safetensors", "optimizer.pt"))
    ) == [
        "checkpoint/adapter_model.safetensors",
        "checkpoint/optimizer.pt",
    ]


def test_staging_is_removed_after_a_verified_upload_but_metadata_is_kept(tmp_path):
    module = hf_backup
    bundle = tmp_path / "visit-x"
    bundle.mkdir()
    (bundle / "a").write_text("a")
    for suffix in (".tar.gz", ".metadata.json"):
        (tmp_path / f"visit-x{suffix}").write_text("x")
    (tmp_path / "hf-download-cache").mkdir()
    (tmp_path / "hf-download-cache/blob").write_text("b")
    module.remove_staging(bundle)
    assert not bundle.exists()
    assert not (tmp_path / "visit-x.tar.gz").exists()
    assert not (tmp_path / "hf-download-cache").exists()
    assert (tmp_path / "visit-x.metadata.json").exists()
