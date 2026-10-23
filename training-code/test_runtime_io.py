"""Result directories: the run directory's owner can write beside what containers write."""

import runtime_io


def test_a_new_result_directory_belongs_to_the_run_directorys_owner(tmp_path, monkeypatch):
    for name in ("run", "artifacts", "runtime/runs"):
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / "run/SOURCE_SHA256SUMS").write_text("frozen\n")
    config = {
        "runtime_host": str(tmp_path / "runtime"),
        "artifact_host": str(tmp_path / "artifacts"),
        "run_host": str(tmp_path / "run"),
    }
    # The controller runs as root; the host launcher, as the owner of run_host, writes logs here.
    owner = (tmp_path / "run").stat()
    changed = []
    monkeypatch.setattr(runtime_io.os, "chown", lambda *args: changed.append(args))
    host = runtime_io.new_result_dir(config, "run-visit-001")
    assert host == tmp_path / "runtime/runs/run-visit-001"
    assert changed == [(tmp_path / "artifacts/run-visit-001", owner.st_uid, owner.st_gid)]
    assert (host / "SOURCE_SHA256SUMS").read_text() == "frozen\n"


def test_a_phase_directory_belongs_to_its_cycle_directorys_owner(tmp_path, monkeypatch):
    # The launcher archives a failed attempt by moving this directory, which needs write access.
    owner = tmp_path.stat()
    changed = []
    monkeypatch.setattr(runtime_io.os, "chown", lambda *args: changed.append(args))
    runtime_io.new_phase_dir(tmp_path / "rollout")
    assert (tmp_path / "rollout").is_dir()
    assert changed == [(tmp_path / "rollout", owner.st_uid, owner.st_gid)]
