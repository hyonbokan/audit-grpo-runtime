import base64
import hashlib
import io
import tarfile
import zipfile

import pytest

from corpus_dependencies import (
    locked_package,
    extract_sources,
    installed_mappings,
    project_mappings,
    extract_git_sources,
)
from corpus_sources import collect_sources


def test_lock_selection_uses_project_selector_instead_of_picking_another_version():
    text = """"@vendor/contracts@^4.5.0":
  version "4.7.3"
  resolved "https://registry.npmjs.org/@vendor/contracts/-/contracts-4.7.3.tgz"
  integrity sha512-AAAA

"@vendor/contracts@4.3.1":
  version "4.3.1"
  resolved "https://registry.npmjs.org/@vendor/contracts/-/contracts-4.3.1.tgz"
  integrity sha512-BBBB
"""
    assert locked_package("@vendor/contracts", "^4.5.0", yarn=text)["version"] == "4.7.3"
    with pytest.raises(ValueError):
        locked_package("@vendor/contracts", "^4.0.0", yarn=text)


def test_npm_lock_uses_root_package_and_requires_integrity():
    lock = {
        "packages": {
            "node_modules/@vendor/contracts": {
                "version": "4.7.3",
                "resolved": "https://registry.npmjs.org/package.tgz",
                "integrity": "sha512-AAAA",
            }
        }
    }
    assert locked_package("@vendor/contracts", "^4.5.0", npm=lock)["version"] == "4.7.3"
    del lock["packages"]["node_modules/@vendor/contracts"]["integrity"]
    with pytest.raises(ValueError):
        locked_package("@vendor/contracts", "^4.5.0", npm=lock)


def test_transitive_package_can_use_only_one_unique_integrity_locked_version():
    text = """"@vendor/contracts@4.7.3":
  version "4.7.3"
  resolved "https://registry.npmjs.org/contracts-4.7.3.tgz"
  integrity sha512-AAAA
"""
    assert locked_package("@vendor/contracts", None, yarn=text)["version"] == "4.7.3"
    second = """
"@vendor/contracts@4.3.1":
  version "4.3.1"
  resolved "https://registry.npmjs.org/contracts-4.3.1.tgz"
  integrity sha512-BBBB
"""
    with pytest.raises(ValueError):
        locked_package("@vendor/contracts", None, yarn=text + second)


def tar_bytes(files):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return stream.getvalue()


def integrity(raw):
    return "sha512-" + base64.b64encode(hashlib.sha512(raw).digest()).decode()


def test_verified_package_extracts_source_without_installing_or_running_scripts(tmp_path):
    raw = tar_bytes(
        {
            "package/contracts/A.sol": b"contract A {}",
            "package/package.json": b'{"scripts":{"postinstall":"evil"}}',
            "package/install.js": b"evil",
        }
    )
    hashes = extract_sources(raw, integrity(raw), tmp_path)
    assert (tmp_path / "contracts/A.sol").read_text() == "contract A {}"
    assert not (tmp_path / "install.js").exists()
    assert set(hashes) == {"contracts/A.sol", "package.json"}


def test_dependency_hash_mismatch_cannot_write_any_source(tmp_path):
    raw = tar_bytes({"package/A.sol": b"contract A {}"})
    with pytest.raises(ValueError, match="integrity"):
        extract_sources(raw, integrity(raw + b"x"), tmp_path)
    assert not list(tmp_path.iterdir())


def test_dependency_archive_cannot_escape_package_root(tmp_path):
    raw = tar_bytes({"package/../A.sol": b"contract A {}"})
    with pytest.raises(ValueError, match="path"):
        extract_sources(raw, integrity(raw), tmp_path)


def test_later_recovery_keeps_previously_installed_imports_resolvable(tmp_path):
    modules = tmp_path / "node_modules/@vendor"
    (modules / "old").mkdir(parents=True)
    (modules / "new").mkdir()
    (modules / "old/A.sol").write_text("contract A {}")
    (modules / "new/B.sol").write_text("contract B {}")
    (tmp_path / "Main.sol").write_text('import "@vendor/old/A.sol"; import "@vendor/new/B.sol";')
    mappings = installed_mappings(tmp_path, {"@vendor/new/": "node_modules/@vendor/new/"})
    result = collect_sources(tmp_path, ["Main.sol"], mappings)
    assert result["unresolved_imports"] == []
    assert len(result["source_hashes"]) == 3


def test_committed_foundry_remappings_and_workspace_packages_resolve_without_downloads(tmp_path):
    (tmp_path / "foundry.toml").write_text(
        '[profile.default]\nremappings = ["solmate/=lib/solmate/src/"]\n'
    )
    (tmp_path / "lib/solmate/src").mkdir(parents=True)
    (tmp_path / "lib/solmate/src/A.sol").write_text("contract A {}")
    (tmp_path / "package.json").write_text('{"workspaces":["packages/*"]}')
    (tmp_path / "packages/v2-library/contracts").mkdir(parents=True)
    (tmp_path / "packages/v2-library/package.json").write_text('{"name":"@example/v2-library"}')
    (tmp_path / "packages/v2-library/contracts/B.sol").write_text("contract B {}")
    (tmp_path / "Main.sol").write_text(
        'import "solmate/A.sol"; import "@example/v2-library/contracts/B.sol";'
    )
    result = collect_sources(tmp_path, ["Main.sol"], project_mappings(tmp_path))
    assert result["unresolved_imports"] == []
    assert len(result["source_hashes"]) == 3


def test_git_snapshot_must_match_pinned_tree_blob_before_source_is_written(tmp_path):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("repo-commit/A.sol", b"abc")
        archive.writestr("repo-commit/install.js", b"ignored")
    raw = stream.getvalue()
    # Git blob object hash of the literal bytes "abc", derived independently with git hash-object.
    expected = {"A.sol": "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f"}
    result = extract_git_sources(raw, expected, tmp_path)
    assert (tmp_path / "A.sol").read_bytes() == b"abc"
    assert not (tmp_path / "install.js").exists()
    assert result == {"A.sol": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}


def test_git_snapshot_with_wrong_object_hash_is_withheld(tmp_path):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("repo-commit/A.sol", b"modified")
    with pytest.raises(ValueError, match="Git blob"):
        extract_git_sources(
            stream.getvalue(), {"A.sol": "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f"}, tmp_path
        )


def test_malformed_submodule_cannot_abort_other_pinned_source_reviews(tmp_path):
    import corpus_dependencies as dependencies

    (tmp_path / ".gitmodules").write_text(
        '[submodule "broken"]\n url = https://github.com/vendor/broken\n'
        '[submodule "valid"]\n path = lib/valid\n url = https://github.com/vendor/valid\n'
        '[submodule "escape"]\n path = ../escape\n url = https://github.com/vendor/escape\n'
    )
    modules, errors = dependencies.submodule_records(tmp_path)
    assert modules == [("lib/valid", "https://github.com/vendor/valid")]
    assert set(errors) == {'submodule "broken"', 'submodule "escape"'}
