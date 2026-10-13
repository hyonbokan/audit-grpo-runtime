from corpus_sources import collect_sources, installed_mappings, project_mappings


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
    assert set(result["source_hashes"]) == {
        "Main.sol",
        "node_modules/@vendor/old/A.sol",
        "node_modules/@vendor/new/B.sol",
    }


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
    assert set(result["source_hashes"]) == {
        "Main.sol",
        "lib/solmate/src/A.sol",
        "packages/v2-library/contracts/B.sol",
    }
