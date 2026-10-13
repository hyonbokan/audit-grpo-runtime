import pytest


def test_exact_declared_npm_version_has_verified_registry_identity():
    import corpus_dependencies as d

    assert hasattr(d, "exact_package"), "Exact declared package recovery is missing"
    meta = {
        "name": "@openzeppelin/contracts",
        "version": "4.8.1",
        "dist": {
            "tarball": "https://registry.npmjs.org/@openzeppelin/contracts/-/contracts-4.8.1.tgz",
            "integrity": "sha512-AAAA",
        },
    }
    pin = d.exact_package("@openzeppelin/contracts", "4.8.1", meta)
    assert pin["version"] == "4.8.1"
    for selector in ["^4.8.1", "~4.8.1", "latest", "4.9.0"]:
        with pytest.raises(ValueError):
            d.exact_package("@openzeppelin/contracts", selector, meta)
