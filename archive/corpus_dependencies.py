"""Recover locked Solidity sources as data; never install or execute dependency packages."""

import base64
import configparser
import hashlib
import hmac
import io
import json
import re
import tarfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit


def submodule_records(root):
    config = configparser.ConfigParser()
    modules, errors = [], {}
    try:
        config.read(Path(root) / ".gitmodules")
    except configparser.Error as error:
        return [], {"configuration": str(error)}
    for section in config.sections():
        try:
            relative, url = config[section]["path"], config[section]["url"]
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts or "\\" in relative or not path.parts:
                raise ValueError("Submodule path escapes snapshot")
            modules.append((relative, url))
        except (KeyError, ValueError) as error:
            errors[section] = str(error)
    return modules, errors


def installed_mappings(root, base):
    mappings = dict(base)
    for vendor in (Path(root) / "node_modules").glob("*"):
        mappings.setdefault(vendor.name + "/", "node_modules/" + vendor.name + "/")
    return mappings


def project_mappings(root):
    root = Path(root).resolve()
    mappings = {}
    for path in root.rglob("remappings.txt"):
        if len(path.relative_to(root).parts) > 3:
            continue
        prefix = path.parent.relative_to(root).as_posix()
        prefix = "" if prefix == "." else prefix + "/"
        for line in path.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.strip().split("=", 1)
                if ":" not in key:
                    mappings[key] = prefix + value
    config = root / "foundry.toml"
    if config.exists():
        foundry = tomllib.loads(config.read_text())
        for remapping in foundry.get("profile", {}).get("default", {}).get("remappings", []):
            key, value = remapping.split("=", 1)
            if ":" not in key:
                mappings[key] = value
    manifest = root / "package.json"
    if manifest.exists():
        workspaces = json.loads(manifest.read_text()).get("workspaces", [])
        if isinstance(workspaces, dict):
            workspaces = workspaces.get("packages", [])
        for pattern in workspaces:
            if PurePosixPath(pattern).is_absolute() or ".." in PurePosixPath(pattern).parts:
                raise ValueError("Workspace path escapes snapshot")
            for workspace in root.glob(pattern):
                if not workspace.resolve().is_relative_to(root):
                    raise ValueError("Workspace path escapes snapshot")
                package = workspace / "package.json"
                if package.exists():
                    name = json.loads(package.read_text()).get("name")
                    if name:
                        mappings[name + "/"] = str(workspace.relative_to(root)) + "/"
    return installed_mappings(root, mappings)


def extract_git_sources(raw, expected, root):
    if len(raw) > 32 * 1024**2:
        raise ValueError("Git source archive exceeds download limit")
    selected, total = [], 0
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for index, member in enumerate(archive.infolist()):
            path = PurePosixPath(member.filename)
            if (
                index >= 10000
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in member.filename
            ):
                raise ValueError("Unsafe Git source archive path")
            if member.is_dir():
                continue
            total += member.file_size
            if total > 128 * 1024**2:
                raise ValueError("Git source archive exceeds expanded size limit")
            relative = str(PurePosixPath(*path.parts[1:]))
            if not (
                relative.endswith(".sol")
                or relative.endswith(".json")
                or relative in ["remappings.txt", ".gitmodules", "foundry.toml"]
            ):
                continue
            if member.file_size > 8 * 1024**2:
                raise ValueError("Git source file exceeds size limit")
            content = archive.read(member)
            blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            if expected.get(relative) != blob:
                raise ValueError("Git blob does not match pinned tree evidence")
            selected.append((relative, content))
    root = Path(root).resolve()
    hashes = {}
    for relative, content in selected:
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Git source path escapes destination")
        if target.exists() and target.read_bytes() != content:
            raise ValueError("Git recovery would overwrite different source bytes")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        hashes[relative] = hashlib.sha256(content).hexdigest()
    return hashes


def locked_package(name, selector, *, yarn=None, npm=None):
    package = None
    if npm is not None:
        package = npm.get("packages", {}).get("node_modules/" + name)
        package = package or npm.get("dependencies", {}).get(name)
    if package is None and yarn:
        choices = []
        for block in re.split(r"\n(?=\S)", yarn):
            lines = block.splitlines()
            if not lines:
                continue
            selectors = [key.strip().strip('"') for key in lines[0].rstrip(":").split(",")]
            if selector is None:
                relevant = any(key.startswith(name + "@") for key in selectors)
            else:
                relevant = name + "@" + selector in selectors
            if not relevant:
                continue
            package = {}
            for field in ["version", "resolved", "integrity"]:
                match = re.search(r"(?m)^  " + field + r'\s+"?([^"\n]+)', block)
                if match:
                    package[field] = match.group(1).strip()
            choices.append(package)
        unique = {
            tuple(choice.get(field) for field in ["version", "resolved", "integrity"])
            for choice in choices
        }
        if len(unique) != 1:
            raise ValueError("No unambiguous project-locked version and integrity")
        package = choices[0]
    if not package or not all(package.get(key) for key in ["version", "resolved", "integrity"]):
        raise ValueError("No unambiguous project-locked version and integrity")
    url = urlsplit(package["resolved"])
    if url.scheme != "https" or url.hostname not in ["registry.npmjs.org", "registry.yarnpkg.com"]:
        raise ValueError("Package is not locked to an approved public registry URL")
    if url.username or url.password:
        raise ValueError("Credential-bearing dependency URL is disallowed")
    return {key: package[key] for key in ["version", "resolved", "integrity"]}


def extract_sources(raw, expected, root):
    if len(raw) > 32 * 1024**2:
        raise ValueError("Dependency archive exceeds download limit")
    choices = expected.split()
    choices.sort(
        key=lambda value: {"sha512": 3, "sha256": 2, "sha1": 1}.get(value.split("-", 1)[0], 0),
        reverse=True,
    )
    algorithm, encoded = choices[0].split("-", 1)
    if algorithm not in ["sha512", "sha256", "sha1"]:
        raise ValueError("Unsupported dependency integrity algorithm")
    wanted = base64.b64decode(encoded, validate=True)
    if not hmac.compare_digest(hashlib.new(algorithm, raw).digest(), wanted):
        raise ValueError("Dependency integrity mismatch")
    root = Path(root).resolve()
    hashes, selected, total = {}, [], 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for index, member in enumerate(archive):
            path = PurePosixPath(member.name)
            if (
                index >= 10000
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in member.name
                or not path.parts
                or path.parts[0] != "package"
            ):
                raise ValueError("Unsafe dependency archive path or member count")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("Dependency links or special files are disallowed")
            total += member.size
            if total > 128 * 1024**2:
                raise ValueError("Dependency archive exceeds expanded size limit")
            relative = PurePosixPath(*path.parts[1:])
            if relative.suffix != ".sol" and str(relative) != "package.json":
                continue
            if member.size > 8 * 1024**2:
                raise ValueError("Dependency source file exceeds size limit")
            selected.append((member, relative))
        for member, relative in selected:
            target = (root / str(relative)).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Dependency source path escapes destination")
            content = archive.extractfile(member).read()
            if target.exists() and target.read_bytes() != content:
                raise ValueError("Dependency recovery would overwrite different source bytes")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            hashes[str(relative)] = hashlib.sha256(content).hexdigest()
    return hashes


def exact_package(name, selector, metadata):
    """Resolve an exact version in the pinned project manifest, never a version range."""
    if not isinstance(selector, str) or not re.fullmatch(r"\d+\.\d+\.\d+(?:-[\w.-]+)?", selector):
        raise ValueError("Dependency selector is not an exact audited version")
    if metadata.get("name") != name or metadata.get("version") != selector:
        raise ValueError("Registry metadata differs from pinned project declaration")
    dist = metadata["dist"]
    return locked_package(
        name,
        selector,
        npm={
            "packages": {
                "node_modules/" + name: {
                    "version": selector,
                    "resolved": dist["tarball"],
                    "integrity": dist["integrity"],
                }
            }
        },
    )
