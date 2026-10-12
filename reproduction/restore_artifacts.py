#!/usr/bin/env python3
"""Restore byte-identical scientific artifacts from anonymous-host-compatible parts.

Uses the Python standard library only. No network access, pickle loading,
model execution, or replacement of existing files is performed.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "reproduction/artifact_parts.json"


def checked_path(root: Path, relative: str) -> Path:
    name = PurePosixPath(relative)
    if not relative or name.is_absolute() or ".." in name.parts or "\\" in relative:
        raise ValueError(f"Invalid artifact path: {relative}")
    root = root.resolve()
    path = root.joinpath(*name.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Artifact path leaves the repository: {relative}")
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError(f"Symlinks are not supported for artifacts: {relative}")
    return path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def matches(path: Path, expected: dict) -> bool:
    return (path.is_file() and path.stat().st_size == expected["bytes"]
            and digest(path) == expected["sha256"])


def restore(root: Path = ROOT) -> dict:
    manifest = json.loads(checked_path(root, MANIFEST).read_text())
    if manifest.get("version") != 1:
        raise ValueError("Unsupported artifact manifest version")
    restored = verified = 0
    for entry in manifest["artifacts"]:
        target = checked_path(root, entry["path"])
        if entry["encoding"] != "gzip":
            raise ValueError(f"Unsupported encoding for {entry['path']}")
        chunks = []
        for part in entry["parts"]:
            path = checked_path(root, part["path"])
            if not matches(path, part):
                raise ValueError(f"Missing or damaged artifact part: {part['path']}")
            chunks.append(path.read_bytes())
        archive = b"".join(chunks)
        if (len(archive) != entry["archive_bytes"] or
                hashlib.sha256(archive).hexdigest() != entry["archive_sha256"]):
            raise ValueError(f"Archive checksum mismatch: {entry['path']}")
        if target.exists():
            if not matches(target, entry):
                raise ValueError(f"Refusing to overwrite a different file: {entry['path']}")
            verified += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # Publish only after validation. Hard-link creation refuses to replace
        # a file created by another process while restoration was in progress.
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".prism-restore-") as output:
            total, checksum = 0, hashlib.sha256()
            with gzip.GzipFile(fileobj=io.BytesIO(archive), mode="rb") as stream:
                while block := stream.read(1024 * 1024):
                    total += len(block)
                    if total > entry["bytes"]:
                        raise ValueError(f"Restored artifact exceeds declared size: {entry['path']}")
                    checksum.update(block)
                    output.write(block)
            if total != entry["bytes"] or checksum.hexdigest() != entry["sha256"]:
                raise ValueError(f"Restored artifact checksum mismatch: {entry['path']}")
            output.flush()
            os.fsync(output.fileno())
            os.link(output.name, target)
        restored += 1
    return {"restored": restored, "already_verified": verified,
            "scientific_bytes_preserved": True}


if __name__ == "__main__":
    try:
        print(json.dumps(restore()))
    except (OSError, ValueError, KeyError, EOFError) as error:
        raise SystemExit(f"Artifact restoration failed: {error}") from error
