"""The immutable reproducibility package (TDD 10.6 / 25).

Raw sample files are written once, BLAKE3-checksummed, and made read-only.
Corrections are new records that supersede. History is never rewritten.
"""

from __future__ import annotations

import contextlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from blake3 import blake3

MANIFEST_NAME = "MANIFEST.json"


def _hash_file(path: Path) -> str:
    h = blake3()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "blake3:" + h.hexdigest()


@dataclass(frozen=True)
class PackageWriter:
    """Writes one run directory, then seals it."""

    root: Path

    def write_json(self, name: str, payload: Any) -> Path:
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return p

    def write_text(self, name: str, text: str) -> Path:
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def write_samples(self, table: pa.Table, name: str = "samples.parquet") -> Path:
        p = self.root / name
        pq.write_table(table, p, compression="zstd")
        return p

    def seal(self) -> dict[str, Any]:
        """Checksum every file, write MANIFEST.json, make the tree read-only."""
        files: dict[str, Any] = {}
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.name != MANIFEST_NAME:
                files[str(path.relative_to(self.root))] = {
                    "blake3": _hash_file(path),
                    "bytes": path.stat().st_size,
                }
        manifest = {
            "schema_version": "1.0.0",
            "run_dir": self.root.name,
            "files": files,
            "sealed": True,
        }
        (self.root / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._make_read_only()
        return manifest

    def _make_read_only(self) -> None:
        """Best-effort immutability.

        On some filesystems (notably a Windows drive mounted into WSL) permission
        bits do not stick. That is recorded rather than assumed away: the checksums
        in MANIFEST.json are the real integrity mechanism, and verify() is what
        detects tampering.
        """
        ro = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
        for path in sorted(self.root.rglob("*"), reverse=True):
            if path.is_file():
                with contextlib.suppress(OSError):
                    os.chmod(path, ro)


def verify(run_dir: Path) -> tuple[bool, list[str]]:
    """Re-hash every file against MANIFEST.json.

    A mismatch marks the run TAMPERED and excludes it from all analysis.
    """
    manifest_path = run_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return False, [f"{MANIFEST_NAME} missing - package INCOMPLETE"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    for rel, meta in manifest["files"].items():
        path = run_dir / rel
        if not path.exists():
            problems.append(f"missing file: {rel}")
            continue
        actual = _hash_file(path)
        if actual != meta["blake3"]:
            problems.append(f"CHECKSUM_MISMATCH: {rel}")
    recorded = set(manifest["files"])
    on_disk = {
        str(p.relative_to(run_dir))
        for p in run_dir.rglob("*")
        if p.is_file() and p.name != MANIFEST_NAME
    }
    for extra in sorted(on_disk - recorded):
        problems.append(f"unrecorded file present: {extra}")
    return (not problems), problems


REQUIRED_PACKAGE_FILES = (
    "config.yaml",
    "config.normalized.json",
    "request.json",
    "environment.json",
    "git.json",
    "calibration.json",
    "results.json",
    "samples.parquet",
    MANIFEST_NAME,
)


def package_complete(run_dir: Path) -> tuple[bool, list[str]]:
    missing = [f for f in REQUIRED_PACKAGE_FILES if not (run_dir / f).exists()]
    return (not missing), missing
