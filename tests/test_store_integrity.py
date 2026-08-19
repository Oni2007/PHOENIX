from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa

from phoenix.store.package import PackageWriter, package_complete, verify


def _make_package(tmp_path: Path) -> Path:
    run = tmp_path / "run-1"
    run.mkdir()
    w = PackageWriter(run)
    w.write_json("results.json", {"run_id": "run-1", "status": "COMPLETED"})
    w.write_samples(pa.table({"host_wall_ns": pa.array([1, 2, 3], type=pa.int64())}))
    w.seal()
    return run


def test_sealed_package_verifies(tmp_path: Path) -> None:
    run = _make_package(tmp_path)
    ok, problems = verify(run)
    assert ok, problems


def test_mutated_file_is_detected_as_tampered(tmp_path: Path) -> None:
    run = _make_package(tmp_path)
    target = run / "results.json"
    target.chmod(0o644)
    target.write_text(json.dumps({"run_id": "run-1", "status": "TOTALLY_FINE"}))
    ok, problems = verify(run)
    assert not ok
    assert any("CHECKSUM_MISMATCH" in p for p in problems)


def test_added_file_is_detected(tmp_path: Path) -> None:
    run = _make_package(tmp_path)
    (run / "sneaky.json").write_text("{}")
    ok, problems = verify(run)
    assert not ok
    assert any("unrecorded file" in p for p in problems)


def test_removed_file_is_detected(tmp_path: Path) -> None:
    run = _make_package(tmp_path)
    target = run / "samples.parquet"
    target.chmod(0o644)
    target.unlink()
    ok, problems = verify(run)
    assert not ok
    assert any("missing file" in p for p in problems)


def test_incomplete_package_is_reported(tmp_path: Path) -> None:
    run = _make_package(tmp_path)
    complete, missing = package_complete(run)
    assert not complete
    assert "environment.json" in missing


def test_manifest_records_a_checksum_for_every_file(tmp_path: Path) -> None:
    run = _make_package(tmp_path)
    manifest = json.loads((run / "MANIFEST.json").read_text())
    assert manifest["files"]
    for meta in manifest["files"].values():
        assert meta["blake3"].startswith("blake3:")
        assert meta["bytes"] > 0
