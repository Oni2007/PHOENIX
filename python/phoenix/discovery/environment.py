"""Environment discovery and fingerprinting (TDD 25).

Absent facts are recorded as null with a reason. They are never guessed, and a
missing GPU is reported as explicitly absent rather than silently omitted.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from blake3 import blake3


def _run(cmd: list[str]) -> str | None:
    exe = shutil.which(cmd[0])
    if exe is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _first_line(text: str | None) -> str | None:
    """First line of a command's output, or None if the command was absent.

    An absent tool must be recorded as null, never crash the manifest and never be
    guessed at. `"".splitlines()[0]` raises IndexError, which is precisely the
    failure this helper exists to prevent.
    """
    if not text:
        return None
    lines = text.splitlines()
    return lines[0] if lines else None


def _detect_wsl() -> bool:
    release = platform.uname().release.lower()
    return "microsoft" in release or "wsl" in release


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def _accelerators() -> dict[str, Any]:
    """Probe for accelerators. Absence is a recorded fact, not an omission."""
    nvidia = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    return {
        "nvidia_gpu_present": nvidia is not None,
        "nvidia_smi_output": nvidia,
        "cuda_toolkit": _run(["nvcc", "--version"]),
        "rocm_present": shutil.which("rocminfo") is not None,
        "note": (
            "Absence here is a measured fact about this host, recorded so that a "
            "CPU-only result is never mistaken for a GPU-capable environment."
        ),
    }


def collect_environment(build_info: dict[str, Any] | None = None) -> dict[str, Any]:
    uname = platform.uname()
    virtualisation = "wsl2" if _detect_wsl() else "none-detected"
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "collected_at": datetime.now(UTC).isoformat(),
        "host": {
            "os": uname.system,
            "kernel": uname.release,
            "machine": uname.machine,
            "node_hash": blake3(uname.node.encode()).hexdigest()[:16],
            "virtualisation": virtualisation,
        },
        "cpu": {
            "model": _cpu_model(),
            "logical_cores": os.cpu_count(),
            "affinity_cores": len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
        },
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "toolchain": {
            # null means "not on PATH in this process", which is a fact worth
            # recording -- not an error, and not something to guess at.
            "gcc": _first_line(_run(["gcc", "--version"])),
            "cmake": _first_line(_run(["cmake", "--version"])),
        },
        "accelerators": _accelerators(),
        "build_info": build_info,
        "power_telemetry": {
            "available": False,
            "mechanism": None,
            # ADR-024: no energy figure ships from an uncharacterised sensor.
            "note": "NOT YET MEASURED - no characterised power source on this platform",
        },
        "measurement_host_qualified": False,
        "measurement_host_disqualifiers": _disqualifiers(virtualisation),
    }
    return manifest


def _disqualifiers(virtualisation: str) -> list[str]:
    """Why this host cannot produce publishable measurements (ADR-032).

    Recorded per run so a PROVISIONAL result can never be mistaken for a clean one.
    """
    reasons: list[str] = []
    if virtualisation == "wsl2":
        reasons.append("WSL2_VIRTUALISATION_TIMING_UNCHARACTERISED")

    model = (_cpu_model() or "").lower()
    # Intel hybrid P/E-core parts: 12th gen and later Core, and all Core Ultra.
    hybrid_markers = ("core(tm) ultra", "core ultra")
    hybrid_gens = tuple(f"i{tier}-1{gen}" for tier in (3, 5, 7, 9) for gen in (2, 3, 4, 5))
    if any(m in model for m in hybrid_markers) or any(g in model for g in hybrid_gens):
        reasons.append("HYBRID_P_E_CORES_SCHEDULER_ARTEFACT")

    # Mobile/low-power SKUs throttle under sustained load.
    if any(model.rstrip().endswith(sfx) for sfx in ("u", "p", "h", "hx", "hs")):
        reasons.append("MOBILE_THERMAL_ENVELOPE")

    return reasons


def environment_fingerprint(manifest: dict[str, Any]) -> str:
    """BLAKE3 over the parts of the manifest that affect comparability.

    Deliberately excludes timestamps and paths: two runs a day apart on an unchanged
    machine must share a fingerprint, or nothing is ever comparable.
    """
    material = {
        "os": manifest["host"]["os"],
        "kernel": manifest["host"]["kernel"],
        "machine": manifest["host"]["machine"],
        "virtualisation": manifest["host"]["virtualisation"],
        "cpu_model": manifest["cpu"]["model"],
        "logical_cores": manifest["cpu"]["logical_cores"],
        "python": manifest["python"]["version"],
        "toolchain": manifest["toolchain"],
        "build_info": manifest.get("build_info"),
        "nvidia_gpu_present": manifest["accelerators"]["nvidia_gpu_present"],
    }
    blob = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    return "blake3:" + blake3(blob).hexdigest()


def collect_git(repo_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    status = git("status", "--porcelain")
    dirty = bool(status)
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": dirty,
        # Running dirty is permitted - research requires it - but the diff is
        # embedded and the run is marked provisional. A provisional run may never
        # be the source of a value in a published figure.
        "diff": git("diff", "HEAD") if dirty else None,
        "provisional": dirty,
    }
