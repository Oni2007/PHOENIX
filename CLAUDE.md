# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current repository state

**Weeks 1–4 of the implementation plan are complete.** The repository contains a
working C++ compute plane, Python control plane, nanobind boundary, immutable store,
analysis pipeline, and CLI. Weeks 5–12 (CUDA onward) are not started.

### Commands that actually work

Run everything from **inside WSL2**, from the repository root, with the venv active:

```bash
source .venv/bin/activate
make build          # cmake + ninja
make test           # 80 C++ tests + 32 Python tests
make lint           # ruff check + format --check
make typecheck      # mypy --strict
```

CLI: `phoenix discover | validate <cfg> | run <cfg> | analyze EXP-001 | catalogue |
report EXP-001 | verify-package <run_dir> | run-info <run_dir>`.

Invoke as `python -m phoenix.cli.main <cmd>` with
`PYTHONPATH=$PWD/python:$PWD/build` until the package is pip-installed.

Do not invent commands beyond these. Weeks 5+ toolchain (CUDA, Nsight, HIP) does not
exist yet — verify a tool is present before running it.

## The design document is the specification

`docs/design/PHOENIX_v0.1_TDD.md` is the authoritative source for every architectural decision. Read it before writing code. It fixes the technology stack (§5), the Python/C++ boundary (§7), the repository tree (§8), all schemas (§9–§11), the CUDA GEMM roadmap (§13), the measurement methodology (§15–§16), and the 12-week plan (§31). Decisions are recorded with rationale in the ADR register (§34) — change one only by updating the ADR and stating the revisit trigger, not by silently diverging in code.

`docs/design/PHOENIX_MASTER_PROMPT.md` is the brief the TDD answers, retained so the TDD can be audited against its requirements. It is not itself a specification.

`docs/research/PHOENIX_MASTER_RESEARCH_CONTEXT.md` is the long-horizon research context. **It is superseded by the TDD wherever the two disagree** — including its provenance enum, its flatter top-level `cuda/` / `photonic/` directory sketch, and its 30/60/90-day roadmap. Its own header records the divergences. Use it for research direction beyond v0.1, never as a v0.1 specification. CUDA belongs under `cpp/backends/cuda/` because it is one backend among several, not a peer of the whole system.

## Implementation gate

The gate at TDD §38 was opened for **Weeks 1–4 only**, and those are done.

Weeks 5–9 require a physical NVIDIA GPU that does not exist yet, and week 10+ requires
AMD/TPU access. Do not begin them on this host. The next authorised slice needs its own
instruction naming the weeks.

## What PHOENIX v0.1 is

Measurement, benchmarking, and provenance infrastructure for comparing AI accelerators — **not** an accelerator, compiler, or photonic simulator. v0.1 covers GEMM only, on CPU and CUDA, single device. Photonics is v0.6; neuromorphic is v0.8; the compiler is v0.5. See TDD §2–§3 for scope and the explicit non-goals.

The reason the ordering matters: a photonic energy advantage claimed against a badly-measured GPU baseline is worth nothing. The baseline is the point of v0.1.

## Non-negotiable rules for any code written here

These govern every contribution and are enforced by schema and CI, not by convention. Violating one invalidates research output, so they are not stylistic preferences.

1. **Never fabricate a number.** No benchmark results, FLOPS, latency, bandwidth, power, or hardware specs — including in documentation, examples, and test fixtures. Unknown values are `UNKNOWN` or `NOT YET MEASURED`, and `null` is a legal schema value precisely so nobody is pressured into inventing one.
2. **Every numeric value carries a machine-readable provenance class**: `MEASURED`, `VENDOR_REPORTED`, `SIMULATED`, `ANALYTICAL`, `HYPOTHETICAL`, `DERIVED`. Never mix classes in one field or one plot series. This is the single load-bearing decision in the design (ADR-004) — it is what makes a later `SIMULATED` photonic vs `MEASURED` GPU comparison defensible.
3. **Raw data is immutable.** Samples are written once, BLAKE3-checksummed, never mutated. Corrections are new records with a `supersedes` pointer. Invalid runs are flagged and retained, never deleted — a pattern of thermal throttling is a finding about the machine.
4. **Derived values are never `MEASURED`.** Store the formula, its inputs, and its assumptions; recompute rather than cache authoritatively.
5. **No optimization without profiling first.** The loop is measure → profile → hypothesize → modify → measure → validate → document. A kernel change without a linked profiling artifact is rejected in review.
6. **Every kernel stage is retained forever.** GEMM-1 is not deleted when GEMM-7 exists; earlier stages are the baselines that make later claims meaningful.
7. **Tolerances are never loosened to make a failing kernel pass.** Tolerance derives from `C · eps(dtype) · sqrt(K)`; raising `C` requires written numerical justification.
8. **Profiled runs are not performance measurements.** Nsight Compute serializes and replays kernels; profiled timings carry `timing_valid_for_reporting = false` and are structurally barred from the performance dataset.
9. **No cross-vendor or cross-environment performance claim** without an explicit comparability declaration covering tuning effort, precision semantics, and power-measurement semantics.
10. **Figures are generated from raw data only.** Never hand-edit a data file or type a number into a plot.

## Architecture in brief

Two planes with a deliberate split (TDD §7):

- **Python control plane** (`python/phoenix/`) — config validation, sweep expansion, environment discovery, orchestration, storage, statistics, plotting, reporting.
- **C++ compute plane** (`cpp/`) — timing, kernel launch, memory, correctness checking, sample capture.

**Python is never inside a measured region.** Interpreter jitter is comparable to the quantities being measured at small matrix sizes. One JSON crossing per run: Python hands `execute()` a fully-specified `RunRequest` and receives a `RunResult`, both validated against versioned schemas on each side. The boundary is JSON strings rather than rich bound types so every crossing is auditable and byte-reproducible into the run's package.

Backends satisfy a C++20 `Backend` concept. **It is designed against the TPU execution model, not CUDA** (ADR-025): `prepare()` is split from `execute_once()` because XLA compile latency is one of the most important things to report about a TPU and a CUDA-shaped interface would hide it. CUDA-shaped abstractions break on other substrates; the reverse holds.

Data flows through four separated tiers (TDD §10) — raw samples → aggregates → derived metrics → reported values — so any published figure traces back to individual iteration timings. Tiers 2–4 are always recomputable from Tier 1.

Storage is local-first: immutable Parquet under `results/raw/<run_id>/` is the system of record; the DuckDB catalogue is a rebuildable query layer over it, never a copy.

## Planned toolchain (does not exist yet)

C++20 / Python 3.11 / CUDA 12.x / CMake ≥ 3.24 / nanobind / GoogleTest / pytest + hypothesis / Pydantic v2 / Parquet + DuckDB / Polars / Matplotlib / ruff + mypy --strict / clang-format + clang-tidy. Full rationale and alternatives-considered in TDD §5.

## Environment notes

- **CUDA work requires a physical NVIDIA GPU.** None is present. Weeks 1–4 are deliberately GPU-free so development is not blocked by hardware availability.
- **Which GPU PHOENIX will run on is still undecided** (TDD §37 item 1). It determines compute capability, available precisions, `CMAKE_CUDA_ARCHITECTURES`, and EXP-006's dtype coverage. Needed by Week 4.
- **Do not put the working tree in a cloud-synced folder.** TDD §10 requires read-only checksummed directories under `results/`; sync clients fight that, and build trees thrash them.
- **The tree currently lives on the Windows filesystem** (`C:\Users\Bruker\PHOENIX`, i.e. `/mnt/c/Users/Bruker/PHOENIX` from WSL2). That is fine while the repository is documents only. **Before Week 1's first build, move it onto the WSL2 filesystem** (e.g. `~/PHOENIX`): building across `/mnt/c` is slow, and its file-metadata semantics are not the ones §10's read-only immutability enforcement assumes. Clone or `git mv` — do not run a build in place.
- Verify the CUDA toolkit and driver versions against NVIDIA's current release rather than trusting the TDD's `⚠`-marked pins.
- AMD/ROCm and TPU are interface-only in v0.1. Do not claim support for hardware that has never executed a run.

## Host platform — resolved

**All development happens inside WSL2 / Ubuntu 26.04** (ADR-031). Windows-native MSVC is not supported: the entire §5 stack, §22 containers, and §21 CI are Linux-native, and the Windows side has no C++ toolchain at all. Run builds, tests, and `phoenix` from the WSL2 shell, not from PowerShell or Git Bash.

Verified toolchain in WSL2: gcc 15.2.0, g++, make, git, python3 3.14.4. **cmake and ninja are absent and must be installed in Week 1.** The project interpreter is pinned by `uv` to 3.11/3.12 independently of the system Python 3.14 (ADR-033).

## This machine cannot produce publishable measurements

**ADR-032, and it is blocking.** The dev host is an i7-1355U laptop with hybrid P/E cores, a 15 W thermal envelope, and a WSL2 hypervisor between the timers and the hardware. Each of those independently violates §32's validity criteria:

- Hybrid cores produce bimodal timing distributions that look like a code property but are a scheduler artefact (R-19).
- Sustained sweeps will thermally throttle, which §32 correctly marks INVALID (R-21).
- WSL2's timing distortion is `UNKNOWN` in magnitude and uncharacterised (R-20, §37 item 11).

So: running EXP-001 here **validates the pipeline, not the CPU**. Its output is `PROVISIONAL` and must never be published or used as a baseline. Do not "fix" this by loosening a validity gate — the gates are working correctly. A publishable CPU baseline needs a controlled measurement host (§37 item 12).

There is also **no NVIDIA GPU and no discrete GPU of any kind** here, so Weeks 5–9 need a separate host that does not yet exist. Weeks 1–4 are GPU-free and unblocked today.

## Known issues to resolve at implementation time

- `.gitignore` deliberately does **not** ignore `Makefile` — TDD §8 specifies a committed top-level `Makefile` as a convenience wrapper over cmake/uv. Leave it that way.
- `results/` is git-ignored with a `!results/**/MANIFEST.json` exception. Confirm at Week 4, when the store is first written, that this is the intended manifest policy.
- The repository has one commit and no remote. Add the GitHub remote before Week 1 — TDD §21 and ADR-021 assume GitHub Actions.
