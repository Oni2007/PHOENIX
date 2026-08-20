# PHOENIX

**Photonic-Heterogeneous Optimized Engine for Neural Intelligence eXecution**

A research platform for measuring, benchmarking, and modelling computation across
heterogeneous AI accelerators — with the long-term goal of determining, from evidence
rather than assumption, which computational substrate suits which workload.

> **Status: Weeks 1–4 of the v0.1 plan are complete**, plus a real BLAS correctness
> cross-check, a hardened hardware schema, an analytical (non-hardware) GPU roofline
> model, and the start of the literature database. Weeks 5–12 (CUDA execution onward)
> await physical GPU access. See [Current state](#current-state) below.

---

## What PHOENIX v0.1 is

The **measurement, benchmarking, and provenance infrastructure** that every later
PHOENIX claim will rest on.

The design's central premise: the hard part of comparing a GPU against a photonic
accelerator is not simulating the photonics. It is producing an electronic baseline a
skeptical reviewer cannot dismantle. A photonic energy advantage claimed against a
badly-measured GPU baseline is worth nothing. So v0.1 builds the baseline first, and
builds it to a standard where the measurement methodology is itself the research asset.

v0.1 delivers:

1. A vendor-neutral hardware database with per-field provenance, enforced by type —
   a peak-compute figure literally cannot validate without stating its conditions.
2. Environment discovery capturing the full software/hardware manifest of any run.
3. A C++ benchmark runtime with a backend abstraction (CPU and a synthetic
   known-duration backend implemented; CUDA, HIP, and a TPU adapter await hardware).
4. A GEMM implementation ladder — reference, naive, blocked, OpenMP — each stage
   validated against both PHOENIX's own FP64 reference and an independent vendor BLAS.
5. An immutable measurement store separating raw samples from aggregates from derived
   metrics from reported values, sealed and checksummed per run.
6. A structured research-literature database recording claims, evidence, and evidence
   *class* — not URLs — with 54 real papers seeded so far.
7. Roofline analysis distinguishing theoretical roofs (`VENDOR_REPORTED`) from
   empirically-attained ceilings (`MEASURED`), demonstrated against a real GPU's
   published specifications without ever claiming to have run on it.

## What PHOENIX v0.1 is **not**

Stated as prominently as the above, because scope creep is the main risk to the project:

- **Not an accelerator, a compiler, or a photonic simulator.** Photonic simulation is
  v0.6. Neuromorphic is v0.8. The compiler is v0.5. The scheduler is v0.3–v0.5.
- **Not an attempt to beat cuBLAS.** The kernel ladder exists to *understand* the
  performance gap, not close it. A kernel that reaches a documented fraction of cuBLAS
  with a full bottleneck analysis is a success; a faster kernel with no explanation is
  a failure.
- **Not a source of cross-vendor performance claims.** Not NVIDIA vs AMD, not GPU vs
  TPU. The backend interface exists to make such comparisons possible *later, under
  controlled conditions*.
- **Not a general tensor library.** No user-facing tensor API, no autograd.
- **Not a leaderboard.**

PHOENIX asserts nothing about photonics — neither that it will win nor that it will lose.
Finding the conditions under which each architecture is advantageous is the entire point.

---

## Non-negotiable rules

These are enforced by schema and CI, not by convention. Violating one invalidates
research output. Full text in [CLAUDE.md](CLAUDE.md); rationale in the TDD.

1. **Never fabricate a number** — including in documentation, examples, and test
   fixtures. Unknown values are `UNKNOWN` or `NOT YET MEASURED`.
2. **Every numeric value carries a machine-readable provenance class** — `MEASURED`,
   `VENDOR_REPORTED`, `SIMULATED`, `ANALYTICAL`, `HYPOTHETICAL`, `DERIVED`. Never mixed
   within a field or a plot series.
3. **Raw data is immutable.** Written once, checksummed, never mutated. Invalid runs are
   flagged and retained, never deleted.
4. **Derived values are never `MEASURED`.**
5. **No optimisation without profiling first.**
6. **Every kernel stage is retained forever** — earlier stages are the baselines that
   make later claims meaningful.
7. **Tolerances are never loosened to make a failing kernel pass.**
8. **Profiled runs are not performance measurements.**
9. **No cross-vendor claim without an explicit comparability declaration.**
10. **Figures are generated from raw data only.**
11. **A device this project has never run on gets `VENDOR_REPORTED`/`ANALYTICAL` data
    only, marked *purely theoretical* — never presented as if it were measured.**

---

## Current state

| | |
|---|---|
| Phase | Weeks 1–4 complete; weeks 5–12 blocked on GPU access |
| Compute plane | C++20, CPU + synthetic backends, GEMM ladder (naive / blocked / OpenMP) |
| Control plane | Python 3.12 — config, discovery, orchestration, store, analysis, literature, CLI |
| Boundary | nanobind, one JSON crossing per run |
| Tests | 94 C++ (GoogleTest) + 68 Python (pytest/hypothesis), all passing |
| Correctness cross-check | Independent vendor BLAS (CBLAS), gated on availability (ADR-036) — CI-verified against real `libopenblas-dev` |
| Hardware schema | Typed structures (not loose dicts); peak-compute fields cannot validate without stated conditions (ADR-037) |
| Static analysis | `ruff` clean, `mypy --strict` clean |
| Experiments executed | EXP-001 — 72 runs, 2 520 raw samples, all correctness passing, marked `PROVISIONAL` (ADR-032) |
| Literature database | **54 papers** — 2 `SKIMMED` with extracted claims, 52 `ABSTRACT_ONLY` (metadata only, no claims — enforced by schema) |
| Theoretical hardware records | NVIDIA H100 SXM5 80GB — `VENDOR_REPORTED` specs + `ANALYTICAL` roofline model, explicitly never run on (ADR-037) |

**What is not built:** every CUDA backend (weeks 5–9), HIP and TPU adapters
(weeks 10–11), profiling infrastructure (week 8), roofline against *measured* GPU data
(needs hardware), and power/energy — which stays `NOT YET MEASURED` because no
characterised power source exists on this host (ADR-024).

### Try it

From inside WSL2, at `/mnt/c/Users/Bruker/OneDrive/PHOENIX`:

```bash
source ~/.phoenix/venv/bin/activate
export PHOENIX_BUILD_DIR=~/.phoenix/build PYTHONPATH=$PWD/python:~/.phoenix/build
make build && make test

python -m phoenix.cli.main discover
python -m phoenix.cli.main run experiments/EXP-001_cpu_reference_gemm/config.yaml
python -m phoenix.cli.main report EXP-001

# literature corpus: validate + regenerate research/papers.csv
python -m phoenix.cli.main papers

# analytical (non-hardware) roofline against a real GPU's published specs
python -m phoenix.cli.main roofline hardware/gpu/nvidia-h100-sxm5-80gb.yaml FP16_TENSOR_CORE_DENSE
```

The build tree and virtualenv live at `~/.phoenix/`, outside the synced folder, on
purpose — see [ADR-035](docs/adr/ADR-035-cloud-synced-working-tree.md).

## Repository layout

```text
PHOENIX/
├── CLAUDE.md                 Working rules for AI-assisted development
├── LICENSE                   Apache-2.0
├── README.md                 This file
├── CMakeLists.txt, Makefile  Build (cmake + ninja; make wraps both planes)
├── bindings/                 nanobind module (phoenix_core)
├── cpp/
│   ├── include/, src/        core, timing, bench, correctness, schema
│   ├── backends/cpu/         GEMM ladder + BLAS cross-check + synthetic backend
│   └── tests/                GoogleTest suite (94 tests)
├── python/phoenix/
│   ├── config/, discovery/   Pydantic config + environment/hardware schema
│   ├── runner/, store/       orchestration + immutable sealed packages
│   ├── analysis/             statistics, figures, roofline
│   ├── literature/           paper corpus schema + loader + CSV export
│   └── cli/                  the `phoenix` command
├── schemas/                  JSON Schema exported from every Pydantic model
├── hardware/
│   ├── cpu/                  the dev host's own record (MEASURED where possible)
│   └── gpu/                  NVIDIA H100 — VENDOR_REPORTED, purely theoretical
├── experiments/EXP-001_.../  version-controlled experiment definitions
├── research/
│   ├── papers/                54 paper records (PAP-0001..PAP-0054)
│   ├── papers.csv              generated export — do not edit directly
│   └── notes/                  findings (e.g. FIND-001)
├── docs/
│   ├── design/                the TDD and the master prompt it answers
│   ├── research/               long-horizon context, superseded by the TDD
│   ├── adr/                    architecture decision records (ADR-034..037)
│   └── analysis/                generated reports (e.g. the H100 roofline)
├── tests/                    pytest suite (68 tests)
└── .github/workflows/        CI (installs real libopenblas-dev, runs both suites)
```

`results/` (immutable run packages) is git-ignored by design — benchmark data does
not belong in git history (TDD §8).

## Documents

- **[docs/design/PHOENIX_v0.1_TDD.md](docs/design/PHOENIX_v0.1_TDD.md)** — the authoritative
  specification. Fixes the technology stack (§5), the Python/C++ boundary (§7), the
  repository tree (§8), all schemas (§9–§11), the CUDA GEMM roadmap (§13), the measurement
  and statistical methodology (§15–§16), the first ten experiments (§27), the risk register
  (§28), the Definition of Done (§30), the 12-week plan (§31), and the ADR register (§34).
  Change a decision by updating its ADR, not by diverging in code.
- **[docs/design/PHOENIX_MASTER_PROMPT.md](docs/design/PHOENIX_MASTER_PROMPT.md)** — the
  engineering brief the TDD was written against. Retained so the TDD can be audited
  against its requirements.
- **[docs/research/PHOENIX_MASTER_RESEARCH_CONTEXT.md](docs/research/PHOENIX_MASTER_RESEARCH_CONTEXT.md)**
  — long-horizon research scope covering photonics, neuromorphic computing, and the
  heterogeneous runtime. **Superseded by the TDD for all v0.1 decisions**; retained for
  research direction beyond v0.1.
- **[docs/adr/](docs/adr/)** — decision records for everything that deviated from or
  extended the TDD during implementation (ADR-034 through ADR-037 so far).

## Open questions

Items that genuinely cannot be settled without hardware or credentials are listed in
TDD §37. The ones on the critical path:

- **Where will publishable measurements be taken?** The development host — a hybrid
  P/E-core laptop under WSL2 — cannot produce them (ADR-032). Development is unaffected;
  publication is blocked until a controlled measurement host exists.
- Which specific NVIDIA GPU(s) PHOENIX will run on — needed before Weeks 5–9 can start.
  The H100 record under `hardware/gpu/` is theoretical scaffolding, not a decision.
- Whether a self-hosted GPU CI runner will be available.
- Whether an external power meter is available. If not, energy stays `NOT YET MEASURED`
  in v0.1 — an acceptable outcome that must not be worked around with estimates.

## Development platform

All development happens inside **WSL2 / Ubuntu 26.04** (ADR-031). Windows-native builds
are not supported.

## Licence

Apache-2.0. See [LICENSE](LICENSE).

## Findings along the way

Two things were caught and fixed rather than smoothed over, each with its own ADR:

- **The first EXP-001 execution failed 36 of 72 runs on correctness** — every fp32 run,
  while every fp64 run passed bit-exactly. The cause was not a kernel: the comparator
  scaled error by `|C_ij|`, measuring catastrophic cancellation in the input data rather
  than implementation error. The tolerance constant was **not** raised; the error model
  was corrected to scale by `(|A|·|B|)_ij`, after which fp32 error lands at ≈2·eps(fp32),
  flat in K. See [ADR-034](docs/adr/ADR-034-gemm-error-scale.md) and
  [FIND-001](research/notes/FIND-001-fp32-cancellation.md).
- **A pydantic union-type bug silently coerced booleans into floats.** Building the H100
  hardware record required hardening the schema; validating an existing record against
  the tightened schema then surfaced a real bug — `False` was becoming `0.0` before any
  validator ran, because Python's `bool` is a subclass of `int`. Fixed by giving `bool`
  its own branch in the type union. See
  [ADR-037](docs/adr/ADR-037-analytical-vendor-hardware-records.md).

This is the kind of thing the v0.1 ordering exists to catch: found on a CPU reference
kernel and a schema validator, at the cheapest possible point, rather than on a GPU
months later.
