# PHOENIX

**Photonic-Heterogeneous Optimized Engine for Neural Intelligence eXecution**

A research platform for measuring, benchmarking, and modelling computation across
heterogeneous AI accelerators — with the long-term goal of determining, from evidence
rather than assumption, which computational substrate suits which workload.

> **Status: Weeks 1–4 of the v0.1 plan are complete.** A working measurement pipeline
> runs end to end on CPU. Weeks 5–12 (CUDA onward) await hardware. See
> [Current state](#current-state) below.

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

1. A vendor-neutral hardware database with per-field provenance.
2. Environment discovery capturing the full software/hardware manifest of any run.
3. A C++ benchmark runtime with a backend abstraction (CPU, CUDA implemented;
   HIP and a TPU adapter as interfaces only).
4. A GEMM implementation ladder from CPU reference through Tensor Core and cuBLASLt,
   each stage validated against a trusted reference.
5. An immutable measurement store separating raw samples from aggregates from derived
   metrics from reported values.
6. A profiling subsystem in which profiled timings are structurally prevented from
   contaminating benchmark timings.
7. A per-run reproducibility package sufficient to reconstruct execution conditions.
8. A structured research-literature database recording claims, evidence, and evidence
   *class* — not URLs.
9. Roofline analysis distinguishing theoretical roofs from empirically-attained ceilings.

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

---

## Current state

| | |
|---|---|
| Phase | Weeks 1–4 complete; weeks 5–12 not started |
| Compute plane | C++20, CPU + synthetic backends, GEMM ladder (naive / blocked / OpenMP) |
| Control plane | Python 3.12 — config, discovery, orchestration, store, analysis, CLI |
| Boundary | nanobind, one JSON crossing per run |
| Tests | 80 C++ (GoogleTest) + 32 Python (pytest/hypothesis), all passing |
| Static analysis | `ruff` clean, `mypy --strict` clean |
| Experiments executed | EXP-001 — 72 runs, 2 520 raw samples, all correctness passing |

**What is not built:** every CUDA backend (weeks 5–9), HIP and TPU adapters
(weeks 10–11), profiling infrastructure (week 8), the literature database
(week 11), roofline analysis, and power/energy — which stays `NOT YET MEASURED`
because no characterised power source exists on this host (ADR-024).

### Try it

From inside WSL2:

```bash
source .venv/bin/activate
export PYTHONPATH=$PWD/python:$PWD/build
python -m phoenix.cli.main discover
python -m phoenix.cli.main run experiments/EXP-001_cpu_reference_gemm/config.yaml
python -m phoenix.cli.main report EXP-001
```

## Repository layout

```text
PHOENIX/
├── CLAUDE.md                 Working rules for AI-assisted development
├── LICENSE                   Apache-2.0
├── README.md                 This file
└── docs/
    ├── design/
    │   ├── PHOENIX_v0.1_TDD.md      Authoritative v0.1 specification
    │   └── PHOENIX_MASTER_PROMPT.md The brief the TDD answers
    └── research/
        └── PHOENIX_MASTER_RESEARCH_CONTEXT.md   Long-horizon research context
```

The full target tree — `cpp/`, `python/`, `schemas/`, `hardware/`, `experiments/`,
`results/`, `research/`, `containers/` — is specified in TDD §8 and will be created when
implementation begins.

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

## Open questions

Items that genuinely cannot be settled without hardware or credentials are listed in
TDD §37. The ones on the critical path:

- **Where will publishable measurements be taken?** The development host — a hybrid
  P/E-core laptop under WSL2 — cannot produce them (ADR-032). Development is unaffected;
  publication is blocked until a controlled measurement host exists.
- Which specific NVIDIA GPU(s) PHOENIX will run on — needed by Week 4.
- Whether a self-hosted GPU CI runner will be available.
- Whether an external power meter is available. If not, energy stays `NOT YET MEASURED`
  in v0.1 — an acceptable outcome that must not be worked around with estimates.

## Development platform

All development happens inside **WSL2 / Ubuntu 26.04** (ADR-031). Windows-native builds
are not supported. `cmake` and `ninja` are not yet installed and are a Week 1 task.

## Licence

Apache-2.0. See [LICENSE](LICENSE).

## A finding from the first run

The first execution of EXP-001 failed 36 of 72 runs on correctness — every fp32 run,
while every fp64 run passed bit-exactly. The cause was not a kernel: it was the
comparator, which scaled error by `|C_ij|` and so measured catastrophic cancellation
in the input data rather than implementation error.

The tolerance constant was **not** raised. The error model was corrected to scale by
`(|A|·|B|)_ij`, after which fp32 error lands at ≈2·eps(fp32), flat in K. See
[ADR-034](docs/adr/ADR-034-gemm-error-scale.md) and
[FIND-001](research/notes/FIND-001-fp32-cancellation.md).

This is the kind of thing the v0.1 ordering exists to catch: it was found on a CPU
reference kernel, at the cheapest possible point, rather than on a GPU months later.
