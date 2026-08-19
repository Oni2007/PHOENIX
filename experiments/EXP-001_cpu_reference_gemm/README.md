# EXP-001 — CPU Reference GEMM

Establishes the ground-truth reference implementation and validates the entire
measurement pipeline on hardware that is always available.

**What this experiment can conclude:** that the pipeline works end-to-end, and how
these specific CPU implementations scale on this specific host.

**What it cannot conclude:** anything about CPU architecture generally, anything
comparative with GPUs, and — on the current development host — anything publishable
at all. See ADR-032.

**Result:** see `results/raw/<run_id>/results.json`. No result is transcribed into
this file; figures and numbers are generated from raw data only.
