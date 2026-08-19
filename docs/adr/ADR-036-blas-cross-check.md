# ADR-036 — Independent vendor-BLAS cross-check, gated on runtime availability

**Status:** Accepted
**Date:** 2026-08-19
**Closes:** the gap recorded in an earlier README/CLAUDE.md note ("the TDD calls
for a BLAS-backed CPU GEMM... `libopenblas-dev` could not be installed")

## Context

TDD §14.1 wants the correctness hierarchy to include an externally-maintained
vendor BLAS, not just PHOENIX's own FP64 reference: a shared misunderstanding of
the math in GEMM-0 and in naive/blocked/omp would pass every test that only
checks PHOENIX against PHOENIX.

The development host has no root access (`sudo` requires an interactive password
this session cannot supply), so `apt-get install libopenblas-dev` is not possible
here — confirmed absent from the ld cache. The GitHub Actions runner **does**
have passwordless root.

## Decision

1. `CMakeLists.txt` calls `find_package(BLAS)` (cmake's stock module) and,
   if found, links it and defines `PHOENIX_HAVE_BLAS`.
2. `cpp/backends/cpu/gemm_blas.{hpp,cpp}` wraps `cblas_dgemm`/`cblas_sgemm`,
   declared locally against the standard, stable CBLAS ABI rather than depending
   on a system `cblas.h` — the integer constants (101/102/111/112) are fixed by
   the reference CBLAS specification, not a PHOENIX choice.
3. `blas_available()` is a single runtime source of truth, used by both the
   GEMM correctness tests and `build_info()` (so `phoenix discover` reports it
   honestly per run) — never two independently-maintained flags that could drift.
4. Correctness tests call `blas_available()` first and `GTEST_SKIP()` with a
   named reason when false, rather than being `#ifdef`'d out (invisible) or
   asserting a hard failure (wrong on a host that legitimately has no BLAS).
5. `.github/workflows/pr-checks.yml` installs `libopenblas-dev` via `apt-get`
   before the build, then runs a dedicated step that fails the build if the
   BLAS tests skipped anyway — catching a broken `find_package(BLAS)` wiring,
   not just a broken kernel. That step deliberately excludes
   `UnavailableBlasThrowsRatherThanSilentlyReturningGarbage`, which is *supposed*
   to skip itself whenever BLAS is present.

## Consequence

The vendor-BLAS cross-check is real and CI-verified, not aspirational. It is
provably inert (all tests skip, none silently pass) on this development host,
and provably exercised on the CI runner where root exists. Locally: 13 tests
skip with a message naming this ADR. In CI: the same 13 must produce a real
comparison or the build fails.

## What this does NOT do

Does not make BLAS a benchmarked implementation in `EXP-001`'s config — that
would make `phoenix run` fail on hosts without BLAS via `ConfigurationError`, an
error-handling change out of scope here. The cross-check is a correctness gate,
not a measured backend implementation. If BLAS throughput becomes an interesting
research question later, that is a separate, deliberate decision (candidate for
the GEMM-6 vendor-baseline slot the CUDA ladder already reserves).

## Revisit trigger

A controlled measurement host is provisioned (§37 item 12) with root access —
at that point the local dev/test split described here becomes unnecessary and
BLAS can simply be required.
