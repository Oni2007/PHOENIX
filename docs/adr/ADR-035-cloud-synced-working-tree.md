# ADR-035 — The working tree lives on a cloud-synced drive; immutability is checksum-only there

**Status:** Accepted (with a recorded weakening)
**Date:** 2026-08-19
**Affects:** TDD §10.6, §25, P4

## Context

The repository lives at `C:\Users\Bruker\OneDrive\PHOENIX`, a OneDrive-synced
Windows folder, reached from WSL2 as `/mnt/c/Users/Bruker/OneDrive/PHOENIX`.

TDD §10.6 specifies two independent immutability mechanisms for a sealed run
package: the directory is made **read-only**, and every file is **BLAKE3
checksummed** into `MANIFEST.json`.

Measured on this filesystem:

```text
$ stat -c '%A' results/raw/<run_id>/samples.parquet
-rwxrwxrwx
```

`chmod` is accepted and silently discarded — DrvFs presents every file as 0777.
An append to a sealed `samples.parquet` succeeded. The OS-level half of the
immutability guarantee **does not exist here**.

The checksum half was verified to work: the same tampered file was reported as
`CHECKSUM_MISMATCH`, and all 72 migrated EXP-001 packages verify clean.

## Decision

Accept the working tree on the synced drive, with these consequences recorded
rather than assumed away:

1. **Immutability on this host is enforced by checksum only, not by permissions.**
   `PackageWriter._make_read_only()` is best-effort and already suppresses the
   failure; that suppression is now a documented platform limitation, not an
   oversight.
2. **`verify()` is therefore not optional.** Any analysis consuming a package must
   verify it first. A future measurement host on a POSIX filesystem regains the
   second mechanism for free.
3. **The build tree and virtualenv live outside the synced folder**, at
   `~/.phoenix/{build,venv}`. Linux ELF artefacts have no business being uploaded
   to cloud storage, and a sync client rewriting a build directory mid-compile is a
   real failure mode. `PHOENIX_BUILD_DIR` and the `Makefile` point at them.
4. **The folder is pinned** with `attrib +P` so Files On-Demand cannot turn source
   files into cloud-only placeholders, which would fail a build or an analysis in a
   confusing way.

## Risks accepted

| Risk | Detection | Mitigation |
|---|---|---|
| Sync client rewrites a sealed file | `verify()` reports `CHECKSUM_MISMATCH` | Verify before every analysis |
| Two machines sync conflicting copies of `results/` | OneDrive conflict-copy files appear | `results/` is git-ignored; treat a conflict copy as `TAMPERED` |
| Files On-Demand dehydrates a file | Build or read fails | Folder pinned with `attrib +P` |
| Slow builds reading source over `/mnt/c` | Wall-clock only | Out-of-tree build absorbs most of it |

## Revisit trigger

A controlled measurement host is provisioned (§37 item 12). At that point the
working tree should move to a local POSIX filesystem and both immutability
mechanisms should be restored and re-verified.
