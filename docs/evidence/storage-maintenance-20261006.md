# Authorized storage maintenance and queue observation

The user approved the reviewed cleanup on 2026-10-06. At 2026-10-06T12:44:11.890261+08:00,
all three maintenance operations and their postchecks passed. The unattended
measurement manager and its current child remain active in background.slice.
[Reviewed record](storage-maintenance-20261006.json).

## Storage result

| Operation | Allocated space |
| --- | ---: |
| Remove 6,416 selected old development-profile, profiler-SQLite and CMake-object files | 26.925694 GiB freed |
| Deduplicate 5,774 hash-checked raw-file pairs, retaining both original paths | 5.412189 GiB freed |
| Total deletion plus deduplication | 32.337883 GiB freed |
| Move 40 inactive root-filesystem trees to the data filesystem | 2.714954 GiB transferred; not additional total space freed |

The data filesystem now has 166.250 GiB free and root has
12.566 GiB free. Other workloads can change these observations.
Root destinations are under `TASK/retired-root-artifacts-20261006`; each old path
remains usable through a symlink. Contents were hashed before copying, after
copying and after an atomic directory/symlink exchange, before removing the old
copy. Relative symlink destinations were checked to remain within each moved tree.

All 644 frozen queue input hashes and 109 retained record hashes still match.
All 6,416 selected files are absent; all 5,774 raw pairs share the intended inode;
all 40 relocated paths resolve correctly. The protected historical worker 2686919
remains stopped (TNsl), with its service, source, builds and artifacts untouched.
Current builds, formal results, failed experiment records, SDKs, environments and
reference repositories were not cleaned.

The first deduplication precheck exited 1 before mutation because 32 zero-byte
completion markers had different collection/export mtimes. Its failed service
and log remain. The v2 plan preserved these markers byte-for-byte and with their
metadata, and deduplicated only matching hashes, modes and mtimes. Original and
recovery raw paths are now hardlinks: keep them immutable and copy into a new
working directory before re-export operations that write raw inputs. The removed
SQLite files are derived databases; original raw data, report CSVs and analyses
remain. Old development builds retain binaries, libraries, archives and source;
rebuilding them requires CMake regeneration and recompilation.

Full reviewed inventories, helper hashes, service logs, receipts and postchecks:
`TASK/plans/storage-cleanup-20261006-01/`. Maintenance ran with Nice 19, idle IO,
one CPU 318 and finite service limits. It overlapped background measurements;
this is not evidence of exclusive host or memory-fabric use.

## Queue observation

The queue has accepted 41 new FP32 first processes, retained 27 failures,
skipped already-attempted cell 60, and is running `formal-bound-auto01-cell097-r1`.
43 first-process slots have not started. With the six prior bound results,
**47/120 cells have an accepted numa-bound-solo-v1 first process**; the eight
older unbound observations remain a different series. No cell has three-process
recommendation evidence yet. The FP32 Attention inference profile also passed.

Of the 27 new failures, 19 never acquired 11 devices within the declared wait; three
exceeded execution bounds; two completed but measured 3104.020558 s / 3087.987731 s
against unchanged 3000 s allowances; two retained ProcessLookupError without an
established root cause; one hit NPU OOM in an optimizer clone. Prior cells 36/60
remain failed. No automatic retries, budget changes or numerical-policy changes
were made. Ending the finite queue will not by itself close these evidence gaps.

The eleven-device original-B512 resident FP16 Attention complete-training first
process passed terminal audit: measured 6190.644420 s, warmup 5804.552517 s,
construction 393.357138 s, and 1.984931 input tokens/s. This remains one process in
its separate FP16 series; no cross-series speed or strict-equivalence claim follows.
