# Historical cancellation and build cleanup — 2026-10-12

Maintenance against implementation HEAD
`0edcd2bdd8149b9b18b1a2c1387b2f89e30f6847`. No implementation or qualification
scope changed. `TASK=/mi/data2T/zlong/tide-execution-flows` below.

## Historical CPU task cancelled

The user's explicit cancellation supersedes the former keep-paused constraint
for `historical-cpu-attention-01`. Its purpose was an old full-size CPU Attention
training timing against an old NPU baseline; it is no longer required by the
finite selection acceptance. No measured update was recorded, so that historical
comparison remains unavailable.

`systemctl --user stop --no-block
 tide-execution-flows-historical-cpu-attention-01.service` terminated only this
service group. The final unit has MainPID 0, empty ControlGroup and exit 143
(SIGTERM cancellation; systemd reports failed/exit-code). All four original PIDs
2686854, 2686877, 2686878 and 2686919 are gone. The worker's pre-cancel RSS was
136,109,900 KiB, about 129.80 GiB; this is process memory, not disk reclaimed.

The job wrapper and child run recorded cancellation. The stopped parent left a
stale running matrix; it was finalized from the observed cancellation, with
unstarted repeats 2–3 explicitly cancelled. No performance result was fabricated.
The original pause record, pre-cancel snapshot, logs, inputs and failures remain.
Final status/run/matrix, original pause and snapshot hashes were checked.

Records:

- `TASK/plans/historical-cpu-attention-cancel-20261012-01/pre-cancel.json`
- `TASK/plans/historical-cpu-attention-cancel-20261012-01/result.json`
- `TASK/runs/historical-cpu-attention-01/cancellation.json`
  SHA256 `bcc2e0a4b4992966b0ae942b76f1959554b26c729e45a948c8a0e2c15e513930`.

No historical repeat or restart is scheduled.

## Conservative build cleanup

The inspected dry run covered only unreferenced, private regular `.o` files
under `TASK/builds/**/CMakeFiles/*.dir/` and their paired `.o.d` files. Each
candidate had exact ownership, link count and metadata checks. Cited object
basenames were conservatively protected in every build, excluding 5,684 objects.
No source, linked binary/library, manifest, log or trace was deleted.

The apply service completed at 2026-10-12 01:39:50 UTC with Result success,
MainPID 0 and an empty cgroup. It removed 19,347 objects and 15,829 dependencies
across 177 builds, **35,176 files / 17,826,045,952 allocated bytes (16.60 GiB)**.
Build storage decreased from 51,348,828,160 to 33,522,782,208 bytes (31.22 GiB
remaining). Filesystem free space was 77,280,985,088 bytes (71.97 GiB) at finish;
this is a shared-filesystem observation, not a reservation.

Post-delete checks confirmed every listed target absent, **1,297 preserved
records unchanged by SHA256** and **3,380 linked products unchanged by metadata**.
Linked products were not rehashed during this cleanup. Active-service and
same-user process cwd/open-file/mapping guards passed before deletion. Linked
programs remain usable; a later build may regenerate the removed intermediates.

Receipt directory: `TASK/plans/storage-cleanup-20261012-build-objects01/`.

| Record | SHA256 |
| --- | --- |
| `dry-run.json` | `4965fe1ae14aca99d62d034580c51c1b9359652ced01311e879e1c153aa55a00` |
| `result.json` | `508c85695a960fa8056fbd1f9ca5953e9138355a9c6cdbc98169b450d2552ab5` |
| `cleanup.py` | `2bdc03d2023911c83af7172f22d7f1669bb2c5e58025c617887284f9b3b2a4ca` |
| `durable_records.py` | `a87ea2f7ea8a4c025487c0176d048d6a8963a98d95124a29b2e36fbbdc03ad89` |

`gef-storage-objects-plan-20261012-01.service` failed before planning/deletion
because its minimal PATH lacked `rg`; its original log remains. Plan02 used the
absolute tool path and passed. `gef-storage-objects-apply-20261012-01.service`
then passed. No prior failure is relabeled.

Final inspection found no active `tide-*`/`gef-*` units, no queued systemd jobs
and no user timers. `scripts/status.py` still reports the known malformed
historical `build-reverse-gather-python-dev01` record and exits 1; that record
was preserved. No new experiment was started, and no runtime tests were needed
for these maintenance-only documentation changes.
