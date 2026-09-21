# Development gates and immutable qualification

All current targets are CPU FP64/FP32. Use the matching Torch Python for scripts
and LibTorch discovery, one ATen/BLAS thread and two build workers. The active
source, unit, output paths and exact next commands belong in `STATUS.md`.

## Gate scopes

| Entry point | Scope |
| --- | --- |
| `develop.py TESTS...` | Archive dirty source, build, run explicitly selected pytest paths |
| `develop_lh.py --component iocortex --scope smoke` | Archive source, run six IOCortex configurations per dtype with original assertions on and off; no exports |
| `develop_lh.py --component iocortex` | Full IOCortex matrix, both assertion variants, complete fixtures and independent Python comparison |
| `qualify.py --lh-snapshot SNAPSHOT` | Full CPU regression and all original LH component gates, including IOCortex/Python |

`full` remains the default; the clean qualification entry point has no smoke
switch. The six smoke configurations collectively include all six pooling
profiles, both clear/lead settings, three original execution modes and three
native schedules. Every selected configuration still runs actual `think` and
ragged `think_single_step`, complete state/message comparisons and every-cut
single-PDG checks. It is not the Cartesian full matrix or a qualification claim.
The assertions-on FP64 diagnostic limitation is avoided in smoke by selecting
original single mode for active-softmax; full qualification retains and reports
the 24 unavailable configurations, and tests them in the assertions-off build.

`check_lh_selector.py` records scope and exact commands in its manifest. Smoke
exports no fixtures. `check_lh_iocortex_python.py` requires an explicitly full,
passed gate, correct source and binary fingerprints, and the exact hashed
fixture inventory. A partial or old unscoped manifest is rejected.

## Source and result lifecycle

1. Use short directed tests while editing. Preserve a failed reproducer and its
   source identity. Commit a coherent implementation after its directed gate.
2. Create a detached worktree at that exact commit for long qualification. Give
   it its own build directory and a new durable output directory. Pass the
   immutable original LH snapshot by absolute path. Do not share mutable build
   caches with a concurrent development or qualification job.
3. Mark tracked worktree files read-only and leave that checkout unchanged until
   termination. Run `scripts/job.py --output-dir ... -- python scripts/qualify.py
   --output-dir ... --lh-snapshot ...` through the documented detached user
   service, in `background.slice`, with `Nice=10` and explicit environment.
4. Confirm the live service's PID, slice/control group and persistent running
   record. Record its exact source/worktree/command in STATUS; development may
   continue on the main branch because the job reads only the frozen worktree.
5. Inspect terminal service status, workload exit code and every expected result.
   Check source identity, clean state, inventories and acceptance counts. Commit
   reviewed evidence separately. A submitted or live job has no pass result.

Retain artifacts referenced by evidence and unresolved failure reproducers.
After terminal verification, remove only redundant frozen worktrees/builds whose
commit and logs remain available. Inspect the cleanup dry run; do not remove
active jobs or reference sources. Project `build`/`artifacts` may be symlinks to
a filesystem with space; never assume the shared repository volume has room.
