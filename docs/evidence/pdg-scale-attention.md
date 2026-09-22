# First graph-only LH-scale PDG Attention measurement

Implementation: f6686c124c8f588a44c1825603ffa5c8843dd422. The user explicitly
requested similar topology, parameter/compute scale and modules without LH
weight import. This report covers a completed wide parallel case; the enclosing
pilot is still running serial and then narrow cases. It is not a full-matrix
or training-performance qualification.

## Source, correctness and records

CPU aarch64, Torch/LibTorch2.10.0+cpu, FP32, CPUs160–319, node workers160,
ATen/inter-op/OpenBLAS1, Nice10/background.slice. Shared host, default NUMA policy.
The source is a clean read-only worktree under
/var/tmp/zlong-graph-execution-foundation/qualification/pdg-scale-20260922-1020.
Binaries built during the development gate were copied to its isolated build;
their recorded C++ source hash equals this commit's exact C++ hash. The build
manifest preserves its original development-time revision instead of relabeling it.

Development: artifacts/pdg-scale-dev-20260922-1120/;57 tests in24.91s passed,
including new FP32/FP64 trace/state/history/pending/route comparisons between
scalar slot, scalar row, packed row and parallel packed row, plus streaming and
cursor regression. The retained dirty-source archive identifies that gate.
The clean pilot independently passed FP32/FP64 small checks on the actual wide
topology, and a small grad-enabled forward (no backward). A deliberate1s timeout
produced a failed run with11 retained steps, native exit-15, complete records
and no unreaped child. That expected failure remains labeled failed.

Job: tide-pdg-scale-20260922-1020; records:
artifacts/pdg-scale-20260922-1020/. Completed per-case manifests, original native
metrics and root copies, summaries and logs are under comparison/. analyze.py
hashes its own source and LH inputs, validates terminal records with the retained
validator identity, and writes comparison.json. All currently completed cases
validate and native metric copies match. Trackio best-effort is explicitly
Degraded/ModuleNotFoundError; local records are complete. No dashboard was started.

## Workload and comparable window

The retained four original graph blocks have232 static nodes per cortex,
2208 logical edges and the single-PDG encoding has465 nodes/4418 physical edges.
Four-head fiber Attention, all-softmax source pooling, SiLU/RMSNorm, clear after
activation and localnum32/selectnum1. Fresh normal(std=.02) parameters;
no LH parameters are read. Exactly17,269,426,339 learned elements, including
embedding and vocabulary head, equal the native LH count. Physical phase wires
alias logical weights. Body and readout use the existing positive-delay clocks.

PDG row Emit uses one dense Linear per node for its combined cortex/bridge row;
LH uses one Linear per nonempty block row. PDG timing includes embedding, two
body ticks, readout and vocabulary projection. Validation, metrics and model
construction are outside token timing. PDG uses fixed external token IDs and
on-demand KV; original LH uses its unseeded greedy loop and initial KV reserve.
Both run without grad. See ../pdg-scale-benchmark.md for the detailed contract.

Both timing rows below use **token indices4–11**, batch512 and hidden2048.
The means divide measured wall time by both batch and token steps.

| Implementation | Mean ms/sample-token | Median | Min–max | Aggregate sample-tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Original LH a10fdb1 | 24.58203 | 24.82520 | 22.28320–25.98242 | 40.6801 |
| Native PDG cursor, row Emit, packed,160 workers | 33.75842 | 33.79813 | 31.83653–35.57944 | 29.6222 |

PDG took 1.3733x the amortized time in this short window
(about37.3% more). This is a comparable-scale observation; different random
weights/input trajectories, schedules, kernel organization, cache occupancy and
shared-host conditions prevent attributing the entire difference to scheduling.
No speed ceiling, model quality, exact LH parity or long-context claim follows.

Both executed32 selected **body** node-sample events per sample-token in that
window: PDG selected_events minus readout_rows, and four original LH activation
counts per token independently parsed from stdout. PDG additionally averaged
146.20532 candidate events and
165.00781 source rows per sample-token (including
readout). Its average rows per packed update call was
81.24078. Equal selected counts do not establish
identical candidate/attention FLOPs or route distributions.

PDG completed12/12steps, native exit0. Construction took
226.84463s. Whole-process peak RSS was
109.39629GiB, including construction and all12steps. LH's prior
193.50579GiB peak covers100steps; these peaks are different-length observations,
so this report does not claim a matched memory reduction.

Earlier ramps also completed: width256/batch32,6steps; and the same17.27B wide
model at batch64,8steps. The latter's token2–7 mean was108.85616ms/sample-token,
peak95.58398GiB and construction226.92974s. Its time rose from63.10936 to140.35034
within the measured window. Do not compare that different batch/window as a
matched LH speed ratio.

## Live continuation

At this checkpoint the wide serial case is running (workers1, ATen/BLAS1,
batch512,4steps,600s process limit); it is an intentionally serial baseline.
The narrow parallel case is queued:115,713 PDG nodes, width128/batch512,
nominal2/128 leaf budget,8steps,1800s. The overall pilot has no terminal result.
The process address-space limit1280GiB is separate from measured RSS.

The durable command is recorded verbatim as an argv array in status.json,
working directory is the frozen checkout above; runner is
/home/zlong/anaconda3/bin/python scripts/job.py, wrapping scripts/pilot_pdg_scale.py
with the wide/narrow input paths under artifacts/pdg-scale-input-20260922/ and
comparison output under this job directory. Inspect or stop with:

```sh
systemctl --user show tide-pdg-scale-20260922-1020 -p ActiveState -p SubState -p Result -p MainPID
tail -n 30 artifacts/pdg-scale-20260922-1020/task.log
systemctl --user stop tide-pdg-scale-20260922-1020
```

After each terminal case, rerun analyze.py, inspect its status/exit code and
update STATUS plus this evidence. Preserve failed/partial cases; a live job is
not passed. Large grad-forward/backward, longer context and attribution by
operator/scheduling phase remain future measurements.
