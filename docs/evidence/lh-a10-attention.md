# Original a10fdb1 Attention: local CPU measurement

This report records completed original-LH tests, separate from the earlier
[Add pilot](lh-local-scale-pilot.md). It establishes timing at the declared
configuration; it does not yet reproduce the historical 8.8B/8.5B models or
measure a matched Tide/LH speed ratio. Current live jobs belong to
[STATUS](../STATUS.md), and workload details to the
[original-test contract](../lh-original-test.md).

## Source and method

- Frozen Tide runner: `6ade84dffa2101bb18aa66afd12d167cdad0816e` at
  `/var/tmp/zlong-graph-execution-foundation/qualification/lh-a10-20260922-0830`.
  All 372 tracked files match its Git archive; the frozen checkout is clean.
- Original LH: `a10fdb1883fccd63ec21e36cc9cffa294c63c9e2`. The current LH
  workspace is read-only and has unrelated local BatchHidden.cpp edits;
  this run uses an independent clone of the exact selected revision.
- Changes: CPU CMake/Release/OpenMP build, eight emitD/receiveD JSON values,
  and explicit selector/batch/step constants where required. Original kernels,
  cache handling, scheduling, test inputs and Think timers are preserved.
  The large wide test retains batch 512 / selectnum 1 / 100 steps.
- Graph: original Graph.py/BaseUtils.py, preserved seed7 adjacency,
  `[0,1,7,224]` with 232 total static nodes and four block edge counts
  `984/984/232/8`. Historical graph bytes and RNG state are unavailable.
- Host: HiSilicon aarch64, 320 physical cores, about 2 TiB RAM; Torch 2.10.0+cpu,
  FP32, CPUs 160–319 (sockets 2/3, NUMA 4–7), ATen/OpenMP 160, OpenBLAS 1.
  Default first-touch, shared host, no exclusive resource reservation.
- Bounds: large cases sequential, 1800 s each, RLIMIT_AS 1280 GiB. This is a
  virtual-address bound; actual child RSS is measured with Linux wait4.
  The user authorized roughly half the machine's cores and memory.
- Each ms/sample-token is the original integer-ms Think time divided by 512.
  Whole-process elapsed time includes construction and teardown outside Think.
  The original test performs forward only; grad-forward has no backward or
  optimizer call. It does not set a deterministic model/input seed.

## Completed observations

| Case | Actual parameters | Steps | All steps ms/sample-token | Steps 4–99 ms/sample-token | Peak RSS GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| width 2048, nograd | 17,269,426,339 | 100/100 | 26.6342 | 27.1745 | 193.506 |

The full 100-step aggregate throughput is 37.5457 sample-tokens/s; the corresponding
per-batch step is 13.6367 s. The process took 1429.701 s. All 100 native timer/token
pairs completed and the native parameter count matches the independent build
preflight. Process exit and wrapper exit are both 0.

The first step is 6.58984 ms/sample-token; it includes state creation but begins
with less routed work. Steps 4–11 average 24.5820 and steps 80–99 average 28.5503,
about 16.1% higher. This is not a controlled context-length experiment: local KV
history, routes, batch occupancy and host conditions can all change. Neither
block_size=4096 nor 100 calls to think establish long-context performance.

For steps 4–99, the original non-overlapping leaf timer groups average
11.0834 ms/sample-token for BaseEmitToEdges Total (40.8% of Think) and 14.6896
for chal (54.1%). The latter includes node Attention processing and associated
cache work, not only attention-score arithmetic. Do not sum nested parent and
child timers. Whole-process CPU time averages 53.19 equivalent occupied cores;
160 is the available thread/core budget, not evidence of full utilization.

## Evidence and boundaries

Artifacts: `artifacts/lh-a10-attention-20260922-0830/`.
The completed case is `pilot/wide-nograd/`, containing run.json, summary.json,
metrics.jsonl, resource.json and original stdout.log. analyze.py rechecks source,
parameter/configuration, binary, graph, graph-input and vendor hashes, validates
run records and writes comparison.json and record-validation.json. The frozen
Tide tree audit is in tide-source-audit.json. plot.py renders the raw time series
to original-attention-times.png/svg and records its source hashes.

The two width 64 / batch 4 / 4-step smoke modes also passed (23,133,347 parameters).
Five directed timer/resource tests passed before the durable run. Three
completed run records validate at this checkpoint. Trackio is best-effort and
explicitly degraded because it is not installed; local observations are intact.

Retain the earlier 0820 failed launcher separately: its smoke build succeeded,
but /usr/bin/time was missing and the native test did not start. The current
runner uses wait4; success here does not change that historical failed record.

No full tensors, states, routes or backward values are exported by the original
test. Completed token inventory and timing do not establish numerical or training
equivalence. The current homogeneous Attention model has 17.27B parameters,
whereas the historical wide reference was approximately 8.8B. Machine, thread
count, graph RNG and historical grad scope also differ. No historical timing
ratio or Tide speedup is claimed. The matched weight-preserving Tide importer
and paired timer remain separate unfinished work in ROADMAP.
