# Bounded native streaming measurement

This first M8 workload measures execution and graph/state management for the
existing EMA/tanh profile on CPU FP64/FP32. It is separate from correctness
qualification and does not establish LLM quality, Attention/SSM performance or
the full scale target. Exact measured evidence is added only after a controlled
run; the current handoff is STATUS and remaining work is ROADMAP.

## Workload and correctness

`parallel-ring-prefix-v1` has disjoint eight-node positive-delay rings. Each
node has two distinct parallel unit-delay edges to its successor and its own
budget-one region. Every node shares the same immutable local weights; edge
send scale is 0.2. Boundary inputs/outputs belong only to the selected active
prefix of rings. One input per ring/sample at time zero launches a continuing
wave. Increasing the dormant ring count preserves every active physical ID,
input, source slot, route and expected event/message count.

This sharing isolates graph/runtime overhead from the cost of independent model
weights. It must not be used to estimate the memory of a large unshared LLM.
Batch samples are independent. Several rings give real independent node jobs;
packed mode batches samples at each node. No time-prefill claim applies to this
cyclic streaming workload. The first CLI bounds are nodes <= 1,000,000 (multiple
of eight), active rings <= 128, batch <= 1024, width <= 4096, ticks <= 100,000,
node workers <= 64 and at most 10,000,000 candidate events per repetition.
These are guards, not qualified scale claims.

Before timing, a serial/unpacked functional executor on only the active rings
checks a traced run of the requested configuration on the full graph. Comparison
covers candidates, all trace tensors, routes, fibers/slots, state clocks/values,
history, ledger, pending messages and labeled outputs. The declared dormant
prefix embedding omits only whole-graph identity equality. Each timed run still
checks its own identity against its engine and compares its complete final state,
history, pending, ledger and outputs to the anchor outside the timer. Independent
Python literal recurrence tests check output, state and pending checksums as well.

## Timing and resource definitions

- One process/run has one variant, workload and dtype. Construction measures
  graph/model creation, compile/validation, kernels and worker-pool creation once.
- Every warmup and measured repetition starts with an empty continuation. Reset
  includes identity initialization and cursor import, where applicable.
- Advance is the sum of per-cut native API intervals. Functional continuation
  replacement is included. Collection, checks, checksums, JSON and Trackio are
  outside these intervals. Returned trace/messages are disabled while timing.
- Cursor snapshot is timed separately after the final cut; it includes graph
  identity copies, pending sort and tensor clones. Functional results already
  export complete continuations on every cut; their extra snapshot time is zero.
- Inference uses no_grad, one ATen/inter-op/OpenBLAS thread and explicit node
  workers. Packed semantic autograd replay counters must be zero.
- Peak RSS is the whole child process high-water mark, including construction,
  traced correctness checks, warmup and measurements. It is not isolated
  steady-state execution memory. Identity bytes and retained-state/message counts
  are recorded separately. No allocator/allocation-count claim is made.

Each repetition preserves raw latency, throughput in candidate events/second,
work counters and process RSS. Summary uses median/min/max/population standard
deviation. Construction and initial validation are single observations repeated
as run-level context; their repeated fields are not independent timing samples.

## Entry points and records

Build with `scripts/build.py`, then use an immutable clean checkout:

```sh
python scripts/benchmark_streaming.py --device cpu --dtype float32 \
  --nodes 4096 --active-rings 4 --batch 4 --ticks 32 --width 16 \
  --api cursor --workers 3 --packed 1 --warmup 2 --repetitions 5 \
  --output-dir artifacts/streaming-example
```

The standalone `build/tidegraph-streaming-bench` uses the same workload/runtime
flags, plus `--run-id ID` and its own new `--output-dir`. It writes flushed metric
events to `metrics.jsonl` and uses a nonzero exit for correctness/CLI/I/O errors.
The Python wrapper owns `run.json`, `stdout.log` and `summary.json`; C++ alone
writes `native/metrics.jsonl`. After the child exits, the wrapper retains an exact
copy at the run root for portable record validators and optional Trackio import.
During a live run, inspect the native file. A failed native exit remains failed
even if metrics exist. Source/build/binary identities must match and stay fixed.

`--allow-dirty` is only a development smoke path; it archives source and records
its hash. `--tracking best-effort` is the default optional local Trackio projection;
`required` fails before expensive work if initialization fails, and `off` imports
no Trackio. Local raw metrics survive projection failure. The adapter requests
fresh runs and disables automatic collectors, records destination/version/health,
and makes no durable-acknowledgement promise. It starts no dashboard/listener.
Runtime code does not import any installed skill file.

Use the detached job workflow in `verification.md` for retained measurements.
Keep system load and source/workload differences visible when comparing runs;
the small first pilot does not justify a large sweep or a speedup claim.

The fixed first comparison is `scripts/pilot_streaming.py --device cpu --dtype
float32 --output-dir NEW_PATH`. It runs nodes 32/4096 x API functional/cursor x
workers 1/3 x packed 0/1, holding all other workload settings fixed. It rebuilds
from a clean commit and stops at the first failed child. This is 16 declared
comparisons, two warmups and five measured repetitions each, without adaptive
search or automatic scale enlargement. Per-case manifests retain CPU affinity
and initial system load; co-running work and limited size keep it a pilot.
