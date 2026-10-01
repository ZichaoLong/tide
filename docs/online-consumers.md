# Continuous model consumers

`scripts/run_execution_flow.py` consumes a hash-validated v2 workload with one
command interface for independent Python execution, a Python native adapter,
and an independently linked LibTorch program. These are experiment consumers of
the public library. They do not add a scheduler, a loss requirement or a model
head to core graph semantics. CPU FP32/FP64 and directed mixed NPU FP32 complete-training qualification is
recorded at [clean fe2d886](evidence/online-consumers-20261002.md):120 CPU checks
and18 NPU cases, no skips. This is small-model correctness, not full-size throughput.

## Workload and parameters

`prepare_execution_flow.py` now defaults to `--protocol continuous`, schema
`tide-complete-flow-workload-v2`. The old `--protocol reset-v1` still emits the
unchanged v1 reset-window contract. The continuous runner explicitly rejects v1.
Both records include the same reachable topology and exact parameter formula:

```
Add:       edges*D*D + nodes*D + edges + input_ports + 2*vocab*D
Attention: Add + 4*nodes*D*D
```

Every physical body edge has its own learned D×D output projection. Body Full is
SiLU followed by learned RMS normalization, with fixed zero projection biases;
output boundary projections are fixed identities. Read is the declared FP32 norm
profile. Add uses fixed .99 retention and learned all-source-softmax logits.
Attention uses four heads, learned QKV/output matrices and all-source-softmax
fiber pooling, fixed zero biases and .01 decay. Embedding and head are learned.
Non-learned scaffolding is shared where safe and excluded from the count.
Both implementations assert the actual learned tensor count against the packet.

`named-lcg31-v1` defines CPU FP32 initial values independently of framework RNG:
start `key=seed mod 2147483647`, fold each ASCII parameter-name byte using
`key=(131*key+byte) mod 2147483647`, then initialize element i to `x=(i+key) mod
2147483647`. Apply `x=(1103515245*x+12345) mod 2147483647` three times. Its FP32
value is `((x mod 65536)-32768)*2^-20`, then explicitly cast to payload dtype.
All integer intermediates fit signed int64. Norm and pooling coefficients start
at one. Canonical initializer names are in the matching Python/C++ builders.
No candidate runs another implementation to get numerical inputs or routes.

The wide packet remains 480 reachable body nodes, 2,208 body edges, D2048,
B512/T12/V50304: Add9,468,053,696 and Attention17,521,117,376 learned elements.
This is the reachable cross-family fixture in the execution contract, separately
identified from the historical two-cortex 8.8B/17B workload. Parameter count alone
does not establish model equivalence. `representative` offers a bounded D128
fixture; packet generation is independent of Torch and device availability.

Rank-aligned packets have equivalent Settle/TimedDAG/PDG encodings. Delayed packets
exercise TimedDAG/PDG unaligned arrivals and refuse a Settle equivalence claim.
Settle encoding shares body owners and allocates only the two boundary nodes;
it no longer temporarily constructs another full set of body parameters.
The fixtures do not constrain the underlying general online greedy scheduler.

## Execution and training boundary

The runner exposes `--schedule streaming|prefill`; prefill selects the public
general online greedy algorithm. It continues numerical state, history, KV,
pending messages and the absolute input ledger through warmup and measured
windows. At each optimizer boundary it explicitly detaches the graph, preserving
the numerical continuation. There is no per-window state reset.

Input token IDs are `(absolute_position*7+sample*3)%vocab`. The external head
predicts the next position. Cross entropy sums over present outputs and divides
by the requested token count across that update's windows, including positions
with no output. A wholly absent loss remains disconnected. SGD and AdamW include
all learned graph, embedding and head parameters; None gradients are preserved.
Gradients must be finite before any optimizer update. This short synthetic loss
checks the training mechanism, not downstream convergence or model quality.

FP32/FP64 eager training is implemented. FP16 master/head consumer updates remain
pending and are explicitly refused. This does not remove the separately verified
public resident FP16 training API. The consumer is a bounded benchmark runner;
it does not claim to serialize an application bundle with head/data cursor.

CPU and mixed A/B/C use public Read/control/selection placement, with the fine
switches `--read`, `--control`, `--selection`, `--events`, `--scoring-dtype` retained.
The current consumer is single-device. The consumer has not yet integrated resident
training and sharding; `--preset resident` fails explicitly.
Broadcast resident qualifications cannot certify this per-edge model. [HARD slot-affine reverse/publication](resident-emission-vjp.md) is being developed
and tested separately; compact projection owners and consumer integration are
prerequisites to
the resident wide comparison, alongside total memory and chunk admission.

## Commands and records

From the repository with the intended Torch/runtime environment active:

```bash
python scripts/prepare_execution_flow.py --preset smoke --memory attention --output-dir artifacts/flow-input
PYTHONPATH=python python scripts/run_execution_flow.py --packet artifacts/flow-input/workload.json \
  --output-dir artifacts/flow-python --device cpu --dtype float32 --family settle \
  --implementation python --preset cpu --schedule prefill --training --optimizer adamw
```

The native Python client uses `--implementation native --native-library BUILD_DIR`.
Build the independent C++ program against an installed matching public core:

```bash
cmake -S tools/online_bench -B build/online -DCMAKE_PREFIX_PATH=/path/to/TideGraph/prefix
cmake --build build/online --parallel 2
python scripts/run_execution_flow.py --packet artifacts/flow-input/workload.json \
  --output-dir artifacts/flow-libtorch --device cpu --dtype float32 --family pdg \
  --implementation libtorch --native-binary build/online/tidegraph-online-bench \
  --preset cpu --schedule streaming --training
```

The LibTorch launcher does not import Torch. It derives and hashes the exact v2
text input from the validated JSON, records the executable digest, and launches
the independent runtime. Direct C++ invocation validates structure but treats the
header SHA as a declared identity; use the shared launcher for a verified packet.

`--steps`, `--warmup`, `--windows-per-step`, `--threads` and new output directories
are explicit. `--parameter-budget` defaults to1GiB of learned payload storage,
checked before allocation; it is not a total peak-memory estimate. Raising it
alone does not certify safe full-size execution. Native and Python runs preserve
failures in `result.json`; the standalone log remains in `consumer.log`.

Construction and warmup durations are separate. A measured step includes token
preparation/upload, online graph execution, head/loss, backward, finite checks,
detach/optimizer and final synchronization. No independent reference is timed.
Actual output counts and available scheduler counters accompany each sample.
`--diagnostics` adds bounded states/routes/gradients/updates JSONL and is not a
formal timing mode. Profiler runs and three fresh-process recommendation repeats
remain separate requirements; one process's step timings do not satisfy them.
Full-size peak memory, aggressive-safe chunking and complete F6 comparisons remain
pending. The deliberately paused historical CPU job is not managed by this CLI.
