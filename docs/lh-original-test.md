# Original a10fdb1 Attention test reproduction

The user identified `a10fdb1` as the original LH code baseline, with only a few
configuration parameters needed for the requested wide/narrow cases. This
supersedes selecting old Add configurations by their hidden width. The earlier
[Add pilot](evidence/lh-local-scale-pilot.md) remains separate evidence.

Use an isolated clone of the exact revision; do not use the current dirty LH
tree's BatchHidden.cpp. The actual reference tree remains read-only. Preserve
the original `test-cortexnet.cpp` execution and Think timer. Changes are limited
to the CPU CMake build and an explicit list of width, selector, batch and step
parameters; original numerical kernels and scheduling stay unchanged.

## Parameter mapping

The revision's default model is width512, all Attention in both channel
segments and Pronounce, four heads, clear=true, SiLU/RMSNorm, allsoftmax,
two body layers and vocabulary50304. The test has batch512, selectnum1,
100 calls to think, CROSSBATCH attention and no backward/optimizer call.
`block_size=4096` is not evidence of a 4096-token test. Node KV lengths also
count local received fibers, not simply the number of output tokens.

| Profile | Width | Per-level graph counts | localnum | selectnum | Expected parameters |
| --- | ---: | --- | ---: | ---: | ---: |
| wide | 2048 | [0,1,7,224] | 32 | 1 | 17,269,426,339 |
| narrow | 128 | [0,1,7,56,448,57344] | 128 | 2 | 16,608,289,021 |

Graph recipes and their preserved seed7 bytes come from the original graph
generator. The original graph-data directory was not committed (only .gitkeep),
so exact historical adjacency/seed is still unknown. The C++ test's selectnum
literal changes to2 for the narrow nominal1/64 case. Only emitD/receiveD change
in the model JSON; attention, clear, heads and other equations remain as in
a10fdb1. Distinct parameter counts are checked against the instantiated model.
The reported historical 8.8B/8.5B sizes remain approximate references; do not
silently change memory modules to force those numbers.

The original CROSSBATCH implementation eagerly allocates initial KV capacity
for every node and batch row. At FP32/batch512, initial key/value tensors alone
are about58.125GiB/904.008GiB, before parameters, metadata, cache growth and
autograd. Use the user's expanded budget: at most160 physical cores on this
320-core host, large cases sequentially, with explicit memory/time bounds.
Initial planned placement is CPUs160–319, OpenMP160, OpenBLAS1. This differs
from the earlier56-core pilot and cannot isolate a module-only speed difference.

## Tools and qualification

`scripts/build_lh_original.py` clones the selected revision, checks the original
generator against the retained graph input, copies the graph bytes and vendored
JSON headers, lists parameter changes, and replaces CMake with
`cpp/lh_original/CMakeLists.txt`. It builds grad/nograd executables sharing the
same original library. The build manifest hashes original/configured sources,
graph, vendor headers and binaries. Git archive comparison permits only CMake
and the recorded parameter-file hashes to differ from the selected revision.

`scripts/benchmark_lh_original.py` runs the original executable with explicit
CPU/FP32, thread counts, timeout and RLIMIT_AS. It requires all configured token
completions, pairs each original Think timer with its token, checks native
parameter counts and rechecks source/binary/graph identities. The timer has
integer-millisecond resolution and includes original inner timer printing and
first-step state initialization. Input IDs and model RNG follow the original
test without an added seed. No tensor/route parity is claimed from timing logs.

The raw log remains authoritative. Metrics are projected after process exit;
their timestamps are projection time, not original token emission times.
Report the full100-step mean, a separately labeled steps4–99 mean, and early/
late windows. Never erase the original cold first step. GNU time records whole
process RSS and elapsed/resource usage; it is distinct from summed Think time.
Best-effort Trackio can degrade while complete local records are retained.

Small bring-up may override width64/batch4/steps4; those overrides remain
explicit in the manifest. They do not substitute for the requested scale.
Run nograd before grad-forward and retain timeouts/failed cases. The original
LH autograd implementation is not a Tide training reference; reproducing its
grad-enabled forward does not qualify its backward correctness.

Build/run from a frozen Tide commit and use fresh output directories:

```sh
python scripts/build_lh_original.py --source /home/zlong/llm/lh \
  --revision a10fdb1 --profile wide \
  --graph-input /absolute/artifacts/lh-local-wide-input-20260922-0720/prepared/input.json \
  --json-include /absolute/lh-snapshot/vendor --output-dir /absolute/new-prepared --jobs 2
python scripts/benchmark_lh_original.py --device cpu --dtype float32 \
  --prepared /absolute/new-prepared --output-dir /absolute/new-run \
  --mode nograd --threads 160 --blas-threads 1 --memory-gib 1280 --timeout-seconds 1800
```

Execution status and exact durable jobs belong to STATUS; implementation and
remaining comparison obligations belong to ROADMAP. No result is implied by
the configuration estimates or a successful build.
