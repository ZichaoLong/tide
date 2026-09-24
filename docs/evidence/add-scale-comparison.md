# Add scale comparison and foundation status review

Clean implementation source: `0c053ebc7f8b3be03220c54aebf56502f5f0f6f1`. All4 fresh native smoke cases and
12 large cases completed with exit0; three independent processes per engine and
mode. Terminal audit: `artifacts/add-scale-comparison-20260924-a/reviewed-audit.json`.
Unit `tide-add-scale-comparison-20260924-a` inactive/dead, MainPID0/exit0;
no descendants remain. Only CPU/aarch64 FP32 performance is claimed here.

## Configuration and measurement

Fixed wide graph, D2048/B512/V50304, nominal leaf selection1/32. Each cortex has
224 leaves+8 hubs. Four logical edge blocks984/984/232/8; PDG465 nodes and4418
phase-expanded physical edges. Both observe **9,468,020,899 parameters**. This is
9.468B in conventional decimal notation, or8.818 when divided by1024³. It is a
parameter count, not a storage size; bare FP32 values occupy35.27GiB.

Add/all-softmax in both cortex namespaces and readout, fixed computed retention
0.99, SiLU/RMS body Full, selected clear, two body ticks per token. Same graph,
parameter count, dimensions and module profiles; independently initialized
weights. No exact cross-engine function/training or complete two-clock
continuation equivalence is claimed by this scale test.

This host reports HiSilicon/aarch64,320 physical CPUs/4 sockets/8 NUMA nodes,
Torch/LibTorch2.10.0+cpu and GCC10.3.1. Both processes use the same56 physical
CPUs160–215, from the host aggregate half-budget160. Dynamic half-memory launch
bound752GiB; builds use2 jobs; background.slice/Nice10. No heavy project job
runs concurrently with timing. Shared-host interference remains possible.

Actual LH ATen/OpenMP/OpenBLAS pools report56; interop reports320, which does
not mean320 active workers. OpenBLAS uses OpenMP and ignores the intended
OPENBLAS_NUM_THREADS=1 override in this setting. PDG reports all these compute
pools1, interop1, with56 node and56 vocabulary-head workers in separate phases.
CPUAffinity applies to every child/thread. Optional live process samples record
actual RSS, threads and interval CPU use; they are not a per-operator profile.

Twelve token steps, warmup4, indices4–11 measured; state/history and the grad
forward graph persist across the window. Metric is synchronous forward wall
seconds*1000/512. Both intervals exclude construction, token-ID generation,
previous-logits disposal, finite-value/checksum checks and logging; embedding,
body and vocabulary head are included. RAII/phase/operator timers and optional
work accounting are off. Native process limit1200s, separate build limits1800s.
Modes are no_grad forward and **grad-enabled forward only**: no backward,
optimizer, zero_grad or detach. No complete training-throughput result follows.

## Results

Median below is the median of three independent process-window means, not the
median individual token. Range spans those three means. RSS is wait4 whole-process
peak, including setup/checks/cleanup. Raw token observations remain in JSONL.

| Engine / mode | Median ms/sample-token | Repeat mean range | Sample-tokens/s | Seconds/batch-step | Peak RSS GiB range |
| --- | ---: | ---: | ---: | ---: | ---: |
| lh-nograd | 5.82417 | 5.67728–5.98666 | 171.698 | 2.9820 | 50.18–50.37 |
| pdg-nograd | 5.59769 | 5.40132–6.00031 | 178.645 | 2.8660 | 55.19–57.02 |
| lh-grad-forward | 7.12832 | 7.11879–7.14263 | 140.286 | 3.6497 | 66.08–67.02 |
| pdg-grad-forward | 108.23288 | 98.36605–109.83500 | 9.239 | 55.4152 | 84.11–85.17 |

- lh-nograd: 5.98666128, 5.82416724, 5.67728296 ms/sample-token.
- pdg-nograd: 5.59768576, 6.00031386, 5.40132032 ms/sample-token.
- lh-grad-forward: 7.12832012, 7.11878827, 7.14262729 ms/sample-token.
- pdg-grad-forward: 108.23287934, 109.83500226, 98.36605319 ms/sample-token.

PDG no_grad median latency is 3.89% lower than LH. Repeat ranges overlap, so
these observations do not establish a consistent performance advantage. PDG grad-forward is
15.18× the LH latency in this fixed short window. This identifies a
large current implementation cost, not a requirement to adopt LH training
semantics. Historical3.3/3.6ms and the older Intel Attention17ms report remain
user references, with different hardware/software/window details; they are not
reproduction thresholds or a controlled cross-machine/module speed ratio.

## Why training performance remains open

In the first PDG measured window, mean candidate events are
74997.625 per batch-token; selected Full events are
16896. Grad mode adds that candidate count of Aggregate,
state and Read semantic replays, plus16896 Full
replays. No_grad replay counts are zero. Packed Full calls remain only
881.500 per batch-token.

`cpp/src/full_evaluate.cpp` first evaluates the packed numeric batch under
NoGradGuard, then calls FullKernel::step separately for every selected event to
construct its semantic graph. The scale RowEmit step invokes its batch method
with one row, repeating dense projection as individual GEMV-like work. Aggregate,
Read and state have related replay paths. This is a strong optimization target;
this experiment did not time each replay category, so it does not assign an
exact percentage of the total slowdown to any single one.

The replay protects isolated roots, None versus connected-zero and parameter
ownership. Its cost must not be removed by weakening those semantics. A focused
next implementation would retain this oracle while introducing packed VJPs with
explicit dependency/absence handling, then recheck isolated roots, aliases,
optimizer updates and continuation before repeating the same workload.

## Verification, provenance and reproduction

472 directed CPU FP64/FP32 tests passed/113.88s, including Add formulas, source
domains, state clocks, scale scalar/packed/node-worker parity, original/default
runner compatibility and timer/parameter accounting. Gate:
`artifacts/add-compare-dev-20260924-d/`. Earlier-a failed for a wrong pytest path;
-b/-c exposed a FP32-derived FP64 norm descriptor being checked at FP64 tolerance
(absolute difference2.455e-9). The scale comparator now uses payload precision
for such derived values; execution formulas and exact routes/identities did not
change. All failure source archives/logs remain. This extension does not claim a
fresh rerun of the older7741-test global gate.

Fresh exported PDG and LH builds precede four real small runs. PDG small checks
compare scalar-slot/scalar-row/packed/parallel complete traces and continuation;
LH small grad/nograd complete logits match exactly. Across all large repetitions
and both modes, each engine's full12-token logits checksum sequence matches
exactly; there is no LH-versus-PDG checksum equality claim. Checksums supplement,
not replace, the complete small tensor/state/gradient anchors.

The terminal auditor checks all tracked frozen source files against Git archive,
the exported packet inventory, all3 prepared-source/build/binary records,16
completed run identities and schemas, parameter counts, timing denominators,
actual pool settings, RSS bounds and complete unit/process termination. Source,
logs, unrounded metrics, host/load/NUMA records, build commands and checksums live
under `artifacts/add-scale-comparison-20260924-a/`. Launch and host preflight:
`artifacts/add-scale-comparison-a-launch.json` and
`artifacts/add-scale-host-preflight-20260924.json`. Retained drivers/reviewer are
hashed. Reference repositories were read-only; LH is the audited original-a10
snapshot, not the dirty current reference working tree.

Trackio best-effort projection is degraded because the package is unavailable;
all16 local manifests/JSONL/summaries are complete and independently validated.
Project tide-add-scale-comparison, intended local root under this run's trackio/,
storage mode auto. No dashboard delivery is claimed or required for these results.

Portable source archive:
`artifacts/add-scale-comparison-20260924-a/export/cpu-attention-compare.tar.gz`.
SHA256: `3794a7e5d8f5a869c837ad2253510e9c2da88a02dffd2ee183df4387e91c59cb`; 404828 bytes.
Archive contents and hashes were independently checked. It retains the historical packet name but supports Add. Rebuild on the target
CPU against its Torch/LibTorch. See [runner instructions](../../tools/cpu_compare/README.md).
Use `--memory add --work-count 0` and `--mode nograd` or
`--mode grad-forward`; LH `--lh-timer outer`, PDG `--phase-profile 0`.
No old commit checkout or LH tree modification is necessary.

## Overall objective assessment

The [six-stage finite CPU foundation](foundation-final.md) is complete for its
declared profiles: six generic/specialized classes, independent schedules,
native Settle construction/encoding, representative modules, first-order
semantics/checkpoints and migrated applicable performance options. The global
evidence is7741 CPU FP64/FP32 tests,17 relocated smoke variants and36 fresh-process
checkpoint trajectories; the medium assessment has12 configurations/108 runs.

That acceptance does not establish uniformly efficient large-scale training or
success at every large preset. Large evaluation retained7 completions,5 timeouts
and1 unlaunched stage. Its300s limits included construction/warmup/profile, and
one narrow PDG baseline actually used1 worker despite a32-worker limit. Native
self-loop/ring specializations also visit nodes before their Aggregate/Full
pool dispatch; available workers do not imply multi-node speedup for every
specialization. These limits are explicit in the current capability documents.

Recommended priority: optimize packed training replay first; next separate
startup/steady-state budgets and measure actual parallel large sparse workloads;
then address costly Linear/Delta scans and validate one concrete open-weight
model adapter end to end. The present tiny adapter is not arbitrary pretrained
model import. CUDA/Ascend, higher-order AD and whole-controller/RNG/data-cursor
resume remain extensions. Any Ascend follow-up retains the maximum8-card bound
and starts with CPU parity and one real card. CPU/memory half-budget rules and
small correctness gates should continue across all graph families.
