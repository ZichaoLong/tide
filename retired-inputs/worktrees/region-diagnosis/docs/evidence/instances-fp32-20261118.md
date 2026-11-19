# Current FC/HB FP32 delivery — 2026-11-18

Model implementation: `c17c613f178dd076d963da98017abdea8c09db16`.
Fixed mathematics: `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`.
Numerical diagnostic: `d2090654f0c83037d0c0454327994306dcd4b2e8`;
packing diagnostic: `a2c1e08b1617d0e2714e6272e12420e2c5d931ec`;
optimizer replay example: `cc2b5767948d3156a97c7ca729da9b29c0251646`.
Those later commits change diagnostic/example facilities only; model files remain
identical to the qualified implementation. [Complete reviewed observations](instances-fp32-20261118.json)
include configurations, inputs/parameter hashes, samples, work, memory, device
checks, finite deviations and exact launch/source identities. Reading this report
or reproducing its configurations does not require retained artifact directories.

## Delivered and checked

The fresh public `tide` package implements current FC, HB-Lattice, HB-Exchange
and HB-Multiscale Settle/Timed bindings. K/S/X, all four reception profiles,
quad/Shannon/pool, independent/region_shared/layer_shared and the main legal
Hub/partition/topology options are configurable. All execution/performance here
is FP32, with HF32 disabled on NPU. CPU FP64 is used only for accuracy checks.

Frozen CPU qualification: **308 tests passed in 32.39s**. This includes independent
FC and chronological Timed formulas, complete values/strengths and private state,
all parameter/input gradients, parameter inventories, original-time long KV
windows, zero-reception no-write gradients, SSD/Attention formulas, allocation
boundary VJPs, sharing, continuation/flush and optimizer/checkpoint checks.
A separate venv installed the frozen package and ran the public consumer example:
three AdamW updates and exact next-window replay passed. These are mechanism
checks, with a synthetic scalar objective, not evidence of trained model quality.

NPU: **10/10** short/long Attention forward+VJP cases passed. Six whole-model
cases all matched support, metadata and gradient connectivity; all had finite
three-step optimizer updates and exact checkpoint parameters, optimizer state,
next results and final states. **Two cases passed every strict elementwise
numerical threshold; four retain finite deviations.** Outer job exit zero does
not relabel those deviations as passes.

The six-case checkpoint check covers exact saved data and same-device next-window
forward replay. A separate installed FC-K example also executed a fourth AdamW
update after restoring parameters, private history and optimizer state. Both CPU
and NPU completed it with finite values and optimizer-state comparisons within
the stated tolerance. Parameter differences remain: the observed maximum is
1.20e-5 on CPU and 1.79e-5 on NPU, with `regional.1.second.bias` outside the
1e-6/2e-3 threshold. The original exact-update and finite-threshold CPU attempts
are retained. Tiny near-null common-bias gradient differences can be amplified
by AdamW; this is not a claim of bitwise FP32 optimizer replay. The CPU FP64
unit test does verify an exact subsequent optimizer update.

| Whole-model case (d64, B2/L5) | Numerical status |
|---|---|
| FC K, source_residual, quad, independent | within threshold |
| FC S, incoming_wide, pool, region_shared | finite deviation |
| Exchange X, incoming_light_64, Shannon, layer_shared | finite deviation |
| Multiscale Timed X, source_residual, quad, region_shared | finite deviation |
| FC K, incoming_light_128, pool, region_shared | within threshold |
| Multiscale Timed all-K, mixed regions, quad, region_shared | finite deviation |

Output/state/parameter checks keep `atol=1e-6, rtol=2e-3`; the input gradient uses
`rtol=1e-2`. Most deviations occur near zero. The largest raw Full difference in
the six cases is 1.3864e-4 (Timed X); its raw Full relative L2 is 4.40e-6.
One gradient tensor (`describe.encode.left`) exceeds the parameter threshold at
one element; its maximum absolute difference is 3.22e-6 and relative L2 2.29e-5.
Near-zero common score-bias gradients can have large relative ratios while their
absolute errors are about 1e-9; the absolute+relative gate handles them explicitly.

One bounded higher-precision replay of that worst Timed case found:

- CPU FP32 versus CPU FP64: raw Full max error 1.0069e-4, relative L2 3.19e-6.
- NPU FP32 versus CPU FP64: raw Full max error 3.9381e-5, relative L2 1.81e-6.
- Both FP32 variants' complete parameter gradients pass against FP64, and all
  support/metadata agree. FP64 split continuation max error is 4.22e-14.

This supports FP32 accumulation/reordering as the explanation in this case.
It is the same function at higher precision, not an independent semantic oracle;
the separate formula tests supply independent evidence. The original strict
failures remain in the JSON. No execution path was silently changed to FP64.

## Configuration and measurement

All performance cases use n8, source_residual, encoded regional Attention,
role quad with ordinary k1 and `attention_backend=auto`. Read+ and the bound
state/update rules are the fixed main-instance choices. K means KV/sum, S SSD/state,
X ordinary KV plus Hub SSD. FC-K uses independent parameters; FC-S and all HB
cases use region_shared bases with private adapters and private state. The other
reception/allocation/sharing candidates are correctness-tested, not performance-swept.

| Configuration file | Nodes / regions / layers | d | B / L | Ordinary region widths by layer |
|---|---|---:|---|---|
| [fc-k](../../configs/fc-k.json), [fc-s](../../configs/fc-s.json) | 32 / 4 / 4 | 128 | 8 / 128 | 1,1,1,1 |
| [hb-lattice-k](../../configs/hb-lattice-k.json) | 80 / 10 / 5 | 128 | 8 / 128 | 1,2,4,2,1 |
| [hb-exchange-x](../../configs/hb-exchange-x.json) | 92 / 15 / 5 | 128 | 8 / 128 | 1,2,2,2,1 |
| [hb-multiscale-x-settle](../../configs/hb-multiscale-x-settle.json) | 188 / 31 / 9 | 128 | 4 / 64 | 1,1,2,2,4,2,2,1,1 |
| [hb-multiscale-x-timed](../../configs/hb-multiscale-x-timed.json) | 188 / 31 / 9 | 128 | 4 / 64 | same geometry, Timed clock |
| [fc-k-149m](../../configs/fc-k-149m.json) | 32 / 4 / 4 | 512 | 8 / 128 | 1,1,1,1 |

Exchange has 64 ordinary nodes and 28 Hubs; Multiscale has 128 ordinary nodes and 60
Hubs. These are the current mathematical bindings, not historical FC16/LH176.

Execution is the new PyTorch region-window evaluator with ATen tensor math;
NPU uses TorchNPU and its native fusion adapter when applicable. B/L stay tensor
axes. Selected Full queries are packed per region, with rectangular padding;
there is no B/L event dispatch loop. Native LibTorch, resident control or old
mixed-a/b/c labels are not implied by this implementation.

The actual NPU logs report int64 Argsort on **AiCPU**, notably the KV cache's
time ordering; this is a device-side processor, not the host CPU. Allocation
padding and base tensor-format warnings are also retained in the JSON. The NPU
path is therefore not an all-AiCore claim. Do not follow the vendor's generic
FP32-cast suggestion for logical timestamps: it can destroy their int64 order.

Host: aarch64, Torch 2.12.0+cpu, TorchNPU 2.12.0.post3, CANN 9.2. CPU timing used 32
ATen threads pinned to CPUs 0–31. NPU host work used 8 threads on 64–71 and shared
physical NPU 5 (logical npu:0), with the existing occupant and lock identities
bound by the admitted lease. Other host/device work was present. CPU and NPU
jobs ran concurrently on disjoint host affinities; this is not exclusive timing.

Each subject/mode has one warmup and three measured fresh windows, with the
median reported. Throughput is **B×L input positions / seconds**. B counts
independent samples; L counts positions in each input. Model creation/input
transfer are outside timing. Forward includes routing, packing, local math,
state and returns; train additionally includes loss, backward, foreach AdamW
and synchronization. Grad-forward only records autograd. Timed returns the
online tick window plus pending state; the finite flush tail is not timed.
These are prefill measurements, not decode or first-token latency.

All 45 paired CPU/NPU subject/mode rows have identical input and initial-parameter
hashes. Floating-point trajectories need not stay identical: seven Tide audit
rows have different work counts. For example Timed inference has 100,293 CPU
versus 100,292 NPU Fulls. Keep that distinction when interpreting throughput;
these observations do not certify strict equal-trajectory speedups. Counters
come from one audit window after timing, after optimizer updates in train mode.

## Throughput (input positions/s)

| Model | Parameters | CPU: inference / grad-forward / train | NPU: inference / grad-forward / train |
|---|---:|---:|---:|
| FC-K | 9,334,304 | 9,298 / 8,543 / 1,380 | 11,129 / 6,522 / 5,449 |
| FC-S | 2,197,904 | 3,575 / 3,185 / 818 | 5,349 / 5,210 / 3,139 |
| HB-Lattice K | 5,758,704 | 2,303 / 2,007 / 387 | 5,531 / 4,161 / 2,217 |
| HB-Exchange X | 7,647,248 | 1,743 / 1,248 / 231 | 2,687 / 2,393 / 1,218 |
| HB-Multiscale X Settle | 15,694,224 | 292 / 223 / 61 | 440 / 348 / 187 |
| HB-Multiscale X Timed | 15,694,224 | 129 / 111 / 25 | 236 / 196 / 114 |
| FC-K d512 | 149,207,072 | 1,863 / 1,922 / 383 | 5,162 / 4,587 / 2,299 |

The comparison stacks are mathematically different: batched pre-RMS Transformer
with RoPE, native Attention, fused gate/up SwiGLU and final RMSNorm; or a
Mamba2Simple-style SSD block stack. They have no Tide allocation, summaries,
private node histories or state returns. The SSD baseline uses Torch chunk
math, not an official fused Mamba kernel. Matching parameters does not match work.

| Comparison | Actual parameters | CPU: inference / grad-forward / train | NPU: inference / grad-forward / train |
|---|---:|---:|---:|
| Transformer36 d128, FC parameter match | 9,446,528 | 10,551 / 3,299 / 704 | 24,263 / 19,264 / 7,750 |
| Transformer4 d128, FC layer match | 1,049,728 | 17,068 / 40,410 / 18,001 | 208,577 / 164,460 / 43,557 |
| SSD20 d128, FC-S parameter match | 2,190,688 | 8,314 / 6,021 / 1,332 | 8,099 / 7,821 / 3,345 |
| Transformer22 d128, Lattice parameter match | 5,772,928 | 17,967 / 14,090 / 1,174 | 40,770 / 33,614 / 12,131 |
| Transformer80 d128, Lattice node match | 20,992,128 | 2,850 / 2,537 / 280 | 11,668 / 9,275 / 3,744 |
| Transformer29 d128, Exchange parameter match | 7,609,728 | 12,850 / 4,780 / 769 | 30,952 / 24,562 / 9,252 |
| Transformer60 d128, Multiscale parameter match | 15,744,128 | 2,039 / 1,146 / 310 | 3,602 / 3,002 / 1,273 |
| Transformer36 d512, FC parameter match | 151,032,320 | 2,511 / 1,837 / 521 | 15,293 / 15,006 / 4,956 |

The raw samples matter. Some short CPU cases vary by 2–3×; for example the
Transformer4 inference median is slower than its grad-forward median in this
single workflow. That is not an algorithmic conclusion. No exhaustive tuning
or stable scaling claim is made. Peak recorded NPU allocation across these
Tide observations is 7.32GiB (Timed); FC149M peaks at 5.93GiB. The CPU workflow's
sampled process-tree RSS peaked at 9.30GiB. Per-row CPU RSS is cumulative process
high-water RSS, not a resettable per-model peak.

## What the bounded optimization changed

[NodeLinear](../../src/tide/parameters.py) uses a single flattened GEMM when
static owner metadata proves one shared base; private adapters retain BMMs.
It avoids gathering the same base matrix repeatedly and then accumulating its
repeated-index gradients. Mixed groups remain BMM. No per-event autograd splitting
is introduced. Independent tensor/explicit-matrix gradient tests cover both paths.

[Quad allocation](../../src/tide/allocation.py) now sorts the 2n enter/leave
breakpoints of its clipped-linear response, then reconstructs the selected
face from the remaining mass. Normal work is O(n log n), O(n) storage, with
safeguarded bisection retained for ill-conditioned rows. Shannon remains
iterative. The first-order VJP is the solution-mapping derivative, independent
of the solver's finite iterations.

An initial sorted implementation invented a2.22e-16 sender at an exact [0,1]
corner. That changed raw Full support and failed four integrated checks. Stable
face reconstruction and exact endpoint handling fixed it; eight FP32/FP64
regressions preserve both exact zeros and genuine tiny positive senders.
There is no epsilon support clipping and no weakened oracle threshold.

The same new harness/inputs/initial parameters compared the previous fresh
implementation `d7437c21` with `c17c613f` on Lattice:

| Device | Inference before → after | Grad-forward before → after | Train before → after |
|---|---:|---:|---:|
| CPU32 | 2,261 → 2,303 | 1,762 → 2,007 | 334 → 387 |
| shared NPU | 3,956 → 5,531 | 3,540 → 4,161 | 2,020 → 2,217 |

CPU audit work matches in all three modes. NPU inference/grad-forward work
matches; its post-training audit differs by 7 Fulls out of about 75,000. The
training numbers remain measured observations, not proof of identical routes.
The two changes were measured together; individual gains are not isolated.
An 8-thread Lattice calibration gave 1,102/971/276 positions/s, below the 32-thread
case here; it does not establish a universal CPU thread recommendation.

Intrusive synchronized Lattice diagnosis locates remaining cost: CPU forward
0.696s, backward 1.521s, AdamW 0.0149s; NPU forward 0.245s, backward 0.220s,
AdamW 0.0060s. NPU forward includes KV Full 0.113s and allocation 0.0398s. The
separate CPU operator observation is dominated by shared FFN matrix gradients
and private-adapter BMMs; it records CPU times, not NPU kernel times. Intrusive
step timing is separate from normal throughput.

Packing is not free sparsity. With the identical inference input, FC has 9,894
selected Full rows but 19,968 rectangular rows processed (2.02×); Lattice has
74,665 selected versus 80,000 padded rows (1.07×). The optional profiler now
records both counts. FC's remaining padding is a concrete optimization target.

The ordinary budget k/n=12.5% is a mass ratio, not a hard active-node limit.
At equal scores, all eight nodes may send 1/8 and all require Full. Measured
Full/candidate ratios are 30.2% for FC-K,99.9% for FC-S and 91.3% for Lattice.
Shared bases reduce stored parameters, not the number of candidate updates.
Lattice executes roughly 73 Fulls per input position, while its parameter-matched
Transformer has 22 blocks. No 3%-activity throughput conclusion follows here.

## Reproduction and remaining scope

Use a fresh checkout of the recorded source with the project environment from
[development](../development.md). Long runs use [execution-flows](../execution-flows.md),
frozen source, background.slice and finite bounds. Native device admission must
precede NPU commands; the historical shared lease is not a new-session permission.

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q
python scripts/check_upstream.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m benchmarks.run \
  --config configs/fc-k.json --device cpu --threads 32 \
  --subjects tide transformer_parameters transformer_layers --output NEW-fc.json
# In an admitted NPU job:
python tests/device_attention.py --device npu:0 --output NEW-attention.json
python scripts/check_device.py --device npu:0 --output NEW-models.json
python -m benchmarks.run --config configs/hb-lattice-k.json --device npu:0 \
  --threads 8 --subjects tide transformer_parameters transformer_nodes --output NEW-lattice.json
# Higher-precision diagnosis and packing observation (later harness commits):
python -m scripts.diagnose_precision --config configs/precision-timed-x.json \
  --device npu:0 --output NEW-precision.json
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m benchmarks.profile \
  --config configs/fc-k.json --device cpu --threads 4 --mode inference --output NEW-packing.json
```

The strict finite and resumed-optimizer deviations, larger-scale capacity/performance, exclusive
hardware measurement, CUDA/multi-device and candidate-specific kernel tuning
remain unqualified. Optional native implementations need demonstrated benefit.
Current [ROADMAP](../ROADMAP.md) owns follow-up priorities. Old teaching bindings,
LH-DAG/PDG, arbitrary graph engines, FP16/BF16 and retired backend matrices are
outside this delivery. No whole-combination or billion-parameter claim is made.

Earlier failures remain recorded: the first NPU job failed admission before
compute; `8fb66221` then produced 6 Attention runtime errors/4 passes because
CANN forbids a nondefault PSE mode when PSE is null. `d7437c21` fixed the mode;
current native Attention checks pass. Early device-local random-input performance
and diagnostics remain exploratory and are not used as paired speedup evidence.
