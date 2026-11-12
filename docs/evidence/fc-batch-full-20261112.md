# Direct FC Full across batch and sequence

Implementation `cc6ccbf7948362686ba27166aedf48ce17af1841`, compared with
`bb23a9fae7aa3b02e59344d9f798f89f4e551214`. Mathematics remains
`0770593838a37c790a568226930567ed60f99b83`. This changes the independent direct
FC anchor, not the default graph executor. [Machine record](fc-batch-full-20261112.json)
retains all 20 measured points, commands, configuration, numerical differences,
qualification cases, source identities and terminal job records.

## Change and independent checks

[`fc_tensor_baseline.full`](../../scripts/fc_tensor_baseline.py) previously
dispatched selected queries separately for each sample. It now collects selected
`(sample,time)` rows, projects Q once, packs `[B,H,Qmax,Dh]`, and contracts with
one `[B,H,K,Dh]` KV bank per sample. Output projection and FFN consume only actual
selected rows. One `index_copy` publishes `[B,L,d]`; no Python B/L loop remains
in Full. Attention alone pads query counts to the batch maximum. It never copies
the entire history once per query. All-inactive output remains disconnected.

The padding has finite softmax rows because this checked FC domain has a positive
write at every position, including its first position. This is not a general
empty-history algorithm. Dynamic `nonzero`/maximum-count extraction, per-node
dispatch and the remaining source/Read/region work are still present.

The authority is [KV §4 equations (7)–(8)](../upstream/tide-core-5/local-semantics/weighted-message/kv-memory-learning-note.md).
[`test_fc_full_batch`](../../tests/test_fc_full_batch.py) independently evaluates
the weighted exponential quotient per query/head, not the candidate's packing
or log-weight softmax. Uneven, single, all-active and all-empty supports cover
manual/SDPA outputs and current input, K/V/write-weight and five projection VJPs,
including None gradients. CPU FP64/FP32 and NPU FP32 use the existing tolerances.

Fixed-source CPU qualification: **31 passed**. This includes independent scalar
FC outputs/controls/raw state, nonempty left state, sharing, retained cross-window
gradients, three SGD/momentum steps and fresh-process restoration. Fixed-source
NPU qualification: **9 passed**, for selected Full and small scalar FC output/state
formulas. It does not certify NPU multi-step restoration or all public intermediate
roots. The direct anchor's output/state projection contract is unchanged.

## Configuration and timing

FC16 is four regions of four nodes, 48 physical edges, B8/L128 (1024 actual input
positions), d128, two heads, FFN512, KV qry/Read+/cat/ind/context/cat,
regional_total/quad, k=2, allocation temperature .25. The model has **5,092,788**
trainable parameters including the task head/embedding; its core has 4,564,404.
Transformer16 has **4,722,688** parameters and 16 serial pre-RMS/causal SDPA/SwiGLU
blocks at the same dimensions. It is a different mathematical function, not an
equivalence oracle or parameter-matched model.

Both are pure PyTorch, host scheduled, FP32. CPU uses one ATen thread; NPU uses
two host ATen threads, with tensors on physical NPU0 mapped to logical npu:0.
This is not LibTorch native or device-resident scheduling. Each point uses a new
process, one warmup, reset initial parameters/optimizer, and one measured interval.
`--phase-fences step` synchronizes the complete interval. Training includes input
preparation, packing, forward/loss, backward, finite/clip checks, AdamW, cleanup
and synchronization; construction and warmup are excluded.

Inference uses the last-position head and returns the first token. Grad-forward
and training use the full-sequence head/loss, so their difference from inference
does not isolate autograd construction cost. Throughput is **1024 / elapsed
seconds**, not autoregressive generated tokens/s. Each cell below is
**seconds / input positions per second**.

| Device | Mode | Before direct FC | After direct FC | Speedup | Transformer16 |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU | inference | .414144 / 2472.6 | .291181 / 3516.7 | 1.42× | .167318 / 6120.1 |
| CPU | grad-forward | .479715 / 2134.6 | .342995 / 2985.5 | 1.40× | .195280 / 5243.8 |
| CPU | training | 1.364575 / 750.4 | 1.044402 / 980.5 | 1.31× | .746452 / 1371.8 |
| NPU | inference | .242287 / 4226.4 | .114741 / 8924.4 | 2.11× | .009275 / 110399.9 |
| NPU | grad-forward | .280904 / 3645.4 | .125617 / 8151.8 | 2.24× | .010451 / 97978.7 |
| NPU | training | .673608 / 1520.2 | .291277 / 3515.6 | 2.31× | .042086 / 24331.0 |

Replacing only new Full's manual Attention with 4D SDPA gave CPU training
1.043579s / 981.2 positions/s and NPU .303310s / 3376.1. This single observation
does not justify changing the manual default. It neither replaces the two Read
attentions nor proves which backend kernel was selected by the weighted mask.

Measured batch first-token latency, before → after → Transformer: CPU
414.058 → 291.082 → 167.302ms; NPU 242.071 → 114.529 → 9.213ms. It includes local
input creation/upload, prefill, last-position head, selection, finite check and
host return, excluding queueing/tokenization and subsequent decode.

## Remaining difference and static interpretation

New direct FC training is 1.40× the Transformer interval on CPU and 6.92× on NPU;
the NPU difference is 249.191ms. No measurement here assigns that whole difference
to necessary Tide mathematics. The static call paths differ as follows:

| Boundary | Direct FC | Transformer16 |
| --- | --- | --- |
| Outer order | 4 regions, 4 nodes per region; nodes still separately dispatched | 16 serial blocks |
| Source/Aggregate | 52 physical input/edge incidences; per-incidence transform when enabled, then weighted mean; each tensor spans B×L | no corresponding multi-source operation |
| Prepare/Read/Describe | per-node KV write, query/key/value transforms, short and long weighted Read, score and evidence | QKV projection followed by one attention per block |
| Region | 4 batched context/control heads and exact quad interval solves, each spanning B×L | none |
| Full | selected Q/output/FFN rows; B×H×Qmax×K attention and index publication | dense B×H×L×L attention and dense output/FFN |

The old Full path had at most 16×B sample groups; the new path has at most
16 node groups. These are source-level dispatch bounds, not observed kernel
counts. The numerical sample/sequence work still grows with B/L. Padding,
source transforms, node grouping, checks/host synchronization and optimizer
parameter granularity remain implementation choices to investigate. Distinct
Read/Full semantics and regional mathematics remain obligations, but do not
require a separate Tensor or kernel for every equation. This task does not
establish a lower bound on their implementation cost.

All recorded per-phase intervals are host calling-thread/enqueue times; NPU work
may cross them. They cannot be presented as device kernel shares. The present
bounded conclusion is that the avoidable B dispatch had material cost, and that
the remaining Transformer gap still requires explanation before the direct
anchor can be called an efficient target for the foundation.

## Numerical and resource scope

Before/after FC initial parameter hashes agree, as do task-archive identities.
The task is read from `tide-learning-lab@0439efd2af31cf10b07ce9e6423b0195f7449968`
Git objects; its working tree is untouched. Inputs use the same fixed recipe and
seeds; the old runner did not hash input tensor bytes. Work agrees within each
device: CPU candidates/selected/edges = 16384/8344/25060; NPU = 16384/8345/25064.
Thus about 51% of candidate events run Full despite the dense FC topology; this
is not the future 3% sparse case. Cross-device routes are not claimed identical.

CPU training loss is 10.9906949997 before/after; gradient norms are
1362.9216309 / 1362.9218750. NPU loss is 10.9906730652 / 10.9906768799;
gradient norms are 1360.0449219 / 1360.3856201. These are finite observations,
not a full tensor-gradient comparison. Full-scale `strict_numerical=not_checked`
is retained. Small independent tests do not upgrade that field or erase older
strict numerical failures. No tolerance was changed.

Two isolated `background.slice` services used disjoint host affinity 0–7 and
8–15, each 128GiB/1800s, with 240s per-point limits. The NPU queue admitted idle
physical0 after two observations; external work used other cards/shared host.
This is not exclusive peak or thread-tuned CPU performance. Environment:
CANN9.2.0, Torch2.12.0+cpu, TorchNPU2.12.0.post3. Source worktrees remained clean.
Both services ended inactive, MainPID0, empty cgroup, exit0; all 20 inner cases
completed. Disk/shm capacity did not block or serialize the work.

## Reproduce without the old runs

Create clean worktrees at the two commits and use the project environment from
[development](../development.md). No native build is required. Keep an unchanged
clone containing the fixed downstream Git object. Run the following workload in
a bounded service; substitute checkout/output paths and an admitted device:

```sh
python NEW/scripts/benchmark_fc_batch_pair.py \
  --before OLD --after NEW --downstream-repo DOWNSTREAM \
  --device cpu --threads 1 --sdpa-training --output-dir NEW_OUTPUT
```

For NPU use `--device npu:0 --threads 2` after site allocation/activation. From
the new source, CPU checks use `PYTHONPATH=python:scripts python -m pytest -q
tests/test_fc_full_batch.py tests/test_fc_tensor_baseline.py
tests/test_tensor_baseline_continuation.py -k 'fc or finite_interval' --dtype both`.
NPU sets `TIDE_FC_DEVICE=npu:0`, selects the first two files,
`-k 'selected_batch or formula_outputs' --dtype float32`.
The machine record contains exact commands and the executed workload script.
