# Finite CPU batching assessment, 2026-10-10

Batched components were faster in the retained single observations, but the
complete standalone Add consumer showed no speed gain. The Attention baseline
was refused by memory admission before allocation; no Attention timing or
candidate result exists. This does not justify a full-size follow-up or change
the old CPU/NPU route ranking. [Reviewed raw identities and observations](batched-selection-cpu-20261010.json).

## Question, sources and budget

`g2-cpu-assessment01` declared10 serial fresh processes after the clean CPU and
backend build gates terminated. It ended failed/exit1 with an empty cgroup after
8 successful measurements and1 admission refusal; the tenth case never started.
No case was repeated, retried or given a larger timeout/budget. Trackio was off;
existing project records contain configurations, logs, results and raw traces.

The component source is clean ecafdf4, model/input seeds7/19, FP32,32 independent
rows,width128. Read uses norm-fp32-v1; State uses Add-repeat or Attention with an
8-row initial cache. Replay reproduces the old scalar gradient attachment on the
same packed numeric forward. Every process checks forward/VJP/None parity before
one warmup and one synchronized measured step, then a separate untimed CPU
profiler pass. These are **pure PyTorch** components, not Python calling native
and not independent LibTorch graph timings. No device kernels were observed on CPU.

Complete consumers use clean c0ce4ee baseline and a02c18c batched core, with exact
core/client/binary/input/loader identity checks. Both are **standalone LibTorch**;
Python only launches and records the executable. Packets use TimedDAG/prefill,
D128/B32/T4/V257,128 body nodes/544 edges,8,995,632 Add or17,384,240 Attention
parameters, SGD/Add or AdamW/Attention. One warmup and one measured optimizer
step each include two connected windows, input preparation, independent online
scheduling, head/loss, backward, finite checks, detach/update and synchronization.
There is no timed CPU oracle. Native workers4, ATen1, physical batch32,
explicit aggressive chunking,6GiB static admission; logical batch/Attention
visibility/gradient/optimizer boundaries are unchanged.

Budget120s/case,8GiB supervised process group,2GiB outputs;30min service with
8CPU affinity/quota,12GiB memory and192 tasks. External host load was uncontrolled
(1-minute load64.02–70.62 before cases); project heavy jobs were serial. These
single observations have no confidence interval or repeatability claim.

## Results

| Pure PyTorch component | Replay ms | Batched ms | Observed replay/batched ratio |
| --- | ---: | ---: | ---: |
| read | 3.3500 | 1.0607 | 3.16× |
| state-add | 4.9553 | 3.8671 | 1.28× |
| state-attention | 40.3159 | 13.5991 | 2.96× |

The separate CPU trace helps explain the component results without treating
operator count as a device bottleneck measurement:

- Read's recorded forward/attachment stage changed4.205→0.923ms and backward
  2.570→1.428ms. Finite checks and scalar extraction each changed64→1; this is
  host evidence for grouped validation, not a measured NPU synchronization gain.
- Add's packed numeric stage stayed near1.1ms. Attachment changed2.072→2.196ms,
  while backward changed4.157→3.180ms. Clones increased64→96 and selects96→288;
  container/view/structural assembly remains a concrete batching cost.
- Attention attachment changed17.012→6.756ms and backward45.606→10.229ms.
  Scalar matmul calls changed262→12, with batched matmul2→8. The measured stage
  durations support less host graph work in this fixture; they do not prove
  fusion or accelerated whole-graph performance.

Profiler durations are perturbed observations from a different pass and must not
be substituted for, added to or directly compared with unprofiled step timings.

| Standalone CPU LibTorch Add | Before G1 | Batched G1 |
| --- | ---: | ---: |
| Complete measured optimizer step | 3.810711s | 4.090873s |
| Process peak RSS after measured step | 701,222,912B | 814,317,568B |
| Loss | 16.050273895263672 | 16.050273895263672 |
| Output count / final cut | 682 /272 | 682 /272 |

The observed elapsed time increased7.35%; one observation under external load
cannot establish a stable regression size. Input hashes and all semantic work
counts match exactly, including17,202 candidate events,4,972 selected events,
21,148 source rows and20 greedy stages. The batched consumer replaced17,202
State and Read replay attachments with their batch paths; the timing difference
is not an unused-optimization comparison. Their observed RSS stayed below the
same static estimate; the candidate's peak was higher in this case.

Attention's explicit32-row physical chunk had a25,662,653,552B static peak estimate
against the6GiB admission request (5,663,988,122B usable). It was rejected before
model allocation, not timed out or OOM. This identifies a configuration/admission
limit, not Attention execution speed or incorrectness. The candidate was not run.
No budget expansion, automatic rechunking or repeat followed. A future distinct
Attention experiment must first select and calibrate an admitted physical chunk
while keeping logical B32 and all visibility/update semantics; no further run was included in this tranche. Any follow-up needs a separately
recorded finite budget; the old queue remains closed.

## Selection guidance and limits

Retain CPU, mixed-a/b/c and resident plus their fine controls. Batch Read and
State are reusable correct implementations with component evidence; do not sell
this patch as a universal CPU speedup. Preserve the independent reference and
custom/nondefault-policy replay. Add still exposes small-tensor/assembly costs;
NPU mixed and explicit resident device VJPs require their own real-device
profiles before selecting further kernel or packing changes.

This tranche gives no basis to rank new NPU mixed against resident, to extrapolate
components to full-size models, or to choose CUDA for speed. Current-source NPU
qualification/profiling is resource-blocked and every GPU execution claim is
pending a real target. The [old selection table](selection-review-20261009.md)
remains an old-source/configuration baseline, not a route's performance ceiling.
No additional full-size case is launched from these observations alone.

Raw root is `TASK` in STATUS. `runs/g2-cpu-assessment01` retains all8 accepted
results,9 case logs,6 traces and the admission witness; the audit is
`plans/audit-g2-cpu-assessment01.py`, output
`runs/g2-cpu-assessment-reviewed01.json`. Its review status is passed while the
experiment status remains failed. [Current correctness/build scope](batched-vjp-cuda-local-20261010.md)
and [execution contract](../execution-flows.md) apply independently of timing.
