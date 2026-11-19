# ROADMAP HB placement, transport and capacity evidence

The current two-NPU host-staging path completed the recorded FP32/BF16 HB
checks, including continuing training and an actual update after checkpoint
restore. Finite numerical failures remain. Native direct transport failed in
full-graph backward; the first-order synchronized candidate did not complete
Timed qualification. [Machine-readable evidence](roadmap-placement-20261118.json)
preserves source identities, result/log hashes, commands, admissions, original
thresholds and full v8 numerical failures. This is current-cycle evidence, not
an investigation of retired implementations.

Fixed mathematics: `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`.
The API and ownership limits are in [HB placement](../placement.md). This is one
process with contiguous HB layer ownership, private state on its layer's device,
and differentiable predecessor-bank transport. It does not shard within a layer
or establish peer DMA, replica all-reduce, a resident scheduler or CUDA hardware
qualification. All NPU jobs below used the cooperative whole-group exclusive
queue; external activity was not controlled. They are correctness and failure
diagnostics, not exclusive-hardware performance measurements.

## Original transport failures and bounded candidates

| Artifact under `artifacts/` | Fixed source | Outcome and first decisive observation |
|---|---|---|
| `placement-npu-v4` | `bdcb81de615e9aa4b21e7443871d33149b4f1553` | FP32 Exchange backward failed: `rtStreamWaitEvent`, `EE1017`, invalid event handle. The subsequent Timed finite-weight error occurred in the already-failed process. |
| `placement-copy-probes-v6` | `7bcb0df17f0613793d82511d9517e37e1870850a` | Direct, host_stage and synchronized each completed eight finite FP32 updates of an isolated two-GEMM/transport graph. |
| `placement-transport-diagnosis-v6` | `7bcb0df17f0613793d82511d9517e37e1870850a` | Separate candidate processes: direct failed at `update.0.placed.backward`; host_stage completed both cases within thresholds; synchronized completed Exchange with finite deviation, then failed Timed `initial_forward`. |

These three jobs used physical NPUs 1 and 9 as logical `npu:0/1`. The v6 full
graph harness had mandatory phase synchronization, so its direct failure persisted
with forward/backward fences. The isolated probe's success establishes only that
small graph, not complete HB transport, pending state or restored training.

The synchronized Timed failure occurred while flushing, copying a pending tensor
through `_SynchronizedCopy.forward`. The first exception reports ACL error 500001
and `EC0010`: `No module named 'tbe'` during lazy compiler initialization. This is a
recorded runtime/environment failure, not evidence that the mathematical operation
is invalid. The public environment was left unchanged. Subsequent local launch inspection
identified that `scripts/launch.py` replaced the module-provided `PYTHONPATH` with
only the frozen `source/src`, omitting the CANN runtime/TBE directories. This
launch-environment defect can be repaired while preserving the public installation;
the separate recheck below records its new-source outcomes. It does not relabel
the original failure or establish the synchronized candidate beyond those checks.
After a device runtime failure, subsequent tensors in that process cannot provide
an independent diagnosis; the later harness stops before the next configuration.

## Host staging without extra phase fences

`placement-host-stage-npu-v8` fixes source
`20062c170e85ef0c5fd91130b6df724f90622913`. It admitted physical NPUs 6 and 8,
logical `npu:0/1`, four CPU threads on CPUs 112–115. Torch was `2.12.0+cpu`,
TorchNPU used the local 2.12/CANN 9.2 workspace, and HF32 was disabled. The run
uses explicit `host_stage`, `phase_sync=false`, B2/L5, d64, seed 807 and three
continued AdamW updates. Transport itself synchronizes the necessary
accelerator→CPU→accelerator copies and retains ordinary autograd.

Exchange-X uses `incoming_light_64`, `layer_shared`, four layers partitioned
`0,0,1,1` and 1,011,083 parameters. Timed Multiscale-X uses `source_residual`,
`region_shared`, `compact-5-dd4`, five layers partitioned `0,0,0,1,1` and
1,150,527 parameters. Each has an ordinary one-device reference created from the
same CPU parameters and input. Parameter/input hashes are retained. BF16 means
actual BF16 projection arithmetic with FP32 parameters, gradients, masters and
AdamW moments; sensitive controls/SSD state remain FP32 and times int64.

All four rows have exact initial whole results and, where applicable, exact
flush results. Initial input and parameter gradients satisfy the original
thresholds. The same split schedule (`2+3`) has exact single-device/placed
readout and state. All retained comparisons are finite and have zero logical
metadata or semantic-support mismatches.

| Precision / case | Initial parameter-gradient max abs | Whole versus split readout max abs | Three-update parameter max abs | Original overall outcome |
|---|---:|---:|---:|---|
| FP32 Exchange | 4.6566e-10 | 1.9073e-6, within | 3.4749507904e-5, exceeds | finite_deviation |
| FP32 Timed | 2.4214e-8 | 1.9670e-6, within | 6.6608190536e-6, within | within_threshold |
| BF16 Exchange | 9.3132e-10 | 0.04505980015, exceeds | 5.8207660913e-11, within | finite_deviation |
| BF16 Timed | 5.5879e-9 | 0.03686928749, exceeds | 3.0963867903e-4, exceeds | finite_deviation |

The comparison remains `atol=1e-6, rtol=2e-3`, with input-gradient `rtol=1e-2`;
the saved-data and checkpoint-next-result checks use zero tolerances. Max absolute
error alone does not decide a relative-tolerance result. FP32 Exchange fails at
updates 2 and 3; BF16 Timed also fails at updates 2 and 3. No threshold was relaxed.

The two BF16 whole/split readout deviations occur identically in the ordinary
reference and placed model. They therefore are not specific to multi-device
transport. Their state differences are also retained: Exchange max `0.02380371094`,
Timed max `0.05607414246`; Timed flush max is `0.05133605003`. Same-schedule exact
initial values and close first gradients do not prove identical later routing or
training trajectories. The retained multi-update differences do not have a blanket
single-bias explanation. The separate [bounded numerical investigations](roadmap-numerics-followup-20261118.md)
provide their own source/shape-scoped witnesses, without replacing these failures.

Each update carries detached private state; Timed flush contributes to the loss
and advances the next segment's clock. Checkpoint parameters, private state and
optimizer slots are exact after reload, and the next forward result is exact for
all four rows. Both original and restored placed owners then perform a real
update. Updated state is exact, while parameters and optimizer slots remain within
the original tolerances. Across the four rows the maximum restored parameter
difference is `2.9103830457e-11`; it is not bitwise identical optimization.

The v8 allocator peaks include the ordinary reference, placed candidate, restored
model and diagnostic calls. Its raw phase times omit diagnostic-record gaps and
detach, so they are not used as complete-training performance figures. Source
`0992b467584c1f603314ed04d8efe8596a14f576` corrects that complete wall-time boundary,
including detach and intervening record I/O. Its separate capacity-train mode uses
one placed model, no large duplicate reference or checkpoint, and records timed
per-device peaks plus actual parameter/gradient/slot/state storage outside timing.

## Launcher repair and independent v18 recheck

Source `ca6199d427ea452492422bb7be21c03ce45aab5c` fixes only the local launch
environment: clear caller `PYTHONPATH`, activate the selected workspace, then
prepend frozen `source/src` while retaining the baseline CANN runtime/TBE paths.
The public installation is unchanged. Its `src/tide` Git tree is identical to
`0992b467` (`1ef1183be8b131a9bbfd806ce4d9240ef25ea502`). `environment.json`
records the actual executable, Python path and Tide/TBE module origins;
`launch.json` records the launcher SHA256. Existing 0992 jobs are not relabelled
or modified. Successful old-scope results keep their original environment identity.

`environment-cpu-v18` completed the actual generated service in 5.064 seconds:
Tide imports from its frozen source, and TBE imports from
`/opt/software/ascend-runtime/9.2.0-r1/python/tbe/__init__.py`. This is an import
check, without accelerator operations. The NPU recheck below is separate.

`placement-environment-npu-v18` admitted physical NPUs 0 and 10, logical
`npu:0/1`, NUMA2 and CPUs 80–83, four threads, a 24 GiB process-tree RSS bound
and 1800-second time bound. It ran three fresh processes, B2/L5, d64, three
continuing updates plus checkpoint and a real next update, with `phase_sync=false`.

| New-source candidate | Completed observations | First failure / remaining numerical outcome |
|---|---|---|
| auto / BF16 → host_stage | Exchange and Timed, including real resumed update | Both retain finite_deviation; original whole/split BF16 deviations remain, and Timed three-update max parameter difference is `3.0963867903e-4`. |
| synchronized / FP32 | Exchange initial values/gradients, three updates, exact checkpoint data and exact next result | `checkpoint.original_update.backward`: `EE1017`, `rtStreamWaitEvent` invalid event handle. No complete restored update; Timed not run after failure. |
| direct / FP32 | Initial Exchange checks and reference update | `update.0.placed.backward`: `EE1017`, `rtStreamWaitEvent` invalid event handle. Timed not run after failure. |

The TBE import failure did not recur. The synchronized candidate now has a
distinct event-lifetime failure after its three finite continuing updates; those
updates were within tolerance, with maximum parameter difference `5.8207660913e-11`.
Exact saved data and next forward do not establish the subsequent backward/update.
The two auto/BF16 rows retain exact initial/same-schedule and checkpoint data/next
results; actual resumed parameters remain within threshold, max `2.9103830457e-11`.
The outer v18 service is **failed** because two independent candidate processes
failed, not because the successful auto/BF16 process failed. The machine record
retains its terminal status, attempts, partial comparisons and first exceptions.

This bounded transport investigation ends here. Default auto remains host_stage.
Synchronized BF16 was not run, and the unrun Timed candidates are not qualified.
Neither the repaired environment nor isolated-copy success overrides any original
failure or establishes native peer-DMA performance.

## Default policy and exact static partition planning

Source `c172c606b8e3bc574910aa1f948b4b129fed2f35` makes `transport="auto"`
select `host_stage` for multiple distinct NPUs and `direct` for CPU/CUDA or a
repeated same NPU. Requested and resolved choices are recorded. Checkpoint loading
re-resolves auto for its target devices; older checkpoints retain their saved
explicit/resolved transport. The v8 evidence above tests explicit host_stage at its own source; the later
auto-policy qualification is recorded separately in the v18 recheck above.

The same source adds exact meta-only unique-parameter ownership planning. The
9B model has **8,778,358,224** parameters and layer node counts
`144,144,288,144,128`; the 17B model has **16,707,220,176** parameters and
`272,272,544,272,256`. Both have five layers and period 6. The following vectors
are per partition in GiB. FP32 master parameters stay FP32 under both arithmetic
policies. Dense AdamW planning includes parameters, gradients and two moments;
the separate foreach allowance is one additional parameter payload.

| Model / cards | Layer owners | Parameters GiB | Dense AdamW persistent GiB | Persistent + foreach allowance GiB |
|---|---|---|---|---|
| hb-9b / 2 | `0,0,0,1,1` | 22.210, 10.492 | 88.840, 41.968 | 111.050, 52.460 |
| hb-9b / 4 | `0,0,1,2,3` | 11.107, 11.103, 5.551, 4.941 | 44.428, 44.411, 22.206, 19.762 | 55.536, 55.514, 27.757, 24.703 |
| hb-9b / 5 | `0,1,2,3,4` | 5.556, 5.551, 11.103, 5.551, 4.941 | 22.223, 22.205, 44.411, 22.206, 19.762 | 27.779, 27.757, 55.514, 27.757, 24.703 |
| hb-17b / 2 | `0,0,0,1,1` | 41.901, 20.338 | 167.606, 81.351 | 209.507, 101.689 |
| hb-17b / 4 | `0,0,1,2,3` | 20.953, 20.949, 10.474, 9.863 | 83.811, 83.795, 41.897, 39.454 | 104.764, 104.743, 52.372, 49.317 |
| hb-17b / 5 | `0,1,2,3,4` | 10.479, 10.474, 20.949, 10.474, 9.863 | 41.914, 41.897, 83.795, 41.897, 39.454 | 52.393, 52.371, 104.743, 52.372, 49.317 |

These are exact parameter counts and declared storage budgets, not execution or
allocator measurements. State bounds separately account for BF16/FP32 KV storage,
FP32 SSD state, int64 timestamps and padded SSD convolution slots up to
`3*period+1`; assuming only four events would undercount the current representation.
The JSON retains both precisions' cold/steady bounds and remaining capacity against
a planning limit of 60 GiB per card. It excludes autograd/activations, prepared
banks, parameter gathering/casting, copy workspaces and allocator overhead.

At 9B/four cards, disabling foreach leaves approximately `15.449,15.466,37.733,40.179`
GiB after dense AdamW and steady BF16 state. The two busiest partitions each have
a largest Parameter shard of 2.25 GiB. This is room for a bounded capacity attempt,
not proof that backward/optimizer workspaces fit. Five cards leave the same
288-node layer as the bottleneck.

At 17B/two cards, parameter-plus-steady-BF16-state headroom is
`17.619,39.425` GiB for inference. At 17B/five cards, the 544-node layer alone needs
83.795 GiB of dense AdamW persistent payload, before activations. The current
contiguous layer split cannot divide it. CPU alternatives and the scale comparison
table belong to the scale workstream; this evidence does not introduce node sharding.

Directed CPU implementation checks passed 23 tests covering placement and capacity
planning, including independent counts of actually constructed shards for all
sharing profiles, both precision policies, Timed state retention and a 17B meta-only
construction guard. Six later capacity-train tests cover two actual updates,
detach/clock boundaries, actual storage, nonfinite-slot stopping and full timing.
The checked sources and commands are in the JSON. CPU checks do not confer NPU
qualification on unrun shapes.

## Large execution status and reproduction

The original `scale-two-npu-hb-17b-v12` and
`scale-four-npu-hb-9b-train-v14` were stopped before computation to reassign NUMA
resources. Their original stop/cleanup states and `superseded.json` remain in the
machine record; they are neither hardware failures nor capacity passes.

Replacement jobs `scale-two-npu-hb-17b-v16` and
`scale-four-npu-hb-9b-train-v16` both fix source
`0992b467584c1f603314ed04d8efe8596a14f576`. The former uses B1/L8 inference,
one warmup and three samples per precision; the latter uses B1/L8, two continued
updates per precision and `--no-foreach`, without a duplicate reference or large
checkpoint. Their current lifecycle, actual completed results and failures are
owned by the [large-scale report](roadmap-scale-large-20261118.md), along with
the one-NPU 9B A–D and CPU 17B workloads. No pending job is a capacity pass.

All five final large-workload jobs are now terminal: 20 completed Tide/A–D inference comparison rows and 5 completed capacity rows, with 0 execution/nonfinite failures in this group. The CPU17B BF16 capacity row includes two real updates and finite/storage scans. Detailed mode, precision, memory and unmeasured-checkpoint boundaries remain in that report.

At the current captured state, the 17B two-device BF16, FP32 inference rows completed with recorded `auto`→`host_stage`. This is actual large-shape inference evidence; it does not extend the small-model gradient/restore comparisons to 17B.

The 9B four-device BF16, FP32 capacity rows completed two real continued updates each. Their actual gradients/moments, state advancement, finite scans and full wall times are in the large-scale report; no large reference or checkpoint qualification was performed.

Use the recorded frozen source and a fresh output identity. Hardware commands run
only through the bounded detached/admitted [execution flow](../execution-flows.md):

```sh
python -m scripts.check_placement --devices npu:0 npu:1 --precision bf16 \
  --transport host_stage --output NEW-placement-bf16.json
python -m scripts.probe_placement_copy --devices npu:0 npu:1 \
  --transport direct --steps 8 --output NEW-direct-probe.json
python -m scripts.check_placement --devices npu:0 npu:1 --mode inference \
  --config configs/scale/hb-17b.json --precision bf16 --batch 1 --length 8 \
  --warmup 1 --steps 3 --output NEW-17b-inference.json
python -m scripts.check_placement --devices npu:0 npu:1 npu:2 npu:3 \
  --mode train --no-foreach --config configs/scale/hb-9b.json --precision bf16 \
  --batch 1 --length 8 --steps 2 --output NEW-9b-train.json
```

Static planning uses `benchmarks.capacity.placement_budget` with the chosen
configuration, precision, mode and `partitions=2`, `4` or `5`; it allocates only
meta tensors. These entry points do not restart experiments merely to view the
retained evidence. Current handoff and subsequent scale outcomes belong to
[STATUS](../STATUS.md).
