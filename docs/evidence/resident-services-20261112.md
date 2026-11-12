# Resident bounded services: fixed-source qualification and original-budget scale

Production source **042ea369d9e3cac63d0fb0ccec87c4d872ae3c63** completes the previously failing
KV flow and the original FC16/LH176 budget witnesses. Small independent checks pass; the two
scale runs establish execution and exact recovery, **not independent mathematical equivalence**.
The [machine record](resident-services-20261112.json) embeds configurations, generator identity,
commands, build manifests, terminal states and key outputs. Raw records were not changed.

## What changed and which source was tested

41c42759 replaced the bank-by-capacity branch tree by one fixed numerical tile body driven by
actual device bank/tile cursors. Padding has the bound `P < A + G*T`; the old less-than-two-times
claim does not apply. It passed messages, Read/selection and KV sequence, but complete flow
still failed while recording its backward program with CANN 507002.

042ea369 places each regional forward/reverse numerical body in a bounded device-called
program. The coordinator still chooses and commits on device; reverse uses the actual forward
journal. Storage admission covers the union of coordinator and called programs, with shared
workspace. This removes the observed recording failure for the checked flow and scale cases;
it does not remove every device task or memory limit.

See [flow](../../tools/device_online/block_settle_flow.cpp),
[reverse](../../tools/device_online/block_settle_flow_vjp.cpp),
[services](../../tools/device_online/cann_task_services.cpp),
[banked GEMM](../../tools/device_online/weighted_bank_gemm.cpp) and the
[session contract](../resident-block-session.md). The mathematics stays at
`0770593838a37c790a568226930567ed60f99b83`.

Both resident builds use 042ea369; the unchanged core is reused from
`1f0e6e43a540a78f2fe810e97b7562b73481fb22`. The 317 reused installed Python files were checked
byte-for-byte against 042ea369 and the actual installation. **7fd1dd1008ae581d4dfcbd543b1c8b992b4541b4**
is a separate CPU test-harness fix, described below; it is not the source of these NPU binaries.

## Results and preserved failures

| Check | Result | Scope |
| --- | --- | --- |
| 042ea369 native and SDK resident builds | Both passed | Source/build/loader evidence; device evidence is separate |
| 042ea369 directed CPU checks | **29 passed, 9 failed** | Original test-order contamination retained |
| 7fd1dd10 directed CPU checks | **39 passed**, no skip/failure | Original 38 plus launch-context regression |
| Standalone SDK qualification | **21 passing processes** | Components, complete flow/session, public qry/sum and fresh-process restore; includes one reused-core check and static Model lowering |
| Python native installed consumer | **4 passing processes** | qry/sum, independent Python schedule, three AdamW steps and each fresh-process restore |
| Original-budget FC16/LH176 | Both completed, exact recovery passed | Single measured training window per configuration; scale mathematics `not_checked`, `throughput_qualified=false` |

Counts use different units and are not added together. The SDK service check includes repeated
and empty windows, three nested levels, sibling services and shared workspace. Complete flow
checks an independent scalar formula, ten roots, three Read snapshots, B2/L2, duplicate edges
and ports, shared parameters and exact int64. Session checks two-window BPTT, three AdamW
steps and raw checkpoint reconstruction. Public consumers exercise actual Model/source-factor
bindings and independent streaming schedules. Each sum/qry archive is also consumed by a
new process. The machine record preserves each process's own scope and output.

The initial SDK qualification job stopped after its **120-second build-prerequisite timeout**.
It never admitted an NPU or ran checks. A new job after build completion passed; the initial
job remains failed. Original 41c42759 projection and flow failures also remain failed. Its flow
trace ended after vendor task 2041, `aclnnInplaceZero`, then 507002 while recording raw-field
adjoints. This is an observed trace boundary, not a claimed universal device task ceiling.

### Projection oracle and CPU harness

For the original 193-row / three-bank / width-32 bias-gradient witness, the independent FP64
row formula uses the actual FP32 inputs. At unchanged `atol=1e-6`, `rtol=1e-5`:

| Comparison | Failing coordinates | Maximum absolute difference |
| --- | ---: | ---: |
| NPU candidate versus FP64 formula | 0 | 7.48318e-7 |
| Old sequential FP32 oracle versus FP64 formula | 1 | 1.81212e-6 |
| NPU candidate versus old FP32 oracle | 1 | 2.27243e-6 |

Thus the original disagreement was not silently tolerated or relabeled. The successor oracle
has independent accuracy evidence, including `math.fsum` and a cancellation counterexample in
[test_resident_weighted_bank_oracle.py](../../tests/test_resident_weighted_bank_oracle.py).
The new projection check also records 193-row forward/reverse storage of 126,144/106,093 bytes
within the same 131,072-byte component allowance.

The nine CPU failures arose because a generated multiblock adapter left thread-local
`worker/workers` at the exhausted loop state, contaminating subsequent raw-source probes in
the same process. 7fd1dd10 restores the caller's context on success and exception and checks
zero-grid behavior. Only the CPU generator branch changed; this was not a numerical tolerance
adjustment or a production NPU correction. Exact failing test names/messages remain in JSON.

## Original-budget scale costs

Both points use Python native, single NPU FP32, resident `settle-block-v2`, B4/L16,
`staggered-v1`, packed input `B×L×P`, outputs return mode, intraop 2 / interop 1 / worker 1.
The task is `1e-6 × Σ‖cG‖²`, with SGD lr .001, momentum .3 and weight decay .01.
Each has one measured complete training window; FC16 also has one warmup, LH176 has none.

| Configuration | Complete training cost | Actual candidates / Full / messages | Exact next-window restore |
| --- | ---: | ---: | --- |
| FC16: four regions × four nodes, 48 edges, d128, H2, FFN512, KV qry | **3.944763638 s** | 915 / 306 / 912 | Passed; 65,691,604-byte checkpoint |
| LH176: 22 regions × eight nodes, 1,058 edges, d64, H1, FFN256, 56 qry + 120 sum | **22.591799125 s** | 8,368 / 2,572 / 15,340 | Passed; 237,404,617-byte checkpoint |

FC has 4,564,404 trainable parameters. LH has 14,902,694; it is not a 9B test. FC uses ind region
scoring; LH opening/closing regions use att/content_stats and its eight middle regions use ind.
Both use read `+`, cat local score/evidence, context control, regional_total/quad and `k=1`.
`k` constrains sending strength, not a fixed count of selected nodes. Exact configurations and
original-time window/position settings are reproducible from the embedded fixed-source recipe.
This FC16 point is the earlier **B4/L16 resident refusal witness**, not the main B8/L128 FC16 task.

The original reservations were retained: forward 6,375,342,080 bytes, backward 11,744,051,200,
optimizer 402,653,184 and total **18,522,046,464 bytes**, with cache 49, one retained window and
workspace limit 536,870,912. Requested and effective limits match. The LH partition was borrowed
from the original finite witness; this is not a claim that its partition is optimal.
Measured allocator peaks were 1,517,363,712 bytes for FC and 8,276,641,792 for LH; allocator
figures exclude untracked vendor/driver memory and are not total reservation accounting.

The timer includes input preparation/transfer, scheduling, packing, forward, requested returns,
finite checks, loss, VJP, optimizer and synchronization. Construction (0.9272 / 3.4404 s), durable
checkpoint publication, independent validation and recovery checks are outside it. Recovery
checks are serial new-owner checks in the same process: decoded raw state, actual next-window
returns/loss, parameters, gradients, optimizer and task all agree exactly. Fresh-process coverage
comes from the small public consumers above, not from these two scale runs.

**Denominator correction:** raw `metrics.input_positions=64` reports rectangular B×L capacity.
The fixed input pattern actually has lengths `[16,15,14,16]`, hence **61 input positions**;
physical input atoms are 183 for FC and 366 for LH. The fixed generator and full resolved JSON
were reconstructed without executing a model. Raw JSON stays unchanged. Reviewed arithmetic
therefore gives **15.4635** and **2.7001 positions/s**, respectively, as **unqualified cost rates**.
Neither point performed independent scale mathematics: all validation fields are `not_checked`
and `throughput_qualified=false`. No qualified throughput, TTFT, inference rate, CPU/mixed
speedup or direct-formula parity is inferred.

## Environment and reproduction

The Python-owned runtime is `ascend/dev-workspace-9.2.0`, Python 3.12.13,
Torch 2.12.0+cpu and TorchNPU 2.12.0.post3. Standalone SDK uses
`libtorch-npu/2.12.0-cann9.2.0`; both builds are aarch64 ABI1, AscendC `Ascend910_9392`.
The scale processes report `TASK_QUEUE_ENABLE`, `PER_STREAM_QUEUE` and
`ASCEND_LAUNCH_BLOCKING` unset; this does not infer vendor defaults or qualify ordinary
multicard asynchronous behavior.

Builds used separate background.slice services, CPU masks 0–159 / 160–319, each reporting
160 build jobs with 512 GiB / 2,400-second limits. SDK/native qualification used CPU20–23 /
24–27, 256 GiB / 1,200 seconds, per-process 300 seconds. Scale used CPU24–27, 256 GiB /
1,500 seconds, 600 seconds per case. Cooperative NPU admission chose physical 1 for SDK and
scale, physical 11 for Python native. Scale started after these builds/checks ended. Host load
and allocator samples are retained; external device contention was not independently traced,
so these single measurements make no exclusive or peak-performance claim.

From clean fixed checkouts, follow [development](../development.md) to build the matching core
and resident runtimes. JSON `reproduction` contains the complete commands and original launch
scripts, and `device_qualification` contains each argv and key output. Replace historical
checkout/build/install/output paths with new paths. Public qry/sum commands are followed by
separate `--restore-check` processes. CPU checks use 7fd1dd10; NPU evidence uses 042ea369.

For scale, copy `scale[].configuration`; reconstruct LH's `model.definition` using its embedded
`model_definition_reproduction`, then replace only `site.paths`. Run:

```sh
python -m tidegraph run NEW_CONFIG.json --output-dir NEW_OUTPUT
```

The LH definition hash is `6de707d9304585c659b10a6fd185a756bee735a3a74c07a8c16f1a2cdcf2ee67`.
Both complete resolved JSON configurations were independently regenerated from the fixed source
with original site paths. Historical raw directories are identity references, not prerequisites.

This evidence closes the specified small-flow and original-budget witnesses. It does not certify
recurrent/half/multiple-device/CUDA blocks, every general PDG/TimedDAG program, large-model
throughput or all P1–P6 work. Current remaining scope stays in [ROADMAP](../ROADMAP.md).
