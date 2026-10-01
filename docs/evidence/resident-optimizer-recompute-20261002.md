# Device optimizer without full numerical proposal banks

Source `81f4736b3ed3afd3ed3cf2a7b56a243d3f10b3f2`, 2026-10-02.
All eight immutable-source jobs passed; [audit record](resident-optimizer-recompute-20261002.json).
Both runtime backends reuse recursively source-verified affected objects and two
verified Ascend C kernels, with fresh links and loader checks. The installed
standalone consumer is relinked against the matching backend. This is not a
from-scratch rebuild of the unchanged core or every kernel.

The first device pass evaluates candidate updates and records finite flags.
After the existing all-owner consensus, the same numerical kernel recomputes
and writes masters/slots. A separate following kernel commits int64 counters and
bias corrections, preventing a race with SGD first-use momentum. Gradients,
connection bits and old state stay frozen between the two numerical passes.
There are no full proposed master/momentum/moment/max banks. Ordinary AdamW
removes12 bytes per active parameter; SGD removes its corresponding proposal
banks. Numerical refusal still leaves every card's live state unchanged.

Qualification covers:

- Local SGD/AdamW:32 trajectories per payload,256 FP32 updates and248 FP16
  updates. The two expected FP16 representability refusals remain; comparison
  uses independent CPU FP32/FP64, with unchanged tolerances.
- Two-device canonical optimizer:4 trajectories/32 updates per payload,
  shared aliases, None/zero, empty partitions, poisoned and nonfinite sources,
  global finite/half-overflow refusal and replay.
- Public training:32 trajectories,512 windows,128 updates, both schedules,
  FP32/FP16, SGD/AdamW, explicit Full/state maps and2→3 device restoration.
- Actual native Python client and standalone LibTorch consumers:24 complete
  training trajectories across three families and both payloads, plus4 bounded
  head-splitting cases. All28 passed without skips.

Four1,048,579-element allocator calibrations also compare real updates and slots
against CPU and inject infinity into the final tile. Byte comparisons confirm
that this late refusal changes no master, slot or counter. The measured deltas
are approximately4.26MiB(SGD),8.26MiB(momentum),12.26MiB(AdamW) and16.26MiB(AMSGrad),
within one live state set plus2MiB. Input gradients pre-exist in the baseline;
untracked vendor/driver allocations are outside this allocator scope.

An actual128-node/544-edge Attention consumer uses D128/B2/T2/V257 and17,384,240
parameters, two devices, online prefill and AdamW. It runs four continued windows
and two updates, including warmup. Its loss, output counts and final cut match
both the previous fixed-source NPU implementation and independent CPU execution.

| Peak allocated phase | Previous npu:0 / npu:1 | New npu:0 / npu:1 |
| --- | ---: | ---: |
| Construction | 386.91 / 346.90 MiB | 284.91 / 244.90 MiB |
| Warmup training | 998.77 / 908.40 MiB | 896.77 / 806.40 MiB |
| Measured training | 997.22 / 907.37 MiB | 895.22 / 805.37 MiB |

Every noninitial phase drops106,955,264 bytes per device. These single samples
establish memory behavior, not a throughput recommendation or full-size result.

The separate actual D32 FP16 two-device training profile observes18,453
AI_VECTOR_CORE,704 AI_CORE and258 MIX_AIV operators, with no observed AiCPU.
Each card records four numerical optimizer passes and two counter commits for
two updates. The trace includes construction and is not throughput timing.

Retained development failure: the new large test initially supplied an empty
named optimizer group, which correctly selected no parameters. Naming its
`weight` owner fixed the test; production code was unchanged. No tolerance was
relaxed or fallback added.

Artifacts use source `optimizer-recompute-clean01`, standalone/Python builds
`optimizer-recompute-clean01` / `optimizer-recompute-python-clean01`, and installed
client `optimizer-recompute-consumer-clean01`. Three build jobs and five
`optimizer-recompute-{component,session,consumer,calibration,profile}-clean01`
jobs are audited. The CPU/prior NPU calibration uses source `fdfc748` and the same
hashed input packet. Total per-device admission, safe scale configuration and
the F6 performance matrix remain pending.
