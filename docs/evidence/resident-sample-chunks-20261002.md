# Resident consumer physical sample slicing

Qualified source: `75543a7fd75bccbbbf26305f3f1c55725ca28048`.
[Audited receipts](resident-sample-chunks-20261002.json),
[consumer contract](../online-consumers.md).

The Python and standalone C++ resident consumers accept `--sample-chunk-rows`.
One live graph parameter/optimizer owner switches independent device continuations.
All connected windows of a physical slice are differentiated together; gradients
accumulate before one logical-batch graph update and one embedding/head update.
Global sample coordinates and the full logical-batch loss denominator are retained.
The uneven tail submits only real samples; unused owner capacity has no input events.
Dense saved continuations and simultaneously live accumulators are included in
capacity admission. The explicit maximum is not an automatic sample-size search.

Five immutable jobs passed: consumer build, CPU and NPU gates, separate profile
and two fresh-process memory observations. The consumer was freshly linked against
public installed headers/library; exact-source checked objects were reused. Core,
resident library and CANN kernels are unchanged from their previously qualified
identities. Standalone loader closure excludes Python-owned libraries.

- CPU: 15 affected checks passed; NPU: 45 passed, no skips. The cases cover all
  three families, both schedules, Add/Attention, FP32/FP16, SGD/AdamW, one/two
  devices, multiple updates and connected windows, nonaligned arrivals, clear
  policies, uneven B5/chunk2 tails and continued warmup. Independent CPU schedules
  check complete observable windows, all parameter gradients including None and
  each parameter generation; the same-dtype unsliced FP16 path is also independent.
- FP32 retains elementwise atol1e-6/rtol1e-5. Cross-dtype FP16 uses tensor maximum
  absolute error <= .002 + .02 times reference infinity-norm; discrete fields and
  None remain exact. Separately executed whole FP16 retains the original
  elementwise atol.002/rtol.02 comparison against sliced FP16.
- Eight initial FP16 near-zero threshold failures are retained. A separate
  diagnostic reproduced them in unsliced FP16, while sliced-versus-whole FP16
  passed the original thresholds. The precision-policy correction changed no
  production code. An earlier run with 23 skips from a missing device selector
  is excluded from qualification.

The fixed D128/B8/T4/V257 Attention packet executed one FP32 AdamW update over two
connected windows per process. Whole batch and chunk2 both produced 64 outputs,
3,145 events, the same cut and loss 5.6127676964. Maximum observed allocated memory
fell from 2,409,101,824 bytes (2.244 GiB) to 1,975,361,024 bytes (1.840 GiB), a
18.0% reduction. Both observed peaks fit their admission estimates. This is one
memory calibration per configuration, not a throughput recommendation. Splitting
increased stages from 26 to 104 and Full chunks from 36 to 88 in this observation.

A separate FP32 two-device B5/chunk2 Attention training trace observed 46,281
Vector Core, 1,797 AI Core and 586 MIX_AIV tasks, zero observed AiCPU tasks and no
CPU-fallback warning. It includes construction and diagnostics, not timed throughput.

Persistent and saved KV remain dense. Compact cache storage, automatic sample
admission, original wide execution and full-size performance are still pending.
This qualification makes no new CUDA or other-CANN-version claim.
