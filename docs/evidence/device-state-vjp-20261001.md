# Device state-chain VJP qualification

Source **3e2d54d728e82bb90bb1d4c7704d6a0e7599cd77**, clean standalone NPU build;
core source d412541 (unchanged source and binary fingerprints checked).
AArch64, Ascend910_9392, CANN 9.0.0, LibTorch/torch_npu 2.10.0.
See [audited identities and terminal records](device-state-vjp-20261001.json).

All **34 component cells** passed, with four build CTests. The state-chain check
passed **216 independent CPU FP32/FP64 autograd cases** and **four actual NPU
forward tapes** (streaming/greedy, EMA/Add). Cases include None/connected-zero,
poisoned absent/padding values, identity/adopt/clear, periodic int64 clocks above
2^55, retention zero/one/negative/fractional, literal replay chunks 2/32, empty
reuse, malformed metadata and tensor/tick-work refusal. No tolerance changed.

The independent profile contains **7,743 AI_VECTOR_CORE tasks**, including 438
state VJP and 438 metadata-plan tasks. No AiCPU or host-fallback diagnostic.
This includes setup, forward construction and assertions; it is placement
evidence, not throughput or a whole-model engine distribution.

The [contract](../resident-state-vjp.md) covers a first-order local state-chain
component. Parameter sample partials still require owner/alias reduction.
Full/Aggregate/message dependencies, complete graph reverse progression,
HST/SOFTP, other modules, retained windows, optimizer and peer reverse remain.
Public resident sessions are still inference-only. Portable CPU/Python code is
unchanged; the existing 8,954-test CPU gate at 622dbb2 was not repeated.

Earlier state-vjp-dev02 cache-line/connection failure and the 0459195 clean gate
and profile failures remain recorded. The latter and Add-dev01 failed because a
diagnostic edit made a test demand zero for an already matching valid gradient;
this source corrects that assertion branch, preserving tolerances and None checks.
