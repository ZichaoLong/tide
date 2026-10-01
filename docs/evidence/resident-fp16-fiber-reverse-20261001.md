# FP16 actual fiber cache/source reverse

Implementation: **2d7cee1ab9838dcaeacf639781053a53a822c308**.
[Contract](../resident-fiber-vjp.md), [audit](resident-fp16-fiber-reverse-20261001.json).
All six fixed-clean-source jobs passed/exit0 on LibTorch-NPU2.10.0/CANN9.0.0,
Ascend910_9392. The four runtime jobs used physical9/1/13/11, each mapped to
logical `npu:0`. Independent CPU FP32/FP64 quantized-forward autograd receives
only common inputs and parameters, never a candidate route or gradient.

| Gate | Result |
| --- | --- |
| Actual fiber reverse, each dtype |90 cases/180 replays:five pools,seven root modes,streaming/greedy,adopt/clear,width1/4/257,multihead,large int64 time,permuted slots,missing/zero-scale sources |
| Cache-bias bridge, each dtype |3 cases/6 replays/6 refusals:None/zero,padding,capacity257,FP32 sums beyond half range |
| Reverse links, each dtype |64 actual feedback/parallel-edge/empty/policy/continuation windows |
| FP32 event/fiber training |66/172 root cases,8/20 optimizer trajectories |
| Python-owned client |61 passed,zero skips:precision,event-training,fiber-training |
| Separate half profile |162004 AI_VECTOR_CORE,11706 AI_CORE,1560 MIX_AIV tasks;no observed AiCPU or logged CPU fallback |

The candidate packs actual messages, rounds physical source products to half,
and reads half parameters/cache/bias. Roots, cache carry, source/parameter
adjoints and boundary sums stay FP32. Half roots are multiplied by256; comparison
uses rtol2e-3/atol2e-5, while FP32 retains1e-5/1e-6. Structural connections and
identities remain exact. Complete half graph reverse is still explicitly rejected.

Two development failures remain recorded:dev01 checker compilation used a
string where a const-char pointer was required;dev02 FP32 training regression
found that an identity-only graph has no tanh weight bank. The corrected dtype
gate reads the always-present source scales. No tolerances changed. Dev03
build,component gate and FP32 training regression passed.

Clean standalone qualification recompiled checkers and reused byte-matched
terminal production archives; Python qualification rebuilt six host objects.
The audit reauthenticates the completed production objects recovered from the
checker-failed dev01 without relabeling that failed job, along with source,
archive members,kernels,loader closure,logs and profile CSVs. This is not a
from-scratch vendor rebuild. The unchanged portable core was not retested.

Artifacts: `TASK=/mi/data2T/zlong/tide-execution-flows`;
`build-low-precision-fiber-reverse-{clean01,python-clean01}` and
`low-precision-fiber-reverse-{components,regression,python,profile}-clean01`.

```bash
python "$TASK/launchers/precision_fiber_reverse_evidence.py" 2d7cee1ab9838dcaeacf639781053a53a822c308
```

This qualifies the fiber reverse integration and affected FP32 clients.
Complete half graph/retained-window/public training,checkpoints,peers and
throughput remain separate. No full-size speed ratio changed.
