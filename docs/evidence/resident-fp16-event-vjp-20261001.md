# FP16 event cache and projection adjoints

Implementation: **4f195d22961d66495b30ecc447ececf684eac969**.
[Contract](../resident-event-vjp.md), [machine audit](resident-fp16-event-vjp-20261001.json).
All six fixed-clean-source jobs passed/exit0 on the public LibTorch-NPU2.10.0 /
CANN9.0.0 stack, Ascend910_9392, using logical `npu:0` after exclusive selection.

The component consumes actual journals from its own ContentFlow forward.
QKV and attention reproduce half forward rounding. Cache roots, carry, content
and parameter adjoints remain FP32. Independent CPU FP32/FP64 quantized-forward
autograd is only an oracle, never a candidate data source.

| Check | Result |
| --- | --- |
| Event VJP FP32 and FP16 | Each51 cases/102 replays;2 samples,3 events,large int64 times, GQA,widths1/4/7/257,adopt/selection/clear,window1/3,streaming/greedy,1/2-owner reverse batches |
| Cache boundary per dtype |6 cases/12 replays/12 refusals; FP32 sums beyond half range,None/connected-zero/empty roots,poison padding,reset,length/budget/root rejection |
| FP32 event/fiber complete-training regression |66/172 roots,8/20 optimizer trajectories; original CPU FP32/FP64 thresholds retained |
| Python-owned client regression |61 passed,zero skips: precision,event-training,fiber-training |
| Separate FP16 profile |52140 AI_VECTOR_CORE,2088 AI_CORE,450 MIX_AIV tasks; no observed AiCPU or logged CPU fallback |

Floating tolerances for event VJP are half rtol2e-3/atol2e-5 and unchanged
FP32 rtol1e-5/atol1e-6. Cache boundary sums and connection/padding checks are
exact. None and connected-zero remain distinct even for empty caches.

Standalone checker rebuilt; source-byte-matched terminal event host/kernel
objects were reused. The Python-owned host object was rebuilt and its client
relinked against matching CANN archives. Audit checks the actual immutable
fixture archive link, source/core/archive/object/binary hashes, loader closures,
raw case logs and profile CSV hashes. This is not a from-scratch vendor rebuild.
The unchanged portable core's8,954 CPU checks were not repeated.

Runs under `TASK=/mi/data2T/zlong/tide-execution-flows`:
`build-low-precision-event-vjp-clean01`, `build-low-precision-event-vjp-python-clean01`,
`low-precision-event-vjp-{components,regression,python,profile}-clean01`.
Physical devices for these four runtime jobs were1,3,9,13 respectively.
`RUN/status.json`, `task.log`, `gate/result.json` or `profile/result.json`
own original lifecycle/results. Audit command:

```bash
python "$TASK/launchers/precision_event_vjp_evidence.py" 4f195d22961d66495b30ecc447ececf684eac969
```

Preserved test-scaffold failures: build dev02 rejected ambiguous empty Tensor
assignment; gate dev03 rejected LibTorch vector-to-bool construction. Explicit
Tensor{} and int64-then-bool conversion fixed them; candidate math/tolerances
were unchanged. Dev01 event/regression and dev04 expanded cache gates passed.
An audit attempted while the regression was still running refused that run;
it wrote no pass evidence. The terminal audit subsequently passed.

This qualifies local event/cache/projection components and affected FP32 clients.
It does not qualify complete FP16 graph or retained-window training, public
FP16 training/checkpoints, fiber integration, peers or throughput. Profiled
correctness cases are not performance samples. No full-size speed ratio changed.
