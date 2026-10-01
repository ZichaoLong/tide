# FP16 consumer and bounded packed head qualification

Implementation `d178b863402157f0f92b60b7dbff12c8e8d7a60e`; immutable clean source.
[Machine-readable audit](resident-consumer-head-20261002.json).

The actual standalone LibTorch and Python-owned resident consumers now accept
FP16 payloads for inference and complete training. Head matmul preserves payload
rounding; loss,explicit adjoints and optimizer masters use FP32. Updates remain
staged across graph/head/embedding and preserve None versus connected-zero gradients.
Python is a client of the C++/CANN scheduler,not an independent PyTorch scheduler.
Eager FP16 consumer training remains explicitly unavailable.

The packed head selects physical output-row chunks within
`--head-workspace-bytes` (default4GiB). It accounts for root storage,head gradient
accumulation/partial,optional FP32 head copy,row scratch,32MiB calibrated operator
allowance and10% aggressive/25% conservative headroom. Only present outputs enter
arithmetic; each row retains the full vocabulary normalization. It does not split
a logical window,loss denominator or optimizer update. A single-row budget failure
is reported before model construction. Records expose precision,head reservations
and actual head chunk counts.

Six fixed-source jobs passed: installed standalone client build,10 independent CPU
autograd/optimizer cases,32 Python-owned and27 standalone/NPU checks,separate large
head calibration and separate two-device FP16 profiling. No skips or fallback
warnings were accepted. The59 NPU checks include48 whole-model trajectories:
three families,Add/Attention,both runtime owners,FP32/FP16,24 inference and24 training,
with both schedules and one/two-device placement. Four additional split-head
training trajectories retain strict discrete/None comparisons and unchanged update
boundaries. CLI checks retain failed graph/head budget receipts. FP32 tolerances
remain1e-6/1e-5; FP16 uses the already declared2e-3/2e-2 scope.

Four small NPU head cases calibrate inference/training at both dtypes. Larger
D2048,V50304 head VJPs use256 actual rows with the following results:

| Payload | Selected rows/chunk | Head budget | Allocated peak delta |
| --- | ---: | ---: | ---: |
| FP32 | 108 | 1536MiB | 915,439,104 bytes |
| FP16 | 151 | 2048MiB | 1,340,316,160 bytes |

The delta excludes pre-existing caller tensors. These are head-only allocator
measurements,not whole-model or driver HBM measurements. FP16 retains an additional
FP32 head copy for its declared FP32 VJP; these numbers do not imply a throughput
comparison or universally lower half-precision memory.

Independent D32 Attention/two-device FP16 profiling runs four continued windows
and two AdamW updates without diagnostics,with head chunk1 and12 head chunks per
update. It observes17,399 AI_VECTOR_CORE,667 AI_CORE,258 MIX_AIV tasks and no AiCPU.
Construction and instrumentation are included; no formal speedup is claimed.

The first head calibration failed: an8MiB tensor-only plan observed20,538,880
allocated bytes. A bounded row-count probe identified a16MiB operator floor,so the
planner now reserves32MiB before applying policy headroom. The failed profile
fixture is also retained: its consumer succeeded but an8-row budget did not force
splitting; the fixed-source trace explicitly selects1 row. Neither correction
changes model mathematics or relaxes numerical comparisons.

The unchanged backend/core artifacts remain qualified at acb84f3. Their component,
source and binary hashes were checked against this source; they were not rebuilt.
The standalone client uses verified source/header/compiler dependencies for object
reuse,then a fresh link and loader check. Total per-device memory admission,
representative preset screening and the full-size F6 matrix remain outstanding.
