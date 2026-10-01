# FP16 forward components and resident FP32 regression

Implemented source: `9f010c9cf41d6077df90cb8228b8c250663a333b`.
All jobs below completed with exit0 from a clean immutable checkout on aarch64,
Ascend910_9392, Torch/TorchNPU2.10 and CANN9.0.0. The [audit](resident-fp16-forward-components-20261001.json)
checks source/core/binary/loader/log and profiler CSV identities.

This qualifies FP16 state, Read, SwiGLU, emission and normalized Aggregate
**components**, plus the existing complete FP32 resident regression. It does not
qualify complete FP16 resident inference/training, peer progression or throughput.
The Python entry is a client of C++/CANN device scheduling.

## Gates

- Normal full standalone and separately owned Python builds passed; standalone
  passed five CTests. Loader checks exclude conflicting Python/standalone owners.
- All62 registered standalone cells passed in four independent device jobs:
  components23 (physical9), windows20 (physical1), adjoints13 (physical3), updates6
  (physical13), each using logical0. Full tests use the same frozen implementation.
- Python/NPU resident inference/training:196 passed,zero skipped,physical9.
- CPU interfaces:76 passed,193 explicitly optional NPU skips. The unchanged
  portable core's8,954 tests were not rerun; its source/binaries were hash-verified.

| New coverage per dtype | FP32 and FP16 |
| --- | --- |
| State/Read | 48 configurations,336 windows,18 refusals |
| Normalized Aggregate | 64 configurations,192 replays |
| SwiGLU | 16 component cases,4 refusals; additional256 FP32 graph windows |
| Emission | 16 component cases,6 refusals; additional66 FP32 graph windows |

State products/additions round at every declared operation, event and Add-repeat
tick. Packed prefill cannot retain extra precision between events. Read products,
reductions and scores stay FP32, including scores beyond FP16's finite range.
Normalized Aggregate stores source products in payload dtype; normalization and
ordered accumulation use FP32. Contribution/summary storage rounds separately.
These FP32 normalization banks still need low-precision optimizer publication.

FP32 thresholds remain unchanged. Only the existing Full training dependency
uses its declared conditioned-control comparison:13 frames,max abs4.470348358154297e-6.
Other complete training gates retain strict controls. Existing AdamW trajectory
eps1e-5 and public default eps1e-8 are unchanged.

## Separate FP16 traces

| Component | Observed task engines/counts |
| --- | --- |
| State/Read | AI_VECTOR_CORE98 |
| Aggregate | AI_VECTOR_CORE4736,MIX_AIV128 |
| SwiGLU | AI_VECTOR_CORE326,AI_CORE40 |
| Emission | AI_VECTOR_CORE665,AI_CORE27 |

No AiCPU task or logged CPU fallback was observed. These bounded traces include
construction and CPU assertions; neither task sums nor instrumented wall times
are throughput measurements. Collection was capped at256MB per trace and180s.

Raw records are under `artifacts/execution-flows-NAME`: builds
`build-low-precision-forward[-python]-clean01`; gates
`low-precision-forward-{components,windows,adjoints,updates,python,host}-clean01`;
profiles `low-precision-{state,aggregate,swiglu,emission}-profile-clean02`.
The two development failures remain unchanged:state-dev01 failed on an implicit
half scalar Read operand (fixed by explicit widening);state-dev02 was cancelled
because Make repeated vendor work across top-level targets. A single selected
aggregate target removes that redundant build path. The later passed dev03 and
these clean builds do not relabel either historical record.
