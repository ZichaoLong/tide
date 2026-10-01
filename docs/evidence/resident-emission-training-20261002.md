# HARD resident slot-affine training qualification

Implementation: `96c75f8d9c2bee54a5000f4c410fe3d5764ec552`.
All eight immutable-source jobs passed; [machine-readable audit](resident-emission-training-20261002.json).
[Mathematical/API scope](../resident-emission-vjp.md).

Two isolated builds freshly link byte-verified terminal host objects, the six
changed/new device kernels and unchanged core dependencies. Standalone loader
closure excludes Python; the Python-owned build excludes the standalone NPU runtime.
Public ABI users were rebuilt according to recursive header dependencies.

| Gate | Passed scope |
| --- | --- |
| Two-card matrix | FP32/FP16 each16 trajectories; total512 windows/128 updates |
| Owner/card transitions | Three trajectories,48 windows/12 updates;2→legacy single for both dtypes,1→2 FP32 |
| Legacy single FP16 |12 trajectories,192 windows/48 updates |
| Broadcast/cache regression | Two trajectories,32 windows/8 updates |
| Public Python client |10 cases,no skips; three families,both schedules,refusals,zero-output slots |
| Separate two-card FP16 profile | One trajectory,16 windows/four updates |

Independent CPU FP32/FP64 and rounded-forward references check complete states,
pending/output values,physical edge identity,all parameter/boundary gradients,
None versus connected zero,shared owners,explicit phases,optimizer state and
checkpoint suffixes. Actual unscaled forward values preserve delivery-scale
adjoints even at zero scale. No numerical thresholds were relaxed.

The trace records19,022 AI_VECTOR_CORE,357 AI_CORE and288 MIX_AIV tasks; no AiCPU
was observed. Actual emission reverse link/plan/payload tasks execute16/42/56 times.
It includes construction,checks and checkpoint work, so neither task counts nor
elapsed time are a throughput comparison or a bottleneck-share measurement.

Projection banks and their physical partial gradients still occupy the
coordinator. Full/state/KV and canonical optimizer owners may span devices.
This does not qualify compact projection sharding,HST/SOFTP slot-affine emission,
the full model consumer,full-size throughput,CUDA or other CANN versions.

Run records: `artifacts/execution-flows-build-emission-reverse-{clean01,python-clean01}`
and `artifacts/execution-flows-emission-reverse-{matrix,placements,legacy,regression,python,profile}-clean01`.
The audit verifies clean source,build and loader hashes,archive members,binaries,
raw trajectory counts,queue terminal states and profiler CSV hashes. Development
records remain separate and unchanged.
