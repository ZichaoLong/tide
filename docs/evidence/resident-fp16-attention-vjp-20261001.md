# FP16 packed attention adjoint

Implementation: `f20c2cc227a400d08c200475be56976ffe4ccc07`.
[Audited records](resident-fp16-attention-vjp-20261001.json) bind the fixed clean
source, binaries, dependencies, loader closure, tests and profiler CSVs.
All six bounded jobs passed with exit0. This qualifies the local attention
component, not complete half event/fiber cache reverse or graph training.

Q/K/V use FP16 and QK recomputation preserves its half rounding. Additive bias,
public cotangents, global normalization and all adjoints use FP32. The softmax
correction uses the complete unrounded weighted sum across physical key tiles;
the returned saved forward output is half-rounded only afterward. This keeps
downstream projection operands consistent with the existing half forward.

Both FP32 and FP16 gates cover four geometries and12 changing-prefix replays:
GQA, key capacity257, width257, partial/empty prefixes, disconnected poisoned
roots, connected zeros, unused padding and explicit capacity refusal. Independent
CPU FP32/FP64 autograd references compare all four gradients and forward output.
Half uses rtol2e-3/atol2e-5; FP32/FP64 retains2e-5/2e-6. Two additional half
fixtures, with physical tile1 and2, pass the original strict tolerance and must
distinguish missing QK rounding before checking the candidate.

Affected regressions passed: FP32 event/fiber training66/172 root cases and8/20
trajectories, plus61 Python precision/event/fiber tests with zero skips. The
unchanged portable core and the preceding215-case client qualification were
not repeated. Standalone reused byte-matched terminal host/kernel archives and
rebuilt the checker; Python-owned rebuilt its affected attention host object
and relinked the client against matching CANN archives. These are affected-path
builds, not complete vendor rebuilds.

Environment: aarch64 Ascend910_9392, CANN9.0.0, Torch2.10.0 and matching
standalone/Python-owned NPU stacks. Component/training/Python checks used
physical9/13/9 and the independent profile used physical1, all mapped to logical0.
The half profile contained1178 AI_VECTOR_CORE,94 AI_CORE and24 MIX_AIV tasks;
no AiCPU or logged CPU fallback was observed. It includes construction and
CPU assertions and is not a throughput measurement.

The raw `low-precision-attention-vjp-dev01` failure is retained: all ordinary
FP32/FP16 cases passed, but its proposed strict fixture failed its own sensitivity
test. Two score-rounding errors were nearly equal and mostly canceled in softmax.
Independent CPU arithmetic showed that changing one public key from-2.71875 to
-2.703125 exposes the missing-rounding error. Candidate arithmetic and tolerances
were unchanged. The corrected fixture passes both physical tilings.

Reaudit with `TASK/launchers/precision_attention_vjp_evidence.py FULL_HASH`, where
`TASK=/mi/data2T/zlong/tide-execution-flows`; retained snapshots, build/reuse
manifests, raw lifecycle logs and traces live there. Half cache integration,
control/graph reverse, retained training, public resume, peer progression and
the full-size performance matrix remain separate work.
