# FP16 normalized Aggregate and extended Full adjoints

Implementation: `5ce5346208fe73b33c9a5a85f1fef56441d76073`.
[Audited records](resident-fp16-extended-vjp-20261001.json) bind the clean source,
reused dependencies, archive members, loader closure, gates and profiler CSVs.
Seven bounded jobs passed with exit0. This is single-device component and
affected FP32 training/client evidence, not complete FP16 training or throughput.

The local environment was aarch64 Ascend910_9392, CANN9.0.0, Torch2.10.0 and its
matching standalone LibTorch-NPU/Python-owned TorchNPU stacks. The two runtimes
were built and checked separately. Builds rebuilt affected host/checker/fixture
objects and reused terminal, byte-matched CANN archives and other dependencies;
they were not complete vendor rebuilds. The portable core was unchanged.

| Gate | Verified result |
| --- | --- |
| Aggregate FP32 / FP16 | Each39 configurations,117 replays; CPU FP32/FP64 references |
| LH/SwiGLU FP32 / FP16 | Each90 configurations,270 replays; half adds3 strict fixtures, each with3 replays |
| FP32 event/fiber training |66/172 root cases,8/20 trajectories, unchanged strict controls |
| Python-owned affected client |215 passed, zero skips across five resident test modules |

Aggregate checks all four normalized pooling rules, physical-scale gradients,
absent domains, present-zero and zero-scale messages, poisoned padding,
duplicate logical-source refusal and memory admission. Its half fixture uses
non-dyadic stored payloads and a larger public cotangent to expose source-product
rounding. A CPU comparison must distinguish omitting that rounding. The candidate
and both CPU references still meet the original rtol1e-5/atol1e-6.

Extended Full checks all nine LH profiles and SwiGLU, widths1/7/257,
disconnected and connected-zero roots, unused poisoned parameters, ordered
owner reduction and19/7/0-row replay. Saved forward activations/products use
half arithmetic; cotangents and owner accumulation use FP32. Ordinary half
checks use2e-3/2e-5; existing FP32/FP64 checks remain1e-5/1e-6. Separate RMSNorm,
LayerNorm and SwiGLU fixtures distinguish full-FP32 recomputation and pass
strict1e-5/1e-6, including normalization-gradient cancellation.

The original half LayerNorm implementation failed: CANN returned half-rounded
mean/rstd values in FP32 output buffers. A width257 ReLU/LayerNorm row had
rstd32.09375 versus the FP32 reference32.0815 and an input-gradient error around
8.75e-4. The fix retains the real half normalization output for weight gradients
and computes only the Jacobian's statistics in FP32 on the actual half activation.
Neither the half forward nor the comparison tolerance was changed to pass.

Two separate half profiles passed:

| Component | AI_VECTOR_CORE | AI_CORE | MIX_AIV |
| --- | ---: | ---: | ---: |
| Aggregate |2277|0|182|
| LH/SwiGLU |8966|88|195|

No AiCPU tasks or logged CPU fallback were observed. These profiles include
construction and CPU assertions; their task counts are not throughput measures.
Component, training and Python checks used physical9/13/1 respectively;
profiles used physical3/9, each remapped to logical0. No performance comparison
is made across those allocations.

Raw failures remain under the task's `runs/` directory: Aggregate dev01 build
name collision, extended dev01 test-helper linkage, and extended dev02/dev03
half LayerNorm numerical/diagnostic failures. They were not relabeled as passes.
The audit script is `TASK/launchers/precision_extended_vjp_evidence.py FULL_HASH`;
`TASK=/mi/data2T/zlong/tide-execution-flows`. Source snapshots, build/reuse
manifests, lifecycle records and raw logs remain available there.

Whole-graph FP16 reverse, event/fiber cache integration, retained-window training,
public FP16 training/resume, peer progression and full-size performance remain
separate work. Passing local adjoints does not remove those guards or certify
training convergence.
