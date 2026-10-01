# Resident FP16 extended graph reverse qualification

Source `db9e985e9d7163563ed8877be28b5f8148f490fd`; all four immutable-source
jobs passed with exit0. The [audit](resident-fp16-extended-graph-20261001.json)
binds source, archives, oracle object, CANN kernels, loader closure, six gate
cells, raw failures and profiler CSVs. Production code is unchanged from the
previous qualified graph reverse implementation `25e996c`.

| Check, each FP32/FP16 | Completed result |
| --- | --- |
| Extended retained reverse | 110 trajectories,440 windows, replay after closing/poisoning live forward buffers |
| Base graph reverse regression | 122 windows, replay |
| Base retained reverse regression | 42 trajectories,168 windows, replay |

The extension covers all four normalized Aggregate profiles, nine LH
activation/normalization profiles, SwiGLU and mixed modules; shared parameter
owners, feedback/parallel edges, streaming/greedy prefill, HARD/HST/SOFTP,
widths3/257, None/connected-zero, input/initial-state/parameter gradients and
full forward observables. Independent CPU FP32/FP64 autograd references consume
the common public fixture. The half reference preserves actual forward rounding;
half roots are scaled by256. VJP tolerances remain rtol2e-3/atol2e-5 for half and
1e-5/1e-6 for FP32. Discrete identities and gradient connectivity remain exact.

The half extended profile recorded254,571 AI_VECTOR_CORE,2,391 AI_CORE and
6,057 MIX_AIV tasks, with no observed AiCPU or logged CPU fallback. This trace
includes CPU references and construction; it establishes placement, not throughput.
Runtime: aarch64 Ascend910_9392, public LibTorch/TorchNPU2.10.0/CANN9.0.0.
Only affected oracle/checker objects were rebuilt; terminal source-matched
production/CANN/core dependencies were authenticated and reused. Unchanged
Python clients and the portable core were not requalified redundantly.

Two development failures remain retained. Dev01 correctly rejected an
insufficient128MiB wide SwiGLU reverse budget; the width257 checker now explicitly
admits256MiB. Dev02 exposed a native CPU FP32 LayerNorm cancellation residual
(-6.1035e-5 in an analytically zero component). An independent CPU probe and
FP64 derivative confirmed it. The half oracle keeps FP32 statistics and half
forward storage, differentiating normalization with native CPU FP64 autograd;
an analytic zero-component anchor checks this choice. Neither fix changed
production code or loosened tolerances.

Attention cache whole-graph/retained integration, public half training and
master/checkpoint lifecycle, peer progression and full-size performance remain
pending. This report does not certify those paths.
