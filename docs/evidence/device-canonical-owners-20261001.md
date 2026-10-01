# Canonical multi-device owners and internal training

Source **a365e2f1635f3ef8f9b66176662b058ef1b39645**, clean fixed checkout.
[Machine audit](device-canonical-owners-20261001.json) records all nine successful
terminal jobs, exact source/build/binary/kernel/log identities and raw failures.
Contract: [resident peers](../resident-peers.md#canonical-owners-and-complete-internal-training-steps).

Actual retained candidate gradients now reduce on canonical owner NPUs. SGD/AdamW
propose on every card, reach a common device error decision, then commit and
publish to each used forward alias. Shared Full/state/Read uses update once.
No CPU reference trajectory, gradient sum or updated parameter is a candidate
input. This is internal Full-sharded training; state/KV and online scheduling
remain on the coordinator. Public multi-device clients, whole-model placement
and medium/full-size throughput remain unqualified work.

## Qualification

Environment: aarch64 Ascend910_9392, CANN9.0.0, Torch/LibTorch-NPU2.10.0.
Standalone SDK and Python-owned runtime builds remain separate. The standalone
build reused host objects only after terminal dev05 object/source/header hashes
matched; both variants relinked against their own verified dependencies. The
Python runtime rebuilt eight affected host objects. New CANN kernels match the
passed dev02 build. Standard CMake configuration also generated the three new/
affected checker link dependencies successfully; that preflight is not a full
fresh compiler run. The machine audit checks archive membership and contents.

| Fixed check | Result |
| --- | --- |
| Two-card retained graph VJP | FP32/FP16 each50 trajectories,200 retained windows; canonical device gradients compared to independent CPU FP32/FP64 |
| Two-card canonical optimizer | Each dtype4 trajectories,32 updates; SGD/AdamW,aliases,groups,None/zero,empty partitions,each source's nonfinite refusal and FP16 publication overflow |
| Two-card continued training | Each dtype40 trajectories,640 windows,160 updates; streaming/prefill,mixed tanh/LH/SwiGLU,normalized Aggregate,HARD/HST/SOFTP,event/fiber caches |
| Three-card memory/locality | Four policy/dtype cells,each2 trajectories,32 windows,8 updates |
| One-owner degeneration | Two dtype cells,each2 trajectories,32 windows,8 updates |
| Single-device regression | Original optimizer gates in both dtypes; public half-cache1 trajectory,16 windows,4 updates |
| Python-owned public regression | Five independent scalar-oracle/fresh-process checkpoint tests passed,no skips |

Training checks compare actual device gradients against separately computed CPU
FP32/FP64 references, then compare masters, slots and counters against a CPU
optimizer using its own gradients. Device banks equal the candidate's rounded
masters exactly, including HARD Read aliases, strided Q/K/V and FP32 normalization
banks containing widened FP16 values. Subsequent windows consume those live banks
and the candidate's own continuation. Connected-zero and disconnected windows
stay distinct. A connected nonfinite contribution changes no card's master,
slot,counter or forward bank. Malformed sources/groups/publication and capacity
requests fail explicitly. No numerical tolerance was widened.

## Three-card profile

A separate two-trajectory training smoke includes32 windows,eight accepted
updates and two deliberately refused update replays. All three physical devices
executed `tide_owner_gradient_pack`, `tide_owner_gradient_reduce`, optimizer
proposal/commit and `tide_owner_parameter_publish`. Per device there were ten
reduction/commit/publication kernel invocations; error-gated invocations do not
mean ten accepted updates.

Observed task counts: **28187 AI_VECTOR_CORE,802 AI_CORE,409 MIX_AIV; zero AiCPU**.
There were198 host model submissions,1282 matching notify records/waits,
4419 device switches and10575 asynchronous memory-copy tasks. Counts include
construction, assertions and exports; they are not a throughput comparison or
proof that all shared-server operations avoid CPU work. Full raw CSVs remain in
the task artifact store with SHA256 receipts. Profiling and correctness runs are
separate from future uninstrumented performance measurements.

## Retained development failures and limits

Three original failures are preserved: a missing brace in the new optimizer test
initializer; an empty-partition test arena set to64KiB where CANN IndexSelect
requires77312bytes (only its arena was raised to1MiB); and a CPU byte comparison
attempting a dtype view on a zero-dimensional Float bank (reshape fixed). These
were test construction/assertion failures, not evidence of candidate numerical
success. The later immutable gates establish the reported result.

The final gate ran only affected checks; unchanged8,954 CPU checks and unrelated
Python suites were not repeated. There is no speedup, full-model NPU residency,
public multi-device checkpoint or CUDA execution claim. All nine jobs reached
exit0. The older intentionally paused CPU Attention process is outside this
increment and remains preserved.
