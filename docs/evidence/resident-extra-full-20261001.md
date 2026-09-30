# Resident LH/SwiGLU training qualification

Exact implementation **5143de38c9e82b75b8173e5d01e2f5dc0e85b743**, aarch64
Ascend910_9392, Torch/TorchNPU2.10 and CANN9.0.0. Both clean standalone and
Python-owned builds passed. [The audit](resident-extra-full-20261001.json)
checks frozen source, matching core, binaries, loader closure, results, logs
and profiler CSV hashes. The Python plugin does not link the standalone SDK.

The standalone gate passed all 44 component cells and five CTests. Isolated
LH/SwiGLU VJPs passed 90 configurations and 270 long/short/empty replays, each
against independent CPU FP32/FP64 autograd: 540 comparisons. Public C++ training
passed 46 trajectories, 736 retained windows and 184 updates, covering all nine
LH profiles, SwiGLU, mixed modules, shared owners, streaming/greedy, checkpoint
restore and width 257. Wide retained SwiGLU explicitly uses a 2 GiB reverse budget.
The Python client passed 54 device tests with no skips, including all three graph
families, real NPU loss cotangents, disk and fresh-process continuation. Its CPU
interface gate passed 76 tests with 51 optional-NPU skips.

These are conditional numerical results under the [declared comparison
contract](../resident-full-vjp.md), not a strict uniform-tolerance claim:

- Full multi-step trajectories explicitly use AdamW epsilon 1e-5. The public
  default 1e-8, normalization epsilons and ordinary tensor tolerances are unchanged.
- The retained near-zero RMSNorm case at epsilon 1e-8 passes the independent VJP
  and same-device-gradient CPU optimizer oracle. Its independently computed
  end-to-end parameter trajectories fail tolerance and remain labelled failures.
- The gate explicitly selects `--full-training-control-check conditioned`.
  Thirteen control frames require this policy; maximum absolute difference is
  4.470348e-6. Scores, other tensors, gradients and updates keep their original
  tolerances; routes remain exact. Each side's complete-candidate softmax and
  the score-error propagation bound are checked independently. No CPU result
  enters candidate execution or continuation. Strict comparison remains default.

The separate correctness profile completed the same Full checker and observed
48,566 AI_VECTOR_CORE, 533 AI_CORE and 760 MIX_AIV records, including 527 extra-Full,
1,018 optimizer, 1,788 graph-reverse and 292 window-bridge records. No AiCPU record
or host-fallback diagnostic was observed. These are observed trace records,
including construction and CPU assertions, not a throughput measurement or a
claim that task-time sums equal wall time.

This extends the single-NPU FP32 HARD,sum/broadcast training profile to
LH/SwiGLU. The Python interface is a C++/CANN client, not an independently
qualified pure-PyTorch resident scheduler. Normalized Aggregate and attention
adjoints,HST/SOFTP,FP16,peer progression and the full performance matrix remain
separate required work. Original numerical failures retain their raw records
and identities in the audit; later passing scopes do not relabel them.
