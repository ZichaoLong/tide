# SettleGraph and specialization qualification, 2026-09-21

Clean tested source: `a1da7fe1886dde6c55cbd016c0f4c95e1f9249f7`.
Environment: aarch64 CPU, Python 3.11.15, Torch/LibTorch 2.10.0+cpu; GCC 10.3.1,
C++17 Release. Command: `python scripts/qualify.py --output-dir artifacts/m4-20260921-0830`.
Build and tests exited 0: **301 tests passed**, pytest 70.93 seconds.
Native adapter SHA256: `a951b6898452a8111278c54440a0a1b259f4e75d56727ee7915cfb042dde1282`.
Retain `artifacts/m4-20260921-0830/{status.json,task.log,verification/}`.

Adds direct SettleGraph vs encoded Python/native streaming/frontier, including
multiple output sources, selected-only state adoption/clear, all three Emit
modes, complete projected traces, final states/history and VJPs. Chunk/decode
tests preserve gradients across position cuts. Partial-cut projection is rejected.
Regular encoded graphs automatically form input/body/output region blocks.

Independent Python/native self-loop and chain schedules match the general
streaming oracle, and the Python SettleGraph chain specialization matches its
generic executor. The C++ binary `tidegraph-smoke` independently loads LibTorch,
executes a delayed loop and computes an input VJP. Both dtypes match Python;
unsupported backend/dtype, unknown options and output overwrite fail explicitly.

The operator profile is still EMA plus a tanh FFN, source-weighted sum Agg and
HARD/HST/SOFTP Emit. Identity adapters add no trainable parameters. Advanced
memory modules, general operator plug-ins, original-LH numerical alignment and
large-workload performance remain pending. Native SettleGraph graph compilation
currently uses a Python frontend; native execution is implemented by the exact
TimedDAG encoding, as documented in `../settle-embedding.md`.
