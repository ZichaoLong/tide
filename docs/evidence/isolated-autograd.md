# Isolated-root packed autograd, 2026-09-21

Clean source: `c04b89eb5e2ab414a3367ae4ae8b65f24cc08768`.
Command: `python scripts/qualify.py --output-dir artifacts/autograd-20260921-1034`.
Environment: aarch64 CPU, Python 3.11.15, Torch/LibTorch 2.10.0+cpu, GCC 10.3.1,
C++17 Release, two build workers, ATen/BLAS threads 1, node workers up to 3.
Build and tests exited 0: **1267 passed**, pytest 180.29 seconds.
Retain `artifacts/autograd-20260921-1034/{status.json,task.log,verification/}`.
Native source and binary hashes are recorded in `verification/result.json`.

The earlier 903-test suite did not detect cross-event autograd connections from
packing when only one public tensor roots the loss. Dense operations could give
unused initial memories and upstream parameters connected-zero gradients where
the scalar reference returned None. Local semantic autograd replay now preserves
packed forward values and routes backward through independent event graphs.
See `../packed-autograd.md` for the implementation, cost and qualification scope.

The 364 added cases cover EMA, diagonal SSM, attention/GQA/window, Linear
Attention and gated DeltaRule in FP64/FP32, with HARD/HST/SOFTP. They separately
root early outputs, pending messages, persistent slots and trace content,
proposal, descriptor, control and next tensors. Comparisons preserve gradient
presence for separate inputs, initial-state leaves and upstream parameters;
future observations and other samples remain disconnected when specified.

Repeated retained-graph VJPs include losses multiplied by zero. Mixed frozen and
trainable lanes preserve requires_grad. AdamW skips disconnected parameters and
matches reference optimizer state. In-memory cuts, pending delivery and native
cursor snapshots preserve connectivity. Native serial/node-parallel streaming,
packed/per-sample/step frontier, independent chain/self-loop schedules and
direct/encoded SettleGraph are covered. Caller-owned stacked input tensors retain
their ordinary connected-zero row VJPs. Inference mode and no_grad report zero
semantic replays; existing 903 tests also pass.

This establishes the tested first-order root-to-input/parameter/initial-state
contract. It does not establish arbitrary internal adjoints, higher-order AD,
stochastic replay, optimized packed training backward, throughput or LH parity.
Training currently pays the extra scalar/event replay cost reported in counters.
