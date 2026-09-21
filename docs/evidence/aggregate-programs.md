# Source-aware Aggregate, 2026-09-21

Clean source: `4fa29b6d2291da006781aa456a15647e874d4fb1`.
Command: `python scripts/qualify.py --output-dir artifacts/aggregate-20260921-1203`.
Environment: aarch64 CPU, Python 3.11.15, Torch/LibTorch 2.10.0+cpu, GCC 10.3.1,
C++17 Release, two build workers, ATen/BLAS threads 1, node workers up to 3.
Build and tests exited 0: **1727 passed**, pytest 184.67 seconds.
Retain `artifacts/aggregate-20260921-1203/{status.json,task.log,verification/}`.
The verification manifest includes C++ source and all five native binary hashes.
The unit is inactive and no worker remains.

The five built-in profiles (sum, mean, positive weighted mean, active-source and
all-source softmax) pass independent forward/VJP formulas, absent versus present
zero, isolated output/contribution/pending roots and AdamW behavior. Active-only
normalizers disconnect absent parameters; all-source logits remain connected via
the denominator. Python/native streaming, native serial/parallel, batch packing,
frontier prefill/step, independent chain/self-loop schedules, SettleGraph direct/
encoded execution, shared parameters, cursor cuts and checkpoint round trips
are covered in FP64/FP32. Inference performs no semantic Aggregate replay.

Standalone native custom Aggregate reads source tags, positions, time and local
slots, returns source contributions and uses a trainable gain. Hand-computed loss
136, input gradients 52 and 4, gain gradient 116 and disconnected second sample
all agree. Python custom-program registration and malformed-result guards pass.

Qualification does not cover arbitrary custom Aggregate under SettleGraph
embedding. Review after this run found a counterexample: converting a boundary
input to an edge changes its kind and origin position as observed by the program.
The built-ins use stable slots and are unaffected semantically. A graph-owned
source-origin view is required before claiming tag-sensitive custom-program
embedding; this is the immediate next fix. Existing state modules consume only
summary content. Full typed-content propagation, Next/Read and region interfaces
remain later work. No LH parity, higher-order AD or performance claim is made.
