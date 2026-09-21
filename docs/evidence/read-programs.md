# Region Read modes and independent readouts, 2026-09-21

Clean source: `4b5e14f6412c587d3ccec4ce75f1b6c88dbc3820`.
Command: `python scripts/qualify.py --output-dir artifacts/read-20260921-1311 --jobs 2`.
Environment: aarch64 CPU, Python 3.11.15, Torch/LibTorch 2.10.0+cpu, GCC 10.3.1,
C++17 Release, two build workers, ATen/BLAS threads 1, node workers up to 3.
Build and tests exited 0: **1918 passed**, pytest 162.73 seconds.
Retain `artifacts/read-20260921-1311/{status.json,task.log,verification/}`.
Both manifests identify clean source; verification retains native source and
binary hashes. Unit `tide-foundation-read-20260921-1311` is inactive, MainPID 0.
Development build `artifacts/read-build-20260921-1305/` passed; 186 targeted
checks passed in 17.72 seconds before the implementation commit.

Read is independent from the memory kernel. Requests expose complete content and
only the region-mode-appropriate state, including slots and clocks. Every
candidate still computes its proposal. A hand-computed three-candidate case
selects nodes 1, 0 and 2 respectively for content, old and proposal modes; its
second event selects 0, 2 and 0. Descriptor VJPs distinguish absent connections
from connected-zero, including independent samples and future inputs. An idle
fourth node preserves its initial state and never executes Read.

The standalone custom native Read consumes source positions, contributions,
logical time, selected state and observation clock. For two events, its descriptor
loss and per-element VJPs are:

| Mode | Loss | First input | Second input | Initial state | Decay | Read gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| content | 54 | 4 | 6 | None | None | 27 |
| old | 72 | 6 | 6 | 3 | 1 | 36 |
| proposal | 94 | 7 | 8 | 1.5 | 2.5 | 47 |

Scalar streaming, packed parallel streaming and frontier match these formulas;
the other sample remains disconnected. No-grad execution matches and does not
replay. Scalar custom Read batch fallbacks are counted. Python's custom Read also
checks named SSM slots, mode-specific state clocks, shared registered parameters,
AdamW and checkpoint state. Malformed scalar shape/dtype, nonfinite scores,
changed batch counts and unmatched native overrides fail explicitly.

Differential tests cover observe-all versus selected-only adoption, selected
clear, causal versus prefilled state, first-order trace/output/state VJPs,
self-loop and chain specializations, SettleGraph direct/encoded/chain execution,
cursor cuts, detach and checkpoint continuation in FP64/FP32. All earlier
attention, matrix-memory, Aggregate/Emit and isolated-root checks pass.

Native graph identity format is v9; checkpoint payload remains v3. Region mode
changes reject old checkpoint identity before weights change. Current selectors
require scalar descriptors of the payload dtype. Tuple descriptors, LH's FP64
norm policy, full Next requests, original-LH parity and performance remain later
work. This result makes no higher-order AD or pretrained-model compatibility claim.
