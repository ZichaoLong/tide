# LH Full activation, normalization and signaling qualification

2026-09-21; clean implementation `5df88e7b91985092870d0197888d8d95311ee7b1`.
**2596 tests passed in 193.61 seconds**. Outer job, verification and original-code
oracle all passed with exit 0 and empty dirty status at that source. Unit
`tide-foundation-lh-full-20260921-1535` is inactive, MainPID 0.
Artifacts: `artifacts/lh-full-20260921-1535/{status.json,task.log,verification/,oracle/}`.
Both subdirectories retain result JSON and logs; oracle logs are named
`{selector,add,full}-{float64,float32}.log`.

Command: `python scripts/qualify.py --output-dir artifacts/lh-full-20260921-1535
--jobs 2 --lh-snapshot artifacts/lh-source-20260921-1428`, via scripts/job.py in
background.slice. CPU aarch64, Python 3.11.15, Torch/LibTorch 2.10.0+cpu, C++11 ABI;
two build jobs, ATen/BLAS single-thread, backend autoload disabled.

## Original inference component

The original snapshot remains identity
`ac7c878a56aeb55eec9f919da1962be6134fc5e3d1872f6d2303917306edb87f`, capturing
actual dirty LH source at HEAD `5fd237d40c9880ccb6e511e4bf20799c7022fd1e`.
Its ModuleUtils factories construct the actual activation, norm and signaling
modules. Source inventory/hashes, core library and executable hashes are retained;
runtime assertions are enabled. No reference worktree files changed.

Each dtype passed **72 configurations, 504 rows**: ReLU/SiLU, identity/RMS/Layer
normalization, widths 1/3/5, output degrees 1/3 and signaling bias on/off. Compare
auxiliary normalized activations and each mapped output slot for scalar and
packed Full. Rows include zero variance, all-negative, constant and mixed-sign
values. The original Selector and Add components also passed again in both
dtypes (12/54 cases respectively; see their prior evidence for domains).

## Training and graph schedules

- Independent value and analytic derivative anchors for activation and RMS/Layer
  normalization, HARD/SOFTP/HST, zero variance and malformed norm parameters.
  Held content is absent under HARD and connected zero under HST for a Full-only
  root; unused FFN weights remain disconnected.
- Add memory, selected clear with preserved comparison, per-slot signaling,
  competing nodes, parallel edges and ragged/missing samples; output, in-flight
  and persistent-state roots with independent parameter/input/initial leaves.
- Python/native, native serial/node-parallel packed streaming, frontier, direct
  and encoded SettleGraph, independent Python/native fixed chain/self-loop and
  Python SettleGraph specialization.
- Cut composition, detach, shared normalization across different output slot
  domains, AdamW and checkpoint alias/value/clock round-trip.

Before commit, 421 focused tests and 8 additional fixed-topology checks passed.
All are included in the clean full suite. Contracts are in `../lh-full.md`.
Profiles use the original default eps (RMS 1e-7, LayerNorm 1e-5) and equal payload
widths. This does not qualify arbitrary LH configurations, whole IOCortexNet,
same-fiber attention, Pronounce, higher-order AD or performance. Training authority
and first-order public-root replay limitations remain Tide's own contracts.
