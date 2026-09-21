# Original IOCortexNet whole-model inference projection

2026-09-22 (Asia/Shanghai), clean implementation
`776fc2e597dac872f4204ac3a09966db39ab74c0`.
**3296 tests passed in 240.23 seconds**. All seven qualification records passed
with exit 0 and empty dirty status at that source under
`artifacts/iocortex-20260921-1855/`: outer status, verification, oracle,
oracle-release, pronounce-release, iocortex-release and iocortex-python.
Unit `tide-foundation-iocortex-20260921-1855` is inactive, MainPID 0, exit 0.
The outer job finished at 2026-09-21T19:28:09Z.

Command: `python scripts/qualify.py --output-dir artifacts/iocortex-20260921-1855
--jobs 2 --lh-snapshot artifacts/lh-source-20260921-1428`, through scripts/job.py
in background.slice. CPU aarch64, Python 3.11.15, Torch/LibTorch 2.10.0+cpu,
C++11 ABI, two build workers, OMP/OpenBLAS single-thread, backend autoload off.
Schemas remain graph v11/checkpoint v4; this gate adds an oracle and mapping,
without changing core executor behavior. Build, source, CMake, binary, original
snapshot, Python-source and exported-data hashes are retained in the manifests.

## Qualified projection

The actual unchanged original `IOCortexNet::think` and `think_single_step` run
embedding, four adjacency blocks, CHAL updates, original heap/tensor selection,
activation/norm, selected clear and Pronounce. The Tide body has unit-delay wires,
explicit intra-CSC/bridge-CSC/token-last input slots and original CSR signaling
weight slices. Original edge IDs are reversed relative to CSR order and parallel
wires remain distinct. See `../lh-iocortex.md` for the contract and code map.

Compare complete candidate/proposal sets, FP64 descriptors, exact selected routes,
post-selection activations, all-node Add physical hidden or KV/log-bias (including
idle and cleared samples), both original counter maps, every emitted message,
final pending coordinates/payloads and labeled logits. Token cuts and individual
ragged ticks also match whole-window execution and complete continuation.
Four-token cases include two actual greedy feedback inputs, sealed after readout.

| Original variant / dtype | Cases | Candidates | Messages | Token/sample logits | Unavailable |
| --- | ---: | ---: | ---: | ---: | ---: |
| assertions on / FP64 | 180 | 215100 | 848052 | 2880 | 24 |
| assertions on / FP32 | 204 | 243780 | 961188 | 3264 | 0 |
| assertions off / FP64 | 204 | 243780 | 961188 | 3264 | 0 |
| assertions off / FP32 | 204 | 243780 | 961188 | 3264 | 0 |

The 24 unavailable configurations are original FP64 active-softmax multi-batch
diagnostic failures, not numerical passes. The component oracle checks the known
hardcoded-FP32 denominator exception; the separate consistent build disables
ENABLE_RUNTIME_ASSERTION and fills numerical coverage. Ordinary C/C++ assertions
remain enabled via -UNDEBUG in both variants. No original source was modified.
The 69-file snapshot identity is
`ac7c878a56aeb55eec9f919da1962be6134fc5e3d1872f6d2303917306edb87f`.

Original exports are versioned, dtype/shape-explicit JSON, protected by hashes and
an exact scenario inventory. Independent Python time-major body execution and
reference/frontier readout passed **48 fixtures / 28680 candidate events**, including
whole/cut state, clocks, occurrence ledgers, hidden, counters, pending and logits.
Its checks reject disabled Python assertions and changed source/oracle/input files.
All earlier tests and original Selector/Add/Full/Attention/Pronounce gates reran.

## Scope and numerical policy

N=11 per cortex, width 4, two heads, batch 4, vocabulary 7; four tokens plus ten
ragged ticks. Each case uses the same Add or one of five attention pooling
profiles throughout body/readout. Clear, lead and native/original schedules vary;
clear is coupled to relu/L=3 versus silu/L=2, and lead to bias+RMS versus
no-bias+identity. Native schedules are serial, three-worker unpacked and
three-worker packed. Original uses single/multi PACKED and attention CROSSBATCH;
tensor selection is exercised in CROSSBATCH. Python exports use single projection
and lead+bias+RMS, both clear settings and all profiles/token/ragged scenarios.

Each Read must equal the FP64 norm of its own candidate at FP64 tolerance.
Cross-implementation norm differences are bounded by the actual candidate L2
perturbation plus FP64 roundoff. Candidate tensors retain the declared payload
tolerance (FP64 1e-10/1e-8, FP32 1e-6/1e-5); route mismatches still fail exactly.
Promoting a norm does not remove upstream FP32 arithmetic error.

This qualifies a bounded, equal-width, fixed-weight two-clock inference
composition. It does not prove single-PDG containment, arbitrary original
configuration/pretrained import, unequal widths, whole-model training or composite
checkpoint/optimizer ownership, high-order AD, or large-sparse performance.

## Retained development records

`iocortex-dev-20260921-1820` failed on missing hidden-type namespace aliases.
`iocortex-dev-20260921-1822` passed 180 FP64 cases, then its FP32 descriptor test
incorrectly demanded FP64 upstream accuracy. The corrected norm check above
preserves the payload and exact-routing contract. Both source archives and logs
remain failed and reproducible.

`iocortex-dev-20260921-1832` passed both original FP32 variants (204 cases each)
and 24 Python fixtures/14340 events before the clean commit. Artifact-cleanup
dry run found no eligible entries; no files or reference trees were removed.
