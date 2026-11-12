# Incremental source-bank admission

**9ae62d0d** replaces repeated pending-prefix layout construction with incremental
unique-source and row-domain sets. The final incidence is constructed once per
accepted numerical group. It preserves the byte formula, accepted group order,
source identities, scalar fallback and numerical forward/VJP. This removes the
previous quadratic row rescanning and repeated growing dense-incidence construction;
final incidence storage still costs G×S and set membership work remains.

[Fixed records](source-planning-20261112.json) contain two successful CANN core
builds, **75 CPU checks**, **7 standalone CPU targets**, and an installed NPU native
consumer's three AdamW steps plus fresh-process exact restore. The 72-receiver
regression verifies one final incidence construction; other checks cover exact
budget boundaries, repeated/independent owners, ragged rows, versions, None/zero,
all source roles and complete training/continuation.

Qualification used 9ae62d0d in a frozen checkout. Builds ran in separate
background.slice services, CPU0–159/160–319, each 512 GiB / 2,400 seconds; fixed
checks used CPU0–7/16–19, 256 GiB / 600 seconds. NPU admission chose physical1,
logical `npu:0`, Python TorchNPU2.12/CANN9.2; the SDK build was separate.
No new cost measurement was made; [72e1d22c costs](source-bank-20261112.md)
and its finite FP32 strict differences keep their original identities.

Reproduce by [building](../development.md) the fixed source, running the recorded
pytest/CTest arguments and `examples/consumer_core5/tensor_window.py --client native
--device npu:0 --dtype float32 --positions --native-library NEW_BUILD --output-dir NEW`.
Commands, manifests, runtime admission and exact result JSON are embedded; historical
run directories are not required. Broader backend and scale qualification is not inferred.
