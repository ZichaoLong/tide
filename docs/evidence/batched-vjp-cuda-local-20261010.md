# Batched VJP and CUDA local qualification, 2026-10-10

The G1 CPU gate passed on clean `a02c18c1bf274a8542461efe486bd43e02aac8b1`.
The G3 CUDA/NPU builds passed on clean
`bda25de29465140dc47df8f488514447cbcb64b3`; target entry checks passed on clean
`afbc07556c693e9809bb005154a4b9b806d16c51`. This is new-source evidence.
It establishes no current-source NPU device execution or real GPU execution.
[Reviewed identities, counts and artifact hashes](batched-vjp-cuda-local-20261010.json).

## Implemented scope

Built-in State attaches the actual packed numeric outputs to batched structural
first-order VJPs, grouping defined cotangents by output connectivity. Sequences
batch independent owners at the same causal depth. Add preserves each tick's
multiply and rounding. Read has explicit batched linear/norm VJPs and one finite
check per scoring device/dtype group. Disconnected gradients remain None;
connected zero gradients remain connected. Custom State/Read and nondefault
native fiber policies retain replay. Full/Aggregate batching was already active.
These are composed autograd/ATen paths, without a fusion claim.

The Norm correction preserves `dy*(x/norm)` ordering before converting a FP32
score derivative to FP64 payload dtype. The earlier `x*(dy/norm)` changed
rounding; the exact regression witness and original tolerances remain. No
selector, graph semantics, discrete comparison or near-tie policy was expanded.

CUDA implements the common DeviceProgram/DeviceSequence boundary with candidate
owned conditional WHILE/SWITCH graphs, cuBLAS numerical stages, exact integer
controls and peer mailboxes. All102 shared semantic kernel sources compile
through its adapter. Source scope includes the declared three families, two
schedules, modules, inference/full first-order training, continuation,
checkpoint, capacity, optimizers and multi-owner/locality paths. Actual GPU
operation is unverified. One thread per semantic worker, serial reductions and
scatter scans, bounded32KiB scratch and uncalibrated device peaks are explicit
limits, not performance guarantees.

## Qualification results

| Gate | Fixed source | Result and boundary |
| --- | --- | --- |
| CPU FP64/FP32 complete pytest | a02c18c | 9595 passed,654 optional hardware/feature skips;2109.16s |
| Complete topology suite | a02c18c | all22 cells passed, including active64 mixed TimedDAG/Settle and32-node positive-delay feedback with parallel edges, plus original benchmark topologies |
| Installed library consumers | a02c18c | all10 Python/native/C++ checks passed |
| Standalone CPU CTest | exact same core, afbc075 launch | all12 FP32/FP64 runtime, greedy, Settle, Read precision, placement and payload ownership checks passed |
| CUDA core and resident packages | bda25de | standalone66 manifest targets and Python2 libraries; installed and combined consumers built,7 host CTests |
| CUDA CPU semantic adapter | bda25de | all102 sources compiled,1 host contract CTest passed |
| CUDA-linked directed CPU pytest | bda25de | 229 passed,84 optional device skips; no skipped case becomes GPU evidence |
| NPU core and resident packages | bda25de | separate standalone/Python owners;66 standalone targets and2 Python libraries; installed/combined/scale clients built,7 resident host CTests and5 scale CTests |
| Target-entry identity and refusal | afbc075 | 6 stale-consumer regressions passed; absent CUDA fails at the only preflight stage |
| Python dynamic imports | afbc075 with exact bda25de binaries | CUDA and NPU matching native/resident extensions and host placement metadata load; neither initializes a device |

The CPU qualifier reuses the immutable `g1-norm-dev06` build only after exact
C++/binary digest checks. Its construction manifest points to the earlier dirty
norm-fix development snapshot; the qualified source is clean a02c18c with the
same bytes. Common core digest is
`dbc370539ed261846524c427b7b9a37a38f6857ba6a6ccb78ad365168140a150`.
G3 component digest is
`53dfba264427b829ead61e59524e28d490488f1ce854c767be43b00b1ccf058b`.
The audit checks core/backend/consumer binaries, loader records, clean source
identities and terminal units with MainPID0 and empty cgroups. Python-only
consumer files execute from the fixed qualification checkout; compiled consumer
C++/header/CMake inventories must match their installed executables.

The environment is aarch64/GCC10.3.1/C++17/Python3.11.15, CPU Torch2.10.0,
CUDA Torch2.10.0+cu128/toolkit12.8.1/architectures80,90,100(+PTX), and NPU
Torch/TorchNPU2.10/CANN9.0.0. NPU uses the public SDK module with separate wheel
and standalone owners. Shared-driver/module installations were not modified.
Python NPU import retained shared-installation owner warnings; no operator was
run, so this import provides no device health or numerical evidence.

## Reproduction and retained failures

The raw root is the `TASK` path in the canonical STATUS. Relevant jobs are
`g1-cpu-qualification03`, `g4-cpu-standalone01`, `g3-{cuda,npu}-clean01`,
`g3-target-identity-clean01` and `g3-plugin-import01`; each has its own immutable
source, plan, durable state and log. `plans/audit-g1-g3-clean01.py` verifies the
recorded artifacts and refuses overwriting its existing output.

The CPU release command at a02c18c was:

```bash
python scripts/qualify_library.py --reuse-build --build-dir "$MATCHING_CPU_BUILD" --output-dir "$NEW_GATE"
ctest --test-dir "$MATCHING_CPU_BUILD" --output-on-failure --timeout 60
```

`g1-cpu-qualification01` remains failed73/passed9451/skipped723:69 accounting,
omitted-consumer and inherited-RSS harness failures plus4 real Norm VJP rounding
regressions. The fixes were committed before qualification03. Gate02 remains
cancelled after diagnosis. Earlier NPU qualification01/02 never acquired a device.
The original unavailable-CUDA wrapper remains failed because of an obsolete
exception substring; its retained preflight rejection was independently audited,
and the corrected entry check passed on afbc075. None of these records was relabeled.

All16 local NPUs reported external processes in the retained resource inventory;
no current-stage NPU workload was submitted to those occupied devices. The
current NPU target status is implemented/build-tested, with old verified evidence
still tied to its old commits. New module, complete-training, resident/multicard
and profiling gates await an allocation. Actual GPU tests await a GPU host.

Target build, complete zero-skip gate, explicit conditioned-comparison option and
residency trace commands are in [device-control](../device-control.md). The full
entry needs3 remapped logical devices and default strict comparisons; an absent
explicit device fails. CUDA requires12.8 conditional SWITCH support, CC8.0+ and
bidirectional P2P/native system atomics for peers. Real correctness, residency,
multicard, memory and performance must each be recorded on that target.

The [original strict route failure](original-add-route-witness-20261004.md)
remains intact. Old qualification and performance are not promoted to this source.
[New finite CPU selection evidence](batched-selection-cpu-20261010.md) is separate.
