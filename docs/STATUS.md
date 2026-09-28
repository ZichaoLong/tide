# Current handoff

Updated: 2026-09-28 (Asia/Shanghai). Branch: graph-execution-foundation.

## Active authorized work

Implement Python/LibTorch × CUDA/Ascend NPU for the reusable Tide library.
Single-device eager FP32 is the accelerator baseline; CPU FP32/FP64 remains
required. Preserve independent schedules, graph semantics and installed clients.
CUDA device execution will be qualified on another machine; never infer it from
local compilation. User also authorizes matched CANN version testing/installations.

Starting source: 277b2d3. A1 Python implementation is ready for a coherent commit.
Development snapshot python-dev01 passed 30 directed CPU tests (94.49 s).
python-dev02 passed the mixed TimedDAG NPU configuration gate on CANN 9.0.0,
physical 9 -> logical npu:0, including VJPs, three updates, fresh-process resume
and CPU checkpoint handoff. Both jobs terminated with exit 0. The earlier NPU
python-dev01 failed due to a self-loop fixture size; its record is preserved.

Source snapshots, snapshot hashes, launchers and run records live under
/mi/data2T/zlong/tide-accelerator; project artifacts/accelerator-python-*-dev*
links expose status.json/task.log and gate reports. No active job at this boundary.
Uncommitted A2 work is adapting native device validation, optimizer/checkpoint
placement and worker stream propagation. No new native build is verified yet.
No subagents. Reference repositories remain read-only.

## Next actions

1. Run directed Python CPU tests with the matching immutable existing native build
   /var/tmp/zlong-graph-execution-foundation/library-l3-build; add device boundary tests.
2. Freeze and qualify Python NPU on CANN 9.0.0 using the account device queue.
3. Adapt native tensor validation, runtime lifecycle, checkpoint placement and
   build/consumer paths; validate standalone C++ separately from Python bindings.
4. Complete backend/configuration gates, full CPU regression, installed clients,
   CUDA compile/link checks and available matched CANN version matrix.
5. Commit implementation and exact-source evidence separately; update public
   support matrix with verified, implemented and target-machine-pending scopes.

## Environments and prior evidence

CANN modules 8.5.0/8.5.1/8.5.2 with Torch/TorchNPU 2.9.0 and 9.0.0 with 2.10.0
are installed under /opt. Standalone NPU SDK module: libtorch-npu/2.10.0-cann9.0.0.
CUDA private module: torch-cuda/2.10.0-cu128 under ~/privatemodules.
GPU Toolkit/Python import/nvcc/C++ CPU checks passed; no NVIDIA GPU is present.

Previous CPU library gate: 8621 tests and 22 complex cells at ff708a1; validator
follow-up at aa03a03 (18 directed tests and installed consumers). These remain
historical evidence, not acceptance of this extension. See evidence/library-foundation.md.
Historical Python NPU evidence at 7811418 covers only its finite older eager cases.
No experiment data/head/loss/controller is moved into the library.
