#!/bin/bash
set -euo pipefail
source /usr/share/Modules/init/bash
module purge
module load libtorch-npu/2.12.0-cann9.2.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cmake -S "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-development/build-v1/source" -B "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-development/build-v1/cpu" -DCMAKE_BUILD_TYPE=Release -DTIDE_NATIVE_BACKEND=CPU
cmake --build "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-development/build-v1/cpu" --parallel 2
ctest --test-dir "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-development/build-v1/cpu" --output-on-failure
cmake -S "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-development/build-v1/source" -B "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-development/build-v1/npu" -DCMAKE_BUILD_TYPE=Release -DTIDE_NATIVE_BACKEND=NPU
cmake --build "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-development/build-v1/npu" --parallel 2
ctest --test-dir "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-development/build-v1/npu" --output-on-failure
ldd "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-development/build-v1/cpu/tide-regional"
ldd "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-development/build-v1/npu/tide-regional"
