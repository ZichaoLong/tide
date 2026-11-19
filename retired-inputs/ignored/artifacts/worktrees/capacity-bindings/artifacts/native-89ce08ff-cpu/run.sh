#!/bin/bash
set -euo pipefail
source /usr/share/Modules/init/bash
module purge
module load libtorch-npu/2.12.0-cann9.2.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
python3 /mi/data2T/zlong/home/.agents/skills/develop-portable-torch/scripts/inspect_libtorch_npu_sdk.py "$LIBTORCH_NPU_ROOT" --format json --check-loader > "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-89ce08ff-cpu/sdk-inspection.json"
for backend in CPU NPU; do
 cmake -S "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-89ce08ff-cpu/source/native/regional" -B "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-89ce08ff-cpu/build-$backend" -DCMAKE_BUILD_TYPE=Release -DTIDE_NATIVE_BACKEND="$backend"
 cmake --build "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-89ce08ff-cpu/build-$backend" --parallel 2
 ctest --test-dir "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-89ce08ff-cpu/build-$backend" --output-on-failure
 ldd "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-89ce08ff-cpu/build-$backend/tide-regional" > "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-89ce08ff-cpu/loader-$backend.txt"
 sha256sum "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-89ce08ff-cpu/build-$backend/tide-regional" > "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-89ce08ff-cpu/binary-$backend.sha256"
done
module purge
module load ascend/dev-workspace-9.2.0
/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-89ce08ff-cpu/source/scripts/check_native_region.py" --binary "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-89ce08ff-cpu/build-CPU/tide-regional" --device cpu --output-dir "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-89ce08ff-cpu/checks"
cd "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-89ce08ff-cpu/source"
PYTHONPATH="$PWD/src" /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -m pytest -q tests/test_native_region.py
python scripts/check_upstream.py
