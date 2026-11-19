#!/bin/bash
set -euo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PIP_CONFIG_FILE=/dev/null PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1
install_python=$(command -v python)
"$install_python" -m pip wheel --no-deps --no-build-isolation --no-index --no-cache-dir --wheel-dir "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/installed-check/artifacts/capacity-installed-8a22530d-v1/dist" "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/installed-check/artifacts/capacity-installed-8a22530d-v1/build-source"
"$install_python" -m venv --system-site-packages "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/installed-check/artifacts/capacity-installed-8a22530d-v1/env"
"/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/installed-check/artifacts/capacity-installed-8a22530d-v1/env/bin/python" -m pip install --no-deps --no-index --no-cache-dir --ignore-installed "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/installed-check/artifacts/capacity-installed-8a22530d-v1/dist/tide_instances-0.1.0-py3-none-any.whl"
cd /
env -u PYTHONPATH "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/installed-check/artifacts/capacity-installed-8a22530d-v1/env/bin/python" -I "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/installed-check/artifacts/capacity-installed-8a22530d-v1/check_installed_capacity.py" --wheel "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/installed-check/artifacts/capacity-installed-8a22530d-v1/dist/tide_instances-0.1.0-py3-none-any.whl" --venv "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/installed-check/artifacts/capacity-installed-8a22530d-v1/env" --source-manifest "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/installed-check/artifacts/capacity-installed-8a22530d-v1/source-manifest.json" --output-dir "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/installed-check/artifacts/capacity-installed-8a22530d-v1/results"
"$install_python" -I "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/installed-check/artifacts/capacity-installed-8a22530d-v1/verify-source.py" "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/installed-check/artifacts/capacity-installed-8a22530d-v1"
cd "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/installed-check/artifacts/capacity-installed-8a22530d-v1/frozen-source"
"$install_python" scripts/check_upstream.py
