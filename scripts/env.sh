#!/usr/bin/env bash
# Site activation only. Caller supplies this project's immutable environment.
set -euo pipefail
: "${TIDE_LAB_VENV:?Set the project-private venv absolute path}"
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source "$TIDE_LAB_VENV/bin/activate"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
