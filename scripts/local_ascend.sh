# Source in a task shell. Site activation is separate from portable Tide code.
tide_load_ascend() {
  source /usr/share/Modules/init/bash
  module purge
  export TIDE_ASCEND_MODULE="${TIDE_ASCEND_MODULE:-ascend/dev-workspace-9.2.0}"
  module load "$TIDE_ASCEND_MODULE"
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
}

tide_lm_environment() {
  export TIDE_LM_ENV="${TIDE_LM_ENV:-$1/env/transformers-cann${ASCEND_CANN_VERSION:?load baseline first}}"
  local guard
  guard="$(dirname "${BASH_SOURCE[0]}")/ascend_environment.py"
  "$TIDE_LM_ENV/bin/python" -B "$guard" --require-venv >/dev/null
  lm_python="$TIDE_LM_ENV/bin/python"
}
