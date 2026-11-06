#!/usr/bin/env bash
# Run only in a bounded service. Public modules remain read-only.
set -euo pipefail
experiment_root=$1
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/local_ascend.sh"
tide_load_ascend
environment_path="${TIDE_LM_ENV:-$experiment_root/env/transformers-cann${ASCEND_CANN_VERSION}}"
/bin/bash "$script_dir/prepare_local_ascend.sh" "$environment_path"
source "$environment_path/bin/activate"
python -m pip install --disable-pip-version-check --no-cache-dir \
  --index-url https://pypi.org/simple -c "$ASCEND_CONSTRAINTS_FILE" \
  --report "$environment_path/install-report.json" \
  transformers==4.57.1 huggingface-hub==0.36.0 tokenizers==0.22.1 \
  accelerate==1.11.0 peft==0.17.1
python -I -m pip check
python -B - "$environment_path/ready.json" "$script_dir" <<'PY'
import importlib.metadata as md
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[2])
from durable_records import write_json
from ascend_environment import inspect
versions = {k: md.version(k) for k in ['torch','torch-npu','transformers','huggingface-hub','tokenizers','accelerate','peft','safetensors']}
write_json(Path(sys.argv[1]), dict(state='ready', environment=inspect(require_venv=True), python=sys.executable,
    packages=versions, scope='dependency installation only; operators and models not yet checked'))
PY
