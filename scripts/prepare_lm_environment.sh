#!/usr/bin/env bash
# Run only in a bounded service. Public modules remain read-only.
set -euo pipefail
experiment_root=$1
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.1.0
export PYTHONDONTWRITEBYTECODE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
environment_path="$experiment_root/env/transformers"
mkdir -p "$experiment_root/env"
if test -e "$environment_path"; then
  test -f "$environment_path/pyvenv.cfg" || exit 1
else
  python -m venv --system-site-packages "$environment_path"
fi
source "$environment_path/bin/activate"
python -m pip install --disable-pip-version-check --no-cache-dir \
  --index-url https://pypi.org/simple -c "$ASCEND_CONSTRAINTS_FILE" \
  --report "$experiment_root/env/install-report.json" \
  transformers==4.57.1 huggingface-hub==0.36.0 tokenizers==0.22.1 \
  accelerate==1.11.0 peft==0.17.1 pytest==9.0.2
python -I -m pip check
python -B - "$experiment_root/env/ready.json" <<'PY'
import importlib.metadata as md
from pathlib import Path
import sys
from tidegraph.experiment.records import now, write_json
versions = {k: md.version(k) for k in ['torch','torch-npu','transformers','huggingface-hub','tokenizers','accelerate','peft','safetensors']}
assert versions['torch']=='2.12.0+cpu' and versions['torch-npu']=='2.12.0.post2', versions
write_json(Path(sys.argv[1]), dict(state='ready', finished=now(), python=sys.executable,
    packages=versions, scope='dependency installation only; operators and models not yet checked'))
PY
