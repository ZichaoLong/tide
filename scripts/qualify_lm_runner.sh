#!/usr/bin/env bash
set -euo pipefail
experiment_root=$1
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.1.0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
lm_python="$experiment_root/env/transformers/bin/python"
implementation_commit=$(git rev-parse HEAD)
install_root="$experiment_root/install/$implementation_commit"
mkdir -p "$install_root"
"$lm_python" -m pip wheel --no-deps --no-build-isolation --no-cache-dir --wheel-dir "$install_root" .
"$lm_python" -m pip install --no-deps --force-reinstall "$install_root/tidegraph-0.3.0-py3-none-any.whl"
"$lm_python" -I -m tidegraph lm matrix > "$install_root/matrix.json"
"$lm_python" -I -m tidegraph lm plan examples/external_lm/qwenmoe-two-npu.json > "$install_root/plan.json"
"$lm_python" -B -m pytest -q tests/test_lm_assets.py tests/test_lm_matrix.py tests/test_lm_runtime.py \
  tests/test_lm_queue.py tests/test_lm_recovery.py tests/test_experiment_cli.py tests/test_experiment_html.py \
  tests/test_lm_retry.py tests/test_lm_engines.py tests/test_shared_npu_policy.py \
  tests/test_experiment_report.py tests/test_container_job.py tests/test_durable_records.py \
  --junitxml="$install_root/tests.xml"
"$lm_python" -I - "$install_root" <<'PY'
import subprocess,sys
from pathlib import Path
from tidegraph.experiment.identity import package_identity
from tidegraph.experiment.records import now,write_json,digest
root=Path(sys.argv[1])
write_json(root/'ready.json',dict(state='passed',finished=now(),
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    package=package_identity(),wheel_sha256=digest(root/'tidegraph-0.3.0-py3-none-any.whl'),
    scope='runner/CLI/random-tiny mechanisms only; no pretrained model or NPU execution inferred'))
PY
