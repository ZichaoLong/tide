#!/usr/bin/env bash
# Directed installed-wheel qualification, not a full backend/core-5 matrix.
set -euo pipefail
if [[ $# != 2 || ( $2 != cpu && $2 != npu ) ]]; then
  echo "usage: $0 NEW_OUTPUT cpu|npu (activate a private fixed-wheel environment first)" >&2
  exit 2
fi
out=$(realpath -m "$1")
target=$2
mkdir "$out"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python - "$out" <<'PY'
import json, os, platform, subprocess, sys
from pathlib import Path
import importlib.metadata as md
import tidegraph
sys.path.insert(0,"scripts")
from durable_records import write_json
root=Path.cwd().resolve()
package=Path(tidegraph.__file__).resolve()
if package.is_relative_to(root):
    raise RuntimeError("qualification requires a separately installed wheel, not this source tree")
dirty=subprocess.check_output(["git","status","--porcelain"],text=True)
if dirty:
    raise RuntimeError("qualification source must be clean")
try:
    torch_npu=md.version("torch-npu")
except md.PackageNotFoundError:
    torch_npu=None
write_json(Path(sys.argv[1])/"environment.json",dict(
    source=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    python=sys.version,executable=sys.executable,platform=platform.platform(),
    package=str(package),torch=md.version("torch"),torch_npu=torch_npu,
    affinity=sorted(os.sched_getaffinity(0)),threads=1,
    visibility=os.environ.get("ASCEND_RT_VISIBLE_DEVICES")))
PY
if [[ $target == npu ]]; then
  # Physical group admission/locks are owned by the enclosing site queue.
  : "${ASCEND_RT_VISIBLE_DEVICES:?run inside the site NPU queue}"
  export TIDE_POSITIVE_SUPPORT_DEVICE=npu:0
else
  export TIDE_POSITIVE_SUPPORT_DEVICE=cpu
fi
python -m pytest -q -o pythonpath= tests/test_positive_support.py \
  tests/test_positive_support_config.py tests/test_positive_support_graph.py \
  --junitxml="$out/support.xml"
if [[ $target == cpu ]]; then
  python -m pytest -q -o pythonpath= tests/test_weighted_entropy_allocation.py \
    tests/test_weighted_allocation.py tests/test_weighted_allocation_batch.py \
    tests/test_weighted_region_batch.py tests/test_weighted_aggregate_batch.py \
    tests/test_weighted_aggregate_vjp.py tests/test_weighted_kv_formulas.py \
    tests/test_weighted_execution.py tests/test_weighted_compute_sharing.py \
    tests/test_weighted_query_training.py tests/test_weighted_placement.py \
    tests/test_core5_planning.py tests/test_experiment_config.py \
    -k 'not native' --junitxml="$out/regression.xml"
  # This older file uses boolean parameter IDs, not python/native labels.
  python -m pytest -q -o pythonpath= tests/test_allocation_failure.py \
    -k 'not True' --junitxml="$out/allocation-failure.xml"
fi
python examples/consumer_core5/positive_support.py \
  --row tests/data/tl_req_003_shannon_row.json --device "$TIDE_POSITIVE_SUPPORT_DEVICE" \
  > "$out/public-row.json"
python scripts/check_upstream.py > "$out/upstream.json"
