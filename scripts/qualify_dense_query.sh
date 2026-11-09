#!/usr/bin/env bash
# Fixed-source installed-wheel gate; finite exploration retains strict failures.
set -euo pipefail
if [[ $# != 3 || ( $2 != cpu && $2 != npu ) ]]; then
  echo "usage: $0 NEW_OUTPUT cpu|npu MATCHING_NATIVE_BUILD (activate fixed wheel)" >&2
  exit 2
fi
out=$(realpath -m "$1")
target=$2
export TIDE_BUILD_DIR=$(realpath "$3")
mkdir "$out"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python - "$out" <<'PY'
import hashlib, importlib.metadata as md, json, os, platform, subprocess, sys
from pathlib import Path
import tidegraph
sys.path.insert(0,"scripts")
from build_identity import source_hash
from durable_records import write_json
root=Path.cwd().resolve(); package=Path(tidegraph.__file__).resolve()
if package.is_relative_to(root) or subprocess.check_output(['git','status','--porcelain'],text=True):
    raise RuntimeError('qualification requires clean source and separately installed wheel')
build=Path(os.environ['TIDE_BUILD_DIR']); manifest=json.loads((build/'build-manifest.json').read_text())
if manifest['cpp_source_sha256'] != source_hash(root):
    raise RuntimeError('native build source mismatch')
for name,expected in manifest['binary_sha256'].items():
    if hashlib.sha256((build/name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('native binary changed: '+name)
write_json(Path(sys.argv[1])/'identity.json',dict(source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    package=str(package), python=sys.version, torch=md.version('torch'), torch_npu=md.version('torch-npu'),
    native_build=manifest, visibility=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), platform=platform.platform()))
PY
if [[ $target == npu ]]; then
  : "${ASCEND_RT_VISIBLE_DEVICES:?run inside the site NPU queue}"
  export TIDE_DENSE_DEVICE=npu:0
  dtype=float32
else
  export TIDE_DENSE_DEVICE=cpu
  dtype=both
fi
export TIDE_DENSE_NUMERIC_REPORT="$out/numeric"
python -m pytest -q -o pythonpath= tests/test_weighted_dense_query.py tests/test_weighted_dense_training.py \
  --dtype "$dtype" --tb=short --junitxml="$out/checks.xml"
for client in python native; do
  python examples/consumer_core5/dense_query.py --device "$TIDE_DENSE_DEVICE" --client "$client" \
    --native-library "$TIDE_BUILD_DIR" --output-dir "$out/consumer-$client"
done
python scripts/check_upstream.py > "$out/upstream.json"
