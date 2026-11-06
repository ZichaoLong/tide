#!/usr/bin/env bash
# Fixed clean checkout; invoke through job.py in a bounded background.slice unit.
set -euo pipefail
out=${1:?usage: qualify_ascend_migration.sh NEW_OUTPUT NPU_QUEUE_HELPER [all|half|N]}
npu_helper=${2:?absolute cooperative queue helper required}
jobs=${3:-all}
test -z "$(git status --porcelain)" || { echo 'fixed clean source required' >&2; exit 1; }
mkdir "$out"
out=$(realpath "$out")
root=$PWD
source scripts/local_ascend.sh
tide_load_ascend
test "$ASCEND_CANN_VERSION" = 9.2.0 || { echo 'this migration gate selects CANN9.2.0' >&2; exit 1; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/bin/bash scripts/prepare_local_ascend.sh "$out/env"
py="$out/env/bin/python"
"$py" -m pip install --disable-pip-version-check -c "$ASCEND_CONSTRAINTS_FILE" nlohmann_json==3.11.2
mkdir "$out/wheels"
"$py" -m pip wheel --no-deps --no-build-isolation --no-cache-dir --wheel-dir "$out/wheels" .
"$py" -m pip install --no-deps "$out"/wheels/tidegraph-*.whl
"$py" scripts/build.py --backend npu --npu-runtime python --build-dir "$out/build-python" --jobs "$jobs"
json_include=$("$py" -c 'import nlohmann_json; print(nlohmann_json.get_include())')
build_path=$PATH
# No Python Torch import in this standalone build, nor its runtime loader.
(
  module purge
  module load libtorch-npu/2.12.0-cann9.2.0
  export PATH="$build_path:$PATH"
  /usr/bin/python3 "$root/scripts/build.py" --backend npu --npu-runtime standalone \
    --libtorch-prefix "$LIBTORCH_ROOT" --cxx11-abi 1 --build-dir "$out/build-sdk" --jobs "$jobs"
  /usr/bin/python3 "$root/scripts/build_weighted_benchmark.py" --core-build "$out/build-sdk" \
    --json-include "$json_include" --output-dir "$out/weighted" --jobs "$jobs"
  /usr/bin/python3 - "$out/sdk-environment.json" "$root" <<'PY'
import os,sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[2])/'scripts'))
from durable_records import write_json
# Exactly the public site's runtime allowlist; omit credentials and visibility
# (the cooperative queue supplies physical visibility later).
keys=('PATH','LD_LIBRARY_PATH','PYTHONPATH','ASCEND_HOME_PATH','ASCEND_OPP_PATH',
      'ASCEND_AICPU_PATH','ASCEND_TOOLKIT_HOME','TORCH_DEVICE_BACKEND_AUTOLOAD')
write_json(sys.argv[1],{k:os.environ[k] for k in keys if k in os.environ})
PY
)
checks_failed=0
TIDE_BUILD_DIR="$out/build-python" "$py" -m pytest -q tests/test_ascend_migration.py \
  tests/test_semantics.py tests/test_weighted_native.py --dtype both --junitxml="$out/cpu-tests.xml" || checks_failed=1
"$py" scripts/check_ascend_migration.py --build-root "$out" --output-dir "$out/cpu" --device cpu || checks_failed=1
"$py" "$npu_helper" --job-id "gef-cann92-$(basename "$out")" --project graph-execution-foundation \
  --priority 0 --npu-count 1 --max-wait-seconds 1800 --state-file "$out/npu-queue.json" -- \
  "$py" "$root/scripts/check_ascend_migration.py" --build-root "$out" --output-dir "$out/npu" --device npu:0 || checks_failed=1
# New optional LM increment; never mutate the old 9.1 experiment environment.
/bin/bash scripts/prepare_lm_environment.sh "$out/lm"
tide_lm_environment "$out/lm"
"$lm_python" -m pip install --no-deps "$out"/wheels/tidegraph-*.whl
"$lm_python" -m pytest -q tests/test_lm_runtime.py tests/test_lm_recovery.py tests/test_lm_queue.py \
  --junitxml="$out/lm-tests.xml" || checks_failed=1
exit "$checks_failed"
