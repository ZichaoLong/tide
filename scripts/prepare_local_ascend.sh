#!/usr/bin/env bash
# Create/reuse only a venv belonging to the selected read-only public baseline.
set -euo pipefail
environment_path=${1:?usage: prepare_local_ascend.sh PRIVATE_VENV}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/local_ascend.sh"
tide_load_ascend
python -B "$script_dir/ascend_environment.py"
if test -e "$environment_path"; then
  test -f "$environment_path/pyvenv.cfg" || { echo 'existing path is not a venv' >&2; exit 1; }
else
  python -m venv --system-site-packages "$environment_path"
fi
"$environment_path/bin/python" -B "$script_dir/ascend_environment.py" --require-venv \
  --output "$environment_path/baseline.json"
"$environment_path/bin/python" -m pip check
