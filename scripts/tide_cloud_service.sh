#!/usr/bin/env bash
# Install the current committed controller on this host. Cloud jobs own their workers.
set -euo pipefail
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
control="$repo/.local/modelarts-r3/control"
python="$repo/.venv-pilot-r3/bin/python"
maflow=$(realpath "$(command -v maflow || echo "$HOME/.local/bin/maflow")")
tool=$(dirname "$maflow")
rev=$(git -C "$repo" rev-parse HEAD)
ma_rev=$(git -C "$tool" rev-parse HEAD)
release="$control/releases/$rev-$ma_rev"
mkdir -p "$control/releases"
if [[ ! -d $release ]]; then
    tmp=$(mktemp -d "$control/releases/new-XXXXXXXX")
    trap 'rm -rf -- "$tmp"' EXIT
    git -C "$repo" archive "$rev" pilotlab scripts | tar -x -C "$tmp"
    mkdir "$tmp/maflow-tool"
    git -C "$tool" archive "$ma_rev" | tar -x -C "$tmp/maflow-tool"
    mv -T "$tmp" "$release"
    trap - EXIT
fi
unit=tide-r3-guiyang-controller-20261128.service
if systemctl --user -q is-active "$unit" && \
        systemctl --user show "$unit" --property=ExecStart --value | grep -Fq -- "$release/scripts/tide_cloud_controller.py"; then
    exit 0
fi
seconds=$("$python" -B -c 'import json,sys,time; print(max(1,int(json.load(open(sys.argv[1]))["deadline_unix"]-time.time())+86400))' "$control/payload/production.json")
cache="${TIDE_CLOUD_CHECKPOINT_CACHE:-/mi/sfs_turbo/zlong/tide-learning-lab-cloud/r3-20261128/objects}"
foundation=$("$python" -B - "$control" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

area = Path(sys.argv[1])
spec = json.loads((area / "payload/production.json").read_bytes())
inventory = (area / "initial-tasks.json").read_bytes()
if hashlib.sha256(inventory).hexdigest() != spec["initial_tasks_sha256"]:
    raise ValueError("frozen task inventory changed")
# The catalog reader must understand every frozen policy carried by the campaign.
# Each scientific worker still loads the exact foundation named by its own task.
for row in json.loads(inventory):
    raw = Path(row["path"]).read_bytes()
    task = json.loads(raw)
    if task["model"].get("config", {}).get("shannon_support_policy") == "weak-log-v1":
        if hashlib.sha256(raw).hexdigest() != row["fingerprint"]:
            raise ValueError("frozen weak-log task changed")
        src = area / "payload/bases" / row["source"] / "foundation/src"
        config = src / "tide/config.py"
        if hashlib.sha256(config.read_bytes()).hexdigest() != spec["files"][str(config.relative_to(area / "payload"))]:
            raise ValueError("frozen model configuration reader changed")
        print(src.resolve(strict=True))
        break
else:
    raise ValueError("frozen R3 weak-log foundation missing")
PY
)
if systemctl --user -q is-active "$unit"; then
    systemctl --user stop "$unit"
fi
systemd-run --user --unit="$unit" --slice=background.slice --collect --working-directory="$repo" \
    --setenv="TIDE_CLOUD_PROJECT_ROOT=$repo" --setenv="PATH=$release/maflow-tool:/usr/local/bin:/usr/bin:/bin" \
    --setenv="PYTHONPATH=$foundation" \
    --setenv="OPENBLAS_NUM_THREADS=1" --setenv="OMP_NUM_THREADS=1" --setenv="TORCH_DEVICE_BACKEND_AUTOLOAD=0" \
    --property="RuntimeMaxSec=$seconds" --property=Restart=on-failure --property=RestartSec=15 \
    --property="StandardOutput=append:$control/controller.log" --property="StandardError=append:$control/controller.log" \
    "$python" -B -u "$release/scripts/tide_cloud_controller.py" --control "$control" --checkpoint-cache "$cache"
