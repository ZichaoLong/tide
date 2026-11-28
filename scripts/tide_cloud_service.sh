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
if systemctl --user -q is-active "$unit"; then
    if systemctl --user show "$unit" --property=ExecStart --value | grep -Fq -- "$release/scripts/tide_cloud_controller.py"; then
        exit 0
    fi
    systemctl --user stop "$unit"
fi
seconds=$("$python" -B -c 'import json,sys,time; print(max(1,int(json.load(open(sys.argv[1]))["deadline_unix"]-time.time())+86400))' "$control/payload/production.json")
cache="${TIDE_CLOUD_CHECKPOINT_CACHE:-/mi/sfs_turbo/zlong/tide-learning-lab-cloud/r3-20261128/objects}"
systemd-run --user --unit="$unit" --slice=background.slice --collect --working-directory="$repo" \
    --setenv="TIDE_CLOUD_PROJECT_ROOT=$repo" --setenv="PATH=$release/maflow-tool:/usr/local/bin:/usr/bin:/bin" \
    --setenv="OPENBLAS_NUM_THREADS=1" --setenv="OMP_NUM_THREADS=1" \
    --property="RuntimeMaxSec=$seconds" --property=Restart=on-failure --property=RestartSec=15 \
    --property="StandardOutput=append:$control/controller.log" --property="StandardError=append:$control/controller.log" \
    "$python" -B -u "$release/scripts/tide_cloud_controller.py" --control "$control" --checkpoint-cache "$cache"
