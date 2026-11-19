#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
unset PYTHONPATH
module load libtorch-npu/2.12.0-cann9.2.0
exec "$@"
