#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/scale-cpu-hb-17b-v15/source/src
export TIDE_SOURCE_COMMIT=0992b467584c1f603314ed04d8efe8596a14f576
export TIDE_HARNESS_COMMIT=0992b467584c1f603314ed04d8efe8596a14f576
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TIDE_RESOURCE_IDENTITY='pinned_cpu; shared host'
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/scale-cpu-hb-17b-v15/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/scale-cpu-hb-17b-v15/status.json --source 0992b467584c1f603314ed04d8efe8596a14f576 --max-rss-gib 512.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/scale-cpu-hb-17b-v15/source/scripts/wait_units.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/scale-cpu-hb-17b-v15/dependency.json --seconds 25200 --unit tide-attention-sites-cpu-v14.service -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import json,math,pathlib,statistics,subprocess,sys
from scripts.job import atomic
root=pathlib.Path("/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/scale-cpu-hb-17b-v15")
resource=dict(meminfo=pathlib.Path("/proc/meminfo").read_text(),quota=subprocess.check_output(["quota","-s"],text=True),units=subprocess.check_output(["systemctl","--user","list-units","tide-*","--state=running","--no-pager"],text=True))
atomic(root/"resource-preflight.json",resource)
available=int(next(line.split()[1] for line in resource["meminfo"].splitlines() if line.startswith("MemAvailable:")))*1024
if available<700*2**30: raise RuntimeError("large CPU programme needs at least 700 GiB currently available; observed "+str(available))
scores={}
rows=[]
for threads in (16,32,64):
    path=root/("calibration-"+str(threads)+".json")
    command=[sys.executable,"-m","benchmarks.run","--config","configs/scale/hb-1b.json","--device","cpu","--precision","fp32","--threads",str(threads),"--batch","1","--length","8","--warmup","1","--steps","2","--modes","inference","train","--subjects","tide","--output",str(path)]
    code=subprocess.call(command)
    if code: raise RuntimeError("large workload thread calibration failed: "+str(threads))
    data=json.loads(path.read_text())
    seconds=[row["median_seconds"] for row in data["observations"]]
    scores[threads]=math.exp(sum(map(math.log,seconds))/len(seconds))
    rows.append(dict(threads=threads,command=command,latencies=seconds,score=scores[threads]))
    atomic(root/"calibration.json",dict(state="running",rows=rows))
selected=min(scores,key=scores.get)
atomic(root/"calibration.json",dict(state="completed",rows=rows,selected_threads=selected,scope="finite B1/L8 HB1B inference/training calibration for larger CPU capacity work; no universal thread optimum"))
command=[sys.executable,"-m","benchmarks.programme","--suite","scale","--configs","configs/scale/hb-17b.json","--device","cpu","--precision","fp32","bf16","--modes","inference","--threads",str(selected),"--batch","1","--length","8","--warmup","1","--steps","3","--child-timeout","3600","--output-dir",str(root/"comparison")]
a=subprocess.call(command)
mem=pathlib.Path("/proc/meminfo").read_text()
available=int(next(line.split()[1] for line in mem.splitlines() if line.startswith("MemAvailable:")))*1024
atomic(root/"training-resource-preflight.json",dict(meminfo=mem,required_available_bytes=700*2**30,available_bytes=available))
if available<700*2**30: raise RuntimeError("17B CPU training not admitted: available RAM below planning margin")
b=subprocess.call([sys.executable,"-m","scripts.check_placement","--mode","train","--devices","cpu","--no-foreach","--config","configs/scale/hb-17b.json","--precision","bf16","--threads",str(selected),"--batch","1","--length","8","--steps","2","--output",str(root/"training-bf16.json")])
sys.exit(int(bool(a or b)))'
