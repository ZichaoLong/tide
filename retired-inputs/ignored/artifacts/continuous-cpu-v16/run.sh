#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/continuous-cpu-v16/source/src
export TIDE_SOURCE_COMMIT=0992b467584c1f603314ed04d8efe8596a14f576
export TIDE_HARNESS_COMMIT=0992b467584c1f603314ed04d8efe8596a14f576
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TIDE_RESOURCE_IDENTITY='pinned_cpu; shared host'
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/continuous-cpu-v16/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/continuous-cpu-v16/status.json --source 0992b467584c1f603314ed04d8efe8596a14f576 --max-rss-gib 64.0 -- numactl --membind=7 --physcpubind=280-295 /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import os,pathlib,subprocess
from scripts.job import atomic
root=pathlib.Path("/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/continuous-cpu-v16")
command=['"'"'/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python'"'"', '"'"'-c'"'"', '"'"'import json,math,pathlib,statistics,subprocess,sys\nfrom scripts.job import atomic\nroot=pathlib.Path("/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/continuous-cpu-v16")\nrows=[]\nscores={}\nfor threads in (4,8,16):\n    latencies=[]\n    for label,mode in (("fc-k-small","inference"),("hb-timed-small","train")):\n        path=root/("calibration-"+str(threads)+"-"+label+".json")\n        command=[sys.executable,"-m","benchmarks.continuous","--config","configs/continuous/"+label+".json","--device","cpu","--precision","fp32","--subject","tide","--mode",mode,"--threads",str(threads),"--batch","1","--prefill","128","--prefill-chunk","64","--decode","8","--warmup","1","--updates","3","--output",str(path)]\n        code=subprocess.call(command)\n        row=dict(threads=threads,label=label,command=command,exit_code=code)\n        if code: raise RuntimeError(row)\n        result=json.loads(path.read_text())\n        latency=statistics.median([entry["complete_seconds"] for entry in result["observations"]])\n        row["median_complete_seconds"]=latency\n        latencies.append(latency)\n        rows.append(row)\n        atomic(root/"calibration.json",dict(state="running",rows=rows))\n    scores[threads]=math.exp(sum(map(math.log,latencies))/len(latencies))\nselected=min(scores,key=scores.get)\natomic(root/"calibration.json",dict(state="completed",rows=rows,geometric_scores=scores,selected_threads=selected,scope="finite compact CPU continuous schedule; same seed and three stateful segments, not a universal thread optimum"))\nconfigs=["configs/continuous/"+name+".json" for name in ("fc-k-small","fc-s-small","hb-settle-small","hb-timed-small")]\ncodes=[]\nfor batch in (1,4):\n    command=[sys.executable,"-m","benchmarks.programme","--suite","continuous","--configs",*configs,"--device","cpu","--precision","fp32","bf16","--threads",str(selected),"--batch",str(batch),"--length","128","--prefill-chunk","64","--decode","8","--warmup","1","--steps","3","--continuous-joint","A","B","C","D","--checkpoints","--child-timeout","600","--output-dir",str(root/("b"+str(batch)))]\n    codes.append(subprocess.call(command))\nsys.exit(int(any(codes)))'"'"']
atomic(root/"numa-policy.json",dict(policy='"'"'--membind=7'"'"',affinity=sorted(os.sched_getaffinity(0)),maps=pathlib.Path("/proc/self/numa_maps").read_text(),status=pathlib.Path("/proc/self/status").read_text(),numactl=subprocess.check_output(["numactl","--show"],text=True),supersedes='"'"'continuous-cpu-v14'"'"'))
os.execv(command[0],command)'
