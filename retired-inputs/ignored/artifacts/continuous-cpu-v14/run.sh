#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/continuous-cpu-v14/source/src
export TIDE_SOURCE_COMMIT=0992b467584c1f603314ed04d8efe8596a14f576
export TIDE_HARNESS_COMMIT=0992b467584c1f603314ed04d8efe8596a14f576
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TIDE_RESOURCE_IDENTITY='pinned_cpu; shared host'
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/continuous-cpu-v14/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/continuous-cpu-v14/status.json --source 0992b467584c1f603314ed04d8efe8596a14f576 --max-rss-gib 64.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/continuous-cpu-v14/source/scripts/wait_units.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/continuous-cpu-v14/dependency.json --seconds 18000 --unit tide-representative-cpu-v5.service --unit tide-scale-npu-hb-9b-v12.service --unit tide-scale-two-npu-hb-17b-v12.service --unit tide-scale-four-npu-hb-9b-train-v14.service -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import json,math,pathlib,statistics,subprocess,sys
from scripts.job import atomic
root=pathlib.Path("/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/continuous-cpu-v14")
rows=[]
scores={}
for threads in (4,8,16):
    latencies=[]
    for label,mode in (("fc-k-small","inference"),("hb-timed-small","train")):
        path=root/("calibration-"+str(threads)+"-"+label+".json")
        command=[sys.executable,"-m","benchmarks.continuous","--config","configs/continuous/"+label+".json","--device","cpu","--precision","fp32","--subject","tide","--mode",mode,"--threads",str(threads),"--batch","1","--prefill","128","--prefill-chunk","64","--decode","8","--warmup","1","--updates","3","--output",str(path)]
        code=subprocess.call(command)
        row=dict(threads=threads,label=label,command=command,exit_code=code)
        if code: raise RuntimeError(row)
        result=json.loads(path.read_text())
        latency=statistics.median([entry["complete_seconds"] for entry in result["observations"]])
        row["median_complete_seconds"]=latency
        latencies.append(latency)
        rows.append(row)
        atomic(root/"calibration.json",dict(state="running",rows=rows))
    scores[threads]=math.exp(sum(map(math.log,latencies))/len(latencies))
selected=min(scores,key=scores.get)
atomic(root/"calibration.json",dict(state="completed",rows=rows,geometric_scores=scores,selected_threads=selected,scope="finite compact CPU continuous schedule; same seed and three stateful segments, not a universal thread optimum"))
configs=["configs/continuous/"+name+".json" for name in ("fc-k-small","fc-s-small","hb-settle-small","hb-timed-small")]
codes=[]
for batch in (1,4):
    command=[sys.executable,"-m","benchmarks.programme","--suite","continuous","--configs",*configs,"--device","cpu","--precision","fp32","bf16","--threads",str(selected),"--batch",str(batch),"--length","128","--prefill-chunk","64","--decode","8","--warmup","1","--steps","3","--continuous-joint","A","B","C","D","--checkpoints","--child-timeout","600","--output-dir",str(root/("b"+str(batch)))]
    codes.append(subprocess.call(command))
sys.exit(int(any(codes)))'
