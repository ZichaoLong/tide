"""One subprocess owns one exact configuration/variant/repeat, CPU only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from durable_records import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',required=True,choices=['cpu'])
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--build-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--variant',required=True)
    parser.add_argument('--modes',nargs='+',required=True,choices=['nograd-forward','grad-forward','backward','optimizer','train-step'])
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--warmup',type=int,default=2)
    args=parser.parse_args()
    if args.workers<1 or args.threads<1 or args.warmup<0:parser.error('positive worker/thread counts and nonnegative warmup required')
    sys.path.insert(0,str(args.build_dir.resolve()))
    os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
    import torch
    import _tide_native as core
    from foundation_workloads import estimate
    from foundation_execute import Execution
    from foundation_measure import measure
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    config=json.loads(args.config.read_text());out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    record={'state':'starting','pid':os.getpid(),'config':config,'variant':args.variant,'modes':args.modes,
            'preflight':estimate(config),'model_constructed':False}
    def save():
        write_json(out/'worker.json',record)
        assert json.loads((out/'worker.json').read_text())==record
    save();started=time.perf_counter()
    try:
        before_threads={p.name for p in Path('/proc/self/task').iterdir()}
        execution=Execution(config,args.variant,args.workers)
        after_threads={p.name for p in Path('/proc/self/task').iterdir()}
        total=sum(p.numel() for p in execution.model.parameters())
        if total!=record['preflight']['parameters']:raise RuntimeError('allocated parameter owners disagree with exact preflight')
        record.update(state='running',model_constructed=True,parameters=total,
                      graph_identity=execution.graph.identity,options=execution.options,
                      constructor_thread_ids=sorted(after_threads-before_threads),
                      constructor_threads_created=len(after_threads-before_threads),
                      actual_affinity=sorted(os.sched_getaffinity(0)),
                      worker_resolution='serial Python/baseline variants use1; worker variants use the requested limit',
                      head_workers=0,head_worker_note='No vocabulary DenseLinear in this graph-only suite',
                      setup_seconds=time.perf_counter()-started,
                      trace_policy='Python reference retains full semantic traces; native wall pass trace=false',
                      clock_stride=execution.spec.stride if execution.spec else execution.stride,
                      actual_input_lengths=execution.lengths)
        save()
        def progress(stage):
            record['stage']=stage;save()
        measured,profile=measure(execution,args.modes,args.warmup,progress)
        write_json(out/'measured.json',measured);write_json(out/'profile.json',profile)
        record.update(state='passed',exit_code=0,seconds=time.perf_counter()-started,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      actual_threads=torch.get_num_threads(),actual_interop_threads=torch.get_num_interop_threads(),
                      parallel_info=torch.__config__.parallel_info(),process_threads=next(s for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('Threads:')),
                      native_binary_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest())
        try:
            from threadpoolctl import threadpool_info
            record['thread_pools']=threadpool_info()
        except ImportError:record['thread_pools']='threadpoolctl unavailable; see native parallel_info'
    except BaseException as error:
        record.update(state='failed',exit_code=1,error=f'{type(error).__name__}: {error}')
        raise
    finally:save()
    return record['exit_code']


if __name__=='__main__':sys.exit(main())
