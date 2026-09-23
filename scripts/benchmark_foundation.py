#!/usr/bin/env python3
"""One bounded entry for the frozen three-family CPU foundation suite.

Warmups reset state/cache/ledgers. Training uses detached fixed windows, explicit
AdamW epsilon1e-5 and measured forward/backward/optimizer/complete-step phases.
Large evaluation retains failures and stops scaling a family at its first limit.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from build_identity import source_hash
from source_identity import source_state
from durable_records import write_json
from foundation_control import digest,run_variant
from foundation_workloads import VARIANTS,large_config,estimate


def configurations(args,suite):
    if args.tier=='large':
        selected=suite['large_presets'] if not args.ids else [p for p in suite['large_presets'] if p['id'] in args.ids]
        if not selected or (args.ids and {p['id'] for p in selected}!=set(args.ids)):raise ValueError('unknown large preset')
        result=[]
        for preset in selected:
            for family in args.families or ['pdg','timed-dag','settle']:
                target=large_config(preset,family);denom=preset['nominal_selection_denominator']
                for nodes in sorted({4*denom,min(16*denom,target['body_nodes']),target['body_nodes']}):
                    config=large_config(preset,family,nodes);config['target_body_nodes']=target['body_nodes']
                    config['evaluation_stage']='target' if nodes==target['body_nodes'] else 'resource-stage'
                    result.append((config,['native-settle' if family=='settle' else 'native-frontier' if family=='timed-dag' else 'native-stream-packed']))
        return result
    ids=args.ids or (['P01','T01','S01','TR01'] if args.tier=='smoke' else [c['id'] for c in suite['configurations']])
    if not set(ids)<={c['id'] for c in suite['configurations']}:raise ValueError('unknown frozen workload id')
    result=[]
    for saved in suite['configurations']:
        if saved['id'] not in ids:continue
        config=dict(saved)
        if args.tier=='smoke':
            config.update(width=16,batch=4,sequence=6)
            if config['training_window']:config['training_window']=3
        families=['pdg','timed-dag','settle'] if config['graph']=='all-three' else [config['graph']]
        for family in families:
            if args.families and family not in args.families:continue
            item=dict(config,graph=family)
            variants=VARIANTS[config['id']]
            if config['id']=='TR01':
                variants=['native-settle','native-settle-step'] if family=='settle' else ['native-stream-scalar','native-stream-packed'] if family=='pdg' else variants
            if args.variants:
                variants=[v for v in variants if v in args.variants]
                if not variants:continue
            if args.modes:
                if not set(args.modes)<=set(item['modes']):raise ValueError('requested mode is not defined for selected frozen workload')
                item['modes']=args.modes
            result.append((item,variants))
    if not result:raise ValueError('no applicable frozen configuration/variant')
    if args.variants and set(args.variants)-{v for _,vs in result for v in vs}:raise ValueError('unsupported variant for selected workloads')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',required=True,choices=['cpu'])
    parser.add_argument('--tier',required=True,choices=['smoke','medium','large'])
    parser.add_argument('--ids',nargs='+')
    parser.add_argument('--families',nargs='+',choices=['pdg','timed-dag','settle'])
    parser.add_argument('--variants',nargs='+')
    parser.add_argument('--modes',nargs='+',choices=['nograd-forward','grad-forward','backward','optimizer','train-step'])
    parser.add_argument('--build-dir',type=Path,default=Path('build'))
    parser.add_argument('--build',action='store_true',help='build the target-local native core before running')
    parser.add_argument('--build-jobs',type=int,choices=[1,2],default=2)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--suite',type=Path,default=Path(__file__).resolve().parents[1]/'benchmarks/foundation-v1.json')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--repeats',type=int)
    parser.add_argument('--timeout-seconds',type=int)
    parser.add_argument('--tracking',choices=['best-effort','off','required'],default='best-effort')
    parser.add_argument('--describe',action='store_true',help='print concrete topology/owner/resource estimates without building or timing')
    parser.add_argument('--allow-dirty-smoke',action='store_true',help='development smoke only; exact source status/hashes retained')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    args.repeats=args.repeats if args.repeats is not None else (3 if args.tier=='medium' else 1)
    args.timeout_seconds=args.timeout_seconds if args.timeout_seconds is not None else (300 if args.tier=='large' else 180)
    if not 1<=args.repeats<=3 or not 1<=args.timeout_seconds<=1800 or args.workers<1 or args.threads<1 or args.warmup<0:
        parser.error('invalid bounded resources/repetition settings')
    if args.tier=='medium' and (args.repeats!=3 or args.warmup!=2):parser.error('frozen medium suite requires three independent repeats and two reset warmups')
    if args.tier=='large' and (args.variants or args.modes):parser.error('large preset schedule and inference mode are fixed')
    suite=json.loads(args.suite.read_text())
    try:cases=configurations(args,suite)
    except ValueError as error:parser.error(str(error))
    if args.describe:
        print(json.dumps([dict(config=c,variants=v,preflight=estimate(c)) for c,v in cases],indent=2));return 0
    if args.output_dir is None:parser.error('--output-dir required for execution')
    commit,dirty=source_state(root)
    if dirty and not (args.tier=='smoke' and args.allow_dirty_smoke):parser.error('formal timing requires clean frozen source')
    build=args.build_dir.resolve()
    if args.build:
        from foundation_resources import resources
        if args.build_jobs>resources()['cpu_budget']:parser.error('build jobs exceed aggregate CPU budget')
        subprocess.run([sys.executable,str(root/'scripts/build.py'),'--build-dir',str(build),
                        '--jobs',str(args.build_jobs)],cwd=root,check=True)
    manifest=json.loads((build/'build-manifest.json').read_text())
    if manifest['cpp_source_sha256']!=source_hash(root):parser.error('native C++ source/build identity mismatch')
    for name,sha in manifest['binary_sha256'].items():
        if digest(build/name)!=sha:parser.error('native build artifact changed: '+name)
    source=dict(repository='tide/graph-execution-foundation',commit=commit,dirty=bool(dirty),dirty_status=dirty,
                cpp_source_sha256=source_hash(root),native_binary_sha256=digest(build/'_tide_native.so'),build=manifest,
                files_sha256={str(p.relative_to(root)):digest(p) for base in ('python','scripts') for p in (root/base).rglob('*.py')})
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    record={'schema':'tide-foundation-evaluation-v1','state':'running','tier':args.tier,'source':source,
            'suite_sha256':digest(args.suite),'command':sys.argv,'runs':[],'bounded_stops':[]}
    write_json(out/'suite.json',record);stopped=set()
    try:
        for config,variants in cases:
            key=(config['id'],config['graph'])
            if key in stopped:
                record['bounded_stops'].append({'config':config,'reason':'earlier declared resource stage failed; larger allocation not launched'})
                write_json(out/'suite.json',record);continue
            for variant in variants:
                for repeat in range(args.repeats):
                    name=f"{config['id']}-{config['graph']}-n{config['body_nodes']}-{variant}-r{repeat}"
                    summary=run_variant(root,build,out/name,config,variant,config['modes'],args,source,repeat)
                    record['runs'].append({'directory':name,'config':config,'variant':variant,'repeat':repeat,**summary})
                    write_json(out/'suite.json',record)
                    print(json.dumps({'run':name,'status':summary['status'],'error':summary['error']}),flush=True)
                    if summary['unreaped_child_pid']:raise RuntimeError('child was not reaped; refusing another workload')
                    if summary['status']!='completed' and args.tier=='large':stopped.add(key);break
                    if summary['status']!='completed' and args.tier=='smoke':
                        record['state']='failed';write_json(out/'suite.json',record);return 1
                if key in stopped:break
    except BaseException as error:
        record.update(state='cancelled' if isinstance(error,(SystemExit,KeyboardInterrupt)) else 'failed',
                      error=f'{type(error).__name__}: {error}')
        write_json(out/'suite.json',record)
        raise
    record.update(state='evaluated',completed_runs=sum(r['status']=='completed' for r in record['runs']),
                  failed_runs=sum(r['status']!='completed' for r in record['runs']))
    write_json(out/'suite.json',record)
    # Evaluated means bounded assessment ended; each failed run stays failed.
    return 0


if __name__=='__main__':sys.exit(main())
