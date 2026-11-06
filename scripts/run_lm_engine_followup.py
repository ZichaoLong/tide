#!/usr/bin/env python3
"""Finite engine/placement continuation with observable dependency waits."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
from tidegraph.lm.assets import ready
from tidegraph.lm.engine_matrix import manifest,ASCEND_REVISION
from tidegraph.lm.selection import validate
from tidegraph.lm.queue import compute_lock,execute,placeholder
from tidegraph.lm.report import save_all
from tidegraph.experiment.records import write_json,read_json,fingerprint,now

TERMINAL={'passed','failed','cancelled','timeout','timed_out'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--kind',choices=('llama','placement'),required=True);p.add_argument('--attempt',required=True)
    p.add_argument('--manifest',type=Path,help='explicit subset; does not add old completed cases')
    p.add_argument('--small-job',default='assets-small-d');p.add_argument('--moe-job',default='assets-moe-d')
    p.add_argument('--network-job',default='assets-small-d',help='empty string allows independent direct source download')
    p.add_argument('--npu-helper',type=Path)
    p.add_argument('--previous-build',type=Path,help='require exact tiny C++/state regression against this build before costs')
    args=p.parse_args();root=args.root.resolve();out=root/'recovery'/args.attempt;out.mkdir(exist_ok=False)
    value=validate(args.manifest) if args.manifest else manifest(args.kind)
    if any((x['config']['execution']['engine']=='llama_cpp') != (args.kind=='llama') for x in value['cases']):
        p.error('manifest engine differs from selected continuation kind')
    keys=list(dict.fromkeys(x['config']['model'] for x in value['cases']))
    state=dict(state='running',kind=args.kind,started=now(),models={},phase='preparing')
    def save():write_json(out/'status.json',state)
    def terminal(job):
        path=root/'jobs'/job/'status.json'
        return path.exists() and read_json(path).get('state') in TERMINAL
    def reject(cases,key,reason):
        base=root/'runs'/f'{args.kind}-{key}-{args.attempt}-unrun';base.mkdir()
        for case in cases:
            d=base/case['name'];d.mkdir();write_json(d/'case.json',case);placeholder(case,d,'dependency_failed',reason)
        save_all(base)
    begin=time.monotonic();save()
    try:
        if args.kind=='llama':
            state['phase']='waiting_network_slot_after_small_asset_service';save()
            # No third large download stream alongside the two model services.
            while args.network_job and not terminal(args.network_job) and time.monotonic()-begin<30*3600:time.sleep(30)
            if args.network_job and not terminal(args.network_job):raise TimeoutError('30h network-slot dependency limit')
            state['phase']='waiting_compute_lock_for_build';save()
            with compute_lock(root/'compute.lock',21600):
                state['phase']='building';save()
                # Network proxy environment comes from the service's netpath wrapper.
                subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('prepare_lm_llama.py')),str(root)],check=True,timeout=7200)
                if args.previous_build:
                    state['phase']='progress_regression';save()
                    subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('verify_lm_llama_progress.py')),str(root),
                        '--previous-build',str(args.previous_build.resolve()),'--output',str(out/'progress-regression')],check=True,timeout=1200)
        else:
            from ascend_environment import inspect
            state['vllm_audit']=dict(historical_engine_revision=ASCEND_REVISION,hardware='A3',
                public_baseline=inspect(require_venv=True),
                decision='this continuation measures Transformers only; a separately published vLLM service stack is not selected or qualified here',
                restart='explicit separate engine configuration and acceptance; do not mix vLLM dependencies into this environment')
            save()
        for key in keys:
            cases=[x for x in value['cases'] if x['config']['model']==key]
            state.update(phase='waiting_asset',current=key);save()
            job=args.moe_job if key=='qwenmoe' else args.small_job
            while not ready(key,root/'models') and not terminal(job) and time.monotonic()-begin<30*3600:time.sleep(30)
            if not ready(key,root/'models'):
                reject(cases,key,'asset failed or finite dependency deadline; no model execution')
                state['models'][key]='dependency_failed';save();continue
            if args.kind=='llama':
                state['phase']='waiting_compute_lock_for_conversion';save()
                try:
                    with compute_lock(root/'compute.lock',21600):
                        state['phase']='converting';save()
                        subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('prepare_lm_llama.py')),str(root),'--convert',key],check=True,timeout=3600)
                except Exception as error:
                    reject(cases,key,'conversion failed: '+str(error)[:1000]);state['models'][key]='conversion_failed';save();continue
            # Checks first. A finite numerical disagreement may proceed; runtime/nonfinite failures may not.
            for phase in ('check','run'):
                selected=[x for x in cases if (x['action']=='run')==(phase=='run')]
                if not selected:continue
                path=out/(key+'-'+phase+'.json')
                write_json(path,{**value,'cases':selected,'identity':fingerprint(selected)})
                state['phase']=phase;save()
                backend='cpu' if args.kind=='llama' else 'npu';attempt=f'{args.kind}-{key}-{phase}-{args.attempt}'
                result=execute(root,backend,selection=path,attempt=attempt,hours=6,
                               npu_helper=str(args.npu_helper or root/'site/run_on_free_npu.py') if backend=='npu' else None)
                if phase=='check':
                    base=root/'runs'/(backend+'-'+attempt)
                    complete=all((base/x['name']/'run/check.json').exists() and
                                 read_json(base/x['name']/'run/check.json').get('state')=='completed' for x in selected)
                    if not complete:
                        reject([x for x in cases if x['action']=='run'],key,'small engine/placement check did not execute completely')
                        state['models'][key]='check_execution_failed';save();break
                state['models'][key]=dict(last_phase=phase,queue_state=result['state']);save()
        state.update(state='completed',phase='terminal',finished=now());save()
    except BaseException as error:
        state.update(state='failed',error=str(error)[:2000],finished=now());save()
        if args.kind=='llama' and state.get('phase') in {'building','progress_regression','waiting_compute_lock_for_build','waiting_network_slot_after_small_asset_service'}:
            for key in keys:reject([x for x in value['cases'] if x['config']['model']==key],key,'engine preparation failed: '+str(error)[:1000])
        raise


if __name__=='__main__':main()
