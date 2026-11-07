#!/usr/bin/env python3
"""Wait inside a finite service, validate the frozen HB preparation, then exec one pilot."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from tide_lab.records import atomic_json,digest,now


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation',required=True,type=Path)
    parser.add_argument('--preparation-job',required=True,type=Path)
    parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--send',required=True,choices=['dense','sparse'])
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--recording-cli',required=True)
    args=parser.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    state=dict(schema='tide-lab-dependent-pilot-v1',state='waiting_dependency',started_at=now(),
               prerequisite=str(args.preparation),variant=args.send,wait_budget_seconds=1200)
    start=time.monotonic()
    def publish():
        state['wait_elapsed_seconds']=time.monotonic()-start;atomic_json(root/'dependency.json',state)
    publish()
    try:
        while True:
            lifecycle=json.loads((args.preparation_job/'state.json').read_text())
            if lifecycle['state'] in ['failed','timed_out','stopped','interrupted']:
                raise RuntimeError('preparation failed; paired learning admission is not established')
            if args.preparation.exists():
                preparation=json.loads(args.preparation.read_text())
                if preparation['state']=='failed': raise RuntimeError('preparation retained a failure')
                if preparation['state']=='completed' and lifecycle['state']=='completed' and lifecycle.get('termination_verified'):
                    break
            if time.monotonic()-start>=1200: raise TimeoutError('bounded prerequisite wait exhausted')
            time.sleep(2);publish()
        spec=json.loads((args.preparation_job/'job.json').read_text())
        if spec['source']['commit']!='aff03eee27960113694c7c3d3b7a5c4cc82f8434':
            raise ValueError('unexpected preparation source identity')
        if preparation['plan_sha256']!=digest(args.plan): raise ValueError('preparation plan differs')
        if not preparation['selection']['eligible']: raise ValueError('chosen common batch is ineligible')
        for send in ['dense','sparse']:
            axes=preparation['admission'][send]['axes']
            if any(axes[k]['state']!='passed' for k in ['discrete_semantics','mechanisms','recovery']):
                raise ValueError('paired mechanism/discrete/recovery gate failed')
        resolved=next(r for r in preparation['resolved'] if r['send']==args.send)
        config=Path(resolved['config'])
        if digest(config)!=resolved['sha256'] or config.stat().st_mode & 0o222:
            raise ValueError('resolved config differs or is not frozen read-only')
        c=json.loads(config.read_text())
        if (c['device']!='cpu' or c['dtype']!='float32' or c['threads']!=1 or c['seed']!=11
            or c['model']['topology']!='hb' or c['model']['send']!=args.send
            or c['training']['steps']!=80 or c['training']['max_seconds']>2400
            or c['training']['batch_size']!=preparation['selection']['batch']):
            raise ValueError('resolved run exceeds the admitted paired pilot')
        run_id=f'tl-a2-hb-{args.send}-s11-pilot'
        matrix=root/'matrix.json'
        atomic_json(matrix,dict(schema='tide-lab-matrix-v1',study_id='TL-A2',max_seconds=2700,
                               stop_on_failure=True,runs=[dict(run_id=run_id,config=str(config))]))
        matrix.chmod(0o444)
        state.update(state='admitted',admitted_at=now(),preparation_sha256=digest(args.preparation),
                     config_sha256=digest(config),run_id=run_id,strict_numerical=c['evidence_axes']['strict_numerical']['state'])
        publish()
        # Same frozen train/batch package as admission. The supervisor retains
        # the entire process group; exec avoids an orphan wrapper on stop.
        os.execv(sys.executable,[sys.executable,'-m','tide_lab.batch','--matrix',str(matrix),
                 '--output',str(root/'learning'),'--recording-cli',args.recording_cli])
    except BaseException as error:
        state.update(state='failed',error=repr(error),finished_at=now());publish()
        print(json.dumps(state),flush=True);return 1


if __name__=='__main__': raise SystemExit(main())
