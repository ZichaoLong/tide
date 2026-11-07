"""Sequential finite runs with a separate, explicitly selected JSONL viewer writer."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .records import atomic_json, digest, now


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--asset-path')
    p.add_argument('--recording-cli',required=True)
    a=p.parse_args()
    matrix=json.loads(Path(a.matrix).read_text())
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
    if not 1<=len(matrix['runs'])<=12 or not 0<matrix['max_seconds']<=14400:
        raise ValueError('Matrix exceeds first-batch finite limits')
    state=dict(schema='tide-lab-batch-v1',matrix_sha256=digest(a.matrix),started_at=now(),state='running',runs=[])
    atomic_json(root/'batch.json',state)
    started=time.monotonic();worker=recorder=None
    try:
        for item in matrix['runs']:
            if time.monotonic()-started>=matrix['max_seconds']:
                state['state']='budget_exhausted';break
            directory=root/item['run_id']
            command=[sys.executable,'-m','tide_lab.train','--config',item['config'],'--output',str(directory),
                     '--run-id',item['run_id']]
            if a.asset_path: command+=['--asset-path',a.asset_path]
            if item.get('resume'): command+=['--resume',item['resume']]
            worker=subprocess.Popen(command)
            # The recorder lives outside the training interpreter and its dependency graph.
            # No HTTP token or Lens import enters the model code.
            while worker.poll() is None and not (directory/'metrics.jsonl').exists():
                time.sleep(.1)
            recorder_log=(root/(item['run_id']+'-recording.log')).open('x')
            recorder=None
            if (directory/'metrics.jsonl').exists():
                recorder=subprocess.Popen([a.recording_cli,'record','--events',str(directory/'metrics.jsonl'),
                    '--manifest',str(directory/'manifest.json'),'--report',str(directory/'projection.json'),
                    '--follow','--done-file',str(directory/'completed.json')],stdout=recorder_log,stderr=subprocess.STDOUT,
                    env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','VIRTUAL_ENV')})
            remaining=max(1,matrix['max_seconds']-(time.monotonic()-started))
            try:
                code=worker.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                worker.terminate()
                try: worker.wait(timeout=30)
                except subprocess.TimeoutExpired: worker.kill();worker.wait()
                code=124
            if not (directory/'completed.json').exists() and directory.exists():
                atomic_json(directory/'completed.json',dict(producer_finished=True,state='failed',exit_code=code))
            recording_code=None
            if recorder is not None:
                try: recording_code=recorder.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    recorder.terminate()
                    try: recorder.wait(timeout=10)
                    except subprocess.TimeoutExpired: recorder.kill();recorder.wait()
                    recording_code=124
            recorder_log.close()
            result=json.loads((directory/'result.json').read_text()) if (directory/'result.json').exists() else {}
            state['runs'].append(dict(run_id=item['run_id'],config=item['config'],exit_code=code,
                state=result.get('state','failed'),recording_exit_code=recording_code,
                result=str(directory/'result.json'),projection=str(directory/'projection.json')))
            atomic_json(root/'batch.json',state)
            if code and matrix.get('stop_on_failure',True):
                state['state']='failed';break
        if state['state']=='running':
            state['state']='completed' if all(r['exit_code']==0 for r in state['runs']) else 'failed'
    finally:
        for child in (worker,recorder):
            if child is not None and child.poll() is None:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill();child.wait()
        if state['state']=='running': state['state']='interrupted'
        state.update(finished_at=now(),elapsed_seconds=time.monotonic()-started)
        atomic_json(root/'batch.json',state)
    return 0 if state['state']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
