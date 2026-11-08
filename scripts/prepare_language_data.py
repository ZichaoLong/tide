#!/usr/bin/env python3
"""Bounded CPU data preparation with fixed raw files and a separate Arrow reader."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import Metrics,atomic_json,digest,now,project_to_lens


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('raw','output','extract-python','asset-path','recording-cli'):
        p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    source=json.loads((Path(__file__).resolve().parents[1]/'.tide-source.json').read_text())
    run_id='tl-lm0-data-v1'
    manifest=dict(schema='tide-lab-run-v1',project='tide-learning-lab',study_id='TL-LM0',group='TL-LM0',
        run_id=run_id,name=run_id,created_at=now(),config=dict(raw_lock_sha256=digest(a.raw/'source.json'),
        source_commit=source['commit'],vocab_size_a=4096,context=128,validation_documents=8,test_documents=16))
    atomic_json(a.output/'manifest.json',manifest);metrics=Metrics(a.output,run_id);result=dict(state='running')
    atomic_json(a.output/'state.json',result)
    try:
        subprocess.run([str(a.extract_python),'-m','tide_lab.language_data','extract','--input',str(a.raw),
            '--output',str(a.output/'documents')],check=True,timeout=120)
        subprocess.run([sys.executable,'-m','tide_lab.language_data','tokenize','--input',str(a.output/'documents'),
            '--output',str(a.output/'tokens'),'--asset-path',str(a.asset_path)],check=True,timeout=450)
        lock=json.loads((a.output/'tokens/data-lock.json').read_text())
        for i,(route,r) in enumerate(lock['routes'].items()):
            metrics.log(i,{route+'/'+s+'/valid_tokens':v['valid_tokens'] for s,v in r['splits'].items()})
        result.update(state='completed',data_lock_sha256=digest(a.output/'tokens/data-lock.json'),
            routes=lock['routes'],document_counts=lock['source']['article_counts'])
        for path in (a.output/'tokens').rglob('*'):
            if path.is_file():path.chmod(0o444)
    except BaseException as error:
        result.update(state='failed',error=repr(error),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        result.update(finished_at=now(),elapsed_seconds=time.monotonic()-started)
        atomic_json(a.output/'result.json',result);atomic_json(a.output/'state.json',result)
        metrics.close();atomic_json(a.output/'completed.json',dict(producer_finished=True,state=result['state']))
        result['lens_projection_completed']=project_to_lens(a.output,str(a.recording_cli))
        atomic_json(a.output/'result.json',result)
    return 0 if result['state']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
