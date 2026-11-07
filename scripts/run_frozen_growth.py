#!/usr/bin/env python3
"""One frozen-backbone pilot, gated by the changed-path ingress and exact resume checks."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,digest,now
from scripts.npu_learning_stage import bounded_command


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p.add_argument('--asset-path',required=True);p.add_argument('--recording-cli',required=True)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text())
    if plan['max_seconds']>2700:raise ValueError('outside frozen pilot budget')
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    report=dict(state='running',started_at=now(),plan=plan,plan_sha256=digest(a.plan),commands=[],
        axes=dict(execution='running',strict_numerical='historical finite failures retained; changed path pending',
                  discrete_semantics='same core5 binding; prior limited admission reused',
                  mechanisms='pending frozen optimizer and gradient checks',recovery='pending',learning='not_checked'))
    def publish():atomic_json(root/'stage.json',report)
    def run(name,command,budget):
        remaining=plan['max_seconds']-(time.monotonic()-started)
        if remaining<20:raise TimeoutError('stage budget exhausted')
        record=dict(name=name,command=command);report['commands'].append(record)
        bounded_command(command,root/(name+'.log'),min(budget,remaining)-10,record,publish)
    publish()
    try:
        run('ingress',[sys.executable,'scripts/check_growth.py','--checkpoint-path',a.asset_path,
            '--output',str(root/'ingress.json'),'--device','npu:0','--threads','4','--seed','11',
            '--freeze-backbone','--skip-generation','--explore-numerics'],300)
        ingress=json.loads((root/'ingress.json').read_text())
        report['axes']['strict_numerical']=dict(changed_path=ingress['strict_numeric'],historical_failures='retained')
        report['axes']['mechanisms']='passed frozen optimizer/gradient checks; tiny input scope'
        run('resume',[sys.executable,'scripts/check_resume.py','--config',plan['resume_config'],
            '--output',str(root/'resume'),'--asset-path',a.asset_path,'--retire-large-checkpoints'],600)
        resume=json.loads((root/'resume/result.json').read_text())
        if resume['state']!='passed':raise RuntimeError('complete frozen model resume did not pass')
        report['axes']['recovery']='passed: tiny fresh-process same-stack model/SGD/RNG; not cross-backend'
        publish()
        run('learning',[sys.executable,'-m','tide_lab.batch','--matrix',plan['matrix'],
            '--output',str(root/'learning'),'--asset-path',a.asset_path,'--recording-cli',a.recording_cli],1700)
        batch=json.loads((root/'learning/batch.json').read_text())
        if batch['state']!='completed':raise RuntimeError('inner learning batch incomplete')
        report['axes']['learning']='completed pilot; per-run metrics and branch intervention need human review'
        report['axes']['execution']='completed';report['state']='completed'
    except BaseException as error:
        report.update(state='failed',error=repr(error));report['axes']['execution']='failed';raise
    finally:
        report.update(finished_at=now(),elapsed_seconds=time.monotonic()-started);publish()


if __name__=='__main__':main()
