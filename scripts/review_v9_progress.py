#!/usr/bin/env python3
"""Resolve original/recovery ancestry and summarize existing validation, without execution."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.review_direct_v9 import read,review
from scripts.review_io import write_review_json
from tide_lab.records import digest


def summarize(root):
    observation=review(root);original=observation['campaign']
    cases={c['id']:c for c in original['phases']['main']['cases']}
    for recovery in observation['recovery_campaigns']:
        cases.update({c['id']:c for c in recovery.get('resolved_main_cases',[])})
    rows=[];groups=defaultdict(list)
    for ident,case in cases.items():
        output=Path(case['output']);train=output/'train';result=read(train/'result.json')
        progress=read(train/'progress.json');val=read(train/'validation-final.json');cfg=read(Path(case['config']))
        state=result.get('state','not_started')
        if state=='not_started' and output.exists():state='preparing'
        work=val.get('work',{});den=work.get('work/body_node_slots');all_den=work.get('work/node_slots')
        row=dict(id=ident,study=cfg['study_id'],state=state,config=case['config'],config_sha256=digest(case['config']),
            output=str(output),tokens=progress.get('tokens'),target_tokens=int(case['train_arguments'][1]),
            source_commit=result.get('identity',{}).get('source_commit'),
            validation=dict(state=val.get('state','unchecked'),nll=val.get('nll'),tokens=val.get('tokens'),
                record=str(train/'validation-final.json'),sha256=digest(train/'validation-final.json') if val else None,
                body_full=work.get('work/body_full',0)/den if den else None,
                body_receipt=work.get('work/body_candidates',0)/den if den else None,
                whole_full=work.get('work/full',0)/all_den if all_den else None,
                physical_full=work.get('work/physical_full_rows',0)/all_den if all_den else None),
            last_metrics=progress.get('last_update'),recovery=read(output/'recovery.json').get('state'),
            qualifications_reused={k:case.get(k) for k in ('reuse_admission','reuse_recovery')},
            lens=read(train/'projection.json'),axes=result.get('axes'),error=result.get('error'))
        rows.append(row)
        if state=='completed' and val.get('state')=='completed':groups[ident.rsplit('-s',1)[0]].append(row)
    means=[]
    for name,items in groups.items():
        mean=lambda k:statistics.mean(r['validation'][k] for r in items) if items[0]['validation'][k] is not None else None
        means.append(dict(name=name,seeds=len(items),ids=[r['id'] for r in items],
            training_tokens=items[0]['target_tokens'],validation_tokens=items[0]['validation']['tokens'],
            **{k:mean(k) for k in ('nll','body_full','body_receipt','whole_full','physical_full')}))
    return dict(schema='tide-v9-progress-review-v1',observed_at=observation['observed_at'],
        counts=dict(Counter(r['state'] for r in rows)),
        by_study={g:dict(Counter(r['state'] for r in rows if r['study']==g)) for g in sorted({r['study'] for r in rows})},
        original_controller={k:original.get(k) for k in ('state','error','finished_at','deadline_epoch')},
        recovery_campaigns=observation['recovery_campaigns'],jobs=observation['jobs'],cases=rows,group_means=means,
        scope='Existing same131072-target validation used for selection; no new evaluation or scientific-support automation. Original failures preserved; main trajectories resolve to explicit recovery descendants.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--write',type=Path);a=p.parse_args();result=summarize(a.root)
    if a.write:write_review_json(a.write,result)
    print(json.dumps({k:result[k] for k in ('observed_at','counts','by_study','original_controller','group_means')},ensure_ascii=False,indent=2))
