#!/usr/bin/env python3
"""Read v4 status and completed evidence; never mutate workload directories."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.job import inspect
from scripts.review_io import write_review_json
from scripts.review_language_results import paired_documents
from tide_lab.records import digest,now
from tide_lab.supervisor import verify_source


def review(root):
    root=Path(root);hashes={}
    def read(path):
        path=root/path;value=json.loads(path.read_text())
        hashes[str(path.relative_to(root))]=digest(path)
        return value
    source=verify_source(root/'.local/source-language-confirm-v4')
    plan=read('configs/studies/language/fc-confirm-v4.json')
    out=dict(schema='tide-language-v4-review-v1',observed_at=now(),source_commit=source['commit'],
        source_tree_sha256=source['content']['tree_sha256'],configuration='configs/studies/language/fc-confirm-v4.json',
        revisions=plan['revisions'],routes={},lens=[],quality_margin=.05,hardware_speedup_required=False,
        checkpoint_inspection='docs/evidence/language-v4-checkpoint-inspection.json',
        scope='snapshot of execution plus completed comparisons; active sparse/scale outcomes remain unchecked')
    used=0.
    for route in 'ab':
        name=f'lm-{route}-language-v4';base=Path('runs')/name
        lifecycle=inspect(root/'artifacts/jobs'/name/'job.json')
        chain=json.loads((root/base/'stage.json').read_text())
        elapsed=lifecycle['state'].get('elapsed_seconds',0.)
        if lifecycle['state']['state']=='running':
            from datetime import datetime
            elapsed=time.time()-datetime.fromisoformat(lifecycle['state']['started_at']).timestamp()
        used+=elapsed
        result=dict(state=chain['state'],lifecycle=lifecycle,chain_snapshot=chain,
            conservative_used_seconds=elapsed,training={},evaluations={},checks={},inference={},comparisons={})
        for step in chain['steps']:
            folder=base/step['name']
            if step['state']!='completed':continue
            row=read(folder/'result.json');manifest=read(folder/'manifest.json')
            if row['state']!='completed' or step['exit_code']!=0:
                raise ValueError('outer and inner completed state disagree: '+str(folder))
            if manifest['identity']['source']['commit']!=source['commit']:
                raise ValueError('unexpected execution source')
            events=[json.loads(line) for line in (root/folder/'metrics.jsonl').read_text().splitlines()]
            projection=root/folder/'projection.json'
            if not manifest['resolved']['probe']:
                p=read(projection)
                if not (p['project']=='tide-learning-lab' and p['source_run_id']==manifest['run_id']
                        and p['status']=='verified' and p['readback_matches'] and p['event_count']==len(events)):
                    raise ValueError('completed metric projection mismatch: '+str(folder))
                out['lens'].append(dict(run_id=manifest['run_id'],record=str(folder/'projection.json'),
                    **{k:p[k] for k in ('destination_name','destination_run_id','status','event_count','readback_matches')}))
            if 'evaluation' in row:result['evaluations'][step['name']]=row['evaluation']
            if step['name']=='unigram':
                result['evaluations']['unigram']=row['evaluations']['test']
                result['unigram_validation_nll']=row['evaluations']['validation']['nll']
            if step['action']=='checks':result['checks'][step['name']]=row
            if step['action']=='inference':result['inference'][step['name']]=row
            if step['action']=='train' and not manifest['resolved']['probe']:
                result['training'][step['name']]={k:row[k] for k in ('steps','schedule_steps','valid_tokens',
                    'initial_validation_nll','final_validation_nll','validation_curve','training_wall_seconds',
                    'inventory','checkpoint_sha256','retained_checkpoints','process_peak_rss_mib')}
                sampled=[r['metrics'] for r in events if 'train/update_seconds' in r['metrics']]
                result['training'][step['name']].update(
                    sampled_update_seconds_mean=statistics.mean(r['train/update_seconds'] for r in sampled),
                    sampled_updates=len(sampled),stage_seconds=step['elapsed_seconds'],
                    npu_peak_allocated_mib=max(r.get('resource/npu_peak_allocated_mib',0) for r in sampled))
        if chain['state']=='running':
            active=next((s for s in reversed(chain['steps']) if s['state']=='running'),None)
            if active:
                result['active_stage']=active
                path=Path(active['output'])/'metrics.jsonl'
                text=path.read_text() if path.exists() else ''
                lines=text.splitlines() if text.endswith('\n') else text.splitlines()[:-1]
                rows=[json.loads(line) for line in lines]
                result['latest_metrics']=rows[-1] if rows else None
                result['live_training_validation']=[r for r in rows if 'validation/nll' in r['metrics']]
                ckpt=path.parent/'checkpoint.json'
                result['latest_checkpoint_metadata']=json.loads(ckpt.read_text()) if ckpt.exists() else None
                p=path.parent/'projection.json'
                result['live_projection']=json.loads(p.read_text()) if p.exists() else None
        ev=result['evaluations']
        if route=='a':
            pairs=[('dense-minus-transformer512','test-tide-dense-512','test-transformer-512'),
                ('dense-minus-extra2048','test-tide-dense-512','test-transformer-extra2048'),
                ('extra2048-minus-unigram','test-transformer-extra2048','unigram'),
                ('sparse-minus-dense','test-tide-sparse-512','test-tide-dense-512')]
            for kind in ('tide-dense','tide-sparse','transformer'):
                pairs.append(('context-reset-minus-'+kind,'context-reset-'+kind,'test-'+kind+'-512'))
            pairs += [('context-reset-minus-extra2048','context-reset-extra2048','test-transformer-extra2048'),
                ('zero-core-minus-dense','zero-core-tide-dense','test-tide-dense-512')]
            for kind in ('tide-dense','tide-sparse'):
                pairs.append(('FC8-minus-FC4-prefix128-'+kind,'test-fc8-'+kind,'test-fc4-prefix128-'+kind))
        else:
            if not lifecycle['termination_verified_now']:raise ValueError('B terminal review found active descendants')
            pairs=[('sparse-minus-dense','test-v3-tide-sparse','test-v3-tide-dense'),
                ('dense-minus-bottleneck','test-v3-tide-dense','test-bottleneck'),
                ('sparse-minus-bottleneck','test-v3-tide-sparse','test-bottleneck'),
                ('bottleneck-minus-mlp','test-bottleneck','test-v3-mlp'),
                ('dense-minus-mlp','test-v3-tide-dense','test-v3-mlp'),
                ('sparse-minus-mlp','test-v3-tide-sparse','test-v3-mlp'),
                ('sparse-minus-original','test-v3-tide-sparse','test-v3-original')]
            for kind in ('bottleneck','v3-tide-dense','v3-tide-sparse','v3-mlp','v3-original'):
                pairs.append((kind+'-minus-frozen','test-'+kind,'test-frozen'))
            dense=ev['test-v3-tide-dense']['work'];sparse=ev['test-v3-tide-sparse']['work']
            result['structure']=dict(dense=dense,sparse=sparse,full_reduction=1-sparse['work/full']/dense['work/full'],
                scope='same v3 weights, new documents; FC4 single region, zero internal messages, candidate/Upd/Read unchanged; not FLOPs or throughput ratio')
        for label,left,right in pairs:
            if left in ev and right in ev:result['comparisons'][label]=paired_documents(ev[left]['rows'],ev[right]['rows'])
        out['routes'][route]=result
    out['budget']=dict(total_npu_seconds=28800,conservative_used_seconds=used,
        conservative_remaining_seconds=28800-used,deadline_epoch=plan['confirmation']['deadline_epoch'],
        a_compute_deadline_epoch=out['routes']['a']['chain_snapshot']['compute_deadline_epoch'],
        old_v1_v2_v3_seconds=11990.574,new_jobs_submitted=False)
    out['input_sha256']=hashes
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,default=Path('docs/evidence/language-v4-status.json'))
    a=p.parse_args();r=review(a.root);write_review_json(a.output,r,root=a.root)
    print('observed',r['observed_at'],'budget',r['budget'],'verified completed projections',len(r['lens']))
    for route,rr in r['routes'].items():
        for name,row in rr['comparisons'].items():
            print(route,name,round(row['delta_nll'],6),'CI95',[round(x,6) for x in row['bootstrap_two_sided_95']],
                  'upper95',round(row['bootstrap_one_sided_95_upper'],6))
