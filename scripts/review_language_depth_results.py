#!/usr/bin/env python3
"""Review terminal v5 records without running models or changing run directories."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.review_io import write_review_json
from scripts.review_language_depth import review as inspect_depth
from scripts.review_language_results import paired_documents
from tide_lab.records import digest, now


def build(root):
    root = Path(root)
    handoff = inspect_depth(root)
    if not all(j['lifecycle']['termination_verified_now'] for j in handoff['jobs'].values()):
        raise ValueError('terminal review requires all recorded jobs and descendants to stop')
    hashes = {}

    def read(path):
        path = root / path
        hashes[str(path.relative_to(root))] = digest(path)
        return json.loads(path.read_text())

    out = dict(schema='tide-language-depth-terminal-v1', observed_at=now(),
        scope='v5 single-seed finite comparison; v4 A completion reviewed separately on its own test documents',
        handoff=handoff, evaluations={}, training={}, inference={}, failed_stages={},
        comparisons={}, structure={}, quality_margin=.05, activity_reduction_margin=.20,
        hardware_speedup_required=False, new_jobs_submitted=False)
    evaluations = {}
    for job_id, job in handoff['jobs'].items():
        for step in job.get('chain', {}).get('steps', []):
            if step.get('action') == 'diagnose':
                # Its separate schema and numerical witness are reviewed in TL-REQ-003.
                continue
            name = step['name']
            folder = Path(step['output']).relative_to(root)
            if not (root/folder/'result.json').exists():
                continue
            row = read(folder/'result.json')
            key = job_id + '/' + name
            if step['state'] != 'completed':
                out['failed_stages'][key] = dict(state=step['state'], error=row.get('error'),
                    record=str(folder/'result.json'))
                continue
            if row['state'] != 'completed' or step['exit_code'] != 0:
                raise ValueError('inconsistent completion: ' + key)
            ev = row.get('evaluation', row.get('evaluations', {}).get('test'))
            if ev:
                evaluations[key] = ev
                out['evaluations'][key] = {k:v for k,v in ev.items() if k != 'rows'}
                out['evaluations'][key].update(checkpoint_step=row.get('checkpoint_step'),
                    checkpoint_sha256=row.get('checkpoint_sha256'),
                    documents=len({r['document_id'] for r in ev['rows']}))
            if step['action'] == 'train' and step.get('category') == 'training':
                training = {k:row.get(k) for k in ('steps','schedule_steps','valid_tokens',
                    'initial_validation_nll','final_validation_nll','validation_curve',
                    'training_wall_seconds','checkpoint_sha256','process_peak_rss_mib')}
                training['inventory'] = {k:v for k,v in row['inventory'].items() if k != 'shared_groups'}
                metrics = root/folder/'metrics.jsonl'
                hashes[str(metrics.relative_to(root))] = digest(metrics)
                events = [json.loads(line) for line in metrics.read_text().splitlines()]
                sampled = [e['metrics'] for e in events if 'train/update_seconds' in e['metrics']]
                training.update(sampled_updates=len(sampled),
                    sampled_update_seconds_mean=statistics.mean(m['train/update_seconds'] for m in sampled),
                    npu_peak_allocated_mib=max(m.get('resource/npu_peak_allocated_mib',0) for m in sampled))
                regions = [e for e in events if any(k.startswith('region/') for k in e['metrics'])]
                training['last_region_observation'] = regions[-1] if regions else None
                meta = read(folder/'checkpoint.json')
                checkpoint = root/folder/'checkpoint.pt'
                actual = digest(checkpoint)
                if actual != meta['sha256'] or actual != row['checkpoint_sha256']:
                    raise ValueError('final checkpoint checksum mismatch: '+key)
                training['checkpoint_file_sha256_verified'] = True
                training['checkpoint_scope'] = 'file integrity only; not decoding or independent long-trajectory recovery'
                out['training'][key] = training
            if step['action'] == 'inference':
                out['inference'][key] = {k:v for k,v in row.items() if k not in
                    ('generated_token_ids','prefix_token_ids')}
    for mapping in handoff['lens']:
        path = Path(mapping['projection'])
        p = read(path)
        manifest = read(path.parent/'manifest.json')
        metrics = root/path.parent/'metrics.jsonl'
        hashes[str(metrics.relative_to(root))] = digest(metrics)
        count = len(metrics.read_text().splitlines())
        if not (p['project'] == 'tide-learning-lab' and p['source_run_id'] == manifest['run_id']
                and p['status'] == 'verified' and p['readback_matches'] and p['event_count'] == count):
            raise ValueError('projection identity or event count mismatch: '+str(path))
    out['projection_count_verified'] = len(handoff['lens'])
    a = 'lm-a-reference-v5/'
    b = 'lm-b-reference-v5/'
    hb = 'lm-a-hb16-sparse-v5/'
    fc = 'lm-a-fc16-v5/'
    tf = 'lm-a-transformer-cpu-v5/'
    bf = 'lm-b-fc8-v5-budget2/'
    old = 'lm-a-language-v4/'
    pairs = [
        ('A-fc16-dense-minus-fc4-dense', fc+'test-tide-dense', a+'test-fc4-tide-dense-prefix128'),
        ('A-hb16-sparse-minus-fc4-sparse', hb+'test-tide-sparse', a+'test-fc4-tide-sparse-prefix128'),
        ('A-hb16-sparse-minus-fc16-dense', hb+'test-tide-sparse', fc+'test-tide-dense'),
        ('A-fc16-dense-minus-transformer128', fc+'test-tide-dense', tf+'test-transformer-prefix128'),
        ('A-hb16-sparse-minus-transformer128', hb+'test-tide-sparse', tf+'test-transformer-prefix128'),
        ('A-fc16-dense-minus-transformer2048', fc+'test-tide-dense', tf+'test-transformer'),
        ('A-hb16-sparse-minus-transformer2048', hb+'test-tide-sparse', tf+'test-transformer'),
        ('A-hb16-reset-minus-full', hb+'context-reset-tide-sparse', hb+'test-tide-sparse'),
        ('A-transformer-reset-minus-full', tf+'context-reset-transformer', tf+'test-transformer'),
        ('A-hb16-128-minus-64', hb+'test-tide-sparse', hb+'test-tide-sparse-prefix64'),
        ('B-fc8-sparse-minus-dense', bf+'test-tide-sparse', bf+'test-tide-dense'),
        ('v4-A-sparse-minus-dense', old+'test-tide-sparse-512', old+'test-tide-dense-512'),
        ('v4-A-sparse-minus-transformer2048', old+'test-tide-sparse-512', old+'test-transformer-extra2048'),
        ('v4-A-sparse-reset-minus-full', old+'context-reset-tide-sparse', old+'test-tide-sparse-512'),
        ('v4-A-sparse-zero-minus-full', old+'zero-core-tide-sparse', old+'test-tide-sparse-512')]
    for kind in ('dense','sparse'):
        pairs.append(('B-fc8-'+kind+'-minus-fc4', bf+'test-tide-'+kind,b+'test-v3-tide-'+kind))
        for control in ('frozen','v3-mlp','v3-original','v4-bottleneck'):
            pairs.append(('B-fc8-'+kind+'-minus-'+control,bf+'test-tide-'+kind,b+'test-'+control))
    for label,left,right in pairs:
        out['comparisons'][label] = dict(candidate=left, reference=right,
            **paired_documents(evaluations[left]['rows'], evaluations[right]['rows']))
    for label,prefix,dense_name,sparse_name in [
            ('B-FC8',bf,'test-tide-dense','test-tide-sparse'),
            ('v4-A-FC4',old,'test-tide-dense-512','test-tide-sparse-512')]:
        d = evaluations[prefix+dense_name]['work']
        s = evaluations[prefix+sparse_name]['work']
        out['structure'][label] = dict(dense=d,sparse=s,
            full_reduction=1-s['work/full']/d['work/full'],
            message_reduction=1-s['work/messages']/d['work/messages'] if d['work/messages'] else None,
            scope='same topology and heldout targets; Full and message counts, not total FLOPs or hardware speedup')
    work = evaluations[hb+'test-tide-sparse']['work']
    out['structure']['A-HB16-sparse'] = dict(work=work,
        full_fraction=work['work/full']/work['work/candidates'],
        scope='observed Full/candidate ratio only; no completed matched HB16 dense128 evaluation')
    out['input_sha256'] = hashes
    out['review_scope'] = 'reproducible statistics; scientific interpretation is in the reviewed reports, never inferred from exit codes'
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,default=Path('docs/evidence/language-depth-terminal.json'))
    args = p.parse_args()
    result = build(args.root)
    write_review_json(args.output,result,root=args.root)
    for name,row in result['comparisons'].items():
        print(name,round(row['delta_nll'],6),
              [round(v,6) for v in row['bootstrap_two_sided_95']],
              'upper95',round(row['bootstrap_one_sided_95_upper'],6))
    print('verified projections',result['projection_count_verified'])
