#!/usr/bin/env python3
"""Read-only v6 HB/FC learning review; never evaluates models or opens test."""
import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.review_io import write_review_json
from tide_lab.records import digest, now


def read(p):
    return json.loads(p.read_text())


def records(p):
    return {(r['block_id'], r['tokens']): r for r in read(p)['records']}


def paired(left, right):
    a, b = records(left), records(right)
    common = sorted(a.keys() & b.keys()); docs = {}
    for key in common:
        row = a[key]; item = docs.setdefault(row['document_id'], [0., 0., 0])
        item[0] += row['nll_sum']; item[1] += b[key]['nll_sum']; item[2] += row['tokens']
    values = list(docs.values()); total = sum(r[2] for r in values)
    if not total:
        raise ValueError('no identical evaluation blocks')
    delta = (sum(r[0]-r[1] for r in values))/total
    rng = random.Random(711); draws = []
    for _ in range(2000):
        sample = [rng.choice(values) for _ in values]
        draws.append(sum(r[0]-r[1] for r in sample)/sum(r[2] for r in sample))
    draws.sort()
    return dict(left=str(left), right=str(right), matched_blocks=len(common),
        documents=len(values), tokens=total, complete_same_blocks=a.keys()==b.keys(),
        left_nll=sum(r[0] for r in values)/total, right_nll=sum(r[1] for r in values)/total,
        delta_left_minus_right=delta, bootstrap_document_ci95=[draws[49], draws[1949]],
        scope='validation only; paired document bootstrap conditional on one seed/checkpoints; not seed or model-selection uncertainty')


def main(a):
    root=a.root.resolve(); scale=[]; coverage=[]
    for p in sorted((root/'runs').glob('v6-scale-*/train/metrics.jsonl')):
        run=p.parent
        manifest=read(run/'manifest.json'); config=manifest['config']
        result=read(run/'result.json') if (run/'result.json').exists() else {}
        progress=read(run/'progress.json') if (run/'progress.json').exists() else {}
        raw=[json.loads(line) for line in p.read_text().splitlines()]
        curves=[dict(step=r['step'], **r['metrics']) for r in raw]
        initial=run/'validation-initial.json'; milestone=run/'validation-262144.json'
        validated=[]
        for v in sorted(run.glob('validation-*.json')):
            d=read(v)
            validated.append(dict(path=str(v.relative_to(root)), sha256=digest(v),
                tokens=d['tokens'], target_tokens=d['target_tokens'], nll=d['nll'],
                state=d['state'], progress=d.get('progress'), test_access=d['test_access']))
            if d['test_access']:
                raise ValueError('unexpected test evidence')
        training=[r['metrics'] for r in raw if 'train/loss' in r['metrics']]
        owner=root/'artifacts/jobs'/run.parent.name/'state.json'
        item=dict(id=config['id'], run=str(run.relative_to(root)), source=manifest['identity'],
            config_path=str(run.relative_to(root)/'manifest.json'), config_sha256=digest(run/'manifest.json'),
            metrics_sha256=digest(p), inner_state=result.get('state'),
            outer_lifecycle=read(owner) if owner.exists() else None,
            parameters=result.get('inventory',{}).get('trainable_parameters'),
            progress=progress, recipe=config.get('core',{}).get('recipe'),
            training=config['training'], validations=validated, curves=curves,
            first_training=training[0] if training else None,
            last_logged_training=training[-1] if training else None,
            max_logged_gradient_norm=max([r['train/gradient_norm'] for r in training],default=None))
        if initial.exists() and milestone.exists():
            item['matched_initial_vs_262k']=paired(initial,milestone)
        scale.append(item)
    for p in sorted((root/'runs').glob('v6-cover-*/train/result.json')):
        d=read(p);v=p.parent/'validation-final.json'
        coverage.append(dict(id=d['config_id'], path=str(p.relative_to(root)),sha256=digest(p),
            state=d['state'], tokens=d.get('progress',{}).get('tokens'),
            validation_tokens=read(v)['tokens'] if v.exists() else None,
            final_nll=d.get('final_validation_nll'),parameters=d.get('inventory',{}).get('trainable_parameters')))
    hb88=root/'runs/v6-scale-hb88-kv-quad-memory2/train/validation-262144.json'
    comparisons=[]
    for name in ('hb40-kv-quad-retry1','hb40-shannon-retry1','fc16-kv-quad-retry1',
                 'fc16-gdn-quad-retry1','fc32-kv-quad-retry1','transformer-10m-retry1'):
        comparisons.append(paired(hb88,root/f'runs/v6-scale-{name}/train/validation-262144.json'))
    hb40=root/'runs/v6-scale-hb40-kv-quad-retry1/train'
    result=dict(schema='tide-hb-learning-review-v1',at=now(),state='reviewed_with_limits',
        scope='Original d32 coverage and d128 scale kept separate. Validation only; no new training/evaluation/test access. Raw running inner records preserved alongside outer lifecycle.',
        scale=scale,coverage=coverage,comparisons_at_262k=comparisons,
        hb40_133k_to_658k=paired(hb40/'validation-133571.json',hb40/'validation-658436.json'),
        hb88_133k_to_262k=paired(hb88.with_name('validation-133571.json'),hb88),
        caveats=['One seed per candidate; no architecture-wide scaling verdict.',
                 'd32/L32/65K candidate screen does not isolate capacity against d128/L128 scale.',
                 'Initial 2048-token validation and later 32768-token validation differ; matched intersection reported.',
                 'Gradient norm is before global clip; AdamW is not SGD, so clip factor is not an effective learning-rate estimate.',
                 'Finite numerical tolerance failures and untested complete trajectory resume remain separate.',
                 'Transformer controls also have prolonged plateaus and nonmonotone capacity ordering.'])
    write_review_json(a.output,result,root=root)
    for item in comparisons:
        print(Path(item['right']).parent.parent.name,'HB88-minus NLL',round(item['delta_left_minus_right'],4),
              'tokens',item['tokens'],'CI',item['bootstrap_document_ci95'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
