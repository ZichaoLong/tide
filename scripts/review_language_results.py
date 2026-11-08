#!/usr/bin/env python3
"""Read fixed language results; write statistical review only under docs/evidence."""
import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.job import inspect
from scripts.review_io import write_review_json
from tide_lab.records import digest, now


def paired_documents(candidate, reference, *, draws=20000, seed=20261109):
    """Candidate minus reference, resampling whole documents with token weights."""
    def keyed(rows):
        result = {}
        for row in rows:
            key = (row['document_id'], row['block_id'])
            if key in result or row['tokens'] <= 0 or not math.isfinite(row['nll_sum']):
                raise ValueError('invalid or repeated heldout block')
            result[key] = row
        return result
    left, right = keyed(candidate), keyed(reference)
    if not left or left.keys() != right.keys():
        raise ValueError('paired evaluation must use exactly the same document/block targets')
    docs = {}
    for key in sorted(left):
        a, b = left[key], right[key]
        if a['tokens'] != b['tokens']:
            raise ValueError('paired target counts differ')
        delta, tokens = docs.get(key[0], (0., 0))
        docs[key[0]] = (delta + a['nll_sum'] - b['nll_sum'], tokens + a['tokens'])
    rows = list(docs.values()); point = sum(r[0] for r in rows) / sum(r[1] for r in rows)
    rng = random.Random(seed); samples = []
    for _ in range(draws):
        chosen = rng.choices(rows, k=len(rows))
        samples.append(sum(r[0] for r in chosen) / sum(r[1] for r in chosen))
    samples.sort()
    def quantile(q):
        position = q * (len(samples) - 1); lo = math.floor(position); hi = math.ceil(position)
        return samples[lo] + (position - lo) * (samples[hi] - samples[lo])
    return dict(delta_nll=point, documents=len(rows), tokens=sum(r[1] for r in rows),
        bootstrap_two_sided_95=[quantile(.025), quantile(.975)],
        bootstrap_one_sided_95_upper=quantile(.95), draws=draws, seed=seed,
        resampling='whole paired documents, token-weighted ratio per draw',
        scope='descriptive uncertainty conditional on these documents, checkpoint and single training seed; no multiplicity adjustment or seed/generalization guarantee')


def build(root):
    root = Path(root); hashes = {}
    def read(path):
        path = root / path; hashes[str(path.relative_to(root))] = digest(path)
        return json.loads(path.read_text())
    result = dict(schema='tide-language-terminal-review-v1',observed_at=now(),
        source_commit='6cd4f856df02e9c90645198cf5998a27a877b4ec',
        mathematics='0770593838a37c790a568226930567ed60f99b83',
        foundation='85d161d5c9148eac2b164f89a0ba4204b4331898',
        configuration='configs/studies/language/fc-explore-v3.json',
        jobs=[],routes={},lens=[],quality_margin=.05,hardware_speedup_required=False,
        checkpoint_inspection=read('docs/evidence/language-v3-checkpoint-inspection.json'))
    cumulative = 0.
    for route in 'ab':
        for version in (1, 2, 3):
            name = f'lm-{route}-language-v{version}'
            lifecycle = read(f'artifacts/jobs/{name}/state.json')
            observed = inspect(root / f'artifacts/jobs/{name}/job.json')
            if not observed['termination_verified_now']:
                raise ValueError('terminal review found live or unverified descendants')
            cumulative += lifecycle['elapsed_seconds']
            result['jobs'].append(dict(job_id=name,state=lifecycle['state'],ended_at=lifecycle['ended_at'],
                conservative_seconds=lifecycle['elapsed_seconds'],termination_verified=True))
        base = Path('runs') / f'lm-{route}-language-v3'; stage = read(base / 'stage.json')
        route_result = dict(state=stage['state'],spent_seconds=stage['spent_seconds'],evaluations={},training={},inference={},comparisons={})
        for step in stage['steps']:
            name = step['name']; folder = base / name
            row = read(folder / 'result.json')
            if row['state'] != 'completed' or step['exit_code'] != 0:
                raise ValueError('inner stage failure: ' + name)
            manifest = read(folder / 'manifest.json')
            projection = read(folder / 'projection.json')
            events = (root / folder / 'metrics.jsonl').read_text().splitlines()
            if (projection['project'] != 'tide-learning-lab' or projection['source_run_id'] != manifest['run_id']
                    or projection['status'] != 'verified' or not projection['readback_matches']
                    or projection['event_count'] != len(events)):
                raise ValueError('incomplete/misdirected metric projection')
            result['lens'].append(dict(run_id=manifest['run_id'],record=str(folder / 'projection.json'),
                destination_name=projection['destination_name'],destination_run_id=projection['destination_run_id'],
                status=projection['status'],readback_matches=projection['readback_matches'],event_count=len(events)))
            if 'evaluation' in row:
                route_result['evaluations'][name] = row['evaluation']
            elif name == 'unigram':
                route_result['evaluations'][name] = row['evaluations']['test']
            elif name.startswith('train-'):
                training = {k:row[k] for k in ('steps','valid_tokens','initial_validation_nll','final_validation_nll',
                    'validation_curve','training_wall_seconds','checkpoint_sha256','inventory')}
                logged = [json.loads(e)['metrics'] for e in events if 'train/update_seconds' in json.loads(e)['metrics']]
                training['sampled_update_seconds_mean'] = statistics.mean(r['train/update_seconds'] for r in logged)
                training['sampled_updates'] = len(logged)
                training['npu_peak_allocated_mib'] = max(r.get('resource/npu_peak_allocated_mib',0) for r in logged)
                training['process_peak_rss_mib'] = row['process_peak_rss_mib']
                training['stage_seconds_including_startup_and_recording'] = step['elapsed_seconds']
                route_result['training'][name.removeprefix('train-seed11-')] = training
            elif name.startswith('inference-'):
                route_result['inference'][name] = row
        evaluation = route_result['evaluations']
        pairs = [('sparse-minus-dense','evaluate-seed11-tide-sparse','evaluate-seed11-tide-dense')]
        if route == 'a':
            pairs += [('sparse-minus-transformer','evaluate-seed11-tide-sparse','evaluate-seed11-transformer'),
                      ('dense-minus-transformer','evaluate-seed11-tide-dense','evaluate-seed11-transformer'),
                      ('sparse-minus-unigram','evaluate-seed11-tide-sparse','unigram'),
                      ('transformer-minus-unigram','evaluate-seed11-transformer','unigram')]
            for kind in ('tide-dense','tide-sparse','transformer'):
                pairs.append(('context-reset-minus-full-'+kind,'context_reset-'+kind,'evaluate-seed11-'+kind))
            for kind in ('tide-dense','tide-sparse'):
                pairs.append(('zero-core-minus-full-'+kind,'zero_core-'+kind,'evaluate-seed11-'+kind))
        else:
            for kind in ('tide-dense','tide-sparse','mlp','original'):
                pairs.append((kind+'-minus-frozen','evaluate-seed11-'+kind,'evaluate-frozen-baseline'))
            for control in ('original','mlp'):
                pairs.append(('sparse-minus-'+control,'evaluate-seed11-tide-sparse','evaluate-seed11-'+control))
        for name, left, right in pairs:
            route_result['comparisons'][name] = paired_documents(evaluation[left]['rows'],evaluation[right]['rows'])
        dense = evaluation['evaluate-seed11-tide-dense']['work']; sparse = evaluation['evaluate-seed11-tide-sparse']['work']
        route_result['structure'] = dict(dense=dense,sparse=sparse,
            full_reduction=1-sparse['work/full']/dense['work/full'],message_reduction=None,
            scope='FC4 one region, no internal edges/messages; candidate/Upd/Read unchanged; not total FLOPs reduction')
        result['routes'][route] = route_result
    result['budget'] = dict(total_npu_seconds=28800,conservative_used_seconds=cumulative,
        conservative_remaining_seconds=28800-cumulative,
        shared_deadline_epoch=json.loads((root/'.local/language-batch-budget.json').read_text())['deadline_epoch'],
        current_active_jobs=0,new_training_submitted=False)
    result['input_sha256'] = hashes
    result['review_scope'] = 'statistics and observed states; scientific conclusions require the accompanying human-reviewed reports'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=Path('docs/evidence/language-v3-terminal.json'))
    args = parser.parse_args(); review = build(args.root)
    write_review_json(args.output,review,root=args.root)
    for route,row in review['routes'].items():
        print(route,'Full reduction',row['structure']['full_reduction'])
        for name,value in row['comparisons'].items():
            print(name,round(value['delta_nll'],6),[round(x,6) for x in value['bootstrap_two_sided_95']],
                  'upper95',round(value['bootstrap_one_sided_95_upper'],6))
    print('budget',review['budget'],'verified projections',len(review['lens']))
