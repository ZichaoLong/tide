#!/usr/bin/env python3
"""Inspect the accepted-policy completion batch without modifying workload records.

This collects observations and paired statistics, not scientific support. Human
review still owns interpretation and changes to research/map.json.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.job import inspect
from scripts.review_io import write_review_json
from scripts.review_language_results import paired_documents
from tide_lab.records import digest, now
from tide_lab.supervisor import verify_source


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def complete_events(path):
    """Do not interpret a concurrently appended partial JSONL record."""
    if not path.exists():
        return []
    content = path.read_text()
    lines = content.splitlines()
    if content and not content.endswith('\n'):
        lines = lines[:-1]
    return [json.loads(line) for line in lines]


def metric_summary(events):
    training = [e for e in events if 'train/loss' in e['metrics']]
    validation = [e for e in events if 'validation/nll' in e['metrics']]
    out = dict(events=len(events), logged_updates=len(training),
        latest_training=training[-1] if training else None,
        latest_validation=validation[-1] if validation else None)
    if training:
        out.update(mean_update_seconds=statistics.mean(e['metrics']['train/update_seconds'] for e in training),
            support_repaired_coordinates=sum(e['metrics'].get('support/repaired_coordinates', 0) for e in training),
            support_repaired_rows=sum(e['metrics'].get('support/repaired_rows', 0) for e in training),
            updates_with_support_repair=sum(e['metrics'].get('support/repaired_coordinates', 0) > 0 for e in training),
            support_scope='logged updates only, includes autograd replay; validation/inference counted separately')
    return out


def build(root, *, terminal=False):
    root = Path(root).resolve()
    config = read(root/'configs/studies/language/completion-v5/budget.json')
    names = [n for n in config['jobs'] if n.startswith('lm-complete-')]
    names.append('lm-completion-budget-v5')
    out = dict(schema='tide-language-completion-review-v1', observed_at=now(),
        scope='Previously blocked v5 candidates; original 24 worker-hour budget and deadline; no next scaling batch',
        observations_only=True, scientific_review='pending human interpretation', jobs={}, sources={},
        lens=[], checks={}, calibration={}, training={}, evaluations={}, inference={}, comparisons={}, structure={},
        quality_margin=.05, activity_reduction_margin=.20, hardware_speedup_required=False)
    inputs, evaluations, hashes = {}, {}, {}

    def frozen_json(path):
        hashes[str(path.relative_to(root))] = digest(path)
        return read(path)

    for name in names:
        record = root/'artifacts/jobs'/name/'job.json'
        if not record.exists():
            out['jobs'][name] = dict(status='not_submitted')
            continue
        job = read(record)
        life = inspect(record)
        source = job['source']['path']
        if source not in out['sources']:
            identity = verify_source(source)
            out['sources'][source] = dict(commit=identity['commit'], tree_sha256=identity['content']['tree_sha256'])
        for item in [job['environment_script'], *job['inputs'], *([job['queue_helper']] if job.get('queue_helper') else [])]:
            if item['path'] in inputs and inputs[item['path']] != item['sha256']:
                raise ValueError('conflicting fixed input identity: '+item['path'])
            inputs[item['path']] = item['sha256']
        base = root/'runs'/name
        chain = read(base/'stage.json')
        wait = read(root/'runs'/(name+'-wait')/'state.json')
        status = chain['state'] if chain else life['state']['state']
        if wait and wait['state'] == 'waiting_dependency':
            status = 'queued'
        row = dict(status=status, lifecycle=life, source_commit=job['source']['commit'],
            device=job['device'], resources=job['resources'], dependency_wait=wait, chain=chain)
        if terminal and not life['termination_verified_now']:
            raise ValueError('terminal review requires empty cgroups and descendants: '+name)
        out['jobs'][name] = row
        for step in (chain or {}).get('steps', []):
            folder = Path(step['output'])
            key = name+'/'+step['name']
            events = complete_events(folder/'metrics.jsonl')
            if step['action'] == 'train' and step['category'] == 'training':
                row['latest_metrics'] = metric_summary(events)
            projection = read(folder/'projection.json')
            if projection:
                manifest = read(folder/'manifest.json')
                if projection['project'] != 'tide-learning-lab' or projection['source_run_id'] != manifest['run_id']:
                    raise ValueError('Lens projection identity mismatch: '+key)
                verified = (projection.get('status') == 'verified' and projection.get('readback_matches')
                    and projection.get('event_count') == len(events))
                out['lens'].append(dict(job_id=name, study_id=manifest['study_id'], run_id=manifest['run_id'],
                    projection=str((folder/'projection.json').relative_to(root)),
                    local_event_count=len(events), final_projection_verified=bool(verified),
                    **{k:projection.get(k) for k in ('destination_name','destination_run_id','status','event_count','readback_matches')}))
            result = read(folder/'result.json')
            if not result or step['state'] == 'running':
                continue
            hashes[str((folder/'result.json').relative_to(root))] = digest(folder/'result.json')
            if step['state'] != 'completed' or result['state'] != 'completed' or step.get('exit_code') != 0:
                out.setdefault('failed_stages', {})[key] = dict(state=step['state'], error=result.get('error'),
                    record=str((folder/'result.json').relative_to(root)))
                continue
            if step['action'] == 'checks':
                out['checks'][key] = result
            if step['action'] == 'calibrate':
                out['calibration'][key] = {k:result.get(k) for k in ('batch','microbatch','accumulation',
                    'length','warmup_updates','timed_updates','mean_step_seconds','phase','process_peak_rss_mib',
                    'npu_peak_allocated_mib','evaluation_preview')}
            if step['action'] == 'train' and step['category'] == 'training':
                training = {k:result.get(k) for k in ('steps','start_step','updates_this_attempt','valid_tokens',
                    'schedule_steps','initial_validation_nll','final_validation_nll','learning_origin_validation_nll',
                    'learning_signal','validation_curve','training_wall_seconds','checkpoint_sha256')}
                training['inventory'] = {k:v for k,v in result['inventory'].items() if k != 'shared_groups'}
                training['metrics'] = metric_summary(events)
                training['transfer_record'] = str((folder/'transfer.json').relative_to(root)) if (folder/'transfer.json').exists() else None
                meta = read(folder/'checkpoint.json')
                if not meta or digest(folder/'checkpoint.pt') != meta['sha256'] or meta['sha256'] != result['checkpoint_sha256']:
                    raise ValueError('final checkpoint checksum mismatch: '+key)
                training['checkpoint_file_verified'] = True
                training['checkpoint_scope'] = 'file integrity; same-stack three-step recovery is in chain.admission, not long-trajectory equivalence'
                out['training'][key] = training
            if 'evaluation' in result:
                evaluations[key] = result['evaluation']
                out['evaluations'][key] = {k:v for k,v in result['evaluation'].items() if k != 'rows'}
                out['evaluations'][key].update(checkpoint_step=result.get('checkpoint_step'), checkpoint_sha256=result.get('checkpoint_sha256'))
            if step['action'] == 'inference':
                out['inference'][key] = {k:v for k,v in result.items() if k not in ('generated_token_ids','prefix_token_ids')}

    mismatches = [p for p,h in inputs.items() if not Path(p).is_file() or digest(p) != h]
    if mismatches:
        raise ValueError('fixed input changed: '+str(mismatches))
    out['fixed_inputs'] = dict(unique_files=len(inputs), all_hashes_match=True, files_sha256=inputs)
    lock = read(root/'dependencies/positive-floor-v1/installation.lock.json')
    package = lock['installed_identity']['package']
    bad = [n for n,h in package['files_sha256'].items() if digest(Path(package['path'])/n) != h]
    if bad:
        raise ValueError('accepted foundation installation changed: '+str(bad))
    out['installation'] = dict(lock_id=lock['lock_id'], files_checked=len(package['files_sha256']), all_hashes_match=True,
        lock='dependencies/positive-floor-v1/installation.lock.json')
    out['budget'] = read(root/'runs/lm-completion-budget-v5/state.json')
    out['terminal_verified'] = all(j.get('lifecycle', {}).get('termination_verified_now', False) for j in out['jobs'].values())
    out['completed_projections_verified'] = sum(p['final_projection_verified'] for p in out['lens'])

    # Same old weights before/after installation changes are implementation
    # comparisons, distinct from independently trained dense/sparse models.
    out['reference_implementation_checks'] = {}
    for label, old in [
            ('hb16-sparse128', 'lm-a-hb16-sparse-v5/test-tide-sparse'),
            ('hb16-sparse64', 'lm-a-hb16-sparse-v5/test-tide-sparse-prefix64'),
            ('fc4-dense128', 'lm-a-reference-v5/test-fc4-tide-dense-prefix128'),
            ('fc4-sparse128', 'lm-a-reference-v5/test-fc4-tide-sparse-prefix128')]:
        key = 'lm-complete-references-v5/test-'+label
        if key not in evaluations:
            continue
        old_record = frozen_json(root/'runs'/old/'result.json')
        old_ev, new_ev = old_record['evaluation'], evaluations[key]
        if old_record['checkpoint_sha256'] != out['evaluations'][key]['checkpoint_sha256']:
            raise ValueError('implementation comparison used different checkpoint files: '+label)
        # Validates exact document/block target pairing, not just aggregate NLL.
        paired_documents(new_ev['rows'], old_ev['rows'], draws=1)
        out['reference_implementation_checks'][label] = dict(old_record='runs/'+old+'/result.json',
            new_record='runs/'+key+'/result.json', checkpoint_sha256=old_record['checkpoint_sha256'],
            delta_nll=new_ev['nll']-old_ev['nll'],
            per_block_results_exact_equal=new_ev['rows'] == old_ev['rows'], work_exact_equal=new_ev['work'] == old_ev['work'],
            scope='same fixed old checkpoint on v5 test targets; does not certify all tensors, training or strict core equivalence')

    # Reuse already accepted FC16 ordinary/reset evaluation; do not rerun it.
    accepted = frozen_json(root/'runs/support-v1-cpu/fc/result.json')
    for label, ev in accepted['evaluations'].items():
        if 'rows' in ev:
            key = 'accepted-fc16-dense128/'+label
            evaluations[key] = ev
            out['evaluations'][key] = {k:v for k,v in ev.items() if k != 'rows'}
    refs = 'lm-complete-references-v5/'
    fc = 'lm-complete-fc16-sparse-v5/'
    hb = 'lm-complete-hb16-dense-v5/'
    wide = 'lm-complete-wide-dense-v5/'
    pairs = [
        ('FC16-sparse-minus-dense', fc+'test-tide-sparse', 'accepted-fc16-dense128/normal'),
        ('HB16-sparse-minus-dense', refs+'test-hb16-sparse128', hb+'test-tide-dense'),
        ('FC16-sparse-minus-FC4-sparse', fc+'test-tide-sparse', refs+'test-fc4-sparse128'),
        ('HB16-dense-minus-FC4-dense', hb+'test-tide-dense', refs+'test-fc4-dense128'),
        ('wide-dense-minus-FC4-dense', wide+'test-tide-dense', refs+'test-fc4-dense128')]
    for topology in ('fc16-sparse','hb16-dense','wide-dense','hb40-dense','hb40-sparse'):
        prefix = 'lm-complete-'+topology+'-v5/'
        kind = topology.rsplit('-', 1)[1]
        pairs.append((topology+'-reset-minus-normal', prefix+'context-reset-tide-'+kind, prefix+'test-tide-'+kind))
    for kind in ('dense','sparse'):
        pairs.append(('HB40-'+kind+'-minus-HB16-prefix64', 'lm-complete-hb40-'+kind+'-v5/test-tide-'+kind, refs+'test-hb16-'+kind+'64'))
    pairs.append(('HB40-sparse-minus-dense', 'lm-complete-hb40-sparse-v5/test-tide-sparse', 'lm-complete-hb40-dense-v5/test-tide-dense'))
    for label, candidate, reference in pairs:
        if candidate in evaluations and reference in evaluations:
            out['comparisons'][label] = dict(candidate=candidate, reference=reference,
                **paired_documents(evaluations[candidate]['rows'], evaluations[reference]['rows']))
            if label in ('FC16-sparse-minus-dense','HB16-sparse-minus-dense','HB40-sparse-minus-dense'):
                s, d = evaluations[candidate]['work'], evaluations[reference]['work']
                out['structure'][label] = dict(dense=d, sparse=s, full_reduction=1-s['work/full']/d['work/full'],
                    message_reduction=1-s['work/messages']/d['work/messages'] if d['work/messages'] else None,
                    scope='actual Full/internal messages on matching targets; not total FLOPs or hardware speedup')
    out['result_sha256'] = hashes
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=Path('docs/evidence/language-completion-handoff.json'))
    parser.add_argument('--terminal', action='store_true')
    args = parser.parse_args()
    result = build(args.root, terminal=args.terminal)
    write_review_json(args.output, result, root=args.root)
    print('observed', result['observed_at'], 'terminal', result['terminal_verified'],
        'fixed files', result['fixed_inputs']['unique_files'], 'installed files', result['installation']['files_checked'])
    for name, row in result['jobs'].items():
        event = row.get('latest_metrics', {}).get('latest_training') or {}
        print(name, row['status'], 'latest step', event.get('step'))
    print('verified completed Lens projections', result['completed_projections_verified'])
