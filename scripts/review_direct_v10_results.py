#!/usr/bin/env python3
"""Read-only terminal review of v10, including explicit repairs and extensions."""
import argparse
from collections import Counter, defaultdict
import datetime
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.job import inspect
from scripts.review_direct_v10 import review
from scripts.review_v10_repair import review as review_repairs
from scripts.review_moe_results import endpoint
from scripts.review_hb_learning import paired
from scripts.review_io import write_review_json
from tide_lab.quota import account_quota
from tide_lab.records import digest, now

ROOT = Path(__file__).resolve().parents[1]
MI = 2**20


def read(path):
    return json.loads(Path(path).read_text())


def audit():
    root = ROOT
    rel = lambda path: str(Path(path).resolve().relative_to(root))
    observation = review()
    repairs = review_repairs()
    controller = read(root/'runs/v10-controller/state.json')
    owner = inspect(root/'artifacts/jobs/v10-controller/job.json')
    if (controller['state'] != 'completed' or not owner['termination_verified_now']
            or any(not r['descendants_terminated'] for r in observation['jobs'])
            or any(r['execution'] not in ('completed', 'failed') for r in observation['jobs'])):
        raise ValueError('v10 is not terminal')
    for r in repairs['cases']:
        if (not r['descendants_terminated'] or r['training_state'] != 'completed'
                or r['confirmation'].get('state') != 'completed'
                or r['recovery_state'] != 'passed' or r['migration_state'] != 'completed'):
            raise ValueError('repair lifecycle/recovery incomplete: '+r['id'])
    signatures = {}
    rows = []
    paths = {}
    projections = []
    data_sha = digest(root/'runs/v9-data-r3/data-lock.json')

    def projection(path):
        value = read(path)
        if value.get('status') != 'verified' or not value.get('readback_matches'):
            raise ValueError('Lens readback missing: '+str(path))
        projections.append(dict(record=rel(path), **value))

    for phase, group in controller['phases'].items():
        for c in group['cases']:
            path = Path(c['output'])
            projection(path/('projection.json' if phase.startswith('confirm') else 'train/projection.json'))
    for r in repairs['cases']:
        for stage in ('pilot', 'train', 'confirmation'):
            projection(root/'runs'/r['job']/stage/'projection.json')

    def add(cid, train, conf, config_path, exposure, repaired=False):
        config = read(config_path)
        result = read(train/'result.json')
        manifest = read(train/'manifest.json')
        identity = result['identity']
        cr = read(conf/'result.json')
        if (result['state'] != 'completed' or result['progress']['tokens'] != exposure
                or result['inference']['state'] != 'completed'
                or identity['config_sha256'] != digest(config_path)
                or identity['data_lock_sha256'] != data_sha
                or manifest['config'] != config or manifest['identity'] != identity
                or cr['state'] != 'completed' or cr['training_identity'] != identity
                or cr['config_sha256'] != identity['config_sha256']
                or cr['exposure'] != exposure or cr['test_access']):
            raise ValueError('endpoint identity/exposure mismatch: '+cid)
        checkpoint = read(train/'checkpoint.json')
        cp = train/checkpoint['file']
        sha = digest(cp)
        if sha != checkpoint['sha256'] or sha != cr['checkpoint_sha256']:
            raise ValueError('checkpoint/confirmation SHA mismatch: '+cid)
        values = {}
        for split, path, count in [('validation', train/'validation-final.json', 131072),
                                   ('confirmation', conf/'confirmation.json', 262144)]:
            value, signature = endpoint(path, count)
            if split not in signatures:
                signatures[split] = signature
            if signature != signatures[split]:
                raise ValueError('evaluation blocks differ: '+cid)
            values[split] = dict(nll=value['nll'], tokens=count, record=rel(path),
                                 sha256=digest(path), work=value['work'])
        if cr['nll'] != values['confirmation']['nll']:
            raise ValueError('confirmation aggregate mismatch')
        shape = cid.rsplit('-s', 1)[0]
        paths[(shape, config['seed'], exposure)] = conf/'confirmation.json'
        work = values['validation']['work']
        parameters = result['inventory']['trainable_parameters']
        activity = dict(active_parameters=parameters)
        if work:
            activity.update(
                active_parameters=work['work/active_parameter_uses']/work['work/prediction_positions'],
                full_all=work['work/full']/work['work/node_slots'],
                receipt_all=work['work/candidates']/work['work/node_slots'],
                full_candidates=work['work/full']/work['work/candidates'],
                physical_full_all=work['work/physical_full_rows']/work['work/node_slots'])
        admission = root/'runs'/('v10-repair-'+cid if repaired else 'v10-main-'+cid)
        recovery_path = admission/('recovery-compare/recovery.json' if repaired else 'recovery.json')
        recovery = read(recovery_path)
        if recovery['state'] != 'passed':
            raise ValueError('missing bounded recovery: '+cid)
        check = read(admission/'check/result.json')
        checks = check['checks']
        curves = {}
        for amount in (8*MI, 32*MI, 128*MI, 256*MI):
            path = train/f'validation-{amount}.json'
            if amount > exposure or not path.is_file():
                continue
            v, signature = endpoint(path, 131072)
            if signature != signatures['validation']:
                raise ValueError('curve blocks differ: '+cid)
            curves[str(amount)] = dict(nll=v['nll'], record=rel(path), sha256=digest(path))
        parent = result.get('ancestry')
        if exposure == 256*MI:
            expected = read(root/'runs'/('v10-main-'+cid)/'train/checkpoint.json')['sha256']
            if not parent or parent['sha256'] != expected:
                raise ValueError('extension ancestry mismatch: '+cid)
        rows.append(dict(id=cid, shape=shape, seed=config['seed'], exposure=exposure,
            repaired=repaired, config=rel(config_path), identity=identity, ancestry=parent,
            parameters=parameters, activity=activity, training=config['training'],
            microbatch=result['microbatch'], global_batch=result['global_batch'],
            validation=values['validation'], confirmation=values['confirmation'], curves=curves,
            stage_training_seconds=result['training_seconds'], stage_elapsed_seconds=result['elapsed_seconds'],
            training_time_scope='this stage only; repaired prefixes and extensions are not fresh trajectories',
            recovery=dict(state=recovery['state'], scope=recovery['scope'], record=rel(recovery_path)),
            strict_forward=checks['forward'],
            strict_vjp_failures=sum(v['state']=='failed' for v in checks['parameter_vjp'].values()),
            check_record=rel(admission/'check/result.json'), axes=result['axes'],
            checkpoint=dict(record=rel(cp), sha256=sha, verified_bytes=cp.stat().st_size)))

    for phase, count in [('main', 128*MI), ('extend', 256*MI)]:
        confirmed = {c['id']: c for c in controller['phases']['confirm-'+phase]['cases']}
        for c in controller['phases'][phase]['cases']:
            if c['id'] not in confirmed:
                continue
            add(c['id'], Path(c['output'])/'train', Path(confirmed[c['id']]['output']),
                Path(c['config']), count)
    for r in repairs['cases']:
        cid = r['id']
        out = root/'runs'/r['job']
        add(cid, out/'train', out/'confirmation', root/'runs/v10-repair-control'/cid/'config.json',
            128*MI, repaired=True)
    if Counter(r['exposure'] for r in rows) != {128*MI: 20, 256*MI: 7}:
        raise ValueError('unexpected endpoint inventory')
    groups = defaultdict(list)
    for r in rows:
        groups[(r['shape'], r['exposure'])].append(r)
    means = []
    for (shape, exposure), items in sorted(groups.items()):
        means.append(dict(shape=shape, exposure=exposure, seeds=[r['seed'] for r in items],
            parameters=items[0]['parameters'],
            activity={k: statistics.mean(r['activity'][k] for r in items) for k in items[0]['activity']},
            validation=statistics.mean(r['validation']['nll'] for r in items),
            confirmation=statistics.mean(r['confirmation']['nll'] for r in items),
            seed_confirmation={str(r['seed']): r['confirmation']['nll'] for r in items}))
    comparisons = []

    def compare(left, right, reason):
        value = paired(paths[left], paths[right])
        if not value['complete_same_blocks'] or value['tokens'] != 262144:
            raise ValueError('incomplete paired confirmation')
        value.update(left=rel(value['left']), right=rel(value['right']), reason=reason,
            left_identity=list(left), right_identity=list(right),
            scope='independent confirmation; paired document bootstrap (2000), conditional on fixed checkpoints; excludes seed/model-selection uncertainty')
        comparisons.append(value)

    for seed in (11, 22):
        for route in ('r28-q10', 'r64-q3'):
            compare((f'hb-d128-{route}', seed, 128*MI), (f'hb-d64-{route}', seed, 128*MI),
                    'width growth at similar Full/all; active computation also increases')
            compare((f'hb-d128-{route}', seed, 128*MI),
                    (f'hb-d128-{route.split("-")[0]}-pool', seed, 128*MI), 'same-shape activity cost')
        for width, dense in [(64, 'r4'), (128, 'r3')]:
            compare((f'hb-d{width}-r28-q10', seed, 128*MI),
                    (f'hb-d{width}-{dense}-dense-match', seed, 128*MI), 'approximate active capacity, more total parameters')
            compare((f'hb-d{width}-r64-q3', seed, 128*MI),
                    (f'hb-d{width}-r28-q10', seed, 128*MI), 'approximate active capacity, lower Full/all')
        compare(('hb-d128-r64-pool', seed, 128*MI), ('hb-d128-r28-pool', seed, 128*MI),
                'pool capacity growth; repaired histories explicitly retained')
        compare(('hb-d64-r64-q3', seed, 256*MI), ('tf-d192-l6', seed, 256*MI),
                'same 256Mi exposure, low-activity HB versus calibrated Transformer')
    for shape, seed, exposure in sorted(paths):
        if exposure == 256*MI:
            compare((shape, seed, exposure), (shape, seed, 128*MI), 'same trajectory exposure 128Mi to 256Mi')
    budget = read(root/'configs/studies/language/direct-v10/budget.json')
    end = datetime.datetime.fromisoformat(controller['finished_at']).timestamp()
    return dict(schema='tide-v10-terminal-review-v1', observed_at=now(), state='terminal_reviewed',
        finished_at=controller['finished_at'], wall_hours_from_budget_start=(end-budget['started_epoch'])/3600,
        all_services_terminated=True, controller_terminated=True, observation=observation, repairs=repairs,
        phase_state_counts=dict(Counter(r['phase']+':'+r['execution'] for r in observation['jobs'])),
        failures=[r for r in observation['jobs'] if r['execution']=='failed'],
        rows=rows, means=means, comparisons=comparisons, lens_projections=projections,
        confirmation_targets=262144, confirmation_blocks=len(signatures['confirmation']),
        data_lock_sha256=data_sha, test_access=False, quota=account_quota(),
        limits=['Original 3 nonfinite pool failures plus 1 extension feasibility failure remain.',
                'Repairs are explicit continuations, not independent seeds.',
                'q10 seed11 has no 256Mi endpoint; do not present seed22 as a two-seed mean.',
                'Strict finite numerical failures, discrete checks, bounded recovery and learning remain separate.',
                'Dense physical Full, logical activity and achieved throughput are distinct.',
                '256Mi exposure repeats the fixed 128Mi training pool; it is not 256Mi independent data.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', type=Path)
    args = parser.parse_args()
    value = audit()
    if args.write:
        write_review_json(args.write, value)
    print(json.dumps({k: value[k] for k in ('finished_at', 'wall_hours_from_budget_start',
                                          'phase_state_counts', 'means')}, ensure_ascii=False, indent=2))
