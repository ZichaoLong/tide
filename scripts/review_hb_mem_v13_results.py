#!/usr/bin/env python3
"""Audit available v13 endpoints without changing jobs or loading live weights."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.review_hb_mem_v13 import review, read
from scripts.review_hb_learning import paired
from scripts.review_moe_results import endpoint
from scripts.review_io import write_review_json
from tide_lab.quota import account_quota
from tide_lab.records import digest, now

M = 2**20


def audit(root):
    live = review(root)
    table_path = root / 'configs/studies/language/hb-mem-v13/table.json'
    table = read(table_path)
    training_sha = digest(root / 'runs/v9-data-r3/data-lock.json')
    confirmation_sha = digest(root / 'runs/v13-data/data-lock.json')
    frozen = read(root / '.local/source-hb-mem-v13-main/.tide-source.json')
    rows, paths, signatures, recoveries, pending = [], {}, {}, [], []

    def rel(path):
        return str(Path(path).resolve().relative_to(root))

    def require(ok, message):
        if not ok:
            raise ValueError(message)

    def projection(path):
        p = read(path)
        require(p.get('status') == 'verified' and p.get('readback_matches'),
                'Lens readback missing: ' + rel(path))
        return {k: p.get(k) for k in ('status', 'readback_matches', 'source_run_id',
                                     'destination_name', 'destination_run_id')}

    def collect(case, out, training, reused=False):
        conf = out if reused else out / 'confirmation'
        cr = read(conf / 'result.json')
        tr = read(training / 'result.json')
        manifest = read(training / 'manifest.json')
        cfg_path = root / case['config']
        cfg, identity = read(cfg_path), tr['identity']
        require(tr['state'] == 'completed' and tr['progress']['tokens'] == 128*M
                and tr['inference']['state'] == 'completed', 'training incomplete: ' + case['id'])
        require(manifest['config'] == cfg and manifest['identity'] == identity
                and digest(cfg_path) == identity['config_sha256']
                and identity['data_lock_sha256'] == training_sha,
                'training config/data identity mismatch: ' + case['id'])
        require(cr['state'] == 'completed' and cr['exposure'] == 128*M
                and cr['training_identity'] == identity
                and cr['training_lock_sha256'] == training_sha
                and cr['confirmation_lock_sha256'] == confirmation_sha
                and cr['config_sha256'] == identity['config_sha256']
                and not cr['test_access'] and cr['source']['commit'] == frozen['commit'],
                'confirmation identity mismatch: ' + case['id'])
        cp = training / 'latest.pt'
        require(Path(cr['checkpoint']).resolve() == cp.resolve(), 'checkpoint path mismatch')
        sha = digest(cp)  # Only completed, immutable training endpoints reach this path.
        require(sha == read(training / 'checkpoint.json')['sha256'] == cr['checkpoint_sha256'],
                'checkpoint SHA mismatch: ' + case['id'])
        if reused:
            require(sha == case['sha256'] and identity == case['training_identity'],
                    'reused parent differs from frozen table: ' + case['id'])
        else:
            require(identity['source_commit'] == frozen['commit'], 'new training source mismatch')
        values = {}
        for name, path, count in [('validation', training / 'validation-final.json', 131072),
                                  ('confirmation', conf / 'confirmation.json', 262144)]:
            val, sig = endpoint(path, count)
            require(name not in signatures or signatures[name] == sig,
                    'different evaluation blocks: ' + case['id'])
            signatures[name] = sig
            values[name] = dict(nll=val['nll'], record=rel(path), sha256=digest(path))
            if name == 'confirmation':
                work = val['work']
                require(val['nll'] == cr['nll'], 'confirmation result/aggregate mismatch')
                paths[(case['group'], case['mem'], case['seed'])] = path
        n = work['work/prediction_positions']
        parts = {role: work[f'work/{role}_parameter_uses']/n for role in ('full', 'receive', 'shared')}
        active = work['work/active_parameter_uses']/n
        require(math.isclose(sum(parts.values()), active, abs_tol=1e-6), 'active partition mismatch')
        lens = dict(train=projection(training / 'projection.json'),
                    confirmation=projection(conf / 'projection.json'))
        diagnostic = dict(state='not_repeated_for_reused_parent')
        if not reused:
            lens['pilot'] = projection(out / 'pilot/projection.json')
            lens['diagnostic'] = projection(out / 'diagnostic/projection.json')
            dg = read(out / 'diagnostic/result.json')
            require(dg['state'] == 'completed' and dg['parent_identity'] == identity
                    and dg['case']['sha256'] == sha
                    and dg['observer_output_equality'] == 'passed_exact_same_batch',
                    'diagnostic identity mismatch: ' + case['id'])
            require('diagnostic' not in signatures or signatures['diagnostic'] == dg['block_ids'],
                    'different diagnostic train window')
            signatures['diagnostic'] = dg['block_ids']
            coverage = [v['coverage_summary'] for v in dg['observations'].values()]
            diagnostic = dict(state=dg['state'], record=rel(out / 'diagnostic/result.json'),
                              zero_full_nodes=sum(v['zero'] for v in coverage),
                              nodes=sum(v['units'] for v in coverage),
                              mean_region_effective_units=statistics.mean(v['effective_units'] for v in coverage),
                              scope=dg['scope'])
        rows.append(dict(id=case['id'], group=case['group'], mem=case['mem'], seed=case['seed'],
                         reused=reused, config=rel(cfg_path), checkpoint=rel(cp), checkpoint_sha256=sha,
                         identity=identity, total_parameters=tr['inventory']['trainable_parameters'],
                         active_parameters=active, active_parts=parts,
                         full_all=work['work/full']/work['work/node_slots'],
                         receipt_all=work['work/candidates']/work['work/node_slots'],
                         work=work, exposure=cr['exposure'], validation=values['validation'],
                         confirmation=values['confirmation'], inference=tr['inference']['state'],
                         lens=lens, diagnostic=diagnostic,
                         main_training_seconds=tr['training_seconds'] if not reused else None,
                         timing_scope='new main segment8Mi-to128Mi update time; excludes pilot/checks/evaluation/I/O; shared cards',
                         preserved_source_scope=case.get('scope', 'new frozen source')))

    for case, state in zip(table['cases'], live['new_training']):
        require(case['id'] == state['id'], 'live/table ordering mismatch')
        out = root / 'runs' / ('v13-main-' + case['id'])
        recovery = [dict(record=rel(out / p), state=read(out / p).get('state'))
                    for p in ('recovery.json', 'trained-compare/recovery.json')]
        recoveries.append(dict(id=case['id'], checks=recovery))
        if state['execution'] == 'completed' and state['terminated']:
            require(all(r['state'] == 'passed' for r in recovery), 'completed endpoint recovery missing')
            collect(case, out, out / 'train')
        else:
            pr = read(out / 'train/progress.json')
            logs = out / 'train/metrics.jsonl'
            raw = [json.loads(line) for line in logs.read_text().splitlines()] if logs.exists() else []
            recent = [r for r in raw if 'train/tokens_per_second' in r['metrics']][-8:]
            rate = statistics.median(r['metrics']['train/tokens_per_second'] for r in recent) if recent else None
            wallrate = ((recent[-1]['metrics']['train/tokens']-recent[0]['metrics']['train/tokens']) /
                        (recent[-1]['elapsed_seconds']-recent[0]['elapsed_seconds'])) if len(recent)>1 else None
            latest = read(out / 'train/validation-33554432.json')
            pending.append(dict(id=case['id'], group=case['group'], mem=case['mem'], seed=case['seed'],
                                state=state['execution'], stage=state['stage'], device=state['device'],
                                tokens=pr.get('tokens'), progress_updated_at=pr.get('updated_at'),
                                median_last8_update_targets_s=rate, recent_wall_targets_s=wallrate,
                                estimated_remaining_train_h=(128*M-pr['tokens'])/wallrate/3600 if wallrate else None,
                                validation_at_32Mi=latest.get('nll'),
                                lens_status=state['lens'], error=state['error']))
    for case, state in zip(table['qry_reuse'], live['reused_qry_confirmation']):
        require(case['id'] == state['id'], 'reuse/table ordering mismatch')
        if state['execution'] == 'completed' and state['terminated']:
            collect(case, root / 'runs' / ('v13-reuse-' + case['id']),
                    (root / case['checkpoint']).parent, reused=True)
    comparisons, groups = [], []
    for group in sorted({r['group'] for r in table['cases']}):
        selected = [r for r in rows if r['group'] == group]
        if {(r['mem'], r['seed']) for r in selected} != {('sum', 11), ('sum', 22), ('qry', 11), ('qry', 22)}:
            continue
        means = {}
        for mem in ('sum', 'qry'):
            rs = [r for r in selected if r['mem'] == mem]
            means[mem] = dict(seed_nll={str(r['seed']): r['confirmation']['nll'] for r in rs},
                              mean_nll=statistics.mean(r['confirmation']['nll'] for r in rs),
                              total_parameters=rs[0]['total_parameters'],
                              active_parameters=statistics.mean(r['active_parameters'] for r in rs),
                              full_all=statistics.mean(r['full_all'] for r in rs),
                              receipt_all=statistics.mean(r['receipt_all'] for r in rs))
        for seed in (11, 22):
            p = paired(paths[(group, 'sum', seed)], paths[(group, 'qry', seed)])
            require(p['complete_same_blocks'] and p['tokens'] == 262144, 'unmatched pair')
            p.update(group=group, seed=seed, left=rel(p['left']), right=rel(p['right']),
                     scope='v13 fresh confirmation document bootstrap conditional on fixed seed/checkpoints; excludes seed/model-selection uncertainty')
            comparisons.append(p)
        groups.append(dict(group=group, means=means,
                           delta_sum_minus_qry=means['sum']['mean_nll']-means['qry']['mean_nll']))
    fs = os.statvfs(root)
    return dict(schema='tide-hb-mem-v13-endpoint-review-v1', observed_at=now(),
                state='partial_endpoints_audited' if pending else 'all_endpoints_audited',
                source_commit=frozen['commit'], table_sha256=digest(table_path),
                training_lock_sha256=training_sha, confirmation_lock_sha256=confirmation_sha,
                live=live, recoveries=recoveries, endpoints=rows, paired_groups=groups,
                comparisons=comparisons, pending=pending,
                resources=dict(account_quota=account_quota(), filesystem_available_gib=fs.f_bavail*fs.f_frsize/2**30),
                limits=['This audit does not create scientific support automatically; the associated narrative/map is reviewed separately.',
                        'Logical active parameters include Full/receipt/shared per model position including padding; not FLOPs or measured physical sparsity.',
                        'Different model functions and realized routing; sum has fewer parameters. Not an equal-parameter or equivalent-implementation experiment.',
                        'Historical qry parents retain source/repair ancestry. Same evaluator does not make training trajectories identical.',
                        'Only v13 confirmation is used for paired NLL; do not subtract v12 or earlier absolute confirmation values.',
                        'Original finite numerical failures and3 calibration OOMs remain. Recovery and task quality do not waive strict thresholds.',
                        'ETA uses last8 logged updates, excludes later confirmation/diagnosis and is subject to shared-device contention.',
                        'No new training or evaluation was launched; test remains sealed.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    write_review_json(args.output, result, root=args.root.resolve())
    print(json.dumps({k: result[k] for k in ('state', 'observed_at', 'paired_groups', 'pending', 'resources')},
                     indent=2, ensure_ascii=False))
