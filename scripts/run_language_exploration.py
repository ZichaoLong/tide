#!/usr/bin/env python3
"""Continue the bounded language comparison using verified existing calibration."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
from tide_lab.records import atomic_json, digest
from tide_lab.language_driver import StageRunner
from tide_lab.language_reuse import admission
from scripts.run_language_route import choose_case, forecast


def prepare(a, plan):
    import torch
    from tide_lab.train import installed_identity
    from tide_lab.language_training import load_tokens, indices
    torch.set_num_threads(1)
    reused = admission(plan, a.route, a.reuse_from, a.data,
                       Path(__file__).resolve().parents[1], installed_identity())
    limits = {k: plan['budget'][a.route + '_' + k + '_seconds'] for k in ('admission', 'training', 'final')}
    prior = []
    for path in a.prior_attempt:
        state = json.loads(path.read_text())
        if state['route'] != a.route or state['state'] not in ('completed', 'failed'):
            raise ValueError('prior spending must come from a terminal same-route record')
        for key in limits:
            limits[key] = max(0., limits[key] - state['spent_seconds'].get(key, 0.))
        prior.append(dict(path=str(path), sha256=digest(path), spent_seconds=state['spent_seconds']))
    rows, _ = load_tokens(a.data, a.route, 32)
    counts = (rows['train']['labels'] != -100).sum(1)
    for case in reused['cases']:
        total = 0; case['exact_exposures'] = []
        for step in range(plan['training']['maximum_updates']):
            total += int(counts[indices(len(counts), case['batch'], plan['initial_seed'], step)].sum())
            case['exact_exposures'].append(total)
    selection = choose_case(reused['cases'], plan, limits['training'] - 60)
    if selection is None:
        raise ValueError('reused cases do not fit the explicitly smaller-exposure comparison')
    if (selection['batch'], selection['length'], selection['steps']) != (8, 32, 128):
        raise ValueError('this continuation fixes B8/L32/128; no undeclared automatic recipe')
    if a.compute_seconds <= selection['forecast_seconds'] + limits['final'] + 60:
        raise ValueError('complete training plus independent evaluation reserve does not fit route cap')
    if a.compute_seconds > plan['budget'][a.route + '_allocated_seconds']:
        raise ValueError('route cap exceeds original allocation')
    return reused, limits, prior, selection


def run(a):
    plan = json.loads(a.plan.read_text())
    reused, limits, prior, selection = prepare(a, plan)
    exposure = selection['exact_exposures'][selection['steps'] - 1]
    summary = dict(route=a.route, batch=8, length=32, steps=128, valid_tokens=exposure,
                   forecast_seconds=selection['forecast_seconds'], limits=limits,
                   reused_source_commit=reused['source_commit'], reused_files=len(reused['source_files']),
                   evidence_files=len(reused['evidence']), prior_attempts=prior)
    if a.preflight:
        print(json.dumps(summary, indent=2)); return 0
    runner = StageRunner(a, plan, limits, a.compute_seconds - 30)
    runner.state.update(candidates=reused['candidates'], training=[], evaluations=[], prior_attempts=prior,
        reused_admission=dict(source_commit=reused['source_commit'], source_files=reused['source_files'],
                              evidence=reused['evidence'], scope=reused['scope']),
        axes=dict(execution='running', strict_numerical='prior small checks reused; historical failures retained',
                  discrete_semantics='prior small checks reused; trained inference pending',
                  recovery='prior B2/L3 three-update exact recovery reused; long trajectory unchecked',
                  learning='unchecked', performance='reused calibration; actual training timings pending'))
    resolved = copy.deepcopy(plan)
    order = plan['exploration']['execution_order'][a.route]
    resolved['resolved'] = dict(route=a.route, kinds=order, batch=8, microbatch=8, accumulation=1,
        global_batch=8, length=32, steps=128, seed=11, depth=1, exact_valid_token_exposure=exposure,
        selection_source=str(a.reuse_from / 'stage.json'), selection_reason='B32 has higher measured throughput but cannot finish matched 128 updates within training budget; B8 fits',
        forecast_seconds=selection['forecast_seconds'], checkpoint_rule='final declared update; no test selection',
        strict_mode='finite exploration; prior thresholds and failures retained',
        original_65536_token_admission_met=False)
    a.plan = a.output / 'resolved-seed11.json'
    atomic_json(a.plan, resolved); a.plan.chmod(0o444)
    runner.state['decisions'].append(dict(config=str(a.plan), sha256=digest(a.plan), **summary))
    runner.publish()
    def flags(kind):
        argv = ['--kind', kind, '--length', '32', '--batch', '8', '--depth', '1', '--seed', '11']
        if a.route == 'b': argv += ['--cache', str(reused['cache'])]
        return argv
    def evaluate(kind, checkpoint=None, baseline=False):
        label = 'frozen-baseline' if baseline else 'seed11-' + kind
        extra = flags(kind) + (['--baseline-only'] if baseline else ['--checkpoint', str(checkpoint)])
        for action, cap in (('evaluate', 240), ('inference', 180)):
            path, result = runner.stage(action + '-' + label, action, 'final', cap, extra)
            runner.state['evaluations'].append(dict(name=action + '-' + label, kind=kind,
                state=result['state'], output=str(path), baseline=baseline))
            runner.publish()
    try:
        if a.route == 'b': evaluate('original', baseline=True)
        else: runner.stage('unigram', 'unigram', 'final', 90, ['--length', '32'])
        completed = {}
        for index, kind in enumerate(order):
            reserve = sum(forecast(selection['calibrations'][k], 128) for k in order[index + 1:])
            predicted = forecast(selection['calibrations'][kind], 128)
            cap = min(plan['training']['per_attempt_seconds'], predicted * 1.15 + 30,
                      runner.remaining('training') - reserve - 15)
            path, result = runner.stage('train-seed11-' + kind, 'train', 'training', cap,
                                        flags(kind) + ['--steps', '128'])
            runner.state['training'].append(dict(kind=kind, output=str(path), state=result['state'],
                seed=11, steps=result.get('steps'), valid_tokens=result.get('valid_tokens'),
                learning_signal=result.get('learning_signal'), config=str(a.plan)))
            runner.publish()
            if result['state'] == 'completed':
                if result['steps'] != 128 or result['valid_tokens'] != exposure:
                    raise ValueError('actual exposure differs from predeclared paired workload')
                completed[kind] = path
                evaluate(kind, path / 'checkpoint.pt')
        if a.route == 'a':
            for kind, path in completed.items():
                for intervention in (['zero_core', 'context_reset'] if kind.startswith('tide') else ['context_reset']):
                    runner.stage(intervention + '-' + kind, 'evaluate', 'final', 180,
                        flags(kind) + ['--checkpoint', str(path / 'checkpoint.pt'), '--intervention', intervention])
        runner.state['skipped'].append(dict(name='scale-and-seed22',
            reason='this named continuation first obtains the complete seven-candidate learning comparison; retain unspent original budget for evidence review'))
        execution = (len(completed) == len(order) and all(r['state'] == 'completed' for r in runner.state['evaluations']))
        runner.state['axes']['execution'] = 'completed' if execution else 'failed or partial; inspect each stage'
        runner.state['axes']['learning'] = 'heldout results available for review; no automatic quality/structure claim'
        runner.state['science_review'] = 'pending'
        runner.finish('completed' if execution else 'failed')
        return 0 if execution else 1
    except BaseException as error:
        runner.state['axes']['execution'] = 'failed'; runner.finish('failed', repr(error)); raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'data', 'output', 'asset-path', 'recording-cli', 'reuse-from'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--route', choices=['a', 'b'], required=True)
    p.add_argument('--deadline-epoch', type=float, required=True)
    p.add_argument('--compute-seconds', type=float, required=True)
    p.add_argument('--prior-attempt', type=Path, action='append', required=True)
    p.add_argument('--preflight', action='store_true')
    return run(p.parse_args())


if __name__ == '__main__': raise SystemExit(main())
