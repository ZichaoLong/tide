#!/usr/bin/env python3
"""Fixed four-candidate growth comparison; independent paths retain separate gates."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bounded_workload import remaining_seconds
from scripts.npu_learning_stage import bounded_command, ChildFailure, CleanupFailure
from tide_lab.records import atomic_json, digest, now
from tide_lab.supervisor import group_processes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'output', 'asset-path', 'recording-cli'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--deadline-epoch', type=float, required=True)
    a = p.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    if plan['max_seconds'] != 10800 or len(plan['runs']) != 4:
        raise ValueError('outside the four-candidate three-hour study')
    root = Path(a.output).resolve(); root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(state='running', started_at=now(), plan=plan, plan_sha256=digest(a.plan),
        deadline_epoch=a.deadline_epoch, commands=[], runs=[], gates={},
        axes=dict(execution='running', strict_numerical='historical failures retained; changed path pending',
                  discrete_semantics='prior bounded core5 binding checks reused; no per-step CPU reference',
                  recovery='pending separate frozen/full paths', learning='not_checked'))
    def publish():
        report['elapsed_seconds'] = time.monotonic()-started
        atomic_json(root/'stage.json', report)
    def interrupted(signum, _):
        raise InterruptedError(f'growth stage received signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    def run(label, command, cap):
        wall = min(cap, remaining_seconds(plan['max_seconds']-20, a.deadline_epoch-30, started))
        if wall <= 20:
            raise TimeoutError('stage/common deadline exhausted')
        row = dict(name=label, command=command, wall_budget_seconds=wall, cleanup_reserve_seconds=10)
        report['commands'].append(row); report['phase']=label; publish()
        bounded_command(command, root/(label+'.log'), wall-10, row, publish)
    def gate(label, command, cap, result_path, accepted):
        try:
            run(label, command, cap)
            value = json.loads(result_path.read_text())
            ok = accepted(value)
            report['gates'][label] = dict(state='passed' if ok else 'failed', result=value,
                                         result_sha256=digest(result_path))
            return ok
        except ChildFailure as error:
            value = json.loads(result_path.read_text()) if result_path.exists() else None
            report['gates'][label] = dict(state='failed', error=repr(error), result=value)
            return False
        finally:
            publish()
    publish()
    try:
        frozen_ingress = gate('frozen-ingress', [sys.executable, 'scripts/check_growth.py',
            '--checkpoint-path', a.asset_path, '--output', str(root/'ingress.json'), '--device', 'npu:0',
            '--threads', '4', '--seed', '11', '--freeze-backbone', '--skip-generation', '--explore-numerics'],
            600, root/'ingress.json', lambda r:r['execution']=='completed')
        ingress = report['gates']['frozen-ingress'].get('result')
        report['axes']['strict_numerical'] = dict(historical='all original thresholds and failures retained',
            changed_frozen_path=ingress.get('strict_numeric','not_checked') if ingress else 'not_checked',
            details='gates.frozen-ingress.result')
        frozen_resume = False
        if frozen_ingress:
            frozen_resume = gate('frozen-resume', [sys.executable, 'scripts/check_resume.py',
                '--config', plan['frozen_resume_config'], '--output', str(root/'frozen-resume'),
                '--asset-path', a.asset_path, '--retire-large-checkpoints'], 750,
                root/'frozen-resume/result.json', lambda r:r['state']=='passed')
        full_resume = gate('full-resume', [sys.executable, 'scripts/check_resume.py',
            '--config', plan['full_resume_config'], '--output', str(root/'full-resume'),
            '--asset-path', a.asset_path, '--retire-large-checkpoints'], 750,
            root/'full-resume/result.json', lambda r:r['state']=='passed')
        report['axes']['recovery'] = dict(frozen='passed' if frozen_resume else 'failed_or_unchecked',
            full='passed' if full_resume else 'failed', scope='3 updates/B2 fresh-process same-stack full model/SGD/RNG with intermediate diagnostics; not cross-backend or full horizon')
        for item in plan['runs']:
            ready = full_resume if item['path']=='full' else frozen_ingress and frozen_resume
            row = dict(run_id=item['run_id'], config=item['config'], path=item['path'])
            report['runs'].append(row)
            if not ready:
                row.update(state='not_started', reason='applicable ingress/recovery gate failed or unchecked'); publish(); continue
            config = json.loads(Path(item['config']).read_text())
            if (config['training']['steps'] != 200 or config['training']['batch_size'] != 32
                or config['evaluation']['diagnostic_steps'] != [40]):
                raise ValueError('fixed horizon/batch/diagnostic contract changed')
            matrix_path = root/(item['run_id']+'-matrix.json')
            atomic_json(matrix_path, dict(schema='tide-lab-matrix-v1', study_id=plan['study_id'],
                max_seconds=item['wall_seconds']-80, stop_on_failure=True,
                runs=[dict(config=str(Path(item['config']).resolve()), run_id=item['run_id'])]))
            matrix_path.chmod(0o444)
            out = root/item['run_id']
            try:
                run(item['run_id'], [sys.executable, '-m', 'tide_lab.batch', '--matrix', str(matrix_path),
                    '--output', str(out), '--asset-path', a.asset_path, '--recording-cli', a.recording_cli], item['wall_seconds'])
                batch = json.loads((out/'batch.json').read_text())
                result = json.loads((out/item['run_id']/'result.json').read_text())
                if batch['state'] != 'completed' or result['state'] != 'completed' or result['steps'] != 200:
                    raise ChildFailure('inner budget incomplete')
                row.update(state='completed', result=str(out/item['run_id']/'result.json'),
                    result_sha256=digest(out/item['run_id']/'result.json'),
                    validation=result['last_validation'], test=result.get('test'),
                    learning_gate_met=result['training_acceptance']['learning_gate_met'])
            except ChildFailure as error:
                row.update(state='failed', error=repr(error))
            publish()
        report['state'] = 'completed' if all(r['state']=='completed' for r in report['runs']) else 'failed'
        report['axes']['execution'] = report['state']
        report['axes']['learning'] = 'observed_unreviewed; 40-step and 200-step judgments separate; inspect each run and branch intervention'
    except BaseException as error:
        report.update(state='failed', error=repr(error)); report['axes']['execution']='failed'; publish()
        if isinstance(error, CleanupFailure):
            while any(group_processes(row.get('process_group')) for row in report['commands']):
                time.sleep(.2)
        raise
    finally:
        report['finished_at']=now(); publish()
    return 0 if report['state']=='completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
