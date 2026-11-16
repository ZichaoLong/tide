#!/usr/bin/env python3
"""Administrative continuation; execute all scientific stages from frozen 62e32602.

This file is a separately pinned orchestration input. The job cwd/PYTHONPATH,
stage scripts, models and checkpoint identity remain the original repair source.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path.cwd()))
from scripts.run_hb_v12_repair_worker import RepairWorker
from tide_lab.records import atomic_json, digest, now
from tide_lab.supervisor import verify_source

MODEL_COMMIT = '62e32602b23e371eb1669f7b99409db626a1c4f8'
DEVICES = [4, 6, 9, 11, 12, 14]


def validate_plan(plan, source, policy, deadline):
    if source['commit'] != MODEL_COMMIT or plan['model_source'] != source['commit']:
        raise ValueError('administrative continuation cannot migrate model source')
    if (policy['candidate_ids'] != DEVICES or policy['max_new_concurrent_npus'] != 6
            or policy['slot_namespace'] != 'tide-direct-v10'
            or policy['deadline_epoch'] != deadline or plan['deadline_epoch'] != deadline):
        raise ValueError('six-card physical placement/deadline contract changed')
    if len(plan['cases']) != 1 or plan['administrative_mode'] not in ('resume_train', 'requeue_unstarted'):
        raise ValueError('one original repair chain per service')
    case = plan['cases'][0]
    config = json.loads(Path(case['config']).read_text())
    if config['training']['target_tokens'] != 128*2**20:
        raise ValueError('original128Mi endpoint required')
    if digest(case['config']) != plan['original_config_sha256']:
        raise ValueError('original configuration changed')
    if digest(case['parent_checkpoint']) != case['parent_sha256']:
        raise ValueError('fixed parent changed')


class SixCardWorker(RepairWorker):
    def stage(self, case, name, action, cap, extra=(), record=False, script='scripts/campaign_stage.py'):
        # Run IDs distinguish the administrative segment; learning inputs do not change.
        rename = lambda value: str(value).replace('tl-v12-repair-', 'tl-v12-six-')
        if 'train_arguments' in case:
            case['train_arguments'] = [rename(x) for x in case['train_arguments']]
        return super().stage(case, name, action, cap, [rename(x) for x in extra], record, script)

    def run(self):
        if self.plan['administrative_mode'] == 'requeue_unstarted':
            return super().run()
        case = self.plan['cases'][0]
        out = Path(case['output']); out.mkdir(parents=True, exist_ok=False)
        row = dict(id=case['id'], state='running', stages=[], started_at=now(),
                   administrative_parent=case['administrative_parent'])
        self.state['cases'].append(row); self.publish(); self.guard()
        config = json.loads(Path(case['config']).read_text())

        def require(name, action, cap=1200, extra=(), record=False, script='scripts/hb_v12_repair_stage.py'):
            result = self.stage(case, name, action, cap, extra, record, script)
            if not result or result.get('state') != 'completed':
                raise RuntimeError('administrative continuation stage failed: '+name)
            return result

        parent = Path(case['parent_checkpoint'])
        # Verify the saved trained state on the allocated card, before scientific continuation.
        require('recovery-full', 'segment', extra=['--resume', parent, '--steps', '2'])
        require('recovery-one', 'segment', extra=['--resume', parent, '--steps', '1'])
        require('recovery-resumed', 'segment', extra=['--resume', out/'recovery-one/checkpoint.pt', '--steps', '1'])
        require('recovery-compare', 'compare', extra=['--checkpoint-a', out/'recovery-full/checkpoint.pt',
                                                   '--checkpoint-b', out/'recovery-resumed/checkpoint.pt'])
        temporary = [out/name/'checkpoint.pt' for name in ('recovery-full', 'recovery-one', 'recovery-resumed')]
        atomic_json(out/'recovery-retirement.json', dict(at=now(), reason='successful disposable recovery; administrative parent retained',
            files=[dict(path=str(p), sha256=digest(p), bytes=p.stat().st_size) for p in temporary]))
        for path in temporary: path.unlink()
        case['train_arguments'] = ['--until-tokens', str(128*2**20), '--run-id', 'tl-v12-six-main-'+case['id'],
                                   '--recording-group', config['study_id']]
        final = require('train', 'train', self.a.deadline_epoch-time.time()-1800,
            ['--resume', parent, '--admission', case['admission'], '--recovery', out/'recovery-compare/recovery.json'],
            True, 'scripts/campaign_stage.py')
        if final['progress']['tokens'] != 128*2**20:
            raise ValueError('original endpoint not reached')
        import torch
        saved = torch.load(out/'train/latest.pt', map_location='cpu', weights_only=False)
        if saved['progress']['tokens'] != 128*2**20 or 'optimizer' not in saved:
            raise ValueError('complete final recovery artifact missing')
        del saved
        training_data = self.a.data; self.a.data = Path(self.plan['confirmation_data'])
        try:
            require('confirmation', None, 1200, ['--training-data', training_data, '--checkpoint', out/'train/latest.pt',
                '--lens-cli', self.a.recording_cli, '--run-id', 'tl-v12-six-confirm-'+case['id'],
                '--recording-group', 'TL-HB-CONFIRM-v12'], script='scripts/evaluate_capacity_v11.py')
        finally:
            self.a.data = training_data
        table = out/'diagnostic-table.json'
        atomic_json(table, dict(cases=[dict(id=case['id'], config=case['config'], checkpoint=str(out/'train/latest.pt'),
            sha256=digest(out/'train/latest.pt'), history=dict(administrative_parent=str(parent),
                                                            original_run=case['original_run']))]))
        require('diagnostic', None, 1800, ['--table', table, '--root', self.plan['root'], '--lens-cli', self.a.recording_cli,
            '--run-id', 'tl-v12-six-diag-'+case['id'], '--recording-group', 'TL-HB-DIAG-v12'], script='scripts/diagnose_hb_v12.py')
        row.update(state='completed', finished_at=now())
        self.state.update(state='completed', finished_at=now()); self.publish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'data', 'output', 'recording-cli'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    policy = json.loads(Path(os.environ['TIDE_LAB_RESOURCE_POLICY']).read_text())
    model_source = verify_source(Path.cwd())
    validate_plan(plan, model_source, policy, args.deadline_epoch)
    orchestration = verify_source(Path(__file__).resolve().parents[1])
    worker = SixCardWorker(args)
    worker.state.update(model_source=model_source['commit'], orchestration_source=orchestration['commit'],
                        administrative_mode=plan['administrative_mode'])
    worker.publish()
    try:
        worker.run()
    except BaseException as error:
        if worker.state['cases']:
            worker.state['cases'][-1].update(state='failed', error=repr(error), finished_at=now())
        worker.state.update(state='failed', error=repr(error), finished_at=now()); worker.publish()
        raise
