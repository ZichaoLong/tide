#!/usr/bin/env python3
"""Wait for fixed HB runs, evaluate unique test mappings, then time paired inference."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tide_lab.records import Metrics, atomic_json, digest, now, project_to_lens


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'project-root', 'output', 'recording-cli'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--deadline-epoch', type=float, required=True)
    a = p.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    if (plan['samples'] != 2048 or plan['max_seconds'] > 3600 or len(plan['checkpoints']) != 6
        or plan['timing'] != dict(batch_size=32, samples=64, warmups=2, repetitions=7)):
        raise ValueError('outside the third-batch evaluation budget')
    root, project = Path(a.output).resolve(), Path(a.project_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(state='waiting_dependency', started_at=now(), plan_sha256=digest(a.plan),
                  deadline_epoch=a.deadline_epoch, runs=[], missing=[],
                  axes=dict(execution='waiting_dependency', strict_numerical='original finite failures retained; no independent reference rerun',
                    discrete_semantics='held-out invariants pending; public workload counters recorded',
                    recovery='checkpoint decoding only; not exact application continuation', learning='not_checked'))
    def publish():
        atomic_json(root/'result.json', report)
    publish()
    metrics = None
    try:
        waiting = [item for item in plan['checkpoints'] if item.get('job')]
        wait_end = min(a.deadline_epoch-120, time.time()+plan['max_wait_seconds'])
        terminal = {'completed', 'failed', 'timed_out', 'stopped', 'interrupted'}
        while waiting and time.time() < wait_end:
            pending = []
            for item in waiting:
                state_path = project/item['job']/'state.json'
                state = json.loads(state_path.read_text()) if state_path.exists() else {}
                if state.get('state') not in terminal or not state.get('termination_verified'):
                    pending.append(item)
            waiting = pending
            report['waiting_for'] = [item['run'] for item in waiting]
            publish()
            if waiting:
                time.sleep(min(5, max(.01, wait_end-time.time())))
        report['wait_seconds'] = time.monotonic()-started
        import torch
        from tide_lab.heldout import validate_dataset
        from tide_lab.tasks import RetrievalTask
        from tide_lab.train import runtime, make_model, installed_identity, source_identity
        runtime('cpu', 1, 11)
        identity = installed_identity()
        for relative, expected in plan['model_source_sha256'].items():
            if digest(Path(__file__).resolve().parents[1]/relative) != expected:
                raise ValueError('fixed model/task implementation differs')
        data_path = Path(plan['dataset'])
        if digest(data_path) != plan['dataset_sha256']:
            raise ValueError('held-out dataset changed')
        data = json.loads(data_path.read_text())
        report['dataset_checks'] = validate_dataset(data)
        report['dataset_sha256'] = digest(data_path)
        report['axes']['discrete_semantics'] = 'passed held-out labels/split/unique maps/exclusions; prior HB binding checks reused'
        x = torch.tensor([r['tokens'] for r in data['examples']], dtype=torch.long)
        y = torch.tensor([r['target'] for r in data['examples']], dtype=torch.long)
        compute_start = time.monotonic()
        compute_end = min(time.monotonic()+plan['max_seconds']-70,
                          time.monotonic()+a.deadline_epoch-time.time()-70)
        def budget():
            if time.monotonic() >= compute_end:
                raise TimeoutError('evaluation budget including common deadline exhausted')
        run_id = 'tl-a2-repeat-eval2048'
        atomic_json(root/'manifest.json', dict(project='tide-learning-lab', name=run_id, run_id=run_id,
            study_id=plan['study_id'], group=plan['study_id'], config=plan, created_at=now(),
            source=source_identity(), installation=identity))
        metrics = Metrics(root, run_id)
        report.update(state='running', host_load_start=list(os.getloadavg()))
        models, rows = {}, {}
        for item in plan['checkpoints']:
            budget()
            directory = project/item['run']
            result_path = directory/'result.json'
            old = json.loads(result_path.read_text()) if result_path.exists() else {}
            if item.get('job'):
                state_path = project/item['job']/'state.json'
                life = json.loads(state_path.read_text()) if state_path.exists() else {}
                if life.get('state') != 'completed' or not life.get('termination_verified'):
                    report['missing'].append(dict(run=item['run'], reason='job not successfully terminal', lifecycle=life.get('state')))
                    publish(); continue
            if old.get('state') != 'completed' or old.get('steps') != 400:
                report['missing'].append(dict(run=item['run'], reason='400-step training not completed', state=old.get('state')))
                publish(); continue
            path = directory/'checkpoint.pt'
            config = json.loads((directory/'config.json').read_text())
            meta = json.loads((directory/'checkpoint.json').read_text())
            sha = digest(path)
            if sha != meta['sha256'] or (item.get('checkpoint_sha256') and sha != item['checkpoint_sha256']):
                raise ValueError('checkpoint hash mismatch')
            saved = torch.load(path, map_location='cpu', weights_only=False)
            if (digest(directory/'config.json') != item['config_sha256']
                or saved['config_sha256'] != item['config_sha256'] or saved['step'] != 400
                or saved['identity']['installation'] != identity or config['task'] != data['task']
                or config['dtype'] != 'float32' or config['device'] != 'cpu'
                or config['seed'] != item['seed'] or config['model']['send'] != item['send']):
                raise ValueError('checkpoint config/step/installation mismatch')
            model = make_model(config, RetrievalTask(config['task']), None).float().eval()
            model.load_state_dict(saved['model'], strict=True)
            for module in model.modules():
                if hasattr(module, 'observe'):
                    module.observe = False
            key = f"s{item['seed']}-{item['send']}"
            models[key] = model
            row = dict(key=key, seed=item['seed'], send=item['send'], checkpoint_sha256=sha,
                config_sha256=item['config_sha256'], training_run=item['run'],
                checkpoint_source=saved['identity']['source'], predictions=[], losses=[], work_totals={})
            del saved
            begin = time.monotonic()
            with torch.no_grad():
                for offset in range(0, len(x), plan['batch_size']):
                    budget()
                    out = model(x[offset:offset+plan['batch_size']])
                    if not torch.isfinite(out).all():
                        raise FloatingPointError('nonfinite held-out logits')
                    row['predictions'] += out.argmax(-1).tolist()
                    row['losses'] += torch.nn.functional.cross_entropy(out, y[offset:offset+len(out)], reduction='none').tolist()
                    for k, v in model.last_metrics.items():
                        if k in ('work/candidates', 'work/full', 'work/state_updates', 'work/messages', 'work/read_events'):
                            row['work_totals'][k] = row['work_totals'].get(k, 0)+v
            row.update(samples=len(x), accuracy=sum(a == b for a, b in zip(row['predictions'], y.tolist()))/len(x),
                       loss=statistics.mean(row['losses']), evaluation_seconds=time.monotonic()-begin)
            report['runs'].append(row); rows[key] = row
            metrics.log(len(rows), {key+'/accuracy':row['accuracy'], key+'/loss':row['loss']})
            publish()
        # Separate matched-input timed windows, alternating paired order.
        timing = plan['timing']; count = timing['samples']; batch = timing['batch_size']
        report['timing'] = dict(config=timing, records=[],
            scope='CPU FP32 one thread, no_grad full task forward and public internal counters; prepared CPU inputs, traces off; excludes loading/loss/argmax/JSONL and Lens; nonexclusive host, trained candidates have different work',
            repeated_output_check='bitwise same-model logits against own warmup; not cross-model/reference equivalence')
        reference = {}
        def forward_window(model):
            return [model(x[offset:offset+batch]) for offset in range(0, count, batch)]
        with torch.no_grad():
            for key, model in models.items():
                for _ in range(timing['warmups']):
                    budget(); reference[key] = forward_window(model)
            for rep in range(timing['repetitions']):
                for seed in (11, 22, 33):
                    order = ('dense', 'sparse') if rep%2 == 0 else ('sparse', 'dense')
                    for send in order:
                        key = f's{seed}-{send}'
                        if key not in models:
                            continue
                        budget(); begin = time.monotonic()
                        outputs = forward_window(models[key])
                        seconds = time.monotonic()-begin
                        if any(not torch.isfinite(z).all() or not torch.equal(z, ref)
                               for z, ref in zip(outputs, reference[key])):
                            raise RuntimeError('repeated fixed-checkpoint logits differ or are nonfinite')
                        report['timing']['records'].append(dict(key=key, repetition=rep, seconds=seconds,
                                                               samples=count, order=list(order)))
                        metrics.log(10+rep, {key+'/inference_seconds':seconds})
                publish()
        report['timing']['summary'] = {}
        for key in models:
            values = [r['seconds'] for r in report['timing']['records'] if r['key'] == key]
            report['timing']['summary'][key] = dict(median=statistics.median(values),
                minimum=min(values), maximum=max(values), stdev=statistics.stdev(values), repetitions=len(values))
        report['paired_seeds'] = []
        for seed in (11, 22, 33):
            dense, sparse = rows.get(f's{seed}-dense'), rows.get(f's{seed}-sparse')
            if dense and sparse:
                report['paired_seeds'].append(dict(seed=seed,
                    sparse_minus_dense_accuracy=sparse['accuracy']-dense['accuracy'],
                    sparse_minus_dense_ce=sparse['loss']-dense['loss'],
                    sparse_only_correct=sum(s == t and d != t for s, d, t in zip(sparse['predictions'], dense['predictions'], y.tolist())),
                    dense_only_correct=sum(d == t and s != t for s, d, t in zip(sparse['predictions'], dense['predictions'], y.tolist()))))
        report.update(state='completed' if not report['missing'] else 'partial',
                      compute_seconds=time.monotonic()-compute_start, host_load_end=list(os.getloadavg()))
        report['axes']['execution'] = report['state']
        report['axes']['learning'] = 'observed_unreviewed: paired seed quality and work; no automatic support or noninferiority claim'
    except BaseException as error:
        report.update(state='failed', error=repr(error)); report['axes']['execution'] = 'failed'
        raise
    finally:
        report.update(finished_at=now(), elapsed_seconds=time.monotonic()-started); publish()
        if metrics:
            metrics.close()
            atomic_json(root/'completed.json', dict(producer_finished=True, state=report['state']))
            project_to_lens(root, a.recording_cli)
    return 0 if report['state'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
