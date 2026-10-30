"""Copyable core-5 CPU training/continuation client; not a model-quality recipe."""
import argparse
import json
from pathlib import Path
from recording import application_identity, json_record, sha256, tensor_record


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe', type=Path, default=Path(__file__).with_name('recipe.json'))
    p.add_argument('--foundation-lock', type=Path, required=True)
    p.add_argument('--device', choices=('cpu',), required=True)
    p.add_argument('--dtype', choices=('float32', 'float64'), default='float64')
    p.add_argument('--client', choices=('pytorch', 'python-native'), default='pytorch')
    p.add_argument('--native-library')
    p.add_argument('--workers', type=int, default=1, help='native forward workers; PyTorch requires 1')
    p.add_argument('--schedule', choices=('streaming', 'prefill'), default='prefill')
    p.add_argument('--packing', choices=('scalar', 'packed'), default='packed')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--steps', type=int, default=2, help='additional updates; bounded toy check, 1..8')
    p.add_argument('--resume', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    return p


def setup(a):
    import torch
    from tidegraph.planning import ExecutionSpec
    from dependency import verify
    from task import Task
    if not 1 <= a.steps <= 8 or not 0 <= a.seed < 2**63-2:
        raise ValueError('steps must be 1..8 and seed in [0,2**63-2)')
    if (a.client == 'python-native') != (a.native_library is not None):
        raise ValueError('python-native requires an explicit matching native library; pytorch uses none')
    lock, observed = verify(a.foundation_lock, native_library=a.native_library)
    execution = ExecutionSpec(client=a.client, device=a.device, dtype=a.dtype, scoring_dtype=a.dtype,
        schedule=a.schedule, packing=a.packing, workers=a.workers).record()
    recipe = json.loads(a.recipe.read_text())
    task = Task(recipe, execution, a.seed, native_library=a.native_library)
    if lock['mathematics'] != dict(version='tide-core-5', commit=task.plan['upstream']):
        raise ValueError('locked mathematics differs from actual configuration')
    return task, observed


def main():
    a = parser().parse_args()
    import torch
    from checkpoint import save_application, restore_application
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    task, dependency = setup(a)
    if a.resume:
        restore_application(a.resume, task, dependency)
    # Invalid lock/config/resume must not leave a final directory blocking retry.
    a.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(schema='example-core5-run-v1', state='running', dependency=dependency,
        application=application_identity(), facts=task.facts(), initial_step=task.step_index,
        resume_sha256=sha256(a.resume) if a.resume else None,
        scope='small CPU integration only; no throughput or model-quality claim')
    try:
        losses = []
        for _ in range(a.steps):
            actual = task.step()
            losses.append(float(actual['loss']))
        tensor_record(a.output_dir/'observation.pt', actual)
        save_application(a.output_dir/'checkpoint.pt', task, dependency)
        predictions = task.infer()
        if not torch.isfinite(predictions).all():
            raise FloatingPointError('nonfinite inference')
        report.update(state='passed', final_step=task.step_index, losses=losses,
            inference_shape=list(predictions.shape), checkpoint_sha256=sha256(a.output_dir/'checkpoint.pt'),
            observation_sha256=sha256(a.output_dir/'observation.pt'))
    except BaseException as error:
        report.update(state='failed', error=repr(error))
        raise
    finally:
        json_record(a.output_dir/'result.json', report)
    print(json.dumps(dict(state=report['state'], final_step=task.step_index)))


if __name__ == '__main__':
    main()
