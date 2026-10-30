"""Independent oracle and fresh-process application-resume gate for this starter."""
import json
from pathlib import Path
import subprocess
import sys
import torch
from train import parser, setup
from recording import json_record, sha256
from task import Task


def equivalent(a, b, *, exact=False):
    # Validation belongs to the experiment; no benchmark-script dependencies.
    if isinstance(a, torch.Tensor):
        tol = (0, 0) if exact or not a.is_floating_point() else ((1e-10, 1e-8) if a.dtype == torch.float64 else (1e-6, 1e-5))
        torch.testing.assert_close(a, b, atol=tol[0], rtol=tol[1], equal_nan=False)
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        for key in a: equivalent(a[key], b[key], exact=exact)
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b): equivalent(x, y, exact=exact)
    else:
        assert a == b


def main():
    a = parser().parse_args()
    if a.resume:
        raise ValueError('check creates its own fresh-process restore input')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    task, dependency = setup(a)
    oracle = Task(json.loads(a.recipe.read_text()), task.execution, a.seed, reference=True)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    record = dict(schema='example-core5-check-v1', state='running', dependency=dependency, facts=task.facts(), checks={})
    try:
        families = set(task.facts()['configuration']['families'])
        assert task.plan['graph']['edges'], 'starter gate requires actual body edges'
        for _ in range(2):
            rng = torch.get_rng_state()
            actual = task.step()
            torch.set_rng_state(rng)
            expected = oracle.step()
            equivalent(actual, expected)
            touched = {task.plan['memories'][e['node']].get('memory_family', 'kv') for e in actual['result']['trace']}
            assert touched == families
        record['checks']['independent_two_step_complete_records_gradients_sgd_scheduler'] = True
        base = [sys.executable, str(Path(__file__).with_name('train.py')), '--device', a.device,
                '--recipe', str(a.recipe.resolve()), '--foundation-lock', str(a.foundation_lock.resolve()),
                '--client', a.client, '--dtype', a.dtype, '--seed', str(a.seed), '--workers', str(a.workers),
                '--schedule', a.schedule, '--packing', a.packing]
        if a.native_library: base += ['--native-library', str(Path(a.native_library).resolve())]
        commands = [('whole', ['--steps','3']), ('prefix', ['--steps','2']),
                    ('resumed', ['--steps','1','--resume',str(a.output_dir.resolve()/'prefix/checkpoint.pt')])]
        for name, extra in commands:
            with (a.output_dir/(name+'.log')).open('w') as log:
                subprocess.run([*base,*extra,'--output-dir',str(a.output_dir.resolve()/name)],
                               check=True, stdout=log, stderr=subprocess.STDOUT, timeout=90)
        load = lambda name: torch.load(a.output_dir/name, map_location='cpu', weights_only=True)
        equivalent(load('whole/observation.pt'), load('resumed/observation.pt'), exact=True)
        # Compare exact decoded graph state too, not serialization byte order.
        import io
        whole, restored = load('whole/checkpoint.pt'), load('resumed/checkpoint.pt')
        for record_value in (whole, restored):
            record_value['graph'] = torch.load(io.BytesIO(record_value['graph']), map_location='cpu', weights_only=True)
        equivalent(whole, restored, exact=True)
        record['checks']['fresh_process_application_resume_exact'] = True
        record['checks']['inference'] = all(json.loads((a.output_dir/n/'result.json').read_text())['inference_shape'][0] > 0
                                            for n in ('whole','prefix','resumed'))
        record.update(state='passed', resume_checkpoint_sha256=sha256(a.output_dir/'prefix/checkpoint.pt'))
    except BaseException as error:
        record.update(state='failed', error=repr(error)); raise
    finally:
        json_record(a.output_dir/'result.json', record)
    print(json.dumps(dict(state=record['state'], checks=record['checks'])))


if __name__ == '__main__':
    main()
