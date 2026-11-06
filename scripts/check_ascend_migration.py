#!/usr/bin/env python3
"""Small installed three-client gate for a newly selected Ascend baseline.

Reuse tidegraph check/run and the tracked configuration. Separate processes
compare uninterrupted and resumed checkpoints; no reference schedules are fed
to a candidate. This does not certify resident, multi-card or performance.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from durable_records import write_json
from source_identity import digest


def configuration(root, build, sdk, device, client):
    value = json.loads((root/'configs/checks/ascend-migration.json').read_text())
    value['execution'].update(client=client, device=device,
                              placement='cpu' if device == 'cpu' else 'mixed-c')
    value['site'] = dict(schema='tide-site-v1', name='cann92-migration', launcher='foreground',
        paths=dict(native_library=str(build/'build-python'),
                   standalone_binary=str(build/'weighted/client/weighted-benchmark')),
        environment={'PYTHONPATH':os.environ.get('PYTHONPATH', '')}, standalone_environment=sdk)
    return value


def compare_checkpoints(left, right, client):
    if client == 'libtorch':
        def payload(run):
            path = run/'client/checkpoint'
            name = json.loads((path/'latest.json').read_text())['generation']
            return path/name/'state.pt'
        assert digest(payload(left)) == digest(payload(right)), 'standalone checkpoint bytes differ'
        return dict(state='passed', scope='fresh C++ process: identical complete native checkpoint bytes after next update')
    import torch
    from tidegraph.experiment.checkpoint import locate
    from tidegraph.compare import equivalent
    a, _ = locate(left)
    b, _ = locate(right)
    for name in ('state.pt', 'task.pt'):
        equivalent(torch.load(a/name, weights_only=True, map_location='cpu'),
                   torch.load(b/name, weights_only=True, map_location='cpu'), atol=0, rtol=0)
    equivalent(torch.load(left/'outputs-000002.pt', weights_only=True, map_location='cpu'),
               torch.load(right/'outputs-000002.pt', weights_only=True, map_location='cpu'), atol=0, rtol=0)
    return dict(state='passed', scope='fresh Python process: complete checkpoint/task values and next output exact')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'npu:0'), required=True)
    args = parser.parse_args()
    root, build, out = Path(__file__).resolve().parents[1], args.build_root.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sdk = json.loads((build/'sdk-environment.json').read_text())
    report = dict(state='running', device=args.device, clients=[], physical_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                  scope='small FP32 eager migration, not resident, multi-card or throughput qualification')
    def save():
        write_json(out/'result.json', report)
    save()
    for client in ('pytorch', 'python-native', 'libtorch'):
        row = dict(client=client, state='running', commands=[])
        report['clients'].append(row)
        case = out/client
        case.mkdir()
        config = configuration(root, build, sdk, args.device, client)
        write_json(case/'config.json', config)
        def run(label, command):
            command = list(map(str, command))
            with (case/(label+'.log')).open('x') as log:
                process = subprocess.run(command, cwd=case, stdout=log, stderr=subprocess.STDOUT, timeout=900)
            row['commands'].append(dict(label=label, argv=command, exit_code=process.returncode,
                                        log_sha256=digest(case/(label+'.log'))))
            save()
            if process.returncode:
                raise RuntimeError(f'{client}/{label} exit {process.returncode}; see {case/(label+".log")}')
        command = [sys.executable, '-B', '-m', 'tidegraph']
        try:
            run('related', [*command, 'check', case/'config.json', '--scope', 'related', '--output-dir', case/'related'])
            checked = json.loads((case/'related/result.json').read_text())
            assert checked['state'] == 'passed'
            row['related'] = checked
            for label, steps in (('whole', 3), ('resumed', 1)):
                selected = deepcopy(config)
                selected['measurement'].update(mode='training', steps=steps, checkpoint_check=False)
                write_json(case/(label+'.json'), selected)
                extra = ['--resume', case/'related/training'] if label == 'resumed' else []
                run(label, [*command, 'run', case/(label+'.json'), '--output-dir', case/label, *extra])
                result = json.loads((case/label/'result.json').read_text())
                assert result['state'] == 'passed', result.get('validation')
            row['fresh_process_resume'] = compare_checkpoints(case/'resumed', case/'whole', client)
            row['state'] = 'passed'
        except Exception as error:
            row.update(state='failed', error=repr(error))
        save()
    report['state'] = 'passed' if all(x['state'] == 'passed' for x in report['clients']) else 'failed'
    save()
    print(json.dumps(report, indent=2))
    return int(report['state'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
