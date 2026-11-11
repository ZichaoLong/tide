#!/usr/bin/env python3
"""Bounded tap-VJP and exact fresh-process continuation checks, not timing.

Use --config with a public experiment JSON to check actual output/loss, raw
state, parameters, gradients and optimizer. The global deterministic flag is
recorded and left unchanged; ordinary gather is a diagnostic comparison only.
"""
import argparse
from dataclasses import fields, is_dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import _planning_path  # noqa: F401
import torch
from durable_records import write_json
from tidegraph.comparison import collect_comparison
from tidegraph.runtime import resolve_device


def plain(value):
    if is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(plain(v) for v in value)
    return value


def observe(owner, index):
    returned, loss = owner.step(index)
    return plain(dict(returned=returned, loss=loss, state=owner.decoded_state(),
                      training=owner.training_state(), effective=owner.manifest()))


def gather_probe(device, repetitions):
    from tidegraph.weighted.column_layout import gather_column
    from tidegraph.weighted.tap_gather import tap_gather
    # True fixed-time slots: absent past is a sentinel, never clamp to row 0.
    n, queries, taps = 72, 64, 3
    index = torch.arange(queries)[:, None]-torch.arange(1, taps+1)[None, :]
    index = torch.where(index >= 0, index, n)
    values = torch.randn(n, 128, generator=torch.Generator().manual_seed(7)).to(device).requires_grad_()
    bar = torch.randn(queries, taps, 128, generator=torch.Generator().manual_seed(8)).to(device)
    record = {}
    for name, gather in (("ordinary", lambda x: gather_column(x, index.to(device))),
                         ("ordered_taps", lambda x: tap_gather(x, index))):
        grads = [torch.autograd.grad(gather(values), values, bar)[0].cpu() for _ in range(repetitions)]
        record[name] = [collect_comparison(grads[0], x, atol=0, rtol=0) for x in grads[1:]]
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--config", type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--repetitions", type=int, default=4)
    p.add_argument("--windows", type=int, default=2)
    p.add_argument("--child", type=int, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.repetitions < 2 or a.windows < 1:
        p.error("need repetitions >=2 and windows >=1")
    out = a.output_dir.resolve()
    if a.child is None:
        out.mkdir(parents=True, exist_ok=False)
    config = None
    if a.config:
        from tidegraph.experiment.config import resolve
        from tidegraph.experiment.engine import _configure
        from tidegraph.experiment.eager import Eager
        config = resolve(json.loads(a.config.read_text()), base_dir=a.config.resolve().parent)
        if config['measurement']['mode'] != 'training':
            p.error("continuation probe requires training")
        _configure(config)
        device = torch.device(config['execution']['effective']['device'])
    else:
        device, _ = resolve_device(a.device)
        torch.set_num_threads(1 if device.type == 'cpu' else 2)
        torch.set_num_interop_threads(1)
    if a.child is not None:
        owner = Eager(config, checkpoint=out/'checkpoint')
        value = observe(owner, a.windows)
        owner.close()
        torch.save(value, out/f'child-{a.child}.pt')
        return
    record = dict(state='running', deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  visibility=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), device=str(device),
                  config=config, scope='finite VJP and same-stack exact fresh-process resume; no performance claim')
    write_json(out/'result.json', record)
    try:
        record['gather'] = gather_probe(device, a.repetitions)
        if config is not None:
            owner = Eager(config)
            for index in range(a.windows):
                owner.step(index)
            (out/'checkpoint').mkdir()
            owner.save(out/'checkpoint')
            expected = observe(owner, a.windows)
            owner.close()
            record['resume'] = []
            record['commands'] = []
            for i in range(2):
                command = [sys.executable, str(Path(__file__).resolve()), '--config', str(a.config.resolve()),
                           '--windows', str(a.windows), '--output-dir', str(out), '--child', str(i)]
                record['commands'].append(command)
                with (out/f'child-{i}.log').open('w') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
                actual = torch.load(out/f'child-{i}.pt', weights_only=True)
                record['resume'].append(collect_comparison(expected, actual, atol=0, rtol=0))
                write_json(out/'result.json', record)
        checks = record['gather']['ordered_taps']+record.get('resume', [])
        record['state'] = 'passed' if all(x['strict_passed'] for x in checks) else 'failed'
    except BaseException as error:
        record.update(state='failed', error=repr(error))
        raise
    finally:
        write_json(out/'result.json', record)
    print(json.dumps(dict(state=record['state'], output=str(out/'result.json'))))
    if record['state'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
