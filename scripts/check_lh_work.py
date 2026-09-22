#!/usr/bin/env python3
"""Execute bounded original-LH counting on/off and serial/parallel parity."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from durable_records import write_json
from lh_work_instrument import parse_work


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=False)
    build = json.loads((Path(args.prepared)/'build-manifest.json').read_text())
    if build['width'] > 64 or build['batch']*build['model']['vocab_size'] > 4096:
        p.error('small prepared accounting fixture required')
    runs = []
    for threads, counted in [(1, 0), (1, 1), (4, 1)]:
        target = out/f't{threads}-count{counted}'
        command = [sys.executable, 'scripts/benchmark_lh_original.py', '--device', 'cpu',
                   '--dtype', 'float32', '--prepared', args.prepared, '--output-dir', str(target),
                   '--mode', 'nograd', '--threads', str(threads), '--work-count', str(counted),
                   '--seed', '7', '--audit-logits', '--memory-gib', '32', '--timeout-seconds', '120']
        subprocess.run(command, check=True)
        events = parse_work((target/'stdout.log').read_text(), build['steps'])
        runs.append(events)
    checks = []
    for off, on, parallel in zip(*runs):
        assert off['logits'] == on['logits']
        # No tensor initialization is parallel in this original constructor.
        # Serial/parallel floating reduction may differ in the last bits.
        import torch
        torch.testing.assert_close(torch.tensor(on['logits']), torch.tensor(parallel['logits']), atol=1e-6, rtol=1e-5)
        a, b = on['metrics'], parallel['metrics']
        for key in a:
            if key.startswith('op/'):
                assert a[key] == b[key], (key, a[key], b[key])
        d = build['width']
        assert a['op/qkv_flops'] == 6*d*d*a['op/qkv_rows']
        assert a['op/out_flops'] == 2*d*d*a['op/out_rows']
        assert a['op/emit_flops'] == 2*d*d*a['op/emit_edge_rows']
        assert a['op/executed_score_elements'] >= a['op/valid_score_elements'] > 0
        assert a['op/head_rows'] == build['batch']
        checks.append(dict(step=on['step'], complete_logits=True, work_equal=True))
    write_json(out/'checks.json', dict(state='passed', checks=checks))


if __name__ == '__main__':
    main()
