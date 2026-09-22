#!/usr/bin/env python3
"""Validate original Think/token pairs and summarize the saved reproduction log."""
import argparse
import json
from pathlib import Path
import re
import statistics


def summarize(text, config, native_exit_code):
    times, pending, parameters, diagnostics = [], None, None, {}
    for line in text.splitlines():
        if match := re.fullmatch(r'Think took (\d+) ms\s*', line):
            if pending is not None:
                raise ValueError('Think without token completion')
            pending = int(match[1])
        elif match := re.fullmatch(r't: (\d+)\s*', line):
            if pending is None or int(match[1]) != len(times):
                raise ValueError('invalid Think/token sequence')
            times.append(pending)
            pending = None
        elif match := re.search(r'参数量 = (\d+)', line):
            parameters = int(match[1])
        elif line.startswith(('LH_ENV ', 'LH_MODEL ', 'LH_DONE ')):
            name, value = line.split(' ', 1)
            diagnostics[name] = json.loads(value)
    batch, steps = config['batch'], config['steps']
    result = dict(status='completed' if native_exit_code == 0 and len(times) == steps
        and parameters == config['expected_parameters'] and pending is None else 'failed-or-incomplete',
        native_exit_code=native_exit_code, parameters=parameters, completed_steps=len(times),
        configured_steps=steps, batch=batch, raw_think_ms_per_batch=times,
        timing='original integer-ms Think timer; includes cold first step and inner timer printing',
        seed_policy=config['seed_policy'], diagnostics=diagnostics, windows={})
    for name, lo, hi in [('all', 0, steps), ('after-first-4', 4, steps),
                         ('early-4-11', 4, 12), ('late-80-99', 80, 100)]:
        sample = times[lo:hi]
        if hi > lo and len(sample) == hi-lo and sum(sample) > 0:
            result['windows'][name] = dict(token_indices=[lo, hi-1],
                mean_ms_per_sample_token=sum(sample)/(batch*len(sample)),
                tokens_per_second=1000*batch*len(sample)/sum(sample),
                median_ms_per_sample_token=statistics.median(sample)/batch,
                min_ms_per_sample_token=min(sample)/batch, max_ms_per_sample_token=max(sample)/batch)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', required=True)
    p.add_argument('--configured', required=True)
    p.add_argument('--native-exit-code', required=True, type=int)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    config = json.loads(Path(a.configured).read_text(encoding='utf-8'))
    result = summarize(Path(a.log).read_text(encoding='utf-8'), config, a.native_exit_code)
    with Path(a.output).open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('diagnostics','raw_think_ms_per_batch')}, indent=2))
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
