"""Live native event ingestion and comparable warm-window summaries."""
import json
import math
import re
import statistics
import time
from pathlib import Path
from experiment_record import utc_now


class Collector:
    def __init__(self, engine, args, out, run_id, expected):
        self.engine, self.args, self.out, self.run_id = engine, args, Path(out), run_id
        self.expected = expected; self.events = []; self.pending_ms = None; self.token = None
        self.parameters = None; self.native_offset = 0; self.started = time.monotonic()
        self.runtime = None
        (self.out/'metrics.jsonl').touch()

    def add(self, event):
        m = event['metrics']; index = len(self.events)
        if (event['step'] != index or event['sequence'] != index or event['run_id'] != self.run_id
                or event.get('schema_version') != 1 or not m
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in m.values())):
            raise ValueError('invalid/nonfinite/out-of-order native metric event')
        if m.get('model/parameters') != self.expected:
            raise ValueError('native parameter count differs from configured count')
        self.events.append(event)
        with (self.out/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(event, allow_nan=False)+'\n'); f.flush()
        runtime = {k: v for k, v in m.items() if k.startswith('runtime/')}
        if self.runtime is None:
            self.runtime = runtime
            print('RUNTIME '+json.dumps(runtime, sort_keys=True), flush=True)
        ms = m['perf/ms_per_sample_token']; phase = 'warmup' if index < self.args.warmup else 'measure'
        work = ''
        if 'op/qkv_rows' in m:
            work = f" qkv_rows/sample={m['op/qkv_rows']/self.args.batch:.3f} attention_calls/batch={m['op/attention_calls']:.0f}"
        print(f'{self.engine.upper()} token={index} {phase} ms/sample-token={ms:.6f}'+work, flush=True)

    def line(self, line):
        outer = getattr(self.args, 'lh_timer', 'original') == 'outer'
        if line.startswith(("OpenBLAS ", "ATen parallel backend:")):
            print(line, flush=True)
        if self.engine == 'pdg':
            if line.startswith('MODEL '):
                print(line, flush=True)
            if line.startswith('STEP '):
                self.drain_native()
            return
        if '参数量 = ' in line:
            self.parameters = int(re.search(r'参数量 = (\d+)', line)[1])
            print('MODEL parameters='+str(self.parameters), flush=True)
        if match := re.fullmatch(r'Think took (\d+) ms\s*', line):
            if self.pending_ms is not None:
                raise ValueError('Think timer missing completion')
            self.pending_ms = int(match[1])
        elif match := re.fullmatch(r't: (\d+)\s*', line):
            if (not outer and self.pending_ms is None) or self.token is not None or int(match[1]) != len(self.events):
                raise ValueError('Think timer/token order mismatch')
            self.token = int(match[1])
        elif line.startswith('WORK '):
            data = json.loads(line[5:])
            if self.token is None or data['step'] != self.token:
                raise ValueError('work event without completed Think')
            m = data['metrics']; m['model/parameters'] = self.parameters
            milliseconds = m['perf/token_seconds']*1000 if outer else self.pending_ms
            m['perf/ms_per_sample_token'] = milliseconds/self.args.batch
            m['perf/think_ms_per_batch'] = milliseconds
            if milliseconds:
                m['perf/sample_tokens_per_second'] = 1000*self.args.batch/milliseconds
            event = dict(schema_version=1, run_id=self.run_id, sequence=len(self.events), step=self.token,
                         timestamp=utc_now(), elapsed_seconds=time.monotonic()-self.started, metrics=m,
                         context=dict(phase='warmup' if self.token < self.args.warmup else 'measure'))
            self.add(event); self.pending_ms = self.token = None

    def drain_native(self):
        path = self.out/'native/metrics.jsonl'
        if not path.exists():
            return
        with path.open('rb') as stream:
            stream.seek(self.native_offset)
            for line in stream:
                if not line.endswith(b'\n'):
                    break
                self.add(json.loads(line)); self.native_offset = stream.tell()

    def finish(self):
        if self.engine == 'pdg':
            self.drain_native()
        if len(self.events) != self.args.steps or self.pending_ms is not None or self.token is not None:
            raise ValueError('incomplete token inventory')
        if self.args.work_count:
            for event in self.events:
                m = event['metrics']
                add = getattr(self.args, 'memory', 'attention') == 'add'
                if (not add and m.get('op/qkv_flops', 0) <= 0
                    or add and (m.get('op/qkv_flops', 0) != 0 or m.get('op/head_flops', 0) <= 0)
                    or m['op/executed_score_elements'] < m['op/valid_score_elements']):
                    raise ValueError('work inventory missing or inconsistent')
        return summarize(self.events, self.args.batch, self.args.warmup)


def summarize(events, batch, warmup):
    measured = [e['metrics'] for e in events if e['step'] >= warmup]
    if not measured:
        return dict(observations=len(events), measured_observations=0)
    times = [m['perf/ms_per_sample_token'] for m in measured]
    result = dict(observations=len(events), measured_observations=len(measured),
        token_indices=[warmup, events[-1]['step']], mean_ms_per_sample_token=statistics.mean(times),
        median_ms_per_sample_token=statistics.median(times), min_ms_per_sample_token=min(times),
        max_ms_per_sample_token=max(times), population_stdev_ms=statistics.pstdev(times),
        sample_tokens_per_second=len(times)*1000/sum(times) if sum(times) else None, raw_ms=times)
    keys = [k for k in measured[0] if k.startswith(('op/', 'profile/', 'detail/'))]
    result['mean_metrics_per_batch_token'] = {k: statistics.mean(m[k] for m in measured) for k in keys}
    if 'op/qkv_rows' in measured[0]:
        counts = result['mean_metrics_per_batch_token']
        result['mean_counted_gflops_per_sample_token'] = counts['op/executed_matmul_flops']/batch/1e9
        result['attention_padding_ratio'] = (counts['op/executed_score_elements']/counts['op/valid_score_elements']
                                             if counts['op/valid_score_elements'] else None)
    return result
