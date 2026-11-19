#!/usr/bin/env python3
"""Executable teaching binding: instances/explicit-residual-moe.md.

Exact arithmetic, a small timed message evaluator, and an independent direct
formula check. No dependencies; not a training or performance benchmark.
"""
from fractions import Fraction as F
from itertools import combinations

EXPERTS = ('E1', 'E2', 'E3')
EDGES = tuple([('S', e, 1) for e in EXPERTS]
              + [(e, 'M', 1) for e in EXPERTS] + [('S', 'M', 2)])


def expert(node, x):
    return {'E1': lambda: 2*x, 'E2': lambda: -x, 'E3': lambda: x+5}[node]()


def select(scores):
    active = sorted(scores, key=lambda e: (-scores[e], e))[:2]
    # Checks use integer scores, so base-2 softmax is rational.
    powers = {e: F(2) ** scores[e] for e in active}
    total = sum(powers.values())
    return active, {e: powers[e]/total if e in active else F(0) for e in scores}


def run_one(x=2, counts=None, start=1):
    """Run S -> experts -> M; state is optional and copied before updating."""
    old = None if counts is None else dict(counts)
    new = None if old is None else dict(old)
    pending = {start: {'S': [('input', F(x))]}}
    trace = dict(input=F(x), old=old, messages=[], events=[], full=[], selections=0)

    def emit(node, value, time):
        for source, target, delay in EDGES:
            if source == node:
                pending.setdefault(time+delay, {}).setdefault(target, []).append((source, value))
                trace['messages'].append((time, source, target, time+delay, value))

    while pending:
        time = min(pending)
        incoming = pending.pop(time)
        trace['events'].extend((time, node) for node in incoming)
        if 'S' in incoming:
            trace['selections'] += 1
            trace['full'].append((time, 'S'))
            emit('S', sum(value for _, value in incoming['S']), time)
        candidates = {e: sum(value for _, value in incoming[e])
                      for e in EXPERTS if e in incoming}
        if candidates:
            trace['selections'] += 1
            scores = {e: (candidates[e] if e == 'E1' else 1 if e == 'E2' else 0)
                      - (3*old[e] if old is not None else 0) for e in candidates}
            active, weights = select(scores)
            values = {e: expert(e, candidates[e]) for e in active}
            weighted = {e: weights[e]*values[e] for e in active}
            trace.update(scores=scores, active=active, weights=weights,
                         values=values, weighted=weighted)
            for e in candidates:
                if new is not None:
                    new[e] = old[e] + int(e in active)
            for e in active:
                trace['full'].append((time, e))
                emit(e, weighted[e], time)
        if 'M' in incoming:
            trace['selections'] += 1
            baseline = sum(value for source, value in incoming['M'] if source == 'S')
            contribution = sum(value for source, value in incoming['M'] if source != 'S')
            trace['full'].append((time, 'M'))
            trace.update(baseline=baseline, contribution=contribution,
                         output=baseline+contribution, output_time=time)
    trace['new'] = new
    return trace


def check():
    first = run_one()
    assert first['active'] == ['E1', 'E2']
    assert first['weighted'] == {'E1': F(8, 3), 'E2': F(-2, 3)}
    assert first['output'] == 4 and first['new'] is None
    remembered = run_one(counts=dict.fromkeys(EXPERTS, 0))
    second = run_one(counts=remembered['new'], start=4)
    assert remembered['new'] == {'E1': 1, 'E2': 1, 'E3': 0}
    assert second['active'] == ['E3', 'E1'] and second['output'] == 8
    assert second['weighted'] == {'E3': F(14, 3), 'E1': F(4, 3)}
    assert second['new'] == {'E1': 2, 'E2': 1, 'E3': 1}
    assert run_one(counts=dict.fromkeys(EXPERTS, 0), start=4)['output'] == 4
    assert select({}) == ([], {})
    for size in range(1, 4):
        for subset in combinations(reversed(EXPERTS), size):
            active, weights = select(dict.fromkeys(subset, 0))
            assert active == sorted(subset)[:2]  # Independent of enumeration order.
            assert sum(weights.values()) == 1
    samples = 0
    for x in range(-2, 5):
        for counts in (None, dict.fromkeys(EXPERTS, 0), {'E1': 1, 'E2': 1, 'E3': 0}):
            result = run_one(x, counts, start=7)
            raw = [x, 1, 0]
            logits = [raw[i] - (3*counts[e] if counts else 0) for i, e in enumerate(EXPERTS)]
            indices = sorted(range(3), key=lambda i: (-logits[i], i))[:2]
            denom = sum(F(2)**logits[i] for i in indices)
            direct = F(x) + sum(F(2)**logits[i] * [2*x, -x, x+5][i] for i in indices)/denom
            assert result['output'] == direct
            assert len(result['events']) == 5 and len(result['full']) == 4
            assert result['selections'] == 3 and len(result['messages']) == 6
            assert len({(s, d) for _, s, d, _, _ in result['messages']}) == 6
            assert {arrival for _, _, d, arrival, _ in result['messages'] if d == 'M'} == {9}
            assert result['output_time'] == 9
            samples += 1
    return f'PASS: two-input replay, reset comparison, ties/subsets, {samples} timed/direct checks; 5 candidates / 2 expert Full / 6 messages'


if __name__ == '__main__':
    print(check())
