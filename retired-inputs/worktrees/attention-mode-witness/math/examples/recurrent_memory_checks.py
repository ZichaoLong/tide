#!/usr/bin/env python3
"""Finite checks for the short-convolution, GDN and SSD textbook identities.

Python 3.10+, standard library only. Inputs here are already projected
features and positive gate parameters. This checks a single-head recurrence,
not a complete neural module, regional allocator, GPU kernel or Tide graph.
The textbooks supply the general proofs and complete interface bindings.
"""

from dataclasses import dataclass
import math
import random


Vector = tuple[float, ...]
Matrix = tuple[Vector, ...]
Cache = dict[int, Vector]


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def scale(a, factor):
    return tuple(factor * x for x in a)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def outer(a, b):
    return tuple(scale(b, x) for x in a)


def madd(a, b):
    return tuple(add(x, y) for x, y in zip(a, b))


def mscale(a, factor):
    return tuple(scale(row, factor) for row in a)


def mv(a, x):
    return tuple(dot(row, x) for row in a)


def mm(a, b):
    return tuple(tuple(dot(row, col) for col in zip(*b)) for row in a)


def identity(width):
    return tuple(tuple(float(i == j) for j in range(width)) for i in range(width))


def norm(a):
    return math.sqrt(sum(x * x for row in a for x in row))


def error(a, b):
    if isinstance(a, (int, float)):
        return abs(a - b)
    if len(a) != len(b):
        raise AssertionError("shape mismatch")
    return max((error(x, y) for x, y in zip(a, b)), default=0.0)


def close(a, b, tolerance=1e-10):
    distance = error(a, b)
    assert distance <= tolerance, (distance, tolerance)


def conv(cache: Cache, time: int, current: Vector, taps: tuple[Vector, ...], stride: int):
    result = tuple(a * b for a, b in zip(taps[0], current))
    zero = (0.0,) * len(current)
    for lag, tap in enumerate(taps[1:], 1):
        history = cache.get(time - lag * stride, zero)
        result = add(result, tuple(a * b for a, b in zip(tap, history)))
    return result


def conv_write(cache: Cache, time: int, weight: float, current: Vector, size: int, stride: int):
    if weight == 0:
        return cache
    horizon = (size - 1) * stride
    result = {t: value for t, value in cache.items() if max(0, time - horizon) <= t < time}
    result[time] = scale(current, weight)
    return result


def delta_update(matrix, key, value, retain, beta):
    retained = mscale(matrix, retain)
    residual = add(value, scale(mv(retained, key), -1))
    return madd(retained, mscale(outer(residual, key), beta))


def l2(vector):
    return scale(vector, 1 / math.sqrt(1e-6 + dot(vector, vector)))


def silu(vector):
    return tuple(x / (1 + math.exp(-x)) for x in vector)


@dataclass
class State:
    time: int
    matrix: Matrix
    cache: Cache
    summary: tuple[Vector, Vector]
    mass: tuple[float, float]


@dataclass(frozen=True)
class Event:
    time: int
    weight: float
    projected: Vector
    delta: float
    beta: float


def initial():
    return State(0, ((0.0, 0.0),) * 3, {}, ((0.0, 0.0),) * 2, (0.0, 0.0))


def features(family, state, event, taps, stride):
    values = silu(conv(state.cache, event.time, event.projected, taps, stride))
    if family == "gated_delta":
        return scale(l2(values[:2]), 1 / math.sqrt(2)), l2(values[2:4]), values[4:]
    # SSD projection layout is x (value), B (write), C (read).
    return values[5:], values[3:5], values[:3]


def advance(family, state, event, taps, stride):
    if event.weight == 0:
        return state
    _, key, value = features(family, state, event, taps, stride)
    gap = event.time - state.time
    previous = mscale(state.matrix, math.exp(-0.03 * gap))
    retain = math.exp(-0.7 * event.weight * event.delta)
    if family == "gated_delta":
        matrix = delta_update(previous, key, value, retain, event.weight * event.beta)
    else:
        matrix = madd(mscale(previous, retain), mscale(outer(value, key), event.weight * event.delta))
    decays = (math.exp(-0.2 * gap), 1.0)
    summary = tuple(add(scale(z, decay), scale(event.projected[:2], event.weight))
                    for z, decay in zip(state.summary, decays))
    mass = tuple(decay * w + event.weight for w, decay in zip(state.mass, decays))
    cache = conv_write(state.cache, event.time, event.weight, event.projected, len(taps), stride)
    return State(event.time, matrix, cache, summary, mass)


def read(family, state, event, taps, stride):
    query, _, _ = features(family, state, event, taps, stride)
    return mv(mscale(state.matrix, math.exp(-0.03 * (event.time - state.time))), query)


def effective(state, time, horizon):
    gap = time - state.time
    return (mscale(state.matrix, math.exp(-0.03 * gap)),
            scale(state.summary[0], math.exp(-0.2 * gap)), state.summary[1],
            (state.mass[0] * math.exp(-0.2 * gap), state.mass[1]),
            tuple(state.cache.get(t, (0.0,) * 7) for t in range(max(0, time - horizon), time + 1)))


def run(family, events, taps, stride, state=None):
    state = initial() if state is None else state
    states, outputs = [], []
    for event in events:
        state = advance(family, state, event, taps, stride)
        states.append(state)
        outputs.append(read(family, state, event, taps, stride))
    return state, states, outputs


def check_convolution(rng):
    for size in range(1, 6):
        for stride in (1, 2, 5):
            taps = tuple(tuple(rng.uniform(-1, 1) for _ in range(3)) for _ in range(size))
            series = [tuple(rng.uniform(-1, 1) for _ in range(3)) for _ in range(13)]
            cache = {}
            for i, current in enumerate(series):
                time = 3 + i * stride
                expected = tuple(sum(taps[lag][c] * series[i - lag][c]
                                     for lag in range(min(size, i + 1))) for c in range(3))
                close(conv(cache, time, current, taps, stride), expected)
                written = conv_write(cache, time, 1, current, size, stride)
                close(conv(written, time, current, taps, stride), expected)
                assert len(written) <= size
                cache = written
    # Textbook example, including the disappearing middle event.
    taps = ((1.0,), (2.0,), (3.0,))
    for weak in (0.0, 1e-4, 0.3, 1.0):
        cache = conv_write({}, 0, 1, (4.0,), 3, 2)
        cache = conv_write(cache, 2, weak, (5.0,), 3, 2)
        close(conv(cache, 4, (6.0,), taps, 2), (18 + 10 * weak,))
    # An event-count FIFO would discard 4 even as the inserted weight vanishes.
    assert abs(4.0 - 1e-9 * 5.0) > 3.9


def check_matrix_identities(rng):
    for _ in range(200):
        matrix = tuple(tuple(rng.uniform(-1, 1) for _ in range(2)) for _ in range(3))
        key = l2(tuple(rng.uniform(-1, 1) for _ in range(2)))
        value = tuple(rng.uniform(-1, 1) for _ in range(3))
        retain, beta = rng.random(), rng.random()
        transition = mscale(madd(identity(2), mscale(outer(key, key), -beta)), retain)
        injection = mscale(outer(value, key), beta)
        close(delta_update(matrix, key, value, retain, beta), madd(mm(matrix, transition), injection))
        perturbation = tuple(tuple(rng.uniform(-1, 1) for _ in range(2)) for _ in range(3))
        assert norm(mm(perturbation, transition)) <= norm(perturbation) + 1e-12
        tiny = 1e-5
        difference = mscale(madd(delta_update(madd(matrix, mscale(perturbation, tiny)), key, value, retain, beta),
                                 mscale(delta_update(matrix, key, value, retain, beta), -1)), 1 / tiny)
        close(difference, mm(perturbation, transition), 1e-9)
        second = mscale(identity(2), rng.random())
        offset = outer(value, key)
        sequential = madd(mm(madd(mm(matrix, transition), injection), second), offset)
        composed = madd(mm(matrix, mm(transition, second)), madd(mm(injection, second), offset))
        close(sequential, composed)

        # SSD recurrence versus the explicit causal historical kernel.
        coefficients = [rng.random() for _ in range(8)]
        writes = [outer(tuple(rng.uniform(-1, 1) for _ in range(3)),
                        tuple(rng.uniform(-1, 1) for _ in range(2))) for _ in coefficients]
        query = tuple(rng.uniform(-1, 1) for _ in range(2))
        state = ((0.0, 0.0),) * 3
        for i, (coefficient, write) in enumerate(zip(coefficients, writes)):
            state = madd(mscale(state, coefficient), write)
            direct = tuple(sum(math.prod(coefficients[j + 1:i + 1]) * mv(writes[j], query)[c]
                               for j in range(i + 1)) for c in range(3))
            close(mv(state, query), direct)
    close(delta_update(((0.0,),), (0.5,), (2.0,), 0.5, 0.5), ((0.5,),))
    close(delta_update(((0.5,),), (0.5,), (2.0,), 0.5, 0.5), ((23 / 32,),))


def check_continuation_and_weak_events(rng):
    for family in ("gated_delta", "ssd"):
        for stride in (1, 2, 3):
            taps = tuple(tuple(rng.uniform(-0.8, 0.8) for _ in range(7)) for _ in range(4))
            times = (0, 2, 5, 6, 11, 17)
            events = [Event(t, w, tuple(rng.uniform(-1, 1) for _ in range(7)), 0.4 + rng.random(), 0.6)
                      for t, w in zip(times, (1.0, 0.2, 0.0, 0.6, 1.0, 0.3))]
            final, states, outputs = run(family, events, taps, stride)
            for cut in range(len(events) + 1):
                boundary, left_states, left_outputs = run(family, events[:cut], taps, stride)
                # Explicitly reconstruct all fields at the boundary.
                restored = State(boundary.time, tuple(tuple(row) for row in boundary.matrix),
                                 dict(boundary.cache), tuple(tuple(z) for z in boundary.summary), tuple(boundary.mass))
                continued, right_states, right_outputs = run(family, events[cut:], taps, stride, restored)
                assert continued == final and left_states + right_states == states
                close(left_outputs + right_outputs, outputs)
            # Delayed reads must use each saved snapshot, even after the whole run.
            close([read(family, state, event, taps, stride) for state, event in zip(states, events)], outputs)
            assert states[2] is states[1]  # zero event preserves the raw tuple

            base_events = [events[0], events[3], events[4]]
            base, base_states, base_outputs = run(family, base_events, taps, stride)
            errors = []
            raw = tuple(rng.uniform(-1, 1) for _ in range(7))
            for weak in (0.0, 1e-3, 1e-6, 1e-9):
                inserted = Event(3, weak, raw, 0.8, 0.7)
                result, trace, observed = run(family, [base_events[0], inserted, *base_events[1:]], taps, stride)
                horizon = (len(taps) - 1) * stride
                distance = max(error(effective(a, e.time, horizon), effective(b, e.time, horizon))
                               for a, b, e in zip([trace[0], *trace[2:]], base_states, base_events))
                distance = max(distance, error([observed[0], *observed[2:]], base_outputs))
                if weak == 0:
                    assert result == base
                    close(distance, 0)
                else:
                    errors.append(distance)
            assert errors[-1] < 1e-7 and errors[-1] < errors[0] * 1e-4, errors


def main():
    rng = random.Random(20261020)
    check_convolution(rng)
    check_matrix_identities(rng)
    check_continuation_and_weak_events(rng)
    print("PASS: regular-grid convolution, snapshot reads and weak-event example")
    print("PASS: 200 Delta affine/derivative checks and SSD historical expansions")
    print("PASS: both recurrences across 3 strides, every cut and disappearing events")
    print("Scope: finite single-head formula checks; no complete model or GPU validation")


if __name__ == "__main__":
    main()
