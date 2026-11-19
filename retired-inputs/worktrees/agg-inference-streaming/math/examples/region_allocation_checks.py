#!/usr/bin/env python3
"""Finite checks for the regional allocation textbook, sections 2--7.

Python 3.10+, standard library only. Scores, caps and total bounds are inputs;
this is not a neural selector or a Tide graph implementation. The quadratic
solver is checked against independent enumeration of lower/free/upper faces.
Textbook examples, including one explicit evidence/head construction, and
smooth-branch directional derivatives are also checked. Other regularizers,
general neural heads, whole-graph limits and training are outside this
program's scope; the textbooks supply the general proofs.
"""

from itertools import product
import math
import random


Vector = tuple[float, ...]


def close(actual: Vector, expected: Vector, tolerance: float = 1e-9) -> float:
    assert len(actual) == len(expected), (actual, expected)
    assert all(math.isfinite(x) for x in actual + expected)
    error = max((abs(x - y) for x, y in zip(actual, expected)), default=0.0)
    assert error <= tolerance, (actual, expected, error)
    return error


def quad_alloc(
    scores: Vector, caps: Vector, lower: float, upper: float, tau: float
) -> Vector:
    """Alloc for R(c)=-sum(c*c)/2, via the scalar multiplier (section 6.1)."""
    assert len(scores) == len(caps)
    assert all(math.isfinite(x) for x in scores + caps + (lower, upper, tau))
    assert tau > 0 and all(0 <= cap <= 1 for cap in caps)
    assert 0 <= lower <= upper and lower <= sum(caps)
    if not caps or upper == 0 or sum(caps) == 0:
        return (0.0,) * len(caps)
    if lower == sum(caps):
        return caps

    def response(multiplier: float) -> Vector:
        return tuple(
            min(cap, max(0.0, (score - multiplier) / tau))
            for score, cap in zip(scores, caps)
        )

    unconstrained = response(0.0)
    mass = sum(unconstrained)
    if lower <= mass <= upper:
        return unconstrained
    target = lower if mass < lower else upper
    # At these endpoints every coordinate is at its cap or at zero.
    left = min(score - tau * cap for score, cap in zip(scores, caps))
    right = max(scores)
    for _ in range(90):
        middle = (left + right) / 2
        if middle == left or middle == right:
            break
        if sum(response(middle)) > target:
            left = middle
        else:
            right = middle
    return response((left + right) / 2)


def face_oracle(
    scores: Vector, caps: Vector, lower: float, upper: float, tau: float
) -> Vector:
    """Independent small-region oracle: stationary points of every face.

    Each coordinate is zero, free, or at its cap. On each such face the
    total is free, fixed at lower, or fixed at upper. A concave quadratic's
    maximum is among these stationary points, including all vertices.
    This uses neither clipping responses nor root finding.
    """
    best = None
    best_value = -math.inf
    for statuses in product((0, 1, 2), repeat=len(caps)):
        free = [i for i, status in enumerate(statuses) if status == 1]
        fixed = [cap if status == 2 else 0.0 for status, cap in zip(statuses, caps)]
        for total in (None, lower, upper):
            candidate = fixed.copy()
            if free:
                multiplier = 0.0 if total is None else (
                    sum(scores[i] for i in free) - tau * (total - sum(fixed))
                ) / len(free)
                for i in free:
                    candidate[i] = (scores[i] - multiplier) / tau
            elif total is not None and abs(sum(candidate) - total) > 1e-10:
                continue
            if any(x < -1e-10 or x > cap + 1e-10 for x, cap in zip(candidate, caps)):
                continue
            if not lower - 1e-10 <= sum(candidate) <= upper + 1e-10:
                continue
            value = sum(s * x - tau * x * x / 2 for s, x in zip(scores, candidate))
            if value > best_value:
                best_value, best = value, tuple(candidate)
    assert best is not None, (scores, caps, lower, upper, tau)
    return best


def total_zero(scores: Vector, caps: Vector, total: float) -> Vector:
    """regional_total_zero; tuple order is the fixed tie order."""
    assert len(scores) == len(caps) and 0 <= total <= sum(caps)
    result = [0.0] * len(caps)
    remaining = total
    for i in sorted(range(len(caps)), key=lambda i: (-scores[i], i)):
        result[i] = min(caps[i], max(0.0, remaining))
        remaining -= result[i]
    return tuple(result)


def hard_select(adjusted_scores: Vector, caps: Vector, k: int) -> Vector:
    """Shared hard rule, with either regional or node-gain caps."""
    assert len(adjusted_scores) == len(caps) and k >= 1
    order = sorted(
        (i for i, cap in enumerate(caps) if cap > 0),
        key=lambda i: (-adjusted_scores[i], i),
    )
    positive = [i for i in order if adjusted_scores[i] > 0]
    selected = set(positive[:k] if positive else order[:1])
    return tuple(cap if i in selected else 0.0 for i, cap in enumerate(caps))


def central_difference(function, step: float = 1e-5) -> Vector:
    plus, minus = function(step), function(-step)
    return tuple((x - y) / (2 * step) for x, y in zip(plus, minus))


def check_textbook_examples() -> int:
    cases = []
    # Sections 2.5 and 5.2: total control and the fixed-threshold rule.
    scores, caps = (2.0, 1.0, 100.0), (1.0, 1.0, 0.0)
    for total, expected in (
        (1.4, (0.95, 0.45, 0.0)),
        (1.5, (1.0, 0.5, 0.0)),
        (1.6, (1.0, 0.6, 0.0)),
    ):
        cases.append((scores, caps, total, total, 2.0, expected))
    cases.extend([
        (scores, caps, 1.0, 1.0, 2.0, (0.75, 0.25, 0.0)),
        (scores, caps, 1.4, 1.4, 0.5, (1.0, 0.4, 0.0)),
        (tuple(s - 0.8 for s in scores), caps, 1.0, 2.0, 0.5, (1.0, 0.4, 0.0)),
        (scores, caps, 1.0, 1.0, 0.5, (1.0, 0.0, 0.0)),
    ])
    # Section 5.3: weak receive leaves room for a regional total head.
    for total, expected in (
        (0.4, (0.4, 0.0)), (0.6, (0.4, 0.2)), (0.8, (0.4, 0.4))
    ):
        cases.append(((1.0, 0.0), (0.4, 0.4), total, total, 1.0, expected))
    cases.append(((1.0, 0.0), (0.4, 0.4), 0.8, 1.0, 1.0, (0.4, 0.4)))
    # Sections 5.5 and 7.3.2: exact sparsity at finite positive temperature.
    for tau, expected in (
        (5.0, (0.8, 0.6, 0.4, 0.2)),
        (2.0, (1.0, 0.75, 0.25, 0.0)),
        (0.4, (1.0, 1.0, 0.0, 0.0)),
    ):
        cases.append(((4.0, 3.0, 2.0, 1.0), (1.0,) * 4, 1.0, 2.0, tau, expected))
    for tau, expected in (
        (2.0, (0.75, 0.25, 0.0, 0.0)), (1.0, (1.0, 0.0, 0.0, 0.0))
    ):
        cases.append(((4.0, 3.0, 2.0, 1.0), (1.0,) * 4, 1.0, 1.0, tau, expected))
    for scores, caps, lower, upper, tau, expected in cases:
        close(quad_alloc(scores, caps, lower, upper, tau), expected)
        close(face_oracle(scores, caps, lower, upper, tau), expected)
    return len(cases)


def check_evidence_example() -> int:
    """Sections 2.5 and 3.4--3.5: actual cat projection and two-unit SiLU head."""
    def normalize(x: float) -> float:
        return x / math.sqrt(1 + x*x)  # d=1, epsilon=1

    def silu(x: float) -> float:
        return x / (1 + math.exp(-x))

    content, received = (1.0, -1.0, 0.0), (1.0, 1.0, 0.0)
    projection = (-math.sqrt(2/3), math.sqrt(2/3), 0.0)
    for scale, expected_z, expected_fraction, expected_control in (
        (1.0, -0.5, 0.4, (0.95, 0.45, 0.0)),
        (0.0, 0.0, 0.5, (1.0, 0.5, 0.0)),
        (-1.0, 0.5, 0.6, (1.0, 0.6, 0.0)),
    ):
        evidence = tuple(normalize(scale*a*normalize(u)) for a, u in zip(projection, content))
        close(evidence, (expected_z, expected_z, 0.0))
        pooled = sum(w*z for w, z in zip(received, evidence)) / max(1, sum(received))
        hidden = 2*math.log(3/2)*pooled
        head = silu(hidden) - silu(-hidden)
        fraction = 1 / (1 + math.exp(-head))
        close((pooled, fraction), (expected_z, expected_fraction))
        total = 1 + fraction
        scores, caps = (2.0, 1.0, 100.0), (1.0, 1.0, 0.0)
        close(quad_alloc(scores, caps, total, total, 2.0), expected_control)
        close(face_oracle(scores, caps, total, total, 2.0), expected_control)
    return 3


def check_small_regions() -> tuple[int, float]:
    rng = random.Random(20261022)
    cases = [
        ((), (), 0.0, 1.0, 1.0),
        ((4.0, -2.0), (0.0, 0.0), 0.0, 2.0, 1.0),
        ((4.0, -2.0), (0.7, 0.8), 0.0, 0.0, 1.0),
        ((-4.0, -2.0), (0.7, 0.8), 1.5, 2.0, 1.0),
        ((0.4, 0.5, 0.6), (1.0,) * 3, 1.0, 3.0, 1.0),
    ]
    for index in range(80):
        size = 1 + index % 4
        scores = tuple(rng.uniform(-2, 3) for _ in range(size))
        caps = tuple(0.0 if rng.random() < 0.2 else rng.uniform(0.1, 1) for _ in range(size))
        mass = sum(caps)
        lower = rng.uniform(0, mass)
        upper = lower if index % 3 == 0 else rng.uniform(lower, mass)
        cases.append((scores, caps, lower, upper, rng.uniform(0.2, 3)))
    maximum_error = 0.0
    for scores, caps, lower, upper, tau in cases:
        actual = quad_alloc(scores, caps, lower, upper, tau)
        oracle = face_oracle(scores, caps, lower, upper, tau)
        maximum_error = max(maximum_error, close(actual, oracle))
    return len(cases), maximum_error


def check_derivatives() -> tuple[int, float]:
    scores, caps = (2.0, 1.0, 100.0), (1.0, 1.0, 0.0)
    checks = [
        (lambda h: quad_alloc(scores, caps, 1.4 + h, 1.4 + h, 0.5), (0.0, 1.0, 0.0)),
        (lambda h: quad_alloc(tuple(s - 0.8 - h for s in scores), caps, 1.0, 2.0, 0.5),
         (0.0, -2.0, 0.0)),
        (lambda h: quad_alloc((2.0, 1.0 + h, 100.0), caps, 1.4, 1.4, 0.5),
         (0.0, 0.0, 0.0)),
        (lambda h: quad_alloc((1.2, 0.2 + h, 99.2), caps, 1.0, 2.0, 0.5),
         (0.0, 2.0, 0.0)),
        # Two free coordinates with a fixed total: centered score derivative.
        (lambda h: quad_alloc((2.0 + h, 1.0, 100.0), caps, 1.4, 1.4, 2.0),
         (0.25, -0.25, 0.0)),
        # Slack total: a common score shift is no longer canceled.
        (lambda h: quad_alloc((0.4 + h, 0.5 + h, 0.6 + h), (1.0,) * 3, 1.0, 3.0, 1.0),
         (1.0, 1.0, 1.0)),
        # Temperature derivative uses actual response points, not bracket points.
        (lambda h: quad_alloc((0.6, 0.4), (1.0, 1.0), 1.0, 1.0, 1.0 + h),
         (-0.1, 0.1)),
        # Simultaneous score, cap, total and temperature directions (section 6.2).
        # At h=0 the active faces are: upper, free, free, zero.
        (lambda h: quad_alloc(
            (1.5 + 0.2*h, 0.7 - 0.1*h, 0.4 + 0.05*h, -1.0 + 0.3*h),
            (0.5 + 0.04*h, 0.9 + 0.02*h, 0.8 - 0.01*h, 0.8 + 0.07*h),
            1.4 + 0.06*h, 1.4 + 0.06*h, 1.0 + 0.03*h),
         (0.04, -0.0695, 0.0895, 0.0)),
    ]
    maximum_error = max(close(central_difference(f), expected, 1e-8) for f, expected in checks)
    return len(checks), maximum_error


def check_zero_and_hard() -> int:
    count = 0
    for total, expected in ((1.4, (1.0, 0.4, 0.0)), (0.0, (0.0, 0.0, 0.0))):
        close(total_zero((2.0, 1.0, 100.0), (1.0, 1.0, 0.0), total), expected)
        count += 1
    close(total_zero((), (), 0.0), ())
    close(hard_select((), (), 1), ())
    close(hard_select((3.0, -2.0), (0.0, 0.0), 1), (0.0, 0.0))
    count += 3
    # Fractional caps separate the constrained zero limit from hard selection.
    scores, caps = (-2.0, -1.0), (0.9, 0.8)
    close(total_zero(scores, caps, 1.0), (0.2, 0.8))
    close(hard_select(scores, caps, 1), (0.0, 0.8))
    for tau in (0.5, 0.1, 0.01):
        close(quad_alloc(scores, caps, 1.0, 1.0, tau), (0.2, 0.8))
        count += 1
    count += 2
    # Equal scores retain a fractional optimum; deterministic ties pick a vertex.
    for tau in (2.0, 0.5, 0.01):
        close(quad_alloc((1.0, 1.0), (1.0, 1.0), 1.0, 1.0, tau), (0.5, 0.5))
        count += 1
    close(total_zero((1.0, 1.0), (1.0, 1.0), 1.0), (1.0, 0.0))
    close(hard_select((1.0, 1.0), (1.0, 1.0), 1), (1.0, 0.0))
    count += 2
    # A score tending to the threshold can retain half strength at every tau.
    for tau in (0.5, 0.1, 0.01):
        close(quad_alloc((1.0, tau/2), (1.0, 1.0), 1.0, 2.0, tau), (1.0, 0.5))
        count += 1
    close(hard_select((1.0, 0.0), (1.0, 1.0), 2), (1.0, 0.0))
    return count + 1


def main() -> None:
    examples = check_textbook_examples()
    evidence = check_evidence_example()
    regions, solver_error = check_small_regions()
    derivatives, derivative_error = check_derivatives()
    limits = check_zero_and_hard()
    print(f"PASS: {examples} textbook allocation examples")
    print(f"PASS: {evidence} evidence/head examples from content through allocation")
    print(f"PASS: {regions} small regions vs independent face enumeration; max error {solver_error:.3g}")
    print(f"PASS: {derivatives} smooth-branch derivatives; max error {derivative_error:.3g}")
    print(f"PASS: {limits} zero/hard values and limit counterexample checks")


if __name__ == "__main__":
    main()
