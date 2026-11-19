#!/usr/bin/env python3
"""Finite witnesses for architecture §2.3–2.4 and attention parameter counts.

Standard-library scalar reference, not a training or device implementation.
Closed-form one-/two-head cases and invariances check different requirements.
"""
from dataclasses import dataclass, replace
from itertools import permutations
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference'))
from backbones import capacity


def matvec(matrix, vector):
    return [math.fsum(a*b for a, b in zip(row, vector)) for row in matrix]


def normalize(vector, epsilon):
    scale = math.sqrt(epsilon+math.fsum(x*x for x in vector)/len(vector))
    return [x/scale for x in vector]


def silu(x):
    return x/(1+math.exp(-x)) if x >= 0 else x*math.exp(x)/(1+math.exp(x))


@dataclass(frozen=True)
class Parameters:
    d: int
    heads: int
    epsilon: float
    q: list
    k: list
    value: list
    out: list
    source_embedding: dict
    source_a: list
    source_b: list
    source_bias: dict
    scale: float = 1.0
    alpha: float = 1.0

    def residual(self, vector, source):
        hidden = matvec(self.source_a, normalize(vector, self.epsilon))
        hidden = [silu(x+b) for x, b in zip(hidden, self.source_bias[source])]
        correction = matvec(self.source_b, hidden)
        return [x+self.alpha*y for x, y in zip(vector, correction)]


def aggregate(messages, params):
    """Messages are (strength, content, source-class); atom order is arbitrary."""
    positive = [(w, list(z), a) for w, z, a in messages if w > 0]
    total = math.fsum(w for w, _, _ in positive)
    if total == 0:
        return 0.0, [0.0]*params.d
    width = len(params.q)
    assert width > 0 and params.heads > 0 and width % params.heads == 0
    head_width = width//params.heads
    q, k, values = [], [], []
    for _, z, source in positive:
        x = [a+b for a, b in zip(normalize(z, params.epsilon),
                                 params.source_embedding[source])]
        q.append(matvec(params.q, x))
        k.append(matvec(params.k, x))
        values.append(matvec(params.value, x))
    processed = []
    for i, (_, z, source) in enumerate(positive):
        contexts = []
        for head in range(params.heads):
            start, stop = head*head_width, (head+1)*head_width
            logits = [
                math.fsum(a*b for a, b in zip(q[i][start:stop], key[start:stop]))
                / math.sqrt(head_width) + math.log(w)
                for key, (w, _, _) in zip(k, positive)
            ]
            maximum = max(logits)
            weights = [math.exp(x-maximum) for x in logits]
            denominator = math.fsum(weights)
            contexts.extend(
                math.fsum(weight*value[col] for weight, value in zip(weights, values))
                / denominator for col in range(start, stop)
            )
        delta = matvec(params.out, contexts)
        y = [a+params.scale*b for a, b in zip(z, delta)]
        processed.append(params.residual(y, source))
    return total, [
        math.fsum((w/total)*value[col]
                  for (w, _, _), value in zip(positive, processed))
        for col in range(params.d)
    ]


def fixture(seed=17):
    rng = random.Random(seed)
    def matrix(rows, columns):
        return [[rng.uniform(-0.7, 0.7) for _ in range(columns)]
                for _ in range(rows)]
    return Parameters(4, 2, 0.1, matrix(4, 4), matrix(4, 4), matrix(4, 4),
                      matrix(4, 4), {a: matrix(1, 4)[0] for a in (0, 1)},
                      matrix(3, 4), matrix(4, 3),
                      {a: matrix(1, 3)[0] for a in (0, 1)})


MESSAGES = [(0.25, [1, 0, -2, 0.5], 0),
            (0.75, [0.5, -1, 0, 2], 1),
            (1.5, [-0.5, 0.3, 1, -1], 0)]


class IncomingAttentionChecks(unittest.TestCase):
    def assert_vector(self, actual, expected, tolerance=1e-11):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertTrue(math.isclose(a, b, abs_tol=tolerance, rel_tol=tolerance),
                            (actual, expected))

    def test_textbook_closed_form(self):
        root = math.sqrt(2)
        p = Parameters(1, 1, 1, [[root]], [[root]], [[root]], [[1]],
                       {0: [0]}, [[1]], [[1]], {0: [0]})
        messages = [(1, [1], 0), (1, [-1], 0)]
        for scale in (0.0, 1.0):
            a = 1+scale*math.tanh(1)
            c = a/math.sqrt(1+a*a)
            total, result = aggregate(messages, replace(p, scale=scale))
            self.assertEqual(total, 2)
            self.assert_vector(result, [c/2*math.tanh(c/2)])
        self.assert_vector(aggregate(messages, replace(p, alpha=0))[1], [0])

    def test_two_heads_unequal_strengths(self):
        eye = [[1, 0], [0, 1]]
        p = Parameters(2, 2, 1, eye, eye, eye, eye,
                       {0: [0, 0]}, [[0, 0]], [[0], [0]], {0: [0]}, alpha=0)
        root, e = math.sqrt(2), math.e
        messages = [(1, [root, 0], 0), (3, [0, root], 0)]
        # Head 1 sees scalar features (1, 0); head 2 sees (0, 1).
        expected = [root/4 + e/(4*(e+3)) + 3/16,
                    3*root/4 + 3/16 + 9*e/(4*(1+3*e))]
        self.assert_vector(aggregate(messages, p)[1], expected)

    def test_permutation_scale_and_zero_atoms(self):
        for seed in (17, 42, 71):
            p = fixture(seed)
            total, expected = aggregate(MESSAGES, p)
            for order in permutations(MESSAGES):
                self.assert_vector(aggregate(order, p)[1], expected)
            for scale in (1e-12, 0.2, 3, 1e6):
                r, u = aggregate([(scale*w, z, a) for w, z, a in MESSAGES], p)
                self.assertTrue(math.isclose(r, scale*total))
                self.assert_vector(u, expected)
            zeros = [(0, [1e3, -1e3, 0, 1], 1), (-2, [3, 4, 5, 6], 0)]
            self.assertEqual(aggregate(zeros, p), (0.0, [0.0]*4))
            self.assertEqual(aggregate([], p), (0.0, [0.0]*4))
            self.assert_vector(aggregate(zeros+MESSAGES, p)[1], expected)
            # Splitting one weighted atom into two identical, distinct atoms
            # preserves its mass; no source-class deduplication is performed.
            split = [(0.1, MESSAGES[0][1], 0), (0.15, MESSAGES[0][1], 0)]
            self.assert_vector(aggregate(split+MESSAGES[1:], p)[1], expected)

    def test_disabled_branch_and_single_message(self):
        p = fixture()
        total = sum(w for w, _, _ in MESSAGES)
        residuals = [(w, p.residual(z, a)) for w, z, a in MESSAGES]
        expected = [sum(w*z[col] for w, z in residuals)/total for col in range(p.d)]
        self.assert_vector(aggregate(MESSAGES, replace(p, scale=0))[1], expected)
        w, z, source = MESSAGES[0]
        x = [a+b for a, b in zip(normalize(z, p.epsilon), p.source_embedding[source])]
        delta = matvec(p.out, matvec(p.value, x))
        expected = p.residual([a+p.scale*b for a, b in zip(z, delta)], source)
        self.assert_vector(aggregate([(w, z, source)], p)[1], expected)

    def test_weak_nonconvergent_content(self):
        p = fixture()
        expected = aggregate(MESSAGES, p)[1]
        for exponent in range(2, 10):
            weight = 10.0**(-exponent)
            content = [(-1)**exponent*x for x in (2, -3, 1, 4)]
            actual = aggregate(MESSAGES+[(weight, content, 1)], p)[1]
            error = max(abs(a-b) for a, b in zip(actual, expected))
            self.assertLess(error, 20*weight)
        # A vanishing total does not force u to zero: this is the old contract.
        r, u = aggregate([(1e-12*w, z, a) for w, z, a in MESSAGES], p)
        self.assertLess(r, 1e-10)
        self.assert_vector(u, expected)

    def test_large_logits_and_context_dependence(self):
        p = fixture()
        large = replace(p, q=[[1e4*x for x in row] for row in p.q],
                        k=[[1e4*x for x in row] for row in p.k])
        self.assertTrue(all(math.isfinite(x) for x in aggregate(MESSAGES, large)[1]))
        # Mixed differences vanish for any independent per-message transform
        # followed by fixed-weight summation. They need not vanish for this Agg.
        a, b, c = [list(z) for _, z, _ in MESSAGES]
        d = [0.4, 0.9, -0.2, -1.5]
        for scale in (0, 1):
            pp = replace(p, scale=scale)
            outputs = [aggregate([(1, x, 0), (1, y, 1)], pp)[1]
                       for x, y in ((a, c), (b, c), (a, d), (b, d))]
            mixed = [outputs[0][i]-outputs[1][i]-outputs[2][i]+outputs[3][i]
                     for i in range(p.d)]
            if scale == 0:
                self.assert_vector(mixed, [0]*p.d)
            else:
                self.assertGreater(max(abs(x) for x in mixed), 1e-4)

    def test_parameter_increments(self):
        profile, n, d, width, rank = 'fc-4', 8, 128, 64, 8
        nodes, groups = 32, 4
        independent = capacity.incoming_attention_increment(profile, n, 0, d,
                                                           attention_width=width)
        self.assertEqual(independent['total'], nodes*(4*d*width+d))
        regional = capacity.incoming_attention_increment(profile, n, 0, d,
                                                        'region_shared', rank, width)
        layer = capacity.incoming_attention_increment(profile, n, 0, d,
                                                     'layer_shared', rank, width)
        self.assertEqual(regional, layer)
        self.assertEqual(regional['base'], groups*(4*d*width+d))
        self.assertEqual(regional['D'], nodes*4*rank*(d+width))
        self.assertEqual(regional['H'], 0)
        narrow = capacity.incoming_attention_increment(profile, n, 0, d,
                                                      'region_shared', rank, 2)
        self.assertEqual(narrow['D'], nodes*4*2*(d+2))
        for sharing in ('independent', 'region_shared', 'layer_shared'):
            delta = capacity.incoming_attention_increment('multiscale-compact-9', 8, 4,
                                                         d, sharing, rank, width)
            baseline = capacity.count('multiscale-compact-9', 8, 4, d, 'X', sharing, rank)
            self.assertEqual((delta['nodes_D'], delta['nodes_H']),
                             (baseline['nodes_D'], baseline['nodes_H']))
            self.assertGreater(delta['H'], 0)
            self.assertEqual(delta['total'], delta['base']+delta['D']+delta['H'])
            self.assertEqual(capacity.active(delta, 8, 'pool'), delta['total'])
        for width in (0, -2, 3.5, True):
            with self.assertRaises(ValueError):
                capacity.incoming_attention_increment(profile, n, 0, d, attention_width=width)


if __name__ == '__main__':
    unittest.main(verbosity=2)
