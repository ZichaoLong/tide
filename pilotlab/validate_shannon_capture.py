"""Replay captured Shannon inputs and check an independent scalar solution/VJP.

Runs one allocation forward/backward, never a language-model training job.
The caller owns any NPU allocation. Reference calculations remain on the CPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(scores, w, lower, upper, budget):
    """Python float64 scalar bisection; no production tensor solver calls."""
    high = min(budget, math.fsum(upper))
    low = max(math.fsum(lower), min(1., math.fsum(w), high))

    def response(multiplier):
        return [max(lo, min(hi, math.exp(score - 1 - multiplier)))
                for score, lo, hi in zip(scores, lower, upper)]

    initial = response(0.)
    total = math.fsum(initial)
    target = max(low, min(high, total))
    active = 'low' if total < low else 'high' if total > high else 'slack'
    multiplier = 0.
    if active != 'slack':
        left, right = -128., 128.
        if not math.fsum(response(left)) >= target >= math.fsum(response(right)):
            raise ArithmeticError('independent scalar bracket does not cover this capture')
        for _ in range(120):
            middle = (left + right) / 2
            if math.fsum(response(middle)) > target:
                left = middle
            else:
                right = middle
        multiplier = (left + right) / 2
    return dict(values=response(multiplier), multiplier=multiplier,
                target=target, low=low, high=high, active=active)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--repo-root', type=Path, default=ROOT)
    parser.add_argument('--foundation-src', type=Path,
                        help='defaults to the r2 private source under --repo-root')
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    source = (args.foundation_src or repo_root / '.local/pilot-r2/foundation/src').resolve()
    for name in ('validation.json', 'candidate.pt'):
        if (args.output / name).exists():
            parser.error(f'refusing to replace an existing {name}')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    import torch
    import tide.allocation as allocation
    if Path(allocation.__file__).resolve() != source / 'tide/allocation.py':
        raise RuntimeError('candidate foundation source was not selected')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if args.device.startswith('npu'):
        import torch_npu  # noqa: F401
        torch.npu.set_device(args.device)
    device = torch.device(args.device)
    captured = torch.load(args.capture, map_location='cpu', weights_only=True)
    inputs = captured['allocation_inputs']
    if inputs['kind'] != 'shannon':
        raise ValueError('capture is not a Shannon allocation')
    start = time.time()
    names = ('scores', 'w', 'mu', 'lower', 'upper', 'budget')
    native = {key: inputs[key].to(device).detach().requires_grad_() for key in names}
    answer = allocation.allocate(*(native[key] for key in names), kind='shannon')
    width = answer.shape[-1]
    coefficients = torch.tensor([(-1.)**i * (i + 1) for i in range(width)], dtype=answer.dtype)
    gradients = torch.autograd.grad((answer * coefficients.to(device)).sum(), tuple(native.values()))
    actual = answer.detach().cpu()
    grad_cpu = {key: value.detach().cpu() for key, value in zip(names, gradients)}
    flat = {key: value.reshape(-1, width).tolist() for key, value in inputs.items()
            if isinstance(value, torch.Tensor) and key != 'budget'}
    budgets = inputs['budget'].reshape(-1).tolist()
    refs = [reference(flat['scores'][i], flat['w'][i], flat['lower'][i], flat['upper'][i], budgets[i])
            for i in range(len(budgets))]
    expected = torch.tensor([row['values'] for row in refs], dtype=torch.float64).reshape(actual.shape)
    targets = torch.tensor([row['target'] for row in refs], dtype=torch.float64).reshape(actual.shape[:-1])
    eps = torch.finfo(actual.dtype).eps
    tolerance = 32 * eps * targets.abs().clamp_min(torch.finfo(actual.dtype).tiny)
    residual = (actual.double().sum(-1) - targets).abs()
    reference_error = (actual.double() - expected).abs()
    bounds_ok = bool(((actual >= inputs['lower']) & (actual <= inputs['upper'])).all())
    support_equal = bool(torch.equal(actual > 0, expected > 0))
    max_kkt_ratio = 0.
    for row, ref in enumerate(refs):
        for score, value, lo, hi in zip(flat['scores'][row], actual.reshape(-1, width)[row].tolist(),
                                        flat['lower'][row], flat['upper'][row]):
            if lo == hi:
                continue
            if value <= 0:
                max_kkt_ratio = float('inf')
                continue
            marginal = score - 1 - math.log(value)
            multiplier = ref['multiplier']
            error = (max(0., marginal - multiplier) if value == lo else
                     max(0., multiplier - marginal) if value == hi else abs(marginal - multiplier))
            allowed = 128 * eps * max(1., abs(score), abs(multiplier), abs(marginal))
            max_kkt_ratio = max(max_kkt_ratio, error / allowed)

    original = captured['solver']
    failed = original['active'] & ((original['c'].sum(-1, keepdim=True) - original['target']).abs()
                                   > original['tolerance'])
    indices = failed.squeeze(-1).nonzero().tolist()
    if not indices:
        raise ValueError('capture contains no original failing row')
    vjp = []
    for index_list in indices:
        index = tuple(index_list)
        flat_index = 0
        for dimension, item in zip(actual.shape[:-1], index):
            flat_index = flat_index * dimension + item
        ref = refs[flat_index]
        lo, hi = inputs['lower'][index].tolist(), inputs['upper'][index].tolist()
        c, g = ref['values'], coefficients.tolist()
        # The captured r1 failure is a high-budget active D row with positive
        # interiors and upper faces. Check that scope before using its closed
        # form derivative of (target - capped mass) * softmax(interior scores).
        inside = [lower < value < upper for lower, value, upper in zip(lo, c, hi)]
        upper_face = [value == upper for value, upper in zip(c, hi)]
        if (ref['active'] != 'high' or any(lower == upper for lower, upper in zip(lo, hi))
                or any(value == lower for value, lower in zip(c, lo))
                or budgets[flat_index] >= math.fsum(hi)):
            raise ValueError('captured row is outside the independently derived VJP case')
        average = math.fsum(value * cot for value, cot, free in zip(c, g, inside) if free) / math.fsum(
            value for value, free in zip(c, inside) if free)
        expected_grads = {
            'scores': [value * (cot - average) if free else 0. for value, cot, free in zip(c, g, inside)],
            'w': [0.] * width, 'mu': [0.] * width, 'lower': [0.] * width,
            'upper': [cot - average if face else 0. for cot, face in zip(g, upper_face)],
            'budget': [average],
        }
        comparisons = {}
        for key in names:
            expected_grad = torch.tensor(expected_grads[key], dtype=torch.float64)
            found = grad_cpu[key][index].double().reshape(-1)
            comparisons[key] = {'max_abs': float((found - expected_grad).abs().max()),
                                'within_tolerance': bool(torch.allclose(found, expected_grad, atol=1e-5, rtol=1e-4)),
                                'actual': found.tolist(), 'expected': expected_grad.tolist()}
        vjp.append({'index': index_list, 'reference': ref, 'comparisons': comparisons,
                    'candidate': actual[index].tolist(), 'upper_face': upper_face})
    checks = {
        'finite_forward': bool(torch.isfinite(actual).all()),
        'finite_gradients': all(bool(torch.isfinite(value).all()) for value in grad_cpu.values()),
        'bounds': bounds_ok, 'positive_support_matches_reference': support_equal,
        'total_within_original_tolerance': bool((residual <= tolerance).all()),
        'independent_solution': bool((reference_error <= tolerance[..., None] + 32 * eps * expected.abs()).all()),
        'independent_kkt': max_kkt_ratio <= 1.,
        'independent_vjp': all(value['within_tolerance'] for row in vjp for value in row['comparisons'].values()),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({'allocation': actual, 'gradients': grad_cpu}, args.output / 'candidate.pt')
    record = {
        'schema': 'pilot-r2-shannon-captured-input-validation-v1',
        'status': 'passed' if all(checks.values()) else 'failed',
        'device': str(device), 'cpu_threads': 2, 'torch_version': torch.__version__,
        'physical_visibility': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        'script': str(Path(__file__).resolve()), 'script_sha256': sha256(__file__),
        'capture': str(args.capture.resolve()), 'capture_sha256': sha256(args.capture),
        'allocation_source': str(Path(allocation.__file__).resolve()),
        'allocation_sha256': sha256(allocation.__file__),
        'shape': list(actual.shape), 'checks': checks,
        'reference_method': 'Independent Python float64 scalar bisection, 120 iterations per row',
        'vjp_method': 'Derivative of remaining-mass times interior softmax, conditioned on independently verified active face',
        'original_total_tolerance': '32 * finfo(input_dtype).eps * abs(target)',
        'kkt_tolerance': '128 * eps * max(1, abs(score), abs(multiplier), abs(marginal))',
        'vjp_atol': 1e-5, 'vjp_rtol': 1e-4,
        'max_total_residual': float(residual.max()),
        'max_total_tolerance_ratio': float((residual / tolerance).max()),
        'max_solution_error': float(reference_error.max()), 'max_kkt_ratio': max_kkt_ratio,
        'captured_failed_rows': vjp,
        'candidate_payload_sha256': sha256(args.output / 'candidate.pt'),
        'seconds': time.time() - start,
        'limitations': ['One captured regional input tensor; no model-wide or long-training qualification',
                        'No precision change and no production tolerance change'],
    }
    target = args.output / 'validation.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)
    print(json.dumps({key: record[key] for key in ('status', 'device', 'shape', 'checks',
          'max_total_residual', 'max_solution_error', 'max_kkt_ratio', 'seconds')}))
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
