"""CPU-only minimal witness for Shannon cap/face reconstruction failure.

Run with the pilot environment: python -m pilotlab.diagnose_shannon
The installed solver is called without modifications. Its expected failure is
recorded alongside an independent Decimal bisection, not repaired or suppressed
in any experiment. No model, dataset, training, backward, or NPU is used.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import sys
import time


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scalar_reference(scores, lower, upper, target):
    """Independent 80-digit solve of sum clip(exp(s-1-lambda)) = target.

    Input floats are represented exactly, so the comparison isolates the
    solver from the FP32 reduction that originally produced the total target.
    """
    with localcontext() as context:
        context.prec = 80
        scores, lower, upper = ([Decimal.from_float(x) for x in row]
                                for row in (scores, lower, upper))
        target = Decimal.from_float(target)

        def response(multiplier):
            return [max(lo, min(hi, (score - 1 - multiplier).exp()))
                    for score, lo, hi in zip(scores, lower, upper)]

        left, right = Decimal(-100), Decimal(100)
        if not sum(response(left)) >= target >= sum(response(right)):
            raise ArithmeticError('independent reference bracket is invalid')
        for _ in range(300):
            middle = (left + right) / 2
            if sum(response(middle)) > target:
                left = middle
            else:
                right = middle
        multiplier = (left + right) / 2
        result = response(multiplier)
        return {
            'method': 'Decimal 80-digit scalar bisection, 300 iterations',
            'target': str(target), 'allocation': [str(x) for x in result],
            'sum': str(sum(result)), 'residual': str(abs(sum(result) - target)),
            'multiplier': str(multiplier),
            'upper_face': [x == bound for x, bound in zip(result, upper)],
        }


def witness(torch, allocation, dtype):
    scores = torch.tensor([[5., 0.]], dtype=dtype)
    upper = torch.tensor([[.1, 1.]], dtype=dtype)
    lower, w = torch.zeros_like(upper), upper / 2
    trace_values = {}

    def serial(value):
        return value.detach().tolist() if isinstance(value, torch.Tensor) else value

    def trace(frame, event, arg):
        if frame.f_code is allocation._bisect.__code__:
            if event == 'return':
                trace_values.update({
                    key: serial(value) for key, value in frame.f_locals.items()
                    if key in {'c', 'inside', 'reconstructed', 'remainder', 'target',
                               'multiplier', 'shifted'}
                })
            return trace
        return None

    result = {
        'dtype': str(dtype), 'scores': serial(scores), 'w': serial(w),
        'mu': serial(upper), 'lower': serial(lower), 'upper': serial(upper),
        'budget': 1., 'cap_roundtrip': serial(upper.log().exp()),
        'roundtrip_below_cap': serial(upper.log().exp() < upper),
    }
    previous_trace = sys.gettrace()
    sys.settrace(trace)
    try:
        c = allocation.allocate(scores, w, upper, lower, upper, 1., 'shannon')
        result.update(status='completed', allocation=serial(c))
    except ArithmeticError as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        tb = exc.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code is allocation._Constrained.forward.__code__:
                values = tb.tb_frame.f_locals
                result.update({key: serial(values[key])
                               for key in ('c', 'target', 'tolerance', 'low_active', 'high_active')})
                result['residual'] = serial((values['c'].sum(-1, keepdim=True) - values['target']).abs())
                break
            tb = tb.tb_next
    finally:
        sys.settrace(previous_trace)
    result['bisect_before_final_clipping'] = trace_values
    target = result.get('target', trace_values.get('target'))
    if target is not None:
        result['independent_reference'] = scalar_reference(
            scores[0].tolist(), lower[0].tolist(), upper[0].tolist(), target[0][0])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path,
                        help='optional review JSON path directly under docs/evidence')
    args = parser.parse_args(argv)
    import torch
    import tide.allocation as allocation

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    root = Path(__file__).resolve().parents[1]
    start = time.perf_counter()
    attempts = [witness(torch, allocation, torch.float32) for _ in range(3)]
    reference_precision = witness(torch, allocation, torch.float64)
    expected_message = 'allocation multiplier did not meet the total constraint'
    result = {
        'schema': 'pilot-r1-shannon-scalar-diagnostic-v1',
        'status': ('reproduced' if all(x.get('error') == expected_message for x in attempts)
                   else 'not_reproduced'),
        'device': 'cpu', 'cpu_threads': 2, 'torch_version': torch.__version__,
        'script': str(Path(__file__).resolve().relative_to(root)),
        'script_sha256': sha256(__file__),
        'command': ('source /usr/share/Modules/init/bash\nmodule purge\n'
                    'module load ascend/dev-workspace-9.2.0\n'
                    'OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout 30s '
                    '.venv-pilot/bin/python -m pilotlab.diagnose_shannon '
                    '--evidence docs/evidence/pilot-r1-shannon-scalar-diagnostic.json'),
        'allocation_source': str(Path(allocation.__file__).resolve()),
        'allocation_sha256': sha256(allocation.__file__),
        'fp32_attempts': attempts, 'fp32_attempts_identical': all(x == attempts[0] for x in attempts),
        'fp64_witness': reference_precision,
        'limits': [
            'Original solver and total-constraint tolerance are unchanged',
            'No training, backward, NPU operations or production fixes',
            'A passing FP64 witness is not general FP64 qualification or a training policy change',
            'The original NPU failing allocation row and region were not captured',
            'The same-seed CPU model validation passed; this is a mechanism witness, '
            'not proof of the exact NPU failure cause',
        ],
        'seconds': time.perf_counter() - start, 'created_unix': time.time(),
    }
    cpu_evidence = root / 'docs/evidence/pilot-r1-shannon-cpu-diagnostic.json'
    if cpu_evidence.exists():
        result['same_seed_cpu_evidence'] = {
            'path': str(cpu_evidence.relative_to(root)), 'sha256': sha256(cpu_evidence)}
    npu_result = root / '.local/pilot-r1/round/tasks/npu-exchange-x-shannon/result.json'
    if npu_result.exists():
        result['original_npu_result'] = {
            'path': str(npu_result.relative_to(root)), 'sha256': sha256(npu_result)}
    if args.evidence:
        from scripts.review_io import write_review_json
        write_review_json(args.evidence, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
