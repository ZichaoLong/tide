#!/usr/bin/env python3
"""Independently solve a captured actual allocation row in Decimal arithmetic.

No Torch, reference execution, tensor solver, or input regeneration. JSON numbers
are decoded as the exact captured binary floats; output decimals are strings.
Allocation §4.1/6: broad-bracket monotone solve, not the candidate's inverse
bracket/breakpoint implementation. This certifies only the captured local map.
"""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
from durable_records import write_json


def diagnose(record, precision=120):
    if record.get("schema") != "tide-allocation-failure-v1":
        raise ValueError("unsupported allocation failure schema")
    if not 48 <= precision <= 1024:
        raise ValueError("Decimal precision must be in [48,1024]")
    with localcontext() as ctx:
        ctx.prec = precision
        d = lambda value: Decimal.from_float(float(value))
        def field(name):
            return list(map(d, record["tensors"][name]["values"]))
        scores, caps = field("scores"), field("caps")
        low, high, tau = (field(k)[0] for k in ("lower", "upper", "temperature"))
        shift = field("shift")[0] if "shift" in record["tensors"] else Decimal(0)
        kind = record["details"]["regularizer"]
        alpha = d(record["details"]["alpha"]) if kind == "ent" else None
        if (kind not in {"quad", "bern", "ent"} or len(scores) != len(caps) or not caps
                or any(not x.is_finite() for x in (*scores, *caps, low, high, tau, shift))
                or tau <= 0 or low < 0 or high < low or any(m < 0 or m > 1 for m in caps)):
            raise ValueError("invalid captured allocation inputs")
        if kind == "ent" and not 1 <= alpha <= 2:
            raise ValueError("invalid entropy shape")
        t = [s-shift for s in scores]
        mass = sum(caps)
        base = dict(schema="tide-allocation-decimal-v1", precision=precision,
                    method="exact captured binary inputs; independent expanded bracket and Decimal bisection",
                    row=record["row"], time=record["time"], nodes=record["nodes"],
                    regularizer=kind, exact_mass=str(mass), lower=str(low), upper=str(high))
        if low > mass:
            return dict(base, state="infeasible_exact_inputs", bound_excess=str(low-mass))
        def response(lam):
            values = []
            for s, m in zip(t, caps):
                if not m:
                    values.append(Decimal(0)); continue
                z = (s-lam)/tau
                if kind == "quad":
                    value = min(m, max(Decimal(0), z))
                elif kind == "bern":
                    e = (-abs(z)).exp()
                    value = m/(1+e) if z >= 0 else m*e/(1+e)
                elif alpha == 1:
                    value = m if z-1 >= m.ln() else (z-1).exp()
                else:
                    value = min(m, max(Decimal(0), (alpha-1)*z+1/alpha)**(1/(alpha-1)))
                values.append(value)
            return values
        lam, iterations = Decimal(0), 0
        if not mass or not high:
            c, branch = [Decimal(0)]*len(caps), "zero"
        elif low == mass:
            c, branch = caps[:], "caps"
        else:
            c = response(lam); total = sum(c)
            branch = "equality" if low == high else "lower" if total < low else "upper" if total > high else "slack"
            if branch != "slack":
                target = low if branch in {"lower", "equality"} else high
                step = max(Decimal(1), tau)
                left, right = min(t)-step, max(t)+step
                for _ in range(2048):
                    if sum(response(left)) >= target and sum(response(right)) <= target:
                        break
                    step *= 2
                    left, right = min(t)-step, max(t)+step
                else:
                    raise ArithmeticError("independent bracket expansion exhausted")
                for iterations in range(1, 4097):
                    lam = (left+right)/2
                    if lam in {left, right}:
                        break
                    total = sum(response(lam))
                    if total > target:
                        left = lam
                    else:
                        right = lam
                else:
                    raise ArithmeticError("independent Decimal bisection exhausted")
                c = response(lam)
        actual = field("controls")
        if len(actual) != len(c):
            raise ValueError("captured control shape mismatch")
        finite = all(x.is_finite() for x in actual)
        violation = max(Decimal(0), low-sum(c), sum(c)-high)
        result = dict(base, state="solved", branch=branch, iterations=iterations,
                      multiplier=str(lam), controls=list(map(str, c)), residual=str(violation),
                      candidate_controls=list(map(str, actual)), candidate_finite=finite)
        if finite:
            result["candidate_max_error"] = str(max(abs(a-b) for a, b in zip(actual, c)))
            result["candidate_bound_residual"] = str(max(Decimal(0), low-sum(actual), sum(actual)-high))
        # The actual response vector may reveal cancellation even when its
        # rounded global multiplier alone appears plausible.
        if "response" in record["tensors"]:
            implied = [s-x for s, x in zip(t, field("response"))]
            if all(x.is_finite() for x in implied):
                result["candidate_implied_multiplier_span"] = str(max(implied)-min(implied))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--precision", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("diagnostic output must be new")
    raw = args.failure.read_bytes()
    result = diagnose(json.loads(raw), args.precision)
    result["failure_sha256"] = hashlib.sha256(raw).hexdigest()
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
