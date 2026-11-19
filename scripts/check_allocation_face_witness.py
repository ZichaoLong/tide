"""Same-shape FP32 replay and independent score VJP of the retained quad face.

Only the recorded ordinary-node, budget-one witness is in scope. CPU FP64 and
Decimal are bounded diagnostic roots; production on CPU/NPU remains FP32.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from benchmarks.inference import source_identity
from scripts.diagnose_bf16 import decimal_quad
from scripts.diagnose_routing_divergence import difference
from scripts.job import atomic
from scripts.qualify_precision import THRESHOLDS
from tide.precision import precision_context, precision_policy
from tide.region import region_control


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args(argv)
    if args.output.exists() or args.witness.stat().st_size > 8 << 20:
        parser.error("new output and bounded <=8MiB input required")
    torch.set_num_threads(args.threads)
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401; external queue owns admission.
        torch.npu.set_device(args.device)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    raw = args.witness.read_bytes()
    cases = torch.load(args.witness, map_location="cpu", weights_only=False)
    if not 1 <= len(cases) <= 4:
        raise ValueError("one to four retained regional witnesses")
    record = dict(schema="tide-quad-face-witness-v1", state="running", source=source_identity(),
        input=dict(path=str(args.witness), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)),
        device=args.device, physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        torch=str(torch.__version__), cpu_affinity=sorted(os.sched_getaffinity(0)),
        precision=precision_policy(args.device, "fp32"), tolerance=THRESHOLDS["fp32"], cases=[],
        scope="same complete saved submission shape and selected-row score VJP; no timing or whole-model equivalence claim")
    atomic(args.output, record)
    try:
        for case in cases:
            if any(case["roles"]) or case["budget"] != 1 or case["profile"] != "quad":
                raise ValueError("only the retained ordinary-node budget-one quad witness")
            coordinate = case["coordinate"]
            sample, position, node = (coordinate[k] for k in ("sample", "time_index", "local_node"))
            for index, payload in enumerate(case["paths"]):
                scores = payload["score"].detach().to(args.device, copy=True).requires_grad_()
                received = payload["received"].to(args.device)
                r, n, b, length = scores.shape
                selected = payload["selected_index"]
                def fold(value):
                    return value.permute(1, 0, 2, 3).reshape(n, r*b, length)
                with precision_context(args.device, "fp32"):
                    control = region_control(fold(scores), fold(received),
                        torch.tensor(case["roles"], dtype=torch.bool, device=args.device),
                        1, "quad", case["partition"]).reshape(n, r, b, length).permute(1, 0, 2, 3)
                    loss = control[selected, node, sample, position]
                gradient = torch.autograd.grad(loss, scores)[0].detach().cpu()
                current = control.detach().cpu()[selected, :, sample, position]
                source_score = payload["score"][selected, :, sample, position]
                source_received = payload["received"][selected, :, sample, position]
                oracle = decimal_quad(source_score.tolist(), source_received.tolist(), case["roles"], 1, case["partition"])
                expected = torch.tensor(oracle["control"], dtype=torch.float64)
                cap = torch.minimum(torch.ones_like(source_received.double()), 2*source_received.double().clamp_min(0))
                inside = (expected > 0) & (expected < cap)
                # This recorded row has a strict active fixed total. The VJP
                # is the projection onto the zero-sum tangent of its interior.
                if oracle["low_total"] != 1 or oracle["high_total"] != 1 or not inside.any():
                    raise ValueError("independent fixed-total interior formula is not applicable")
                if float(source_score.double().clamp_min(0).minimum(cap).sum()) == 1:
                    raise ValueError("slack-on-equality derivative needs a separate formula")
                root = torch.zeros(n, dtype=torch.float64); root[node] = 1
                wanted_gradient = inside*(root-root[inside].mean())
                got_gradient = gradient[selected, :, sample, position]
                value_check = difference(expected, current.double(), THRESHOLDS["fp32"])
                gradient_check = difference(wanted_gradient, got_gradient.double(), THRESHOLDS["fp32"])
                row = dict(path_index=index, coordinate=coordinate, score=source_score.tolist(),
                    received=source_received.tolist(), submission_shape=list(fold(scores).shape),
                    submission_stride=list(fold(scores).stride()), control=current.tolist(), oracle=oracle,
                    score_vjp=got_gradient.tolist(), expected_score_vjp=wanted_gradient.tolist(),
                    value_comparison=value_check, score_vjp_comparison=gradient_check,
                    support_matches=torch.equal(current > 0, expected > 0),
                    unchanged_other_rows=bool((gradient.clone().index_put(
                        (torch.tensor([selected]), torch.arange(n), torch.tensor([sample]), torch.tensor([position])),
                        torch.zeros(n)) == 0).all()))
                row["qualified"] = row["support_matches"] and all(check.get("finite", False) and check["threshold_failures"] == 0
                    for check in (value_check, gradient_check)) and row["unchanged_other_rows"]
                record["cases"].append(row)
        record["state"] = "passed" if all(row["qualified"] for row in record["cases"]) else "numerical_failed"
        atomic(args.output, record)
        print(json.dumps(dict(state=record["state"], cases=len(record["cases"]))))
        return 0 if record["state"] == "passed" else 2
    except Exception as error:
        record.update(state="execution_failed", error=repr(error))
        atomic(args.output, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
