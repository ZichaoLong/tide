"""Qualify the frozen routing implementation on the r14 A2 capture and continuation."""
import argparse
from pathlib import Path
import sys

from accept_r13_ingress import ROOT, capture_update, checked, read, replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("capture", "direct", "restored"))
    parser.add_argument("--device", required=True, choices=("cpu", "npu:0"))
    parser.add_argument("--foundation-src", type=Path, required=True)
    parser.add_argument("--source-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnosis = read(ROOT / "docs/evidence/pilot-r3-r14-timed-route-vjp.json")
    plan_ref = diagnosis["bindings"]["launch_plan"]
    recorded_plan = read(ROOT / plan_ref["path"])
    sys.path.insert(0, recorded_plan["config"]["source"])
    frozen = read(checked(plan_ref))["config"]
    worker_source = Path(frozen["source"])
    sys.path[:0] = [str(worker_source), str(args.foundation_src.resolve())]
    bound = diagnosis["bindings"]
    result = read(checked(bound["result"]))
    if checked(bound["checkpoint"]) != checked(result["checkpoint"]) or checked(bound["capture"]) != checked(result["failure_capture"]):
        raise ValueError("official artifact binding changed")
    evidence = dict(task=bound["task"], worker_result=bound["result"], formal_checkpoint=result["checkpoint"],
                    original_failure_capture=result["failure_capture"], stage={"failed_update": diagnosis["attempted_step"]})

    def compare(model, opt, device, task, cap):
        forward = diagnosis["observations"][device.type]["native_primals"]
        controls = {"baseline_logits_sha256": forward["logits_sha256"],
                    "baseline_nextstate_sha256": forward["next_state_sha256"]}
        return capture_update(model, opt, device, task, cap, controls, args.output)

    replay(args, evidence, worker_source, Path(frozen["data"]), compare, "-timed-route-qualification")


if __name__ == "__main__":
    main()
