"""Check the pool repair on the r16 capture and the original CPU checkpoint trajectory."""
import argparse
from dataclasses import asdict
import math
import os
from pathlib import Path
import sys

from accept_r11_quad import ROOT, checked, read, replay


def capture_update(model, optimizer, device, task, capture):
    import torch
    from pilotlab import worker3 as w
    from pilotlab.worker import tree_map
    from tide import ssd_memory
    from unittest.mock import patch

    model.load_state_dict(capture["model"], strict=True)
    optimizer.load_state_dict(capture["optimizer"])
    state = tree_map(capture["private_state"], lambda t: t.to(device))
    selected = []
    select = ssd_memory.select_queries
    def observe(u, control, times):
        packed = select(u, control, times)
        weak = (control == 2.**-149).nonzero()
        if weak.numel():
            active = torch.stack((packed.nodes[packed.node], packed.batch, packed.time), -1)
            if not bool((weak[:, None] == active[None]).all(-1).any(-1).all()):
                raise ArithmeticError("pool floor disappeared before SSD finish")
            selected.extend(weak.tolist())
        return packed
    with patch.object(ssd_memory, "select_queries", observe):
        update = w.train_update(model, optimizer, capture["inputs"], capture["targets"], state, device, task,
                                capture["completed_tokens"])
    w.synchronize(device)
    counts = asdict(model.core._pool_counts)
    values = w.cpu_snapshot((model.state_dict(), optimizer.state_dict(), update["state"],
                             [p.grad for p in model.parameters()]))
    finite = all(bool(torch.isfinite(t).all()) for _, t in w.leaves(values)
                 if isinstance(t, torch.Tensor) and t.is_floating_point())
    scalars = {k:update[k] for k in ("loss","objective","grad_norm","lr")}
    return dict(passed=finite and [0,1,77] in selected and all(math.isfinite(x) for x in scalars.values()),
                finite_model_adamw_state_gradients=finite, selected_floor_coordinates=selected, counts=counts, **scalars)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("capture","direct","restored"))
    parser.add_argument("--device", required=True, choices=("cpu","npu:0"))
    parser.add_argument("--foundation-src", type=Path, required=True)
    parser.add_argument("--source-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnosis = read(ROOT / "docs/evidence/pilot-r3-r16-pool-underflow.json")
    run = ROOT / ".local/pilot-r3/campaign-r16"
    plan = read(run / "campaign-plan.json")
    config = plan["config"]
    source = Path(config.get("worker_source", config["source"]))
    sys.path[:0] = [str(source), str(args.foundation_src.resolve())]
    from pilotlab.prepare_resume3 import reference
    attempt = (ROOT / diagnosis["capture"]).parent
    original = reference(attempt / "task.json")
    worker = reference(attempt / "result.json")
    result = read(checked(worker))
    if result["failure_capture"]["sha256"] != diagnosis["capture_sha256"]:
        raise ValueError("diagnostic capture and official receipt differ")
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = dict(task=original, worker_result=worker, formal_checkpoint=result["checkpoint"],
                    original_failure_capture=result["failure_capture"], stage={"failed_update": diagnosis["failed_step"]})
    replay(args, evidence, source, Path(config["data"]), capture_update, "-pool-qualification",
           model_config={"pool_support_policy":"positive-floor-v1"})
