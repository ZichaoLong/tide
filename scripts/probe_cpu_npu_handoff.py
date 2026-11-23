"""Replay one real CPU checkpoint update with independent CPU and Ascend owners."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import torch

from pilotlab import worker3 as worker
from pilotlab.data3 import DocumentStore3
from pilotlab.io import atomic_json, sha256
from pilotlab.models3 import build_model
from pilotlab.worker import compare_tree


def replay(spec, data, output, backend):
    task_ref, snapshot = spec["task"], spec["snapshot"]
    checkpoint = spec["checkpoint"]
    if sha256(task_ref["path"]) != task_ref["sha256"] or snapshot["sha256"] != checkpoint["sha256"]:
        raise ValueError("probe input does not bind the formal CPU task and checkpoint")
    task = json.loads(Path(task_ref["path"]).read_text())
    worker.validate_source_identity(task, data)
    payload, _ = worker.load_checkpoint(snapshot["path"], task, "cpu", checkpoint["sha256"])
    if (task["device"] != "cpu" or payload["task_id"] != task["id"]
            or any(payload[k] != checkpoint[k] for k in ("steps", "tokens", "state_sha256"))
            or payload["tokens"] >= task["target_tokens"] or payload["rng"]["npu"] is not None):
        raise ValueError("expected an in-progress CPU checkpoint from this task")
    device = worker.initialize(backend, task)
    store = DocumentStore3(data)
    model = build_model(task["model"], store.vocab_size, init_recipe=task["init_recipe"]).to(device)
    optimizer = worker.optimizer_for(model, task)
    if device.type == "cpu":
        state, stream = worker.restore_owners(payload, model, optimizer, store, task, device)
    else:
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        state = worker.tree_map(payload["private_state"], lambda tensor: tensor.to(device))
        stream = worker.StreamBatch(store, task, payload["stream"])
        worker.restore_rng(payload["rng"], torch.device("cpu"))
        torch.npu.manual_seed(task["seed"])
    before_rng = worker.rng_state(device)
    fields = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), private_state=state,
                  stream=stream.state_dict(), rng={**before_rng, "npu": None}, steps=payload["steps"],
                  tokens=payload["tokens"])
    exact = {key: worker.digest_tree(value) == worker.digest_tree(payload[key]) for key, value in fields.items()}
    if not all(exact.values()):
        raise ValueError(f"portable state changed at load: {exact}")
    x, y, reset = stream.next()
    started = time.monotonic()
    updated = worker.train_update(model, optimizer, x, y, None if reset else state, device, task, payload["tokens"])
    worker.synchronize(device)
    after_rng = worker.rng_state(device)
    gradients = {name: p.grad for name, p in model.named_parameters()}
    trees = worker.cpu_snapshot(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                     private_state=updated["state"], gradients=gradients))
    cpu_path = output / "cpu-reference.pt"
    metrics = {key: updated[key] for key in ("loss", "objective", "grad_norm", "lr")}
    report = dict(task=task_ref, checkpoint=checkpoint, backend=backend,
                  relay_step=payload["steps"], relay_tokens=payload["tokens"], next_step=payload["steps"]+1,
                  next_tokens=payload["tokens"]+y.numel(), portable_fields_exact=exact, state_reset=reset,
                  input_sha256=worker.digest_tree((x, y)), next_stream_sha256=worker.digest_tree(stream.state_dict()),
                  host_rng_after_sha256=worker.digest_tree({**after_rng, "npu":None}),
                  device_rng_changed=worker.digest_tree(before_rng["npu"]) != worker.digest_tree(after_rng["npu"]),
                  destination_rng_seed=task["seed"] if device.type == "npu" else None,
                  physical_visibility=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), metrics=metrics,
                  elapsed_seconds=time.monotonic()-started, atol=spec["atol"], rtol=spec["rtol"])
    if device.type == "cpu":
        with (Path(spec["attempt"])/"steps.jsonl").open() as journal:
            formal = next(row for line in journal if (row := json.loads(line))["step"] == report["next_step"])
        report["formal_step"] = formal
        report["passed"] = (formal["tokens"] == report["next_tokens"] and formal["state_reset"] == reset
                            and all(formal[k] == v for k,v in metrics.items()))
        worker.atomic_torch_save(cpu_path, trees)
        report["reference_sha256"] = sha256(cpu_path)
    else:
        cpu = json.loads((output/"cpu.json").read_text())
        if not cpu["passed"] or cpu["reference_sha256"] != sha256(cpu_path) or cpu["checkpoint"] != checkpoint:
            raise ValueError("CPU reference does not reproduce the recorded update")
        expected = torch.load(cpu_path, weights_only=False, map_location="cpu")
        report["comparison"] = {k: compare_tree(v, trees[k], atol=spec["atol"], rtol=spec["rtol"])
                                for k,v in expected.items()}
        report["metrics_close"] = {k: math.isclose(v, cpu["metrics"][k], abs_tol=spec["atol"], rel_tol=spec["rtol"])
                                   for k,v in metrics.items()}
        report["same_data_and_host_rng"] = all(report[k] == cpu[k] for k in
            ("next_step", "next_tokens", "state_reset", "input_sha256", "next_stream_sha256", "host_rng_after_sha256"))
        report["passed"] = (report["same_data_and_host_rng"] and all(report["metrics_close"].values())
                            and all(v["within_tolerance"] for v in report["comparison"].values()))
    atomic_json(output/(device.type+".json"), report)
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "npu:0"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raise SystemExit(0 if replay(json.loads(args.input.read_text()), args.data, args.output, args.device) else 1)
