"""Replay matched training and selection windows on a frozen Stage A failure prefix."""
import argparse
from collections import Counter
from contextlib import ExitStack, nullcontext
import copy
import json
import math
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from pilotlab import worker3 as w
from pilotlab.data3 import DocumentStore3
from pilotlab.io import atomic_json, sha256
from pilotlab.models3 import build_model
from pilotlab.resume_adapter3 import checked
from pilotlab.source_migration3 import read_bound
from tide.precision import precision_context
from tide.shannon_support import observe_shannon_support


def parameter_gradients(model):
    return {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters() if p.grad is not None}


class Observations:
    def __init__(self, model, detailed, stack):
        self.detailed = detailed
        self.model = model
        self.rows = Counter()
        self.logits = self.grads = None
        self.overhead = 0.
        self.sender = model.core._sender_output
        self.clip = torch.nn.utils.clip_grad_norm_
        stack.callback(model.register_forward_hook(self.capture_output).remove)
        if detailed:
            stack.enter_context(patch.object(model.core, "_sender_output", self.count_strengths))
            stack.enter_context(patch.object(torch.nn.utils, "clip_grad_norm_", self.capture_gradients))
            stack.callback(model.register_forward_pre_hook(lambda *_: self.rows.clear()).remove)

    def capture_output(self, model, inputs, result):
        self.logits = result.logits.detach()

    def count_strengths(self, region, values, strengths, face, incoming, silent):
        start = time.monotonic()
        with torch.no_grad():
            codes = strengths.detach()
            weak = (codes != 0) & (codes < 2. ** -60)
            if bool(weak.any()):
                sample = codes[weak].cpu().double()
                log_mass = torch.where(sample < 0, sample, sample.abs().log())
                self.rows["positive_below_2^-60"] += sample.numel()
                self.rows["below_fp32_minsub"] += int((log_mass < -149 * math.log(2) - 1e-5).sum())
            self.rows["positive"] += int((codes != 0).sum())
            self.rows["at_stored_floor"] += int((codes == 2. ** -149).sum())
        self.overhead += time.monotonic() - start
        return self.sender(region, values, strengths, face, incoming, silent)

    def capture_gradients(self, parameters, *args, **kwargs):
        start = time.monotonic()
        self.grads = parameter_gradients(self.model)
        self.overhead += time.monotonic() - start
        return self.clip(parameters, *args, **kwargs)


def run(args, stack):
    plan = json.loads(args.plan.read_bytes())
    old = read_bound(plan["original_task"])
    record = read_bound(plan["old_result"])
    capture = torch.load(checked(record["failure_capture"], "failure prefix"), weights_only=False, map_location="cpu")
    if capture["task"] != old or capture["resumable"] is not False:
        raise ValueError("unexpected failure prefix")
    task = copy.deepcopy(old)
    task["source_identity"] = read_bound(plan["source_identity"])
    task["model"]["config"]["shannon_support_policy"] = args.policy
    if (args.policy not in plan["policies"] or plan["device"] != args.device.split(":")[0]
            or any(task[k] != plan[k] for k in ("batch_size", "sequence_length"))
            or task["cpu_threads"] != plan["threads"]):
        raise ValueError("workload does not match the frozen plan")
    w.validate_source_identity(task, Path(plan["data"]))
    device = w.initialize(args.device, task)
    store = DocumentStore3(plan["data"])
    model = build_model(task["model"], store.vocab_size, task["init_recipe"]).to(device)
    optimizer = w.optimizer_for(model, task)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    observed = Observations(model, plan.get("detailed", True), stack)
    rows = []
    if device.type == "npu":
        torch.npu.reset_peak_memory_stats(device)

    def reset_owners():
        model.load_state_dict(capture["model"])
        optimizer.load_state_dict(copy.deepcopy(capture["optimizer"]))
        w.initialize(args.device, task)

    def batch(case, index, x, y, state, tokens):
        train = case != "validation"
        observed.logits = observed.grads = None
        observed.overhead = 0.
        w.synchronize(device)
        start = time.monotonic()
        with (observe_shannon_support() if observed.detailed else nullcontext()) as counts:
            if train:
                update = w.train_update(model, optimizer, x, y, state, device, task, tokens, sample=False)
                scalars = {k: update[k] for k in ("loss", "objective", "grad_norm", "backward_divisor")}
                state = update["state"]
            else:
                model.eval()
                with torch.no_grad(), precision_context(device, task["precision"]):
                    pred = model(x.to(device))
                    loss = torch.nn.functional.cross_entropy(pred.logits.float().flatten(0, 1), y.to(device).flatten())
                    scalars = {"loss": float(loss)}
        w.synchronize(device)
        elapsed = time.monotonic() - start
        logits = observed.logits.float().cpu()
        per_token = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none")
        if not (logits.isfinite().all() and per_token.isfinite().all()):
            raise ArithmeticError(f"nonfinite output in {case} batch {index}")
        path = out / f"{case}-{index}.pt"
        tensors = {"nll": per_token, "argmax": logits.argmax(-1), "logits": logits}
        if train:
            tensors["grads"] = observed.grads if observed.detailed else parameter_gradients(model)
        w.atomic_torch_save(path, tensors)
        row = dict(case=case, index=index, tokens=y.numel(), input_sha256=w.digest_tree((x, y)),
                   **scalars, timed_seconds=elapsed, outputs=str(path), memory=w.device_memory(device))
        if observed.detailed:
            row.update(strengths=dict(observed.rows), shannon_forward_calls=vars(counts),
                       inside_observer_seconds=observed.overhead)
        rows.append(row)
        atomic_json(out / "rows.json", rows)
        return state

    reset_owners()
    completed = capture["completed_tokens"]
    state = w.tree_map(capture["private_state"], lambda t: t.to(device))
    batch("failure", 0, capture["inputs"], capture["targets"], state, completed)
    reset_owners()
    stream = w.StreamBatch(store, {**task, "seed": plan["ordinary_train_seed"]})
    state = None
    for i in range(plan["train_batches"]):
        x, y, reset = stream.next()
        state = batch("train", i, x, y, None if reset else state, completed + i * y.numel())
    del state
    reset_owners()
    rng = np.random.default_rng(plan["validation_seed"])
    for i in range(plan["validation_batches"]):
        sample = store.sample_batch("selection", task["batch_size"], task["sequence_length"], rng)
        batch("validation", i, torch.from_numpy(sample[:, :-1].copy()), torch.from_numpy(sample[:, 1:].copy()), None, 0)
    atomic_json(out / "result.json", dict(
        status="completed", policy=args.policy, plan_sha256=sha256(args.plan), device=str(device),
        physical_card=os.getenv("ASCEND_RT_VISIBLE_DEVICES"), cpu_affinity=sorted(os.sched_getaffinity(0)),
        torch=torch.__version__, threads=torch.get_num_threads(), memory=w.device_memory(device), rows=rows,
        gradient_point="before_clip" if observed.detailed else "after_clip"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("policy", choices=["weak-log-v1", "positive-floor-v1"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu")
    with ExitStack() as stack:
        run(parser.parse_args(), stack)
