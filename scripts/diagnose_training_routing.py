"""One retained BF16 third-update routing witness, with bounded scalar capture.

Replays baseline and combined from the original real-update anchor. A new
diagnosis never upgrades the original qualification. No full checkpoint or
observation tensor is written; only a few allocation input rows are retained.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch

from tide import ModelConfig
from benchmarks.inference import source_identity
from benchmarks.workloads import tensor_hash
from scripts import check_training_paths as paths
from scripts.diagnose_routing_divergence import (
    CaptureBudget, RoutingTrace, allocation_payload, compare_traces, replay_allocation,
)
from scripts.job import atomic
from scripts.qualify_precision import THRESHOLDS, atomic_tensor


def capture_update(model, optimizer, state, value, precision, device, chunk, budget):
    """Trace just the forward; checkpoint backward runs outside the observer."""
    trace = RoutingTrace(model, budget)
    original = paths.forward

    def captured(*args, **kwargs):
        with trace.install():
            return original(*args, **kwargs, trace=trace)

    with patch.object(paths, "forward", captured):
        observation, following, work = paths.observe(
            model, optimizer, state, value, precision, [device], chunk, update=True)
    trace.validate()
    return observation, following, work, trace


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--candidate", choices=("x", "kv_half"), default="x")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload_path = args.output.with_suffix(".witness.pt")
    if args.output.exists() or payload_path.exists():
        parser.error("choose a new output identity")
    old = json.loads(args.qualification.read_text())
    options = old["arguments"]
    if (options["precision"] != "bf16" or options["steps"] != 3
            or options["batch"] > 2 or options["length"] > 8):
        parser.error("only the existing bounded three-update BF16 witness is in scope")
    case = next(row for row in old["rows"] if row["candidate"] == args.candidate)
    config = ModelConfig.from_dict(case["config"])
    if config.d > 192 or config.topology.n > 2:
        parser.error("small witness only; no large model training")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    paths.configure_devices([args.device])
    record = dict(schema="tide-training-routing-diagnosis-v1", state="running",
        source=source_identity(), original_source=old["source"],
        qualification=dict(path=str(args.qualification), sha256=sha256(args.qualification)),
        anchor=dict(path=str(args.anchor), sha256=sha256(args.anchor), bytes=args.anchor.stat().st_size),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        torch=str(torch.__version__), physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        cpu_affinity=sorted(os.sched_getaffinity(0)), thresholds=THRESHOLDS["bf16"], paths=[],
        boundary="two paths from the retained real-update anchor; three continuing updates each; forward-only scalar capture, no timing claim",
        limitations=["original qualification failures remain unchanged",
            "new source and hardware are recorded; original observation hashes are checked, never assumed",
            "first differing scalar boundary is not the earliest vector/operator cause",
            "allocation replay explains rounded inputs, not the accumulated training trajectory"])
    atomic(args.output, record)
    try:
        bundle = torch.load(args.anchor, map_location="cpu", weights_only=False, mmap=True)
        anchor, inputs = bundle["anchor"], bundle["inputs"]
        original = case["paths"]
        if [tensor_hash(x) for x in inputs] != original["input_hashes"]:
            raise ValueError("retained input hashes differ from the original")
        if paths.screen(anchor)["sha256"] != original["fixed_start"]["sha256"]:
            raise ValueError("retained anchor data differs from the original")
        record["input_and_anchor_bytes_match_original"] = True
        budget, traces = CaptureBudget(64 << 20), []
        for name in ("baseline", "combined"):
            variant = original["resolved_candidates"][name]
            model, optimizer, state = paths.build_candidate(config, name, anchor,
                [args.device], "bf16", "auto", overrides=variant)
            if paths.screen(paths.training_data(model, optimizer, state))["sha256"] != original["fixed_start"]["sha256"]:
                raise RuntimeError("loaded candidate does not match the original anchor")
            prior = next(row for row in original["candidates"] if row["name"] == name)
            row = dict(name=name, config=model.config.to_dict(), updates=[])
            record["paths"].append(row)
            current_traces = []
            for index in range(3):
                observation, state, work, trace = capture_update(model, optimizer, state,
                    inputs[index+1], "bf16", args.device, options["chunk"], budget)
                summary = paths.screen(observation)
                previous = prior["updates"][index]["summary"]
                row["updates"].append(dict(index=index, work=work, routing=trace.summary(),
                    observation_identity={k: summary[k] for k in ("sha256", "discrete_sha256", "support_sha256")},
                    matches_original={k: summary[k] == previous[k] for k in ("sha256", "discrete_sha256", "support_sha256")},
                    returned_strengths=[segment["strengths"].tolist() for segment in observation["forward"]["segments"]],
                    optimizer_update=observation["optimizer_update"]))
                current_traces.append(trace)
                del observation
                atomic(args.output, record)
            traces.append(current_traces)
            del model, optimizer, state
            gc.collect()
        comparisons, payloads = [], []
        for index, (reference, candidate) in enumerate(zip(*traces)):
            compared = compare_traces(reference, candidate, THRESHOLDS["bf16"], max_witnesses=2)
            witnesses = []
            for coordinate in compared["witnesses"]:
                key = coordinate["call"], coordinate["region"]
                graph = reference.model.graph
                roles = [graph.nodes[node].role == "H" for node in graph.regions[key[1]]]
                amount = config.budget or max(1, graph.config.n//32)
                items = [allocation_payload(trace, key) for trace in (reference, candidate)]
                replay = [replay_allocation(item, roles, amount, config.allocation,
                    graph.config.partition, sample=coordinate["sample"], position=coordinate["time_index"],
                    device=args.device, precision="bf16") for item in items]
                witnesses.append(dict(coordinate=coordinate, roles=roles, budget=amount, paths=replay))
                payloads.append(dict(update=index, coordinate=coordinate, roles=roles,
                    budget=amount, profile=config.allocation, partition=graph.config.partition, paths=items))
            comparisons.append(dict(update=index, comparison=compared, allocation_witnesses=witnesses))
        record.update(state="completed", comparisons=comparisons, captured_bytes=budget.used,
            original_observations_reproduced=all(all(update["matches_original"].values())
                for row in record["paths"] for update in row["updates"]))
        if payloads:
            size = sum(value.numel()*value.element_size() for _, value in paths.leaves(payloads)
                       if isinstance(value, torch.Tensor))
            if size > 8 << 20:
                raise RuntimeError("minimal allocation witness exceeds 8MiB")
            atomic_tensor(payload_path, payloads)
            record["witness"] = dict(path=str(payload_path), sha256=sha256(payload_path), bytes=payload_path.stat().st_size)
        if len(json.dumps(record).encode()) > 32 << 20:
            raise RuntimeError("result exceeds 32MiB")
        atomic(args.output, record)
        print(json.dumps(dict(state=record["state"], original_observations_reproduced=record["original_observations_reproduced"],
            support_differences=[row["comparison"]["support_differences"] for row in comparisons])))
    except Exception as error:
        record.update(state="execution_failed", error=repr(error))
        atomic(args.output, record)
        raise


if __name__ == "__main__":
    main()
