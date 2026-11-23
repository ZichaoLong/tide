"""Second finite pilot: paired cost calibration and selected continuation."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .io import atomic_json


GIB = 1024**3


def build_plan(previous: dict, physical_ids: list[int], checkpoint_identity=None, foundation_source=None):
    if len(physical_ids) != 9 or len(set(physical_ids)) != 9:
        raise ValueError("this paired design declares exactly nine eligible candidate cards")
    base = {task["id"]: task for task in previous["tasks"]}
    tasks = []
    phases = [
        dict(id="cpu-isolated", cpu_slots=1, npu_slots=0),
        dict(id="npu-isolated", cpu_slots=0, npu_slots=1),
        dict(id="mixed-concurrent-a", cpu_slots=4, npu_slots=9),
        dict(id="mixed-concurrent-b", cpu_slots=4, npu_slots=9),
        dict(id="capacity-grid", cpu_slots=0, npu_slots=16),
        dict(id="lr-screen", cpu_slots=0, npu_slots=16),
        dict(id="selected-continuation", cpu_slots=0, npu_slots=16),
    ]

    def add(name, original, phase, *, backend="npu", batch=8, length=64,
            threads=8, train_seconds=120, lr=0.0, purpose="cost_calibration", **changes):
        task = copy.deepcopy(base[original])
        task.update(id=name, phase=phase, backend=backend, batch_size=batch,
                    sequence_length=length, cpu_threads=threads, train_seconds=train_seconds,
                    wall_seconds=900, max_steps=20001, lr=lr, purpose=purpose,
                    resume_verify_steps=1, estimated_rss_bytes=8*GIB if batch*length>=2048 else 4*GIB,
                    estimated_output_bytes=2*GIB, closing_reserve_seconds=180)
        task.update(changes)
        if task["resume_verify_steps"] > 1 or batch*length >= 2048:
            task["estimated_output_bytes"] = 4*GIB
        if task["model"].get("config", {}).get("topology", {}).get("timing") == "timed":
            task["estimated_output_bytes"] = 8*GIB
        if checkpoint_identity is not None:
            task["checkpoint_identity"] = copy.deepcopy(checkpoint_identity)
        tasks.append(task)
        return task

    cpu_bindings = [("exchange-x", "npu-exchange-x", 0),
                    ("transformer", "npu-transformer-d128-l4", 40)]
    for short, original, origin in cpu_bindings:
        for threads in (4, 16, 40):
            add(f"cpu-isolated-{short}-t{threads}", original, "cpu-isolated", backend="cpu",
                threads=threads, train_seconds=90,
                cpu_affinity=list(range(origin, origin+threads)),
                comparison_group=f"cpu-{short}-t{threads}", comparison_role="isolated")
    anchors = [("exchange-x", "npu-exchange-x"), ("transformer", "npu-transformer-d256-l6")]
    for repeat in ("a", "b"):
        for slot, (short, original) in enumerate(anchors):
            add(f"npu-isolated-{short}-{repeat}", original, "npu-isolated", batch=16, length=128,
                train_seconds=240, candidate_ids=[physical_ids[slot]],
                cpu_affinity=list(range(80+slot*8,88+slot*8)),
                comparison_group=f"npu-{short}-{repeat}", comparison_role="isolated")
    fills = {
        "a": [
            ("fc32", "npu-fc-k-d128", 32, 64),
            ("x32", "npu-exchange-x", 32, 64),
            ("x64", "npu-exchange-x", 64, 64),
            ("x16-l128", "npu-exchange-x", 16, 128),
            ("x16-l256", "npu-exchange-x", 16, 256),
            ("timed16", "npu-timed-x-group", 16, 64),
            ("shannon8", "npu-exchange-x-shannon", 8, 64),
        ],
        "b": [
            ("fc64", "npu-fc-k-d128", 64, 64),
            ("x32-l128", "npu-exchange-x", 32, 128),
            ("x64-l128", "npu-exchange-x", 64, 128),
            ("x16-l256", "npu-exchange-x", 16, 256),
            ("timed-mixed16", "npu-timed-x-mixed", 16, 64),
            ("timed16-l128", "npu-timed-x-group", 16, 128),
            ("shannon16", "npu-exchange-x-shannon", 16, 64),
        ],
    }
    for repeat in ("a", "b"):
        phase = f"mixed-concurrent-{repeat}"
        for slot, (short, original) in enumerate(anchors):
            add(f"npu-concurrent-{short}-{repeat}", original, phase, batch=16, length=128,
                train_seconds=240, candidate_ids=[physical_ids[slot]],
                cpu_affinity=list(range(80+slot*8,88+slot*8)),
                comparison_group=f"npu-{short}-{repeat}", comparison_role="concurrent")
        for slot, (short, original, batch, length) in enumerate(fills[repeat],2):
            shannon = "shannon" in short
            add(f"npu-load-{short}-{repeat}", original, phase, batch=batch, length=length,
                train_seconds=240, candidate_ids=[physical_ids[slot]],
                cpu_affinity=list(range(80+slot*8,88+slot*8)),
                lr=3e-4 if shannon else 0.0,
                purpose="shannon_learning_and_cost" if shannon else "cost_calibration",
                resume_verify_steps=8 if shannon else 1,
                estimated_rss_bytes=8*GIB)
        for short, original, origin in cpu_bindings:
            add(f"cpu-concurrent-{short}-{repeat}", original, phase, backend="cpu", threads=4,
                train_seconds=240, cpu_affinity=list(range(origin,origin+4)),
                comparison_group=f"cpu-{short}-t4", comparison_role="concurrent")
        add(f"cpu-load-exchange-x-{repeat}", "npu-exchange-x", phase, backend="cpu", threads=16,
            train_seconds=240, cpu_affinity=list(range(160,176)))
        add(f"cpu-load-transformer-{repeat}", "npu-transformer-d128-l4", phase, backend="cpu", threads=40,
            train_seconds=240, cpu_affinity=list(range(200,240)))
    grid = [
        ("x8-l64", "npu-exchange-x", 8,64),
        ("x16-l64", "npu-exchange-x", 16,64),
        ("x32-l64", "npu-exchange-x", 32,64),
        ("x64-l64", "npu-exchange-x", 64,64),
        ("x8-l128", "npu-exchange-x", 8,128),
        ("x8-l256", "npu-exchange-x", 8,256),
        ("fc8-l64", "npu-fc-k-d128", 8,64),
        ("fc16-l64", "npu-fc-k-d128", 16,64),
        ("fc32-l64", "npu-fc-k-d128", 32,64),
        ("tf8-l64", "npu-transformer-d256-l6", 8,64),
        ("tf32-l128", "npu-transformer-d256-l6", 32,128),
        ("tf64-l128", "npu-transformer-d256-l6", 64,128),
    ]
    for short, original, batch, length in grid:
        add("capacity-"+short, original, "capacity-grid", batch=batch, length=length)
    seeds = [20261119,20261120]
    for short, original in [("fc", "npu-fc-k-d128"), ("x", "npu-exchange-x"),
                            ("tf", "npu-transformer-d256-l6")]:
        candidates = []
        for lr in (1e-4, 3e-4, 1e-3):
            for seed in seeds:
                name = f"screen-{short}-lr{lr:g}-s{seed}"
                candidates.append(name)
                add(name, original, "lr-screen", lr=lr, seed=seed, max_steps=257,
                    train_seconds=600, eval_batches=16, resume_verify_steps=8,
                    checkpoint_interval_steps=128,
                    purpose="equal_token_lr_screen", selection_family=short)
        for seed in seeds:
            add(f"continue-{short}-s{seed}", original, "selected-continuation", lr=3e-4, seed=seed,
                max_steps=768, train_seconds=600, eval_batches=16, resume_verify_steps=8,
                checkpoint_interval_steps=128,
                purpose="selected_multistep_continuation", selection_family=short,
                depends_on=candidates,
                resume_select={"rule":"lowest_mean_validation_nll", "candidates":candidates,
                               "seeds":seeds, "updates":257},
                resume_allow_budget_overrides=["max_steps"])
    assert len(tasks)==72 and len({t["id"] for t in tasks})==72
    assert sum(t["backend"]=="npu" for t in tasks)==58
    result = dict(schema="pilot-round-plan-v2", round_id="pilot-r2-20261119", phases=phases, tasks=tasks,
        duration_seconds=5400, dispatch_reserve_seconds=300, finish_reserve_seconds=60,
        term_grace_seconds=25, quota_reserve_bytes=16*GIB, max_output_bytes=512*GIB,
        memory_guard=dict(limit_bytes=192*GIB, reserve_bytes=16*GIB,
                          system_reserve_bytes=128*GIB, poll_seconds=1, admission_factor=1.25),
        candidate_card_snapshot=physical_ids,
        purpose="Shannon fix validation, paired CPU/NPU concurrency and capacity calibration, finite two-seed LR selection and real continuation",
        precision_policy="FP32, original numerical tolerances; only private Shannon cap fix changes the foundation",
        data_policy="unchanged pilot-r1 WikiText103 article holdout and train-only BPE8192",
        cost_policy="lr=0 cost tasks still run forward/backward/clipping/AdamW, with unchanged model weights; not learning evidence",
        comparison_policy="matched cost pairs fix physical card, CPU mask, model, input stream, precision and lr=0; compare shared step prefixes and measured concurrency overlap; unrelated host loads remain uncontrolled",
        learning_policy="per-model lr grid [1e-4,3e-4,1e-3], seeds [20261119,20261120], 257 updates each; pick complete qualified seed-group mean NLL, then continue both seeds for 768 additional updates",
        selection_limits="exploratory within-model LR selection; 4096-token probe, no architecture quality ranking or broad HPO qualification",
        retention="all final application checkpoints; failed per-step recovery comparisons; no deletion of prior-round evidence",
        memory_policy="RSS process-tree software watchdog sampled every second; shared pages may be counted twice; not a kernel hard cap",
        stop_policy="complete this finite round and report for user alignment; no next batch automatically")
    if foundation_source is not None:
        result["foundation_source"] = str(Path(foundation_source).resolve(strict=True))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--first-plan",type=Path,default=Path("configs/pilot-round1/plan.json"))
    p.add_argument("--physical-ids",required=True)
    p.add_argument("--identity",type=Path)
    p.add_argument("--foundation-source",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(argv)
    if args.output.exists():raise FileExistsError(args.output)
    identity=json.loads(args.identity.read_text()) if args.identity else None
    plan=build_plan(json.loads(args.first_plan.read_text()),[int(v) for v in args.physical_ids.split(",")],identity,args.foundation_source)
    atomic_json(args.output,plan)
    print(f"{len(plan['tasks'])} tasks: 58 NPU, 14 CPU; {len(plan['phases'])} phases")


if __name__=="__main__":main()
