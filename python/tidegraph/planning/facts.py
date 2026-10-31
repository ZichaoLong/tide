"""Static configuration facts for reports; never execute or decode model results.

Architecture §2.2 keeps source classification separate from parameter sharing.
KV windows, positions and convolution slots retain original logical-time units.
"""
from collections import Counter
from copy import deepcopy
from .model import fingerprint
from .workload import validate_plan


def _groups(configs):
    groups = {}
    for owner, config in enumerate(configs):
        key = fingerprint(config)
        if key not in groups:
            groups[key] = dict(config=deepcopy(config), owners=[])
        groups[key]["owners"].append(owner)
    return list(groups.values())


def describe(plan):
    """Summarize all node/region configurations, not just their first member."""
    inventory = validate_plan(plan)
    graph, memories, regions = plan["graph"], plan["memories"], plan["regions"]
    classes = plan["classes"]
    source_nodes = [a+1 for a, _, _ in graph["edges"]]
    classification = ("source-node; external=0" if classes["edges"] == source_nodes
                      and all(x == 0 for x in classes["inputs"]) else "explicit")
    return dict(schema="tide-core5-configuration-facts-v1", model_id=plan["model_id"],
        workload_id=plan["sha256"], recipe=deepcopy(plan.get("recipe")),
        nodes=graph["nodes"], regions=graph["regions"], edges=len(graph["edges"]),
        ranks=sorted(set(graph["ranks"])), graph_counts=deepcopy(graph["counts"]),
        parameters=inventory["totals"], samples=plan["samples"], positions=plan["positions"],
        stride=plan["stride"], seed=plan["seed"], input_pattern=plan.get("input_pattern"),
        input_policy=plan["input_policy"],
        families=dict(sorted(Counter(c.get("memory_family", "kv") for c in memories).items())),
        widths=sorted({c["width"] for c in memories}),
        heads=sorted({c["heads"] for c in memories}),
        ffn_widths=sorted({c["ffn_width"] for c in memories}),
        kv_windows=sorted({c["window"] for c in memories if "memory_family" not in c}),
        source_classification=classification, source_class_count=len(set(classes["edges"]+classes["inputs"])),
        source_binding=deepcopy(plan["source_binding"]),
        memories=_groups(memories), regional=_groups(regions))


def from_reviewed_model(model):
    """Reconstruct static recipe facts only when the recorded identity matches.

    Old reviewed records omit the seed. It is not inferred for workload replay:
    only mathematical model identity is certified here; no numerical run occurs.
    """
    from .recipes import prepare
    recipe = model["recipe"]
    request = deepcopy(recipe["requested"])
    if recipe.get("scale"):
        request["target_parameters"] = recipe["scale"]["target_parameters"]
    plan, _ = prepare(request, samples=model["samples"], positions=model["positions"],
                      seed=model.get("seed", 7), input_pattern=model.get("input_pattern"))
    if plan["model_id"] != model["model_id"]:
        raise ValueError("reviewed model identity differs from current recipe; use its fixed source")
    facts = describe(plan)
    if facts["parameters"] != model["parameters"]:
        raise ValueError("reviewed parameter inventory differs")
    facts["workload_id"] = model["workload_id"]
    if "seed" not in model:
        facts["seed"] = None
    return facts
