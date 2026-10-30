"""Views of exact configurations and evidence; not a Cartesian certification."""
from pathlib import Path
import sys


def facts_for(row):
    if row.get("configuration_facts"):
        facts = row["configuration_facts"]
    elif row.get("reviewed_model"):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"python"))
        from tidegraph.planning.facts import from_reviewed_model
        facts = from_reviewed_model(row["reviewed_model"])
    else:
        raise ValueError("configuration facts unavailable; supply reviewed model metadata or a new run record")
    if facts["model_id"] != row["model_id"] or facts["parameters"] != row["parameters"]:
        raise ValueError("configuration facts/record identity mismatch")
    return facts


def view(row):
    facts = facts_for(row)
    request = (facts.get("recipe") or {}).get("requested", {})
    return dict(case=row["case"], kind=row["kind"], source=row["source"],
        model_id=row["model_id"], workload_id=row["workload_id"], execution_id=row["execution_id"],
        topology=request.get("topology", "explicit"), profile=request.get("profile", "baseline"),
        families=list(facts["families"]), nodes=facts["nodes"], regions=facts["regions"],
        edges=facts["edges"], widths=facts["widths"], parameters=facts["parameters"]["trainable"],
        samples=facts["samples"], positions=facts["positions"],
        requested=row["requested"], state=row["state"],
        correctness_qualified=row["correctness_qualified"], throughput_qualified=row["throughput_qualified"],
        evidence=row.get("evidence_file", row["run"]), facts=facts)


def models(rows):
    result = {}
    for row in rows:
        facts = facts_for(row)
        key = (row["case"], row["model_id"], row["workload_id"])
        result.setdefault(key, dict(case=row["case"], facts=facts))
    return list(result.values())


def table(rows, kind):
    """Display fields derived from all configurations; never assign a pass by name."""
    join = lambda xs: ",".join(map(str, xs)) or "-"
    if kind == "models":
        headers = ["case", "nodes/regions/edges", "d/heads/FFN", "parameters", "B/P", "KV windows (logical time)", "source classes / sharing"]
        body = []
        for row in rows:
            f = row["facts"]
            body.append([row["case"], f'{f["nodes"]}/{f["regions"]}/{f["edges"]}',
                "/".join(join(f[k]) for k in ("widths", "heads", "ffn_widths")), str(f["parameters"]["trainable"]),
                f'{f["samples"]}/{f["positions"]}', join(f["kv_windows"]),
                f'{f["source_classification"]} / {f["source_binding"]["mode"]}'])
    else:
        headers = ["case", "kind", "profile/families", "parameters", "B/P", "client", "device/placement/dtype", "state", "correctness", "performance", "source"]
        body = []
        for r in rows:
            e = r["requested"]
            body.append([r["case"], r["kind"], r["profile"]+"/"+join(r["families"]), str(r["parameters"]),
                f'{r["samples"]}/{r["positions"]}', e["client"], "/".join(e[k] for k in ("device", "placement", "dtype")),
                r["state"], str(r["correctness_qualified"]), str(r["throughput_qualified"]), r["source"][:7]])
    return headers, body
