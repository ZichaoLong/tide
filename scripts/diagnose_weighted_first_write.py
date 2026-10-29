#!/usr/bin/env python3
"""Isolate an original wide fixture's first KV writes with bounded tensor reads.

Only an input-only region with identical external input/source parameters is
accepted. No full reference trace, downstream prediction or 9B model allocation.
Per-tensor hashes validate exactly the bytes used, not the unread binary file.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
from durable_records import write_json
from source_identity import source_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1];sys.path.insert(0, str(root/"python"))
    from tidegraph.weighted.tensors import norm
    from tidegraph.weighted.packing import project
    from tidegraph.weighted.kv_projection import prepare
    from tidegraph.weighted.kv_projection_batch import prepare as private
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    path = args.fixture.resolve();data = json.loads(path.read_text())
    if data["schema"] != "tide-weighted-benchmark-fixture-v2":
        p.error("requires exported binary fixture v2")
    blob = data["binary"]; binary = path.parent/blob["file"]
    if Path(blob["file"]).name != blob["file"] or blob["byte_order"] != "little" or sys.byteorder != "little":
        p.error("requires local little-endian binary")
    if binary.stat().st_size != blob["nbytes"]:
        p.error("binary size mismatch")
    graph = data["graph"]
    ids = [i for i, n in enumerate(graph["nodes"]) if n["region"] == 0]
    # Exported graph edges are dataclass dictionaries.
    if any(e["target"] in ids for e in graph["edges"]):
        p.error("first region is not input-only")
    selected = [[x for x in data["inputs"] if graph["inputs"][x["port"]] == i] for i in ids]
    if any(len(xs) != 1 for xs in selected) or len({xs[0]["value"] for xs in selected}) != 1:
        p.error("requires one identical external atom per first-region node")
    extras = [data["nodes"][i]["extra"] for i in ids]
    classes = [data["classes"]["inputs"][xs[0]["port"]] for xs in selected]
    roles = [{role: row[f"aggregate_program.transforms.{label}.{role}"] for role in ("a", "b", "embedding")}
             for row, label in zip(extras, classes)]
    if any(r != roles[0] for r in roles) or len({r["kernel.base"] for r in extras}) != 1:
        p.error("requires actual shared source matrices/base; no equal-value sharing inference")
    out = args.output_dir.resolve();out.mkdir(parents=True, exist_ok=False)
    cache, verified = {}, {}
    def tensor(index):
        if index in cache:return cache[index]
        row = data["tensors"][index]
        if row["dtype"] != "float32" or row["encoding"] != "binary":
            raise ValueError("selected operand requires stored FP32 bytes")
        if row["nbytes"] > 128*1024**2 or row["offset"] < 0 or row["offset"]+row["nbytes"] > blob["nbytes"]:
            raise ValueError("selected tensor exceeds bounded read")
        with binary.open("rb") as stream:
            stream.seek(row["offset"]);raw = stream.read(row["nbytes"])
        sha = hashlib.sha256(raw).hexdigest()
        if sha != row["sha256"]:raise ValueError("selected tensor hash mismatch")
        value = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone().reshape(row["shape"])
        cache[index], verified[str(index)] = value, sha
        if sum(t.numel()*t.element_size() for t in cache.values()) > 512*1024**2:
            raise ValueError("local witness exceeds 512MiB operand budget")
        return value
    epsilon = data["memories"][ids[0]]["epsilon"]
    memories = [SimpleNamespace(config=SimpleNamespace(epsilon=epsilon), base=tensor(r["kernel.base"]),
        adapter_u=tensor(r["kernel.adapter_u"]), adapter_v=tensor(r["kernel.adapter_v"]),
        summary=tensor(r["kernel.summary"])) for r in extras]
    payload = tensor(selected[0][0]["value"]);z = payload[1:]
    a, b, e = [tensor(roles[0][k]) for k in ("a", "b", "embedding")]
    alpha = data["alpha_agg"];m = memories[0]
    def normalized(x):return x/(epsilon+x.square().mean(-1, keepdim=True)).sqrt()
    def literal(x, aa, bb, ee):return x+alpha*(bb@torch.nn.functional.silu(aa@normalized(x)+ee))
    def kv(x, dtype):
        n = normalized(x.to(dtype))
        return m.base.to(dtype)@n+m.adapter_u.to(dtype)@(m.adapter_v.to(dtype)@n)
    with torch.no_grad():
        scalar = literal(z, a, b, e)
        rows = z.expand(len(ids), -1)
        batch = rows+alpha*project(torch.nn.functional.silu(project(norm(rows, epsilon), a)+e), b)
        exact_content = literal(z.double(), a.double(), b.double(), e.double())
        exact_key = kv(exact_content, torch.float64)
        key = kv(scalar, torch.float32)
        stats = {}
        shared = private(memories, prepare(m, torch.cat((payload[:1], scalar))), stats)[0]["values"]
        combined = private(memories, prepare(m, torch.cat((payload[:1], batch[0]))), {})[0]["values"]
        def comparison(x, y):
            d = x.double()-y.double()
            return dict(max_abs=float(d.abs().max()), strict_close=bool(torch.allclose(x.double(), y.double(), atol=1e-6, rtol=1e-5)),
                        failing_coordinates=int((d.abs() > 1e-6+1e-5*y.double().abs()).sum()))
        # New local intervention: all arithmetic from received content through
        # normalization/projection is coherent FP64, then one explicit cast.
        promoted = kv(exact_content, torch.float64).float()
        report = dict(schema="tide-first-write-isolation-v1", source=source_state(root),
            fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), workload=data["workload"],
            binary_declared_sha256=blob["sha256"], verified_tensor_sha256=verified,
            region=0, nodes=ids, operand_bytes=sum(t.numel()*t.element_size() for t in cache.values()),
            source_batch_vs_scalar=comparison(batch[0], scalar),
            private_batch_vs_scalar=comparison(shared, key),
            source_and_private_batch_vs_scalar=comparison(combined, key),
            scalar_vs_exact=comparison(key, exact_key), private_vs_exact=comparison(shared, exact_key),
            combined_vs_exact=comparison(combined, exact_key), coherent_local64_vs_exact=comparison(promoted, exact_key),
            stats=stats, state="diagnosed", scope="first input-only region local maps; no full graph, routing, training or throughput qualification")
    torch.save(dict(payload=payload, source_a=a, source_b=b, source_embedding=e, base=m.base,
        adapter_u=m.adapter_u, adapter_v=m.adapter_v, scalar_content=scalar, batch_content=batch[0],
        scalar_key=key, shared_key=shared, combined_key=combined, exact_key=exact_key), out/"witness.pt")
    report["witness_sha256"] = hashlib.sha256((out/"witness.pt").read_bytes()).hexdigest()
    write_json(out/"result.json", report)


if __name__ == "__main__":main()
