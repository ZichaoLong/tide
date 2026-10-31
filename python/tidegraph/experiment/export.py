"""Static material export for the independent C++ process, without execution."""
from dataclasses import asdict
from pathlib import Path
import shutil
from .records import fingerprint, write_json


def material(resolved, destination):
    from ..planning.materialize import materialize
    from ..native_loader import load_native
    from ..native_model import encode_model
    from .binary_material import TensorWriter
    from .identity import package_identity
    path = Path(destination)
    if path.exists():
        raise FileExistsError(path)
    plan = resolved["plan"]
    minimum = plan["memory_floor"]["logical_parameter_bytes"]
    if minimum+resolved["resources"].get("disk_reserve_bytes", 1024**3) > shutil.disk_usage(path.parent).free:
        raise ValueError("standalone material: insufficient disk for static tensor export and reserve")
    library = resolved["site"]["paths"].get("native_library")
    if library is None:
        raise ValueError("site.paths.native_library: standalone material export needs a matching Python native adapter; the launched C++ runtime stays independent")
    core = load_native(library, backend="cpu")
    binding, inputs, span = materialize(plan, max_bytes=resolved["resources"]["max_materialize_bytes"])
    graph, model = encode_model(core, binding.graph, binding.model)
    writer = TensorWriter(path.with_suffix(".tensors.bin"))
    try:
        tensor = writer.tensor
        weights = []
        for node in model.nodes:
            row = {name: tensor(getattr(node, name), zero=True) for name in ("decay", "weight", "bias", "read")}
            row["extra"] = {name: tensor(x) for name, x in node.extra.items()}
            weights.append(row)
        regions = [{name: tensor(x) for name, x in row.extra.items()} for row in model.regions]
        scales = {name: [tensor(x) for x in getattr(model, name)]
                  for name in ("input_scale", "output_scale", "edge_scale", "agg_scale")}
        external = [dict(batch=x.batch, port=x.port, position=x.position, time=x.time, value=tensor(x.value)) for x in inputs]
        blob = writer.finish()
    except BaseException:
        writer.abort()
        raise
    graph_record = dict(nodes=[asdict(x) for x in binding.graph.nodes], edges=[asdict(x) for x in binding.graph.edges],
        regions=[asdict(x) for x in binding.graph.regions], inputs=list(binding.graph.inputs), outputs=list(binding.graph.outputs),
        layout=asdict(binding.graph.ports), domain=asdict(binding.graph.domain), native_identity=graph.identity)
    families = {c.get("memory_family", "kv") for c in plan["memories"]}
    config = dict(topology=plan["graph"]["kind"], family=next(iter(families)) if len(families) == 1 else "heterogeneous",
                  **{k: plan[k] for k in ("width", "samples", "positions", "seed")}, workload_sha256=plan["sha256"])
    from .config import resident_limits_defaults
    from ..planning.execution import ExecutionSpec
    limits = resident_limits_defaults(plan, ExecutionSpec(**resolved["execution"]["requested"]),
                                     resolved["measurement"], resolved["resources"])
    record = dict(schema="tide-weighted-benchmark-fixture-v2", source="installed:"+package_identity()["installed_files_sha256"],
        dirty=False, config=config, span=span, graph=graph_record, memories=[asdict(x) for x in binding.memories],
        regions=[asdict(x) for x in binding.regions], classes=asdict(binding.source_classes), alpha_agg=plan["alpha_agg"],
        tensors=writer.rows, nodes=weights, region_weights=regions, scales=scales, inputs=external, limits=limits,
        binary=blob, max_materialize_bytes=resolved["resources"]["max_materialize_bytes"],
        parameter_aliases=model.parameters().alias_partitions(), workload=plan)
    record["fixture_sha256"] = fingerprint(dict(workload_id=plan["sha256"], graph=graph_record,
                                             tensors=writer.rows, inputs=external, binary=blob))
    write_json(path, record, exclusive=True)
    return record
