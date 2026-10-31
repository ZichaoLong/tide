"""Resolve human experiment JSON using the public allocation-free validators."""
from copy import deepcopy
from dataclasses import fields as dataclass_fields
import math
import os
from pathlib import Path
import re
from ..planning.execution import ExecutionSpec
from ..planning.hierarchy import execution_view, non_aligned, with_feedback
from ..planning.model import resolve_model, validate_model, workload
from ..planning.recipes import prepare
from .records import fingerprint, read_json
from .site import fields, positive, load as load_site


SCHEMA = "tide-experiment-v1"
RESOLVED_SCHEMA = "tide-experiment-resolved-v1"
RESOURCE_FIELDS = {"max_materialize_bytes", "checkpoint_max_bytes", "devices", "workspace_bytes",
                   "journal_bytes", "activation_bytes", "communication_bytes", "logical_windows",
                   "live_results", "host_results", "retained_windows", "host_capacity_bytes", "host_free_bytes",
                   "host_reserve_bytes", "disk_free_bytes", "disk_reserve_bytes", "device_reserve_bytes",
                   "resident_limits"}


def _object(value, allowed, name):
    return fields(deepcopy(value), allowed, name)


def _model(value, task, base_dir):
    fields(value, {"recipe", "definition", "resolved", "path", "transforms"}, "model")
    choices = set(value) & {"recipe", "definition", "resolved", "path"}
    if len(choices) != 1:
        raise ValueError("model: choose exactly one of recipe, definition, resolved or path")
    if "path" in value:
        if value.get("transforms"):
            raise ValueError("model.transforms: put transformations inside the referenced model document")
        path = (base_dir/Path(value["path"])).resolve()
        data = read_json(path)
        return _model({"resolved": data} if data.get("schema") == "tide-weighted-model-v1" else data,
                      task, path.parent)
    recipe = None
    if "recipe" in value:
        # prepare has its own formal Config validators for all public modules.
        original, _ = prepare(value["recipe"])
        recipe = original["recipe"]
        model, inv = resolve_model(**{k: original[k] for k in (
            "graph", "memories", "regions", "classes", "source_binding", "source_hidden_width", "alpha_agg")})
    elif "definition" in value:
        model, inv = resolve_model(**value["definition"])
    else:
        model = deepcopy(value["resolved"])
        inv = validate_model(model)
    transforms = value.get("transforms", [])
    if not isinstance(transforms, list):
        raise ValueError("model.transforms: expected an ordered list")
    for i, change in enumerate(transforms):
        options = deepcopy(change)
        kind = options.pop("kind", None)
        if kind not in {"non-aligned", "feedback"}:
            raise ValueError(f"model.transforms[{i}].kind: choose non-aligned or feedback")
        graph = (non_aligned if kind == "non-aligned" else with_feedback)(model["graph"], **options)
        definition = {k: model[k] for k in (
            "memories", "regions", "classes", "source_binding", "source_hidden_width", "alpha_agg")}
        if kind == "feedback":
            definition["classes"]["edges"].append(graph["edges"][-1][0]+1)
        model, inv = resolve_model(graph, **definition)
    keys = {k: task[k] for k in ("samples", "positions", "seed", "input_pattern")}
    keys.update({k: task[k] for k in ("stride", "input_origins") if k in task})
    return workload(model, recipe=recipe, **keys)


def _task(value):
    fields(value, {"samples", "positions", "seed", "input_pattern", "stride", "input_origins",
                   "objective", "optimizer", "plugin"}, "task")
    result = dict(samples=1, positions=1, seed=7, input_pattern=None,
                  objective=dict(kind="weighted-output-square", scale=.001),
                  optimizer=dict(kind="sgd", lr=.001, momentum=.3, weight_decay=.01))
    result.update(deepcopy(value))
    for key in ("samples", "positions", "seed"):
        positive(result[key], "task."+key, 0 if key == "seed" else 1)
    objective = fields(result["objective"], {"kind", "scale"}, "task.objective")
    if objective.get("kind") != "weighted-output-square":
        raise ValueError("task.objective.kind: use weighted-output-square or define task.plugin")
    objective.setdefault("scale", .001)
    _finite(objective["scale"], "task.objective.scale", minimum=0)
    optimizer = fields(result["optimizer"], {"kind", "lr", "momentum", "weight_decay"}, "task.optimizer")
    if optimizer.get("kind") != "sgd":
        raise ValueError("task.optimizer.kind: built-in task uses sgd; custom recipes belong in task.plugin")
    for key, default in (("lr", .001), ("momentum", .3), ("weight_decay", .01)):
        optimizer.setdefault(key, default)
        _finite(optimizer[key], f"task.optimizer.{key}", minimum=0)
    if result.get("plugin") is not None:
        plugin = fields(result["plugin"], {"module", "factory", "options"}, "task.plugin")
        if not isinstance(plugin.get("module"), str) or not plugin["module"]:
            raise ValueError("task.plugin.module: specify an installed downstream module")
        plugin.setdefault("factory", "create_task")
        if not isinstance(plugin["factory"], str) or not plugin["factory"].isidentifier():
            raise ValueError("task.plugin.factory: expected a Python function name")
        plugin.setdefault("options", {})
        if not isinstance(plugin["options"], dict):
            raise ValueError("task.plugin.options: expected a JSON object")
    return result


def _finite(value, where, minimum=0):
    if type(value) not in {int, float} or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{where}: expected a finite number >= {minimum}")
    return value


def _measurement(value):
    fields(value, {"mode", "steps", "warmup", "checkpoint_check", "validation", "timeout_seconds", "observation"}, "measurement")
    result = dict(mode="inference", steps=2, warmup=1, checkpoint_check=True,
                  validation="strict", timeout_seconds=1800, observation="none")
    result.update(deepcopy(value))
    if result["mode"] not in {"inference", "training"}:
        raise ValueError("measurement.mode: choose inference or training")
    if result["validation"] not in {"strict", "explore", "none"}:
        raise ValueError("measurement.validation: choose strict, explore or none; none is unchecked")
    if type(result["checkpoint_check"]) is not bool:
        raise ValueError("measurement.checkpoint_check: expected true or false")
    for key in ("steps", "warmup", "timeout_seconds"):
        positive(result[key], "measurement."+key, 0 if key == "warmup" else 1)
    if result["observation"] not in {"none", "packing", "profile"}:
        raise ValueError("measurement.observation: choose none, packing or profile")
    if result["observation"] != "none" and result["steps"] != 1:
        raise ValueError("measurement.steps: observation uses one measured window in a separate process")
    return result


def _resources(value, site, measurement):
    result = deepcopy(site["resources"])
    result.update(deepcopy(value))
    fields(result, RESOURCE_FIELDS, "resources")
    result.setdefault("max_materialize_bytes", 2*1024**3)
    result.setdefault("checkpoint_max_bytes", 2*1024**3)
    windows = measurement["steps"]+measurement["warmup"]+int(measurement["checkpoint_check"])
    result.setdefault("logical_windows", windows)
    positive(result["logical_windows"], "resources.logical_windows")
    if result["logical_windows"] < windows:
        raise ValueError(f"resources.logical_windows: need at least {windows} for warmup, measured and resume windows")
    result.setdefault("live_results", 1)
    host_results = (measurement["warmup"]+measurement["steps"] if measurement["validation"] != "none" else 0)
    host_results += 2 if measurement["checkpoint_check"] else 1
    result["host_results"] = max(result.get("host_results", 0), host_results)
    result.setdefault("retained_windows", 1)
    mappings = {"devices", "workspace_bytes", "journal_bytes", "activation_bytes", "communication_bytes", "resident_limits"}
    for key, val in result.items():
        if key not in mappings:
            positive(val, "resources."+key, 0 if key.endswith("reserve_bytes") else 1)
        elif not isinstance(val, dict):
            raise ValueError(f"resources.{key}: expected per-device mappings")
    for device, row in result.get("devices", {}).items():
        if not isinstance(device, str) or not re.fullmatch(r"cpu|(?:cuda|npu):[0-9]+", device):
            raise ValueError(f"resources.devices.{device}: expected cpu, cuda:N or npu:N")
        fields(row, {"capacity_bytes", "free_bytes", "reserve_bytes"}, f"resources.devices.{device}")
        for key, val in row.items():
            positive(val, f"resources.devices.{device}.{key}", 0)
    resident = result.get("resident_limits", {})
    allowed = {"queue", "stages", "cache", "tensor_bytes", "operator_bytes", "timeout_ms", "prefill"}
    if measurement["mode"] == "training":
        allowed |= {"windows", "retained_bytes", "backward_bytes", "optimizer_bytes", "total_bytes"}
    fields(resident, allowed, "resources.resident_limits")
    for key, val in resident.items():
        if key == "prefill":
            if type(val) is not bool:
                raise ValueError("resources.resident_limits.prefill: expected true or false")
        else:
            positive(val, "resources.resident_limits."+key)
    return result


def resident_limits_defaults(plan, execution, measurement, resources):
    """Structural finite-window bounds only; never a reference event trace."""
    graph, span, batch = plan["graph"], plan["span"], plan["samples"]
    incoming = len(graph["edges"])+len(graph["inputs"])
    default = dict(queue=max(64, batch*max(1, incoming)*span),
        stages=max(32, span*graph["regions"]*batch),
        cache=max(2*span+2, resources["logical_windows"]*span+1),
        tensor_bytes=1024**3, operator_bytes=128*1024**2, timeout_ms=60000,
        prefill=execution.schedule == "prefill")
    if measurement["mode"] == "training":
        default.update(windows=1, retained_bytes=1024**3, backward_bytes=2*1024**3,
                       optimizer_bytes=256*1024**2, total_bytes=5*1024**3)
    default.update(resources.get("resident_limits", {}))
    for key, val in default.items():
        if key != "prefill":
            positive(val, "resources.resident_limits."+key)
    return default


def resolve(value, *, site=None, base_dir=None):
    """Expand one JSON file/dict without Torch, device allocation or numeric prepass.

    A resolved record is self-contained. Reopening it verifies its content hash;
    task/checkpoint and installed-build checks remain runtime responsibilities.
    """
    base = Path(base_dir or Path.cwd())
    if isinstance(value, (str, os.PathLike)):
        path = (base/Path(value)).resolve()
        value, base = read_json(path), path.parent
    if not isinstance(value, dict):
        raise ValueError("configuration: expected an experiment JSON object")
    if value.get("schema") == RESOLVED_SCHEMA:
        body = {k: v for k, v in value.items() if k != "config_id"}
        if fingerprint(body) != value.get("config_id"):
            raise ValueError("config_id: resolved configuration content has changed")
        if site is not None:
            raise ValueError("site: cannot override a frozen resolved configuration; resolve its requested input again")
        return deepcopy(value)
    value = deepcopy(value)
    fields(value, {"schema", "name", "description", "model", "task", "execution", "site",
                   "measurement", "resources"}, "configuration")
    if value.get("schema") != SCHEMA:
        raise ValueError(f"schema: expected {SCHEMA}")
    selected_site = load_site(value.get("site") if site is None else site, base_dir=base)
    task = _task(value.get("task", {}))
    measurement = _measurement(value.get("measurement", {}))
    resources = _resources(value.get("resources", {}), selected_site, measurement)
    try:
        plan, inventory = _model(value.get("model", {}), task, base)
    except (TypeError, KeyError, ValueError) as error:
        raise ValueError(f"model/task: {error}; correct the public model or input definition") from error
    execution_fields = {f.name for f in dataclass_fields(ExecutionSpec)}
    requested = _object(value.get("execution", {}), execution_fields | {"family", "placement_strategy"}, "execution")
    if "device" not in requested:
        raise ValueError("execution.device: specify cpu, cuda:N or npu:N; devices are never silently selected")
    family = requested.pop("family", "auto")
    node_devices = requested.pop("node_devices", None)
    strategy = requested.pop("placement_strategy", "explicit" if node_devices else "balanced-shared-groups")
    for key, val in selected_site["threads"].items():
        requested.setdefault(key, val)
    try:
        spec = ExecutionSpec(**requested)
        if spec.return_mode != "complete" and measurement["validation"] != "none":
            raise ValueError("execution.return_mode: independent validation requires complete return; "
                             "run check separately, then explicitly set measurement.validation=none for outputs/metrics")
        if resources.get("resident_limits") and spec.placement != "resident":
            raise ValueError("resources.resident_limits: only applies to execution.placement=resident")
        if resources.get("resident_limits", {}).get("prefill", spec.schedule == "prefill") != (spec.schedule == "prefill"):
            raise ValueError("resources.resident_limits.prefill: must match execution.schedule")
        if spec.placement == "resident":
            resources["resident_limits"] = resident_limits_defaults(plan, spec, measurement, resources)
        graph_view = execution_view(plan, family=family)
        from ..planning.placement import plan_payload_placement
        devices = spec.peer_devices or (spec.device,)
        # A site can describe more cards than this run selects. Resource budgets
        # are resolved for exactly the selected owners, without summing HBM.
        for key in ("devices", "workspace_bytes", "activation_bytes", "communication_bytes", "journal_bytes"):
            if set(value.get("resources", {}).get(key, {}))-set(devices):
                raise ValueError(f"resources.{key}: contains a device absent from execution.device/peer_devices")
            if key in resources:
                resources[key] = {d: row for d, row in resources[key].items() if d in devices}
        limits = {d: max(0, row.get("free_bytes", row.get("capacity_bytes"))-row.get("reserve_bytes",
                             resources.get("device_reserve_bytes", 4*1024**3 if d != "cpu" else 0)))
                  for d, row in resources.get("devices", {}).items() if "free_bytes" in row or "capacity_bytes" in row}
        mapping = plan_payload_placement(plan, devices, strategy=strategy,
                    node_devices=node_devices, dtype=spec.dtype, device_limits=limits)
        if spec.peer_devices or node_devices:
            spec = ExecutionSpec(**dict(requested, node_devices=tuple(mapping["node_devices"])))
        execution = spec.record()
        from ..planning.capacity import capacity_budget
        budget = capacity_budget(plan, execution, placement=mapping, mode=measurement["mode"],
                    optimizer=task["optimizer"]["kind"], checkpoint=True, resources=resources)
    except (TypeError, KeyError, ValueError) as error:
        raise ValueError(f"execution/resources: {error}; inspect device mapping, module support and explicit strategy") from error
    task["stride"] = plan["stride"]
    task["input_origins"] = graph_view["input_clock"]["origins"]
    execution_id = fingerprint(dict(schema="tide-experiment-execution-v1",
        execution_id=execution["execution_id"], view_id=graph_view["view_id"], placement_id=mapping["placement_id"]))
    body = dict(schema=RESOLVED_SCHEMA, name=value.get("name", "experiment"),
        description=value.get("description", ""), plan=plan, inventory=inventory,
        execution=execution, graph_view=graph_view, payload_placement=mapping, capacity_budget=budget,
        task=task, site=selected_site, measurement=measurement, resources=resources,
        model_id=plan["model_id"], workload_id=plan["sha256"],
        task_id=fingerprint(dict(workload_id=plan["sha256"], task=task)), execution_id=execution_id,
        requested=value,
        resolved=dict(model=plan, task=task, execution=execution["effective"], family=graph_view["family"],
                      site=selected_site, measurement=measurement, resources=resources))
    return dict(body, config_id=fingerprint(body))
