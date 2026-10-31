"""Per-owner tensor reservations for a finite experiment, without allocation.

Exact parameter banks, bounded private state and conservative record estimates
are deliberately separate from unknown arenas/allocator peaks. No result here
admits a device run or predicts its actual activity/communication implementation.
"""
from collections import deque
from .execution import ExecutionSpec
from .legacy_topology import integer
from .model import fingerprint
from .placement import plan_payload_placement
from .workload import validate_plan

GIB = 1024**3


def _bytes(value, name):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError(f"{name} must be nonnegative int64 bytes")
    return value


def _counts(plan, windows):
    """Structural upper counts, never an executed ready/selection trace."""
    g = plan["graph"]
    n, positions = g["nodes"], windows*plan["positions"]
    horizon = windows*plan["span"]+max(g["ranks"])+1
    if horizon >= 2**63:
        raise ValueError("resources.logical_windows overflows the original int64 time domain")
    if g["kind"] == "ranked-local":
        return [positions]*n, "aligned original-time positions; all candidates assumed active"
    if g["kind"] == "positive-delay":
        return [horizon]*n, "one action per node/sample/int64 time; feedback may fill every tick"
    # Acyclic upper bound counts all physical input/edge paths, capped by the
    # finite clock range. Distinct paths can collide; no actual work is inferred.
    counts, incoming, outgoing = [0]*n, [0]*n, [[] for _ in range(n)]
    for v in g["inputs"]:
        counts[v] = min(horizon, counts[v]+positions)
    for a, b, _ in g["edges"]:
        incoming[b] += 1
        outgoing[a].append(b)
    ready = deque(v for v in range(n) if incoming[v] == 0)
    visited = 0
    while ready:
        v = ready.popleft()
        visited += 1
        for target in outgoing[v]:
            counts[target] = min(horizon, counts[target]+counts[v])
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
    if visited != n:
        raise ValueError("timed-local capacity requires an acyclic graph")
    return counts, "nonaligned DAG path-count bound capped by original logical time"


def _state(config, writes, scalar):
    d, s = config["width"], config.get("summary_width", 2)
    family = config.get("memory_family", "kv")
    if family == "kv":
        window = config.get("window", 0)
        # A positive write can retain window older integer stamps plus its new
        # stamp; zero reception preserves raw expired entries without pruning.
        rows = min(writes, window+1) if window else writes
        floating = d+1+rows*(2*d+s+1)
        integers = rows+3
    else:
        basic = family in {"affine", "delta"}
        h, k, v = config.get("heads", 1), config.get("key_width", 2), config.get("value_width", 3)
        conv = 0 if basic else h*(2*k+v) if family == "gated_delta" else h*v+2*config.get("groups", 1)*k
        rows = 0 if basic else min(writes, (config.get("conv_size", 1)-1)*config.get("conv_stride", 1)+1)
        floating = d+1+(2 if basic else h)*k*v+2*s+2+rows*conv
        integers = rows+4
    return dict(bytes=floating*scalar+integers*8, float_elements=floating,
                int64_elements=integers, cache_rows=rows)


def _reservation(resources, name, device):
    value = resources.get(name)
    if isinstance(value, dict):
        value = value.get(device)
    return None if value is None else _bytes(value, f"resources.{name}.{device}")


def capacity_budget(plan, execution, *, placement=None, mode="inference", optimizer="sgd",
                    checkpoint=True, resources=None, checkpoint_generations=2):
    """Return a preview for declared finite windows, not automatic admission.

    ``resources`` accepts logical_windows/live_results/retained_windows;
    devices={name: {capacity_bytes, free_bytes, reserve_bytes}}; global device,
    host and disk reserve bytes; and explicit per-device workspace_bytes,
    activation_bytes, communication_bytes and journal_bytes reservations.
    Missing dynamic reservations remain unknown, including for an empty card.
    """
    inv = validate_plan(plan)
    spec = execution if isinstance(execution, ExecutionSpec) else ExecutionSpec(**execution.get("requested", execution))
    if isinstance(execution, dict) and "execution_id" in execution and execution["execution_id"] != spec.record()["execution_id"]:
        raise ValueError("execution identity mismatch")
    if mode not in {"inference", "training"} or optimizer not in {"sgd", "adamw"} or type(checkpoint) is not bool:
        raise ValueError("capacity mode/optimizer/checkpoint must be explicit supported values")
    if type(checkpoint_generations) is not int or checkpoint_generations not in {1, 2}:
        raise ValueError("checkpoint_generations must be 1 or 2; derive it from the declared publication policy")
    resources = dict(resources or {})
    windows = integer(resources.get("logical_windows", 1), "resources.logical_windows")
    results = integer(resources.get("live_results", 1), "resources.live_results")
    host_results = integer(resources.get("host_results", results), "resources.host_results")
    retained = integer(resources.get("retained_windows", 1), "resources.retained_windows")
    if mode == "inference" and retained != 1:
        raise ValueError("resources.retained_windows applies to training autograd windows")
    devices = spec.peer_devices or (spec.device,)
    for key in ("workspace_bytes", "activation_bytes", "communication_bytes", "journal_bytes"):
        if isinstance(resources.get(key), dict) and set(resources[key])-set(devices):
            raise ValueError(f"resources.{key} contains an undeclared device")
    if placement is None:
        placement = plan_payload_placement(plan, devices, dtype=spec.dtype,
            strategy="explicit" if spec.node_devices else "region-round-robin",
            node_devices=spec.node_devices or None)
    if placement.get("placement_id") != fingerprint({k: v for k, v in placement.items() if k != "placement_id"}):
        raise ValueError("placement identity mismatch")
    if placement["workload_id"] != plan["sha256"] or placement["dtype"] != spec.dtype or tuple(placement["devices"]) != devices:
        raise ValueError("placement workload/dtype/devices disagree with execution")
    if spec.node_devices and tuple(placement["node_devices"]) != spec.node_devices:
        raise ValueError("placement node map disagrees with execution.node_devices")
    # Recompute known ownership; a hand-edited byte total is not a capacity fact.
    checked = plan_payload_placement(plan, devices, strategy="explicit",
                                    node_devices=placement["node_devices"], dtype=spec.dtype)
    if any(placement[key] != checked[key] for key in ("per_device", "region_devices", "parameter_devices")):
        raise ValueError("placement storage/owner inventory mismatch")
    scalar = {"float16": 2, "float32": 4, "float64": 8}[spec.dtype]
    score_scalar = 8 if spec.scoring_dtype == "float64" else 4
    train, resident = mode == "training", spec.placement == "resident"
    resident_layout = None
    if resident:
        from .resident_capacity import resident_capacity
        limits = dict(resources.get("resident_limits") or {})
        limits.setdefault("prefill", spec.schedule == "prefill")
        if limits["prefill"] != (spec.schedule == "prefill"):
            raise ValueError("resources.resident_limits.prefill must match execution.schedule")
        resident_layout = resident_capacity(plan, devices, placement["node_devices"], limits)
    g, batch, payload = plan["graph"], plan["samples"], (plan["width"]+1)*scalar
    totals, count_method = _counts(plan, windows)
    one_window, _ = _counts(plan, 1)
    states = [_state(c, total, scalar) for c, total in zip(plan["memories"], totals)]
    components = {}
    python_inference = not resident and not train and spec.client != "libtorch"
    for device, owner in placement["per_device"].items():
        learned = owner["trainable_elements"]
        master = 4*learned if train and spec.dtype == "float16" else 0
        slot_scalar = 4 if spec.dtype == "float16" else scalar
        slots = learned*slot_scalar*(2 if optimizer == "adamw" else 1) if train else 0
        state = batch*sum(states[v]["bytes"] for v in owner["nodes"])
        components[device] = dict(parameters=owner["parameter_bytes"], inert_buffers=owner["inert_buffer_bytes"],
            private_state=state*(3 if resident else 1), history_int64=8*batch*len(owner["regions"]),
            retained_state_snapshots=state*(retained-1) if train else 0,
            input_payloads=0, pending_messages=0, complete_record_estimate=0, requested_return_estimate=0,
            payload_gradients=learned*scalar if train else 0, fp32_masters=master,
            master_gradients=master, optimizer_slots=slots, checkpoint_restore_staging=0,
            checkpoint_optimizer_preflight=0, checkpoint_snapshot_staging=master+slots if checkpoint and train else 0)
        if spec.finite_check_packing == "packed":
            # One <=8MiB cat plus conservative finite-predicate temporaries.
            components[device]["finite_check_workspace"] = 32*1024**2
    if resident:
        for device, values in components.items():
            values["private_state"] = resident_layout["node_private_preflight_bytes"][device]
            values["history_int64"] = 0  # Included in coordinator control reservation.
            values["resident_queue_control"] = resident_layout["coordinator_queue_control_bytes"] if device == spec.device else 0
            values["resident_forward_journal"] = resident_layout["coordinator_journal_bytes"] if device == spec.device else 0
            values["resident_physical_role_banks"] = None
            if train:
                values["fp32_masters"] = values["master_gradients"] = values["optimizer_slots"] = 0
                values["checkpoint_snapshot_staging"] = 0
        if train:
            coordinator = components[spec.device]
            coordinator["fp32_masters"] = coordinator["master_gradients"] = 4*inv["totals"]["trainable"]
            coordinator["optimizer_slots"] = (8 if optimizer == "adamw" else 4)*inv["totals"]["trainable"]
            if checkpoint:
                coordinator["checkpoint_snapshot_staging"] = coordinator["fp32_masters"]+coordinator["optimizer_slots"]
    for v in g["inputs"]:
        components[placement["node_devices"][v]]["input_payloads"] += batch*plan["positions"]*payload
    copy_bytes = dict.fromkeys(devices, 0)
    for a, b, _ in g["edges"]:
        source, target = placement["node_devices"][a], placement["node_devices"][b]
        amount = batch*totals[a]*(payload+6*8)
        pending_owner = spec.device if resident else target
        components[pending_owner]["pending_messages"] += amount
        if source != target:
            copy_bytes[source] += batch*totals[a]*payload
    for v, count in enumerate(one_window):
        device = placement["node_devices"][v]
        event = 4*states[v]["bytes"]+32*payload+256*score_scalar+64*8
        components[device]["complete_record_estimate"] += results*batch*count*event
    for v in g["outputs"]:
        device = placement["node_devices"][v]
        components[device]["requested_return_estimate"] += results*batch*one_window[v]*(payload+3*8)
    for device, values in components.items():
        if spec.return_mode == "complete":
            values["requested_return_estimate"] = values["complete_record_estimate"]
        elif spec.return_mode == "metrics":
            values["requested_return_estimate"] = 64*results
        if checkpoint:
            values["checkpoint_restore_staging"] = (values["parameters"]+values["inert_buffers"]+
                values["fp32_masters"]+values["optimizer_slots"]+values["private_state"]+values["pending_messages"])
            if python_inference:
                # checkpoint.load decodes weights on CPU and load_state_dict
                # copies into the existing device bank. There is no second
                # full device bank. Host decode/copy budgets below remain.
                values["checkpoint_restore_staging"] -= values["parameters"]+values["inert_buffers"]
            if train and spec.client != "libtorch" and not resident:
                values["checkpoint_optimizer_preflight"] = (placement["per_device"][device]["trainable_elements"]*scalar+
                    values["fp32_masters"]+values["optimizer_slots"])
        for key in ("workspace", "activation", "communication", "journal"):
            value = _reservation(resources, key+"_bytes", device)
            if key == "journal" and not train:
                value = 0 if value is None else value
            values[key] = value
    device_info = resources.get("devices", {})
    if not isinstance(device_info, dict) or set(device_info)-set(devices):
        raise ValueError("resources.devices must map declared devices to capacity/free/reserve bytes")
    per_device, rejections = {}, []
    if resident_layout and resident_layout["exceeds_tensor_bytes"]:
        rejections.append("resources.resident_limits.tensor_bytes is below the declared physical state/journal preflight; "
                          "review queue/stages/cache and calibrate before changing explicit budgets")
    for device, values in components.items():
        info = dict(device_info.get(device, {}))
        if set(info)-{"capacity_bytes", "free_bytes", "reserve_bytes"}:
            raise ValueError(f"unknown resources.devices.{device} field")
        for key, value in info.items():
            _bytes(value, f"resources.devices.{device}.{key}")
        if "free_bytes" in info and "capacity_bytes" in info and info["free_bytes"] > info["capacity_bytes"]:
            raise ValueError(f"resources.devices.{device}.free_bytes exceeds capacity_bytes")
        reserve = info.get("reserve_bytes", _bytes(resources.get("device_reserve_bytes", 4*GIB if device != "cpu" else 0),
                                                   "resources.device_reserve_bytes"))
        available = info.get("free_bytes", info.get("capacity_bytes"))
        usable = None if available is None else max(0, available-reserve)
        bank = values["parameters"]+values["inert_buffers"]
        if not resident and usable is not None and bank > usable:
            rejections.append(f"{device}: parameter bank {bank} exceeds {usable} usable bytes; do not allocate")
        # Both return estimates are shown, but the internal full-record cost is
        # reserved until the selected runtime proves it avoids that allocation.
        estimated = sum(value for key, value in values.items() if value is not None and key != "requested_return_estimate")
        per_device[device] = dict(components=values, static_bank_bytes=bank, estimated_peak_bytes=estimated,
            storage_basis="logical canonical bank plus resident physical state/journal preflight; role banks unknown" if resident else "canonical eager payload and inert bank",
            unknown_components=[key for key, value in values.items() if value is None],
            resource_snapshot=info, reserve_bytes=reserve, usable_bytes=usable,
            estimated_remaining_bytes=None if usable is None else usable-estimated,
            reservation_exceeds_available=usable is not None and estimated > usable,
            forward_cross_owner_payload_bytes_upper=copy_bytes[device])
    bank = sum(row["static_bank_bytes"] for row in placement["per_device"].values())
    role_bank = scalar*inv["totals"]["role_elements"]+sum(row["inert_buffer_bytes"] for row in placement["per_device"].values())
    checkpoint_weights = role_bank if spec.client != "libtorch" and spec.device != "cpu" else bank
    continuation = sum(v["private_state"]+v["pending_messages"]+v["history_int64"] for v in components.values())
    optimizer_bytes = sum(v["optimizer_slots"]+v["fp32_masters"] for v in components.values())
    checkpoint_estimate = checkpoint_weights+continuation+optimizer_bytes if checkpoint else 0
    checkpoint_limit = resources.get("checkpoint_max_bytes")
    if checkpoint_limit is not None:
        _bytes(checkpoint_limit, "resources.checkpoint_max_bytes")
        if checkpoint and checkpoint_weights > checkpoint_limit:
            rejections.append("checkpoint_max_bytes is below the known weight tensor bytes")
    materialize = bank*max(4, scalar)//scalar
    if resources.get("max_materialize_bytes") is not None:
        if materialize > _bytes(resources["max_materialize_bytes"], "resources.max_materialize_bytes"):
            rejections.append("max_materialize_bytes is below the FP32 construction/final bank floor")
    largest = max([row["elements"] for row in inv["owners"].values()]+[(plan["width"]+1)**2])
    if (spec.client == "libtorch" or resident) and checkpoint and largest > 2**28:
        rejections.append("native checkpoint has a tensor exceeding its 2^28 element codec limit")
    host_reserve = _bytes(resources.get("host_reserve_bytes", 8*GIB), "resources.host_reserve_bytes")
    disk_reserve = _bytes(resources.get("disk_reserve_bytes", GIB), "resources.disk_reserve_bytes")
    host = (4*max(materialize, checkpoint_weights)+2*optimizer_bytes+continuation+
            sum(v["complete_record_estimate"] for v in components.values())*host_results//results+host_reserve)
    if spec.finite_check_packing == "packed":
        host += 32*1024**2
    if "cpu" in per_device:
        host = max(host, per_device["cpu"]["estimated_peak_bytes"]+2*materialize+host_reserve)
    disk = checkpoint_generations*checkpoint_estimate+disk_reserve+(4*inv["totals"]["unique"] if spec.client == "libtorch" else 0)
    medium = resources.get("storage_medium", "disk")
    if medium not in {"disk", "tmpfs"}:
        raise ValueError("resources.storage_medium: choose disk or tmpfs")
    # Existing tmpfs files are already charged to live host free memory. Add
    # only the future files; disk reserve is unallocated headroom, not RAM.
    tmpfs_bytes = disk-disk_reserve if medium == "tmpfs" else 0
    host += tmpfs_bytes
    body = dict(schema="tide-experiment-capacity-v1", automatic_admission=False, workload_id=plan["sha256"],
        execution_id=spec.record()["execution_id"], placement_id=placement["placement_id"], mode=mode,
        optimizer=optimizer, checkpoint=checkpoint, per_device=per_device,
        resident_layout=resident_layout,
        effective_owner_contract=dict(nodes=placement["node_devices"],
            regions=[spec.device]*g["regions"] if resident else placement["region_devices"],
            region_scope="resident region banks/control" if resident else "eager region payload; temporary Read/control follow mixed policy",
            optimizer=spec.device if resident and train else "canonical parameter owners" if train else None),
        largest_indivisible_group=placement["largest_indivisible_group"],
        finite_scope=dict(logical_windows=windows, live_results=results, host_results=host_results, retained_windows=retained,
                          candidate_count_method=count_method, candidate_upper=batch*sum(totals),
                          node_cache_rows=[row["cache_rows"] for row in states]),
        host=dict(estimated_tensor_reservation_bytes=host, reserve_bytes=host_reserve,
                  construction_bank_floor_bytes=materialize, free_bytes=resources.get("host_free_bytes")),
        disk=dict(estimated_new_bytes=disk, checkpoint_generations=checkpoint_generations if checkpoint else 0, reserve_bytes=disk_reserve, free_bytes=resources.get("disk_free_bytes")),
        checkpoint_policy=dict(weight_tensor_bytes=checkpoint_weights, estimated_file_tensor_bytes=checkpoint_estimate,
            max_file_bytes=checkpoint_limit, largest_tensor_elements=largest,
            restoration="CPU-decoded weights copied into existing bank; device continuation staged separately" if python_inference else
                "reserve full staged bank/state; Python training also reserves optimizer deepcopy preflight"),
        communication=dict(requested=spec.communication, effective=None,
            status="requires actual runtime evidence", counted="forward physical message payload upper only; excludes VJP, descriptors and transport overhead"),
        resident_peer_protocol="runtime capability gate required; ordinary request is not a peer fallback" if resident else None,
        static_rejections=rejections, status="static-refusal" if rejections else "requires-measurement",
        assumptions=["all possible candidates send; budget and measured activity are distinct",
            "empty factory initial state; imported checkpoint history/cache must be included in the declared finite scope",
            "one SGD momentum slot or two AdamW slots reserved for every trainable canonical leaf",
            "private states include int64 timestamps; no cache is truncated",
            "complete internal records reserved even for requested outputs/metrics",
            "checkpoint staging follows the selected client; remaining counted partitions are conservatively simultaneous"],
        unknown=["allocator/context and packing peaks", "regional Attention temporary matrices and vendor workspace",
            "resident physical role-bank replicas and service/SQ/CQ/task resources" if resident else "transient parameter/scoring copies",
            "training autograd/retained graph and journal peaks" if train else "runtime object overhead"],
        required_before_run=["probe current per-device free memory, host and disk; do not sum free HBM",
            "calibrate construction/window/backward/restore peaks with the selected exact client and map",
            "verify actual owner maps and communication, then enforce timeout and resource reserves"])
    for name, estimate in (("host", host), ("disk", disk)):
        free = resources.get(name+"_free_bytes")
        if free is not None and estimate > _bytes(free, "resources."+name+"_free_bytes"):
            body[name]["reservation_exceeds_available"] = True
    if medium == "tmpfs":
        body["host"]["future_tmpfs_file_bytes"] = tmpfs_bytes
        body["host"]["tmpfs_accounting"] = "added to tensor reservation; existing files already reduce live host free; no extra physical memory"
    return dict(body, capacity_id=fingerprint(body))
