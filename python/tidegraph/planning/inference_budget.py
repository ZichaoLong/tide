"""Static reservations for the bounded FC/KV inference scale study.

Reuse the canonical inventory/lower bound; no tensors or execution trajectory.
This does not admit a run: operator/allocator peaks and current free resources
must still be measured. Checkpoint policies here describe the current clients.
"""
from .input_pattern import coordinates
from .workload import memory_floor

GIB = 1024**3


def fc_inference_budget(record, *, device_capacity_bytes=24*GIB, device_reserve_bytes=4*GIB):
    """Budget the explicit two-region, four-node baseline FC/KV experiment.

The return reservation overcounts shared snapshots and assumes every candidate
is active. It is a tensor envelope, not a proof of Python/C++ object overhead or
vendor workspace. Other graphs/memories/layouts need their own derived bounds.
    """
    p, inv, e, m = record["plan"], record["inventory"], record["execution"]["requested"], record["measurement"]
    g = p["graph"]
    recipe = (p.get("recipe") or {}).get("requested", {})
    if (g["kind"] != "ranked-local" or g["node_regions"] != [0, 0, 1, 1]
            or g["ranks"] != [1, 2] or g["inputs"] != [0, 1] or g["outputs"] != [2, 3]
            or g["edges"] != [[0, 2, 1], [0, 3, 1], [1, 2, 1], [1, 3, 1]]
            or any(c.get("memory_family", "kv") != "kv" or c["memory_read"] != "sum" for c in p["memories"])
            or recipe.get("profile", "baseline") != "baseline"
            or set(recipe)-{"topology", "family", "width", "region_width", "depth", "profile", "history"}):
        raise ValueError("inference budget requires the declared four-node baseline FC/KV graph")
    if (m["mode"] != "inference" or not m["checkpoint_check"] or m["warmup"] != 1
            or e["placement"] not in {"native", "mixed-c"} or e["peer_devices"]
            or e["dtype"] not in {"float32", "float64"}
            or e["scoring_dtype"] != e["dtype"] or e["workers"] != 1):
        raise ValueError("budget requires same-dtype single-owner eager inference and checkpoint check")
    if not (e["device"] == "cpu" or e["device"].startswith("cuda:")):
        raise ValueError("budget covers CPU and single-card CUDA clients")
    for value in (device_capacity_bytes, device_reserve_bytes):
        if type(value) is not int or value < 1:
            raise ValueError("resource reservations must be positive integer bytes")
    if device_reserve_bytes >= device_capacity_bytes:
        raise ValueError("device reserve must leave usable capacity")
    scalar = 8 if e["dtype"] == "float64" else 4
    floor = memory_floor(p, inv, dtype=e["dtype"])
    parameters, inert = floor["logical_parameter_bytes"], floor["legacy_inert_buffer_bytes"]
    bank = parameters+inert
    role_parameters = inv["totals"]["role_elements"]*scalar
    d, b, positions, n = p["width"], p["samples"], p["positions"], g["nodes"]
    # No path can produce a second event at the same node/logical position in
    # this aligned DAG. Do not predict which candidates are actually selected.
    candidates = n*b*positions
    logical_windows = m["warmup"]+m["steps"]+1  # Includes the untimed resume window.
    cache_rows = logical_windows*positions  # Ignore eviction for a safe tensor reservation.
    state_rows = []
    for c in p["memories"]:
        state_rows.append(((d+1)+cache_rows*(2*d+c["summary_width"]+1))*scalar+cache_rows*8+3*8)
    payload = (d+1)*scalar
    messages = len(g["edges"])*b*positions*(payload+6*8)
    outputs = len(g["outputs"])*b*positions*(payload+3*8)
    continuation = b*sum(state_rows)+messages  # Reserve a full message set even if pending is empty.
    # C++ complete returns include old/proposed/comparison/next state, sources,
    # prepared rows, physical fiber/emissions and descriptors. Python has fewer
    # duplicate fields. Charge four states and 32 payload-sized fields per event
    # for both; scalar/int metadata are separate from payload dtype.
    event = 4*max(state_rows)+32*payload+256*scalar+64*8
    one_return = candidates*event+continuation+messages+outputs
    live_returns = m["warmup"]+m["steps"]+2  # Saved measured windows plus two resume/reference results.
    cpu_returns = live_returns*one_return+2*continuation
    cpp = e["client"] == "libtorch"
    cuda = e["device"].startswith("cuda:")
    # Python CUDA save calls .cpu() independently for each state_dict role.
    # CPU save preserves shared storage; the C++ codec writes canonical owners.
    checkpoint_weights = role_parameters+inert if cuda and not cpp else bank
    checkpoint_limit = ((checkpoint_weights+continuation+128*1024**2+GIB-1)//GIB)*GIB
    physical_inputs = list(coordinates(p))
    roots = len({(sample, position) for sample, position, _, _ in physical_inputs})
    fixture = 4*inv["totals"]["unique"]+roots*(d+1)*4 if cpp else 0  # Zero codec omits inert buffers.
    restore_banks = 2 if cpp else 1  # GraphCheckpoint::load stages all new device owners.
    restore_models = restore_banks*bank
    largest = max([row["elements"] for row in inv["owners"].values()]+[(d+1)**2])
    return dict(schema="tide-core5-inference-budget-v1", automatic_admission=False,
        model_id=p["model_id"], workload_id=p["sha256"], execution_id=record["execution"]["execution_id"],
        counts=dict(trainable_parameters=inv["totals"]["trainable"], frozen_parameters=inv["totals"]["frozen"],
                    unique_parameters=inv["totals"]["unique"], owners=len(inv["owners"]),
                    roles=len(inv["roles"]), samples=b, positions=positions, span=p["span"],
                    input_atoms=len(physical_inputs), input_roots=roots,
                    candidate_upper_per_window=candidates, cache_rows_reserved=cache_rows),
        static_bytes=dict(parameters=parameters, legacy_inert_buffers=inert, one_model_bank=bank,
                          role_expanded_parameters=role_parameters, inference_optimizer_grad_master=0),
        return_tensor_reservation=dict(one_complete_window=one_return, simultaneous_results=live_returns,
                                       cpu_all_results=cpu_returns, continuation=continuation),
        checkpoint=dict(weight_tensor_bytes=checkpoint_weights, suggested_max_file_bytes=checkpoint_limit,
                        device_model_banks_during_restore=restore_banks if cuda else 0,
                        device_model_bytes_during_restore=restore_models if cuda else 0,
                        largest_tensor_elements=largest, native_tensor_limit_elements=2**28,
                        native_tensor_limit_exceeded=largest > 2**28),
        resource_reservation=dict(device_capacity=device_capacity_bytes, device_reserve=device_reserve_bytes,
            cuda_after_restore_models_and_return=(device_capacity_bytes-device_reserve_bytes-restore_models-one_return) if cuda else None,
            host=((6*bank) if cpp else 4*max(bank, checkpoint_weights))+cpu_returns+8*GIB,
            new_disk=fixture+checkpoint_limit+GIB, fixture_binary=fixture),
        excluded=["allocator fragmentation/cache and vendor context/workspace", "Python/C++ object overhead",
                  "existing artifacts and unrelated jobs", "concurrent owners or training",
                  "profile buffers; these budgets are for separate unprofiled measurement"],
        required_before_run=["related small-config correctness and exact resume",
            "near-boundary: calibrate construction/window/restore peaks from the admitted 1B configuration",
            "verify current GPU/host/disk free space and retain the device reserve",
            "negative CUDA remainder refuses the current client; positive is not admission"],
        required_for_qualification=["same-configuration independent complete records and exact resume",
            "one bounded measurement flow; measured costs are qualified only after its checks pass"])
