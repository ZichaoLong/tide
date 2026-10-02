"""Common CLI spelling for optional public resident consumer limits."""
FORWARD = ("queue", "arrivals", "outputs", "trace", "stages", "workspace_bytes", "full_chunk_rows",
           "emission_chunk_rows", "aggregate_chunk_rows", "attention_chunk_rows", "attention_key_rows",
           "kv_rows", "kv_trace_rows", "max_repeat_ticks")
TRAINING = ("retained_bytes", "backward_bytes", "optimizer_bytes", "program_workspace_bytes", "reverse_chunk_rows")


def add_arguments(parser):
    from pathlib import Path
    parser.add_argument("--resident-library", type=Path)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--owner-policy", choices=("locality", "memory"), default="locality")
    parser.add_argument("--chunk-policy", choices=("conservative", "aggressive"), default="conservative")
    parser.add_argument("--head-workspace-bytes", type=int, default=4*1024**3)
    parser.add_argument("--device-memory-bytes", type=int, default=0,
                        help="resident per-device incremental HBM cap; 0 uses current driver free memory")
    parser.add_argument("--resident-context-bytes", type=int, default=0,
                        help="Per-device saved continuation pool; positive enables compact rows, 0 keeps dense storage")
    for field in FORWARD+TRAINING:
        parser.add_argument("--resident-"+field.replace("_", "-"), type=int)


def validate(args):
    if not 1 <= args.devices <= 16:
        raise ValueError("devices must be in 1..16")
    changed = (args.resident_library is not None or args.devices != 1 or args.owner_policy != "locality"
               or args.chunk_policy != "conservative" or args.head_workspace_bytes != 4*1024**3 or args.device_memory_bytes != 0 or args.resident_context_bytes != 0
               or any(getattr(args,"resident_"+k) is not None for k in FORWARD+TRAINING))
    if args.preset != "resident" and changed:
        raise ValueError("resident capacities and placement require resident preset")
    if args.implementation == "libtorch" and args.resident_library is not None:
        raise ValueError("standalone LibTorch uses its linked TideResident package")
    if not 0 <= args.resident_context_bytes < 2**63:
        raise ValueError("resident-context-bytes must be a nonnegative int64")
    if not 0 <= args.device_memory_bytes < 2**63:
        raise ValueError("device-memory-bytes must be a nonnegative int64")


def native_arguments(args):
    values = ["--devices="+str(args.devices), "--owner-policy="+args.owner_policy, "--chunk-policy="+args.chunk_policy,
              "--head-workspace-bytes="+str(args.head_workspace_bytes),"--device-memory-bytes="+str(args.device_memory_bytes)]
    if args.resident_context_bytes:
        values.append("--resident-context-bytes="+str(args.resident_context_bytes))
    for field in FORWARD+TRAINING:
        value = getattr(args,"resident_"+field)
        if value is not None:
            values.append("--resident-"+field.replace("_", "-")+"="+str(value))
    return values


def python_arguments(args, device):
    if args.preset != "resident":
        return {}
    from tidegraph import ResidentLimits, ResidentTrainingLimits, ResidentPlacement
    forward = dict(workspace_bytes=512*1024**2, chunk_policy=args.chunk_policy)
    training = dict(windows=args.windows_per_step, backward_bytes=2*1024**3)
    for fields,values in ((FORWARD,forward),(TRAINING,training)):
        for field in fields:
            value=getattr(args,"resident_"+field)
            if value is not None:
                values[field]=value
    devices = tuple(f"npu:{device.index+i}" for i in range(args.devices)) if args.devices>1 else ()
    return dict(resident_library=args.resident_library, resident_limits=ResidentLimits(**forward),
                head_workspace_bytes=args.head_workspace_bytes,
                device_memory_bytes=args.device_memory_bytes,
                training_limits=ResidentTrainingLimits(**training),
                resident_placement=ResidentPlacement(devices=devices,policy=args.owner_policy))
