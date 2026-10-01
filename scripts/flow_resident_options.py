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
    for field in FORWARD+TRAINING:
        parser.add_argument("--resident-"+field.replace("_", "-"), type=int)


def validate(args):
    if not 1 <= args.devices <= 16:
        raise ValueError("devices must be in 1..16")
    changed = (args.resident_library is not None or args.devices != 1 or args.owner_policy != "locality"
               or args.chunk_policy != "conservative" or any(getattr(args,"resident_"+k) is not None for k in FORWARD+TRAINING))
    if args.preset != "resident" and changed:
        raise ValueError("resident capacities and placement require resident preset")
    if args.implementation == "libtorch" and args.resident_library is not None:
        raise ValueError("standalone LibTorch uses its linked TideResident package")


def native_arguments(args):
    values = ["--devices="+str(args.devices), "--owner-policy="+args.owner_policy, "--chunk-policy="+args.chunk_policy]
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
                training_limits=ResidentTrainingLimits(**training),
                resident_placement=ResidentPlacement(devices=devices,policy=args.owner_policy))
