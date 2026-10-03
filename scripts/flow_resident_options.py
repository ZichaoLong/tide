"""Common ownership options and optional public resident consumer limits."""
FORWARD = ("queue", "arrivals", "outputs", "trace", "stages", "workspace_bytes", "full_chunk_rows",
           "emission_chunk_rows", "aggregate_chunk_rows", "attention_chunk_rows", "attention_key_rows",
           "kv_rows", "kv_trace_rows", "max_repeat_ticks")
TRAINING = ("retained_bytes", "backward_bytes", "optimizer_bytes", "program_workspace_bytes", "reverse_chunk_rows")


def owner_map(value):
    import argparse
    parts=value.split(',')
    if any(not x or any(c not in '0123456789' for c in x) or len(x)>2 or int(x)>15 for x in parts):
        raise argparse.ArgumentTypeError('owner-map needs comma-separated logical device indices 0..15')
    return tuple(int(x) for x in parts)


def add_arguments(parser):
    from pathlib import Path
    parser.add_argument("--resident-library", type=Path)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--owner-policy", choices=("locality", "memory"), default="locality")
    parser.add_argument("--owner-map", type=owner_map, default=(),
                        help="fixed node Full/state owners in encoded order including boundaries; eager boundaries and node zero use owner zero")
    parser.add_argument("--chunk-policy", choices=("conservative", "aggressive"), default="conservative")
    parser.add_argument("--head-workspace-bytes", type=int, default=4*1024**3)
    parser.add_argument("--device-memory-bytes", type=int, default=0,
                        help="per-device incremental memory cap; 0 uses driver free memory or half available CPU RAM")
    parser.add_argument("--resident-context-bytes", type=int, default=0,
                        help="Per-device saved continuation pool; positive enables compact rows, 0 keeps dense storage")
    parser.add_argument("--auto-sample-chunks", action="store_true",
                        help="halve physical samples on static memory refusal before model allocation")
    for field in FORWARD+TRAINING:
        parser.add_argument("--resident-"+field.replace("_", "-"), type=int)


def validate(args):
    if not 1 <= args.devices <= 16:
        raise ValueError("devices must be in 1..16")
    changed = (args.resident_library is not None
               or args.resident_context_bytes != 0
               or any(getattr(args,"resident_"+k) is not None for k in FORWARD+TRAINING))
    if args.preset != "resident" and changed:
        raise ValueError("resident capacities require resident preset")
    if args.implementation == "libtorch" and args.resident_library is not None:
        raise ValueError("standalone LibTorch uses its linked TideResident package")
    if not 0 <= args.resident_context_bytes < 2**63:
        raise ValueError("resident-context-bytes must be a nonnegative int64")
    if not 0 <= args.device_memory_bytes < 2**63:
        raise ValueError("device-memory-bytes must be a nonnegative int64")
    if not 0 < args.head_workspace_bytes < 2**63:
        raise ValueError("head-workspace-bytes must be a positive int64")


def native_arguments(args):
    values = ["--devices="+str(args.devices), "--owner-policy="+args.owner_policy, "--chunk-policy="+args.chunk_policy,
              "--head-workspace-bytes="+str(args.head_workspace_bytes),"--device-memory-bytes="+str(args.device_memory_bytes)]
    if args.resident_context_bytes:
        values.append("--resident-context-bytes="+str(args.resident_context_bytes))
    if args.auto_sample_chunks:
        values.append("--auto-sample-chunks")
    if args.owner_map:
        values.append('--owner-map='+','.join(map(str,args.owner_map)))
    for field in FORWARD+TRAINING:
        value = getattr(args,"resident_"+field)
        if value is not None:
            values.append("--resident-"+field.replace("_", "-")+"="+str(value))
    return values


def python_arguments(args, device):
    if args.preset != "resident":
        return dict(devices=args.devices, owner_policy=args.owner_policy, owner_map=args.owner_map,
                    chunk_policy=args.chunk_policy, auto_sample_chunks=args.auto_sample_chunks,
                    device_memory_bytes=args.device_memory_bytes, head_workspace_bytes=args.head_workspace_bytes)
    from tidegraph import ResidentLimits, ResidentTrainingLimits, ResidentPlacement
    forward = dict(workspace_bytes=512*1024**2, chunk_policy=args.chunk_policy)
    training = dict(windows=args.windows_per_step, backward_bytes=2*1024**3)
    for fields,values in ((FORWARD,forward),(TRAINING,training)):
        for field in fields:
            value=getattr(args,"resident_"+field)
            if value is not None:
                values[field]=value
    devices = tuple(f"npu:{device.index+i}" for i in range(args.devices)) if args.devices>1 or args.owner_map else ()
    return dict(resident_library=args.resident_library, resident_limits=ResidentLimits(**forward),
                auto_sample_chunks=args.auto_sample_chunks,
                head_workspace_bytes=args.head_workspace_bytes,
                device_memory_bytes=args.device_memory_bytes,
                training_limits=ResidentTrainingLimits(**training),
                resident_placement=ResidentPlacement(devices=devices,policy=args.owner_policy,
                    full_owners=args.owner_map,state_owners=args.owner_map))
