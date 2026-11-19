"""Four joint Transformer matches, never an implicit single-axis comparison.

The block count is fixed first. Every candidate has real attention and SwiGLU
weights: d is a multiple of 16, FFN width a multiple of 8 between 2d and 6d.
Head width is 16/32/64; it does not change parameter count. The
finite search minimizes relative parameter error, then distance from FFN=4d.
"""
import math


def transformer_parameters(d, layers, ffn_width, final_norm=True):
    return layers * (4*d*d + 3*d*ffn_width + 2*d) + (d if final_norm else 0)


def match_transformer(blocks, parameters, *, window=128):
    if type(blocks) is not int or blocks < 1 or not math.isfinite(parameters) or parameters <= 0:
        raise ValueError("positive block count and finite positive parameter target required")
    # At FFN=2d the quadratic coefficient is ten. Beyond this bound even
    # the smallest FFN cannot improve on the last candidate below the root.
    maximum = max(32, 16 * math.ceil(math.sqrt(parameters / (10*blocks)) / 16) + 16)
    candidates = []
    for d in range(32, maximum+1, 16):
        optimum = (parameters - blocks*(4*d*d+2*d) - d) / (3*blocks*d)
        widths = {2*d, 4*d, 6*d}
        widths.update(8*math.floor(optimum/8)+delta for delta in (0, 8))
        for width in widths:
            if not 2*d <= width <= 6*d:
                continue
            count = transformer_parameters(d, blocks, width)
            candidates.append((abs(count-parameters)/parameters, abs(width/d-4), d, width, count))
    _, _, d, width, count = min(candidates)
    return dict(constructor=dict(d=d, layers=blocks, ffn_width=width,
                                 head_dim=64 if d >= 256 and d % 64 == 0 else (32 if d % 32 == 0 else 16),
                                 window=window),
                targets=dict(blocks=blocks, parameters=parameters),
                actual=dict(blocks=blocks, parameters=count),
                relative_deviation=dict(blocks=0.0, parameters=(count-parameters)/parameters),
                selection_rule="exact block count; nearest parameters over d=16*j >= 32, FFN=8*j in [2d,6d]; ties prefer FFN=4d",
                parameter_scope="main blocks and final RMSNorm; no embedding, task head or artificial unused weights")


def match_transformers(nodes, depth, total_parameters, active_parameters, *, window=128):
    """Return A/B/C/D with targets, deviations and constructor keyword args.

    ``active_parameters`` is a measured value at the declared frozen workload,
    not k/n or the textbook proxy. Timed callers retain the tick/input policy
    with the audit record and must not rematch after optimizer steps.
    """
    targets = {"A": (nodes, total_parameters, "nodes", "total"),
               "B": (nodes, active_parameters, "nodes", "effective_touched"),
               "C": (depth, total_parameters, "depth", "total"),
               "D": (depth, active_parameters, "depth", "effective_touched")}
    return {name: dict(match_transformer(blocks, parameters, window=window),
                       joint_class=name, block_target_kind=block_kind,
                       parameter_target_kind=parameter_kind)
            for name, (blocks, parameters, block_kind, parameter_kind) in targets.items()}


def parameter_memory_budget(parameters, *, mode, precision, optimizer_foreach=True):
    """Lower bound plus explicitly labelled optimizer scratch allowance.

    The delivered BF16 autocast policy has FP32 parameters, gradients and
    AdamW slots. Activation/cache/workspace memory is workload dependent and
    intentionally not invented by this parameter-only calculator.
    """
    if mode not in {"inference", "grad_forward", "train"} or precision not in {"fp32", "bf16"}:
        raise ValueError("unknown mode or precision")
    training = mode == "train"
    components = dict(parameters=4*parameters, gradients=4*parameters if training else 0,
                      adam_first_second_moments=8*parameters if training else 0,
                      separate_master_weights=0)
    return dict(precision=precision, mode=mode, bytes=components,
                persistent_parameter_floor_bytes=sum(components.values()),
                foreach_scratch_allowance_bytes=4*parameters if training and optimizer_foreach else 0,
                excludes=["activations and autograd", "private KV/SSD state", "allocator and operator workspace",
                          "input and output tensors", "checkpoint copy and serialization"])
