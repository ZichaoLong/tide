"""Write a small, editable experiment JSON; no Torch import or model allocation."""
import argparse
import json
import os
from pathlib import Path
from tidegraph.experiment import resolve


def configuration(*, device, mode="inference", client="pytorch", task="regression",
                  samples=2, positions=3, dtype="float32", schedule="prefill", model=None):
    if task == "regression" and client == "libtorch":
        raise ValueError("execution.client: Python regression plugins do not execute in standalone LibTorch; use --task base or a separately built C++ task")
    if task == "regression" and dtype == "float16":
        raise ValueError("execution.dtype: this downstream regression recipe declares FP32/FP64; FP16 requires a new task recipe")
    result = dict(schema="tide-experiment-v1", name=f"consumer-{device.replace(':', '-')}-{mode}",
        description="Small downstream example; qualification comes from actual check/run results, not this template",
        model=model or dict(recipe=dict(topology="fc", family="kv", width=8, depth=2, region_width=2, history=4)),
        task=dict(samples=samples, positions=positions, seed=7, input_pattern="staggered-v1",
                  objective=dict(kind="weighted-output-square", scale=.001),
                  optimizer=dict(kind="sgd", lr=.0001, momentum=.3, weight_decay=.01)),
        execution=dict(family="settle", client=client, device=device,
            placement="cpu" if device == "cpu" else "mixed-c", dtype=dtype, scoring_dtype=dtype,
            schedule=schedule, packing="packed", return_packing="scalar", return_mode="complete",
            workers=1, intraop_threads=1, interop_threads=1),
        measurement=dict(mode=mode, steps=2, warmup=0, validation="strict", checkpoint_check=True, timeout_seconds=600),
        resources=dict(max_materialize_bytes=64*1024**2, checkpoint_max_bytes=64*1024**2))
    if task == "regression":
        result["task"]["plugin"] = dict(module="tidegraph_consumer_core5.task", factory="create_task",
            options=dict(data_seed=8, head_seed=9, input_scale=.2, target_scale=.05, loss_scale=.001))
    resolve(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, help="Explicit cpu, cuda:N or npu:N; select an actually idle logical device")
    parser.add_argument("--mode", choices=("inference", "training"), default="inference")
    parser.add_argument("--client", choices=("pytorch", "python-native", "libtorch"), default="pytorch")
    parser.add_argument("--task", choices=("base", "regression"), default="regression",
                        help="base uses the foundation diagnostic objective; regression adds downstream data/head/loss")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--positions", type=int, default=3)
    parser.add_argument("--dtype", choices=("float16", "float32", "float64"), default="float32")
    parser.add_argument("--schedule", choices=("streaming", "prefill"), default="prefill")
    parser.add_argument("--model", type=Path, help="Optional public model object JSON (recipe, definition or resolved); no benchmark whitelist")
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; refuses to overwrite")
    args = parser.parse_args()
    model = None if args.model is None else json.loads(args.model.read_text())
    values = {key: getattr(args, key) for key in ("device", "mode", "client", "task", "samples", "positions", "dtype", "schedule")}
    try:
        record = configuration(**values, model=model)
    except (TypeError, ValueError, KeyError) as error:
        parser.error(str(error))
    data = json.dumps(record, indent=2, allow_nan=False)+"\n"
    with args.output.open("x") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    print(args.output)


if __name__ == "__main__":
    main()
