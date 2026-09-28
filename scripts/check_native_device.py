#!/usr/bin/env python3
"""Native optimizer/checkpoint device ownership and explicit NPU capability errors."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import torch
from durable_records import write_json
from source_identity import source_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--native-library", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "python"))
    from tidegraph import GraphConfig, GraphRuntime
    from tidegraph.native_loader import load_native
    from tidegraph.runtime import resolve_device, synchronize, manifest
    from tidegraph.qualification_checks import compare_finite, assert_placement
    device, reason = resolve_device(args.device)
    core = load_native(args.native_library, backend=device.type)
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=False)
    source, dirty = source_state(root)
    record = dict(state="running", source=source, dirty=dirty,
                  runtime=manifest(device, reason, torch.float32), checks=[])
    write_json(out / "result.json", record)
    try:
        for kind in ("SGD", "AdamW"):
            def fixture(target):
                value = torch.tensor([.2, -.3, .7], device=target, requires_grad=True)
                registry = core.ParameterRegistry()
                registry.add("a", value); registry.add("alias", value)
                group = core.OptimizerGroup(); group.parameters = ["a"]
                group.lr = .01; group.weight_decay = .02; group.eps = 1e-5
                if kind == "SGD":
                    group.momentum = .8
                else:
                    group.amsgrad = True
                return value, registry, getattr(core, kind)(registry, [group])
            a, ar, ao = fixture(torch.device("cpu"))
            b, br, bo = fixture(device)
            for gradient in (None, [.1, -.5, .8], [0., 0., 0.]):
                for value, opt in ((a, ao), (b, bo)):
                    value.grad = None if gradient is None else torch.tensor(gradient, device=value.device)
                    opt.step()
                compare_finite(a, b)
            path = out / (kind + ".tide")
            core.save_checkpoint(str(path), br, bo, "owners-v1")
            for target in (torch.device("cpu"), device):
                value, registry, opt = fixture(target)
                core.load_checkpoint(str(path), registry, opt, "owners-v1")
                compare_finite(value, b)
                for name, state in opt.state().items():
                    for key in ("momentum_buffer", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        slot = getattr(state, key)
                        if slot is not None:
                            assert_placement(slot, target)
                            compare_finite(slot, getattr(bo.state()[name], key))
                value.grad = torch.ones_like(value)*.2
                opt.step()
                # Compare the resumed update with an independently restored CPU owner.
                ref, rr, ro = fixture(torch.device("cpu"))
                core.load_checkpoint(str(path), rr, ro, "owners-v1")
                ref.grad = torch.ones_like(ref)*.2; ro.step()
                compare_finite(ref, value)
            record["checks"].append(kind + "-aliases-None-zero-updates-device-checkpoint")
        if device.type == "npu":
            config = GraphConfig.from_dict(dict(schema_version=1, family="pdg", topology=dict(kind="self_loop", size=1)))
            for invalid in (replace(config, dtype="float64"), replace(config, graph=replace(config.graph,
                            nodes=tuple(replace(n, readout="norm-fp64-v1") for n in config.graph.nodes)))):
                try:
                    GraphRuntime(invalid, device=args.device)
                except ValueError:
                    pass
                else:
                    raise AssertionError("NPU accepted FP64 payload or FP64 Read")
            record["checks"].append("explicit-NPU-FP64-and-FP64-Read-rejection")
        synchronize(device)
        record["state"] = "passed"
    except BaseException as error:
        record.update(state="failed", error=repr(error)); raise
    finally:
        write_json(out / "result.json", record)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
