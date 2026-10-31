#!/usr/bin/env python3
"""Bounded eager inference replay: raw decode, layouts, then the exact next window.

Use the checkpoint's fixed installed Python/package and matching site. This
diagnostic never replaces a strict failure, changes layout, or runs a reference
trajectory. Without --checkpoint it creates one new public value checkpoint.
"""
import argparse
from pathlib import Path
import sys


def tensor_layout(value):
    row = dict(shape=list(value.shape), stride=list(value.stride()),
               storage_offset=value.storage_offset(), contiguous=value.is_contiguous(),
               dtype=str(value.dtype), device=str(value.device),
               pointer_alignment_mod512=value.data_ptr() % 512)
    if value.device.type == "npu":
        vendor = sys.modules.get("torch_npu")
        getter = getattr(vendor, "get_npu_format", None)
        row["npu_format"] = None if getter is None else int(getter(value))
    return row


def layouts(owner):
    # All state shapes/strides are cheap metadata; retain only three actual
    # samples, while recording total sample/state counts. No tensor values.
    states = {str(key): dict(value=tensor_layout(state.value),
                            slots={name: tensor_layout(v) for name, v in state.slots.items()})
              for key, state in sorted(owner.state.states.items()) if key[0] < 3}
    parameters = {name: tensor_layout(value) for name, value in owner.parameter_owner.named_parameters()}
    return dict(state_count=len(owner.state.states), batch=owner.state.batch_size,
                sampled_states=states, parameters=parameters)


def diagnose(config, output, *, checkpoint=None, windows=3):
    from tidegraph.experiment.config import resolve
    from tidegraph.experiment.records import write_json, now
    from tidegraph.experiment.identity import capture, verify
    from tidegraph.experiment.checkpoint import validate, save
    from tidegraph.experiment.engine import _configure
    from tidegraph.experiment.admission import preflight
    from tidegraph.experiment.eager import Eager
    from tidegraph.compare import equivalent
    from tidegraph.comparison import ComparisonMismatch
    resolved = resolve(config)
    if (resolved["measurement"]["mode"] != "inference" or
            resolved["execution"]["effective"]["placement"] == "resident" or
            resolved["execution"]["effective"]["client"] == "libtorch"):
        raise ValueError("diagnostic requires Python/native eager inference; other contracts need their own witness")
    if type(windows) is not int or windows < 1:
        raise ValueError("windows: specify a positive finite replay bound")
    identity = capture(resolved["site"]["paths"])
    if checkpoint is not None:
        checkpoint, metadata = validate(checkpoint, resolved, identity)
        if metadata["next_window"] > windows:
            raise ValueError("checkpoint exceeds --windows bound; declare its replay cost explicitly")
        windows = metadata["next_window"]
    if resolved["resources"]["logical_windows"] < windows+1:
        raise ValueError("resources.logical_windows must budget replay plus the next window")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = dict(schema="tide-resume-diagnostic-v1", state="running", phase="admission", started=now(),
                  identity=identity, resolved_config=resolved, replay_windows=windows,
                  scope="exact same-stack inference replay and first mismatch; layouts are observations, not a causal proof",
                  independent_reference="not_run", performance_qualification=False)
    def publish():
        write_json(output/"result.json", record)
    def phase(name):
        record["phase"] = name
        publish()
    owner = None
    try:
        record["runtime"] = _configure(resolved)
        record["admission"] = preflight(resolved, output)
        if record["admission"]["refused"]:
            raise ValueError("capacity preflight: "+"; ".join(record["admission"]["refused"]))
        phase("reconstruct-original")
        owner = Eager(resolved)
        for index in range(windows):
            record["window"] = index
            phase("replay")
            owner.step(index)
        record["original_layouts"] = layouts(owner)
        record["original_work"] = owner.last_work
        phase("snapshot-boundary")
        saved = owner.decoded_state()
        if checkpoint is None:
            phase("save-boundary")
            checkpoint, metadata = save(owner, output, resolved, identity, windows)
        record["checkpoint"] = dict(path=str(checkpoint), metadata=metadata)
        phase("original-next")
        actual, _ = owner.step(windows, return_mode="complete")
        record["original_next_work"] = owner.last_work
        owner.close()
        phase("restore")
        owner = Eager(resolved, checkpoint=checkpoint)
        record["restored_layouts"] = layouts(owner)
        phase("exact-raw-boundary")
        equivalent(saved, owner.decoded_state(), atol=0, rtol=0)
        record["raw_boundary_exact"] = True
        del saved
        phase("restored-next")
        expected, _ = owner.step(windows, return_mode="complete")
        record["restored_next_work"] = owner.last_work
        phase("exact-next-complete")
        equivalent(actual, expected, atol=0, rtol=0)
        record.update(state="passed", next_window_exact=True)
        verify(identity, resolved["site"]["paths"])
    except ComparisonMismatch as error:
        record.update(state="strict-failure", mismatch=error.record())
    except BaseException as error:
        record.update(state="failed", error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        if owner is not None:
            owner.close()
        record["finished"] = now()
        publish()
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", help="same resolved experiment/configuration as the witness")
    p.add_argument("--site")
    p.add_argument("--checkpoint", help="existing exact checkpoint; install and config identities must match")
    p.add_argument("--windows", type=int, required=True, help="maximum original windows to replay before one next window")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    from tidegraph.experiment.config import resolve
    record = diagnose(resolve(args.config, site=args.site), args.output_dir,
                      checkpoint=args.checkpoint, windows=args.windows)
    print(record["state"], record["phase"])
    return 0 if record["state"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
