"""Portable CPU record for explicit resident VJP/optimizer continuation."""
import torch
from .checkpoint_values import encode, decode
from .coordinates import integers
from .native_records import from_continuation, to_continuation
from .resident_training_options import GROUP_FIELDS, STATE_FIELDS, optimizer_groups


def export(runtime, checkpoint):
    c = checkpoint
    return dict(schema="tide-resident-training-v1", native_schema=c.schema,
                generation=c.generation, next_token=c.next_token,
                continuation=encode(from_continuation(runtime.execution_graph, c.continuation)),
                parameters=c.parameters, aliases=c.aliases, trainable=c.trainable,
                optimizer="sgd" if c.optimizer == runtime.engine.module.OptimizerKind.sgd else "adamw",
                groups=[{name: getattr(g, name) for name in GROUP_FIELDS} for g in c.groups],
                offsets=c.offsets, state={name: getattr(c.state, name) for name in STATE_FIELDS})


def prepare(runtime, compiled, record, batch_size):
    if not isinstance(record, dict) or record.get("schema") != "tide-resident-training-v1":
        raise ValueError("resident training checkpoint schema mismatch")
    core, module = runtime.engine.core, runtime.engine.module
    try:
        integers("resident checkpoint progress", record["native_schema"], record["generation"], record["next_token"])
        if record["native_schema"] != 1 or min(record["generation"], record["next_token"]) < 0:
            raise ValueError("invalid resident checkpoint progress")
        if record["optimizer"] not in {"sgd", "adamw"}:
            raise ValueError("invalid resident checkpoint optimizer")
        if (not isinstance(record["groups"], list) or not record["groups"]
                or any(not isinstance(g, dict) or set(g) != set(GROUP_FIELDS) for g in record["groups"])):
            raise ValueError("invalid resident checkpoint optimizer groups")
        for name in ("trainable", "offsets", "aliases"):
            if not isinstance(record[name], list):
                raise ValueError("invalid resident checkpoint owner layout")
        integers("resident checkpoint offsets", *record["offsets"])
        if any(x < -1 for x in record["offsets"]) or any(not isinstance(x, str) for x in record["trainable"]):
            raise ValueError("invalid resident checkpoint owner layout")
        if any(not isinstance(a, list) or not a or any(not isinstance(x, str) for x in a) for a in record["aliases"]):
            raise ValueError("invalid resident checkpoint aliases")
        if (not isinstance(record["parameters"], dict)
                or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in record["parameters"].items())
                or not isinstance(record["state"], dict) or set(record["state"]) != set(STATE_FIELDS)
                or any(not isinstance(v, torch.Tensor) for v in record["state"].values())):
            raise ValueError("invalid resident checkpoint tensor records")
        q = decode(record["continuation"], device="cpu")
        if q.batch_size != batch_size:
            raise ValueError("checkpoint batch size mismatch")
        if runtime.spec and (q.cut % runtime.spec.stride or q.pending):
            raise ValueError("Settle training checkpoint requires a complete position boundary")
        native = module.TrainingCheckpoint()
        native.schema = record["native_schema"]
        native.generation, native.next_token = record["generation"], record["next_token"]
        native.continuation = to_continuation(core, runtime.execution_graph, compiled, q)
        for name in ("parameters", "aliases", "trainable", "offsets"):
            setattr(native, name, record[name])
        native.optimizer = getattr(module.OptimizerKind, record["optimizer"])
        native.groups = optimizer_groups(core, record["groups"])
        state = module.DeviceOptimizerState()
        for name in STATE_FIELDS:
            setattr(state, name, record["state"][name])
        native.state = state
        return native
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("malformed resident training checkpoint") from error
