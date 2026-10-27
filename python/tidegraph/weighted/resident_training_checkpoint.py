"""Detached exact weighted cuts; architecture and core-5 finite-cut contract."""
import torch
from ..checkpoint_values import encode, decode
from ..coordinates import integers
from ..native_records import from_continuation, to_continuation
from ..resident_training_options import GROUP_FIELDS, STATE_FIELDS, optimizer_groups


def export(client, c):
    return dict(schema="tide-weighted-resident-training-v1", native_schema=c.schema,
                generation=c.generation, next_token=c.next_token,
                continuation=encode(from_continuation(client.graph, c.continuation)),
                parameters=c.parameters, aliases=c.aliases, trainable=c.trainable,
                optimizer="sgd" if c.optimizer == client.module.OptimizerKind.sgd else "adamw",
                groups=[{name: getattr(g, name) for name in GROUP_FIELDS} for g in c.groups],
                offsets=c.offsets, state={name: getattr(c.state, name) for name in STATE_FIELDS},
                mode=c.mode, zeta=c.zeta)


def prepare(client, record):
    if not isinstance(record, dict) or record.get("schema") != "tide-weighted-resident-training-v1":
        raise ValueError("weighted resident training checkpoint schema mismatch")
    try:
        if record["mode"] != "weighted" or type(record["zeta"]) not in (int, float) or record["zeta"] != 1:
            raise ValueError("weighted checkpoint local binding differs")
        integers("weighted checkpoint progress", record["native_schema"], record["generation"], record["next_token"])
        if record["native_schema"] != 1 or min(record["generation"], record["next_token"]) < 0:
            raise ValueError("invalid weighted checkpoint progress")
        if record["optimizer"] not in {"sgd", "adamw"}:
            raise ValueError("invalid weighted checkpoint optimizer")
        if (not isinstance(record["groups"], list) or not record["groups"]
                or any(not isinstance(g, dict) or set(g) != set(GROUP_FIELDS) for g in record["groups"])):
            raise ValueError("invalid weighted optimizer groups")
        for name in ("trainable", "offsets", "aliases"):
            if not isinstance(record[name], list):
                raise ValueError("invalid weighted owner layout")
        integers("weighted checkpoint offsets", *record["offsets"])
        if (any(x < 0 for x in record["offsets"])
                or any(not isinstance(x, str) for x in record["trainable"])
                or any(not isinstance(a, list) or not a or any(not isinstance(x, str) for x in a) for a in record["aliases"])):
            raise ValueError("invalid weighted owner names/offsets")
        if (not isinstance(record["parameters"], dict)
                or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in record["parameters"].items())
                or not isinstance(record["state"], dict) or set(record["state"]) != set(STATE_FIELDS)
                or any(not isinstance(v, torch.Tensor) for v in record["state"].values())):
            raise ValueError("invalid weighted checkpoint tensors")
        q = decode(record["continuation"], device="cpu")
        if q.batch_size != client.batch_size:
            raise ValueError("weighted checkpoint batch size differs")
        c = client.module.TrainingCheckpoint()
        c.schema, c.mode, c.zeta = record["native_schema"], "weighted", 1.
        c.generation, c.next_token = record["generation"], record["next_token"]
        c.continuation = to_continuation(client.core, client.graph, client.compiled, q)
        for name in ("parameters", "aliases", "trainable", "offsets"):
            setattr(c, name, record[name])
        c.optimizer = getattr(client.module.OptimizerKind, record["optimizer"])
        c.groups = optimizer_groups(client.core, record["groups"])
        state = client.module.DeviceOptimizerState()
        for name in STATE_FIELDS:
            setattr(state, name, record["state"][name])
        c.state = state
        return c
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("malformed weighted resident checkpoint") from error
