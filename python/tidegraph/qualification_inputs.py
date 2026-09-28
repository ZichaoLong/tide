"""Explicit finite workloads for configuration qualification, never training data."""
from dataclasses import asdict, replace
import hashlib
import json
import torch
from .records import External


class Probe:
    def __init__(self, config, *, inputs=None, batch_size=2, positions=4, stop=None, seed=19):
        for name, value in (("batch_size", batch_size), ("positions", positions)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("input seed must be a nonnegative int64")
        self.family, self.batch_size = config.family, batch_size
        self.generated, self.seed = inputs is None, seed
        dtype = getattr(torch, config.dtype)
        generator = torch.Generator().manual_seed(seed)
        if self.family == "settle":
            self.values = (torch.randn(batch_size, positions, config.width, generator=generator, dtype=dtype)
                           if inputs is None else inputs.detach().clone())
            if self.values.ndim != 3 or self.values.shape != (batch_size, positions, config.width):
                raise ValueError("Settle probe dimensions must match batch/positions/width")
            self.records = None
            self.stop = positions if stop is None else stop
            if self.stop != positions:
                raise ValueError("Settle stop must equal positions")
        else:
            self.values = None
            if inputs is None:
                self.records = [External(b, p, t, t, torch.randn(config.width, generator=generator, dtype=dtype))
                                for b in range(batch_size) for p in range(len(config.graph.inputs)) for t in range(positions)]
            else:
                self.records = [replace(x, value=x.value.detach().clone()) for x in inputs]
            if stop is None and inputs is not None:
                raise ValueError("external probes require an explicit final logical stop")
            if stop is None:
                distance = [0] * len(config.graph.nodes)
                if self.family == "timed-dag":
                    offsets, edges = config.graph.adjacency()
                    for node in config.graph.topological_order():
                        for i in edges[offsets[node]:offsets[node+1]]:
                            edge = config.graph.edges[i]
                            distance[edge.target] = max(distance[edge.target], distance[node] + edge.delay)
                else:
                    distance = [4 * max((e.delay for e in config.graph.edges), default=1)]
                stop = positions + max(distance)
            self.stop = stop
        if type(self.stop) is not int or self.stop < 2:
            raise ValueError("qualification requires at least two positions/logical ticks")
        tensors = [self.values] if self.values is not None else [x.value for x in self.records]
        if not tensors:
            raise ValueError("qualification requires external inputs")
        if any(x.device.type != "cpu" or x.dtype != dtype or not torch.isfinite(x).all() for x in tensors):
            raise ValueError("qualification inputs must be finite CPU tensors of configured dtype")

    def clone(self):
        probe = object.__new__(Probe)
        probe.__dict__ = self.__dict__.copy()
        probe.values = None if self.values is None else self.values.detach().clone().requires_grad_()
        probe.records = None if self.records is None else [replace(x, value=x.value.detach().clone().requires_grad_()) for x in self.records]
        return probe

    def leaves(self):
        return {"input":self.values} if self.values is not None else {f"input.{i}":x.value for i,x in enumerate(self.records)}

    def advance(self, session, start=0, end=None, *, cycle=0):
        end = self.stop if end is None else end
        if self.values is not None:
            return session.advance(self.values[:, start:end])
        counts = {}
        for x in self.records:
            key = x.batch, x.port
            counts[key] = max(counts.get(key, 0), x.position + 1)
        records = [replace(x, position=x.position + cycle * counts[x.batch, x.port], time=x.time + cycle * self.stop)
                   for x in self.records if start <= x.time < end]
        return session.advance(records, stop=cycle*self.stop+end, sealed_until=cycle*self.stop+end)

    def payload(self):
        return dict(family=self.family, batch_size=self.batch_size, generated=self.generated, seed=self.seed,
                    stop=self.stop, values=self.values.detach() if self.values is not None else None,
                    records=None if self.records is None else [asdict(x) for x in self.records])

    @classmethod
    def restore(cls, record):
        probe = object.__new__(cls)
        probe.__dict__ = dict(record)
        if probe.records is not None:
            probe.records = [External(**x) for x in probe.records]
        return probe

    def manifest(self):
        metadata = dict(family=self.family, batch_size=self.batch_size, stop=self.stop,
                        source="deterministic-generated" if self.generated else "caller-supplied", seed=self.seed)
        values = [self.values] if self.values is not None else [x.value for x in self.records]
        if self.records is not None:
            metadata["coordinates"] = [(x.batch,x.port,x.position,x.time) for x in self.records]
        metadata["shapes"] = [list(x.shape) for x in values]
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
        for value in values:
            digest.update(str(value.dtype).encode() + value.detach().contiguous().numpy().tobytes())
        return metadata | {"sha256":digest.hexdigest()}
