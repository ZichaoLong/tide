"""Decode actual resident SGD buffers into the eager benchmark's named roots.

No reference computation: static names/shapes/options come from the requested
model; values, connection bits, momentum and update counts come from the owner.
"""
import math
import torch


class SGDLayout:
    def __init__(self, model):
        self.shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}
        self.names = list(self.shapes)
        self.trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        canonical = {id(p): n for n, p in model.named_parameters()}
        self.aliases = {n: canonical[id(p)] for n, p in model.named_parameters(remove_duplicate=False)}
        self.template = torch.optim.SGD(model.parameters(), lr=.001, momentum=.3, weight_decay=.01).state_dict()

    def name(self, native):
        return self.aliases[native.replace(".extra.", ".")]

    def parameters(self, checkpoint):
        # Native checkpoints also retain inert legacy buffers. This projection
        # exposes registered Parameters under Python's canonical alias names;
        # the exact-resume check still compares the complete checkpoint.
        values = {self.name(n): x for n, x in checkpoint["parameters"].items()
                  if n.replace(".extra.", ".") in self.aliases}
        if values.keys() != self.shapes.keys():
            raise ValueError("benchmark checkpoint omits registered parameters")
        return values

    def span(self, offsets, index, name, storage):
        # Native offsets have one start per owner, with no trailing sentinel.
        a = offsets[index]
        b = a + math.prod(self.shapes[name])
        if a < 0 or b > storage.numel():
            raise ValueError("benchmark SGD owner slice exceeds actual storage")
        return a, b

    def decode(self, gradient, checkpoint):
        if checkpoint["optimizer"] != "sgd":
            raise ValueError("benchmark decoder requires requested SGD")
        groups = checkpoint["groups"]
        if len(groups) != 1 or {self.name(n) for n in checkpoint["trainable"]} != self.trainable:
            raise ValueError("benchmark SGD trainable owners differ")
        if {self.name(n) for n in groups[0]["parameters"]} != self.trainable:
            raise ValueError("benchmark SGD group owners differ")
        for name in ("lr", "momentum", "weight_decay", "dampening", "nesterov", "maximize"):
            if groups[0][name] != self.template["param_groups"][0][name]:
                raise ValueError("benchmark SGD option differs: "+name)
        values, connected = gradient.values.detach().cpu(), gradient.connected.cpu()
        gradients = dict.fromkeys(self.names)
        for i, name in enumerate(gradient.names):
            name = self.name(name)
            a, b = self.span(gradient.offsets, i, name, values)
            if connected[i]:gradients[name] = values[a:b].reshape(self.shapes[name]).clone()
        state, counts = {}, dict.fromkeys(self.names, 0)
        raw, offsets = checkpoint["state"], checkpoint["offsets"]
        # trainable contains every named alias; slots/offsets contain one row
        # per physical owner, ordered by the native canonical name.
        owners = sorted(set(checkpoint["trainable"]) & checkpoint["parameters"].keys())
        if len(owners) != len(offsets) or len(owners) != raw["steps"].numel():
            raise ValueError("benchmark SGD canonical owner layout differs")
        for unused in ("second", "maximum", "corrections"):
            if raw[unused].count_nonzero():
                raise ValueError("unused SGD storage is nonzero: "+unused)
        for i, native in enumerate(owners):
            name = self.name(native);a, b = self.span(offsets, i, name, raw["values"])
            count = int(raw["steps"][i]);counts[name] = count
            # Master values must be the actual published parameters.
            torch.testing.assert_close(raw["values"][a:b].reshape(self.shapes[name]),
                                       checkpoint["parameters"][native], rtol=0, atol=0)
            if count:
                state[self.names.index(name)] = {"momentum_buffer": raw["first"][a:b].reshape(self.shapes[name]).clone()}
            elif raw["first"][a:b].count_nonzero():
                raise ValueError("unupdated SGD owner has momentum")
        return dict(gradients=gradients, optimizer=dict(self.template, state=state), update_counts=counts)
