"""Direct token/phase readout anchor; no generic scheduler or state kernel calls."""
import math
import torch
from tidegraph import Continuation, Graph, Node, Region, State
from tidegraph.ops import Model

KINDS = ("add", "sum", "mean", "linear", "active-softmax", "all-softmax")


def fixture(dtype, kind, layers=3):
    memory = "lh-add-repeat-v1" if kind == "add" else f"lh-fiber-attention-{kind}-repeat-v1"
    g = Graph((Node(0, memory=memory, full="lh-identity-identity-v1"),), (), (Region(1),), (0,)*layers, (0,))
    m = Model(g, width=2, dtype=dtype); w = m.nodes[0]
    w.extra["token_head"] = torch.nn.Parameter(torch.arange(10, dtype=dtype).reshape(5, 2)/13-.2)
    w.extra["token_head_bias"] = torch.nn.Parameter(torch.arange(5, dtype=dtype)/30)
    with torch.no_grad():
        if kind != "add": w.extra["fiber_out_bias"].fill_(.5)
        if "fiber_pool" in w.extra: w.extra["fiber_pool"].copy_(torch.linspace(-.3, .4, layers, dtype=dtype))
        for p in m.input_scale: p.fill_(1)
    leaves = dict(m.named_parameters()); body = []
    for token in range(4):
        for b in range(2):
            if b == 1 and token in (0, 2): continue
            for phase in range(layers):
                if phase and (token+b+phase)%3 == 0: continue
                x = torch.tensor([.3+token/10+phase/5, -.2+b/4], dtype=dtype, requires_grad=True)
                if token == 3 and phase == 0 and b == 0: x = torch.zeros(2, dtype=dtype, requires_grad=True)
                body.append((b, token*layers+phase, 0, x)); leaves[f"input.{b}.{token}.{phase}"] = x
    return g, m, Continuation(g.identity, 3), body, leaves


def logits(model, outputs):
    w = model.nodes[0]
    # Each public logit row keeps its own graph. A future packed head needs a
    # separate connectivity contract, like packed state and Full execution.
    return {(t, b): torch.nn.functional.linear(x, w.extra["token_head"], w.extra["token_head_bias"])
            for b, t, _, x in outputs}


def direct(model, body, layers, kind, tokens=4, batch_size=3):
    w = model.nodes[0]; states, outputs = {}, {}
    for token in range(tokens):
        for b in range(batch_size):
            present = sorted((time%layers, x) for bb, time, _, x in body
                             if bb == b and time//layers == token)
            if not present: continue
            old = states.get((b, 0), State(w.bias.new_zeros(2)))
            rows = torch.stack([x*model.input_scale[p] for p, x in present])
            if kind == "add":
                value = old.value
                for _ in range(token-old.last_time): value = value*w.extra["add_retention"]
                value = value+rows.sum(0); slots = {}
            else:
                q, new_k, new_v = (rows@w.extra["fiber_qkv"]+w.extra["fiber_qkv_bias"]).chunk(3, -1)
                k = torch.cat((old.slots.get("key", rows.new_empty(0, 1, 2)).flatten(1), new_k))
                v = torch.cat((old.slots.get("value", rows.new_empty(0, 1, 2)).flatten(1), new_v))
                bias = old.slots.get("log_bias", rows.new_empty(0))
                if bias.numel():
                    for _ in range(token-old.last_time): bias = bias-w.extra["fiber_decay"]
                bias = torch.cat((bias, rows.new_zeros(len(rows))))
                attention = ((q/math.sqrt(2))@k.T+bias).softmax(-1)@v
                if kind == "sum": value = attention.sum(0)
                elif kind == "mean": value = attention.mean(0)
                else:
                    indices = [p for p, _ in present]; weight = w.extra["fiber_pool"]
                    coefficient = weight.softmax(0)[indices] if kind == "all-softmax" else weight[indices]
                    if kind == "active-softmax": coefficient = coefficient.softmax(0)
                    value = coefficient@attention
                value = value@w.extra["fiber_out"]+w.extra["fiber_out_bias"]
                slots = {"key": k.unsqueeze(1), "value": v.unsqueeze(1), "log_bias": bias}
            state = State(value, token, old.observations+1, slots); states[b, 0] = state
            normalized = value*model.output_scale[0]  # Actual Pronounce defaults to identity normalization.
            outputs[token, b] = normalized@w.extra["token_head"].T+w.extra["token_head_bias"]
    return outputs, states
