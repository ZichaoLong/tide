"""Torch-free parameter roles for pinned 0770593 weighted local functions.

These are allocation-free shape formulas, not a replacement for runtime
configuration validation. Runtime inventory parity tests use actual modules.
"""


def dim(config, name, default=None):
    value = config.get(name, default)
    if type(value) is not int or not 0 < value < 2**63:
        raise ValueError(f"{name} must be a positive int64")
    return value


def scalar_head(input_width, hidden):
    return dict(matrix=(hidden, input_width), hidden_bias=(hidden,),
                vector=(hidden,), bias=())


def encoder(input_width, hidden, output):
    return dict(in_weight=(hidden, input_width), hidden_bias=(hidden,),
                out_weight=(output, hidden), out_bias=(output,))


def prefixed(prefix, shapes):
    return {prefix + "." + name: shape for name, shape in shapes.items()}


def memory_shapes(c):
    """KV §1/3 and affine-delta/GDN/SSD §1–3, trainable roles only."""
    d, s = dim(c, "width", 4), dim(c, "summary_width", 2)
    family = c.get("memory_family", "kv")
    mode = c.get("memory_read") or ("state" if family in {"ssd", "gated_delta"} else "sum")
    out = {"summary": (s, d)}
    if family == "kv":
        r, heads = dim(c, "adapter_rank", 2), dim(c, "heads", 1)
        if d % heads or mode not in {"sum", "qry"}:
            raise ValueError("invalid KV layout/read")
        out.update(base=(2*d, d), adapter_u=(2*d, r), adapter_v=(r, d))
        if mode == "qry":
            out.update({name: (s, d) for name in ("read_q", "read_k", "read_v")})
        return out
    if family not in {"affine", "delta", "gated_delta", "ssd"} or mode not in {"sum", "state"}:
        raise ValueError("invalid recurrent family/read")
    k, v = dim(c, "key_width", 2), dim(c, "value_width", 3)
    h, g, size = dim(c, "heads", 1), dim(c, "groups", 1), dim(c, "conv_size", 1)
    if h % g or family != "ssd" and g != 1:
        raise ValueError("invalid recurrent groups")
    if family in {"affine", "delta"}:
        if (h, g, size, c.get("conv_stride", 1)) != (1, 1, 1, 1):
            raise ValueError("basic recurrence has no heads/convolution")
        out.update(key=(k, d), value=(v, d), retain=(d,), retain_bias=())
        if family == "delta":
            out.update(write=(d,), write_bias=())
        if mode == "state":
            out.update(read_query=(k, d), read_out=(2, s, v))
    else:
        conv = h*(2*k+v) if family == "gated_delta" else h*v+2*g*k
        out.update({"conv.weight": (size, conv), "conv.bias": (conv,), "log_rate": (h,)})
        if mode == "state":
            out["read_out"] = (s, h*v)
        if family == "gated_delta":
            out.update(projection=(conv, d), delta=(h, d), delta_bias=(h,),
                       write=(h, d), write_bias=(h,))
        else:
            out.update(projection=(h*v+conv+h, d), delta_bias=(h,))
    return out


def full_shapes(c):
    d, f = dim(c, "width", 4), dim(c, "ffn_width", 8)
    out = dict(gate=(f, d), up=(f, d), down=(d, f))
    family = c.get("memory_family", "kv")
    if family == "kv":
        out.update(query=(d, d), out=(d, d))
    elif family in {"affine", "delta"}:
        out.update(query=(dim(c, "key_width", 2), d), out=(2, d, dim(c, "value_width", 3)))
    elif family in {"gated_delta", "ssd"}:
        h, v = dim(c, "heads", 1), dim(c, "value_width", 3)
        out["out"] = (d, h*v)
        if family == "gated_delta":
            out.update(output_gate=(h*v, d), rho=(h, v))
        else:
            out.update(direct=(h,), rho=(h*v,))
    else:
        raise ValueError("unknown memory family")
    return out


REGIONAL = {"regional_total", "regional_total_zero", "regional_threshold", "regional_threshold_hard"}
SENDING = REGIONAL | {"constrained", "hard_selection", "direct_attention", "unit_attention", "pooled_attention", "proportional"}


def regional_defaults(c):
    out = dict(c)
    mode = out.setdefault("selection_mode", "regional_total")
    if mode not in SENDING:
        raise ValueError("unknown selection_mode")
    out.setdefault("local_score", "cat")
    out.setdefault("region_score", "ind")
    out.setdefault("hidden_width", 4)
    if mode in REGIONAL:
        out.setdefault("region_control_mode", "context")
        out.setdefault("region_control_width", 2)
        if out["region_control_mode"] == "context":
            out.setdefault("region_evidence", "cat")
    if out["region_score"] == "att":
        out.setdefault("region_key", "content_stats")
    if out["local_score"] == "match" or out["region_score"] == "att" or out.get("region_evidence") == "grouped":
        out.setdefault("selection_width", 4)
    return out


def describe_shapes(memory, region):
    c = regional_defaults(region)
    d, s = dim(memory, "width", 4), dim(memory, "summary_width", 2)
    f, hidden = d+4*s+3, dim(c, "hidden_width")
    out = {}

    def encoders(prefix):
        for group, width in (("cur", d), ("mem", 2*s), ("stat", 3)):
            out.update(prefixed(prefix+".encoders."+group, encoder(width, hidden, dim(c, "selection_width"))))

    if c["local_score"] == "cat":
        out.update(prefixed("score", scalar_head(f, hidden)))
    elif c["local_score"] == "match":
        encoders("score")
        for name in ("gate", "up"):
            out["score."+name] = (hidden, 3*dim(c, "selection_width"))
            out["score."+name+"_bias"] = (hidden,)
        out.update({"score.vector": (hidden,), "score.bias": ()})
    else:
        raise ValueError("invalid local_score")
    if c.get("region_control_mode") == "context":
        dr = dim(c, "region_control_width")
        if c["region_evidence"] == "cat":
            out.update(evidence_weight=(dr, f), evidence_bias=(dr,))
        elif c["region_evidence"] == "grouped":
            encoders("evidence")
            out.update({"evidence."+name: (dr, dim(c, "selection_width"))
                        for name in ("cur", "mem", "stat", "interaction")})
            out["evidence.bias"] = (dr,)
        else:
            raise ValueError("invalid region_evidence")
    if c.get("send_gain") == "context":
        out.update(prefixed("gain", scalar_head(f, hidden)))
    elif c.get("gain_logit") is not None:
        out["gain_logit"] = ()
    if c["region_score"] == "att":
        ds = dim(c, "selection_width")
        out.update(current_weight=(ds, d), current_bias=(ds,))
        if c["region_key"] == "content_stats":
            out.update(key_weight=(ds, 2*s+2), key_bias=(ds,))
        elif c["region_key"] == "content":
            out.update(prefixed("key", encoder(2*s, hidden, ds)))
        else:
            raise ValueError("invalid region_key")
    elif c["region_score"] != "ind":
        raise ValueError("invalid region_score")
    return out


def region_shapes(region):
    c = regional_defaults(region)
    out = {}
    if c.get("region_control_mode") == "context":
        out.update(prefixed("head", scalar_head(dim(c, "region_control_width")+2, dim(c, "hidden_width"))))
    if c["region_score"] == "att":
        ds = dim(c, "selection_width")
        out.update(prefixed("query", encoder(ds, ds, ds)))
    return out
