"""LH FP64 norm policy: upstream payloads retain their own numerical precision."""
from dataclasses import replace
from tidegraph.compare import equivalent


def compare(graph, actual, expected, path):
    assert len(actual.trace) == len(expected.trace)
    a_trace, b_trace = [], []
    for a, b in zip(actual.trace, expected.trace):
        equivalent((a["time"], a["batch"], a["node"]), (b["time"], b["batch"], b["node"]))
        a, b = dict(a), dict(b)
        if graph.nodes[a["node"]].readout == "norm-fp64-v1":
            # Every Read must compute its own candidate norm at FP64 accuracy.
            equivalent(a["descriptor"], a["proposal"].double().norm(), path+".actual_norm")
            equivalent(b["descriptor"], b["proposal"].double().norm(), path+".expected_norm")
            delta = (a["proposal"].double()-b["proposal"].double()).norm()
            assert abs(a["descriptor"]-b["descriptor"]) <= delta+1e-10+1e-8*abs(b["descriptor"])
            # The remaining complete event (including proposal, exact route and
            # payload-dtype control) still uses the original comparison policy.
            a.pop("descriptor"); b.pop("descriptor")
        a_trace.append(a); b_trace.append(b)
    equivalent(replace(actual, trace=a_trace), replace(expected, trace=b_trace), path)
