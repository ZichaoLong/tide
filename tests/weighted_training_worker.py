"""Fresh-process weighted disk-cut continuation; invoked under the parent's lease."""
import sys
from dataclasses import replace
import torch
from test_resident_weighted_training import binding, owner, cotangents


def continue_cut(session, base, span):
    xs = [replace(x, time=x.time+3*span, position=x.position+6) for x in base]
    w = session.advance(xs, stop=4*span, sealed_until=4*span)
    session.backward([cotangents(session, w, "outputs")])
    assert session.step().applied


if __name__ == "__main__":
    target, source, destination, dtype = sys.argv[1:]
    b, _, xs, span = binding("kv", "fc", getattr(torch, dtype))
    with torch.no_grad():
        session = owner(b, target, checkpoint=source)
        continue_cut(session, xs, span)
        session.save(destination)
        session.close()
