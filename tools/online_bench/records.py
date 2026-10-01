"""Bounded optional value diagnostics, separate from ordinary timing."""

def tensor(value):
    if value is None:
        return None
    if value.numel()>100000:
        raise ValueError("diagnostic tensor exceeds 100000 elements")
    return dict(shape=list(value.shape), values=value.detach().cpu().reshape(-1).tolist())


def observer(records):
    def observe(kind, step, value):
        row = dict(kind=kind, step=step)
        if kind != "window":
            row["parameters"] = {k: tensor(v.grad if kind == "gradients" else v) for k, v in value.items()}
        else:
            q = value.continuation
            row.update(cut=q.cut, outputs=[[b,t,p,tensor(v)] for b,t,p,v in value.outputs],
                states=[[b,n,s.last_time,s.observations,tensor(s.value),{k:tensor(v) for k,v in sorted(s.slots.items())}]
                        for (b,n),s in sorted(q.states.items())],
                history=[[b,r,h.last_time,{k:sorted([n,c] for n,c in m.items()) for k,m in h.node_maps.items()}]
                         for (b,r),h in sorted(q.history.items())],
                pending=[[a.batch,a.node,a.time,a.kind,a.source,a.position,tensor(a.value)] for a in q.pending],
                events=[[e["batch"],e["node"],e["time"],e["active"]] for e in value.trace],
                ledger=[[b,p,pos,t] for (b,p),(pos,t) in sorted(q.ledger.items())])
        records.append(row)
    return observe

