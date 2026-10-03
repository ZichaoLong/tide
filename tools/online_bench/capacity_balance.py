"""Bounded static owner moves using the complete consumer's memory envelope.

Inputs contain shapes, physical topology and budgets only. Equal-cost nodes
share an envelope trial; topology chooses their stable locality tie-break.
"""
from collections import defaultdict


def rebalance(g, owners, usable, evaluate, max_evaluations=4096):
    owners=list(owners);n=len(g.sources);moves=evaluations=0
    adjacent=[[] for _ in owners]
    for a,b in g.edges:
        if a!=b:
            adjacent[a].append(b);adjacent[b].append(a)
    def score(rows):
        peaks=[r['estimated_peak_bytes'] for r in rows]
        excess=[max(0,p-u) for p,u in zip(peaks,usable)]
        return (sum(excess),max(excess),tuple(sorted(peaks,reverse=True))),excess
    current,excess=score(evaluate(owners))
    for _ in range(2*len(owners)):
        if current[0]==0 or evaluations>=max_evaluations:
            break
        donor=max(range(g.devices),key=lambda d:excess[d])
        if owners.count(donor)<=1:
            break
        groups=defaultdict(list)
        for i,owner in enumerate(owners):
            if owner==donor:
                groups[(int(i<n),g.sources[i] if i<n else 0,g.slots[i] if i<n else 0)].append(i)
        best=None
        for signature in sorted(groups):
            for target in range(g.devices):
                if target==donor or evaluations>=max_evaluations:
                    continue
                def cut_delta(i):
                    return sum(int(owners[j]!=target)-int(owners[j]!=donor) for j in adjacent[i])
                node=min(groups[signature],key=lambda i:(cut_delta(i),i))
                trial=owners.copy();trial[node]=target
                value,trial_excess=score(evaluate(trial));evaluations+=1
                key=(value,cut_delta(node),node,target)
                if value<current and (best is None or key<best[0]):
                    best=(key,trial,trial_excess)
        if best is None:
            break
        key,owners,excess=best;current=key[0];moves+=1
    return owners,moves,evaluations
