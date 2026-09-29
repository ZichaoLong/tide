#!/usr/bin/env python3
"""Prepare an inspectable active-scale workload without importing Torch."""
import argparse
from pathlib import Path
from durable_records import write_json, replace_text
from flow_topology import ranked_graph, make_packet, native_text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--preset', choices=('smoke', 'wide', 'many-nodes'), default='smoke')
    p.add_argument('--memory', choices=('add', 'attention'), default='add')
    p.add_argument('--topology', choices=('ranked-local', 'timed-local'), default='ranked-local')
    defaults = dict(smoke=(4,4,2,1,2,4,8,2,3,17), wide=(15,32,4,1,8,8,2048,512,12,50304),
                    **{'many-nodes':(128,32,4,1,8,8,128,8,12,257)})
    names = ('layers','region-width','fanout','skip','local-span','cross-every','width','batch','tokens','vocab')
    for name in names: p.add_argument('--'+name, type=int)
    p.add_argument('--rank-gap', type=int, default=1)
    p.add_argument('--budget', type=int, default=1)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--clear', action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args(); config = {}
    for name, default in zip(names, defaults[a.preset]):
        field=name.replace('-', '_'); config[field] = default if getattr(a, field) is None else getattr(a, field)
    graph_keys = ('layers','region_width','fanout','skip','local_span','cross_every')
    graph=ranked_graph(**{k:config.pop(k) for k in graph_keys}, rank_gap=a.rank_gap,
                       delayed=a.topology=='timed-local')
    packet=make_packet(graph=graph, memory=a.memory, budget=a.budget, seed=a.seed, clear=a.clear, **config)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(a.output_dir/'workload.json', packet)
    replace_text(a.output_dir/'topology.txt', native_text(packet))
    print(packet['sha256'], packet['counts'])


if __name__ == '__main__': main()
