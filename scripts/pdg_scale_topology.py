#!/usr/bin/env python3
"""Export only LH connectivity to the portable text PDG scale input; no weights."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
from durable_records import replace_text, write_json


def read_csr(path, nodes):
    data = Path(path).read_bytes()
    if len(data) < 24 or len(data) % 8:
        raise ValueError('truncated CSR file')
    values = struct.unpack('<' + 'q' * (len(data)//8), data)
    rows, columns, count = values[:3]
    if rows != nodes or columns != nodes or count < 0 or len(values) != 3+nodes+1+2*count:
        raise ValueError('CSR header/length mismatch')
    ptr = values[3:4+nodes]
    indices, ids = values[4+nodes:4+nodes+count], values[4+nodes+count:]
    if (ptr[0] != 0 or ptr[-1] != count or any(a > b for a, b in zip(ptr, ptr[1:]))
            or any(p < 0 or p > count for p in ptr) or sorted(ids) != list(range(count))
            or any(v < 0 or v >= nodes for v in indices)):
        raise ValueError('invalid CSR pointers/columns/edge identities')
    edges = [None]*count
    for source in range(nodes):
        for k in range(ptr[source], ptr[source+1]):
            edges[ids[k]] = (source, indices[k])
    return edges


def export(manifest, output):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if output.exists() or output.with_suffix(output.suffix+'.json').exists():
        raise ValueError('topology output already exists')
    source = json.loads(manifest.read_text())
    cfg = source['graph_config']; levels = cfg['base_num_or_hpnums']
    if len(levels) < 4 or any(type(n) is not int or n < 0 for n in levels):
        raise ValueError('explicit per-level counts required')
    n, points, forced = sum(levels), sum(levels[:-1]), sum(levels[:-2])
    local, budget, layers = cfg['localnum'], source['selectnum'], 2
    if (not source['with_lead_point'] or n-points != (points-forced)*local
            or forced < 1 or not 1 <= budget <= local):
        raise ValueError('unsupported lead-point/region topology')
    graph_dir = Path(source['graph_dir'])
    edges, counts, hashes = [], {}, {}
    for name, src, dst in (('inputA', 0, 0), ('outputA', n, n), ('ioA', 0, n), ('oiA', n, 0)):
        path = graph_dir/name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes[name] != source['graph_files_sha256'][name]:
            raise ValueError('graph source hash changed: '+name)
        block = read_csr(path, n); counts[name] = len(block)
        edges.extend((a+src, b+dst) for a, b in block)
    text = f'TIDE_PDG_SCALE_1\n{n} {local} {points} {forced} {budget} {layers} {len(edges)}\n'
    text += ''.join(f'{a} {b}\n' for a, b in edges)
    output.parent.mkdir(parents=True, exist_ok=True)
    replace_text(output, text)
    info = dict(schema='pdg-scale-topology-v1', topology_sha256=hashlib.sha256(text.encode()).hexdigest(),
                source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                graph_files_sha256=hashes, edges=counts, static_nodes=n, body_nodes=2*n,
                pdg_nodes=2*n+1, logical_edges=len(edges), physical_edges=layers*len(edges)+layers,
                localnum=local, selectnum=budget, levels=levels, body_ticks_per_token=layers,
                weights='fresh random initialization; none imported from LH')
    write_json(output.with_suffix(output.suffix+'.json'), info)
    return info


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph-input', required=True)
    p.add_argument('--output-file', required=True)
    a = p.parse_args()
    print(json.dumps(export(a.graph_input, a.output_file)))
