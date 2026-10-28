#!/usr/bin/env python3
"""Bounded lossless v2 -> v1 material adapter for older independent consumers.

No forward, recomputation, random initialization or deduplication. Tensor table
indices and every float32 value are preserved; frozen zero buffers are expanded.
This diagnostic does not extend the older consumer's supported semantics.
"""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import sys
from durable_records import write_json


def decode(source, *, max_elements):
    source = Path(source)
    data = json.loads(source.read_text())
    if data.get('schema') != 'tide-weighted-benchmark-fixture-v2':
        raise ValueError('expected binary weighted fixture v2')
    if type(max_elements) is not int or max_elements <= 0:
        raise ValueError('positive element ceiling required')
    rows = data['tensors']
    counts = []
    for row in rows:
        if row['dtype'] != 'float32' or any(type(n) is not int or n < 0 for n in row['shape']):
            raise ValueError('invalid FP32 tensor shape/dtype')
        counts.append(math.prod(row['shape']))
    if sum(counts) > max_elements:
        raise ValueError('legacy JSON element ceiling exceeded')
    blob = data['binary']
    name = Path(blob['file'])
    if name.name != str(name) or name.name in ('', '.', '..') or blob['byte_order'] != 'little':
        raise ValueError('invalid binary material path/byte order')
    path = source.parent/name
    if path.stat().st_size != blob['nbytes']:
        raise ValueError('binary material size mismatch')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    if digest.hexdigest() != blob['sha256']:
        raise ValueError('binary material hash mismatch')
    decoded = []
    with path.open('rb') as stream:
        for row, count in zip(rows, counts):
            if row['encoding'] == 'zeros':
                if row['trainable']:
                    raise ValueError('trainable zero encoding refused')
                values = [0.0]*count
            elif row['encoding'] == 'binary':
                offset, size = row['offset'], row['nbytes']
                if (type(offset) is not int or type(size) is not int or offset < 0 or
                        size != 4*count or offset+size > blob['nbytes']):
                    raise ValueError('invalid tensor byte range')
                stream.seek(offset)
                raw = stream.read(size)
                if hashlib.sha256(raw).hexdigest() != row['sha256']:
                    raise ValueError('tensor material hash mismatch')
                tensor = array.array('f'); tensor.frombytes(raw)
                if sys.byteorder != 'little':
                    tensor.byteswap()
                values = tensor.tolist()
            else:
                raise ValueError('unknown tensor encoding')
            decoded.append({k: row[k] for k in ('shape', 'dtype', 'trainable')} | dict(values=values))
    data.update(schema='tide-weighted-benchmark-fixture-v1', tensors=decoded,
                decoded_from=dict(schema='tide-weighted-benchmark-fixture-v2',
                    manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    binary_sha256=blob['sha256'], policy='exact tensor table/values; no numerical execution'))
    for name in ('binary', 'max_materialize_bytes', 'parameter_aliases'):
        data.pop(name, None)
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-elements', type=int, default=20000000)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = decode(args.input, max_elements=args.max_elements)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, data)


if __name__ == '__main__':
    main()
