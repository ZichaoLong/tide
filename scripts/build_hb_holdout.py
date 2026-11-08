#!/usr/bin/env python3
"""Freeze the predeclared third-batch test set without reading any model."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tide_lab.heldout import make_dataset, validate_dataset
from tide_lab.records import atomic_json, digest

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('do not replace a frozen evaluation set')
    data = make_dataset(dict(n_keys=8, n_values=8, pairs=4, delay=2))
    atomic_json(a.output, data)
    print(dict(sha256=digest(a.output), **validate_dataset(data)))
