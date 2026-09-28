"""Identity of the standalone historical-topology placement client."""
import hashlib
from pathlib import Path


def client_hash(root):
    root = Path(root)
    paths = sorted((root / 'tools/accelerator_scale').glob('*'))
    paths += [root / name for name in ('cpp/scale/config.cpp', 'cpp/scale/model.cpp',
                                      'cpp/scale/scale.h', 'cpp/bench/compare.cpp',
                                      'cpp/bench/streaming.h', 'cpp/bench/metrics_jsonl_writer.h')]
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode() + b'\0' + path.read_bytes() + b'\0')
    return digest.hexdigest()
