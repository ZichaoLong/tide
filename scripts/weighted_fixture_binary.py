"""Streaming little-endian FP32 fixture storage, with per-tensor SHA256.

Actual TensorImpl identities index the manifest; equal values never merge.
Only explicitly identified inert legacy buffers use the exact zero codec.
"""
import hashlib
import os
from pathlib import Path
import sys
import tempfile


class TensorWriter:
    def __init__(self, path):
        if sys.byteorder != "little":
            raise ValueError("weighted binary fixture requires a little-endian host")
        self.path = Path(path)
        if self.path.exists(): raise FileExistsError(self.path)
        fd, name = tempfile.mkstemp(prefix=".weighted-", dir=self.path.parent)
        self.temporary = Path(name)
        self.stream = os.fdopen(fd, "wb")
        self.rows, self.identities, self.digest = [], {}, hashlib.sha256()
        self.handles = []  # Keep TensorImpl addresses alive until publication.
        self.closed = False

    def tensor(self, x, *, zero=False):
        if self.closed: raise ValueError("tensor writer is closed")
        key = x._cdata
        if key in self.identities: return self.identities[key]
        import torch
        if x.dtype != torch.float32:
            raise ValueError("weighted binary fixture requires explicit FP32")
        row = dict(shape=list(x.shape), dtype="float32", trainable=x.requires_grad)
        if zero:
            if x.requires_grad or torch.count_nonzero(x).item():
                raise ValueError("zero codec only accepts actual frozen zeros")
            row["encoding"] = "zeros"
        else:
            value = x.detach().cpu().contiguous().numpy()
            raw = memoryview(value.reshape(-1)).cast("B")
            row.update(encoding="binary", offset=self.stream.tell(), nbytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            self.stream.write(raw); self.digest.update(raw)
        self.identities[key] = len(self.rows)
        self.handles.append(x)
        self.rows.append(row)
        return len(self.rows)-1

    def finish(self):
        self.stream.flush(); os.fsync(self.stream.fileno()); size = self.stream.tell()
        self.stream.close(); self.closed = True
        # Exclusive publication: a competing producer cannot be overwritten.
        os.link(self.temporary, self.path); self.temporary.unlink()
        fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
        return dict(file=self.path.name, nbytes=size, sha256=self.digest.hexdigest(), byte_order="little")

    def abort(self):
        if not self.closed: self.stream.close()
        self.closed = True
        self.temporary.unlink(missing_ok=True)
