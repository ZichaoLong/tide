"""Fixed, read-only downstream inputs for TL-REQ-006 diagnostics.

The external Git object is an explicit source dependency, never an installed
package or a mutable working tree. Configurations are retained in this repo.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

SOURCE = "6e7bd49eb5e35c15445826cac3a9b6ddda75aff4"
REQUEST = "dc943cc7ce773e3dac4d62736ab524953573decb"
REPORT = "8995958728167eee7a4bcc2139f5e1794cbf82ab"
CASES = ("hb88-pool-s11", "hb184-quad-s11", "tree-r64-h2-q3-s11", "tf-hb184-s11")
CONFIG_ROOT = Path(__file__).resolve().parents[1]/"examples/diagnostics/req006"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def configuration(case=None, path=None):
    if (case is None) == (path is None):
        raise ValueError("choose one retained case or explicit configuration")
    if case is not None:
        if case not in CASES:
            raise ValueError("unknown REQ-006 representative")
        manifest = json.loads((CONFIG_ROOT/"manifest.json").read_text())
        entry = manifest["configurations"][case]
        raw = gzip.decompress((CONFIG_ROOT/entry["file"]).read_bytes())
        if sha256(raw) != entry["sha256"]:
            raise ValueError("retained REQ-006 configuration hash mismatch")
        origin = dict(entry, request=REQUEST, report=REPORT)
    else:
        raw = Path(path).read_bytes()
        origin = dict(file=str(Path(path).resolve()), sha256=sha256(raw), custom=True)
    config = json.loads(raw)
    if config.get("kind") not in {"direct_semantic", "transformer"}:
        raise ValueError("only the frozen direct semantic/Transformer implementations are supported")
    return config, origin


def extract_source(repository, destination, *, revision=SOURCE):
    """Extract Python bytes from a fixed Git commit without touching its tree."""
    repository, destination = Path(repository).resolve(), Path(destination)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository), *args])
    resolved = git("rev-parse", revision+"^{commit}").decode().strip()
    if resolved != revision:
        raise ValueError("downstream revision must be a full immutable commit")
    archive = git("archive", "--format=tar", revision, "tide_lab")
    destination.mkdir(parents=True, exist_ok=False)
    files, total = {}, 0
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
        for member in stream:
            path = Path(member.name)
            if member.isdir() or path.suffix != ".py":
                continue
            if not member.isfile() or path.is_absolute() or ".." in path.parts or path.parts[0] != "tide_lab":
                raise ValueError("unexpected downstream source archive member")
            total += member.size
            if total > 32*2**20:
                raise ValueError("downstream Python source exceeds the 32 MiB diagnostic bound")
            data = stream.extractfile(member).read()
            target = destination/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            files[path.as_posix()] = sha256(data)
    if "tide_lab/campaign_runtime.py" not in files:
        raise ValueError("frozen downstream source is incomplete")
    return dict(repository_hint=str(repository), commit=resolved,
                git_tree=git("rev-parse", revision+"^{tree}").decode().strip(),
                python_files_sha256=files, bytes=total,
                content_sha256=sha256(json.dumps(files, sort_keys=True).encode()))


def synthetic_inputs(global_batch, length, seed, *, vocabulary=4096):
    """One CPU material independent of physical microbatch partitioning."""
    import torch
    if min(global_batch, length) < 1 or vocabulary <= 4:
        raise ValueError("invalid synthetic input dimensions")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tokens = torch.randint(4, vocabulary, (global_batch, length+1), generator=generator)
    return dict(input_ids=tokens[:, :-1].clone(), labels=tokens[:, 1:].clone())


def tensor_digest(value):
    """Content identity for tensor/container state, outside every timed region."""
    import torch
    digest = hashlib.sha256()
    def add(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(repr((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict:")
            for key in sorted(item, key=repr):
                add(key); add(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for child in item:
                add(child)
        else:
            digest.update(repr((type(item).__name__, item)).encode())
        digest.update(b"\0")
    add(value)
    return digest.hexdigest()
