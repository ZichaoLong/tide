"""Verify frozen TIDE inputs in the same private SFS cache and locks used by hydrate()."""

import base64
import fcntl
import hashlib
import hmac
import http.client
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from urllib.parse import quote, urlencode, urlsplit

import tide_cloud_relay as relay


class BlobMismatch(OSError):
    pass


def signed_get(region, key, expires_in=600):
    """Sign one GET to one Guiyang object. The returned URL belongs only in the authenticated SSH stdin."""
    relay.require(region.side == "guiyang" and 0 < expires_in <= 900, "a short-lived Guiyang GET is required")
    encoded = quote(region.full_key(key), safe="/~")
    expires = str(int(time.time()) + expires_in)
    _, public_id, secret = region.credentials
    string = "GET\n\n\n" + expires + "\n/" + region.bucket + "/" + encoded
    signature = base64.b64encode(hmac.new(secret.encode(), string.encode(), hashlib.sha1).digest()).decode()
    query = urlencode({"AccessKeyId": public_id, "Expires": expires, "Signature": signature})
    return "https://" + region.host + "/" + encoded + "?" + query


def measure(stream, digest, size, output):
    sha = hashlib.sha256()
    seen = 0
    while True:
        block = stream.read(2**20)
        if not block:
            break
        seen += len(block)
        if seen > size:
            raise BlobMismatch("object exceeds its bound length: " + digest)
        sha.update(block)
        if output is not None:
            output.write(block)
    if seen != size or sha.hexdigest() != digest:
        raise BlobMismatch("object length or SHA-256 differs: " + digest)
    return seen


def read_signed(url, sha256, size, output=None):
    """Read and verify one object directly on the node. No redirects or secret-bearing error strings."""
    validate(sha256, size)
    conn = None
    try:
        target = urlsplit(url)
        if target.scheme != "https" or not re.fullmatch(r"[a-z0-9][a-z0-9.-]*\.obs\.[a-z0-9-]+\.myhuaweicloud\.com", target.netloc):
            raise OSError("invalid regional host")
        conn = http.client.HTTPSConnection(target.netloc, timeout=45)
        conn.request("GET", target.path + "?" + target.query)
        with conn.getresponse() as data:
            if data.status != 200:
                return_code = data.status
            elif int(data.getheader("Content-Length", "-1")) != size:
                raise BlobMismatch("unexpected regional object length for " + sha256)
            else:
                return measure(data, sha256, size, output)
    except BlobMismatch:
        raise
    except (ValueError, UnicodeError, OSError, http.client.HTTPException) as exc:
        raise OSError("regional GET failed for " + sha256 + " (" + type(exc).__name__ + ")") from None
    finally:
        if conn is not None:
            conn.close()
    raise OSError(f"regional GET returned HTTP {return_code} for {sha256}")


def validate(digest, size):
    relay.require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
                  and type(size) is int and size >= 0, "invalid immutable object identity")


def required_objects(root, descriptor):
    """Match frozen hydrate's local-base, inline reference, external reference and runtime endpoint choices."""
    source = descriptor["source"]
    relay.require(isinstance(source, str) and re.fullmatch(r"[a-zA-Z0-9_.-]+", source), "invalid source identity")
    base = relay.member(root, "bases/" + source)
    manifest = descriptor.get("base_manifest")
    if manifest is None:
        manifest = relay.read(relay.member(base, "base.json"))
    relay.require(manifest["identity"] == source, "frozen base identity differs")
    records = [item for name, item in manifest["files"].items() if not relay.member(base, name).is_file()]
    records.extend(item for item in descriptor["references"].values() if "bytes" not in item)
    if descriptor.get("resume"):
        records.append(descriptor["resume"])
    objects = {}
    for item in records:
        digest, size = item["sha256"], item["size"]
        validate(digest, size)
        relay.require(objects.setdefault(digest, size) == size, "two lengths claimed for the same input")
    return objects


def stamp(path):
    info = path.stat()
    relay.require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid(), "untrusted private cache file")
    # hydrate() hardlinks the cache into each attempt, which changes ctime and link count without a data write.
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]


def private_directory(path):
    path.mkdir(mode=0o700, exist_ok=True)
    info = path.stat()
    relay.require(path == path.resolve() and info.st_uid == os.getuid() and not info.st_mode & 0o022,
                  "untrusted native cache directory")


def ensure_one(native, obs, digest, size, url):
    validate(digest, size)
    cache = relay.member(native, "objects")
    private_directory(cache)
    proofs = relay.member(cache, ".verified")
    private_directory(proofs)
    dest = relay.member(cache, digest)
    proof = relay.member(proofs, digest)

    def verified():
        try:
            current = stamp(dest)
            return current[2] == size and current == relay.optional(proof)
        except (FileNotFoundError, ValueError, UnicodeError):
            return False

    # Other workers can hash or link the same immutable inode under their own lock.
    if verified():
        return "cache"
    with relay.member(cache, digest + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if verified():
            return "cache"
        source = None
        if dest.exists() and stamp(dest)[2] == size and relay.digest(dest) == digest:
            source = "cache"
        if source is None:
            with tempfile.NamedTemporaryFile(dir=cache, prefix=".hydrate-", delete=False) as target:
                temp = Path(target.name)
                try:
                    try:
                        mounted = relay.member(obs, "inputs/objects/" + digest)
                        with mounted.open("rb") as stream:
                            measure(stream, digest, size, target)
                        source = "mount"
                    except OSError:
                        if not url:
                            return "missing"
                        target.seek(0)
                        target.truncate()
                        read_signed(url, digest, size, target)
                        source = "regional_get"
                    target.flush()
                    os.fsync(target.fileno())
                    os.replace(temp, dest)
                finally:
                    temp.unlink(missing_ok=True)
        # Only verified bytes, in the recorded inode/mtime, qualify for the cheap subsequent check.
        with tempfile.NamedTemporaryFile(dir=proofs, delete=False, mode="w") as f:
            tmp = Path(f.name)
            try:
                json.dump(stamp(dest), f)
                f.flush()
                os.replace(tmp, proof)
            finally:
                tmp.unlink(missing_ok=True)
        return source


def ensure_inputs(native, obs, objects, grants):
    result = {"ready": {}, "missing": {}, "retry": {}}
    for digest, size in objects.items():
        try:
            source = ensure_one(native, obs, digest, size, grants.get(digest))
            if source == "missing":
                result["missing"][digest] = size
            else:
                result["ready"][digest] = source
        except BlockingIOError:
            result["retry"][digest] = "another worker is filling this input"
        except (ValueError, OSError, http.client.HTTPException) as exc:
            result["retry"][digest] = str(exc)
    return result
