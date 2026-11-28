import base64
import fcntl
import hashlib
import hmac
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
import zlib

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tide_cloud_blobs as blobs
import tide_cloud_submit as submit


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def test_regional_get_signs_one_object_and_method_for_a_fixed_expiry(monkeypatch):
    monkeypatch.setattr(blobs.time, "time", lambda: 1000)
    host = "example.obs.cn-southwest-2.myhuaweicloud.com"
    region = SimpleNamespace(side="guiyang", bucket="example", host=host, credentials=("n", "public", "dummy secret"),
                             full_key=lambda key: "private/" + key)
    url = urlsplit(blobs.signed_get(region, "jobs/one/a b"))
    query = parse_qs(url.query)
    expected = hmac.new(b"dummy secret", b"GET\n\n\n1600\n/example/private/jobs/one/a%20b", hashlib.sha1).digest()
    assert url.netloc == host and url.path == "/private/jobs/one/a%20b"
    assert query == {"AccessKeyId":["public"], "Expires":["1600"], "Signature":[base64.b64encode(expected).decode()]}
    assert "dummy secret" not in url.geturl()
    with pytest.raises(ValueError):
        blobs.signed_get(region, "other", expires_in=3600)


def test_stream_verifies_full_body_without_disclosing_a_url_or_following_redirects(monkeypatch):
    raw = b"1234567"
    secret = "do-not-echo-query"
    url = "https://demo.obs.cn-southwest-2.myhuaweicloud.com/object?Signature=" + secret
    class Response(io.BytesIO):
        status = 200
        def getheader(self, key, default):
            return str(len(raw))
    class Connection:
        calls = []
        def __init__(self, host, timeout):
            pass
        def request(self, op, target):
            self.calls.append(op)
        def getresponse(self):
            return Response(raw)
        def close(self):
            pass
    monkeypatch.setattr(blobs.http.client, "HTTPSConnection", Connection)
    assert blobs.read_signed(url, digest(raw), len(raw)) == len(raw)
    output = io.BytesIO()
    assert blobs.read_signed(url, digest(raw), len(raw), output) == len(raw) and output.getvalue() == raw
    with pytest.raises(OSError, match="SHA-256"):
        blobs.read_signed(url, digest(b"wrong"), len(raw))
    Response.status = 307
    with pytest.raises(OSError, match="HTTP 307") as err:
        blobs.read_signed(url, digest(raw), len(raw))
    assert secret not in str(err.value) and Connection.calls == ["GET"] * 4
    def failed(*args):
        raise OSError("emergency trace: " + url)
    monkeypatch.setattr(Connection, "request", failed)
    with pytest.raises(OSError) as err:
        blobs.read_signed(url, digest(raw), len(raw))
    assert secret not in str(err.value)


def test_signed_object_uses_native_lock_atomic_cache_and_manifest_hash(tmp_path, monkeypatch):
    native, obs = tmp_path / "sfs", tmp_path / "obs"
    native.mkdir()
    contents = [b"visible", b"new binary bytes", b"locked"]
    first, second, held = map(digest, contents)
    objects = {digest(raw):len(raw) for raw in contents}
    mounted = obs / "inputs/objects" / first
    mounted.parent.mkdir(parents=True)
    mounted.write_bytes(contents[0])
    (native / "objects").mkdir(mode=0o700)
    downloaded = []
    def fetch(url, h, size, dest):
        downloaded.append(h)
        return blobs.measure(io.BytesIO(contents[1] if url == "scoped-read" else b"incorrect"), h, size, dest)
    monkeypatch.setattr(blobs, "read_signed", fetch)
    with (native / "objects" / (held + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = blobs.ensure_inputs(native, obs, objects, {})
        assert result["ready"] == {first:"mount"} and result["missing"] == {second:len(contents[1])}
        assert list(result["retry"]) == [held] and not downloaded
        result = blobs.ensure_inputs(native, obs, objects, {second:"invalid"})
        assert second in result["retry"] and not (native / "objects" / second).exists()
        assert list((native / "objects").glob(".hydrate-*")) == []
        result = blobs.ensure_inputs(native, obs, objects, {second:"scoped-read"})
        assert result["ready"] == {first:"cache", second:"regional_get"}
        assert (native / "objects" / second).read_bytes() == contents[1]
        result = blobs.ensure_inputs(native, obs, objects, {first:"unneeded", second:"unneeded"})
        assert result["ready"] == {first:"cache", second:"cache"} and downloaded == [second, second]
        (native / "objects" / ".verified" / second).write_text("{")
        assert blobs.ensure_one(native, obs, second, len(contents[1]), None) == "cache"


def test_allowlist_checks_and_stdin_are_applied_to_regional_readback(monkeypatch):
    h = digest(b"file")
    store = SimpleNamespace(head=lambda key: {"size":4,"sha256":h})
    node = {"id":"job-id", "name":"a", "prefix":"jobs/a"}
    signed = []
    def sign(store, key):
        signed.append(key)
        return "https://demo.obs.cn-southwest-2.myhuaweicloud.com/p?Signature=private-test"
    monkeypatch.setattr(blobs, "signed_get", sign)
    with pytest.raises(ValueError, match="outside the frozen manifest"):
        submit.grant_inputs(node,store,{h:4},{}, {})
    assert not signed
    grants = submit.grant_inputs(node,store,{h:4},{h:4}, {})
    assert signed == ["jobs/a/inputs/objects/" + h]
    def run(argv, **kwargs):
        assert argv[-1] == "-" and all("private-test" not in arg for arg in argv)
        package = json.loads(zlib.decompress(base64.b64decode(kwargs["input"])))
        assert package["files"] == {"action":"verify", "url":grants[h], "size":4, "sha256":h}
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(
            {"exit_code":0, "job_id":"job-id", "node":0, "stdout":json.dumps({"size":4,"sha256":h})}))
    monkeypatch.setattr(subprocess, "run", run)
    assert submit.verify_object(node,store,"store/" + h,h,4) == {"sha256":h,"size":4}
