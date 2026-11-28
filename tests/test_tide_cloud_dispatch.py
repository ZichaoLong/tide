"""Transport retries must not double-spend the evaluation budget or reassign an attempt."""
import base64
import hashlib
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tide_cloud_node import hydrate
from tide_portable import Frozen, sha, write


def test_node_hydrates_a_sha_bound_original_parent_and_distinct_signal_endpoint(tmp_path, monkeypatch):
    import tide_cloud_node
    root = tmp_path / "payload"
    monkeypatch.setattr(tide_cloud_node, "ROOT", root)
    base = root / "bases/source"
    write(base / "base.json", {"identity":"source", "files":{}, "paths":{}, "roots":{}})
    area = tmp_path / "shared/attempts/one"
    obs = tmp_path / "obs"
    objects = obs / "inputs/objects"
    objects.mkdir(parents=True)
    source = b'{"id":"task"}\n'
    parent, endpoint = b"phase-start", b"stopped-state"
    digest = lambda b: hashlib.sha256(b).hexdigest()
    for raw in (parent, endpoint):
        (objects / digest(raw)).write_bytes(raw)
    d = {"task":"task", "source":"source", "original_path":"/previous/task.json", "task_bytes":base64.b64encode(source).decode(),
         "fingerprint":digest(source), "references":{"/previous/parent.pt":{"name":"evidence/parent.pt", "sha256":digest(parent),
          "size":len(parent)}}, "resume":{"sha256":digest(endpoint), "size":len(endpoint)}}
    packed = hydrate(area, obs, d)
    frozen = Frozen(packed)
    assert frozen.request("task", d["fingerprint"])[0]["id"] == "task"
    assert sha(frozen.path("/previous/parent.pt")) == digest(parent)
    assert sha(packed / "runtime-endpoint.pt") == digest(endpoint)
    assert hydrate(area, obs, d) == packed


def test_east_claim_and_regional_retry_spend_one_reservation(tmp_path):
    if not shutil.which("maflow") and not (Path.home() / ".local/bin/maflow").is_file():
        pytest.skip("maflow client is installed on the ModelArts controller")
    from tide_cloud_dispatch import publish_claim

    class Store:
        def __init__(self):
            self.objects = {}
            self.offline = False

        def put_bytes(self, key, data):
            if self.offline:
                raise ConnectionError(key)
            if key in self.objects and self.objects[key] != data:
                raise FileExistsError(key)
            self.objects[key] = data

    east, region = Store(), Store()
    task = tmp_path / "original-task.json"
    write(task, {"phase":"a2", "evaluation_budget_key":"model", "evaluation_token_limit":132})
    fp = sha(task)
    d = {"attempt":"a1", "fingerprint":fp, "job":"node-a"}
    row = {"id":"native-id", "backend":"cpu", "path":str(task)}
    plan = {"evaluation":{"additional_reservation":32}}
    state = {"attempts":{}, "budget":{"evaluation":{"model":100}, "preparation":0,
                                      "preparation_limit":1, "preparation_prior":{}}}
    region.offline = True
    with pytest.raises(ConnectionError):
        publish_claim(tmp_path, region, east, "scratch", state, row, d, plan)
    assert state["budget"]["evaluation"] == {"model":132}
    assert state["attempts"]["a1"]["status"] == "claimed"
    region.offline = False
    publish_claim(tmp_path, region, east, "scratch", state, row, d, plan)
    assert state["budget"]["evaluation"] == {"model":132}
    assert state["attempts"]["a1"]["status"] == "dispatched"
    with pytest.raises(ValueError, match="already claimed"):
        publish_claim(tmp_path, region, east, "scratch", state, row, {**d, "attempt":"a2", "job":"node-b"}, plan)
    other = {**d, "attempt":"a2", "fingerprint":"0"*64}
    with pytest.raises(ValueError, match="evaluation_token_limit"):
        publish_claim(tmp_path, region, east, "scratch", state, row, other, plan)
