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

        def get_bytes(self, key):
            return self.objects[key]

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
    d = {"attempt":"a1", "fingerprint":fp, "job":"node-a",
         "task_bytes":base64.b64encode(task.read_bytes()).decode()}
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

    # Restore after East accepted the claim but before the old host persisted its state.
    replacement = {"attempts":{}, "budget":{"evaluation":{"model":100}, "preparation":0,
                                             "preparation_limit":1, "preparation_prior":{}}}
    winning = publish_claim(tmp_path, region, east, "other-node", replacement, row, {**d, "job":"node-b"}, plan,
                            transport="ssh")
    assert winning["status"] == "claimed" and winning["job"] == "node-a"
    publish_claim(tmp_path, region, east, "scratch", replacement, row, d, plan)
    assert replacement["budget"]["evaluation"]["model"] == 132
    assert "other-node/inputs/authority/a1.json" not in region.objects


def test_preparation_cloud_hours_merge_two_chips_on_one_physical_board(tmp_path, monkeypatch):
    if not shutil.which("maflow") and not (Path.home() / ".local/bin/maflow").is_file():
        pytest.skip("maflow account client")
    import tide_cloud_evidence as evidence
    monkeypatch.setattr(evidence, "REPO", tmp_path)
    write(tmp_path / ".local/pilot-r3/budget.json", {"segments":{"original":{"preparation_device_seconds":20}},
                                                     "preparation_max_device_seconds":480})
    state = {"attempts":{k:{"row":{"phase":phase}, "backend":backend}
                         for k,phase,backend in (("one","preflight","npu"),("two","preflight","npu"),
                                                 ("cpu","preflight","cpu"),("formal","b","npu"))}}
    nodes = {"job":{"queue":{"tasks":[{"key":key,"state":"running","physical_card":1,"started":begin}
                                      for key,begin in (("one",10),("two",15),("cpu",0),("formal",1))]}}}
    observation = evidence.preparation_budget(state, nodes, 30)
    assert observation["remaining_seconds"] == 440
    assert {x["key"] for x in observation["registrations"]} == {"one","two"}
    next_observation = evidence.preparation_budget(state, {"job":{"queue":None}}, 60)
    assert next_observation["remaining_seconds"] == 440 and not next_observation["registrations"]


def test_only_completed_publications_can_advance_reserved_attempts(tmp_path):
    if not shutil.which("maflow") and not (Path.home() / ".local/bin/maflow").is_file():
        pytest.skip("maflow account client")
    from concurrent.futures import Future
    from tide_cloud_controller import Controller
    from tide_portable import read
    ctl = Controller.__new__(Controller)
    ctl.area = tmp_path
    ctl.state = {"attempts":{k:{"fingerprint":k, "status":"claimed"} for k in ("slow","ready","retry")},
                 "retries":{k:{} for k in ("slow","ready","retry")}}
    ctl.publications = {k:Future() for k in ctl.state["attempts"]}
    ctl.publications["ready"].set_result({"status":"awaiting_submission", "published":10})
    ctl.publications["retry"].set_exception(TimeoutError("regional write"))
    ctl.finish_publications()
    saved = read(tmp_path / "state.json")
    assert saved["attempts"]["ready"]["status"] == "awaiting_submission"
    assert saved["attempts"]["slow"]["status"] == saved["attempts"]["retry"]["status"] == "claimed"
    assert set(saved["retries"]) == {"slow", "retry"} and set(ctl.publications) == {"slow"}
    assert "publish:retry" in saved["errors"]
    ctl.publications["retry"] = Future()
    ctl.publications["retry"].set_result({"status":"awaiting_submission", "published":20})
    ctl.finish_publications()
    assert "publish:retry" not in ctl.state["errors"] and ctl.state["retries"] == {"slow":{}}


def test_slow_ingress_batch_leaves_a_second_slot_on_its_node(tmp_path):
    if not shutil.which("maflow") and not (Path.home() / ".local/bin/maflow").is_file():
        pytest.skip("maflow account client")
    from concurrent.futures import Future
    from types import SimpleNamespace
    from tide_cloud_controller import Controller
    ctl = Controller.__new__(Controller)
    ctl.area, ctl.region, ctl.publications, ctl.submissions = tmp_path, None, {}, {}
    ctl.submission_pool = SimpleNamespace(submit=lambda *args: Future())
    attempts = {str(i):{"attempt":str(i), "job":"n", "transport":"ssh", "status":"awaiting_submission",
                        "row":{"priority":i}} for i in range(17)}
    ctl.state = {"attempts":attempts}
    nodes = {"n":{"ready":True, "id":"node-id", "queue":{"tasks":[]}}}
    ctl.deliver(nodes)
    slow, fast = list(ctl.submissions)
    assert slow[2] == tuple(map(str,range(16,10,-1))) and fast[2] == tuple(map(str,range(10,4,-1)))
    ctl.deliver(nodes)
    assert list(ctl.submissions) == [slow,fast]
    ctl.submissions[fast].set_result({a:{"status":"submitted"} for a in fast[2]})
    ctl.deliver(nodes)
    assert all(attempts[a]["status"] == "dispatched" for a in fast[2])
    assert list(ctl.submissions) == [slow, ("n","ssh",tuple(map(str,range(4,-1,-1))))]


def test_positive_no_start_proof_reuses_eval_reservation_and_removes_the_global_claim(tmp_path):
    if not shutil.which("maflow") and not (Path.home() / ".local/bin/maflow").is_file():
        pytest.skip("maflow account client")
    from concurrent.futures import Future
    from tide_cloud_controller import Controller
    ctl = Controller.__new__(Controller)
    future = Future()
    future.set_result({"status":"never_started"})
    ctl.relay_work = {"attempt":future}
    ctl.state = {"retries":{}, "blocked":{}}
    claim = {"status":"dispatched", "reserved_evaluation_tokens":32,
             "row":{"fingerprint":"fingerprint", "reserved_evaluation_tokens":32800}}
    ctl.reconcile("attempt", {"name":"node-one"}, claim, {"state":"failed"})
    assert claim["status"] == "not_started"
    retry = ctl.state["retries"]["fingerprint"]
    assert retry["reserved_evaluation_tokens"] == 32832
    assert retry["unstarted_on"] == ["node-one"]
    assert retry["preparation_already_reserved"]


def test_temporary_capacity_or_an_unsettled_first_update_keeps_the_allocations():
    if not shutil.which("maflow") and not (Path.home() / ".local/bin/maflow").is_file():
        pytest.skip("maflow account client")
    from tide_cloud_controller import idle_review_reason
    data = {"attempts":{}, "accepted":{}, "blocked":{}, "canaries":{"2237":{}, "2907":{}}}
    ready = {"ready":True, "terminal":False}
    forecast = {"phase":"stage-a-gate", "blocked":["conservative makespan estimate exceeds the absolute campaign deadline"]}
    assert idle_review_reason(forecast, data, {"one":ready, "two":{"ready":False,"terminal":False}}, 2) is None
    nodes = {"one":ready, "two":ready}
    assert idle_review_reason(forecast, data, nodes, 2) == "awaiting_scientific_review"
    stale = {"phase":"stage-b-capacity", "blocked":["awaiting measured idle chip for the Stage B capacity probe: m"]}
    assert idle_review_reason(stale, data, nodes, 2) is None
    data["canaries"] = {}
    claim = {"fingerprint":"m", "row":{"legacy_id":2237}, "status":"failed", "needs_relay":True}
    data["attempts"]["one"] = claim
    assert idle_review_reason(forecast, data, nodes, 2) is None
    claim["needs_relay"] = False
    data["accepted"]["m"] = {"runq_status":"failed"}
    assert idle_review_reason(forecast, data, nodes, 2) == "canary_needs_review"


def test_live_heartbeat_reads_metadata_and_body_in_one_get():
    if not shutil.which("maflow") and not (Path.home() / ".local/bin/maflow").is_file():
        pytest.skip("maflow account client")
    from contextlib import contextmanager
    from io import BytesIO
    from tide_cloud_controller import snapshot
    body = b'{"at":30}'

    class UpdatingStatus:
        @contextmanager
        def response(self, method, key):
            assert (method, key) == ("GET", "heartbeat")
            yield BytesIO(body)

        def metadata(self, response):
            return {"size":len(body), "sha256":hashlib.sha256(body).hexdigest()}

    assert snapshot(UpdatingStatus(), "heartbeat") == {"at":30}


def test_waiting_npu_window_follows_physical_chip_map_and_cpu_follows_threads():
    if not shutil.which("maflow") and not (Path.home() / ".local/bin/maflow").is_file():
        pytest.skip("maflow account client")
    from tide_cloud_controller import queued_window
    worker = {"capacity":{"threads":40}, "cards":{"0":{"chips":{"0":{}, "1":{}}},
                                                  "1":{"chips":{"2":{}, "3":{}}}}}
    node = {"queue":{"workers":[worker]}}
    assert queued_window(node, "npu", 4) == 4
    assert queued_window(node, "cpu", 4) == 10
    assert queued_window(node, "cpu", 4, running_npu_threads=16) == 6
    assert queued_window(node, "cpu", 4, running_npu_threads=40) == 1


def test_restored_inputs_clear_only_their_node_error_even_with_a_full_queue(tmp_path, monkeypatch):
    if not shutil.which("maflow") and not (Path.home() / ".local/bin/maflow").is_file():
        pytest.skip("maflow account client")
    import tide_cloud_controller as controller
    ctl = controller.Controller.__new__(controller.Controller)
    ctl.area, ctl.spec, ctl.stage_futures = tmp_path, {}, {}
    ctl.state = {"errors":{"staging:one:fp":"timeout", "staging:two:fp":"timeout", "publish:fp":"timeout"}}
    monkeypatch.setattr(controller, "descriptor", lambda *_: ({}, {}))
    monkeypatch.setattr(controller, "objects_needed", lambda *_: {"a":(5, None), "b":(6, None)})
    rows = {"fp":{}}
    nodes = {"one":{"prefix":"job-1"}, "two":{"prefix":"job-2"}}
    for name in nodes:
        write(tmp_path / "staged" / name / "a", {"sha256":"a", "size":5, "key":"job-1/inputs/objects/a", "etag":"v1"})
        write(tmp_path / "staged" / name / "b", {"sha256":"b", "size":6, "key":"job-2/inputs/objects/b", "etag":"v2"})
    ctl.reconcile_staging_errors(rows, nodes)
    assert len(ctl.state["errors"]) == 3
    write(tmp_path / "staged/one/b", {"sha256":"b", "size":6, "key":"job-1/inputs/objects/b", "etag":"v3"})
    ctl.reconcile_staging_errors(rows, nodes)
    assert ctl.state["errors"] == {"staging:two:fp":"timeout", "publish:fp":"timeout"}


def test_output_hash_uses_a_pinned_cloud_body_with_a_local_fallback(monkeypatch):
    if not shutil.which("maflow") and not (Path.home() / ".local/bin/maflow").is_file():
        pytest.skip("maflow account client")
    import tide_cloud_submit
    from tide_cloud_results import readback
    digest = "a"*64
    node = {"id":"allocated-job"}

    class Store:
        def head(self, key):
            return {"etag":"v1", "version":"1", "size":70}

        def stream(self, key, *, expected, etag):
            assert expected == digest and etag == "v1"
            return self.head(key)

    def remote(node, store, key, checksum, size):
        return {"sha256":checksum, "size":size}
    monkeypatch.setattr(tide_cloud_submit, "verify_object", remote)
    assert readback(Store(), "key", digest, 70, node)["verified_on"] == "guiyang/allocated-job"
    def unavailable(*args):
        raise ConnectionError()
    monkeypatch.setattr(tide_cloud_submit, "verify_object", unavailable)
    assert readback(Store(), "key", digest, 70, node)["verified_on"] == "client"
