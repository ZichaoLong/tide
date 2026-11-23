from types import SimpleNamespace

from pilotlab.cpu_placement3 import CpuRamp, RecipePlacement, completion_times


def test_turnaround_uses_concurrent_lanes_and_queue_wait():
    assert completion_times(2, [6, 2], [3], 2, 10) == 16
    assert completion_times(6, [], [], 6, 18) < completion_times(6, [6]*6, [], 6, 16)


def test_one_started_seed_locks_the_whole_batch(tmp_path):
    placement = object.__new__(RecipePlacement)
    placement.campaign = SimpleNamespace(migration={"completed":{}}, supervisor=SimpleNamespace(state={"tasks":{}}))
    path = tmp_path/"task.json"
    path.write_text("{}")
    placement.previous = {"tasks":{"one":{"status":"pending","attempts":[]}}}
    placement.outstanding = {"one":{"disposition":"not_started","task":{"path":str(path)}}}
    ids = {"one", "two"}
    assert placement.movable(ids)
    placement.previous["tasks"]["one"]["attempts"] = [{"path":"started"}]
    assert not placement.movable(ids)
    placement.outstanding.clear()
    placement.campaign.migration["completed"]["two"] = {}
    assert not placement.movable(ids)


def test_ramp_compares_same_workers_and_honors_host_saturation(monkeypatch):
    ticks = iter([(1000, 500), (2000, 1300), (3000, 2100), (4000, 2100)])
    monkeypatch.setattr("pilotlab.cpu_placement3.host_cpu_ticks", lambda affinity:next(ticks))
    ramp = CpuRamp({"max_cpu_workers":40,"host_cpu_affinity":list(range(160))})
    state = {}
    def workers(now, progress):
        return [{"device":"npu:0","shared_group":"g","heartbeat":{"task_id":str(i),"pid":i+1,
                 "status":"training","updated_unix":now,"tokens":progress}} for i in range(3)]
    assert ramp.admission(workers(10, 0), 0, 10, state) == 6
    assert ramp.admission(workers(55, 4500), 6, 55, state) == 12
    assert ramp.admission(workers(100, 6750), 12, 100, state) == 6
    assert state["cpu_admission"]["median_npu_rate_ratio"] == .5
    assert ramp.admission(workers(145, 11250), 6, 145, state) == 1
