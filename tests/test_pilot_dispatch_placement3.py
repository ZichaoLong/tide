from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from pilotlab.dispatch_placement3 import BACKFILL_POLICY, DispatchPlacement, POLICY, UNBOUND, best_offer
from pilotlab.prepare_resume3 import reference
from pilotlab.scheduler3 import freeze_json, read
from test_pilot_scheduler_dynamic3 import add_task, observe, owner
from test_pilot_scheduler_shared3 import write


def test_new_worker_gets_its_own_calibration_id(tmp_path):
    mid = "model"
    def make_task(model, phase, seed, *, recipe, target, suffix):
        return {"id":mid+suffix, "source_identity":campaign.identity}
    old_source = {"release":"previous"}
    old = reference(freeze_json(tmp_path/"original.json", {"source_identity":old_source}))
    campaign = SimpleNamespace(task=make_task, protocol={"sequence_length":128}, identity=old_source,
        migration={"completed":{mid+"-dispatch-cpu":{"task":old}}, "new_source_digest":"a"*64})
    placement = object.__new__(DispatchPlacement)
    placement.campaign = campaign
    assert placement.spec({"id":mid}, "cpu")["id"] == mid+"-dispatch-cpu"
    campaign.identity = {"release":"current"}
    assert placement.spec({"id":mid}, "cpu")["id"] == mid+"-dispatch-cpu-"+"a"*16
    assert placement.spec({"id":mid}, "npu:0")["id"] == mid+"-dispatch-npu"


def test_contention_can_reverse_the_choice_on_the_same_clock():
    def forecast(npu_delay):
        return {"cpu": {"offer": {"limit": 6}, "expected_seconds": 17.9*60},
                "npu:0": {"offer": {"physical_id": 3}, "expected_seconds": 16.4*60+npu_delay}}
    assert best_offer({"m":forecast(0)}) == ("m", "npu:0")
    assert best_offer({"m":forecast(300)}) == ("m", "cpu")
    assert best_offer({"m":forecast(0)}) == ("m", "npu:0")
    assert best_offer({"small_gain":forecast(0), "large_gain":forecast(-45)}) == ("large_gain", "npu:0")
    busy = forecast(0)
    busy["npu:0"].update(offer=None, predictable_wait=True)
    assert best_offer({"m":busy}) is None
    assert best_offer({"m":busy}, backfill=True) == ("m", "cpu")
    cheaper_cpu = {**busy, "cpu":{**busy["cpu"], "expected_seconds":17*60}}
    assert best_offer({"m":busy, "smaller_gap":cheaper_cpu}, backfill=True) == ("smaller_gap","cpu")
    busy["npu:0"]["expected_seconds"] += 300
    assert best_offer({"m":busy}) == ("m","cpu")
    neither = forecast(0)
    for side in neither.values():
        side["offer"] = None
    assert best_offer({"m":neither}) is None
    assert best_offer({"m":neither}, backfill=True) is None


def placement_for(supervisor):
    output = supervisor.output
    campaign = SimpleNamespace(supervisor=supervisor, output=output, identity={"source": "frozen"},
        migration={"old_scheduler":reference(freeze_json(output/"previous.json", {"tasks":{}})),
                   "outstanding":{}, "completed":{}},
        resources={"model":{"device_by_batch":{"2":"npu:0"}}},
        plan={"stage_a_cpu_placement":{"calibration_models":[]}})
    placement = supervisor.dispatch_placement = DispatchPlacement(campaign)
    placement.sides["model"] = {dev:{"rate":rate, "overhead":0., "evidence":{"fixture":dev}}
                                for dev,rate in (("cpu", 100.), ("npu:0", 2000.))}
    seeds = (1, 2)
    recipes = [{"recipe_id":f"r{i}", "batch_size":2, "device":"npu:0"} for i in range(3)]
    placement.apply({"id":"model"}, recipes, seeds)
    tasks = [add_task(supervisor, f"model-a1-{seed}-{r['recipe_id']}", model_id="model",
                      device=r["device"], batch_size=2, recipe_id=r["recipe_id"]) for r in recipes for seed in seeds]
    for task in tasks:
        placement.register(task,supervisor.state["tasks"][task["id"]])
    return placement, tasks, recipes, seeds


@pytest.mark.parametrize("backend", ["cpu", "npu:0"])
def test_real_admission_leaves_intents_open_until_capacity_and_replays_group(owner, backend):
    s, parent, key = owner
    s.config.update(max_npu_devices=1, max_cpu_workers=0)
    placement, tasks, recipes, seeds = placement_for(s)
    placement.campaign.resources[parent["model_id"]] = {}
    group_path = Path(s.state["shared_groups"][key]["path"])/"group.json"
    # The owner is waiting for a card. CPU has no free lanes.
    group = observe(s,key,target=4)
    group.update(status="queued", allocation_acquired_unix=None)
    write(group_path, group)
    s.start_dynamic_ready_tasks(tasks, 8)
    assert all(t["device"] == UNBOUND for t in tasks)
    assert all(Path(s.state["tasks"][t["id"]]["task_path"]).name == "intent.json" for t in tasks)
    assert not placement.decision_path("model").exists()
    # The physical card has arrived, but its HBM is temporarily occupied.
    group.update(status="running", allocation_acquired_unix=time.time(),
        resources={"device":{"observed_unix":time.time(),"total_bytes":48*2**30,"used_bytes":48*2**30}})
    write(group_path,group)
    s._group_observation_cache = {}
    s.start_dynamic_ready_tasks(tasks,8)
    assert not placement.decision_path("model").exists()
    if backend == "npu:0":
        group["resources"]["device"].update(used_bytes=3*2**30, observed_unix=time.time())
        write(group_path,group)
        s._group_observation_cache = {}
    else:
        s.config["max_cpu_workers"] = 2
    s.start_dynamic_ready_tasks(tasks,8)
    decision = read(placement.decision_path("model"))
    assert decision["policy"] == POLICY and decision["device"] == backend
    assert len(decision["intents"]) == len(tasks) == 6
    assert all(t["device"] == backend and read(decision["intents"][t["id"]]["path"])["device"] == UNBOUND for t in tasks)
    assert sum(len(s.state["tasks"][t["id"]]["attempts"]) for t in tasks) == 1
    if backend == "npu:0":
        assert decision["forecasts"][backend]["offer"]["owner"] == key
    # Simulate the controller losing its resource projection after sealing the
    # receipt. Recipes used by the pipeline and its promoted children stay bound.
    placement.campaign.resources["model"] = {"device_by_batch":{"2":"npu:0"}}
    placement.campaign.plan["stage_a_cpu_placement"]["policy"] = BACKFILL_POLICY
    restarted = DispatchPlacement(placement.campaign)
    assert all(r["device"] == backend for r in restarted.apply({"id":"model"}, recipes, seeds))
    assert placement.campaign.resources["model"]["device_by_batch"]["2"] == backend


def test_backfill_only_registers_missing_probes_for_open_groups(owner):
    supervisor, _, _ = owner
    placement, tasks, _, _ = placement_for(supervisor)
    placement.policy = BACKFILL_POLICY
    placement.calibration_models = {"model", "future"}
    placement.campaign.subjects = [{"id":"model"}, {"id":"future"}]
    placement.sides["model"].pop("npu:0")
    placement.spec = lambda model, dev: {"id":model["id"]+"-dispatch-"+dev, "device":dev}
    placement.campaign.resolve_task = lambda task: None
    placement.campaign.request = lambda stage, tasks: (stage, tasks)
    wave = placement.supplemental_requests({})
    assert wave == [("dispatch-calibration", [{"id":"model-dispatch-npu:0", "device":"npu:0"}])]
    assert not placement.supplemental_requests({})
    assert set(placement.probes) == {"model-dispatch-npu:0"}


def test_capacity_ticket_has_no_task_or_preparation_debit(owner):
    s, _, _ = owner
    s.dispatch_placement = object()
    key = s.start_group([], "npu", dispatch_lease=True)
    entry = s.state["shared_groups"][key]
    assert entry["dynamic_admission"] and entry["task_ids"] == [] and not entry["preparation"]
    assert not read(entry["manifest_path"])["workers"]
    s.queue_admission(key, close_reason="dispatch_demand_satisfied")
    assert s.state["shared_groups"][key]["admission_commands"][0]["worker_ids"] == []


@pytest.mark.parametrize("device", ["cpu", "npu:0"])
def test_opposite_fixed_backlog_cannot_consume_last_shared_write_before_dispatch(owner, device):
    s, parent, key = owner
    s.config.update(max_npu_devices=1, max_cpu_workers=5)
    placement, tasks, _, _ = placement_for(s)
    placement.campaign.resources[parent["model_id"]] = {}
    # Exactly one write fits. The old CPU-first / NPU-first loops could
    # consume it before an available lane on the other pool was considered.
    s.config["max_inflight_checkpoint_bytes"] = s.state["tasks"][parent["id"]]["checkpoint_reservation_bytes"]+16*2**20
    other = "npu:0" if device == "cpu" else "cpu"
    placement.sides["model"][device]["rate"] = 2000.
    placement.sides["model"][other]["rate"] = 1.
    group = observe(s,key,target=4)
    group["resources"] = {"device":{"observed_unix":time.time(), "total_bytes":48*2**30, "used_bytes":3*2**30}}
    write(Path(s.state["shared_groups"][key]["path"])/"group.json",group)
    fixed = add_task(s, "fixed", device=other, batch_size=2)
    placement.register(fixed,s.state["tasks"]["fixed"])
    s.start_dynamic_ready_tasks([fixed,*tasks], 8)
    assert read(placement.decision_path("model"))["device"] == device
    assert s.state["tasks"]["fixed"]["status"] == "pending"
    assert sum(r.get("checkpoint_reservation_bytes",0) for r in s.state["tasks"].values()
               if r["status"]=="running") == s.config["max_inflight_checkpoint_bytes"]
