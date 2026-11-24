from types import SimpleNamespace
import time

from pilotlab.prepare_resume3 import reference
from pilotlab.resource_placement3 import ResourcePlacement, split_calibrations
from pilotlab.round3 import TUNING_SEEDS
from pilotlab.scheduler3 import freeze_json, read
from pilotlab.sharing_policy3 import SharingPolicy
from pilotlab.stage_a_continued3 import MigrationCampaign
from test_pilot_scheduler_dynamic3 import add_task, observe, owner, write


def placement_for(supervisor):
    campaign = SimpleNamespace(output=supervisor.output, supervisor=supervisor,
                               migration={"completed": {}, "outstanding": {}})
    supervisor.resource_placement = ResourcePlacement(campaign)
    return supervisor.resource_placement


def register_pair(supervisor, placement, recipe="r0"):
    pair = []
    for seed in TUNING_SEEDS:
        task = add_task(supervisor, f"model-a1-{seed}-{recipe}", model_id="model",
                        phase="a1", recipe_id=recipe, batch_size=2)
        placement.prepare_task(task)
        placement.register(task, supervisor.state["tasks"][task["id"]])
        pair.append(task)
    return pair


def test_direct_dispatch_fills_both_pools_without_samples(owner):
    supervisor, _, key = owner
    supervisor.config.update(resource_dispatch=True, max_npu_devices=1, max_cpu_workers=2)
    placement = placement_for(supervisor)
    first = register_pair(supervisor, placement)
    second = register_pair(supervisor, placement, "r1")
    group = observe(supervisor, key, target=3)
    group["resources"] = {"device": {"observed_unix": time.time(), "used_bytes": 2**30, "total_bytes": 48*2**30}}
    group_path = supervisor.output / "groups" / key / "group.json"
    write(group_path, group)
    supervisor._group_observation_cache = {}
    supervisor.start_dynamic_ready_tasks(first + second, 4)
    assert [t["device"] for t in first] == ["cpu", "cpu"]
    assert [t["device"] for t in second] == ["npu:0", "npu:0"]
    assert all(supervisor.state["tasks"][t["id"]]["status"] == "running" for t in first + second)
    assert supervisor.state["dispatch_queue"]["unbound_tasks"] == 0
    # Reconstruct each member after an interruption between receipt and start.
    restored = ResourcePlacement(placement.campaign)
    for expected in first + second:
        task = {**expected, "device": "unbound"}
        restored.prepare_task(task)
        restored.register(task, supervisor.state["tasks"][task["id"]])
        assert task == expected


def test_fresh_shannon_seed_pair_uses_its_own_placement_and_source(owner):
    from pilotlab.shannon_recipes3 import cohort_subjects, new_config

    supervisor, _, _ = owner
    placement = placement_for(supervisor)
    old = {"allocation": "shannon", "shannon_support_policy": "positive-floor-v1"}
    assert new_config(old) == old
    model = {"id": "model", "model": {"family": "tide", "config": old}}
    revised, = cohort_subjects([model], {"policy": "weak-log-v1", "model_ids": ["model"]})
    campaign = object.__new__(MigrationCampaign)
    campaign.supervisor = supervisor
    supervisor.worker_runtimes = {}
    campaign.resource_placement = placement
    campaign.fresh_ids = {"model"}
    campaign.fresh_identity = {"source": "qualified"}
    campaign.fresh_runtime = {"source": "/qualified"}
    campaign.identity = {"source": "prior"}
    campaign.protocol = dict(precision="fp32", deterministic=False, sequence_length=128, cpu_threads=4)
    campaign.resources = {}
    pair = [campaign.task(revised, "a1", seed, recipe={"recipe_id": "r0"}, suffix="-r0")
            for seed in TUNING_SEEDS]
    for task in pair:
        assert campaign.resolve_task(task) is None and task["device"] == "unbound"
        assert task["source_identity"] == campaign.fresh_identity
        assert task["model"]["config"]["shannon_support_policy"] == "weak-log-v1"
        assert placement.groups[task["id"]] == [t["id"] for t in pair]
        assert supervisor.worker_runtimes[task["id"]] == campaign.fresh_runtime
    assert model["model"]["config"] == old


def test_unplaced_pair_waits_for_real_resources(owner):
    supervisor, _, key = owner
    supervisor.config.update(resource_dispatch=True, max_npu_devices=1, max_cpu_workers=0)
    placement = placement_for(supervisor)
    pair = register_pair(supervisor, placement)
    group = observe(supervisor, key, target=16)
    group["resources"] = {"device": {"observed_unix": time.time(), "used_bytes": 48*2**30, "total_bytes": 48*2**30}}
    write(supervisor.output / "groups" / key / "group.json", group)
    supervisor._group_observation_cache = {}
    supervisor.start_dynamic_ready_tasks(pair, 4)
    assert all(t["device"] == "unbound" for t in pair)
    assert not placement.decision_path([t["id"] for t in pair]).exists()


def test_existing_checkpoint_pins_both_seeds_without_reassigning_its_receipt(owner):
    supervisor, _, _ = owner
    placement = placement_for(supervisor)
    ids = [f"m-a1-{seed}-r0" for seed in TUNING_SEEDS]
    row = {"disposition": "signal", "receipt": {"path": "existing checkpoint receipt"},
           "task": reference(freeze_json(supervisor.output / "old.json", {"device": "npu:0"}))}
    placement.campaign.migration["outstanding"][ids[0]] = row
    for tid in ids:
        task = {"model_id": "m", "id": tid, "phase": "a1", "recipe_id": "r0", "device": "cpu"}
        placement.prepare_task(task)
        assert task["device"] == "npu:0"
    assert not placement.reassigned


def test_direct_profiles_and_formal_retirement_preserve_training(tmp_path):
    campaign = SimpleNamespace(plan={"config": {"resource_dispatch": True},
                                     "stage_a_cpu_placement": {"unprofiled_batches": [2, 4]}},
                               migration={"completed": {}}, output=tmp_path,
                               resources={"known": {"batches": [2, 64]}})
    assert list(MigrationCampaign.profile_flow(campaign, [{"id": "known"}, {"id": "new"}])) == []
    assert read(tmp_path / "model-resources.json")["known"] == {"batches": [2, 64]}
    assert campaign.resources["new"]["batches"] == [2, 4]
    pending = {}
    for tid, phase, purpose in (("probe", "preflight", "capacity_calibration"), ("training", "a1", None)):
        task = {"id": tid, "phase": phase, "purpose": purpose}
        pending[tid] = {"task": reference(freeze_json(tmp_path / (tid + ".json"), task))}
    assert set(split_calibrations(pending)) == {"probe"}
    assert set(pending) == {"training"}


def test_card_admits_to_resource_limit_immediately():
    policy = SharingPolicy({"resource_dispatch": True}, 16)
    policy.resource_wait("hbm_reservations_full", {"first"})
    policy.observe({}, [])
    policy.backlog(True, False)
    assert policy.snapshot()["admission_limit"] == 16
    assert policy.take_events() == []


def test_large_npu_profile_runs_on_cpu_under_the_campaign_rss_limit(owner):
    supervisor, _, _ = owner
    supervisor.config.update(resource_dispatch=True, max_rss_bytes=10**12)
    task = add_task(supervisor, "large-init", device="cpu", phase="init", batch_size=128,
                    reserved_hbm_bytes=36*2**30, reserved_rss_bytes=4*2**30)
    group = supervisor.start_group([task], "cpu")
    row = supervisor.state["tasks"][task["id"]]
    manifest = read(supervisor.state["shared_groups"][group]["manifest_path"])
    assert manifest["max_rss_bytes"] == 10**12
    assert manifest["workers"][0]["reserved_rss_bytes"] == 40*2**30
    assert supervisor.worker_reservations(row)["reserved_rss_bytes"] == 40*2**30
    assert supervisor.worker_reservations({**task, "device": "npu:0"})["reserved_rss_bytes"] == 4*2**30


def test_both_pools_fit_when_large_pinned_cpu_work_would_consume_the_host_budget(owner):
    supervisor, parent, key = owner
    active_rss = supervisor.state["tasks"][parent["id"]]["reserved_rss_bytes"]
    supervisor.config.update(resource_dispatch=True, max_npu_devices=1, max_cpu_workers=2,
                             max_rss_bytes=active_rss+5*2**30)
    placement = placement_for(supervisor)
    pair = register_pair(supervisor, placement)
    large = add_task(supervisor, "large-cpu", device="cpu", reserved_hbm_bytes=3*2**30)
    fixed = add_task(supervisor, "fixed-npu")
    group = observe(supervisor, key, target=2)
    group["resources"] = {"device": {"observed_unix": time.time(), "used_bytes": 2**30, "total_bytes": 48*2**30}}
    write(supervisor.output / "groups" / key / "group.json", group)
    supervisor._group_observation_cache = {}
    supervisor.start_dynamic_ready_tasks([large, fixed, *pair], 5)
    rows = supervisor.state["tasks"]
    assert rows[large["id"]]["status"] == "pending"
    assert rows[fixed["id"]]["status"] == "running"
    assert all(rows[t["id"]]["status"] == "running" and t["device"] == "cpu" for t in pair)
