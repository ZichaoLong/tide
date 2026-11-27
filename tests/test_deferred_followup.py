from types import SimpleNamespace

import pytest

from pilotlab.deferred_campaign3 import DeferredCampaign, DeferredSupervisor, gate_observation
from pilotlab.deferred_review3 import disposition, evaluation_forecast, phase_counts
from pilotlab.round3 import TUNING_SEEDS
from pilotlab.runq_campaign import QueueSupervisor
from pilotlab.runq_store import connect, submit
from pilotlab.scheduler3 import Blocked, Supervisor
from pilotlab.source_migration3 import read_bound


def test_npu_gate_keeps_jobs_out_of_queue_until_the_bound_worker_starts(tmp_path, monkeypatch):
    db = connect(tmp_path)
    job = submit(db, "primary/task/0", dict(owner="primary", task_id="task", allowed=["npu"], hbm=10, ram={"npu": 8}))
    gate = dict(job=job, job_key="primary/task/0", owner="primary", task_id="task", minimum_generation=1)
    sup = DeferredSupervisor.__new__(DeferredSupervisor)
    sup.db, sup.output, sup.config = db, tmp_path, {"predecessor": gate}
    tasks = [dict(id=k, device=d) for k, d in (("a", "cpu"), ("b", "npu:0"))]
    sup.state = {"tasks": {t["id"]: {"status": "pending"} for t in tasks}}
    seen = []
    monkeypatch.setattr(QueueSupervisor, "start_ready_tasks", lambda self, tasks, slots: seen.append([t["id"] for t in tasks]))
    for state in ("queued", "reserved", "starting", "failed"):
        db.execute("UPDATE jobs SET state=? WHERE id=?", (state, job))
        sup.start_ready_tasks(tasks, 40)
        assert seen[-1] == ["a"] and "npu_release" not in sup.state
    with pytest.raises(Blocked, match="identity"):
        gate_observation(db, {**gate, "owner": "another-round"})
    lease = db.execute("INSERT INTO leases(card,role,state,trigger_job,grant,created,acquired) VALUES(9,'auto','held',?,'null',1,1)",
                       (job,)).lastrowid
    db.execute("UPDATE jobs SET state='completed',backend='npu',lease=?,generation=1,pid=20,ticks=30,started=2 WHERE id=?", (lease, job))
    sup.start_ready_tasks(tasks, 40)
    receipt = read_bound(sup.state["npu_release"])
    assert receipt["physical_card"] == 9 and receipt["predecessor"] == gate and seen[-1] == ["a", "b"]
    del sup.state["npu_release"]  # restart after receipt publication, before scheduler.json was saved
    sup.start_ready_tasks(tasks, 40)
    assert read_bound(sup.state["npu_release"]) == receipt


def test_carry_needs_both_seed_boundaries_and_every_decisive_sample():
    recipes = ["r2", "r1", "r5", "r4", "r3", "r0"]
    ranks = dict(a1=dict(stable=True, selected=recipes[:2], seed_orders=[recipes, recipes], order=recipes,
                         seed_boundary_margins=[0.04, 0.07]), a2=dict(stable=True, selected=["r2"]))
    s = {"next_review": "source_equivalence_on_winning_chain_and_competing_recipes"}
    exact = dict(outcome="sample_exact", equal={"activity": True}, delta={"loss": 0})
    obs = {f"{p}/{r}/{seed}": dict(phase=p, recipe=r, seed=seed, comparison=exact)
           for p, ids in (("a1", recipes), ("a2", recipes[:2]), ("a3", ["r2"])) for r in ids for seed in TUNING_SEEDS}
    def mode(missing=()):
        return disposition(s, ranks, obs, missing)[0]
    assert mode() == "carry_recipe"
    assert mode(["retired-parent"]) == "compare_finalists"
    key = f"a2/r2/{TUNING_SEEDS[0]}"
    obs["second-start"] = {**obs[key], "comparison": {**exact, "outcome": "sample_changed"}}
    assert mode() == "compare_finalists"
    del obs["second-start"], obs[key]
    assert mode() == "compare_finalists"
    ranks["a1"]["stable"] = False
    assert mode() == "six_recipe_selection"


def test_confirmation_envelope_and_old_budget_are_cumulative():
    work = phase_counts("confirm_historical_winner", "complete_a3_confirmation", ["r2"])
    assert work == dict(capacity=0, init=3, a1=2, a2=2, a3=2)
    assert evaluation_forecast(work) == 3 * 544 + 6 * 32800
    sup = Supervisor.__new__(Supervisor)
    sup.config = {"prior_evaluation_reservations": {"model": 2**20 - 32800}}
    sup.state = {"tasks": {"other": {"evaluation_budget_key": "model", "evaluation_reserved_tokens": 1}}}
    with pytest.raises(Blocked, match="evaluation_token_limit"):
        sup.reserve_task(dict(id="t", target_tokens=2**18, evaluation_budget_key="model", evaluation_token_limit=2**20), {})


def test_fresh_task_and_both_children_use_reviewed_source_and_policy(tmp_path):
    campaign = DeferredCampaign.__new__(DeferredCampaign)
    model = {"id": "m", "model": {"config": {"allocation": "shannon"}}}
    campaign.identity = {"source": "primary"}
    campaign.identities = {"weak_log": {"source": "weak"}}
    runtime = dict(source="/frozen/worker", foundation="/frozen/base", adapter={})
    campaign.runtimes = {"weak_log": runtime}
    campaign.rows = {"m": dict(target_arm="weak_log", model_policy={"shannon_support_policy": "weak-log-v1"},
                               batch_requests={}, unmeasured_npu_request=52*2**30, recipes=["r2"])}
    campaign.resources = {"m": {"batches": [2, 4]}}
    campaign.floors = {}
    campaign.supervisor = SimpleNamespace(worker_runtimes={})
    campaign.protocol = dict(precision="fp32", deterministic=False, sequence_length=128, cpu_threads=4)
    recipe, = campaign.recipes(model)
    assert recipe["recipe_id"] == "r2"
    task = campaign.task(model, "a1", 42, recipe=recipe, target=2**18, suffix="-r2")
    assert task["source_identity"] == campaign.identities["weak_log"]
    assert task["model"]["config"]["shannon_support_policy"] == "weak-log-v1"
    assert "shannon_support_policy" not in model["model"]["config"] and "resume_checkpoint" not in task
    assert task["reserved_hbm_bytes"] == 52*2**30
    assert campaign.supervisor.worker_runtimes == {f"m-{p}-42-r2": runtime for p in ("a1", "a2", "a3")}
