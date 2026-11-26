import json
from pathlib import Path
import threading
import time

import pytest

from pilotlab.io import atomic_json, sha256
from pilotlab.hbm_profile3 import GIB, HbmProfile
from pilotlab.round3 import Campaign
from pilotlab.runq_campaign import QueueSupervisor, device_cost, hbm_shape
from pilotlab.runq_devices import admission, free_bytes, grant_for
from pilotlab.runq_place import place
from pilotlab.runq_store import connect, lookup, pin, put, submit, unit, write


def spec(size=20, allowed=("npu",), binding=None):
    return dict(hbm=size, ram={b:10 for b in allowed}, allowed=list(allowed), binding=binding)


def samples(residents=(), used=0, count=2):
    now = time.time()
    cards = {str(i):dict(card=i,health="OK",used=used,total=100,core=0,processes=list(residents) if i==1 else [])
             for i in range(count)}
    return [dict(at=now-2,boot="boot",cards=cards),dict(at=now,boot="boot",cards=cards)]


def held(db, card, job, role="auto"):
    return db.execute("""INSERT INTO leases(card,role,state,trigger_job,grant,created,acquired)
        VALUES(?,?,'held',?,'null',?,?)""",(card,role,job,time.time()-1,time.time())).lastrowid


def run(db, job, lease):
    db.execute("UPDATE jobs SET state='running',backend='npu',lease=?,generation=1 WHERE id=?",(lease,job))


@pytest.mark.parametrize("frozen_name", ["intent.json", "task.json"])
def test_pipeline_handoff_rejoins_queue_before_opening_new_models(tmp_path, frozen_name):
    supervisor = QueueSupervisor.__new__(QueueSupervisor)
    supervisor.db, supervisor.owner, supervisor.output = connect(tmp_path), str(tmp_path), tmp_path
    supervisor.state = {"tasks": {
        "852": {"model_id": "k4", "status": "running", "runq_id": 1},
        "853": {"model_id": "k4", "status": "running", "runq_id": 2},
        "854": {"model_id": "k4", "status": "pending", "runq_id": 3},
        "settled": {"model_id": "fresh-1", "status": "completed", "runq_id": 9},
        "import": {"model_id": "fresh-2", "status": "pending"}}}
    groups = {"k4": ["852", "853", "854"], "k2": ["960", "961", "962"],
              "fresh-1": ["new-1"], "fresh-2": ["new-2"]}
    for tid in groups["k4"] + groups["k2"]:
        job = submit(supervisor.db, tid, {**spec(), "owner": supervisor.owner, "task_id": tid})
        state = "completed" if tid in ("852", "853") else "reserved"
        supervisor.db.execute("UPDATE jobs SET state=? WHERE id=?", (state, job))
        if tid in groups["k2"]:
            # A crash can follow SQLite submission before scheduler.json is saved.
            atomic_json(tmp_path / "tasks" / tid / frozen_name, {"id": tid, "model_id": "k2"})
    submit(supervisor.db, "foreign", {**spec(), "owner": "another-round", "task_id": "other"})
    submit(supervisor.db, "review", {**spec(), "owner": supervisor.owner, "task_id": "old-subject-probe",
                                    "purpose": "deferred_source_review", "preparation": True})
    campaign = Campaign.__new__(Campaign)
    campaign.supervisor = supervisor
    campaign.plan = {"config": {"runq_root": str(tmp_path), "max_workers": 4}, "pipeline_model_limit": 1}
    models = [{"id": mid} for mid in ("fresh-1", "fresh-2", "k4", "k2")]

    def factory(models):
        ids = groups[models[0]["id"]]
        answers = yield [{"stage": "init", "tasks": [{"id": tid} for tid in ids]}]
        assert set(answers) == set(ids)

    def check_restart(stage, produce):
        def offered(verified):
            update = produce(dict.fromkeys(verified, "verified"))
            return [t["id"] for wave in update["requests"] for t in wave["tasks"]], update["done"]

        assert offered([]) == (groups["k4"] + groups["k2"], False)
        assert offered(groups["k4"]) == ([], False)
        verified = groups["k4"] + groups["k2"]
        assert offered(verified) == (["new-1"], False)
        assert offered(verified + ["new-1"]) == (["new-2"], False)
        assert offered(verified + ["new-1", "new-2"]) == ([], True)

    supervisor.run_dynamic = check_restart
    campaign.run_model_flows(models, factory, stage="stage-a-training-pipeline")


def test_handoff_rejoins_model_when_its_last_worker_finished_offline(tmp_path):
    supervisor = QueueSupervisor.__new__(QueueSupervisor)
    supervisor.db, supervisor.owner, supervisor.output = connect(tmp_path), str(tmp_path), tmp_path
    job = submit(supervisor.db, "last", {**spec(), "owner": supervisor.owner, "task_id": "last"})
    supervisor.db.execute("UPDATE jobs SET state='completed' WHERE id=?", (job,))
    record = {"model_id": "m", "status": "running", "runq_id": job}
    supervisor.state = {"tasks": {"last": record}}
    assert supervisor.inflight_model_ids() == {"m"}
    record["status"] = "completed"
    assert not supervisor.inflight_model_ids()


def test_campaign_reserves_measured_shape_on_registration(tmp_path):
    s = QueueSupervisor.__new__(QueueSupervisor)
    s.db, s.owner, s.registered = connect(tmp_path), str(tmp_path), {}
    s.config = {"run_id": "round", "deadline_unix": time.time() + 300}
    s.worker_reservations = lambda task: {"reserved_rss_bytes": 11}
    base = dict(model={"family": "timed"}, source_identity={"code": "v1"}, sequence_length=128,
                stream_chunks=4, batch_size=2, precision="fp32", device="unbound", reserved_hbm_bytes=8)
    put(s.db, "hbm_floors." + s.owner, {hbm_shape(base): 14})
    for i, (change, want) in enumerate([({}, 14), ({"seed": 27, "phase": "a2"}, 14),
            ({"reserved_hbm_bytes": 25}, 25), ({"batch_size": 4}, 8),
            ({"source_identity": {"code": "v2"}}, 8), ({"device": "cpu"}, 8)]):
        task = {**base, "id": str(i), **change}
        job = s.register_job(task, {"attempts": []})
        assert json.loads(lookup(s.db, job)["spec"])["hbm"] == want
        assert s.register_job(task, {"attempts": []}) == job
        assert task["reserved_hbm_bytes"] == change.get("reserved_hbm_bytes", 8)


@pytest.mark.parametrize("steps, saved", [(0, 0), (1, 1), (162, 160)])
def test_reviewed_first_npu_oom_retries_once_with_its_original_progress(tmp_path, steps, saved):
    s = QueueSupervisor.__new__(QueueSupervisor)
    s.db, s.owner = connect(tmp_path), str(tmp_path)
    source = {"code": "v1"}
    task = {"id": "timed", "model": {"family": "timed"}, "source_identity": source, "device": "npu:0",
            "batch_size": 2, "sequence_length": 128, "reserved_hbm_bytes": 8}
    path = tmp_path / "task.json"
    atomic_json(path, task)
    spec0 = {**spec(size=8), "owner": s.owner, "task_id": task["id"]}
    job = submit(s.db, "round/timed/0", spec0)
    s.db.execute("UPDATE jobs SET state='failed',backend='npu',rc=1,pid=2147483647,ticks=1 WHERE id=?", (job,))
    attempt = {"runq_id": job, "path": str(tmp_path), "status": "failed"}
    checkpoint = None
    if steps:
        cp = tmp_path / "step.pt"
        cp.write_bytes(b"preserved worker state")
        checkpoint = {"path": str(cp), "sha256": sha256(cp), "steps": saved, "tokens": 256 * saved}
        atomic_json(tmp_path / "checkpoint.json", {"current": checkpoint})
    result = {"id": "timed", "device": "npu:0", "source_identity": source, "task_sha256": sha256(path),
              "status": "failed", "failure_class": "capacity_error", "error": "NPU out of memory. Tried to allocate",
              "checkpoint": checkpoint, "training": {"steps": steps, "tokens": 256 * steps}}
    output = tmp_path / "result.json"
    atomic_json(output, result)
    row = dict(status="failed", runq_id=job, attempts=[attempt], task_path=str(path), result_path=str(output),
               result_sha256=sha256(output), evaluation_reserved_tokens=32800)
    s.capacity_retry_jobs = set()
    assert not s.retry_transient(task, row, result)
    s.capacity_retry_jobs.add(job)
    assert not s.retry_transient(task, row, result)
    put(s.db, "hbm_floors." + s.owner, {hbm_shape(task): 14})
    assert not s.retry_transient(task, row, {**result, "failure_class": "numerical_error"})
    if checkpoint:
        cp.write_bytes(b"changed")
        assert not s.retry_transient(task, row, result)
        cp.write_bytes(b"preserved worker state")
    assert s.retry_transient(task, row, result) and row["status"] == "pending"
    assert row["evaluation_reserved_tokens"] == 32800 and row["attempts"] == [attempt]
    assert row["capacity_retry"]["result"] == {"path": str(output), "sha256": sha256(output)}
    assert row["capacity_retry"]["replay_steps"] == steps - saved
    assert row["capacity_retry"]["replay_tokens"] == 256 * (steps - saved)
    assert row.get("retry_checkpoint_sha256") == (checkpoint["sha256"] if steps else None)
    row.update(status="failed", attempts=[attempt, {"runq_id": job + 1}])
    assert not s.retry_transient(task, row, result)


def test_shape_trial_waits_only_for_unmeasured_npu_followers(tmp_path):
    s = QueueSupervisor.__new__(QueueSupervisor)
    s.db, s.owner, s.root, s.registered = connect(tmp_path), str(tmp_path), tmp_path, {}
    s.config = {"run_id": "round", "deadline_unix": time.time() + 300}
    s.worker_reservations = lambda _: {"reserved_rss_bytes": 1}
    base = dict(model={"family":"example"}, source_identity={"source":"v1"}, model_id="m", batch_size=4)
    tasks = [{**base, "id": name, "device": device} for name, device in
             (("trial","npu:0"), ("follower","npu:0"), ("host","cpu"))]
    s.state = {"tasks": {t["id"]: {"status":"pending","attempts":[]} for t in tasks}}
    profile = HbmProfile(tmp_path)
    shape = hbm_shape(base)
    assert profile.add_trial(tasks[0], 29*GIB) == shape
    s.shape_stats = profile.summaries()
    s.start_ready_tasks(tasks, 3)
    assert "trial" in s.state["tasks"]["follower"]["queue_wait"] and "runq_id" not in s.state["tasks"]["follower"]
    assert s.state["tasks"]["host"]["runq_id"]
    trial = s.state["tasks"]["trial"]["runq_id"]
    assert json.loads(lookup(s.db, trial)["spec"])["hbm"] == 44*GIB
    result = dict(id="trial", source_identity=base["source_identity"], device="npu:0", task_sha256="task",
                  status="completed", device_memory={"npu_peak_reserved_bytes":20*GIB})
    with profile.db:
        profile.record_result(tasks[0], "task", tmp_path/"trial/result.json", result, "result")
    s.shape_stats = profile.summaries()
    s.start_ready_tasks(tasks, 3)
    follower = s.state["tasks"]["follower"]["runq_id"]
    assert json.loads(lookup(s.db, follower)["spec"])["hbm"] == 31*GIB


def test_matched_driver_high_water_reprices_without_restarting_and_admission_recovers(tmp_path):
    s = QueueSupervisor.__new__(QueueSupervisor)
    s.db, s.root, s.owner = connect(tmp_path), tmp_path, str(tmp_path)
    s.hbm_profile, s.hbm_jobs, s.last_hbm_sample = HbmProfile(tmp_path), {}, None
    task = dict(id="tid", model_id="m", model={"family":"tide"}, source_identity={"src":"v1"}, batch_size=2)
    task_path = tmp_path / "task.json"
    atomic_json(task_path, task)
    job = submit(s.db, "active", {**spec(8*GIB), "owner":s.owner, "task_id":"tid"})
    lease = held(s.db, 1, job)
    run(s.db, job, lease)
    s.db.execute("UPDATE jobs SET pid=543,ticks=72 WHERE id=?", (job,))
    s.state = {"tasks":{"tid":{"task_path":str(task_path),"attempts":[{"path":str(tmp_path),"runq_id":job}]}}}
    name = unit(tmp_path, "job", job, 1)
    procs = [dict(pid=543, unit=name, identity={"start_ticks":72}, bytes=9*GIB),
             dict(pid=544, unit="other", identity={"start_ticks":72}, bytes=14*GIB)]
    observations = samples(procs, used=23*GIB)
    for sample in observations:
        sample["cards"]["1"]["total"] = 27*GIB
    put(s.db, "samples", observations)
    s.refresh_hbm()
    row = lookup(s.db, job)
    assert (row["pid"], row["generation"], json.loads(row["spec"])["hbm"]) == (543, 1, 14*GIB)
    usage = s.hbm_profile.db.execute("SELECT driver,before_first_update FROM attempts").fetchone()
    assert tuple(usage) == (9*GIB, 9*GIB)
    occupants = [{"unit":name, "data":json.loads(row["spec"])}]
    grants = [grant_for(observations[-1], "npu:1", {name}, time.time()+300)]
    assert admission(observations, 1, occupants, GIB, grants) == "waiting for peak HBM"
    s.hbm_profile.set_policy(1., 0)
    s.db.execute("UPDATE jobs SET first_step=1 WHERE id=?", (job,))
    later = samples(procs, used=23*GIB)
    for sample in later:
        sample["cards"]["1"]["total"] = 27*GIB
    procs[0]["bytes"] = 8*GIB
    put(s.db, "samples", later)
    s.refresh_hbm()
    row = lookup(s.db,job)
    assert json.loads(row["spec"])["hbm"] == 9*GIB and json.loads(row["original"])[0]["hbm"] == 8*GIB
    assert admission(later, 1, [{"unit":name, "data":json.loads(row["spec"])}], GIB, grants) is None


def test_cpu_idempotent_manual_extra_and_refill(tmp_path):
    db=connect(tmp_path)
    put(db,"limits",dict(cards=0,cpu=1))
    cpu=spec(allowed=("cpu",))
    first, second = [submit(db,k,cpu) for k in ("first","second")]
    assert submit(db,"first",cpu)==first
    with pytest.raises(ValueError):
        submit(db,"first",cpu,"cpu")
    extra=submit(db,"extra",cpu,"cpu")
    place(db,tmp_path,[])
    assert [lookup(db,i)["state"] for i in (first,second,extra)]==["reserved","queued","reserved"]
    with pytest.raises(ValueError):
        pin(db,first,"cpu")
    db.execute("UPDATE jobs SET state='completed' WHERE id=?",(first,))
    place(db,tmp_path,[])
    assert lookup(db,second)["state"]=="reserved"


def test_auto_card_filter_replaces_one_lease_and_never_reclaims_it(tmp_path):
    db = connect(tmp_path)
    put(db, "limits", {"cards": 2, "cpu": 40, "auto_card_ids": [1, 3]})
    old = submit(db, "old", spec())
    retained = submit(db, "retained", spec())
    moved = submit(db, "moved", spec(), "npu:1")
    old_lease = held(db, 0, old)
    retained_lease = held(db, 3, retained)
    run(db, old, old_lease)
    run(db, retained, retained_lease)
    snap = samples(count=4)
    place(db, tmp_path, snap)
    assert dict(db.execute("SELECT card,role FROM leases")) == {0: "drain", 1: "manual", 3: "auto"}
    assert lookup(db, old)["state"] == "running"
    db.execute("UPDATE jobs SET state='completed' WHERE id=?", (old,))
    db.execute("UPDATE leases SET state='released' WHERE id=?", (old_lease,))
    db.execute("UPDATE leases SET state='held',acquired=? WHERE card=1", (time.time(),))
    extra = submit(db, "extra", spec())
    place(db, tmp_path, snap)
    assert lookup(db, moved)["state"] == "reserved"
    assert lookup(db, extra)["lease"] == retained_lease
    assert {row[0] for row in db.execute("SELECT card FROM leases WHERE state!='released'")} == {1, 3}


def test_targeted_priority_and_atomic_reservations(tmp_path):
    db=connect(tmp_path)
    put(db,"limits",dict(cards=2,cpu=0))
    auto=submit(db,"auto",spec(40))
    manual=submit(db,"manual",spec(70),"npu:0")
    old=submit(db,"old",spec(40))
    first,second=held(db,0,old),held(db,1,auto)
    run(db,old,first)
    sample=samples()
    place(db,tmp_path,sample)
    assert lookup(db,manual)["state"]=="queued"
    assert lookup(db,auto)["lease"]==second
    db.execute("UPDATE jobs SET state='completed' WHERE id=?",(old,))
    place(db,tmp_path,sample)
    assert lookup(db,manual)["lease"]==first
    for i in range(6):
        submit(db,f"competing-{i}",spec(25))
    failures=[]
    def race():
        other=connect(tmp_path)
        try:
            place(other,tmp_path,sample)
        except Exception as error:
            failures.append(error)
        finally:
            other.close()
    threads=[threading.Thread(target=race) for _ in range(3)]
    for thread in threads:thread.start()
    for thread in threads:thread.join()
    assert not failures
    for lease in (first,second):
        assert sum(json.loads(r[0])["hbm"] for r in db.execute(
            "SELECT spec FROM jobs WHERE lease=? AND state='reserved'",(lease,)))<=100


@pytest.mark.parametrize("lend",[False,True])
def test_extra_shared_card_and_borrower_drain(tmp_path,lend):
    db=connect(tmp_path)
    put(db,"limits",dict(cards=1,cpu=0))
    p=dict(pid=44,identity=dict(pid=44,uid=123,start_ticks=456),unit=None,bytes=10)
    sample=samples([p],used=10)
    permission=grant_for(sample[-1],"npu:1",set(),time.time()+60)
    main=submit(db,"main",spec(87))
    primary=submit(db,"primary",spec(20),"npu:1",lend,permission)
    borrower=submit(db,"borrower",spec())
    regular=held(db,0,main)
    extra=held(db,1,primary,"manual")
    run(db,main,regular)
    run(db,primary,extra)
    place(db,tmp_path,sample)
    assert (lookup(db,borrower)["lease"]==extra) is lend
    if lend:
        run(db,borrower,extra)
    db.execute("UPDATE jobs SET state='completed' WHERE id=?",(primary,))
    another=submit(db,"new",spec())
    place(db,tmp_path,sample)
    assert lookup(db,another)["state"]=="queued"
    if lend:
        assert lookup(db,borrower)["state"]=="running"
        assert db.execute("SELECT role FROM leases WHERE id=?",(extra,)).fetchone()[0]=="drain"


def test_identity_change_and_peak_bytes():
    p=dict(pid=44,identity=dict(pid=44,uid=3,start_ticks=8),unit=None,bytes=10)
    current=samples([p],used=25)
    grant=grant_for(current[-1],"npu:1",set(),time.time()+60)
    assert admission(current,1,[],70,[grant]) is None
    assert admission(current,1,[],76,[grant])
    changed=dict(p,identity=dict(p["identity"],start_ticks=9))
    assert admission([current[0],samples([changed],used=25)[-1]],1,[],70,[grant])
    assert admission(current,1,[],70,[dict(grant,until=time.time()-1)])
    card=dict(current[-1]["cards"]["1"],processes=[p,dict(p,unit="our",bytes=12)])
    assert free_bytes(card,[dict(unit="our",data=spec(30))])==100-25-(30-12)


def test_auto_share_fills_and_refills_without_a_manual_job(tmp_path):
    from pilotlab.runq import main
    from pilotlab.runq_runtime import allowed

    db = connect(tmp_path)
    resident = dict(pid=44, identity=dict(pid=44, uid=3, start_ticks=8), unit="other", bytes=25)
    snap = samples([resident], used=25)
    put(db, "limits", dict(cards=1, cpu=0, auto_card_ids=[1]))
    put(db, "samples", snap)
    args = ["--root", str(tmp_path), "auto-share", "1"]
    main([*args, "--until-unix", str(int(time.time())+60)])
    jobs = [submit(db, name, spec(35)) for name in ("first", "second", "next")]
    place(db, tmp_path, snap)
    lease = db.execute("SELECT * FROM leases").fetchone()
    assert lease["role"] == "auto" and lease["card"] == 1
    assert json.loads(lease["grant"])["residents"] == [resident["identity"]]
    db.execute("UPDATE leases SET state='held',acquired=? WHERE id=?", (time.time(), lease["id"]))
    place(db, tmp_path, snap)
    assert [lookup(db, j)["state"] for j in jobs] == ["reserved","reserved","queued"]
    run(db, jobs[1], lease["id"])
    db.execute("UPDATE jobs SET state='completed' WHERE id=?", (jobs[0],))
    place(db, tmp_path, snap)
    job = lookup(db, jobs[2])
    assert job["state"] == "reserved" and not job["manual"]
    assert not allowed(db, tmp_path, job, lease)
    changed = dict(resident, identity=dict(resident["identity"], start_ticks=9))
    put(db, "samples", samples([changed], used=25))
    assert allowed(db, tmp_path, job, lease)
    put(db, "samples", snap)
    main([*args, "--clear"])
    assert allowed(db, tmp_path, job, lease)
    assert lookup(db, jobs[1])["state"] == "running"


def test_binding_survives_pin_and_restart(tmp_path):
    db=connect(tmp_path)
    template=spec(20,("npu","cpu"),"seeds")
    first,second=[submit(db,k,template) for k in ("seed1","seed2")]
    pin(db,second,"cpu")
    with pytest.raises(ValueError):
        pin(db,first,"npu:1")
    put(db,"limits",dict(cards=1,cpu=1))
    place(db,tmp_path,[])
    other=connect(tmp_path)
    assert [lookup(other,i)["backend"] for i in (first,second)]==["cpu","cpu"]


def test_physical_time_on_mixed_lease(tmp_path):
    db=connect(tmp_path)
    a,b,c=[submit(db,k,dict(spec(),owner=owner)) for k,owner in (("a","round"),("b","round"),("c","other"))]
    lease=held(db,0,a)
    db.execute("UPDATE leases SET acquired=10,released=40,state='released' WHERE id=?",(lease,))
    for key,begin,end in ((a,11,20),(b,17,31),(c,29,39)):
        db.execute("UPDATE jobs SET lease=?,assigned=?,finished=?,state='completed' WHERE id=?",(lease,begin,end,key))
    assert device_cost(db,"round",42)==21
    assert device_cost(db,"other",42)==10


def test_eight_regular_cards_ninth_manual_and_promotion(tmp_path):
    db=connect(tmp_path)
    put(db,"limits",dict(cards=8,cpu=0))
    sample=samples(count=10)
    regular=[]
    for card in range(8):
        job=submit(db,f"ordinary-{card}",spec(95))
        lease=held(db,card,job)
        run(db,job,lease)
        regular.append(lease)
    extra=submit(db,"extra",spec(30),"npu:9")
    borrowed=submit(db,"borrowed",spec(40))
    place(db,tmp_path,sample)
    extra_lease=db.execute("SELECT id FROM leases WHERE card=9").fetchone()[0]
    assert db.execute("SELECT COUNT(*) FROM leases").fetchone()[0]==9
    assert lookup(db,borrowed)["state"]=="queued"
    db.execute("UPDATE leases SET state='held',acquired=? WHERE id=?",(time.time(),extra_lease))
    place(db,tmp_path,sample)
    run(db,extra,extra_lease)
    place(db,tmp_path,sample)
    assert lookup(db,borrowed)["lease"]==extra_lease
    db.execute("UPDATE jobs SET state='completed' WHERE id=?",(extra,))
    place(db,tmp_path,sample)
    assert db.execute("SELECT role FROM leases WHERE id=?",(extra_lease,)).fetchone()[0]=="drain"
    db.execute("UPDATE leases SET state='released' WHERE id=?",(regular[0],))
    db.execute("UPDATE jobs SET state='completed' WHERE lease=?",(regular[0],))
    place(db,tmp_path,sample)
    assert db.execute("SELECT role FROM leases WHERE id=?",(extra_lease,)).fetchone()[0]=="auto"


def test_opening_ticket_reserves_host_memory(tmp_path,monkeypatch):
    db=connect(tmp_path)
    put(db,"limits",dict(cards=2,cpu=40))
    npu=submit(db,"accelerator",spec())
    cpu=submit(db,"host",spec(allowed=("cpu",)))
    monkeypatch.setattr("pilotlab.runq_place.ram_available",lambda:15)
    place(db,tmp_path,samples())
    place(db,tmp_path,samples())
    assert db.execute("SELECT trigger_job FROM leases WHERE state='opening'").fetchone()[0]==npu
    assert lookup(db,cpu)["state"]=="queued"


def test_live_cpu_estimate_uses_real_step_and_frozen_task(tmp_path,monkeypatch):
    import os
    from pilotlab.runq_place import active_peak
    from pilotlab.scheduler import proc_identity
    gib=2**30
    task=tmp_path/"task.json"
    task.write_text(json.dumps(dict(id="worker",device="cpu",checkpoint_reservation_bytes=gib)))
    job=dict(pid=os.getpid(),backend="cpu",command=json.dumps(dict(argv=["worker","--task",str(task)])),
             data=dict(owner="round",task_id="worker",ram={"cpu":35*gib},hbm=30*gib))
    rss=proc_identity(os.getpid())["rss_bytes"]
    assert active_peak(job,rss)==35*gib
    job["first_step"]=time.time()
    assert 5*gib <= active_peak(job,rss) < 35*gib
    assert active_peak(job,37*gib)==37*gib


@pytest.mark.parametrize("tamper",[False,True])
def test_cpu_reservation_uses_only_bound_previous_trajectory(tmp_path,tamper):
    from pilotlab.prepare_resume3 import reference
    from pilotlab.runq_place import estimated_spec
    gib=2**30
    tid="continuing"
    original=dict(id=tid,device="cpu",batch_size=32,model={"family":"tide"},source_identity={"source":"same"})
    old=tmp_path/"previous.json"
    old.write_text(json.dumps(original))
    result={k:original[k] for k in ("id","device","source_identity")}
    result.update(task_sha256=reference(old)["sha256"],training=dict(segment_steps=30),device_memory=dict(cpu_rss_peak_bytes=3*gib))
    measured=tmp_path/"result.json"
    measured.write_text(json.dumps(result))
    receipt=tmp_path/"receipt.json"
    receipt.write_text(json.dumps(dict(original_task=reference(old),result=reference(measured),mode="signal",task_id=tid)))
    task=tmp_path/"tasks"/tid/"task.json"
    task.parent.mkdir(parents=True)
    task.write_text(json.dumps(dict(original,checkpoint_reservation_bytes=gib,stage_a_migration=reference(receipt))))
    if tamper:
        measured.write_text(json.dumps(dict(result,id="different")))
    requested=dict(allowed=["cpu"],ram={"cpu":30*gib},owner=str(tmp_path),task_id=tid)
    adjusted=estimated_spec(requested)
    assert adjusted["ram"]["cpu"]==(30*gib if tamper else 5*gib+3*gib//4)


@pytest.mark.parametrize("manual", [True, False])
def test_shared_cooperative_second_sample_checks_identity_and_hbm(monkeypatch, manual):
    from pilotlab import runq_devices
    from pilotlab.shared_npu3 import load_queue_helper
    helper=load_queue_helper()
    occupant={"uid":17,"pid":44,"start_ticks":10}
    monkeypatch.setattr(runq_devices,"resident_identity",lambda pid: dict(occupant,pid=pid))
    monkeypatch.setattr(runq_devices,"boot_id",lambda: "sample-boot")
    grant=dict(card=3,boot="sample-boot",until=time.time()+60,residents=[dict(occupant)])
    samples=runq_devices.install_predicate(helper,3,grant,32*2**20,manual)
    kwargs=dict(max_hbm_mib=4096,max_aicore_percent=5)
    def card(used=30):
        return helper.DeviceStatus(board_id=1,chip_id=1,physical_id=3,health="OK",
                                   aicore_percent=71,hbm_used_mib=used,hbm_total_mib=64,process_ids=(44,))
    assert card().is_free(**kwargs)
    assert card().is_free(**kwargs)
    assert len(samples)==2
    assert not card(used=33).is_free(**kwargs)
    occupant["start_ticks"]+=1
    assert not card().is_free(**kwargs)
