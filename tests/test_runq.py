import json
from pathlib import Path
import threading
import time

import pytest

from pilotlab.runq_campaign import device_cost
from pilotlab.runq_devices import admission, free_bytes, grant_for
from pilotlab.runq_place import place
from pilotlab.runq_store import connect, lookup, pin, put, submit, write


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
