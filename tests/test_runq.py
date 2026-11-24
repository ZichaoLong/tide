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
