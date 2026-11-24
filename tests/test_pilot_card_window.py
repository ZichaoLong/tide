import os
import signal
import subprocess
import sys

from pilotlab.runq_devices import process_unit
from pilotlab.runq_store import connect, put, setting, submit
from pilotlab.scheduler import proc_identity
from scripts.pilot_r3_card_window import close_refill, signal_ready


def test_cutoff_only_selects_new_automatic_workers_of_the_campaign(tmp_path):
    db = connect(tmp_path)
    put(db, "limits", {"cards": 2, "cpu": 40, "auto_card_ids": [3, 7],
                       "auto_share_grants": {"7": {"original": True}}})
    put(db, "window", {"started": 100, "deadline": 200})
    plan = {"queue": str(tmp_path), "extra_cards": [3], "owner": "round"}
    for i, (card, owner, at, target) in enumerate((
            (3, "round", 101, "auto"), (7, "round", 101, "auto"), (3, "other", 101, "auto"),
            (3, "round", 99, "auto"), (3, "round", 105, "npu:3"), (None, "round", 106, "auto"))):
        spec = {"owner": owner, "allowed": ["cpu", "npu"], "ram": {"npu": 1, "cpu": 1}, "hbm": 1}
        job = submit(db, str(i), spec, target)
        lease = None
        if card is not None:
            db.execute("INSERT OR IGNORE INTO leases(id,card,role,state,trigger_job,grant,created) "
                       "VALUES(?,?,'auto','held',?,'null',100)", (card, card, job))
            lease = card
        db.execute("UPDATE jobs SET state='running',backend=?,assigned=?,generation=1,lease=? WHERE id=?",
                   ("npu" if lease else "cpu", at, lease, job))
    cutoff = close_refill(db, plan, "window")
    assert [t["id"] for t in cutoff["targets"]] == [1]
    assert setting(db, "limits") == {"cards": 1, "cpu": 40, "auto_card_ids": [7],
                                     "auto_share_grants": {"7": {"original": True}}}
    assert close_refill(db, plan, "window") == cutoff


def test_checkpoint_signal_binds_pid_lifetime_unit_and_handler():
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        # Some ML interpreters omit Linux pidfd APIs. Exercise the interpreter
        # used by this host's release service, with the same test assertions.
        subprocess.run(["/usr/bin/python3", "-c",
                        "from tests.test_pilot_card_window import test_checkpoint_signal_binds_pid_lifetime_unit_and_handler as check;check()"],
                       check=True)
        return
    code = ("import signal,sys,time\n"
            "signal.signal(signal.SIGTERM,lambda *_:sys.exit(2))\n"
            "print('ready',flush=True)\n"
            "time.sleep(30)")
    with subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True) as proc:
        try:
            assert proc.stdout.readline().strip() == "ready"
            ticks = proc_identity(proc.pid)["start_ticks"]
            name = process_unit(os.getpid())
            assert not signal_ready({"pid": proc.pid, "ticks": ticks + 1}, name)
            assert not signal_ready({"pid": proc.pid, "ticks": ticks}, "another.service")
            assert signal_ready({"pid": proc.pid, "ticks": ticks}, name)
            assert proc.wait(timeout=5) == 2
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGKILL)
