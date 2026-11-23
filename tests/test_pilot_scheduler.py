"""Bounded-controller tests use fake workers and never allocate an NPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from pilotlab import scheduler as s


FAKE_WORKER = r'''
import argparse,hashlib,importlib,importlib.util,json,os,pathlib,signal,subprocess,sys,time
p=argparse.ArgumentParser()
for n in ('task','data','output','deadline-unix','device'):
    p.add_argument('--'+n,required=True)
a=p.parse_args()
task=json.loads(pathlib.Path(a.task).read_text())
out=pathlib.Path(a.output)
started=time.time()
(out/'worker.json').write_text(json.dumps({'started_unix':started}))
if task.get('child'):
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    child=subprocess.Popen([sys.executable,'-c',
        'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)'])
    (out/'descendant.pid').write_text(str(child.pid))
if task.get('rss_child_bytes') or task.get('detached_child'):
    code = ('import os,pathlib,signal,time; '
            + ('pid=os.fork(); os._exit(0) if pid else None; os.setsid(); ' if task.get('detached_child') else '')
            + 'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
            + 'allocation=bytearray(' + str(task.get('rss_child_bytes',0)) + '); '
            + 'pathlib.Path(' + repr(str(out/'descendant.pid')) + ').write_text(str(os.getpid())); '
            + 'time.sleep(30)')
    child=subprocess.Popen([sys.executable,'-c',code],start_new_session=bool(task.get('detached_child')))
    if task.get('detached_child'):
        child.wait()
time.sleep(task.get('sleep',0.03))
if task.get('write_result',True):
    result={
        'status':task.get('status','completed'),'started_unix':started,
        'ended_unix':time.time(),'affinity':sorted(os.sched_getaffinity(0)),
        'device':a.device,'resume_from':task.get('resume_from'),
        'training':{'steps':2,'tokens':4}, 'validation_final':{'nll':task.get('fake_nll',2.0)},
        'recovery':{'status':'within_tolerance'}}
    if task.get('import_probe'):
        def imported(names):
            return {name:{'marker':importlib.import_module(name).MARKER,
                          'file':importlib.import_module(name).__file__} for name in names}
        child_code = ("import importlib,importlib.util,json,os,sys; "
                      "names=json.loads(sys.argv[1]); "
                      "print(json.dumps({'modules':{name:{'marker':importlib.import_module(name).MARKER,"
                      "'file':importlib.import_module(name).__file__} for name in names},"
                      "'pythonpath':os.environ.get('PYTHONPATH'),"
                      "'untrusted_visible':importlib.util.find_spec('untrusted_only') is not None}))")
        result['import_probe'] = {'modules':imported(task['import_probe']),
                                  'pythonpath':os.environ.get('PYTHONPATH'),
                                  'untrusted_visible':importlib.util.find_spec('untrusted_only') is not None,
                                  'child':json.loads(subprocess.check_output(
                                      [sys.executable,'-c',child_code,json.dumps(task['import_probe'])],text=True))}
    if task.get('checkpoint'):
        checkpoint=out/'checkpoint.pt'
        checkpoint.write_bytes(task['id'].encode())
        result['checkpoint']={'path':str(checkpoint),'bytes':checkpoint.stat().st_size,'step':2,
                              'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        if task.get('bad_checkpoint_hash'):
            result['checkpoint']['sha256']='0'*64
    (out/'result.json').write_text(json.dumps(result))
sys.exit(task.get('exit_code',0))
'''


FAKE_QUEUE = r'''
import argparse,json,pathlib,subprocess,sys,time
p=argparse.ArgumentParser()
for n in ('job-id','project','priority','npu-count','max-wait-seconds','state-file'):
    p.add_argument('--'+n,required=True)
p.add_argument('--candidate-ids')
p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args()
state=pathlib.Path(a.state_file)
state.write_text(json.dumps({'status':'queued'}))
argv=a.command[1:] if a.command[0]=='--' else a.command
task=json.loads(pathlib.Path(argv[argv.index('--task')+1]).read_text())
time.sleep(task.get('queue_delay',.05))
physical=[int(a.candidate_ids.split(',')[0])] if a.candidate_ids else [11]
state.write_text(json.dumps({'status':'running','physical_ids':physical}))
rc=subprocess.run(argv).returncode
state.write_text(json.dumps({'status':'completed' if rc==0 else 'failed','physical_ids':physical}))
sys.exit(rc)
'''


def prepared(tmp_path, monkeypatch, tasks, duration=3, cli_foundation_source=None, **overrides):
    source = tmp_path / "source"
    source.mkdir()
    (source / "fake_worker.py").write_text(FAKE_WORKER)
    helper = source / "fake_queue.py"
    helper.write_text(FAKE_QUEUE)
    plan = {"tasks": tasks, "round_id": "test", "worker_module": "fake_worker",
            "queue_helper": str(helper), "dispatch_reserve_seconds": 0,
            "finish_reserve_seconds": 0, "term_grace_seconds": .15,
            "poll_seconds": .025, "storage_poll_seconds": .1,
            "memory_guard": {"limit_bytes": 256*s.GIB, "reserve_bytes": s.GIB,
                             "system_reserve_bytes": 0, "poll_seconds": .025, "admission_factor": 1.25},
            **overrides}
    plan_path = tmp_path / "input-plan.json"
    plan_path.write_text(json.dumps(plan))
    data = tmp_path / "data"
    data.mkdir()
    args = argparse.Namespace(plan=str(plan_path), output=str(tmp_path / "run"),
                              data=str(data), python=sys.executable, source=str(source),
                              duration_seconds=duration, npu_slots=2, cpu_slots=2,
                              foundation_source=cli_foundation_source)
    monkeypatch.setattr(s, "storage_snapshot", lambda *args: {
        "available_for_admission_bytes": 128*s.GIB,
        "soft_grace_expired_or_unknown": False})
    return s.Scheduler(args)


def task(name, **kwargs):
    return {"id": name, "backend": "cpu", "cpu_threads": 1,
            "wall_seconds": 2, "estimated_output_bytes": 1000, **kwargs}


def test_quota_parser_numeric_grace_and_wrapping():
    text = """Disk quotas for user zlong (uid 1003):
 Filesystem blocks quota limit grace files quota limit grace
 /dev/mapper/data 1048576* 2097152 4194304 1987654321 2 0 0 0
 /dev/other
 10 20 30 0 4 0 0 0
"""
    rows = s.parse_quota(text)
    assert rows[0].used_bytes == s.GIB
    assert rows[0].hard_bytes == 4*s.GIB
    assert rows[0].grace_epoch == 1987654321
    assert rows[1].used_bytes == 10240
    with pytest.raises(ValueError):
        s.parse_quota("quota: not available")
    with pytest.raises(ValueError):
        s.parse_quota("/dev/x 10 20 30 7days 1 0 0 0")


def test_storage_checks_quota_and_filesystem_separately(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "current_quota", lambda _: s.Quota("/dev/x", 20*s.GIB, 30*s.GIB, 40*s.GIB, 0))
    monkeypatch.setattr(s.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 100*s.GIB})())
    result = s.storage_snapshot(tmp_path, {"quota_reserve_bytes": 2*s.GIB, "max_output_bytes": 50*s.GIB})
    assert result["available_for_admission_bytes"] == 18*s.GIB
    assert not result["soft_grace_expired_or_unknown"]


def test_quota_soft_limit_nonzero_exit_keeps_hard_limit(monkeypatch):
    monkeypatch.setattr(s.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=1, stdout="/dev/data 400 300 700 1987654321 2 0 0 0\n", stderr=""))
    quota = s.current_quota()
    assert quota.used_bytes == 400*1024
    assert quota.hard_bytes == 700*1024


def test_soft_grace_is_observational(tmp_path, monkeypatch):
    runner = prepared(tmp_path, monkeypatch, [task("a")])
    monkeypatch.setattr(s, "storage_snapshot", lambda *args: {
        "available_for_admission_bytes": s.GIB, "soft_grace_expired_or_unknown": True})
    assert runner.run() == 0
    assert runner.records["a"]["status"] == "completed"


def test_python_venv_symlink_is_not_resolved(tmp_path, monkeypatch):
    runner = prepared(tmp_path, monkeypatch, [task("a")])
    link = tmp_path/"venv-python"
    link.symlink_to(sys.executable)
    runner.args.python = str(link)
    runner.args.output = str(tmp_path/"second-run")
    second = s.Scheduler(runner.args)
    assert second.python == str(link)


@pytest.mark.parametrize("backend,setting", [("cpu", "plan"), ("npu", "plan"),
                                              ("cpu", "cli"), ("cpu", None)])
def test_explicit_foundation_import_root_reaches_worker_and_fresh_child(tmp_path, monkeypatch, backend, setting):
    foundation = tmp_path / "frozen-foundation"
    installed = tmp_path / "installed-foundation"
    untrusted = tmp_path / "external-pythonpath"
    for directory, marker in ((foundation, "patched-frozen"), (installed, "installed-old"),
                              (untrusted, "untrusted")):
        package = directory / "fakefoundation"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f"MARKER = {marker!r}\n")
    (untrusted / "untrusted_only.py").write_text("MARKER = 'must-not-leak'\n")
    (foundation / "source_order_probe.py").write_text("MARKER = 'foundation'\n")
    monkeypatch.setenv("PYTHONPATH", str(untrusted))
    options = {"foundation_source": str(foundation)} if setting == "plan" else {}
    runner = prepared(tmp_path, monkeypatch,
                      [task("imports", backend=backend, import_probe=["fakefoundation", "source_order_probe"])],
                      cli_foundation_source=str(foundation) if setting == "cli" else None, **options)
    (runner.source / "source_order_probe.py").write_text("MARKER = 'project-source'\n")
    # Supply a fallback package just as an installed distribution would.  The
    # old launcher loses the foundation root and imports this wrong copy.
    (runner.source / "sitecustomize.py").write_text(f"import sys\nsys.path.append({str(installed)!r})\n")
    assert runner.run() == 0
    result = json.loads((runner.root / "tasks/imports/result.json").read_text())
    expected_path = str(runner.source) + (os.pathsep + str(foundation) if setting else "")
    expected_package = (foundation if setting else installed) / "fakefoundation/__init__.py"
    for observation in (result["import_probe"], result["import_probe"]["child"]):
        assert observation["pythonpath"] == expected_path
        assert observation["modules"]["fakefoundation"]["marker"] == ("patched-frozen" if setting else "installed-old")
        assert Path(observation["modules"]["fakefoundation"]["file"]) == expected_package
        assert observation["modules"]["source_order_probe"]["marker"] == "project-source"
        assert observation["untrusted_visible"] is False
    record = runner.records["imports"]
    assert record["resolved_environment"] == {"PYTHONPATH": expected_path}
    persisted = json.loads((runner.root / "scheduler.json").read_text())
    assert persisted["resolved_environment"] == record["resolved_environment"]
    assert persisted["foundation_source"] == (str(foundation) if setting else None)


@pytest.mark.parametrize("kind", ["relative", "missing", "file", "conflict"])
def test_invalid_foundation_declaration_fails_before_output(tmp_path, monkeypatch, kind):
    foundation = tmp_path / "foundation"
    if kind == "file":
        foundation.write_text("not a source directory")
    elif kind != "missing":
        foundation.mkdir()
    requested = "relative-foundation" if kind == "relative" else str(foundation)
    cli = None
    if kind == "conflict":
        other = tmp_path / "other-foundation"
        other.mkdir()
        cli = str(other)
    with pytest.raises((ValueError, FileNotFoundError)):
        prepared(tmp_path, monkeypatch, [task("never")], foundation_source=requested,
                 cli_foundation_source=cli)
    assert not (tmp_path / "run").exists()


def test_inner_failure_is_not_erased_by_outer_success():
    assert s.task_outcome(0, {"status": "failed"}, None)[0] == "failed"
    assert s.task_outcome(0, None, None)[1] == "missing_worker_result"
    assert s.task_outcome(0, {"status": "completed"}, None)[0] == "completed"
    assert s.task_outcome(1, {"status": "completed"}, None)[0] == "failed"
    assert s.task_outcome(-15, {"status": "failed"}, "task_deadline")[0] == "failed"


def test_cpu_workers_have_disjoint_affinity_and_overlap(tmp_path, monkeypatch):
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("requires two available CPUs")
    runner = prepared(tmp_path, monkeypatch, [task("a", sleep=.2), task("b", sleep=.2)])
    assert runner.run() == 0
    records = runner.records
    assert all(r["status"] == "completed" for r in records.values())
    assert set(records["a"]["cpu_affinity"]).isdisjoint(records["b"]["cpu_affinity"])
    assert max(r["worker_started_unix"] for r in records.values()) < min(r["worker_ended_unix"] for r in records.values())
    assert all(r["descendants_terminated"] for r in records.values())
    assert json.loads((runner.root/"scheduler.json").read_text())["all_task_processes_terminated"]


def test_npu_launch_goes_through_queue(tmp_path, monkeypatch):
    runner = prepared(tmp_path, monkeypatch, [task("npu-case", backend="npu", sleep=.1)])
    assert runner.run() == 0
    record = runner.records["npu-case"]
    assert record["status"] == "completed"
    assert "--npu-count" in record["command"]
    assert record["queue"]["physical_ids"] == [11]
    result = json.loads((runner.root/"tasks/npu-case/result.json").read_text())
    assert result["device"] == "npu:0"


def test_npu_queue_wait_does_not_consume_worker_runtime(tmp_path, monkeypatch):
    runner = prepared(tmp_path, monkeypatch, [task("delayed", backend="npu", queue_delay=.35,
                                                       sleep=.05, wall_seconds=.2)])
    assert runner.run() == 0
    record = runner.records["delayed"]
    assert record["status"] == "completed"
    assert record["launch_to_worker_seconds"] >= .35
    assert record["worker_elapsed_seconds"] < .2
    assert record["deadline_unix"] == pytest.approx(record["worker_started_unix"] + .2)
    assert record["supplied_deadline_unix"] > record["deadline_unix"]


def test_deadline_kills_process_group_and_records_limit(tmp_path, monkeypatch):
    runner = prepared(tmp_path, monkeypatch, [task("sleeper", child=True, sleep=30, wall_seconds=.25)])
    began = time.monotonic()
    assert runner.run() == 0
    assert time.monotonic()-began < 3
    record = runner.records["sleeper"]
    assert record["status"] == "budget_exhausted"
    assert record["kill_sent_unix"] is not None
    child_pid = int((runner.root/"tasks/sleeper/descendant.pid").read_text())
    identity = s.proc_identity(child_pid)
    assert identity is None or identity["state"] in {"Z", "X"}
    assert record["descendants_terminated"]


def test_round_deadline_and_dispatch_reserve(tmp_path, monkeypatch):
    runner = prepared(tmp_path, monkeypatch, [task("never")], duration=.2, dispatch_reserve_seconds=.3)
    assert runner.run() == 0
    assert runner.records["never"]["status"] == "not_started"
    assert runner.records["never"]["outcome_reason"] == "dispatch_cutoff"


def test_quota_failure_fails_closed(tmp_path, monkeypatch):
    runner = prepared(tmp_path, monkeypatch, [task("never")])
    def unavailable(*args):
        raise RuntimeError("quota unavailable")
    monkeypatch.setattr(s, "storage_snapshot", unavailable)
    assert runner.run() == 1
    assert runner.stop_reason == "quota_unavailable"
    assert runner.records["never"]["status"] == "not_started"
    assert not (runner.root/"tasks").exists()


def test_signal_cleanup_is_bounded(tmp_path, monkeypatch):
    runner = prepared(tmp_path, monkeypatch, [task("a", sleep=30, wall_seconds=30)])
    update = runner.update_running
    def request_after_launch():
        if runner.running:
            runner.request_stop(15, None)
        update()
    monkeypatch.setattr(runner, "update_running", request_after_launch)
    assert runner.run() == 0
    assert runner.records["a"]["status"] == "budget_exhausted"
    assert runner.records["a"]["descendants_terminated"]


def test_existing_output_refused(tmp_path, monkeypatch):
    runner = prepared(tmp_path, monkeypatch, [task("a")])
    with pytest.raises(FileExistsError):
        s.Scheduler(runner.args)


def test_phases_have_barrier_and_separate_slot_limits(tmp_path, monkeypatch):
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("requires two CPUs")
    phases = [{"id": "isolated", "cpu_slots": 1, "npu_slots": 0},
              {"id": "concurrent", "cpu_slots": 2, "npu_slots": 0}]
    runner = prepared(tmp_path, monkeypatch,
                      [task("parallel-a", phase="concurrent", sleep=.2),
                       task("serial-a", phase="isolated", sleep=.1),
                       task("parallel-b", phase="concurrent", sleep=.2),
                       task("serial-b", phase="isolated", sleep=.1)], phases=phases)
    assert runner.run() == 0
    rows = runner.records
    assert rows["serial-b"]["process_started_unix"] >= rows["serial-a"]["process_ended_unix"]
    assert min(rows[key]["process_started_unix"] for key in ("parallel-a", "parallel-b")) >= rows["serial-b"]["process_ended_unix"]
    assert max(rows[key]["worker_started_unix"] for key in ("parallel-a", "parallel-b")) < min(rows[key]["worker_ended_unix"] for key in ("parallel-a", "parallel-b"))
    assert all(phase["status"] == "completed" for phase in runner.phase_records)


@pytest.mark.parametrize("tasks,phases", [
    ([task("a", depends_on=["b"]), task("b", depends_on=["a"])], None),
    ([task("a", depends_on=["missing"])], None),
    ([task("a", phase="first", depends_on=["b"]), task("b", phase="later")],
     [{"id": "first"}, {"id": "later"}]),
    ([task("a", allow_parent_status=["failed"])], None),
])
def test_invalid_dependency_graph_rejected_before_output(tmp_path, monkeypatch, tasks, phases):
    options = {} if phases is None else {"phases": phases}
    with pytest.raises(ValueError):
        prepared(tmp_path, monkeypatch, tasks, **options)
    assert not (tmp_path / "run").exists()


def test_failed_dependencies_propagate_without_launch(tmp_path, monkeypatch):
    runner = prepared(tmp_path, monkeypatch, [task("grandchild", depends_on=["child"]),
                                             task("child", depends_on=["bad"]), task("bad", status="failed")])
    assert runner.run() == 1
    assert runner.records["bad"]["status"] == "failed"
    for key in ("child", "grandchild"):
        assert runner.records[key]["outcome_reason"] == "dependency_failed"
        assert not (runner.root / "tasks" / key).exists()


def test_parent_checkpoint_resolves_only_after_terminal_and_hash_check(tmp_path, monkeypatch):
    runner = prepared(tmp_path, monkeypatch, [task("child", resume_task="parent"),
                                             task("parent", checkpoint=True)])
    assert runner.run() == 0
    parent, child = runner.records["parent"], runner.records["child"]
    assert child["process_started_unix"] >= parent["process_ended_unix"]
    directory = runner.root / "tasks/child"
    declared = json.loads((directory / "declared-task.json").read_text())
    resolved = json.loads((directory / "task.json").read_text())
    assert declared["resume_task"] == "parent" and "resume_from" not in declared
    assert resolved["resume_from"] == str(runner.root / "tasks/parent/checkpoint.pt")
    assert resolved["resume_checkpoint_sha256"] == s.file_sha256(Path(resolved["resume_from"]))
    assert child["task_resolution"]["declared_task_sha256"] == s.content_sha256(declared)
    assert child["task_resolution"]["resolved_task_sha256"] == s.content_sha256(resolved)
    assert child["task_file_sha256"] == s.file_sha256(directory / "task.json")


def test_parent_bad_checkpoint_blocks_continuation(tmp_path, monkeypatch):
    runner = prepared(tmp_path, monkeypatch, [task("parent", checkpoint=True, bad_checkpoint_hash=True),
                                             task("child", resume_task="parent")])
    assert runner.run() == 1
    assert runner.records["parent"]["status"] == "completed"
    assert runner.records["child"]["outcome_reason"] == "launch_failure"
    assert "identity mismatch" in runner.records["child"]["error"]
    assert not (runner.root / "tasks/child/worker.json").exists()


def test_budget_parent_needs_explicit_allowance(tmp_path, monkeypatch):
    runner = prepared(tmp_path, monkeypatch,
                      [task("parent", checkpoint=True, status="budget_exhausted"),
                       task("strict", resume_task="parent"),
                       task("allowed", resume_task="parent", allow_parent_status=["completed", "budget_exhausted"])])
    assert runner.run() == 0
    assert runner.records["strict"]["outcome_reason"] == "dependency_failed"
    assert runner.records["allowed"]["status"] == "completed"


def test_selection_hook_preserves_declaration_and_validates_selected_checkpoint(tmp_path, monkeypatch):
    common = {"batch_size": 1, "sequence_length": 2, "lr": .001}
    runner = prepared(tmp_path, monkeypatch,
                      [task("seed-a", seed=1, checkpoint=True, **common),
                       task("seed-b", seed=2, checkpoint=True, **common),
                       task("continuation", seed=2, **common, depends_on=["seed-a", "seed-b"],
                            resume_select={"rule": "lowest_mean_validation_nll", "candidates": ["seed-a", "seed-b"],
                                           "seeds": [1, 2], "updates": 2})])
    assert runner.run() == 0
    directory = runner.root / "tasks/continuation"
    selected = json.loads((directory / "selection.json").read_text())
    resolved = json.loads((directory / "task.json").read_text())
    assert selected["selected_parent"] == "seed-b"
    assert resolved["resume_task"] == "seed-b"
    assert "resume_select" not in resolved
    assert "resume_select" in json.loads((directory / "declared-task.json").read_text())
    assert Path(resolved["resume_from"]).read_bytes() == b"seed-b"


def test_fixed_affinity_is_reserved_from_automatic_jobs(tmp_path, monkeypatch):
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 2:
        pytest.skip("requires two CPUs")
    runner = prepared(tmp_path, monkeypatch,
                      [task("auto", sleep=.2), task("fixed", cpu_affinity=[cpus[0]], sleep=.2),
                       task("fixed-again", cpu_affinity=[cpus[0]], sleep=.03)])
    assert runner.run() == 0
    assert runner.records["auto"]["cpu_affinity"] == [cpus[1]]
    assert runner.records["fixed"]["cpu_affinity"] == runner.records["fixed-again"]["cpu_affinity"] == [cpus[0]]
    assert runner.records["fixed-again"]["process_started_unix"] >= runner.records["fixed"]["process_ended_unix"]


def test_candidate_restriction_is_forwarded_through_queue(tmp_path, monkeypatch):
    runner = prepared(tmp_path, monkeypatch, [task("npu", backend="npu", candidate_ids=[7, 11])])
    assert runner.run() == 0
    record = runner.records["npu"]
    command = record["command"]
    assert command[command.index("--candidate-ids") + 1] == "7,11"
    assert command[command.index("--npu-count") + 1] == "1"
    assert record["requested_candidate_ids"] == [7, 11]
    assert record["queue"]["physical_ids"] == [7]


def test_memory_estimates_reserve_future_growth_before_parallel_launch(tmp_path, monkeypatch):
    baseline = s.proc_identity(os.getpid())["rss_bytes"]
    guard = {"limit_bytes": baseline + 96*1024**2, "reserve_bytes": 8*1024**2,
             "system_reserve_bytes": 0, "poll_seconds": .025, "admission_factor": 1.25}
    runner = prepared(tmp_path, monkeypatch,
                      [task("a", sleep=.15, estimated_rss_bytes=48*1024**2),
                       task("b", sleep=.15, estimated_rss_bytes=48*1024**2)], memory_guard=guard)
    assert runner.run() == 0
    assert runner.records["b"]["process_started_unix"] >= runner.records["a"]["process_ended_unix"]
    assert runner.records["a"]["reserved_rss_bytes"] == 60*1024**2
    assert runner.memory_samples > 1
    policy = json.loads((runner.root / "scheduler.json").read_text())["memory_guard"]
    assert policy["enforcement"] == "sampled_process_rss_software_watchdog"
    assert policy["kernel_memory_limit_established"] is False


def test_bad_rss_observation_fails_closed_without_launch(tmp_path, monkeypatch):
    runner = prepared(tmp_path, monkeypatch, [task("never")])
    original = s.process_table
    def invalid(*, strict=False):
        if strict:
            raise RuntimeError("unreadable owned process RSS")
        return original()
    monkeypatch.setattr(s, "process_table", invalid)
    assert runner.run() == 1
    assert runner.stop_reason == "memory_observation_failed"
    assert not (runner.root / "tasks").exists()
    assert runner.memory_error is not None


def test_memory_probe_failure_during_work_stops_admission_and_cleans_children(tmp_path, monkeypatch):
    runner = prepared(tmp_path, monkeypatch,
                      [task("active", sleep=30, wall_seconds=30),
                       task("next", depends_on=["active"])])
    original = s.host_memory
    def unavailable_during_work():
        if runner.running:
            raise ValueError("invalid MemAvailable observation")
        return original()
    monkeypatch.setattr(s, "host_memory", unavailable_during_work)
    assert runner.run() == 1
    assert runner.stop_reason == "memory_observation_failed"
    assert runner.records["active"]["status"] == "failed"
    assert runner.records["active"]["descendants_terminated"]
    assert not (runner.root / "tasks/next").exists()
    assert not runner.running and not runner.orphans


def test_rss_watchdog_counts_and_terminates_allocating_descendant(tmp_path, monkeypatch):
    baseline = s.proc_identity(os.getpid())["rss_bytes"]
    guard = {"limit_bytes": baseline + 64*1024**2, "reserve_bytes": 8*1024**2,
             "system_reserve_bytes": 0, "poll_seconds": .025, "admission_factor": 1.25}
    runner = prepared(tmp_path, monkeypatch,
                      [task("memory", rss_child_bytes=96*1024**2, sleep=30,
                            wall_seconds=30, estimated_rss_bytes=1024**2)],
                      duration=5, memory_guard=guard)
    assert runner.run() == 1
    assert runner.stop_reason == "rss_memory_limit"
    record = runner.records["memory"]
    assert record["termination_reason"] == "rss_memory_limit"
    assert record["descendants_terminated"]
    pid = int((runner.root / "tasks/memory/descendant.pid").read_text())
    samples = [json.loads(line) for line in runner.memory_path.read_text().splitlines()]
    assert any(any(p["pid"] == pid and p["rss_bytes"] >= 96*1024**2 for p in sample["processes"]) for sample in samples)
    assert s.proc_identity(pid) is None or s.proc_identity(pid)["state"] in {"Z", "X"}


def test_subreaper_tracks_detached_grandchild_until_cleanup(tmp_path, monkeypatch):
    before = s.child_subreaper()
    runner = prepared(tmp_path, monkeypatch,
                      [task("detached", detached_child=True, sleep=.2, wall_seconds=2)])
    assert runner.run() == 1
    record = runner.records["detached"]
    assert record["outcome_reason"] == "orphaned_descendants"
    pid = int((runner.root / "tasks/detached/descendant.pid").read_text())
    assert any(row["pid"] == pid for row in record["descendant_identities"])
    assert record["descendants_terminated"]
    assert s.proc_identity(pid) is None or s.proc_identity(pid)["state"] in {"Z", "X"}
    assert s.child_subreaper() is before
