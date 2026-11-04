"""Real subprocess writes must not invalidate a later fixed-source submission."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from tidegraph.lm.job_source import source_identity

REPO=Path(__file__).resolve().parents[1]


def clean_source(tmp_path):
    source=tmp_path/'source';source.mkdir()
    subprocess.run(['git','init','-q',str(source)],check=True)
    subprocess.run(['git','-C',str(source),'-c','user.name=Test','-c','user.email=test@localhost',
                    'commit','--allow-empty','-qm','fixed'],check=True)
    return source


def run_bounded(source,out,code,seconds=10):
    script=('import sys;from tidegraph.lm.queue import bounded;'
            'raise SystemExit(bounded([sys.executable,"-c",sys.argv[2]],sys.argv[1],float(sys.argv[3])))')
    env={**os.environ,'PYTHONPATH':str(REPO/'python'),'PYTHONDONTWRITEBYTECODE':'1'}
    return subprocess.run([sys.executable,'-B','-c',script,str(out),code,str(seconds)],
                          cwd=source,env=env,capture_output=True,text=True,timeout=30)


def test_relative_vendor_output_and_delayed_submission_guard(tmp_path):
    source=clean_source(tmp_path);commit=source_identity(source)
    for name in ('first','delayed'):
        out=tmp_path/name
        result=run_bounded(source,out,'from pathlib import Path;Path("fusion_result.json").write_text("vendor output")')
        assert result.returncode==0,result.stderr
        assert (out/'work/fusion_result.json').read_text()=='vendor output'
        assert json.loads((out/'status.json').read_text())['cwd']==str(out/'work')
        assert source_identity(source)==commit
    (source/'unsaved').write_text('user change')
    with pytest.raises(ValueError,match='clean fixed checkout'):source_identity(source)


def test_timeout_reaps_relative_work_directory_descendants(tmp_path):
    source=clean_source(tmp_path);out=tmp_path/'timeout'
    code=('import subprocess,sys,time;from pathlib import Path;'
          'Path("fusion_result.json").write_text("vendor output");'
          'subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);time.sleep(60)')
    result=run_bounded(source,out,code,.5)
    assert result.returncode==124,result.stderr
    record=json.loads((out/'status.json').read_text())
    assert record['termination_reason']=='timeout' and record['remaining_live_pids']==[]
    assert (out/'work/fusion_result.json').exists()
    assert source_identity(source)


def test_relative_paths_in_worker_are_resolved_before_child_launch(tmp_path,monkeypatch):
    from tidegraph.lm import queue
    from tidegraph.lm.matrix import case
    from tidegraph.experiment.records import write_json
    monkeypatch.chdir(tmp_path)
    root=tmp_path/'experiment';root.mkdir();directory=tmp_path/'case';directory.mkdir()
    write_json(directory/'case.json',case('qwen15','cpu'))
    calls=[]
    def bounded(command,output,seconds):
        calls.append((command,output));return 0
    monkeypatch.setattr(queue,'bounded',bounded)
    assert queue.worker('case/case.json','experiment','case')==0
    command,output=calls[0]
    assert str(directory/'case.json') in command and str(directory/'run') in command
    assert str(root/'models') in command and output==directory/'process'
