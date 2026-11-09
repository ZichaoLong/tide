#!/usr/bin/env python3
"""Observable bounded dependency wait, before the compute budget starts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,now

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dependency',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline-epoch',type=float,required=True)
    p.add_argument('--terminal-only',action='store_true');p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    while time.time()<a.deadline_epoch:
        states=[json.loads(f.read_text()).get('state') if f.exists() else 'not_started' for f in a.dependency]
        state=dict(schema='tide-depth-dependency-v1',state='waiting_dependency',dependencies=list(map(str,a.dependency)),observed_states=states,at=now())
        atomic_json(a.output/'state.json',state)
        if not a.terminal_only and 'failed' in states:
            state.update(state='failed',reason='required predecessor failed');atomic_json(a.output/'state.json',state);raise SystemExit(3)
        if all(s in (['completed','failed'] if a.terminal_only else ['completed']) for s in states):break
        time.sleep(5)
    else:
        state.update(state='failed',reason='shared deadline while waiting');atomic_json(a.output/'state.json',state);raise SystemExit(124)
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:raise ValueError('missing command')
    state.update(state='admitted',compute_started_at=now());atomic_json(a.output/'state.json',state)
    child=subprocess.run(command)
    state.update(state='completed' if child.returncode==0 else 'failed',exit_code=child.returncode,finished_at=now())
    atomic_json(a.output/'state.json',state);raise SystemExit(child.returncode)
