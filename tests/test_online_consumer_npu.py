"""Bounded directed complete-consumer target gate, separate from throughput."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from resident_test_target import target_device

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from flow_protocol import native_text
from tools.online_bench.host import run
from online_consumer_support import observer,same
from test_online_consumer import packet

# Cover every family/memory and both schedules; all three mixed presets without
# expanding every fine switch into a Cartesian sweep.
CASES=[("pdg","add","streaming","mixed-a"),
       ("pdg","attention","prefill","mixed-b"),
       ("timed-dag","add","prefill","mixed-c"),
       ("timed-dag","attention","streaming","mixed-a"),
       ("settle","add","streaming","mixed-b"),
       ("settle","attention","prefill","mixed-c")]


def target():
    return torch.device(target_device("TIDE_ONLINE_DEVICE", resident=False))


@pytest.mark.parametrize("case",CASES)
@pytest.mark.parametrize("implementation",["python","native","libtorch"])
def test_mixed_complete_training(case,implementation,tmp_path):
    device=target();family,memory,schedule,preset=case
    p=packet(memory);expected=[];actual=[]
    reference=run(p,family=family,implementation="python",device="cpu",training=True,
                  optimizer="adamw",schedule="streaming",steps=2,warmup=0,
                  diagnostics=True,observer=observer(expected))
    if implementation=="libtorch":
        binary=os.environ["TIDE_ONLINE_BINARY"];path=tmp_path/"packet.txt";path.write_text(native_text(p));out=tmp_path/"standalone"
        cmd=[binary,"--device="+str(device),"--dtype=float32","--packet="+str(path),"--output-dir="+str(out),
             "--family="+family,"--preset="+preset,"--schedule="+schedule,"--training","--optimizer=adamw",
             "--steps=2","--warmup=0","--windows-per-step=2","--diagnostics"]
        result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90)
        assert result.returncode==0,result.stdout
        actual=[json.loads(line) for line in (out/"diagnostics.jsonl").read_text().splitlines()]
        measured=json.loads((out/"result.json").read_text())
    else:
        measured=run(p,family=family,implementation=implementation,device=device,training=True,optimizer="adamw",
                     schedule=schedule,preset=preset,steps=2,warmup=0,diagnostics=True,observer=observer(actual),
                     native_library=os.environ["TIDE_BUILD_DIR"] if implementation=="native" else None)
    same(actual,expected)
    torch.testing.assert_close(torch.tensor(measured["losses"]),torch.tensor(reference["losses"]),atol=1e-6,rtol=1e-5)
    assert measured["parameters"]==reference["parameters"] and measured["outputs"]==reference["outputs"]
