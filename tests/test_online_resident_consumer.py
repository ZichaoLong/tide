"""Actual model/loss/embedding training through public resident boundaries."""
import os
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from test_online_consumer import packet,ROOT
from test_online_consumer_npu import target
from online_consumer_support import observer,same
from tools.online_bench.host import run
from tidegraph import ResidentPlacement
from flow_protocol import native_text


def standalone(p,device,family,schedule,training,optimizer,tmp_path,devices=None):
    binary=os.environ.get("TIDE_ONLINE_BINARY")
    if not binary:pytest.skip("standalone resident consumer not explicitly selected")
    path=tmp_path/"packet.txt";path.write_text(native_text(p));out=tmp_path/"consumer"
    command=[binary,"--device="+str(device),"--dtype=float32","--packet="+str(path),"--output-dir="+str(out),
             "--family="+family,"--preset=resident","--schedule="+schedule,"--steps=2","--warmup=0",
             "--windows-per-step=2","--diagnostics","--optimizer="+optimizer]
    if training:command.append("--training")
    command.append("--devices="+str(devices if devices is not None else (2 if training and schedule=="prefill" else 1)))
    done=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
    assert done.returncode==0,done.stdout
    return json.loads((out/"result.json").read_text()),[json.loads(s) for s in (out/"diagnostics.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("family,memory,schedule",[
    ("pdg","add","streaming"),("pdg","attention","prefill"),
    ("timed-dag","add","prefill"),("timed-dag","attention","streaming"),
    ("settle","add","streaming"),("settle","attention","prefill")])
@pytest.mark.parametrize("implementation",["native","libtorch"])
def test_resident_complete_training(family,memory,schedule,implementation,tmp_path):
    device=target();p=packet(memory);expected=[];actual=[]
    opt="adamw" if schedule=="prefill" else "sgd"
    kw=dict(family=family,training=True,optimizer=opt,steps=2,warmup=0,windows_per_step=2,diagnostics=True)
    reference=run(p,implementation="python",device="cpu",schedule="streaming",observer=observer(expected),**kw)
    owners=ResidentPlacement(devices=(str(device),f"npu:{device.index+1}")) if schedule=="prefill" else None
    if implementation=="libtorch":
        candidate,actual=standalone(p,device,family,schedule,True,opt,tmp_path)
    else:
        candidate=run(p,implementation="native",device=device,schedule=schedule,preset="resident",
            observer=observer(actual),native_library=os.environ["TIDE_BUILD_DIR"],
            resident_library=os.environ["TIDE_RESIDENT_LIBRARY"],resident_placement=owners,**kw)
    same(actual,expected)
    assert candidate["outputs"]==reference["outputs"] and candidate["final_cut"]==reference["final_cut"]
    torch.testing.assert_close(torch.tensor(candidate["losses"]),torch.tensor(reference["losses"]),atol=1e-6,rtol=1e-5)
    (tmp_path/"observed.json").write_text(json.dumps(dict(packet=p,candidate=candidate,observations=actual)))


@pytest.mark.parametrize("memory",["add","attention"])
@pytest.mark.parametrize("implementation",["native","libtorch"])
@pytest.mark.parametrize("family",["pdg","timed-dag","settle"])
def test_resident_continuous_inference(memory,implementation,family,tmp_path):
    p=packet(memory,delayed=family!="settle");device=target();expected=[];actual=[]
    kw=dict(family=family,training=False,steps=2,warmup=0,diagnostics=True)
    schedule="streaming" if memory=="add" else "prefill"
    a=run(p,implementation="python",device="cpu",schedule="streaming",observer=observer(expected),**kw)
    if implementation=="libtorch":
        b,actual=standalone(p,device,family,schedule,False,"sgd",tmp_path,devices=2)
        owners=b["runtime"]["resident"]["devices"]
    else:
        b=run(p,implementation="native",device=device,schedule=schedule,preset="resident",observer=observer(actual),
              native_library=os.environ["TIDE_BUILD_DIR"],resident_library=os.environ["TIDE_RESIDENT_LIBRARY"],
              resident_placement=ResidentPlacement(devices=(str(device),f"npu:{device.index+1}")),**kw)
        owners=b["runtime"]["resident"]["resolved_inference_placement"]["devices"]
    same(actual,expected)
    assert a["outputs"]==b["outputs"] and a["final_cut"]==b["final_cut"]
    assert len(owners)==2 and "VJP" not in b["timing"] and "optimizer" not in b["timing"]
    (tmp_path/"observed.json").write_text(json.dumps(dict(packet=p,candidate=b,observations=actual)))


@pytest.mark.parametrize("implementation",["native","libtorch"])
def test_resident_cli_limits_and_failure_records(implementation,tmp_path):
    device=target();p=packet();path=tmp_path/"packet.json";path.write_text(json.dumps(p))
    command=[sys.executable,str(ROOT/"scripts/run_execution_flow.py"),"--packet",str(path),"--device",str(device),
        "--family","settle","--implementation",implementation,"--preset","resident","--schedule","prefill",
        "--training","--steps","1","--warmup","0","--devices","2","--owner-policy","memory",
        "--chunk-policy","aggressive","--resident-queue","128","--resident-trace","512"]
    if implementation=="libtorch":
        binary=os.environ.get("TIDE_ONLINE_BINARY")
        if not binary:pytest.skip("standalone resident consumer not explicitly selected")
        command.extend(["--native-binary",binary])
    else:command.extend(["--native-library",os.environ["TIDE_BUILD_DIR"],"--resident-library",os.environ["TIDE_RESIDENT_LIBRARY"]])
    out=tmp_path/"run"
    done=subprocess.run(command+["--output-dir",str(out)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
    assert done.returncode==0,done.stdout
    result=json.loads((out/"result.json").read_text());assert result["state"]=="passed" and result["parameters"]==p["counts"]["parameters"]
    failed=tmp_path/"refused"
    done=subprocess.run(command+["--output-dir",str(failed),"--resident-workspace-bytes","1"],
                        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
    assert done.returncode!=0
    record=json.loads((failed/"result.json").read_text());assert record["state"]=="failed"


def test_resident_boundary_int64_coordinates():
    from types import SimpleNamespace
    from tools.online_bench.resident_loss import head_loss,embedding_gradient
    device=target();large=2**54+3
    coordinates=torch.tensor([[0,1,large,0,0,large],[1,1,large+17,0,0,large+5]],dtype=torch.int64)
    values=torch.arange(8,dtype=torch.float32).reshape(2,4)*.03125
    head=torch.arange(68,dtype=torch.float32).reshape(17,4)*.015625
    window=SimpleNamespace(coordinates=coordinates.to(device),values=values.to(device),
                           valid=torch.ones(2,dtype=torch.bool,device=device))
    with torch.no_grad():loss,root,dh,count=head_loss(window,head.to(device),stride=13,denominator=2,backward=True)
    x=values.clone().requires_grad_();w=head.clone().requires_grad_()
    labels=torch.tensor([((int(row[2])//13+1)*7+int(row[0])*3)%17 for row in coordinates])
    ref=torch.nn.functional.cross_entropy(x@w.t(),labels)
    dx,dw=torch.autograd.grad(ref,(x,w))
    torch.testing.assert_close(loss.cpu(),ref,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(root.cpu(),dx,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(dh.cpu(),dw,atol=1e-6,rtol=1e-5)
    assert count==2
    window.connected=window.valid
    gradient=embedding_gradient([window],torch.zeros(17,4,device=device))
    reference=torch.zeros(17,4)
    for c,v in zip(coordinates,values):reference[(int(c[5])*7+int(c[0])*3)%17]+=v
    torch.testing.assert_close(gradient.cpu(),reference,atol=0,rtol=0)
