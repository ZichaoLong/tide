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


def standalone(p,device,family,schedule,training,optimizer,tmp_path,devices=None,dtype_name="float32",head_budget=None,extra=()):
    binary=os.environ.get("TIDE_ONLINE_BINARY")
    if not binary:pytest.skip("standalone resident consumer not explicitly selected")
    path=tmp_path/"packet.txt";path.write_text(native_text(p));out=tmp_path/"consumer"
    command=[binary,"--device="+str(device),"--dtype="+dtype_name,"--packet="+str(path),"--output-dir="+str(out),
             "--family="+family,"--preset=resident","--schedule="+schedule,"--steps=2","--warmup=0",
             "--windows-per-step=2","--diagnostics","--optimizer="+optimizer]
    if training:command.append("--training")
    command.append("--devices="+str(devices if devices is not None else (2 if training and schedule=="prefill" else 1)))
    if head_budget is not None:command.extend(["--head-workspace-bytes="+str(head_budget),"--chunk-policy=aggressive","--resident-outputs=16"])
    command.extend(extra)
    done=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
    assert done.returncode==0,done.stdout
    return json.loads((out/"result.json").read_text()),[json.loads(s) for s in (out/"diagnostics.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("family,memory,schedule",[
    ("pdg","add","streaming"),("pdg","attention","prefill"),
    ("timed-dag","add","prefill"),("timed-dag","attention","streaming"),
    ("settle","add","streaming"),("settle","attention","prefill")])
@pytest.mark.parametrize("implementation",["native","libtorch"])
@pytest.mark.parametrize("dtype_name",["float32","float16"])
def test_resident_complete_training(family,memory,schedule,implementation,dtype_name,tmp_path):
    device=target();p=packet(memory);expected=[];actual=[]
    opt="adamw" if schedule=="prefill" else "sgd"
    kw=dict(family=family,training=True,optimizer=opt,steps=2,warmup=0,windows_per_step=2,diagnostics=True)
    reference=run(p,implementation="python",device="cpu",schedule="streaming",observer=observer(expected),**kw)
    owners=ResidentPlacement(devices=(str(device),f"npu:{device.index+1}")) if schedule=="prefill" else None
    if implementation=="libtorch":
        candidate,actual=standalone(p,device,family,schedule,True,opt,tmp_path,dtype_name=dtype_name)
    else:
        candidate=run(p,implementation="native",device=device,schedule=schedule,preset="resident",
            observer=observer(actual),native_library=os.environ["TIDE_BUILD_DIR"],
            resident_library=os.environ["TIDE_RESIDENT_LIBRARY"],resident_placement=owners,dtype=dtype_name,**kw)
    tolerance=dict(atol=2e-3,rtol=2e-2) if dtype_name=="float16" else dict(atol=1e-6,rtol=1e-5)
    same(actual,expected,**tolerance)
    assert candidate["projection_placement"] == ("compact banks on Full owners" if owners else "coordinator dense")
    assert candidate["outputs"]==reference["outputs"] and candidate["final_cut"]==reference["final_cut"]
    torch.testing.assert_close(torch.tensor(candidate["losses"]),torch.tensor(reference["losses"]),**tolerance)
    assert candidate["precision"]["payload"]==dtype_name and candidate["precision"]["optimizer_masters"]=="float32"
    (tmp_path/"observed.json").write_text(json.dumps(dict(packet=p,candidate=candidate,observations=actual)))


@pytest.mark.parametrize("memory",["add","attention"])
@pytest.mark.parametrize("implementation",["native","libtorch"])
@pytest.mark.parametrize("family",["pdg","timed-dag","settle"])
@pytest.mark.parametrize("dtype_name",["float32","float16"])
def test_resident_continuous_inference(memory,implementation,family,dtype_name,tmp_path):
    p=packet(memory,delayed=family!="settle");device=target();expected=[];actual=[]
    kw=dict(family=family,training=False,steps=2,warmup=0,diagnostics=True)
    schedule="streaming" if memory=="add" else "prefill"
    a=run(p,implementation="python",device="cpu",schedule="streaming",observer=observer(expected),**kw)
    if implementation=="libtorch":
        b,actual=standalone(p,device,family,schedule,False,"sgd",tmp_path,devices=2,dtype_name=dtype_name)
        owners=b["runtime"]["resident"]["devices"]
    else:
        b=run(p,implementation="native",device=device,schedule=schedule,preset="resident",observer=observer(actual),
              native_library=os.environ["TIDE_BUILD_DIR"],resident_library=os.environ["TIDE_RESIDENT_LIBRARY"],
              resident_placement=ResidentPlacement(devices=(str(device),f"npu:{device.index+1}")),dtype=dtype_name,**kw)
        owners=b["runtime"]["resident"]["resolved_inference_placement"]["devices"]
    same(actual,expected,**(dict(atol=2e-3,rtol=2e-2) if dtype_name=="float16" else {}))
    assert a["outputs"]==b["outputs"] and a["final_cut"]==b["final_cut"]
    assert len(owners)==2 and "VJP" not in b["timing"] and "optimizer" not in b["timing"]
    assert b["projection_placement"] == "compact banks on Full owners"
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
    for flag in ("--resident-workspace-bytes","--head-workspace-bytes"):
        failed=tmp_path/flag.removeprefix("--")
        done=subprocess.run(command+["--output-dir",str(failed),flag,"1"],
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


@pytest.mark.parametrize("implementation",["native","libtorch"])
@pytest.mark.parametrize("dtype_name",["float32","float16"])
def test_resident_split_head_keeps_whole_update(implementation,dtype_name,tmp_path):
    from tidegraph import ResidentLimits
    from tools.online_bench.head_budget import head_budget
    device=target();p=packet("attention");expected=[];actual=[]
    budget=(32*1024**2*10+8)//9+6400
    kw=dict(family="settle",training=True,optimizer="adamw",steps=2,warmup=0,windows_per_step=2,diagnostics=True)
    reference=run(p,implementation="python",device="cpu",schedule="streaming",observer=observer(expected),**kw)
    if implementation=="libtorch":
        candidate,actual=standalone(p,device,"settle","prefill",True,"adamw",tmp_path,devices=2,
                                     dtype_name=dtype_name,head_budget=budget)
    else:
        candidate=run(p,implementation="native",device=device,schedule="prefill",preset="resident",
            dtype=dtype_name,observer=observer(actual),native_library=os.environ["TIDE_BUILD_DIR"],
            resident_library=os.environ["TIDE_RESIDENT_LIBRARY"],head_workspace_bytes=budget,
            resident_limits=ResidentLimits(outputs=16,workspace_bytes=512*1024**2,chunk_policy="aggressive"),
            resident_placement=ResidentPlacement(devices=(str(device),f"npu:{device.index+1}")),**kw)
    tolerance=dict(atol=2e-3,rtol=2e-2) if dtype_name=="float16" else dict(atol=1e-6,rtol=1e-5)
    same(actual,expected,**tolerance)
    assert candidate["outputs"]==reference["outputs"] and candidate["final_cut"]==reference["final_cut"]
    plan=head_budget(16,4,7,2 if dtype_name=="float16" else 4,True,budget,True)
    assert candidate["head_memory"]==vars(plan)
    assert all(s["head_chunks"]>candidate["windows_per_step"] for s in candidate["statistics"])
    (tmp_path/"split-head.json").write_text(json.dumps(dict(candidate=candidate,observations=actual)))


@pytest.mark.parametrize("half",[False,True])
@pytest.mark.parametrize("training",[False,True])
def test_head_allocator_calibration(half,training,tmp_path):
    from types import SimpleNamespace
    from tools.online_bench.resident_loss import head_loss
    from tools.online_bench.head_budget import head_budget
    device=target();dtype=torch.float16 if half else torch.float32
    values=torch.randn(128,32,device=device,dtype=dtype)*.125
    head=torch.randn(4096,32,device=device,dtype=dtype)*.125
    window=SimpleNamespace(values=values,valid=torch.ones(128,device=device,dtype=torch.bool),
        coordinates=torch.zeros(128,6,device=device,dtype=torch.int64))
    plan=head_budget(128,32,4096,values.element_size(),training,40*1024**2,True)
    assert plan.rows<128
    torch.npu.synchronize(device);torch.npu.reset_peak_memory_stats(device)
    baseline=torch.npu.memory_allocated(device)
    with torch.no_grad():result=head_loss(window,head,stride=1,denominator=128,backward=training,plan=plan)
    torch.npu.synchronize(device)
    peak=torch.npu.max_memory_allocated(device)-baseline
    assert torch.isfinite(result[0]) and peak<=plan.budget
    (tmp_path/"head-memory.json").write_text(json.dumps(dict(dtype=str(dtype),training=training,
        plan=vars(plan),baseline_allocated=baseline,peak_allocated_delta=peak,
        peak_reserved=torch.npu.max_memory_reserved(device))))
