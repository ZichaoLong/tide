"""Real per-edge model, continuous state and complete optimizer consumers."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
from flow_topology import ranked_graph, make_packet
from flow_protocol import make_continuous_packet, validate_packet, native_text
from tools.online_bench.fixture import build_model, source_values
from tools.online_bench.host import run
from online_consumer_support import observer, same


def packet(memory="add", delayed=False, clear=False):
    return make_continuous_packet(graph=ranked_graph(layers=3,region_width=2,fanout=2,local_span=2,delayed=delayed),
                                  memory=memory,width=4,batch=2,tokens=2,vocab=7,clear=clear)


def test_packet_versions_and_real_parameter_inventory():
    for memory in ("add","attention"):
        p=packet(memory);assert validate_packet(p) is p
        assert native_text(p).startswith("TIDE_COMPLETE_FLOW_2\n"+p["sha256"])
        _,model,embedding,head=build_model(p)
        assert sum(x.numel() for x in model.parameters() if x.requires_grad)+embedding.numel()+head.numel()==p["counts"]["parameters"]
        assert all(v.device.type=="cpu" for v in model.state_dict().values())
        assert model.nodes[0].weight is model.nodes[1].weight
        p["state_boundary"]="reset"
        with pytest.raises(ValueError,match="hash"):validate_packet(p)
    v1=make_packet(graph=ranked_graph(layers=3,region_width=2,fanout=2,local_span=2))
    assert validate_packet(v1) is v1 and native_text(v1).startswith("TIDE_COMPLETE_FLOW_1\n")


def test_named_initializer_independent_scalar_definition():
    name="edge/13/projection";key=7
    for b in name.encode("ascii"):key=(key*131+b)%2147483647
    wanted=[]
    for i in range(16):
        x=(i+key)%2147483647
        for _ in range(3):x=(x*1103515245+12345)%2147483647
        wanted.append(((x%65536)-32768)*2**-20)
    torch.testing.assert_close(source_values(name,(4,4),7).reshape(-1),torch.tensor(wanted),atol=0,rtol=0)


@pytest.mark.parametrize("memory",["add","attention"])
@pytest.mark.parametrize("family",["pdg","timed-dag","settle"])
@pytest.mark.parametrize("optimizer",["sgd","adamw"])
def test_python_continuation_training(memory,family,optimizer,dtype):
    p=packet(memory);wanted=[];got=[]
    kw=dict(family=family,implementation="python",device="cpu",dtype=str(dtype).split('.')[-1],
            training=True,optimizer=optimizer,steps=2,warmup=0,windows_per_step=2,diagnostics=True)
    a=run(p,schedule="streaming",observer=observer(wanted),**kw)
    b=run(p,schedule="prefill",observer=observer(got),**kw)
    same(got,wanted);torch.testing.assert_close(torch.tensor(a["losses"]),torch.tensor(b["losses"]))
    assert b["final_cut"]==4*p["workload"]["tokens"]*p["workload"]["stride"]
    first,last=[r for r in got if r["kind"]=="updated"]
    assert any(first["parameters"][k]!=last["parameters"][k] for k in first["parameters"])


@pytest.mark.parametrize("memory",["add","attention"])
@pytest.mark.parametrize("schedule",["streaming","prefill"])
@pytest.mark.parametrize("family",["pdg","timed-dag","settle"])
def test_standalone_against_independent_python(memory,schedule,family,dtype,tmp_path):
    binary=os.environ.get("TIDE_ONLINE_BINARY")
    if not binary:pytest.skip("standalone consumer not explicitly selected")
    assert Path(binary).is_file(),"explicit standalone consumer unavailable"
    p=packet(memory);path=tmp_path/"packet.txt";path.write_text(native_text(p));out=tmp_path/"native"
    name=str(dtype).split('.')[-1];optimizer="adamw" if dtype==torch.float64 else "sgd"
    command=[binary,"--device=cpu","--dtype="+name,"--packet="+str(path),"--output-dir="+str(out),
             "--family="+family,"--preset=cpu","--schedule="+schedule,"--training","--optimizer="+optimizer,
             "--steps=2","--warmup=0","--windows-per-step=2","--diagnostics"]
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90)
    assert result.returncode==0,result.stdout
    got=[json.loads(row) for row in (out/"diagnostics.jsonl").read_text().splitlines()];wanted=[]
    reference=run(p,family=family,implementation="python",device="cpu",dtype=name,schedule="streaming",training=True,
                  optimizer=optimizer,steps=2,warmup=0,diagnostics=True,observer=observer(wanted))
    same(got,wanted)
    measured=json.loads((out/"result.json").read_text())
    assert measured["parameters"]==reference["parameters"] and measured["outputs"]==reference["outputs"]
    torch.testing.assert_close(torch.tensor(measured["losses"]),torch.tensor(reference["losses"]))


def test_delayed_continuation_and_unsupported_refusal():
    p=packet(delayed=True);a=[];b=[]
    kw=dict(family="timed-dag",implementation="python",device="cpu",training=True,steps=2,warmup=0,diagnostics=True)
    run(p,schedule="streaming",observer=observer(a),**kw)
    run(p,schedule="prefill",observer=observer(b),**kw);same(a,b)
    with pytest.raises(ValueError,match="equivalent"):
        run(p,family="settle",implementation="python",device="cpu")
    with pytest.raises(ValueError,match="resident"):
        run(packet(),family="pdg",implementation="native",device="cpu",preset="resident")
    with pytest.raises(ValueError,match="parameter-budget"):
        run(packet(),family="pdg",implementation="python",device="cpu",parameter_budget=1)


def test_settle_embedding_preserves_body_without_second_body_construction(monkeypatch):
    from tidegraph.settle import SettleGraph
    from tidegraph.ops import Model
    p=packet();g,m,_,_=build_model(p)
    def reject(*args,**kwargs):raise AssertionError("embedding reconstructed full model")
    monkeypatch.setattr(Model,"__init__",reject)
    eg,em=SettleGraph(g,tuple(p["graph"]["ranks"])).embed(m)
    assert all(a is b for a,b in zip(m.nodes,em.nodes))
    assert em.graph_identity==eg.identity and em.width==m.width


@pytest.mark.parametrize("memory",["add","attention"])
def test_standalone_unified_cli_and_delayed_packet(memory,tmp_path):
    binary=os.environ.get("TIDE_ONLINE_BINARY")
    if not binary:pytest.skip("standalone consumer not explicitly selected")
    p=packet(memory,delayed=True);path=tmp_path/"packet.json";path.write_text(json.dumps(p));out=tmp_path/"run"
    command=[sys.executable,str(ROOT/"scripts/run_execution_flow.py"),"--packet",str(path),"--output-dir",str(out),
             "--device","cpu","--dtype","float32","--family","timed-dag","--preset","cpu","--schedule","prefill",
             "--implementation","libtorch","--native-binary",binary,"--training","--steps","2","--warmup","0","--diagnostics",
             "--workers","2","--packed-sources","--batch-next"]
    done=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90)
    assert done.returncode==0,done.stdout
    record=json.loads((out/"result.json").read_text())
    assert record["packet_identity"]=="hash-validated JSON; exact derived v2 text"
    assert record["host_execution"]==dict(workers=2,packed_sources=True,batch_next=True)
    got=[json.loads(x) for x in (out/"consumer/diagnostics.jsonl").read_text().splitlines()];wanted=[]
    run(p,family="timed-dag",implementation="python",device="cpu",schedule="streaming",training=True,
        steps=2,warmup=0,diagnostics=True,observer=observer(wanted))
    same(got,wanted)


def test_python_cli_failure_record_and_diagnostics(tmp_path):
    p=packet();path=tmp_path/"packet.json";path.write_text(json.dumps(p))
    base=[sys.executable,str(ROOT/"scripts/run_execution_flow.py"),"--packet",str(path),"--device","cpu",
          "--family","settle","--preset","cpu","--schedule","prefill","--implementation","python",
          "--steps","1","--warmup","0","--diagnostics"]
    out=tmp_path/"success"
    done=subprocess.run(base+["--output-dir",str(out)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90)
    assert done.returncode==0,done.stdout
    assert (out/"diagnostics.jsonl").is_file()
    failed=tmp_path/"failed"
    done=subprocess.run(base+["--output-dir",str(failed),"--parameter-budget","1"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90)
    assert done.returncode!=0
    record=json.loads((failed/"result.json").read_text());assert record["state"]=="failed" and "parameter-budget" in record["error"]
