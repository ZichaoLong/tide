"""Isolated installed-wheel CPU continuation check; run with Python -I from /.

This standalone harness imports only the installed tide package, Torch and the
standard library. It never adds a checkout to sys.path. Each restored fourth
update runs in a fresh interpreter using the same installed wheel.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields, is_dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import zipfile


def atomic(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,stage=tempfile.mkstemp(dir=path.parent,prefix=".installed-")
    try:
        with os.fdopen(descriptor,"w") as stream:
            json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n")
            stream.flush();os.fsync(stream.fileno())
        os.replace(stage,path)
        directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if os.path.exists(stage):os.unlink(stage)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def leaves(value,prefix="root"):
    if isinstance(value,torch.Tensor):yield prefix,value
    elif is_dataclass(value):
        for field in fields(value):yield from leaves(getattr(value,field.name),prefix+"."+field.name)
    elif isinstance(value,dict):
        for key,item in value.items():yield from leaves(item,prefix+"."+str(key))
    elif isinstance(value,(tuple,list)):
        for index,item in enumerate(value):yield from leaves(item,prefix+"."+str(index))
    else:yield prefix,value


def clone(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if is_dataclass(value):return type(value)(**{f.name:clone(getattr(value,f.name)) for f in fields(value)})
    if isinstance(value,dict):return {key:clone(item) for key,item in value.items()}
    if isinstance(value,tuple) and hasattr(value,"_fields"):return type(value)(*(clone(item) for item in value))
    if isinstance(value,tuple):return tuple(clone(item) for item in value)
    if isinstance(value,list):return [clone(item) for item in value]
    return value


def compare(actual,expected,*,atol=0.,rtol=0.):
    a,b=dict(leaves(actual)),dict(leaves(expected))
    metadata=[]; failures=[]; nonfinite=[]; support=[]; worst=[]; exact=True; tensors=0
    if a.keys()!=b.keys():metadata.append("leaf names differ")
    for name in a.keys() & b.keys():
        x,y=a[name],b[name]
        if isinstance(x,torch.Tensor) and isinstance(y,torch.Tensor):
            tensors+=1
            if x.shape!=y.shape or x.dtype!=y.dtype:
                metadata.append(name+": shape/dtype");exact=False;continue
            if not torch.equal(x,y):exact=False
            if x.is_floating_point():
                if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
                    nonfinite.append(name);continue
                field=name.rsplit(".",1)[-1]
                support_tensor=(field=="strengths" or ".state." in name and
                                field in {"weight","short_weight","long_weight"})
                if support_tensor:
                    count=int(((x>0)!=(y>0)).sum())
                    if count:support.append(dict(name=name,different_entries=count,entries=x.numel()))
                delta=(x.double()-y.double()).abs()
                maximum=float(delta.max()) if delta.numel() else 0.
                if not torch.allclose(x,y,atol=atol,rtol=rtol):failures.append(name)
                worst.append(dict(name=name,max_abs=maximum))
            elif not torch.equal(x,y):metadata.append(name+": integer/bool data")
        elif type(x) is not type(y) or x!=y:
            metadata.append(name+": scalar/None");exact=False
    within=not metadata and not failures and not nonfinite
    return dict(exact=exact and not metadata,within_threshold=within,finite=not nonfinite,
                atol=atol,rtol=rtol,tensors=tensors,metadata_failures=metadata,
                nonfinite=nonfinite,support_differences=support,threshold_failures=failures,
                largest=sorted(worst,key=lambda row:row["max_abs"],reverse=True)[:8])


def structural_comparison_failure(result):
    return bool(result["metadata_failures"] or not result["finite"])


def classify_case(record):
    """Keep exact observations separate from structural and resume contracts.

    Inference/grad-forward uses zero tolerance to record exactness, not to impose
    a new FP32/BF16 qualification threshold. Saved data and the next restored
    forward retain their explicitly exact requirements in resume_case.
    """
    mode=record["inference_grad_forward"]
    structural=(structural_comparison_failure(mode) or not record["saved_checkpoint_data"]["exact"] or
        not record["checkpoint_config_exact"] or not record["projection_dtype_passed"] or
        not record["grad_forward_connected"] or not record["initial_vjp"]["input_connected"] or
        bool(record["initial_vjp"]["nonfinite"]) or bool(record["resume_exit_code"]) or
        record["resume"]["state"]=="structural_failed")
    if structural:return "structural_failed"
    if mode.get("support_differences") or record["resume"]["state"]=="support_deviation":
        return "support_deviation"
    if not mode["within_threshold"] or record["resume"]["state"]=="finite_deviation":
        return "finite_deviation"
    return "within_threshold"


def finite(value):
    return [name for name,tensor in leaves(value) if isinstance(tensor,torch.Tensor)
            and tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())]


def installation(args):
    import tide
    if not sys.flags.isolated or Path.cwd()!=Path("/") or "PYTHONPATH" in os.environ:
        raise RuntimeError("must run Python -I from / with PYTHONPATH removed")
    prefix=Path(args.venv).resolve()
    if Path(sys.prefix).resolve()!=prefix:
        raise RuntimeError("unexpected interpreter prefix")
    if os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD")!="0" or "torch_npu" in sys.modules:
        raise RuntimeError("CPU isolation requires disabled vendor autoload")
    source=json.loads(Path(args.source_manifest).read_text())
    package=Path(tide.__file__).resolve().parent
    if not package.is_relative_to(prefix):raise RuntimeError("tide imported outside new venv")
    wheel_files={}; installed=[]
    with zipfile.ZipFile(args.wheel) as archive:
        for name in archive.namelist():
            if name.endswith("/"):continue
            content=archive.read(name); digest=hashlib.sha256(content).hexdigest()
            wheel_files[name]=digest
            if name.startswith("tide/") and name.endswith(".py"):
                path=package.parent/name
                source_hash=source["files"]["src/"+name]
                match=path.is_file() and sha(path)==digest==source_hash
                installed.append(dict(name=name,path=str(path),sha256=sha(path),matches_wheel_and_source=match))
    if {row["name"] for row in installed}!={name.removeprefix("src/") for name in source["files"] if name.startswith("src/tide/") and name.endswith(".py")}:
        raise RuntimeError("wheel and source package file sets differ")
    modules=[]
    for name,module in sorted(sys.modules.items()):
        if name=="tide" or name.startswith("tide."):
            file=getattr(module,"__file__",None)
            if file is None:raise RuntimeError("unexpected Tide namespace module")
            path=Path(file).resolve()
            relative="tide/"+str(path.relative_to(package)) if path.is_relative_to(package) else None
            matches=relative in wheel_files and sha(path)==wheel_files[relative]
            modules.append(dict(module=name,path=str(path),sha256=sha(path),matches_wheel=matches))
    if not all(row["matches_wheel_and_source"] for row in installed) or not all(row["matches_wheel"] for row in modules):
        raise RuntimeError("installed wheel/source/module hash mismatch")
    return dict(state="passed",wheel=str(Path(args.wheel).resolve()),wheel_sha256=sha(args.wheel),
        wheel_size=Path(args.wheel).stat().st_size,wheel_files=wheel_files,installed_python_files=installed,
        loaded_modules=modules,package_root=str(package),python=sys.executable,prefix=sys.prefix,
        base_prefix=sys.base_prefix,isolated=sys.flags.isolated,cwd=str(Path.cwd()),pythonpath_present=False,
        distribution_version=importlib.metadata.version("tide-instances"),python_version=sys.version,
        torch_version=str(torch.__version__),torch_path=torch.__file__,vendor_imported=False,
        cpu_affinity=sorted(os.sched_getaffinity(0)),threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),host=platform.node())


class DtypeObserver:
    def __init__(self,model):
        self.events=Counter();self.lock=threading.Lock();self.phase="initial"
        self.caller=threading.get_ident();self.handles=[]
        def hook(name):
            def record(module,inputs,output):
                key=(self.phase,name,str(output.dtype),torch.is_autocast_enabled("cpu"),
                     torch.is_grad_enabled(),threading.get_ident()!=self.caller)
                with self.lock:self.events[key]+=1
            return record
        for name,module in model.named_modules():
            if isinstance(module,torch.nn.Linear):self.handles.append(module.register_forward_hook(hook(name)))
    def close(self):
        for handle in self.handles:handle.remove()
    def report(self):
        return [dict(phase=k[0],module=k[1],output_dtype=k[2],autocast=k[3],grad_enabled=k[4],
                     worker_thread=k[5],calls=v) for k,v in sorted(self.events.items())]


def objective(result):
    value=result.readout().float()
    weight=torch.cos(torch.arange(value.numel(),dtype=value.dtype)).reshape_as(value)
    return (value*weight).mean()


def forward(model,x,state=None,*,split=False):
    if split:
        first=model(x[:,:2],state,record_work=True)
        second=model(x[:,2:],first.state,record_work=True)
        values=[first,second]
    else:
        values=[model(x,state,record_work=True)]
    values.append(model.flush(values[-1].state,record_work=True))
    return values,sum(objective(value) for value in values)


def snapshot(model,optimizer,state):
    return clone(dict(parameters=model.state_dict(),optimizer=optimizer.state_dict(),state=state))


def gradients(model):return {name:None if parameter.grad is None else parameter.grad.detach().cpu().clone() for name,parameter in model.named_parameters()}


def update(model,optimizer,state,x,precision,observer):
    optimizer.zero_grad(set_to_none=True)
    before=clone(model.state_dict())
    observer.phase="train"
    inp=x.clone().requires_grad_()
    with precision_context("cpu",precision):
        results,loss=forward(model,inp,state,split=True)
    loss.backward()
    grads=gradients(model)
    missing=sorted(name for name,value in grads.items() if value is None)
    optimizer.step()
    state=detach_state(results[-1].state)
    data=snapshot(model,optimizer,state)
    bad=finite(dict(data=data,gradients=grads,input_gradient=inp.grad,results=results,loss=loss))
    changed=sum(not torch.equal(value,before[name]) for name,value in model.state_dict().items())
    if bad or not changed or inp.grad is None:raise RuntimeError(f"invalid actual update: {bad}, changed={changed}")
    return state,data,grads,dict(loss=float(loss.detach()),none_gradients=missing,changed_state_tensors=changed,
        state_positions=[result.state.position for result in results],input_gradient_finite=not finite(inp.grad),
        optimizer_step_counts=dict(Counter(str(int(value["step"])) for value in optimizer.state.values() if "step" in value)))


def make_config(path, execution_variant="whole_region"):
    from tide import ClockConfig,ModelConfig,TopologyConfig
    reception = execution_variant == "reception"
    training = execution_variant in ("training", "reception")
    return ModelConfig(topology=TopologyConfig(kind="multiscale",profile="compact-9",n=2,hubs=2,
        timing="timed",period=3,terminal_hubs=True),d=64,memory="X",sharing="region_shared",
        clock=ClockConfig(position_period=7,position_origin="absolute",kv_window=11,short_decay=.09,ssd_stride=3),
        hub_allocation="fixed_pass" if path=="batched" else "bounded",
        region_batch_size=4 if path=="batched" or reception else 1,
        checkpoint_regions=execution_variant=="whole_region",
        checkpoint_reception=training,
        ssd_summary_backend="prefix" if training else "block",
        ssd_scan_backend="doubling" if reception else "block",
        aggregation="incoming_light_64" if reception else "source_residual",
        reception_qkv_backend="fused" if reception else "separate",
        reception_mask_padding=reception,
        parameter_indexing="static_view" if reception else "gather",
        local_attention_backend="math" if execution_variant=="whole_region" else None)


def initial_case(args,path,precision,output):
    from tide import TideModel,place_model,placement_report,save_checkpoint
    torch.manual_seed(1631)
    config=make_config(path,args.execution_variant);model=TideModel(config)
    if path=="placed":
        partitions=[(model.graph.nodes[nodes[0]].position+model.graph.nodes[nodes[0]].layer)%2 for nodes in model.graph.regions]
        model=place_model(model,["cpu:0","cpu:1"],region_partitions=partitions,region_concurrency="per_device")
    optimizer=placement_adamw(model,lr=1e-4,foreach=False)
    observer=DtypeObserver(model)
    generator=torch.Generator().manual_seed(1632)
    x=torch.randn(1,4,64,generator=generator)
    record=dict(path=path,precision=precision,execution_variant=args.execution_variant,
                config=config.to_dict(),parameter_count=sum(p.numel() for p in model.parameters()),
                placement=None if path=="batched" else placement_report(model),updates=[],state="running")
    record["input_sha256"]=hashlib.sha256(x.numpy().tobytes()).hexdigest()
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    record["initial_state_dict_sha256"]=digest.hexdigest()
    if path=="batched":
        record["initial_region_batch_sizes"]=[len(batch) for regions in model.layer_regions for batch in model._region_batches(regions,None)]
    observer.phase="inference"
    with torch.inference_mode(),precision_context("cpu",precision):
        result,_=forward(model,x)
        inference=clone(result)
    observer.phase="grad_forward"
    inp=x.clone().requires_grad_()
    with precision_context("cpu",precision):result,loss=forward(model,inp)
    record["inference_grad_forward"]=compare(clone(result),inference)
    record["grad_forward_connected"]=bool(loss.requires_grad and any(value.values.requires_grad for value in result))
    loss.backward()
    record["initial_vjp"]=dict(input_connected=inp.grad is not None,
        nonfinite=finite(dict(input_gradient=inp.grad,gradients=gradients(model),results=result,loss=loss)),
        none_gradients=sorted(name for name,value in gradients(model).items() if value is None))
    state=None
    for _ in range(3):
        state,data,grads,step=update(model,optimizer,state,x,precision,observer)
        record["updates"].append(step)
    checkpoint=output/"checkpoint.pt"
    save_checkpoint(checkpoint,model,state=state,optimizer=optimizer)
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    saved=dict(parameters=payload["parameters"],optimizer=payload["optimizer"],state=payload["state"])
    record["saved_checkpoint_data"]=compare(saved,data)
    record["checkpoint_config_exact"]=payload["config"]==config.to_dict()
    observer.phase="next_forward"
    with torch.no_grad(),precision_context("cpu",precision):next_values,_=forward(model,x,state,split=True)
    expected_next=clone(next_values)
    next_state,next_data,next_grads,last=update(model,optimizer,state,x,precision,observer)
    record["original_fourth_update"]=last
    record["actual_dtype_observations"]=observer.report();observer.close()
    torch.save(dict(input=x,saved=data,next_forward=expected_next,updated=next_data,gradients=next_grads,
                    none_gradients=last["none_gradients"],config=config.to_dict()),output/"expected.pt")
    command=[sys.executable,"-I",str(Path(__file__).resolve()),"--phase","resume","--path",path,
        "--precision",precision,"--output-dir",str(output),"--wheel",args.wheel,"--venv",args.venv,
        "--source-manifest",args.source_manifest,"--execution-variant",args.execution_variant]
    completed=subprocess.run(command,cwd="/",env={k:v for k,v in os.environ.items() if k!="PYTHONPATH"},
                             text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    (output/"resume.log").write_text(completed.stdout)
    record["resume_command"],record["resume_exit_code"]=command,completed.returncode
    if not (output/"resume.json").is_file():raise RuntimeError("fresh-process resume produced no record: "+completed.stdout[-2000:])
    record["resume"]=json.loads((output/"resume.json").read_text())
    record["checkpoint_sha256"],record["expected_sha256"]=sha(checkpoint),sha(output/"expected.pt")
    policy_dtype="torch.bfloat16" if precision=="bf16" else "torch.float32"
    record["projection_dtype_passed"]=bool(record["actual_dtype_observations"]) and all(
        row["output_dtype"]==policy_dtype for row in record["actual_dtype_observations"])
    record["state"]=classify_case(record)
    atomic(output/"result.json",record)
    return record


def resume_case(args):
    from tide import load_checkpoint
    output=Path(args.output_dir)
    expected=torch.load(output/"expected.pt",map_location="cpu",weights_only=False)
    kwargs={"placement_devices":["cpu:0","cpu:1"]} if args.path=="placed" else {}
    model,state,slots=load_checkpoint(output/"checkpoint.pt",**kwargs)
    optimizer=placement_adamw(model,lr=1e-4,foreach=False);optimizer.load_state_dict(slots)
    record=dict(state="running",fresh_process=True,pid=os.getpid(),config_exact=model.config.to_dict()==expected["config"],
        checkpoint_data=compare(snapshot(model,optimizer,state),expected["saved"]))
    observer=DtypeObserver(model);observer.phase="next_forward"
    with torch.no_grad(),precision_context("cpu",args.precision):values,_=forward(model,expected["input"],state,split=True)
    record["next_forward"]=compare(clone(values),expected["next_forward"])
    state,data,grads,step=update(model,optimizer,state,expected["input"],args.precision,observer)
    record.update(actual_fourth_update=step,
        gradients=compare(grads,expected["gradients"],atol=1e-6,rtol=2e-3),
        updated=compare(data,expected["updated"],atol=1e-6,rtol=2e-3),
        none_gradients_preserved=step["none_gradients"]==expected["none_gradients"],
        actual_dtype_observations=observer.report())
    observer.close()
    record["installation"]=installation(args)
    strict=record["config_exact"] and record["checkpoint_data"]["exact"] and record["next_forward"]["exact"] and record["none_gradients_preserved"]
    policy_dtype="torch.bfloat16" if args.precision=="bf16" else "torch.float32"
    record["projection_dtype_passed"]=bool(record["actual_dtype_observations"]) and all(
        row["output_dtype"]==policy_dtype for row in record["actual_dtype_observations"])
    record["state"]=("structural_failed" if not strict or not record["projection_dtype_passed"] or any(
        structural_comparison_failure(row) for row in (record["gradients"],record["updated"])) else
        "support_deviation" if any(row["support_differences"] for row in (record["gradients"],record["updated"])) else
        "within_threshold" if record["gradients"]["within_threshold"] and record["updated"]["within_threshold"] else "finite_deviation")
    atomic(output/"resume.json",record)
    print(json.dumps(dict(path=args.path,precision=args.precision,state=record["state"])))
    return 0 if record["state"] in ("within_threshold","finite_deviation","support_deviation") else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("initial","resume"),default="initial")
    parser.add_argument("--path",choices=("batched","placed"))
    parser.add_argument("--precision",choices=("fp32","bf16"))
    parser.add_argument("--execution-variant",choices=("whole_region","training","reception"),default="whole_region",
        help="training uses prefix summaries and Reception recomputation; reception adds incoming light64, fused QKV, padding mask, doubling scan and placed batches")
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--wheel",required=True)
    parser.add_argument("--venv",required=True)
    parser.add_argument("--source-manifest",required=True)
    args=parser.parse_args()
    global torch,precision_context,placement_adamw,detach_state
    import torch
    from tide import placement_adamw,detach_state
    from tide.precision import precision_context
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    initial_installation=installation(args)
    if args.phase=="resume":return resume_case(args)
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=False)
    record=dict(state="running",schema_revision=2,scope=__doc__,argv=sys.argv,harness_sha256=sha(__file__),
                mode_comparison_tolerance=dict(atol=0.,rtol=0.,purpose="exactness observation; does not replace repository FP32/BF16 qualification thresholds"),
                source_manifest=sha(args.source_manifest),initial_installation=initial_installation,cases=[])
    atomic(output/"result.json",record)
    for path in ("batched","placed"):
        for precision in ("fp32","bf16"):
            case=output/(path+"-"+precision);case.mkdir()
            record["cases"].append(initial_case(args,path,precision,case))
            atomic(output/"result.json",record)
    record["final_installation"]=installation(args)
    states=[case["state"] for case in record["cases"]]
    record["state"]=("structural_failed" if "structural_failed" in states else "support_deviation" if "support_deviation" in states else
                     "finite_deviation" if "finite_deviation" in states else "within_threshold")
    atomic(output/"result.json",record)
    print(json.dumps(dict(state=record["state"],cases=states,output=str(output))))
    return 0 if record["state"] in ("within_threshold","finite_deviation","support_deviation") else 2


if __name__=="__main__":raise SystemExit(main())
