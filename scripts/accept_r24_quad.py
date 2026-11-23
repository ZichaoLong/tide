"""Qualify quad from the recorded CPU failure and fresh same-backend checkpoint owners."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

from accept_r09_shannon import owners, update
from accept_r11_quad import checked, read


def load_origin(worker, payload, model, optimizer, store, old, device):
    # load_state_dict can leave CPU AdamW tensors aliased to its input.
    payload = {**payload, "optimizer":copy.deepcopy(payload["optimizer"])}
    if device.type == "cpu":
        return worker.restore_owners(payload, model, optimizer, store, old, device)
    import torch
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    state = worker.tree_map(payload["private_state"], lambda x:x.to(device))
    stream = worker.StreamBatch(store, old, payload["stream"])
    worker.restore_rng(payload["rng"], torch.device("cpu"))
    torch.npu.manual_seed(old["seed"])
    fields = dict(model=model.state_dict(),optimizer=optimizer.state_dict(),private_state=state,
                  stream=stream.state_dict(),rng={**worker.rng_state(device),"npu":None})
    assert all(worker.digest_tree(v)==worker.digest_tree(payload[k]) for k,v in fields.items())
    return state,stream


def compare_capture(w, task, old, cap, payload, store, device):
    import torch
    from tide import quad, region
    from pilotlab.worker import compare_tree
    import r24_quad_reference as reference

    original = region.allocate
    observed = dict(calls=0,rows=0,positive_mismatches=0,lower_face_mismatches=0,max_abs=0.,
                    disputed_projection_rows=0,exact_projection_rows=0,exact_total_rows=0)
    recover_rows, exact_row = quad.recover_rows, quad._recover
    def count_disputed(scores, w, lower, upper, budget, mask, target):
        observed["disputed_projection_rows"] += int(mask.sum())
        return recover_rows(scores, w, lower, upper, budget, mask, target)
    def count_exact(row, width, dtype, *, totals=False):
        observed["exact_total_rows" if totals else "exact_projection_rows"] += 1
        return exact_row(row, width, dtype, totals=totals)
    def audited(*args, **kwargs):
        result = original(*args, **kwargs)
        with torch.no_grad():
            expected,lf = reference.allocate(*args, **{**kwargs,"return_lower_face":True})
            c,face = result if isinstance(result,tuple) else (result,None)
            observed["calls"] += 1
            observed["rows"] += c.numel()//c.shape[-1]
            observed["positive_mismatches"] += int(((c>0)!=(expected>0)).sum())
            if face is not None:
                observed["lower_face_mismatches"] += int((face!=lf).sum())
            observed["max_abs"] = max(observed["max_abs"],float((c-expected).abs().max()))
            torch.testing.assert_close(c,expected,atol=1e-6,rtol=5e-6)
        return result

    def reproduce(fn):
        model,opt = owners(task,store,device)
        state,stream = load_origin(w,payload,model,opt,store,old,device)
        batch = stream.next()
        assert torch.equal(batch[0],cap["inputs"]) and torch.equal(batch[1],cap["targets"])
        assert w.digest_tree(None if batch[2] else state)==w.digest_tree(cap["private_state"])
        extra = {}
        def capture_logits(module,args,out):
            extra["logits"] = out.logits.detach().cpu().clone()
        handle = model.register_forward_hook(capture_logits)
        clip = torch.nn.utils.clip_grad_norm_
        def capture_before_clip(*args,**kw):
            extra["before_clip"] = w.cpu_snapshot({name:p.grad for name,p in model.named_parameters()})
            return clip(*args,**kw)
        with patch.object(region,"allocate",fn), patch.object(torch.nn.utils,"clip_grad_norm_",capture_before_clip):
            _,trees,info = update(model,opt,state,batch,task,device,payload["tokens"])
        handle.remove()
        return {**trees,**extra},info

    expected, first = reproduce(reference.allocate)
    with patch.object(quad,"recover_rows",count_disputed), patch.object(quad,"_recover",count_exact):
        actual, second = reproduce(audited)
    comparisons = {k:compare_tree(expected[k],v) for k,v in actual.items()}
    metrics = all(math.isclose(first[k],second[k],rel_tol=1e-4,abs_tol=1e-5) for k in ("objective","loss","grad_norm"))
    passed = (all(x["within_tolerance"] for x in comparisons.values()) and metrics and
              not observed["positive_mismatches"] and not observed["lower_face_mismatches"])
    return dict(passed=passed,allocation=observed,comparison=comparisons,metrics=second,reference_metrics=first)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release",type=Path,required=True)
    p.add_argument("--phase",choices=("capture","direct","restored"),required=True)
    p.add_argument("--device",choices=("cpu","npu:0"),required=True)
    args = p.parse_args()
    release = args.release.resolve()
    root = release.parents[2]
    source = release.parent/"campaign-r24/source"
    base = release/"base/src"
    sys.path[:0] = [str(source),str(base)]
    import torch
    import tide
    from pilotlab import worker3 as w
    from pilotlab.data3 import DocumentStore3
    from pilotlab.worker import compare_tree

    diag = read(root/"docs/evidence/pilot-r3-r24-quad-total-constraint.json")
    cfg = read(source.parent/"campaign-plan.json")["config"]
    old = read(checked(diag["observed"]["task"]))
    result = read(checked(diag["files"]["official_result"]))
    cp = result["checkpoint"]
    assert cp == diag["files"]["checkpoint"] and result["failure_capture"]==diag["files"]["capture"]
    cap = torch.load(checked(result["failure_capture"]),map_location="cpu",weights_only=False)
    assert cap["task"]==old and cap["completed_steps"]==cp["steps"] and cap["update_in_progress"]
    assert Path(tide.__file__).resolve().parent==base/"tide" and Path(w.__file__).resolve().parents[1]==source
    files = {f.relative_to(base/"tide").as_posix():w.sha256(f) for f in (base/"tide").rglob("*.py")}
    digest = w.canonical_digest(files)
    assert digest == (release/"source-digest").read_text().strip()
    task = copy.deepcopy(old)
    task.pop("stage_a_migration",None)
    task.update(id=old["id"]+"-quad-qualification",device=args.device)
    identity = task["source_identity"]
    identity["source_files"] = {k:v for k,v in identity["source_files"].items() if Path(k).is_relative_to(source)}
    identity["source_files"].update({str(base/"tide"/k):v for k,v in files.items()})
    w.validate_source_identity(task,Path(cfg["data"]))
    device = w.initialize(args.device,task)
    store = DocumentStore3(Path(cfg["data"]))
    output = release/"replay"/device.type
    output.mkdir(parents=True,exist_ok=True)
    report_path = output/f"{device.type}-{args.phase}.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    fault = cp["steps"]+1
    record = dict(pid=os.getpid(),device=str(device),physical_id=os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
                  files_digest=digest,fault_step=fault,actual_module=tide.__file__)
    if args.phase!="restored":
        saved,receipt = w.load_checkpoint(cp["path"],old,"cpu",cp["sha256"])
        assert not receipt["fallback_used"] and saved["state_sha256"]==cp["state_sha256"]
    if args.phase=="capture":
        record.update(compare_capture(w,task,old,cap,saved,store,device))
    elif args.phase=="direct":
        model,opt = owners(task,store,device)
        state,stream = load_origin(w,saved,model,opt,store,old,device)
        record.update(official_checkpoint=cp,original_state_loaded_exact=device.type=="cpu",
                      portable_origin_fields_equal=True,origin_backend="cpu",original_error=result["error"],updates=[])
        tokens = cp["tokens"]
        for step in range(fault,fault+3):
            batch = stream.next()
            if step==fault:
                record["fault_batch_matched"] = torch.equal(batch[0],cap["inputs"]) and torch.equal(batch[1],cap["targets"])
                assert record["fault_batch_matched"] and w.digest_tree(None if batch[2] else state)==w.digest_tree(cap["private_state"])
                record["fault_state_reset"] = batch[2]
            if step==fault+2:
                state,trees,info = update(model,opt,state,batch,task,device,tokens)
                w.atomic_torch_save(output/"expected.pt",trees)
            else:
                x,y,reset = batch
                out = w.train_update(model,opt,x,y,None if reset else state,device,task,tokens)
                w.synchronize(device)
                info = {k:out[k] for k in ("loss","objective","grad_norm","lr")}
                state = out["state"]
            tokens += batch[1].numel()
            assert all(math.isfinite(info[k]) for k in ("loss","objective","grad_norm","lr"))
            record["updates"].append(dict(info,step=step,tokens=tokens))
            if step==fault+1:
                fresh = w.checkpoint_payload(model,opt,state,stream,task,device,tokens,step,saved["initialization"])
                record["new_checkpoint"] = w.save_checkpoint(output,fresh)
            print(device,step,info["loss"],flush=True)
        record.update(passed=True,stream_sha256=w.digest_tree(stream.state_dict()),rng_sha256=w.digest_tree(w.rng_state(device)))
    else:
        direct = read(output/f"{device.type}-direct.json")
        ref = direct["new_checkpoint"]
        saved,receipt = w.load_checkpoint(ref["path"],task,device,ref["sha256"])
        assert direct["pid"]!=os.getpid() and not receipt["fallback_used"] and saved["state_sha256"]==ref["state_sha256"]
        model,opt = owners(task,store,device)
        state,stream = w.restore_owners(saved,model,opt,store,task,device)
        _,trees,info = update(model,opt,state,stream.next(),task,device,saved["tokens"])
        expected = torch.load(output/"expected.pt",map_location="cpu",weights_only=False)
        checks = {k:compare_tree(expected[k],v) for k,v in trees.items()}
        prior = direct["updates"][-1]
        discrete = (all(prior[k]==info[k] for k in ("input_sha256","reset","counts","lr","gradient_tensors"))
                    and w.digest_tree(stream.state_dict())==direct["stream_sha256"]
                    and w.digest_tree(w.rng_state(device))==direct["rng_sha256"])
        metrics = all(math.isclose(prior[k],info[k],rel_tol=1e-4,abs_tol=1e-5) for k in ("loss","objective","grad_norm"))
        record.update(step=saved["steps"]+1,loaded_state_exact=True,checkpoint_sha256=ref["sha256"],
                      discrete_exact=discrete,metrics=info,comparison=checks,
                      passed=discrete and metrics and all(x["within_tolerance"] for x in checks.values()))
    w.atomic_json(report_path,record)
    print(report_path,record["passed"],flush=True)
    if not record["passed"]:
        raise ArithmeticError("fixed update outside the existing FP32 or discrete acceptance")


if __name__=="__main__":
    main()
