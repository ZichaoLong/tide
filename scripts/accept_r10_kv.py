"""Replay the interrupted Timed/K A1 from its official checkpoint on the repaired KV base."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from accept_r09_shannon import owners, update


ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / ".local/pilot-r3/campaign-r10"
DIAGNOSIS = ROOT / "docs/evidence/pilot-r3-r10-kv-no-write.json"


def read(path):
    return json.loads(Path(path).read_text())


def replay(args):
    import torch
    from pilotlab import worker3 as w
    from pilotlab.data3 import DocumentStore3
    from pilotlab.worker import compare_tree

    evidence = read(DIAGNOSIS)
    checked = {}
    for key in ("task", "original_result", "failure_capture", "official_checkpoint"):
        source = evidence[key]
        path = ROOT / source["path"]
        if w.sha256(path) != source["sha256"]:
            raise ValueError(f"diagnostic input changed: {key}")
        checked[key] = path
    old = read(checked["task"])
    task = copy.deepcopy(old)
    task.pop("stage_a_migration", None)
    task["id"] += "-kv-replay"
    identity = task["source_identity"]
    files = identity["source_files"]
    identity["source_files"] = {k:v for k,v in files.items() if Path(k).is_relative_to(ROUND / "source")}
    base = args.foundation_src.resolve() / "tide"
    base_files = {str(p.relative_to(base)):w.sha256(p) for p in sorted(base.rglob("*.py"))}
    identity["source_files"].update({str(base/k):v for k,v in base_files.items()})
    config = read(ROUND / "campaign-plan.json")["config"]
    data = Path(config["data"])
    w.validate_source_identity(task, data)
    device = w.initialize("npu:0", task)
    store = DocumentStore3(data)
    model, opt = owners(task, store, device)
    fault = evidence["failure_step"]
    save_step = fault + 2
    end_step = save_step + 1
    if args.phase == "direct":
        original = read(checked["original_result"])
        ref = original["checkpoint"]
        saved, origin = w.load_checkpoint(checked["official_checkpoint"], old, device, ref["sha256"])
        if origin["fallback_used"] or saved["state_sha256"] != ref["state_sha256"] or saved["steps"] != evidence["official_checkpoint"]["step"]:
            raise ValueError("failed to load the exact official checkpoint")
        state, stream = w.restore_owners(saved, model, opt, store, old, device)
        capture = torch.load(checked["failure_capture"], map_location="cpu", weights_only=False)
        report = {"official_checkpoint":ref,"original_state_loaded_exact":True, "original_error":original["error"],
                  "updates":[], "files_sha256":base_files, "physical_device":os.getenv("ASCEND_RT_VISIBLE_DEVICES")}
        tokens = saved["tokens"]
        for step in range(saved["steps"]+1, end_step+1):
            batch = stream.next()
            if step == fault:
                if not torch.equal(batch[0], capture["inputs"]) or not torch.equal(batch[1], capture["targets"]):
                    raise ValueError("original failed batch differs from the live stream")
                report["fault_batch_matched"] = True
            state, trees, info = update(model,opt,state,batch,task,device,tokens)
            tokens += batch[1].numel()
            report["updates"].append(dict(info, step=step,tokens=tokens))
            if step == save_step:
                payload = w.checkpoint_payload(model,opt,state,stream,task,device,tokens,step,saved["initialization"])
                report["new_checkpoint"] = w.save_checkpoint(args.output,payload)
        w.atomic_torch_save(args.output/"expected.pt", trees)
        report.update(stream_sha256=w.digest_tree(stream.state_dict()),rng_sha256=w.digest_tree(w.rng_state(device)))
    else:
        direct = read(args.output/"direct.json")
        ref = direct["new_checkpoint"]
        saved, origin = w.load_checkpoint(ref["path"],task,device,ref["sha256"])
        if origin["fallback_used"] or saved["steps"] != save_step or saved["state_sha256"]!=ref["state_sha256"]:
            raise ValueError("failed to load the new checkpoint")
        state, stream = w.restore_owners(saved,model,opt,store,task,device)
        _, actual, info = update(model,opt,state,stream.next(),task,device,saved["tokens"])
        prior = direct["updates"][-1]
        expected = torch.load(args.output/"expected.pt",map_location="cpu",weights_only=False)
        comparisons = {k:compare_tree(v,actual[k]) for k,v in expected.items()}
        scalars = {k:abs(prior[k]-info[k]) for k in ("loss","objective","grad_norm")}
        discrete = (all(info[k]==prior[k] for k in ("input_sha256","reset","counts","gradient_tensors","lr"))
                    and w.digest_tree(stream.state_dict())==direct["stream_sha256"]
                    and w.digest_tree(w.rng_state(device))==direct["rng_sha256"])
        passed = discrete and all(v["within_tolerance"] for v in comparisons.values()) and all(
            math.isclose(info[k],prior[k],abs_tol=1e-5,rel_tol=1e-4) for k in scalars)
        report = {"step":end_step,"checkpoint_sha256":origin["sha256"],"loaded_state_exact":True,"discrete_exact":discrete,
                  "scalar_abs_diffs":scalars,"comparison":comparisons,"passed":passed}
    w.atomic_json(args.output/(args.phase+".json"),report)
    if not report.get("passed",True):
        raise ArithmeticError("fresh-process successor outside existing tolerance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-src", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("direct","restored"))
    args = parser.parse_args()
    if args.phase:
        sys.path[:0] = [str(ROUND/"source"),str(args.foundation_src.resolve())]
        replay(args)
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        for phase in ("direct","restored"):
            subprocess.run([sys.executable, __file__, *sys.argv[1:],"--phase",phase],check=True)
