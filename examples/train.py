"""Installed-public-API training/continuation example; no task or quality claim."""
import argparse
from pathlib import Path

import torch
from tide import ModelConfig,TopologyConfig,TideModel,detach_state,save_checkpoint,load_checkpoint
from tide.precision import precision_context, precision_policy


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--memory",choices=("K","S","X"),default="K")
    parser.add_argument("--precision",choices=("fp32","bf16"),default="fp32")
    parser.add_argument("--steps",type=int,default=3)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--checkpoint",type=Path,required=True)
    args=parser.parse_args()
    if args.steps<1 or args.threads<1:
        parser.error("steps and threads must be positive")
    if args.checkpoint.exists():
        parser.error("choose a new checkpoint path")
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
        torch.npu.set_device(args.device)
    torch.set_num_threads(args.threads);torch.manual_seed(17)
    model=TideModel(ModelConfig(topology=TopologyConfig(n=8,layers=4),d=64,memory=args.memory)).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,foreach=True)
    state=None
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with precision_context(args.device,args.precision):
            result=model(torch.randn(2,8,64,device=args.device),state)
            loss=result.readout().float().square().mean()  # Mechanism-only scalar objective.
        loss.backward();optimizer.step();state=detach_state(result.state)
        print({"step":step,"loss":float(loss.detach())},flush=True)
    save_checkpoint(args.checkpoint,model,state=state,optimizer=optimizer)
    restored,restored_state,opt_state=load_checkpoint(args.checkpoint,device=args.device)
    next_input=torch.randn(2,8,64,device=args.device)
    with torch.no_grad(),precision_context(args.device,args.precision):
        torch.testing.assert_close(restored(next_input,restored_state).readout(),
                                   model(next_input,state).readout(),rtol=0,atol=0)
    restored_optimizer=torch.optim.AdamW(restored.parameters(),lr=1e-4,foreach=True)
    restored_optimizer.load_state_dict(opt_state)
    torch.testing.assert_close(restored.state_dict(),model.state_dict(),rtol=0,atol=0)
    torch.testing.assert_close(restored_optimizer.state_dict(),optimizer.state_dict(),
                               rtol=0,atol=0,check_device=False)
    for current,step_optimizer,current_state in ((model,optimizer,state),
                                                (restored,restored_optimizer,restored_state)):
        step_optimizer.zero_grad(set_to_none=True)
        with precision_context(args.device,args.precision):
            objective=current(next_input,current_state).readout().float().square().mean()
        objective.backward()
        step_optimizer.step()
    # Exact decoding does not make parallel FP32 gradient reductions bitwise.
    tolerance=dict(atol=1e-6,rtol=2e-3)
    failures=[];max_abs=0.0
    for (name,a),(_,b) in zip(restored.named_parameters(),model.named_parameters()):
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise ArithmeticError("non-finite resumed parameter: "+name)
        delta=(a-b).abs();maximum=float(delta.max().detach());max_abs=max(max_abs,maximum)
        if bool((delta>tolerance["atol"]+tolerance["rtol"]*b.abs()).any()):
            failures.append({"parameter":name,"max_abs":maximum})
    for current_optimizer in (optimizer,restored_optimizer):
        for entries in current_optimizer.state.values():
            for value in entries.values():
                if isinstance(value,torch.Tensor) and not bool(torch.isfinite(value).all()):
                    raise ArithmeticError("non-finite resumed optimizer state")
    optimizer_difference=None
    try:
        torch.testing.assert_close(restored_optimizer.state_dict(),optimizer.state_dict(),
                                   **tolerance,check_device=False)
    except AssertionError as error:
        optimizer_difference=str(error)
    print({"checkpoint":str(args.checkpoint),"restored_optimizer":bool(opt_state),
           "precision":precision_policy(args.device,args.precision),
           "continuation":"exact","next_optimizer_update":
           "finite_deviation" if failures or optimizer_difference else "within_threshold",
           "update_threshold":tolerance,"parameter_max_abs":max_abs,
           "parameter_threshold_failures":failures,"optimizer_difference":optimizer_difference})


if __name__=="__main__":
    main()
