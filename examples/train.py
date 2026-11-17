"""Installed-public-API training/continuation example; no task or quality claim."""
import argparse
from pathlib import Path

import torch
from tide import ModelConfig,TopologyConfig,TideModel,detach_state,save_checkpoint,load_checkpoint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--memory",choices=("K","S","X"),default="K")
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
        result=model(torch.randn(2,8,64,device=args.device),state)
        loss=result.readout().square().mean()  # Mechanism-only scalar objective.
        loss.backward();optimizer.step();state=detach_state(result.state)
        print({"step":step,"loss":float(loss.detach())},flush=True)
    save_checkpoint(args.checkpoint,model,state=state,optimizer=optimizer)
    restored,restored_state,opt_state=load_checkpoint(args.checkpoint,device=args.device)
    next_input=torch.randn(2,8,64,device=args.device)
    with torch.no_grad():
        torch.testing.assert_close(restored(next_input,restored_state).readout(),
                                   model(next_input,state).readout(),rtol=0,atol=0)
    print({"checkpoint":str(args.checkpoint),"restored_optimizer":bool(opt_state),"continuation":"exact"})


if __name__=="__main__":
    main()
