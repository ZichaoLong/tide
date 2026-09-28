"""Tiny independent application; copy this directory into an experiment repo.

The generated regression task is an integration example, not model evidence.
No import from Tide's source paths, tests or benchmark scripts is permitted.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
import tidegraph
from tidegraph import GraphConfig, GraphRuntime, External


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="graph.json")
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--implementation", choices=("python", "native"))
    parser.add_argument("--native-library")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(23)
    config = GraphConfig.load(args.config)
    if args.implementation:
        config = replace(config, execution=replace(config.execution, implementation=args.implementation))
    runtime = GraphRuntime(config, device=args.device, native_library=args.native_library)
    dtype = getattr(torch, config.dtype)
    # Application data and task head belong to this repo.
    generator = torch.Generator().manual_seed(23)
    data = torch.randn(2, 4, config.width, generator=generator, dtype=dtype).to(runtime.device)
    head = torch.nn.Linear(config.width, 1, device=runtime.device, dtype=dtype)
    optimizer = torch.optim.AdamW([*runtime.model.parameters(), *head.parameters()], lr=.0001)
    session = runtime.session(2)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    losses = []
    for step in range(3):
        session.reset()  # Each example is an independent sequence.
        optimizer.zero_grad(set_to_none=True)
        if config.family == "settle":
            result = session.advance(data)
        else:
            external = [External(b,p,t,t,data[b,t]) for b in range(2)
                        for p in range(len(config.graph.inputs)) for t in range(4)]
            stop = 4 + len(config.graph.nodes) * max((e.delay for e in config.graph.edges), default=1)
            result = session.advance(external, stop=stop, sealed_until=stop)
        features = torch.stack([value for _,_,_,value in result.outputs])
        predictions = head(features)
        targets = torch.full_like(predictions, .25)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                         for p in [*runtime.model.parameters(), *head.parameters()]):
            raise RuntimeError("nonfinite consumer loss/gradient")
        session.detach()
        optimizer.step()
        losses.append(float(loss.detach()))
    # Tide checkpoint is graph-only. The application owns whole-task serialization.
    session.save(out / "graph.pt")
    torch.save(dict(head=head.state_dict(), optimizer=optimizer.state_dict(), steps=3), out / "task.pt")
    with torch.no_grad():
        session.reset()
        if config.family == "settle":
            prediction = session.advance(data[:, :1])
            if not prediction.outputs:
                raise RuntimeError("inference produced no outputs")
    report = dict(state="passed", library_path=str(Path(tidegraph.__file__).resolve()),
                  runtime=runtime.manifest(), task="external scalar regression smoke", losses=losses,
                  task_parameters=sum(p.numel() for p in head.parameters()))
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(state=report["state"], library_path=report["library_path"], losses=losses)))


if __name__ == "__main__":
    main()
