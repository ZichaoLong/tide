"""Untimed actual new-owner resume, separate from cross-schedule equivalence."""
import torch


def check(candidate, create, config, args, device, out):
    from tidegraph.checkpoint import save, load
    from tidegraph.compare import equivalent
    from weighted_benchmark_runners import Eager, Resident
    path = out/"resume.pt"
    binding, inputs, span = create()
    if isinstance(candidate, Resident):
        with torch.no_grad():
            candidate.owner.save(path)
            restored = Resident(binding, inputs, span, config, args, device, checkpoint=path)
    else:
        save(path, candidate.binding.graph, candidate.binding.model, candidate.state, candidate.optimizer)
        restored = Eager(binding, inputs, span, config, args, device)
    try:
        if isinstance(candidate, Resident):
            with torch.no_grad():
                equivalent(candidate.owner.checkpoint(), restored.owner.checkpoint(), atol=0, rtol=0)
        else:
            restored.state = load(path, binding.graph, binding.model, restored.optimizer)
            equivalent(candidate.state, restored.state, atol=0, rtol=0)
            equivalent(candidate.optimizer.state_dict(), restored.optimizer.state_dict(), atol=0, rtol=0)
            # Auxiliary eager update counts are not PyTorch checkpoint fields.
            candidate.update_counts = dict.fromkeys(candidate.update_counts, 0)
        a = candidate.step(args.steps+1)
        b = restored.step(args.steps+1)
        equivalent(a, b, atol=0, rtol=0)
        equivalent(candidate.parameters(), restored.parameters(), atol=0, rtol=0)
        equivalent(candidate.training_state(), restored.training_state(), atol=0, rtol=0)
    finally:
        restored.close()
    return dict(state="passed", checkpoint="resume.pt", next_window=args.steps+1, exact=True,
                scope="same schedule/placement, raw decode and new-window records/parameter gradients/SGD slots; outside timing")
