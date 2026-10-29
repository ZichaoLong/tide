"""Untimed actual new-owner resume, separate from cross-schedule equivalence."""
import torch


def check(candidate, create, config, args, device, out):
    from tidegraph.checkpoint import save, load
    from tidegraph.compare import equivalent
    from weighted_benchmark_runners import Eager, Resident
    path = out/"resume.pt"
    if isinstance(candidate, Resident):
        # A restart never needs two simultaneous device programs/banks. Preserve
        # the actual CPU checkpoint and next-window observations before closing.
        with torch.no_grad():
            candidate.owner.save(path)
            saved = candidate.owner.checkpoint()
        actual = candidate.step(args.steps+1)
        parameters, training = candidate.parameters(), candidate.training_state()
        candidate.close()
        binding, inputs, span = create()
        restored = Resident(binding, inputs, span, config, args, device, checkpoint=path)
        try:
            with torch.no_grad():
                equivalent(saved, restored.owner.checkpoint(), atol=0, rtol=0)
            equivalent(actual, restored.step(args.steps+1), atol=0, rtol=0)
            equivalent(parameters, restored.parameters(), atol=0, rtol=0)
            equivalent(training, restored.training_state(), atol=0, rtol=0)
        finally:
            restored.close()
        return dict(state="passed", checkpoint="resume.pt", next_window=args.steps+1, exact=True,
                    scope="serial new-owner raw decode and complete next-window/gradient/SGD/master comparison; outside timing")
    binding, inputs, span = create()
    save(path, candidate.binding.graph, candidate.binding.model, candidate.state, candidate.optimizer)
    restored = Eager(binding, inputs, span, config, args, device)
    try:
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
