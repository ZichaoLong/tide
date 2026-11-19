"""Public deterministic inputs and explicitly synthetic quad support fixtures."""
import hashlib

import torch


def paired_input(batch, length, width, seed):
    """The same (sample, position, feature) has the same value at any width.

    Jointly matched Transformers change width, so their feature tensors cannot
    be identical. Feature-wise seeded CPU generation pairs every shared prefix
    exactly; the full tensor hash still records each actual workload.
    """
    return torch.stack([torch.randn(batch, length, generator=torch.Generator().manual_seed(seed+i))
                        for i in range(width)], -1)


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def activity_fixture(model, kind):
    if kind == "natural":
        return dict(kind=kind, synthetic=False)
    if kind not in {"sparse03", "sparse10", "high"}:
        raise ValueError("unknown activity fixture")
    if model.config.allocation != "quad" or model.graph.config.kind != "fc":
        raise ValueError("synthetic support fixtures require the legal FC quad binding")
    fraction = {"sparse03": .03, "sparse10": .10, "high": 1.0}[kind]
    selected = max(1, round(model.graph.config.n*fraction))
    # Equal chosen scores with legal k=1 may produce e.g. three 1/3 senders.
    # k is never changed to the requested support count. The closed-form
    # fixture is an initialization, not a support mask or trained sparsity.
    with torch.no_grad():
        model.describe.readout.value.zero_()
        for region in model.regional:
            region.readout.zero_()
        model.describe.offset.value.fill_(-16)
        for members in model.graph.regions:
            for node in members[:selected]:
                model.describe.offset.value[model.description_ids[node]] = 16
    return dict(kind=kind, synthetic=True, requested_fraction=fraction,
                designated_nodes_per_region=selected, nodes_per_region=model.graph.config.n,
                legal_mass_budget=model.config.budget or max(1, model.graph.config.n//32),
                construction="zero descriptor/regional score readouts; first selected nodes offset +16, others -16; unchanged legal quad budget and random CPU input",
                qualification="measure actual support; construction is not trained sparsity and training may change it")
