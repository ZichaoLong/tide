"""Built-in physical inputs and output-square roots as whole B×L×P blocks."""
import torch


class BlockInputs:
    def __init__(self, plan):
        from ..planning.input_pattern import coordinates
        self.length = plan["positions"]
        batch, length, ports, width = plan["samples"], self.length, len(plan["graph"]["inputs"]), plan["width"]
        self.positions = torch.arange(length, dtype=torch.int64).expand(batch, length).clone()
        self.present = torch.zeros((batch, length, ports), dtype=torch.bool)
        for b, t, p, _ in coordinates(plan):
            self.present[b, t, p] = True
        self.input_positions = int(self.present.any(-1).sum())
        self.input_atoms = int(self.present.sum())
        phase = torch.arange(width)*.19
        # Same seeded material as materialize_inputs/ColumnInputs. Values repeat
        # across windows; only original logical positions and occurrences move.
        content = .2*torch.sin(phase[None, None, :]
            + torch.tensor([.3*t for t in range(length)])[None, :, None]
            + torch.tensor([.1*b for b in range(batch)])[:, None, None])
        payload = torch.cat((torch.ones((batch, length, 1)), content), -1)
        self.payload = payload[:, :, None, :].expand(batch, length, ports, width+1)

    def window(self, index):
        return self.positions+index*self.length, self.payload, self.present


def output_loss(fields, scale, *, training):
    """Σ_(physical output) scale‖cG‖²; strength is multiplied exactly once."""
    c, g = fields["output.strength"], fields["output.content"]
    strength, content = c.values.float(), g.values.float()
    numeric = strength[..., None]*content
    loss = torch.where(c.present[..., None], numeric.square(), torch.zeros_like(numeric)).sum()*scale
    if not training:
        return loss, None
    # Explicit cotangents of this consumer-owned loss, not a second graph VJP.
    strength_bar = torch.where(c.present, 2*scale*strength*content.square().sum(-1), torch.zeros_like(strength))
    content_bar = torch.where(g.present[..., None], 2*scale*strength.square()[..., None]*content,
                              torch.zeros_like(content))
    return loss, {"output.strength": strength_bar, "output.content": content_bar}
