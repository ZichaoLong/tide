"""Architecture §2: source-owner incidence, separate from numerical payloads.

Integer multiplicity counts physical edges. A source owner is shared only for
the same mass/value tensor identities and exact logical row domain.
"""
from dataclasses import dataclass, replace
import torch


@dataclass(frozen=True)
class SourceIncidence:
    rows: tuple
    sources: tuple
    receivers: tuple
    incidence: tuple
    physical_counts: tuple

    @classmethod
    def build(cls, entries):
        sources, keys, incidence, receivers, physical_counts = [], {}, [], [], []
        for atoms, values, rows in entries:
            receivers.append(tuple(rows.tolist()))
            counts = {}
            for (p, _), value in zip(atoms, values):
                domain = tuple(p.rows.tolist())
                key = (domain, id(p.strength), id(value))
                if key not in keys:
                    keys[key] = len(sources)
                    sources.append((domain, p.strength, value))
                slot = keys[key]
                counts[slot] = counts.get(slot, 0)+1
            incidence.append(counts)
            physical_counts.append(len(atoms))
        rows = tuple(sorted({r for domain in receivers for r in domain}))
        return cls(rows, tuple(sources), tuple(receivers),
                   tuple(tuple(counts.get(s, 0) for s in range(len(sources))) for counts in incidence),
                   tuple(physical_counts))

    @property
    def originals(self):
        return tuple(t for _, mass, value in self.sources for t in (mass, value))

    @property
    def lengths(self):
        return tuple(len(domain) for domain, _, _ in self.sources)

    def bytes(self):
        s, r, g = len(self.sources), len(self.rows), len(self.receivers)
        value = self.sources[0][2]
        d, size = value.shape[-1], value.element_size()
        # Unique payload, one centered receiver tile, forward/reverse fields,
        # indices and incidence. No receiver x source x row x D allocation.
        return s*r*((18*d+18)*size+24)+g*r*(6*d+8)*size+g*s*(8+size)

    def metadata(self):
        # Only saved_tensors owns numerical roots across backward. Do not keep
        # extra input/producer graphs alive after autograd releases its saves.
        return replace(self, sources=tuple((domain, None, None) for domain, _, _ in self.sources))

    def tensors(self, device):
        row_index = {r: i for i, r in enumerate(self.rows)}
        sentinel = sum(self.lengths)
        index = torch.full((len(self.sources), len(self.rows)), sentinel, dtype=torch.int64, device="cpu")
        inverse, offset = [], 0
        present = [[0]*len(self.rows) for _ in self.receivers]
        for s, (domain, _, _) in enumerate(self.sources):
            positions = [row_index[r] for r in domain]
            index[s, positions] = torch.arange(len(domain), device="cpu")+offset
            inverse.extend(s*len(self.rows)+i for i in positions)
            for g, counts in enumerate(self.incidence):
                if counts[s]:
                    for i in positions:
                        present[g][i] += 1
            offset += len(domain)
        publish = tuple(torch.tensor([row_index[r] for r in domain], device=device, dtype=torch.int64)
                        for domain in self.receivers)
        return (index.to(device), torch.tensor(inverse, device=device, dtype=torch.int64),
                torch.tensor(self.incidence, device=device, dtype=torch.int64), publish,
                torch.tensor([[n == 1 for n in row] for row in present], device=device, dtype=torch.bool))
