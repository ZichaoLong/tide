"""Narrow strict FP32 failures for the fixed seed-271/general-port fixtures.

These bounds classify expected failures, not passing tolerances. Discrete
support, None/zero, raw states and same-material FP64 remain strict. Raw reports:
artifacts/columns-20261111/strict-fp32-columns; reviewed evidence follows the
fixed implementation commit. Unexpected fields or larger errors still fail.
"""
import torch


def kv(dtype, mode, read, window):
    if dtype != torch.float32:
        return {}
    fields = {("0", "sum", 0): ("content", 1.2e-5),
              ("-", "qry", 7): ("read_program.evidence_weight", 2.4e-5),
              ("+", "sum", 0): ("read_program.evidence_weight", 3.2e-5),
              ("+", "qry", 0): ("content", 1.8e-5)}
    match = fields.get((mode, read, window))
    return {} if match is None else {f"gradient[{match[0]}]": match[1]}


def settle(dtype, mode, implementation):
    if dtype != torch.float32:
        return {}
    fields = {
        "regional_total": {"3.content": 2.4e-5},
        "regional_threshold": {"3.content": 2.4e-5},
        "regional_total_zero": {"nodes.2.read_program.evidence_weight": 3.2e-5,
                                "nodes.2.aggregate_program.transforms.0.embedding": 2.6e-6},
        "hard_selection": {"nodes.0.kernel.base": 1.4e-5, "nodes.3.full_program.content.out": 2.3e-6},
        "proportional": {"nodes.0.kernel.summary": 1.6e-5, "nodes.3.full_program.content.out": 2e-6}
                        if implementation == "python" else {}}
    return {f"gradient[{n}]": limit for n, limit in fields.get(mode, {}).items()}
