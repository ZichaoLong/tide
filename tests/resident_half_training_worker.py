"""Independent process resumes a complete FP16 payload / FP32 master checkpoint."""
import argparse
import torch
from resident_half_training_cases import runtime, limits, update


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", required=True)
    p.add_argument("--memory", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    r = runtime("pdg", a.device, memory=a.memory, mode="softp")
    values = (torch.arange(16).reshape(1, 4, 4)*.005).half()
    with torch.no_grad(), r.training_session(1, checkpoint=a.checkpoint, limits=limits()) as session:
        update(session, values, 2, 4)
        session.save(a.output)


if __name__ == "__main__":
    main()
