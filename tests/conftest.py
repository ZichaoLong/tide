import torch


def pytest_sessionstart(session):
    # Tiny formula cases benefit from one ATen thread; benchmark pools are separate.
    torch.set_num_threads(1)
