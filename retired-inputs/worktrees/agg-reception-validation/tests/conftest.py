import pytest
import torch


def pytest_sessionstart(session):
    # Tiny formula cases benefit from one ATen thread; benchmark pools are separate.
    torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def isolate_torch_thread_count():
    # CLI checks can select their own thread pool. Restore the caller's setting
    # so subsequent formula and exact-restore tests keep the declared CPU scope.
    threads = torch.get_num_threads()
    try:
        yield
    finally:
        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)
