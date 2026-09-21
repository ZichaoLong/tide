import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def pytest_addoption(parser):
    parser.addoption("--dtype", choices=("float32", "float64", "both"), default="both")


def pytest_generate_tests(metafunc):
    if "dtype" in metafunc.fixturenames:
        selected = metafunc.config.getoption("dtype")
        names = ("float64", "float32") if selected == "both" else (selected,)
        metafunc.parametrize("dtype", [getattr(torch, n) for n in names], ids=names)
