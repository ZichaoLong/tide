import os
from pathlib import Path
import random
import sys
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
sys.path.insert(0, str(Path(os.environ.get("TIDE_BUILD_DIR", Path(__file__).resolve().parents[1] / "build"))))
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(int(os.environ.get("TIDE_TEST_SEED", "7")))
random.seed(int(os.environ.get("TIDE_TEST_SEED", "7")))


def pytest_addoption(parser):
    parser.addoption("--dtype", choices=("float32", "float64", "both"), default="both")


def pytest_generate_tests(metafunc):
    if "dtype" in metafunc.fixturenames:
        selected = metafunc.config.getoption("dtype")
        names = ("float64", "float32") if selected == "both" else (selected,)
        metafunc.parametrize("dtype", [getattr(torch, n) for n in names], ids=names)
