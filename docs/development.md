# Development

Python package: src/tide. Tests: tests. No dependency on the retired executor.

CPU checks with an existing matching Torch environment:

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pip install --no-deps -e .
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q
python scripts/check_upstream.py
```

Local published stack: module ascend/dev-workspace-9.2.0 (CANN9.2, Torch2.12,
TorchNPU2.12.post3). Use a private venv with system-site-packages; do not change
public packages. Standalone C++ uses module libtorch-npu/2.12.0-cann9.2.0 in a
separate shell. Long commands follow execution-flows.md.
