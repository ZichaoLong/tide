# Clean development

Current package: `tide-instances`, import `tide`; source `src/tide`, tests `tests`.
No binary or historical experiment directory is a prerequisite.

## This host

Published environment-module baseline: `ascend/dev-workspace-9.2.0`
(CANN9.2, Torch2.12, TorchNPU2.12.post3). Load the site's modules initialization
in a fresh shell if `module` is not already defined. Then, from the checkout:

```sh
module load ascend/dev-workspace-9.2.0
python -m venv --system-site-packages .venv-instances
source .venv-instances/bin/activate
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pip install --no-deps --no-build-isolation -e .
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q
python scripts/check_upstream.py
```

The environment and install are project-private. Shared packages are used without
modification; `--no-build-isolation` uses the published setuptools and avoids an
unnecessary package download. Record the actual module, Python/Torch versions and
resolved installation in each qualification. On another host use a compatible
private Torch environment; do not infer device qualification from installation.

CPU entry points disable vendor autoload. NPU jobs use the local Ascend admission
workflow and an explicitly selected device; loading a module is not allocation.
Standalone C++ is optional future work: the published matching SDK module is
`libtorch-npu/2.12.0-cann9.2.0`, loaded in its own shell when needed.

## Run and resume

See [instances](instances.md) for the API. From an installed source checkout:

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python examples/train.py --checkpoint NEW.pt
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m benchmarks.run --config configs/fc-k.json --device cpu --output NEW.json
```

Use new output identities. The example checks mechanism, including restored
updates, and reports finite numeric deviations separately. The current benchmark
measures FP32 fresh prefill; BF16, joint matching and stateful measurement are
planned in [ROADMAP](ROADMAP.md), not implicit command capabilities.
Long work follows [execution](execution-flows.md); checks follow [verification](verification.md).
Cleanup may remove private environments and outputs; these steps rebuild them.
