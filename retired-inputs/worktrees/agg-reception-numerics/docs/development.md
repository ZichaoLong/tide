# Clean development

Current package: `tide-instances`, import `tide`; source `src/tide`, tests `tests`.
No binary or historical experiment directory is a prerequisite.

## This host

Published environment-module baseline: `ascend/dev-workspace-9.2.0`
(CANN9.2, Torch2.12, TorchNPU2.12.post3). Load the site's modules initialization
in a fresh shell if `module` is not already defined. Then, from the checkout:

```sh
source /usr/share/Modules/init/bash
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
For frozen jobs, prepend the selected Tide `src` to the module-provided
`PYTHONPATH` while preserving its other entries, including CANN's TBE Python
paths. Replacing the entire path can hide installed compiler modules; that is a
launcher environment defect, not evidence that the public baseline needs changes.
Record the effective launch environment with each run. The CPU installed-wheel
smoke intentionally removes `PYTHONPATH` to verify package isolation; it does not
qualify an accelerator compiler environment.
The optional [standalone regional scoring slice](native-regional.md) uses the
published matching SDK module `libtorch-npu/2.12.0-cann9.2.0` in its own shell.
It is a measured subpath experiment, not a complete native Tide backend.
Its [CPU/NPU checks and performance](evidence/native-regional-performance-20261118.md)
retain finite failures and distinguish outer workers from ATen intra-op threads.

## Installed-package validation

The wheel contains `tide`; `benchmarks`, `scripts` and `examples` are checkout
tools. A source-tree import is not evidence that the wheel was installed correctly.
Build a copy of the frozen source with `pip wheel --no-deps --no-build-isolation`,
install that wheel with `--no-deps` into a new private venv, and run the public API
outside the checkout with `PYTHONPATH` removed and Python `-I`. Check that loaded
`tide` modules resolve inside the new venv and match the wheel contents.
Use project artifact directories for build/install outputs; keep the frozen input
unchanged and the shared baseline read-only.

The [current installed-wheel report](evidence/capacity-installed-20261118.md) fixes
production source `8a22530d`, checks 21 package files/modules, and exercises optional
bindings, batching/recomputation and logical CPU placement with actual new-process
fourth updates. Batched FP32/BF16 updates are exact; placed FP32 is within the original
tolerance and placed BF16 retains finite gradient/update deviations. The original
harness exit2 and corrected classification are recorded separately.

The [integrated CPU record](evidence/capacity-validation-20261118.md) identifies
544 full-suite tests and a separate 17-test integration check. Earlier
[installed results](evidence/roadmap-installed-20261118.md) retain their source and
failures. Neither installation nor test completion extends hardware or numerical
qualification beyond those observations.

## Run and resume

See [instances](instances.md) for the API. From an installed source checkout:

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python examples/train.py --precision fp32 --checkpoint NEW-fp32.pt
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python examples/train.py --precision bf16 --checkpoint NEW-bf16.pt
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m benchmarks.run --config configs/fc-k.json --device cpu --precision bf16 --threads 8 --output NEW.json
```

Use new output identities. The example checks mechanism, including restored
updates, and reports finite numeric deviations separately. Use
`python -m benchmarks.continuous --help` for real cached prefill/decode and
complete stream training, and `python -m scripts.qualify_precision --help` for
the explicit FP32/BF16 policy and fixed-fixture comparisons. Source qualification
remains scoped to [fixed numerical evidence](evidence/roadmap-numerics-followup-20261118.md)
and [STATUS](STATUS.md). `scripts.check_continuation` owns the bounded whole/chunk
full-backward witnesses; `scripts.diagnose_restore` separates same-owner backward
variability from restored-owner comparisons. These diagnostics preserve thresholds
and do not prescribe a downstream training recipe.
`benchmarks.run` fixes four joint Transformer matches from each mode's initial
work audit. The [comparison guide](benchmarking.md) defines effective touches,
synthetic support fixtures, timing/memory boundaries and finite suite commands.
Long work follows [execution](execution-flows.md); checks follow [verification](verification.md).
Optional multi-NPU use follows [placement](placement.md); its host-staged path,
direct transport failure and finite comparisons have separate evidence. CUDA
hardware and resident execution remain unqualified; see the
[bounded TorchAir result](evidence/roadmap-resident-20261118.md).
Retire private environments and outputs only after the ownership/dependency checks
in [artifact retention](artifact-retention.md); these setup steps rebuild them.
