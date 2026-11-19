# Clean installed-wheel CPU smoke

Source: `0992b467584c1f603314ed04d8efe8596a14f576`; fixed mathematics:
`4f5b9f20a4030d68ee66ae2d26e34659a5e37234`.
[Reviewed machine-readable evidence](roadmap-installed-20261118.json.gz) includes
source/input hashes, exact commands, isolated harness contents, environment,
lifecycle, wheel files and all observations.

`wheel-installed-cpu-v15` built the wheel with `--no-deps --no-build-isolation`
and no index/cache, installed it in a new private `--system-site-packages` venv,
and ran from **`/`**, with `PYTHONPATH` removed and Python `-I`. All 17 installed
Python package files match the wheel; all 15 loaded `tide` modules resolve inside
that venv. Python 3.12.13/Torch 2.12.0+cpu come from the published baseline;
public packages were only read. The immutable source contents remain unchanged.

Wheel: `tide_instances-0.1.0-py3-none-any.whl` (47,165 bytes).
SHA256: `1b3349526257e87daf1177d88c5a97ea30484b402f88080d00a72b4f521eb0e4`.

The bounded FC-K smoke uses d64, 8 nodes, 4 layers, B2/L8. FP32 and BF16 each
perform an audited finite inference, three continuing AdamW updates through the
installed public API, checkpoint save/restore and a real next update on both
original and restored owners. Model parameters/AdamW slots and the next readout
restore exactly. This reuses the frozen `examples/train.py`; full private-state
and broader precision qualification remain in the separate numerical reports.

| Precision | Actual inference arithmetic | Three update losses | Restored updated parameters |
|---|---|---|---|
| FP32 | FP32 | 3.30140, 2.41763, 2.30952 | finite deviation; max abs `4.999339580535889e-5` |
| BF16 | BF16 with FP32 controls/masters | 3.29593, 2.40533, 2.30924 | within original threshold; max abs `2.117455005645752e-5` |

The unchanged restored-update threshold is `atol=1e-6, rtol=2e-3`.
FP32 retains failures at `regional.0.second.bias` and `regional.1.second.bias`;
both precisions' optimizer-slot comparisons are within threshold. These finite
results do not relabel previous failures or claim bitwise resumed training.

The verified `background.slice` service used cores 224–227, 4 threads, a
600-second runtime limit and an 8 GiB sampled process-tree RSS bound. It finished
in 9.259 seconds with peak sampled RSS 547,110,912 bytes; the unit is inactive
with exit code 0. The account quota, active jobs and resource headroom were checked
before submission. No NPU, public environment or downstream repository was changed.
The exact launcher/build/install/smoke commands and reconstructible harness are
preserved in the JSON; this is installation and execution evidence, not a new
performance matrix or strict numerical-equivalence pass.

A separate clean frozen checkout of the same `0992b467` passed **424 CPU tests**
in 84.87 seconds (`suite-cpu-v15`). This uses the checkout's independent formulas,
VJPs, continuation/update/checkpoint, ownership, capacity-budget and lifecycle
checks; it is distinct from the wheel's installed public-API smoke above. The
CPU-only job used cores 228–231, four threads, 900 seconds and a 32 GiB sampled
RSS bound. Its exact command, terminal status and log hashes are included in the
JSON. Neither the test count nor this execution exit relabels formal device
numerical deviations.
