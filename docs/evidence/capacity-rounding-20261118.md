# Regional batching: bounded first-divergence witnesses

The corrected batching path still changes some BF16 trajectories. This investigation
locates specific operator rounding differences before the first observed support
change, and independently verifies the allocation decisions on each side. It does
not relabel the original execution-v3/v4 failures, relax thresholds or change model
code. All records and capture hashes are in the
[JSON evidence](capacity-rounding-20261118.json.gz).

The earlier `bmm(BF16) + bias(FP32)` promotion was an implementation error and was
fixed before these witnesses. The corrected BF16 operations can still have different
rounding boundaries. A zero-weight/bias=1.001 probe verifies that case and dtype;
it cannot establish the complete rounding behavior for nonzero products.

## Fixed identity and coverage

The diagnostic is [`scripts/diagnose_region_batch.py`](../../scripts/diagnose_region_batch.py),
committed at `a4509760bc21a0378a9289f90058ece351e02168`, based on the corrected
`062a8ab6` implementation. The CPU Settle run used this clean commit. NPU Settle and
Timed ran the integrated source `8a22530dae69d9f1fa5ead4a01d1798fd1dfe0ee`; all three
used the same diagnostic SHA-256 `4661cab1994db9a0f617f264ba56bb7a5a7cc36416e004cbf20babf6f85fe0be`.
Parameters and paired inputs share their recorded hashes, seeds 1107/1108, B1/L8,
compact-9 Multiscale X, d128, region-shared parameters and batch size four.

Capture is intrusive, initially under no-grad BF16, with Timed flush where needed.
It records regional input, projection, scoring and control values; it separately
replays the selected operator shapes with identical captured inputs. Every selected
local replay matches its corresponding full-model capture exactly. No earlier
shape difference precedes the reported first numeric difference. Later selected-query
padding shape differences are counted separately and do not become semantic failures
merely because the internal padding differs.

## CPU: two rounds versus one

Settle first differs in region 4's first regional FFN projection. Incoming values,
descriptions, Q/K/V, SDPA result, O projection and normalization are identical.
Both first-projection outputs are BF16. The actual serial `F.linear` dispatch for
this non-contiguous multidimensional input is `aten.bmm` followed by `aten.add`,
both in BF16. The batched dispatch is one BF16 `aten.baddbmm`.

The independent 90-digit Decimal dot for the selected element is
`0.2532052695751190185546875`; its BF16 product rounds to `0.25390625`.
The already-BF16 bias is `-0.042236328125`:

- Serial: round the product, add bias, then round → `0.2119140625`.
- Batched: exact product plus bias is `0.2109689414501190185546875`, then round → `0.2109375`.

Both scalar results exactly match their captured outputs. Of 8,192 first-projection
entries, 2,282 differ; maximum difference is `0.0078125`. The final region score
differs by at most `0.0014069080352783203`. The two-round interpretation is specific
to this observed CPU dispatch and input layout.

## NPU: adjacent results near rounding midpoints

NPU Settle also first differs in region 4's first FFN projection, after identical
Q/K/V, NPU fusion Attention, O and normalization. Here serial dispatch is BF16
`aten.addmm`, while batched dispatch is BF16 `aten.baddbmm`. Only one of 8,192 entries
differs, by `9.5367431640625e-7`; the region score differs in one entry by
`5.960464477539063e-8`.

The selected independent dot+bias is `-0.0001569092273712158203125`, only
`-2.9802322387695312e-8` from the midpoint between the observed adjacent BF16
outputs. Serial gives `-0.000156402587890625`; batched gives the nearest result
`-0.00015735626220703125`. Explicit two-stage BF16 rounding would give zero, so
the CPU explanation does not apply to this NPU element. Equal operands and the
near-midpoint result locate a finite matrix-operation rounding sensitivity; the
trace does not expose the internal hardware accumulation order or dtype.

NPU Timed first differs earlier, at region 11's SSD `ffn.down`. A uniform shared
base uses `mm` for four nodes; the mixed batch uses grouped `bmm` for eight. The
selected node inputs and effective matrix are exactly equal after quantization.
The private adapter stages match; the base alone differs in one of 8,192 entries,
by one BF16 step (`0.00048828125`).

For its 512-term dot, Decimal gives
`0.071533251903019845485687255859375`, only `4.877801984548569e-8` above the midpoint
`0.071533203125`. Serial produces the nearest `0.07177734375`, batched produces
the adjacent `0.0712890625`. This is a local matrix-arithmetic witness with the
same mathematical operands, not evidence of changed node ownership or indexing.

## Independent support decisions

The existing independent Decimal breakpoint solver consumes each side's actual
recorded score and received-mass row. It reproduces the entire positive-support
pattern on both sides of each first support change:

| Case | First support position `[node,B,time]` | Serial selected margin | Batched selected margin |
|---|---|---:|---:|
| CPU Settle, region 18 | `[7,0,0]` | `+0.0023159931103388467` | `-0.0030318677425384523` |
| NPU Settle, region 30 | `[6,0,5]` | `-0.006051540374755859` | `+0.0003735542297363281` |
| NPU Timed, region 26 | `[2,0,20]` | `-0.0016460021336873372` | `+0.00227433443069458` |

These are actual crossings of the declared allocation boundary for the two rounded
inputs. The investigation separately identifies an early local numeric difference
and the first support decision; it does not claim that one isolated scalar accounts
for every intervening matrix operation, later gradient or training KV timestamp
difference. Batching remains an optional execution choice with the retained
configuration/precision qualification limits.

## Retained native four-worker restored-update deviation

The separate standalone regional NPU same-stream checker completed 29 cases: 28
passed and `native-bfloat16-restored-w4` retained one failing tensor,
`grad.parameter.v_weight`, at `rtol=2e-5, atol=2e-6`. The maximum difference is
`7.62939453125e-6`. Against the serial native result, four entries differ and one
exceeds tolerance: `[9,18]` is `0.0009860992431640625` versus `0.0009784698486328125`.

A read-only comparison found one- and two-worker three-step and restored results
bitwise equal. Before the fourth update, four-worker parameters differ only in
20 `second_weight` entries (maximum `2.9802322387695312e-8`) and one `second_bias`
entry (`1.1641532182693481e-10`). All BF16 parameter casts are equal, but this second
matrix and scalar readout deliberately execute in FP32. Restored output differences
are at most `5.960464477539063e-8`.

One bounded CPU serial replay used each recorded three-step parameter set with the
same fixture and reference operators. It **did not reproduce** the selected
`v_weight` difference: both CPU gradients were `0.0009970664978027344`. The replay
tests sensitivity of that gradient; it does not resume the optimizer. Small initial
FP32 differences and a later BF16-scale discrepancy are observed, but shared
contribution accumulation, backend backward arithmetic and their propagation have
not been isolated. The original failure remains unresolved and is not assigned a
universal cause.

## Execution, identity and reproduction

CPU work used affinity 72–75 and one ATen/OMP thread. The NPU job used the cooperative
queue, physical 5 → logical `npu:0`, affinity 136–139 and one thread, the published
Python Torch 2.12/CANN 9.2 stack, and disabled HF32. It ended normally after 28.32
seconds, sampled process-tree RSS peak 2.284 GiB. These are diagnostic jobs, with
no performance conclusion or new accumulator claim.

NPU raw records retain `source_dirty=true`: `git status` from the archive directory
walked into the integration workspace, whose user untracked files were preserved.
Rather than rewriting that flag, this report separately verified all 23 relevant
frozen Tide/diagnostic files byte-for-byte against declared commit `8a22530d`; all
matched. The environment record locates the loaded Tide module in that archive.

```bash
python -m scripts.diagnose_region_batch \
  --config configs/hb-multiscale-x-settle.json --device cpu \
  --precision bf16 --threads 1 --output-dir NEW_OUTPUT
# Under current cooperative NPU admission, use --device npu:0.
# For the Timed witness, use configs/hb-multiscale-x-timed.json.
```

Rebuild from the recorded revision, use a new output path and preserve the selected
environment. The JSON lists exact capture, launcher, environment, lifecycle and
source-check paths/hashes. CPU captures are under the `region-diagnosis` worktree;
NPU captures are `artifacts/regional-rounding-npu-v1/{settle,timed}`. The native
comparison retains `artifacts/native-regional-npu-same-v1/checks`. No original
execution-v3/v4 result was modified.
