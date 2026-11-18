# Fixed-start training-path checks

`scripts/check_training_paths.py` compares execution choices at one actual trained
point. It is an intrusive correctness diagnostic, with no latency or throughput
claim. The default is one configuration, one precision, and only `baseline` and
`recompute`; broader candidates must be requested explicitly.

## Fixed point and checks

A baseline performs one real AdamW update first. Every candidate then receives
identical parameters, nonempty private state, optimizer moments/steps and input
identities. The diagnostic maps placed slots through the existing original-row
ownership metadata, checks the complete canonical round trip exactly, and rejects
inconsistent shard step counters or slot presence. No production mapping API is
added. Optimizer policy is one AdamW group, lr1e-4, foreach=False.

Each candidate executes inference, two same-owner forward/VJP repeats without an
update, and two continuing real updates by default. Chunks keep full BPTT within
an update; Timed flush contributes to the objective, then the carried state is
detached. Inference and grad-enabled forward are compared separately. Canonical
parameter gradients preserve original-Parameter None semantics; actual shard
None names are additionally recorded without comparing unlike layouts by name.

The script saves a checkpoint after the continuing updates, checks an original
next forward/update, and by default launches a new Python interpreter that loads
and actually updates it. Saved data and the next forward retain exact checks;
source package/harness hashes are checked across the process boundary. The child
prepends the parent's actual Tide source and preserves existing compiler paths.
`--restore-mode same_process` is an explicitly labelled mechanism shortcut.

Candidates:

| Name | Changes relative to the supplied semantic configuration |
|---|---|
| baseline | region_batch_size=1; checkpoint_regions=False |
| batch4 | region_batch_size=4; checkpoint_regions=False |
| recompute | region_batch_size=1; checkpoint_regions=True |
| combined | region_batch_size=4; checkpoint_regions=True |
| placement_serial | region_batch_size=1; checkpoint_regions=False; complete-region placement, serial |
| placement_per_device | Same complete-region assignment; per_device scheduling |

Additional unplaced candidates can be supplied with `--candidate-overrides`, a
JSON name-to-options object. Built-in names cannot be replaced, and semantic
fields such as budget/topology are rejected. For implementations that provide the
new fields, for example:

```sh
--candidates baseline reception prefix \
--candidate-overrides '{"reception":{"checkpoint_reception":true},"prefix":{"ssd_summary_backend":"prefix"}}'
```

The allowed extra fields are region_batch_size, checkpoint_regions,
checkpoint_reception, ssd_summary_backend and local_attention_backend. Unknown
fields in the actual frozen ModelConfig fail before computation; adding this
harness option does not implement or qualify those production candidates.

`local_attention_backend` and other arithmetic/semantic settings are retained in
every candidate. In particular, combined does **not** add local math. Placement
currently remains batch1; comparing these paths does not qualify a future
placement-plus-batching implementation. CPU indices are logical groups, not
independent physical accelerators.

## Differences and bounded attribution

The original `THRESHOLDS` from `qualify_precision` remain unchanged: FP32
atol1e-6/rtol2e-3, BF16 atol2e-3/rtol5e-2. Restored FP32/BF16 updates retain the
stricter atol1e-6/rtol2e-3. CPU FP64 uses atol1e-10/rtol1e-8 throughout and remains
a finite arithmetic diagnostic, not another production precision qualification.

Every observation first has full/discrete/support hashes, support counts, dtype
counts and None identities. A changed observation retains at most eight
coordinates of its first numeric, support, time/index, None-gradient and
structure discrepancies in deterministic returned-tree order. That order does
not establish the first causal operator. KV-time witnesses also show paired key
weights; stored-time changes alone do not prove an illegal attention visibility
mask. Work counts are separate from numerical equality.

At most one differing unplaced candidate receives an optional internal replay
using the existing `diagnose_region_batch.Trace`, dot/projection replays and
Decimal quad witness. Capture defaults to two million elements per path;
`--witness-elements` sets an explicit cap from 1 to 20 million. The requested cap
is recorded even if no internal replay is needed. Only selected tensors are
saved to `first-witness.pt`. `--no-internal-witness`
skips it. Replays explicitly retain their capture-match checks; unmatched replay,
capture limit, or an exact forward with divergent VJP leaves attribution open.
Concurrent placement keeps its returned-field witness and same-owner repeats;
this script does not present a serial instrumented path as the measured
per_device execution.

`structural_failed` means an observed metadata/dtype/None check failed, not a
proof of a mathematical implementation bug. A rounded support change can cause
later KV metadata changes; the original failures and first witnesses remain.
`exact_contract_failed` concerns saved/source/next-forward exact requirements.
Nonfinite values and runtime failures are explicit failures. Finite numerical
failures retain thresholds and are never silently converted to exact claims.
The original reports are neither overwritten nor reclassified by this facility.

## Bounded invocation

Use one new output directory. CLI bounds are B1–8, L1–64, one to four continuing
updates, one to sixteen ATen threads, and an explicit parameter cap (20M by
default). A restore child has a 180s timeout. These guards do not replace the
outer frozen background.slice time/RSS and device admission bounds. Restoring
requires a duplicate small model; do not use this diagnostic as a large-model
capacity benchmark. The main agent owns NPU admission and shared resources.

For a short CPU check, use the matching private Python and prepend the frozen
checkout source while preserving the published environment's other paths:

```sh
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 taskset -c 64-67 python -m scripts.check_training_paths \
  --config configs/hb-multiscale-x-settle.json --device cpu --precision fp32 \
  --candidates baseline recompute --batch 1 --length 4 --chunk 2 --steps 2 \
  --threads 1 --output-dir artifacts/NEW-training-paths-cpu
```

For the explicit four-path NPU matrix, run **inside an admitted frozen service**:

```sh
python -m scripts.check_training_paths \
  --config configs/hb-multiscale-x-timed.json --device npu:0 --precision bf16 \
  --candidates baseline batch4 recompute combined \
  --batch 1 --length 4 --chunk 2 --steps 2 --threads 1 \
  --output-dir artifacts/NEW-training-paths-npu
```

A two-device mechanism comparison uses `--device npu:0 --placement-devices npu:0
npu:1 --candidates baseline placement_serial placement_per_device`; use a separate
output. `--deterministic` explicitly enables deterministic algorithms;
`--no-deterministic` explicitly disables them. Omission preserves the default.
The requested/effective policy is recorded and propagated to restore. Unsupported
operators produce recorded runtime failures; there is no silent fallback or
threshold change. A deterministic comparison is a separate bounded observation
and does not rewrite an earlier nondeterministic result.

The targeted tests are `tests/test_training_paths.py`. They include distinct
optimizer-row values, corrupted shard counters, support/time/None classification,
FP64 continued updates, fresh-process source/restore checks, logical CPU placement,
and the pure BF16 batch4+checkpoint combination. They validate the facility and
mechanisms; formal numerical outcomes still belong to frozen run records.

## Selected Attention mode witness

`scripts/diagnose_attention_modes.py` reuses a training-path `anchor.pt` and its
adjacent `result.json`: exact parameters, nonempty private state and AdamW slots,
`inputs[1]`, chunk boundary, deterministic policy and unchanged production package
hash. It accepts the original FP32 record only and performs no update/backward.
The original mode comparison remains attached to the new result unchanged.

The first inference pass retains only a bounded in-memory index of call hashes.
The grad-enabled pass stops at the first Attention call with changed input or
output values/layout. One inference prefix replay then recovers that call and
must reproduce its original hash. Only this pair is saved in `witness.pt`; the
full call index is not an output event log. Earlier matching Attention calls do
not establish equality of every intervening non-Attention operation.

The selected call records q/k/v/prior/mask/output tensors and hashes, each input's
requires_grad, actual native/custom-forward dispatch, and native QKV shapes/dtypes.
NPU fusion observations distinguish actual PSE from padded rank-one Q/K or unit
prior paths. Two isolated calls use the same captured inference inputs with each
mode's original requires_grad flags; capture-match checks qualify their relevance.
At most eight complete query rows by default (maximum 32) are independently
evaluated with CPU scalar FP64 dot products and weighted exponential quotients.
All visible keys and output coordinates of those rows are included, with no
production Attention/matmul/softmax/autograd in the oracle. This is a local
forward arithmetic witness, not a whole-model FP64 or VJP qualification.

Bounds are at most three model forward attempts, two primitive replays, 4,096
Attention calls per attempt by default (maximum 8,192), and two million captured
input/output elements per call (maximum 20 million). `--capture-elements`,
`--max-calls` and `--oracle-rows` make those finite choices explicit. A capture
limit, changed inference repeat or unmatched invocation remains inconclusive;
there is no retry expansion or production policy change. Source/anchor hashes,
requested device, resource identity and original thresholds are recorded.

Inside an admitted frozen NPU service:

```sh
python -m scripts.diagnose_attention_modes \
  --anchor /ABS/PREVIOUS-training-paths/results/anchor.pt \
  --device npu:0 --output-dir /ABS/NEW-attention-mode-witness
```

`tests/test_attention_mode_diagnosis.py` checks the independent weighted quotient,
zero-prior native/tiled observations, strict capture bounds, and an injected
mode error at a real small training anchor. These short CPU method checks do not
stand in for running the selected original NPU fixture.
