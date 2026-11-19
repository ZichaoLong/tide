# Bounded capacity and throughput search

`python -m benchmarks.search` runs one config/precision/mode and one B or L
axis per identity. It belongs inside the existing frozen `scripts/launch.py`
service; the parent launcher owns device admission, account/host resource checks
and total job bounds. The search does not acquire hardware or change shared
environments. CPU child tests do not qualify NPU capacity.

Two tables answer different questions:

- `--table common`: hold the other axis fixed and double the chosen axis. Each
  shape gets its own initial Tide audit and A–D joint matching. Every subject
  is a fresh child process with its own parameter initialization and input hash.
- `--table capacity --axis batch`: hold L fixed and freeze all A–D constructors
  at the initial common calibration shape. Each subject searches its own best B
  under identical budgets and candidate limits. This is capacity throughput,
  not a same-B speedup. B/D still refer to the calibration's actual activity;
  current-shape Tide activity and post-update changes are recorded separately.
  A baseline may fit a B where Tide was not measured or its audit failed. Such
  current Tide activity stays explicitly unmeasured, never copied from calibration.

Neither table compares model quality or identical mathematical functions.
The Transformer history window stays 128; Tide's adopted logical-time history
remains unchanged. Larger L is a longer input workload, not increased attention
history. Natural routing is the default. `sparse03`, `sparse10`, and `high` reuse
the existing explicitly synthetic FC-quad fixtures; `high` means full activity,
not high padding. Actual padding is a WorkAudit observation, not a fixture label.

Example **command templates** for an immutable source (replace the revision,
identity and CPU affinity with currently admitted resources):

```sh
python scripts/launch.py --name new-common-fc-k --revision COMMIT \
  --npu-count 1 --threads 4 --affinity AVAILABLE_CORES --seconds 10800 --rss-gib 64 -- \
  python -m benchmarks.search --config configs/fc-k.json --table common \
  --axis batch --batch 8 --length 128 --max-axis-value 64 --max-points 8 \
  --device npu:0 --precision bf16 --mode train --threads 4 \
  --device-budget-gib 52 --rss-gib 56 --seconds 9600 --child-timeout 900 \
  --output-dir '{run}/results'

python -m benchmarks.search --config configs/scale/hb-1b.json --table capacity \
  --axis batch --batch 1 --length 128 --max-axis-value 32 --max-points 8 \
  --device npu:0 --precision fp32 --mode train --threads 4 \
  --device-budget-gib 52 --rss-gib 56 --seconds 9600 --child-timeout 900 \
  --output-dir NEW_OUTPUT
```

The second command also requires the durable launcher for actual long work.
Choose a budget from current admission, not from these example numbers. Separate
identities cover FC-K, FC-S, HB Settle/Timed and HB1B; use useful single axes instead
of their full Cartesian product. `--plan-only` persists inputs and candidate bounds
without importing Torch or constructing a model. It consumes that output identity.
`--config-overrides '{"checkpoint_regions":true}'` applies a top-level JSON object
before freezing `config.json`; the current model implementation validates the
resolved options in each child. This also supports execution candidates such as
`local_attention_backend` or `region_batch_size` when present in that source.

The default is at most eight candidate shapes per search line; the hard cap is
twelve. A line stops at its explicit largest value, resource/time limit, failure,
or two stable points without at least 3% throughput improvement. A common-shape
line stops for a failed subject, or when every subject is flat. Noise is raw
sample range/median (default admissible maximum 0.25); noisy points do not prove
a plateau and are excluded from the best-stable table. This finite rule does not
prove a global maximum or confidence interval. Raw samples remain authoritative.

Every audit and subject uses `programme.run_child` ownership tracking. RSS is
sampled for that process tree and excess terminates that child. Device allocator
peak is checked after synchronized construction/audit/warmup/timed steps; excess
raises and exits the child. This is not an allocator reservation or a hard
pre-allocation bound, so OOM remains possible. Exact persistent parameter floors
can reject a candidate without execution; these rejections, observed OOM,
measured budget breaches, timeouts and other errors have separate records.
The outer search completing means its bounded control flow ended, not that all
children passed or that numerical equivalence was established.

FP32/BF16 use the declared policy: FP32 parameters, gradients and AdamW slots.
Full training includes preparation, packing, readout/loss, backward, update and
sync. Warmup updates parameters, and subsequent timed steps restart private state;
this is zero-state prefill, not stream continuation. `--no-optimizer-foreach`
selects the existing non-foreach AdamW without silently changing its dtype.
Initial audits, runtime dtype audits and final work/finite checks remain outside
normal timing. Timed peak is reset after warmup; construction/warmup/audit peaks,
per-sample cumulative allocator snapshots and child RSS are separate quantities.

`search.json` holds commands, limits, terminal failures, hashes, common-shape
comparisons or each fixed model's best observed B. Raw child JSON/logs are retained.
An audit at the current shape may be supplied directly to `benchmarks.run` with
`--reference-record`; `--calibration-record` freezes only the matching models.
Config, seed, precision, activity and flush policy must agree; a current-shape
reference additionally requires equal B/L. Both source identities are recorded.

Follow selected points with `benchmarks.continuous` for real cached decode,
within-segment full BPTT, explicit detach boundaries and checkpoint/resumed update.
Do not turn accumulated stream positions into one prefill/backward length. Its
checkpoint memory and intrusive diagnostics require separate budgets; the search
does not claim continuation or restore qualification from prefill completion.

## Separate memory-lifetime diagnostics and reports

Use `python -m benchmarks.memory_ledger` only for selected points under a separate
launcher identity. It is intrusive and produces no throughput result. For example:

```sh
python -m benchmarks.memory_ledger --config RESOLVED_CONFIG.json \
  --device npu:0 --precision bf16 --mode train --batch 4 --length 128 \
  --device-budget-gib 52 --warmup 1 --output NEW_MEMORY.json

python -m benchmarks.memory_ledger --config RESOLVED_CONFIG.json \
  --device cpu --precision fp32 --mode grad_forward --batch 1 --length 128 \
  --prefill-chunk 64 --decode 8 --continue-state --segments 3 \
  --output NEW_CONTINUED_MEMORY.json

python -m benchmarks.search_report --inputs COMMON/search.json CAPACITY/search.json \
  --output NEW_REPORT.md
```

The ledger uses the normal model, loss, optional complete Timed flush and AdamW.
Normal mode warmup is the selected mode and changes training parameters; continued
state diagnostics instead warm up independent zero-state inference. Every measured
diagnostic segment explicitly detaches state before the next one. Neither path
certifies restored updates. Transformers use `--subject transformer_A` (etc.) with
the actual `--calibration-record`; the constructor and calibration hash are retained.

Each phase separates visible payload and unique retained storage for parameters,
gradients, optimizer, private state, inputs, outputs and saved-for-backward storage.
An ordered exclusive attribution and overall storage union avoid adding aliases
twice. Saved storage is retained until the packed autograd object is destroyed:
an unpack with `retain_graph=True` does not mean it was released. The extra saved
storage metric excludes storage already owned at the incoming segment boundary.
Outer saved-tensor hooks may not see nested hook overrides or opaque backend
allocations, including some checkpoint internals. Actual allocator and RSS values
remain separate; the ledger is not a complete physical-memory accounting oracle.

Allocator-minus-visible-union is measured at the same phase boundary and labelled
unclassified live allocation. Operator scratch is **unmeasured**; the foreach
parameter-size allowance is declared separately as a planning quantity. Peaks at
different times, saved graph storage, private state and persistent owner payload
overlap and cannot be summed. Hooks, runtime auditing and retained
diagnostic outputs can change allocation lifetime, so compare these diagnostics
with the independent normal-time memory measurements instead of replacing them.
Semantic `WorkAudit` is deliberately absent from the ledger pass: its CPU support
copies would contaminate runtime/allocator observations, and checkpoint backward
recomputation could be counted again as semantic work. Use the separate normal
benchmark's initial/final semantic audit; `actual_work` stays explicitly unmeasured
in the ledger. `input_positions` is the declared real B×L denominator only.

`search_report` verifies the recorded raw result/audit/calibration/frozen-config
hashes before rendering the two tables. Failed/limited children and noisy or
unmeasured cells stay visible. It performs no model or device work, and writes via
fsynced staging and atomic replacement.
