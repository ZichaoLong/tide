# Resident KV bank through Read, Full and reverse

Fixed implementation **0c974ccd4921999e6b7f6292888bfcb3478a33e1** passes two
independent NPU builds, 56 directed CPU tests, 11 standalone LibTorch processes
and four installed Python/native processes. All five durable jobs are terminal
passed with exit 0. This is small directed correctness/training qualification;
**there is no new throughput, TTFT or scale measurement**.
The [machine record](resident-kv-owner-20261112.json) contains exact commands,
build/binary/wheel identities, test names, process logs, device admission and
launch scripts. Mathematics remains fixed at 0770593.

## Change and actual default path

`BlockKVLocal` now uses one actual initial/write bank `[B*N,C+L,width]`, snapshot
indices, bank-based Mem/Full, accumulated bank adjoints and one write VJP. The
default numerical path does not decode floating old/proposal prefixes. Query
weights and visibility still occupy `[B*N,L,C+L]`; index tables and metadata are
int64. Values are not expanded across queries. Read0 retains its independent
bound initial bank, Read− takes old, Read+ and Full take their declared snapshots.
See the [representation and mathematical correspondence](../resident-kv-bank.md).

The public `kv-bank-views-v1` interface declares each raw field's shape, presence
and validity before materialization. Named cotangents need no forward decode.
Explicit `observe` decodes requested actual values, without a reference recurrence;
complete Python event records call this boundary. Final raw continuation and
schema-2 checkpoint remain exact. The dense-snapshot standalone component APIs
remain isolated comparisons; their passes do not substitute for the new default
graph tests.

## What was checked

| Gate | Result and scope |
| --- | --- |
| CPU | 56 passed: bank/list recurrence, expiry, huge clocks, empty/no-write/Read0, opaque padding, independent old/proposal/final/zero roots, pool/position/read formulas, roots without decode, complete-record refusal, resources/default dispatch |
| Python-owned NPU build | Passed, new weighted plugin and wheel; exact unchanged core source/binary hashes verified before reuse |
| Standalone NPU build | Passed, eight selected check executables; published SDK kept separate from Python TorchNPU |
| SDK components | Services, Read-selection, dense KV messages/sequence, actual default graph flow and actual Model lowering passed |
| SDK default flow | Independent scalar formula, complete old/proposal raw decode, ten root sets for Read+, all/none roots for Read−/Read0; B2/L2, duplicate edges/ports, shared parameters and huge clocks |
| SDK session | Two-window BPTT, independent formula/VJP, three AdamW steps and exact fresh-owner raw restore passed |
| SDK public qry/sum | Each actual Model/independent Streaming consumer passed three two-window AdamW steps, canonical/input connectivity, actual continuation and a separate fresh-process restore |
| Installed Python qry/sum | Each independent Python schedule passed three AdamW steps and separate fresh-process restore; explicitly decoded old/proposal values and independently rooted proposal keys/values are included |

The Python installed fixture is FC(2,2), B2/L3, d2/ds2/rank1/FFN3, FP32,
KV qry or sum, adapted sources, regional_total with k2 and gamma_max1.6,
including missing inputs and a present zero-strength input. Full and qry Read
have position frequencies. The SDK public fixture adds nonempty raw import,
staggered positions, duplicate physical edges, heterogeneous source hidden/rank
geometry and shared owners. All fixtures are tracked source, not retired data.

Small consumer comparisons retain their existing rtol8e-5/atol8e-6. The stronger
CPU formula checks have their own fixed assertions. No tolerance was raised to
pass this qualification. Previously recorded host FP32 strict failures at c6
remain failed in [their evidence](read-region-owner-20261112.md); this resident
gate neither reruns nor relabels them.

## Resources and reproduction

The two builds ran in separate background.slice services on CPU0–159/160–319,
each with 512 GiB and 2400 seconds. They finished in about three minutes each;
these are build durations, not model performance. The CPU gate used CPU48–51
while builds were finishing; it is not a timing benchmark. Device gates used
CPU24–27/28–31, one ATen/BLAS thread, 1200-second service and 300-second process
limits. Cooperative two-sample idle admission selected physical NPU11 for native
and NPU5 for SDK, each mapped to logical npu:0. No shared-card override was needed.

The resident source was a read-only detached checkout. Both core builds came
from c6afc20c; the build script verified its exact C++ source hash and each binary
hash before reuse. Python used the published CANN9.2/TorchNPU2.12 environment;
SDK used LibTorch-NPU2.12/CANN9.2. Neither shared toolchain was modified.

From a clean checkout at the fixed implementation, follow [development](../development.md)
to build matching core libraries and install the package. Build the weighted
Python component with `build_device_control.py --python-component weighted` and
the SDK with `--checks services resident-block-selection resident-block-kv-messages
resident-block-kv-sequence resident-block-settle-flow resident-block-settle-session
resident-block-public resident-block-model`. Both use the site's explicit SoC.
Run those component checks on an admitted device. Run the installed
`examples/consumer_core5/block_resident.py` with qry/sum and a separate
`--restore-check` process for each; keep its `block_observations.py` beside it.
The machine record preserves full argv and scripts; replace site/output paths,
not fixtures or assertions. No old run directory is needed to reconstruct them.

## Limits retained

The new bank is FP32, KV sum/qry and one device within the block owner's aligned
Settle domain. CPU component witnesses cover finite-window expiry and complete
eviction; the new full-flow device fixture uses a nonempty unbounded-window state.
It does not independently qualify whole-device-graph expiry for every combination.

Dense score/incidence tables, scalar position/metadata/weight kernels, per-call
program construction and observation decoding remain costs. No end-to-end speedup
is claimed from the new storage representation. Half, recurrent memories, multiple
owners and CUDA build/device execution remain separate work. The existing broad
general engine is not replaced with a narrower semantic contract.

Development caught ctypes argument-count/CPU scratch-arena issues and a branch
fixture whose decimal sum crossed W=1 between FP32/FP64; those harness details
remain in the development subsection of the machine record. They were corrected
before the fixed implementation; they are not formal-device failures hidden by
a later run. The allocation estimate was also corrected for known shared and
temporary arrays before freezing.

This closes the resident boundary selected for this round. P1–P6 overall remain
incomplete; the current handoff stops for user alignment before another round.
