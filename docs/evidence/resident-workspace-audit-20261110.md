# Resident storage and service audit — 2026-11-10

## Review scope

Source: `47704c9c44367ee2052cbf397a2b78cfa21314e7`; mathematics remains
[the pinned upstream](../upstream.json). The [hash-bound manifest](resident-workspace-audit-20261110-reviews.json)
records the files actually read. Its `partial_files` are only selected source ranges,
excluded from whole-file review coverage. No exhaustive C++ symbol census, new
runtime result or whole-repository qualification is claimed. This report records
the pre-change source; later implementation and qualification need their own identity.

Follow the public [resident contract](../resident-weighted-training.md),
[execution contract](../execution-flows.md) and original
[deployment failures](prefill-deployment-20261110.md) when reproducing results.

## Budget attribution

`DeviceProgram::retained_tensor_bytes` counts the union of actual `StorageImpl`
identities, including full storage bytes rather than view sizes. CANN and CUDA
both deduplicate owners, additional tensors and cooperating programs. Vendor
workspace is separate. `WeightedRowGraph::push` compares expression bytes and
local program growth as two alternative limits; it does **not** sum them.
Calling this a simple global double count would be incorrect.

There is a narrower potential accounting mismatch. `WeightedFlow::Impl::remaining`
already deducts banks, state, source tensors, all services and unallocated journal
reservation from the owner limit. A new local service then measures its own
program growth. First use of an already-owned borrowed bank enters that local
program growth even though it is not new owner storage. A local rejection can
therefore charge both genuinely new allocations and the first local reference
to borrowed storage against an allowance intended for new global storage.

A bounded witness is one row with 64 FP32 banks of shape 32×32, a 32 KiB local
allowance, and caller-owned input/owners/error/banks. Bank storage alone is
262,144 bytes. The old projection and row wrappers need under 32 KiB of new
storage, but local program growth includes the borrowed bank. Compare local
program growth against the union with caller-owned tensors held from the start.
This source-level witness is a diagnostic proposal, not yet a runtime result,
and does not establish the cause of the historical SSD refusal.

Keep four numbers separate: owner unique storage, unallocated journal reservation,
new component storage, and operator workspace high-water. A total four-part
training quota is not freely available forward memory. Do not lower a guard
without a mode-aware, calculable allocation bound and refusal tests.

## Shared projection storage

The existing projection gathers one `[out,in]` weight per physical row and stores
its transpose. VJP creates another `[rows,out,in]` outer-product tensor before
reducing by bank. In `WeightedBanks::direct`, a single parameter bank is common;
query/key/value therefore need not physically duplicate the same bank per row.
A single-bank path can save one actual copied weight, use a matrix batch over all
rows, and calculate `dW = dYᵀ X`, `dX = dY W`. Multi-bank remains separately scoped.

The existing row guard must not be reused on one shared matrix: its absent-row
sanitization writes zeros in place. With mixed present/absent rows that would
erase the shared matrix for valid rows. A separate bank-use guard must ignore
unused NaN banks, save current values for backward, and maintain row/owner
connection flags independently of numerical zeros. Existing device checks mutate
original weights and inputs after forward; any optimization must survive this.
FP16 storage still has declared FP32 arithmetic/adjoints and one output cast.

For the original FC16 B4/P16/cache49/d128/summary32, the unchanged KV query guard
is 213,600,256 bytes. The key/value row matrices and their transposes alone are
12,845,056 bytes; two saved single-bank matrices and transposes would occupy
65,536 bytes. These are source-size calculations, **not measured device peaks**.
The coarse top-level query guard would remain a separate restriction until
replaced by a justified composed bound.

## Service lifecycle

CANN creates a DeviceProgram and request/done notifications per service.
Nested parent/child calls are serial: coordinator→request→worker→done, with the
parent suspended while its child uses the shared workspace. `finish()` already
binds one workspace arena at the maximum requirement across coordinator and all
services. Sharing workspace again is not an optimization opportunity.

Service loops persist for repeated online windows. Closing services after one
phase would invalidate reuse. Stream/notification reduction requires a separate
bounded dispatcher or pool with nesting and lifetime evidence. Timeout quarantine
preserves resources when completion is unconfirmed; early release is unsafe.
Adjoint programs capture actual window/root/parameter addresses, so shape-only
caching is not valid. CUDA uses one conditional graph, not CANN worker streams.

Useful construction diagnostics include service count, maximum nesting, workspace
high-water, owner used/reserved/available bytes, action/node, and the exact local
expression that rejected its budget. These do not alter online scheduling.

## Historical failures retained

The final accounted FC16 query case had forward quota 1,275,068,416 bytes and
journal reservation 911,557,912 bytes (gross headroom 363,510,504 before other
owners); construction rejected with `resident KV qry tensor budget exceeded`.
The SSD chain had forward quota 335,544,320 bytes, journal 242,932,248 bytes
(gross headroom 92,612,072), and rejected with `local row graph tensor budget exceeded`.
Both were B4/P16/cache49/stages80. Earlier stages272 refusals remain separate.
Neither final refusal produced warmup, training, continuation or throughput.
The historical evidence lacks expression/node byte attribution, so this audit
must not relabel a hypothesis as the measured cause or a proposed fix as passed.
