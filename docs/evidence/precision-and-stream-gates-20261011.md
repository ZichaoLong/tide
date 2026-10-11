# Control precision and component stream gates — 2026-10-11

Clean **00b1775ce15afa20fb0552d51f6afdf4392ccc1c** passes the corrected standalone
scale consumer: **5 CTests,8 CPU and8 NPU complete training cases**, covering
Add/Attention × SGD/AdamW × FP32/FP16,each with three connected windows and
independent forward/gradient/update checks. [Audited receipts](precision-and-stream-gates-20261011.json)
retain commands,binary/source identities,logs,three updates per case and terminal
resource records. These are standalone LibTorch cases,not Python/native calls.

The selection-only model now supplies the declared control metadata for every
region. FP16/FP32 payload uses FP32 controls; CPU FP64 keeps FP64 controls.
The new regression linked against the old adapter fails at the declared dtype
check. Real payload/parameter ownership is unchanged. Standalone scale FP16 uses
its previously documented0.002/0.02 gate; public module tolerances are unchanged.
The original training08 FP16 assertion remains failed.

Clean **e3c86d70795c594eafbc587388d9e984c3f92612** separately passes the corrected
State/Read checker in NPU FP32 and FP16. Each checks **48 configurations,336
windows and18 refusals**,including exact clocks above2^55,both caller streams,
scalar/vector execution,continuation and failed transaction publication.

Resident10 passed33 component invocations then failed the original State/Read
repeat-bound assertion. A finite diagnostic preserved exact int64 writes and
passed both unsynchronized FP32 and synchronized FP32/FP16 checks,so the original
failure was **not deterministically reproduced**. Source inspection confirms the
checker omitted the required Torch-input/private-stream boundary. The correction
adds that boundary,error-code diagnostics and nondefault-stream coverage.
Production ContentFlow already synchronizes; no kernel,algorithm or tolerance
changed. The original failed resident receipt remains intact.

The changed checker was freshly compiled and linked from the clean commit
against unchanged,hashed production libraries whose original build source is
retained. This is not a claim that all production libraries were rebuilt.
The same clean checker also compiled/linked against the CUDA stack and passed
its help entry; **no actual GPU execution** occurred.

All four qualification services are terminal with zero exits,empty cgroups and
no guarded survivors. NPU jobs used authorized shared cards and recorded mapping
and occupancy. Peak summed RSS:scale CPU3,224,432,640B,NPU training4,831,240,192B,
State/Read4,867,731,456B,CUDA host1,649,061,888B. Shared mappings are counted more
than once; these are sampled aggregate limits,not kernel allocation guarantees.

The remaining resident inventory/public clients form a separate gate,not covered
by this report. Correctness job durations and incidental metrics are not new
performance evidence. Original near-tie strict failures and GPU target work remain.
