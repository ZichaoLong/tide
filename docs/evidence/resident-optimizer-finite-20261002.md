# Vector finite checks in the resident optimizer

Implementation `bb40cffe33c4da9d6ce173305ac8556d494dcffc` passed eight clean
qualification jobs. [Audited records](resident-optimizer-finite-20261002.json)
include source/build/loader identities, one changed CANN kernel, retained failures,
independent references, profiler CSVs and every repeated timing.
Audit: `TASK/launchers/optimizer_finite_evidence.py`.

The optimizer previously read each numerical lane with an AIV scalar instruction
to check its FP32 exponent. It now compares absolute values against FLT_MAX in
vector batches and checks packed result bits. NaN and infinity fail the ordered
comparison. The primitive requires64-lane comparison groups on this stack;
private scratch is zero-padded to that boundary, and only valid bits are checked.
No out-of-range numerical input is read. The existing256-element numerical tile,
SGD/AdamW arithmetic, finite proposal/consensus/commit transaction, int64 counters,
FP32 masters, FP16 publication check and budgets are unchanged.

The local optimizer gate passed32 trajectories per payload against independent
CPU FP32/FP64:256 FP32 updates and248 FP16 updates, preserving two expected
FP16 representability refusals. The two-card gate passed4 trajectories/32 updates
per payload. An additional504 actual optimizer transactions cover9 widths,
word/tile tails, both signs of quiet/signalling NaN and infinity, finite extremes,
subnormals and disconnected poison. Refusals preserve all live fields bytewise.
Four1,048,579-element tests also retain the allocator bounds and a final-tile
infinity refusal. Actual Python-native/installed LibTorch consumers passed24
complete-training cases, and2→3-card restore passed32 trajectories/512 windows/
128 updates. Tolerances were unchanged and no tests skipped.

A bounded single-card comparison uses16,777,219 FP32 elements, constant common
initial values and gradients, SGD momentum and AdamW. Each old/new variant runs
in **three independent processes** on the same leased device, alternating variants.
Each process performs2 warmup and5 measured continued updates. Timers bracket the
whole synchronized device optimizer transaction, including finite checks and
submission/wait. Construction and independent scalar CPU reference checks of all
resulting masters/slots are outside timing. Final values agree exactly between
old and new. Old source is1757b90; public/core/host archive and all other kernels
are unchanged, with fresh benchmark links and verified loader closure.

| Optimizer | Old process medians, ms | New process medians, ms | Median throughput ratio |
| --- | --- | --- | ---: |
| SGD momentum | 86.400,86.655,86.635 | 12.819,12.742,12.782 | 6.778× |
| AdamW | 121.886,121.799,121.804 | 17.382,17.195,17.515 | 7.007× |

These are isolated optimizer improvements, not full graph/training ratios or
original B512 evidence. AIV engine attribution alone does not imply that a
kernel's internal work is already vectorized. This concrete scalar bottleneck
was in an AIV kernel, not an observed AiCPU fallback.

A separate two-card FP16 complete-training profile recorded53,190 operators:
50,642 AI_VECTOR_CORE,1,825 AI_CORE and723 MIX_AIV; no AI_CPU row was observed.
Profiling includes construction/cleanup and is separate from the timing table.
This qualifies the local CANN9.0/Ascend910_9392 tuple, not other device generations
or toolkit versions.

Retained failures: a misspelled development check caused an early CLI refusal;
the first kernel version incorrectly passed an unrounded short count to the
comparison primitive and failed a finite trajectory; the task-local benchmark
initially used a nonexistent CLI parsing function and failed compilation before
measurement. Each remains failed with its source/logs. Corrected jobs and
benchmark sources are separately identified; no numerical tolerance was relaxed.
