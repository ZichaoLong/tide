# Actual resident model consumers

Implementation `0d61cb94e7b911ff883de9af7b81042f87897a36` passed six immutable-source
jobs. [Machine-readable audit](online-resident-consumers-20261002.json) records
source, installed libraries, binaries, loader closure, raw observations and traces.
The public consumer executes actual per-edge Add/Attention models independently
from common parameters and input rules; no reference intermediate feeds it.

| Gate | Result |
| --- | --- |
| Installed CPU and standalone resident clients | Two fresh successful builds/links; unchanged core and qualified resident libraries reused after byte verification |
| Affected CPU consumer/loss/optimizer checks | 59 passed, no skips |
| Resident Python client and standalone LibTorch | 19 passed, no skips |
| Existing CPU/NPU mixed consumers | 18 passed, no skips |
| Separate two-device actual Attention training profile | Passed; four continued windows, two complete AdamW updates |

The resident gate includes 12 training trajectories across three families, both
Add/Attention and both clients; four delayed continuous inference trajectories;
two unified CLI success/refusal cases; and an independent integer/gradient case
beyond `2^54` with vocabulary17. The16 model trajectories cover64 windows and24
updates. Tests compare states, pending, routes, output roots, complete learned
gradients including None/zero, parameters and continuation against an independent
CPU schedule. No tolerance was relaxed. Training exercises multiple devices;
inference is single-device. Payload is FP32.

The standalone profile uses D32 Attention with71,128 learned parameters,
diagnostics off and explicit reverse chunk4. It observes15,004 AI_VECTOR_CORE,
633 AI_CORE and218 MIX_AIV tasks, with **no observed AiCPU**. Device task records
include32 model executions,570 matching notify record/wait pairs,2,370 device
switches and4,552 asynchronous copies. Construction and consumer boundaries are
included. This trace establishes execution placement, not full-size throughput
or a bottleneck-time attribution.

Five failed development attempts remain failed: insufficient divided forward
workspace, reverse-packet budget, two retained builtin-handle startup refusals,
and D32 reverse16-row workspace refusal. The last workload passed after selecting
physical chunk4 without changing logical windows, loss reduction or updates.
Automatic budget-aware reverse chunk selection remains pending at this revision.
An additional fixed-source mixed launch omitted the explicit target and skipped
all18 checks; it is excluded from qualification and preserved in the audit.
Corrected `online-resident-mixed-clean02` ran and passed all18 on NPU.

Standalone and Python-owned NPU runtimes remain isolated. Both reuse the resident
implementation at `96c75f8d9c2bee54a5000f4c410fe3d5764ec552`; this is a verified
unchanged dependency, not a claim that the backend was recompiled at0d61cb9.

This qualification does not establish compact projection ownership, consumer
FP16, multi-device inference, total-memory admission, full-size throughput or
another hardware/toolchain version. Graph scheduling is device-controlled;
host input preparation and output compaction at window boundaries remain part
of the complete consumer time. Python resident remains a client of C++/CANN.

Reproduction uses the retained task-local `launchers/online_resident_evidence.py`
with the full implementation hash. Raw runs are `build-online-resident-cpu-clean01`,
`build-online-resident-clean01`, `online-resident-cpu-clean01`,
`online-resident-matrix-clean01`, `online-resident-mixed-clean02` and
`online-resident-profile-clean01` under the project artifact root.
