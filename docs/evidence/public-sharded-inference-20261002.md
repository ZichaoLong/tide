# Public multi-device resident inference

Implementation `7329c71f5f48267fed23a1656a821ae059364b2f` passed seven fixed-source
jobs. The [machine-readable audit](public-sharded-inference-20261002.json) ties
builds, installed libraries, clients, tests, model observations and the separate
profile to that source. No test was skipped and no tolerance was relaxed.

| Gate | Result |
| --- | --- |
| Standalone/Python-owned backend and installed standalone consumer | Three successful builds/links; unchanged dependencies and development objects reused after recursive source/header/byte checks |
| Affected CPU public sessions | 74 passed, FP64/FP32 |
| Resident library | 47 passed, including 18 new placement/continuation/refusal checks |
| Actual Add/Attention consumers | 27 passed across Python-owned and standalone runtimes |
| Separate two-device D32 Attention inference profile | Four continued windows, diagnostics disabled |

The new library cases cover all three families, both schedules and FP32/FP16;
PDG feedback, parallel edges, nonaligned delays and present-zero inputs; separate
Full/state owners; event/fiber caches; and complete-cut export, reset and load.
Two-device sessions resume with one or three devices and another schedule.
Requested placement survives reset/load; resolved ownership is reported. The
lean cases forbid construction of a training owner and implicit CPU continuation
export. Existing single-device session and mutation/refusal checks also pass.

The actual consumer gate compares 12 multi-device inference trajectories and
12 complete training regression trajectories with independent CPU schedules.
These cover Add/Attention, PDG/TimedDAG/Settle and both installed clients, with
96 windows and24 optimizer updates in total. Full observables, loss, gradients,
parameters and continuation retain their existing comparisons. CLI failure
records and the `>2^54` integer/gradient boundary remain covered. These actual
consumer trajectories use FP32; FP16 here qualifies the Python-owned public
library, not an actual FP16 head/embedding consumer.

The separate standalone trace observes4,790 AI_VECTOR_CORE,170 AI_CORE and
16 MIX_AIV tasks across both devices, with **no observed AiCPU** and no
reverse/VJP/optimizer/diagnostic-journal operators. It records12 model executions,
332 notification pairs,1,172 device switches and2,432 asynchronous copies.
Construction is included. This establishes placement and the inference path;
it is not formal throughput or a full-size bottleneck-time attribution.

The implementation reuses qualified device-controlled Full/state/KV completion
chains. It creates no training owner or retained backward tape for inference.
The legacy C++ constructor and Python `TrainingPlacement` binding alias remain
available. CPU reference intermediates never become candidate inputs.

Projection banks remain on the coordinator. Whole-model memory admission,
actual FP16 consumers, full-size performance and other target environments are
still outside this qualification. Python resident remains a C++/CANN client.
The initial development CPU collection failure from a reserved fixture name is
retained; its corrected run and all fixed-source gates passed.

Reproduction uses task-local `launchers/sharded_inference_evidence.py` with the
full implementation hash. Retained runs are `build-sharded-inference-clean01`,
`build-sharded-inference-python-clean01`, `build-sharded-inference-consumer-clean01`
and `sharded-inference-{cpu,library,consumer,profile}-clean01`.
