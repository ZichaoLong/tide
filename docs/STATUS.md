# Current handoff

Updated 2026-10-02. **ACTIVE: user resumed execution and accepted overhead reduction.**
Continue the overall goal; commits/pushes authorized. No pause instruction.
No subagents. Reference repositories and ObsidianVault are read-only.
Repository /home/zlong/llm/graph-execution-foundation resolves to
/var/tmp/zlong-graph-execution-foundation/repository; branch graph-execution-foundation.
[execution-flows.md](execution-flows.md) is the contract; [ROADMAP F1–F7](ROADMAP.md)
is the only backlog. Overall task remains incomplete.

## Contract and focus

Candidates independently consume common inputs/parameters/initial state, never
CPU reference events/routes/results/gradients. General online greedy accepts legal
topology/input including positive-delay feedback and natural streaming fallback.
Preserve int64, stable ordering, parallel-edge identity, missing/zero messages,
None/zero gradients. Matrix: PDG LibTorch; TimedDAG/Settle LibTorch+PyTorch;
CPU/NPU × streaming/prefill × inference/complete training. Python resident is a
C++/CANN client, not an independent PyTorch scheduler. Five presets retain fine
switches; FP32 main, FP16 separate. CUDA execution is target-machine-pending.
Training means loss/VJP/optimizer/continuation/throughput, not model convergence.
Contract outranks run-ml-experiments: minimal existing durable records; Trackio
is not blocking. Public consumers and F6 take priority over internal polishing.
Only affected checks, not unchanged 8,954 CPU checks. Implementation commit →
immutable qualification → evidence commit; push each. Never edit live-job inputs.

## Current source and qualified scale enablers

HEAD46ff66c is pushed. Uncommitted host worker/packed-source/Next consumer wiring and directed tests
are in progress alongside this handoff; not yet built or qualified. Core unchanged.
Latest implementation0b1a5aa68061f37aef1f4387cf28abb0c77a464f, frozen source
TASK/sources/capacity-admission-clean01. All eight capacity jobs PASSED;
CPU8/NPU19, no skips, 24 cross-language shape plans, eight forced-splitting
complete FP32/FP16 training comparisons and D128 Attention peak calibration.
See [report](evidence/consumer-capacity-20261002.md), consumer-capacity.md.
Audit: python TASK/launchers/capacity_evidence.py 0b1a5aa68061f37aef1f4387cf28abb0c77a464f

Earlier qualified components/consumers, public sharded inference/training,
FP16 masters/publication, continuation/checkpoint/repartition, physical banks,
canonical packet streaming and transactional optimizer recomputation are indexed
in ROADMAP F4/F5. Do not repeat completed gates. Recent evidence:
resident-owner-stream-20261002, consumer-memory-20261002,
resident-optimizer-recompute-20261002 and consumer-capacity-20261002.
The complete consumer estimates simultaneous lifetimes and shrinks only physical
chunks. It is not an allocator quota or an arbitrary vendor/module guarantee.
CPU/mixed total admission and multi-device scale placement remain open.

Current standalone consumers:
TASK/builds/capacity-cpu-clean01/consumer/tidegraph-online-bench
TASK/builds/capacity-consumer-clean01/consumer/tidegraph-online-bench
Core builds placement-{cpu,npu,npu-python}-clean01. Resident backends remain
optimizer-recompute-clean01 / optimizer-recompute-python-clean01. Source/header/
options-verified NPU object reuse plus fresh link/loader, fresh CPU consumer compile;
no backend ABI/kernel change. These identities are already audited, not new builds.

## Completed representative screening; current development

All3 representative-screen0{1,2,3} and4 separate representative-profile-
{mixed,resident}-{train,infer}01 jobs PASSED. Clean source0b1a5aa.
72 serial fresh processes:TimedDAG/standalone LibTorch/prefill,Add/Attention,
inference/training,five FP32 presets + separate resident FP16. Each1continued
warmup+3measured steps,2windows/step,CPU/ATen1,one leased NPU. No heavy own
measurements overlap. Three-repeat FP32 event/output/cut checks exact;loss<1e-5.
Four groups select mixed-a for this scope. Resident FP32 steady CPU-relative
throughput1.462/2.198/2.709/2.198 (Add infer/train,Attention infer/train).
Finite cold-process throughput differs:with only4total steps CPU wins most
cells. Report process startup/exit + construction/warmup separately.
FP16 changes events/loss; no same-trajectory FP32 or large-scale parity claim.

Inputs screening-representative-{add,attention}01/workload.json:
128 body nodes/544 edges,D128/B8/T4/V257,clear=true;
8,995,632 Add /17,384,240 Attention parameters. Workload (not file) hashes:
Add b6aca13c802f0b6ce667f1c6d57d4c1492b233e5bd7267e61009881a1c3e1a15
Attention db2257abf2c8326b78ad7029e5e994e4584ed7ca32ccfc33512553289eafcf43
Resident queue/arrivals4096,outputs128,trace8192,KVtrace65536,forward8GiB,
retained8GiB,backward128GiB,optimizer4GiB,head256MiB,aggressive. Local ceilings
are not allocations. Profiles use same shapes/limits,construction+1warmup+
1measured step,msprof runtime/task/AiCPU. No AiCPU observed in any profile.
Mixed training268252tasks/271132host launches/15873sync copies vs resident
80631tasks/1632host launches/263sync copies;8device-program executions.
Device task sums are not wall time. Exact raw CSVs/hashes are retained.

Audit PASSED:
python TASK/launchers/screen_evidence.py 0b1a5aa68061f37aef1f4387cf28abb0c77a464f --profiles representative-profile-mixed-train01 representative-profile-resident-train01 representative-profile-mixed-infer01 representative-profile-resident-infer01 --output docs/evidence/representative-preset-screen-20261002.json
Report/evidence:docs/evidence/representative-preset-screen-20261002.{md,json}.
Audit passed; evidence is ready for its separate commit. Do not rerun these jobs.
Nonblocking TASK/online-measurement.lock differs from historical timing.lock.

Current uncommitted implementation: expose existing native node workers and
packed_sources/batch_next through actual consumer API/CLI; default behavior is
unchanged. Independent Python/resident reject unsupported host switches before
model construction. New tests compare full observations/gradients/updates against
independent Python streaming (directed CPU8/NPU8, plus refusal/CLI cases). No core/
CANN/ABI change. Need builds and directed gates, then implementation commit and
immutable qualification. Do not re-run the CPU full suite.

PASSED development builds (no NPU lease needed for compilation):
TASK/sources/host-execution-dev01 dirty frozen snapshot;2workers per build,
900s timeout. Runs build-host-execution-{cpu,npu}-dev01,units
 tide-execution-flows-build-host-execution-{cpu,npu}-dev01.service;
TASK/runs/NAME/{status.json,task.log};builds host-execution-{cpu,npu}-dev01.
CPU build_online_consumer.py;NPU build_capacity_client.py reuses qualified
optimizer-recompute-clean01 backend and only unchanged verified client objects.
Next/starting directed development gates:host-execution-cpu-dev01 (11 selected
cases) and host-execution-npu-dev01 (8 cases,one leased NPU),same frozen dirty
source. Units tide-execution-flows-host-execution-{cpu,npu}-dev01.service;
TASK/runs/NAME/{status.json,task.log,junit.xml,pytest};NPU queue.json.
Timeout180s/queue120s. Stop and inspect failures,not a full-suite rerun.
No profiling or throughput job remains active; historical suspended task untouched.

## Remaining priority work

1. Commit representative evidence; finish host-policy wiring/gates and bounded tuning.
2. Complete representative family/language/schedule matrix and generic scale capacity/
   placement improvements, then full-size CPU + screened mixed + resident comparisons.
3. F7 clean final qualification/portable migration packet; CUDA and other CANN tuples
   remain pending real target tests. No new blanket rebuild/full-suite loop.

CPU/mixed currently use one payload device;17B FP32 cannot fit one64GiB NPU.
Generic mixed multi-card placement may require implementation. Eager/mixed FP16
training explicitly rejects the unqualified master path. Original wide remains
480 nodes/2208 edges,D2048/B512/T12/V50304,Add9,468,053,696 /Attention17,521,117,376.
Do not lower logical batch and call it original wide. Coordinator queues/journals
and retained buffers may limit full-size training; use shape analysis and calibrated
safe physical splitting. Do not stop at capacity refusal or use offline CPU events.

## Environment and preservation

TASK=/mi/data2T/zlong/tide-execution-flows. Runtime cwd env -C RUN.
Module libtorch-npu/2.10.0-cann9.0.0;Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
Default shell Python cannot run Torch tests. Public /opt stack supersedes dated
personal guide by user authorization. TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0;
preserve module PYTHONPATH,prepend snapshot/python. SoC Ascend910_9392,16 logical
chips ×64GiB; leased/remapped NPUs only. Standalone/Python runtimes stay separate.
freeze_run.py;background.slice/Nice10,2 build workers,queue120s/run600s/build900s.
Last disk:data219GiB/root14GiB;check before heavy writes. Atomic handoff writes:
scripts.durable_records.replace_text. Preserve all cited artifacts/failures.

Historical historical-cpu-attention-01 remains deliberately SIGSTOP; pause.json
outweighs running status,retains memory/TASK/timing.lock. Do not resume/kill/clean.
Historical Add CPU78.793172/NPU4 47.932888ms/token = throughput1.6438× faster;
it does not certify the new online flows. Earlier24 dirty files are preserved on
pushed archive/restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37,
sha256 inventory TASK/restricted-flow-archive.json. No current pause or approval needed.
