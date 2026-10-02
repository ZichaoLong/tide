# Current handoff

Updated 2026-10-02. **ACTIVE: user resumed and accepted overhead reduction.**
Continue the overall goal; commits/pushes authorized. No current pause instruction.
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
is not blocking. Public complete consumers and F6 take priority over repeated
internal polishing. Only affected checks, not unchanged8,954 CPU checks.
Implementation commit → immutable qualification → evidence commit; push each.

## Latest qualified work

All earlier gates/citations are indexed in ROADMAP F1–F5. Do not repeat them.
Actual CPU/mixed and resident consumers, public multi-device inference/training,
FP16 master/publication, retained VJP and checkpoint/repartition are qualified.
Recent scale enablers:

- Projection banks/adjoints/publication: acb84f3, evidence dd24e50,
  [report](evidence/resident-projection-shards-20261002.md).
- FP16 actual consumers/bounded head: d178b86, evidence0c0a428,
  [report](evidence/resident-consumer-head-20261002.md).
- Canonical streaming:7e375f5, evidence7b96565,
  [report](evidence/resident-owner-stream-20261002.md). Eight jobs passed;
  64MiB endpoint cap, ordinal alias accumulation, no full contribution banks.
- Complete-consumer phase memory:fdfc74802934cf145c052eb3ae966ff94c323e0a,
  [report/audit](evidence/consumer-memory-20261002.md). Five fixed-source jobs
  PASSED; CPU4/NPU7, no skips. No backend ABI change or core rebuild.
  C++ CPU rebuilt; NPU client source/header/options-verified object reuse, fresh
  link/loader. Both reuse existing qualified backend. No live task jobs except
  the intentionally suspended historical CPU task.

Memory calibration:128 body nodes/544 edges,D64/B2/T2/V257,4,367,024 parameters,
Attention,2-card FP32 AdamW,online prefill,4 continued windows/2 updates including
warmup. Peak allocated construction146456576/123725312 bytes; measured training
408049152/392550400; peak reserved461373440/432013312. Not formal throughput.
Initial/construction/warmup/measured sampling occurs outside step timers.
Counters exclude untracked vendor/driver HBM; CPU RSS is process-lifetime peak.

Source consumer-memory-clean01; builds consumer-memory-{cpu,npu}-clean01.
Runs build-consumer-memory-{cpu,npu}-clean01 and
consumer-memory-{cpu,npu,calibration}-clean01 all PASSED. Audit:
python TASK/launchers/consumer_memory_evidence.py fdfc74802934cf145c052eb3ae966ff94c323e0a

Retain failures: memory dev01 collection dtype collision; NPU dev02 Python fixture
passed native_library incorrectly (other6 passed, corrected case passed dev03).
Calibration-dev01 reverse-packet budget refusal at trace4096/backward8GiB;
dev02 cache-adjoint boundary refusal at trace1024/backward8GiB. Neither OOM.
Nested budget divisions reviewed; clean calibration passed with32GiB backward,
trace1024 and unchanged KV128. Actual per-card training peak<400MiB demonstrates
that local envelopes cannot simply be summed to obtain total HBM admission.

## Current implementation awaiting immutable qualification

Resident Add/Attention complete-consumer capacity admission is implemented in
matching Torch-free C++/Python planners. It reads per-device driver availability,
accounts for simultaneous tensor lifetimes, reserves headroom and reduces only
physical rows. CLI device-memory-bytes and offline plan_execution_flow.py added.
Core/resident ABI unchanged. Current scope/limits: docs/consumer-capacity.md.

Development source capacity-admission-dev05. Terminal passed: static-dev02(4
checks including24 cross-language shape cases); build-capacity-cpu-dev01;
build-capacity-consumer-dev03; capacity-cpu-dev01(8); capacity-native-dev02(5);
capacity-calibration-dev01; capacity-libtorch-dev01(4). All current jobs terminal.
Observed training peaks857463296/685577216 bytes fit estimates2756542808/2114316504. Builds at TASK/builds/capacity-{cpu-dev01,consumer-dev03}; consumer backend
remains optimizer-recompute-clean01, Python backend optimizer-recompute-python-clean01.

Calibration17384240 parameters,128 body nodes,D128/B2/T2,FP32 AdamW,2NPUs,4
continued windows/2updates:3GiB incremental/card cap automatically reduces
Full/emission/reverse rows16→1,attention rows8→1,keys128→8,head1024→64.
Output8/cut80; loss5.662106990814209 differs4.77e-7 from independent earlierCPU
5.662106513977051. This is capacity calibration, not formal throughput.

Retain failures: invalid initial fanout/local-span test fixture; static-dev01
inconsistent overflow message before checked arithmetic; consumer build-dev01
SDK header path and dev02 direct ACL symbol link; native-dev01 Python limits
has no diagnostics member. Fixed with matched SDK declaration plus lookup in
already loaded runtime and diagnostics from consumer options. No OOM or semantic
tolerance relaxation. Build-dev03 reused completed ELF objects from failed link,
with source/header/compile-option identities verified; failed record unchanged.

Next: commit tested implementation and push; freeze exact commit, source-verified
consumer rebuild/relink, affected CPU/NPU gates and capped calibration; evidence
separate. Do not repeat core/backend or old full CPU matrix. No pause authorized.

## Next work

Total per-device memory admission and safe physical splitting → representative
five-preset screening → full-size F6 → F7 final qualification/target handoff.
Include parameters,state/KV,retained immutable banks/journals/roots,physical and
canonical gradients,FP32 masters/slots/proposals,communication,embedding/head/loss,
construction transients,CANN workspace and calibrated headroom. Existing local
ContentBudget,reverse budgets and head budget are not total admission.
Avoid repeated trial launches for nested budget limits; inspect shape formulas
and distinguish declared envelopes from simultaneous live allocations.

Current CPU/mixed consumers have one payload device; full-size17B FP32 does not
fit one64GiB card. Generic multi-card mixed placement may remain necessary.
Eager/mixed FP16 training explicitly refuses the unqualified master path.
Historical restricted accelerator_scale executors do not replace online flows.
No new formal full-size throughput result. Do not finish merely on capacity refusal.
Memory evidence2b6dae2 is committed/pushed. No pending memory qualification.

Optimizer recomputation implementation81f4736b3ed3afd3ed3cf2a7b56a243d3f10b3f2
is committed/pushed. All8 immutable-source jobs PASSED; audit
optimizer_recompute_evidence.py exited0. [Report/audit](evidence/resident-optimizer-recompute-20261002.md).
No live task jobs except the intentionally suspended historical CPU.

DeviceOptimizer evaluates finite candidates, reaches local/cross-card consensus,
then recomputes and commits from frozen gradient/old-state inputs. A following
counter kernel avoids first-use momentum races. No full numerical proposal banks;
ordinary AdamW removes12 bytes/active parameter. Public ABI is unchanged,private
optimizer-dependent objects and2 CANN kernels are rebuilt or byte-verified reused.
Local32 trajectories per dtype (FP32 updates256,FP16 updates248 with2 expected
half refusals);peer4 trajectories/32 updates per dtype; public32 trajectories/
512 windows/128 updates with2→3-card restore; actual consumers28 passed; no skips.
Four1,048,579-element allocator/last-tile-Inf checks fit one live bank set plus2MiB.
Independent D32 FP16 training profile:18,453 AI_VECTOR_CORE/704 AI_CORE/258 MIX_AIV,
no observed AiCPU; each device4 numerical passes/2 counter commits for2 updates.

Source optimizer-recompute-clean01; backends optimizer-recompute-clean01 and
optimizer-recompute-python-clean01; installed client optimizer-recompute-consumer-clean01.
Runs build-optimizer-recompute{,-python,-consumer}-clean01 plus
optimizer-recompute-{component,session,consumer,calibration,profile}-clean01.
Do not repeat these completed qualification jobs. Implementation/evidence use
existing records and recursive dependency audits,not new full-core builds.

D128 calibration128 nodes/544 edges,B2/T2/V257,17,384,240 parameters,FP32 AdamW,
2-card prefill,4 windows/2 updates including warmup. New measured allocator peaks
938704896/844491776 bytes; construction298747904/256794624. Each card drops
106955264 bytes in every noninitial phase against fixed fdfc748 baseline.
Loss/output/cut also match independent CPU; one timing sample is not a formal
throughput conclusion. Input optimizer-recompute-calibration01. Retain component
dev01 test failure: empty named group correctly selected no parameters; corrected
explicit weight group. No production semantic failure or tolerance relaxation.

Optimizer evidence f1876c1 is committed and pushed; no pending qualification.
Continue total per-device memory planning and safe chunk selection,then F6.
No new pause or user approval is required.

## Environment and job bounds

TASK=/mi/data2T/zlong/tide-execution-flows;RUN=TASK/runs/NAME;
unit=tide-execution-flows-NAME.service;RUN/status.json/task.log and queue.json.
Module libtorch-npu/2.10.0-cann9.0.0;Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
Default shell Python cannot run Torch tests. Public /opt stack supersedes dated
personal guide by user authorization. TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0;
preserve module PYTHONPATH,prepend snapshot/python. SoC Ascend910_9392,16 logical
chips ×64GiB; leased/remapped NPUs only. Standalone/Python runtimes stay separate.
Core builds placement-cpu-clean01,placement-npu-clean01,placement-npu-python-clean01.
Qualified resident backends optimizer-recompute-clean01 / optimizer-recompute-python-clean01.
CPU standalone consumer remains consumer-memory-cpu-clean01 (unchanged core/source).
freeze_run.py;runtime cwd env -C RUN prevents vendor files polluting snapshots.
background.slice/Nice10,2 build workers,queue120s/run600s/build900s. Last disk:
data220GiB/root14GiB;check before heavy writes. Atomic writes:durable_records.replace_text.

Historical historical-cpu-attention-01 remains deliberately SIGSTOP; pause.json
outweighs running status,retains memory/TASK/timing.lock. Do not resume/kill/clean.
Historical Add CPU78.793172/NPU4 47.932888ms/token = throughput1.6438× faster;
it does not certify resident. Earlier24 dirty files are preserved on pushed
archive/restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37,
sha256 inventory TASK/restricted-flow-archive.json. Preserve cited artifacts/failures.
