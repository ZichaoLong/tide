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

Optimizer recompute increment ready for implementation commit. DeviceOptimizer
now evaluates finite candidates, reaches the existing local/cross-card gate, then
recomputes and commits from frozen gradient/old-state inputs. A separate following
counter kernel avoids first-use momentum races. SGD/AdamW,FP16 representability,
None/zero and sticky errors remain. Ordinary AdamW removes12 bytes/active parameter;
this is a fixed-memory reduction,not total-memory admission or a throughput claim.
Changed private optimizer layout/C++ and2 Ascend C kernels; public ABI unchanged.

Development standalone builds dev01/dev02,Python and installed consumer dev01
PASSED. Frozen optimizer-recompute-dev02 fixes only a calibration test group;
production source is identical to dev01. Component-dev02 PASSED4 cells:32 local
trajectories per payload plus peer optimizer checks. Four1,048,579-element
calibrations fit one live state set plus2MiB,including last-tile-Inf byte-identical
refusal. Actual consumer-dev01 PASSED28 native/LibTorch training/split-head cases.
Retain component-dev01 failure: new test supplied an empty named group, correctly
updated nothing; explicitly naming weight fixed it. No production semantic failure.

Fixed old-source D128 baseline and independent CPU calibration both PASSED at
fdfc748. Input optimizer-recompute-calibration01 has128 body nodes/544 edges,
D128/B2/T2/V257,17,384,240 parameters. Four windows/two AdamW updates including
warmup. New calibration-dev01 PASSED; losses/outputs/cut match both old NPU and
independent CPU. Per-card allocated peak decreases106955264 bytes in construction,
warmup and measured phases. New measured peaks938704896/844491776 bytes.
One sample,concurrent development: no formal performance claim.

Next commit implementation,then frozen optimizer-recompute-clean01; byte-verified
reuse of both development backends' affected objects/kernels with fresh links,
installed client,component4/public2→3 checkpoint/consumer28 gates,new D128
calibration and separate actual FP16 training profile. All development jobs terminal.
Then evidence commit and continue total-memory planning/F6; no pause authorized.

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
Qualified resident backends owner-stream-clean01 / owner-stream-python-clean01.
freeze_run.py;runtime cwd env -C RUN prevents vendor files polluting snapshots.
background.slice/Nice10,2 build workers,queue120s/run600s/build900s. Last disk:
data220GiB/root14GiB;check before heavy writes. Atomic writes:durable_records.replace_text.

Historical historical-cpu-attention-01 remains deliberately SIGSTOP; pause.json
outweighs running status,retains memory/TASK/timing.lock. Do not resume/kill/clean.
Historical Add CPU78.793172/NPU4 47.932888ms/token = throughput1.6438× faster;
it does not certify resident. Earlier24 dirty files are preserved on pushed
archive/restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37,
sha256 inventory TASK/restricted-flow-archive.json. Preserve cited artifacts/failures.
