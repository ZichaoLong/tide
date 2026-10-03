# Graph execution foundation acceptance roadmap

This is the only backlog and stage acceptance index. STATUS owns the current
handoff; semantics owns the contract. The six-stage acceptance scope was frozen
on 2026-09-23 and is now complete; [final qualification](evidence/foundation-final.md)
is the delivery report. No unit of that acceptance version remains pending;
the separately authorized cross-family extension below is also complete.
Historical M0–M8 evidence remains indexed by architecture.md and Git history;
its finite profiles do not certify the broader acceptance version.
Status: verified = cited completed evidence; implemented = code without the full
required gate; planned = required work remains. No submitted/running job passes.

## Authorized extension: reusable experiment library

Authorized 2026-09-28. Consumers live in separate repositories and depend on a
pinned Tide version/source. Runtime use must not write into the Tide checkout.
Existing graph/kernel semantics and independent reference schedules remain.
L1-L4 are complete; [immutable-source qualification](evidence/library-foundation.md)
records the full gate and the subsequent validator-only hardening separately.

| Unit | Delivery and acceptance | Status |
| --- | --- | --- |
| L1 | Versioned graph/module configuration, public runtime/session API, explicit backend/policy checks, external inputs and checkpoint lifecycle | verified; library qualification |
| L2 | Package-owned configuration equivalence gate: full observables, independent gradients, chunking, multi-step updates and checkpoint continuation, explicit bounded scope | verified; library qualification + validator follow-up |
| L3 | Python wheel and optional native consumption from another directory; CMake install/export for C++ clients; runnable consumer and version boundary | verified; installed Python/native/C++ consumption |
| L4 | Mixed modules and fully active complex topology gates, actual topology with reduced tensor sizes, clean full CPU regression and consumer qualification | verified; 8621-test full gate and 22 complex cells |

Full-scale performance runs and new backend support are not required for this
library increment. Reports identify the actual topology, dimensions, inputs,
source, options, backend and comparisons; reduced-width evidence does not
certify all shapes or other backends.

## Authorized extension: accelerator library paths

Authorized 2026-09-28 after toolchain preparation. Develop on the local aarch64
host, verify NPU here, and deliver the same qualification entry points for CUDA
and other host/software versions. Additional compatible CANN stacks may be
installed in isolation. Local site paths belong in launch records, not library code.

| Unit | Delivery and acceptance | Status |
| --- | --- | --- |
| A1 | Python CUDA/NPU public runtime and configuration CPU parity, live placement, VJPs, training and checkpoint handoff | NPU verified across four CANN stacks; CUDA implemented, device gate pending |
| A2 | Native device-neutral execution, isolated backend builds, standalone C++ CUDA/NPU lifecycle and clients | NPU verified, standalone SDK on 2.10/CANN9.0 and2.9/CANN8.5.0/.1/.2; CUDA build/CPU checks passed, device gate pending |
| A3 | Configuration matrix across families/modules/schedules, placement/fallback evidence, installed Python/native/C++ use | verified locally: 328 positive NPU cases, four CSR rejections, hardware traces and installed consumers |
| A4 | Clean full CPU regression, local NPU acceptance and CUDA compilation without device claims | verified: 8636 CPU tests, 22 complex cells, installed consumers; no CUDA device claim |
| A5 | Matched local CANN versions and portable target-machine environment/qualification reports | four local stacks verified; additional [standalone2.9 gates](evidence/standalone-sdk29-20260928.md) passed; migration fixtures/commands delivered, target-machine executions pending |

[Immutable accelerator evidence](evidence/accelerators-20260928.md) records exact
source, finite scope, failures and environmental boundaries. The remaining external
acceptance is a rebuild and full device suite on NVIDIA hardware and on each new
host/software combination, including x86_64; CUDA FP64 needs its own gate.
Use [the target-machine commands](accelerators.md#target-machine-acceptance).

Initial accelerator contract: single-device eager FP32; CPU FP32/FP64 preserved.
CUDA FP64 is a separately qualified capability. FP16/BF16/AMP, distributed jobs,
graph compilation and custom fused kernels are subsequent extensions. Independent
schedules and exact discrete graph semantics remain mandatory. Floating tolerance
and numerical backend settings are explicit. Report finite coverage and unknown
fallback visibility instead of extrapolating from a small device smoke.

## Authorized extension: full-topology NPU performance

Authorized 2026-09-28: independent benchmarks on multiple NPUs and one model
across multiple NPUs, using the historical D2048/B512/V50304 Attention17.269B
and Add9.468B (historical binary-unit label8.818) PDG topology. Keep the
single-device runtime contract explicit; retain historical CPU FP64 Read as a
reference while adding declared FP32 Read/control placements.

| Unit | Delivery and acceptance | Status |
| --- | --- | --- |
| P1 | Standalone C++ resident/host placement clients; exact parameter/topology identities; CPU oracle at small sizes | verified: immutable CPU/2/8-NPU small and real-topology gates; explicit VJP policy, see [evidence](evidence/accelerator-scale-20260928.md) |
| P2 | Warm synchronized timing, transfer and allocator metrics, durable records/Trackio, bounded processes | verified: local Trackio success and timeout cleanup; immutable consumer build |
| P3 | Independent concurrent configuration benchmarks, with resource interference recorded | verified finite assessment: three fixed-group inference pairs per model and one concurrent Add2/Attention4 workflow; [54-cell performance evidence](evidence/accelerator-performance-20260928.md), no causal parallel-speedup claim |
| P4 | Same-model 2/4/8-device memory/locality placement; full-size no-grad and grad-forward windows, explicit limits | verified finite assessment: both models no-grad at 2/4/8 devices, memory/locality at four; retained Attention2 grad-forward OOM and successful Attention4 follow-up; descriptive placement/scaling, not causal speedup |
| P5 | Explicit CPU/model-device Read and softmax controls, FP64/FP32, independent CPU reference and cross-precision diagnostics | verified: [34-cell scoring gates](evidence/accelerator-scoring-20260928.md), seven full-size inference candidates/model and three fixed-group CPU64/CPU32 pairs/model; reference/default retained where selection is unresolved |
| P6 | Independent NPU node-selection/event-scheduling candidates; validate then choose by inference/training performance | verified: [200-cell correctness gates](evidence/accelerator-dispatch-training-20260928.md), all seven inference and cold-training candidates/model; finite performance choices documented, public CPU defaults unchanged |
| P7 | Full-size backward and optimizer-step performance, independently checked before implementation selection | verified: full-window training correctness on CPU/2/4/8 devices; seven cold updates/model and three warmed processes/model. Add4 CPU32 median 42.080 and Attention8 mixed32 median 115.678 ms/sample-token; synthetic throughput, not convergence |

The bounded local assessment is complete. Screens and placement/scaling remain
descriptive; matched inference uses three fresh process pairs per model and
warmed training uses three processes per selected configuration. The single
concurrent workflow is not a causal parallel-speedup measurement.
Resident state/KV/message payloads remain on the assigned devices. Integer
histories, tensor handles and C++ dispatch remain host-owned. Configured Read,
controls, dispatch, executor guards and transfers contribute to the declared
synchronized timing; explicit byte counters do not measure all transport.
See [the consumer contract](accelerator-scale.md).

## Authorized extension: public/consumer FP16 and profiling interpretation

Authorized 2026-09-29. Existing FP32 acceptance remains immutable.

| Unit | Delivery and acceptance | Status |
| --- | --- | --- |
| H1 | Public Python/native and consumer FP16 payload; separate consumer Read/control precision; FP32 master optimizer/checkpoints and static loss scale; unchanged integer semantics | verified; [FP16 qualification](evidence/fp16-qualification-20260929.md) |
| H2 | Independent same-dtype CPU oracle; complete observables, routes, isolated VJPs, None/zero and three optimizer updates; clean CPU/public NPU/consumer gates | verified;8645 CPU tests,82 public NPU cases and124 consumer cells |
| H3 | Bounded same-placement FP32/FP16 full-size inference and warmed training for Add/Attention; preserve failures and variance | verified; [eight accepted full-size cells](evidence/accelerator-fp16-performance-20260929.md), fixed allocation per pair; shared-load exploratory results and all resource failures retained |
| H4 | Existing CANN trace engine/API summary; explain host orchestration and attribution limits | verified; [profile analysis](evidence/accelerator-profile-analysis-20260929.md) and two FP16 hardware traces |

## Authorized extension: CPU comparison and bounded device scheduling

Authorized 2026-09-29 after the read-only CPU/NPU comparison. Preserve the
foundation's independent CPU schedules, exact discrete semantics and optional
backend boundaries. This increment is not a claim of arbitrary dynamic-graph
compilation or support for unbounded device queues.

| Unit | Delivery and acceptance | Status |
| --- | --- | --- |
| D1 | Same-source full-size Add/Attention pure-CPU and NPU inference/complete-training baselines; effective CPU parallelism, identical shapes/windows/owners, three process repeats for speed claims | verified finite assessment:21 passed/1 CPU Attention timeout,2 later repeats skipped; [three-process CPU/NPU report](evidence/accelerator-cpu-npu-comparison-20260929.md), no measured CPU Attention training ratio |
| D2 | Bounded full-size NPU trace windows, host decisions/copies/barriers and device engines; instrumentation timing separated from throughput | verified finite scope:4 full-size collected/analyzed windows with exact device coverage; original export/analysis failures and offline recoveries retained in the CPU/NPU report |
| D3 | Optional bounded scheduler: device queue/history/masks, exact int64 keys/ties/edge identities, explicit capacities, single-device Add/Attention inference and independent complete-observable oracle | verified for finite static expansion/device predication; [84 immutable CPU/NPU cells](evidence/bounded-scheduler-qualification-20260929.md), both dtypes; general device event queue not implemented |
| D4 | Device-controlled continuation of tasks within a window; graph/capture or compiled backend, no per-event host scalar/index decisions; verify actual replay path and explicit unsupported combinations | verified finite scope; native one/two/eight-device replay, input changes and explicit unsupported combinations; full-size capacity separate |
| D5 | Multi-device execution with locality and device completion dependencies; complete training with isolated VJPs, None/zero, master/optimizer updates and three-step oracle | verified finite scope; tiny one/two/eight-device FP32/FP16 and actual-topology CPU/two-device three-token gates, SGD/AdamW trajectories |
| D6 | Full-size accepted-path FP32/FP16 comparison, immutable qualification, source/build/profile/record audit and final support/performance report | finite assessment complete: [capacity report](evidence/bounded-scheduler-capacity-20260929.md) and CPU/NPU/profile audit; Add bounded-eager inference pair passed, full-size captured replay remains unqualified |

This closes the fixed assessment, not the broader full-size resident-scheduling
objective. The current backend uses host-built static expansion/device predication;
a general preallocated device event queue is not implemented. Full-size captured
replay remains unqualified after notification-capacity and memory failures.
Notification reuse/completion design and lower-memory execution need further
implementation and independent gates before a usable full-size claim. Uncontended
12-device fit is unknown. Baseline/profiling and backend development overlapped;
the original recommendation to finish baselines first was not followed in order.

Start with the installed Torch/TorchNPU2.10/CANN9.0 stack; capabilities must be
probed before relying on graph capture, conditional streams or custom kernels.
A device tensor sort followed by host dispatch does not pass D4. Forward-only
capture does not qualify a device-resident backward/optimizer. Preserve the
independent eager reference and reject unsupported requests explicitly.

Bounds: historical D2048/B512/V50304, 12-token inference (4 warmup/8 measured),
complete 12-token AdamW windows (1 warmup/1 measured update per process).
Initial CPU node/head budgets:56 for Add,160 for Attention, ATen/BLAS1; report this
intentional difference from the NPU16-host-worker configuration. No overlapping
heavy project jobs during formal timing. Three repetitions establish descriptive
spread; shared external load prevents a hardware-limit or convergence claim.
Use small independent analytic/trace/VJP gates before full-size backend runs.
Profile one declared token or training phase at a time, with finite wall/RSS/disk
bounds, and keep these timings out of uninstrumented throughput tables.

## Authorized extension: independent complete flows across all three families

Authorized and confirmed 2026-09-30; consolidated contract in
[execution-flows.md](execution-flows.md). The user resumed execution after the device-window alignment and authorizes pushing
tested commits. The current contract takes priority over the experiment skill;
retain only minimal useful recording and bounded execution; see STATUS.
This general-online delivery supersedes the earlier finite-static assessment scope;
all historical evidence remains scoped and unchanged. STATUS owns current job/source
state. The contract owns definitions, matrices, presets, batching/memory and test scope.

| Unit | Delivery and acceptance | Status |
| --- | --- | --- |
| F1 | General online greedy node-time prefill and independent streaming for every legal family topology/input; complete continuation/configuration/timing contracts | host greedy qualified;8849 clean CPU checks + four CTests + six NPU FP32 fixtures passed; public resident and continued consumers subsequently qualified under F4/F5; all ten representative submatrices complete under F6, with full-size and final integration acceptance still open |
| F2 | General topology/input fixtures including PDG feedback and unaligned arrivals; active-scale/locality packets, equivalent family mappings and actual work counts; no fixture-specific scheduler | in progress; reachable v1 reset-window and v2 continuous packets, real edge-affine Add/Attention public consumers implemented ([contract](online-consumers.md)); CPU120 and NPU18 directed complete-training checks qualified on cleanfe2d886 ([evidence](evidence/online-consumers-20261002.md)); scale/performance pending |
| F3 | Independent CPU FP64/FP32 schedules, full observables/isolated VJPs/None/zero/multi-update optimizer gates; public Python/native and standalone consumers, changes of input and continuation | host greedy immutable fixtures qualified; real matched-model CPU120/NPU18 consumer checks qualified on cleanfe2d886 ([evidence](evidence/online-consumers-20261002.md)); HARD resident slot-affine qualified on clean96c75f8 ([evidence](evidence/resident-emission-training-20261002.md)); scale gates pending |
| F4 | Device-resident online queues/readiness/selection/batching/progression, packed transport and peer completion, continuous state and complete training; explicit capacities and safe chunking | single-device FP32 inference qualified: packed slot-affine/phase emission ([evidence](evidence/device-emission-20260930.md)), stable InputOrigin projection ([evidence](evidence/device-origins-20260930.md)), sum Aggregate, identity/EMA/Add state, linear/norm-fp32 Read, count/positive selection, identity/tanh/LH/SwiGLU Full ([SwiGLU evidence](evidence/device-swiglu-20260930.md), [LH evidence](evidence/device-lh-full-20260930.md)), periodic clocks and optional journals; vector sum/state/Read, all20 cells on clean8a735ef ([Read evidence](evidence/device-read-20260930.md)), clocks on clean49ff108 ([evidence](evidence/device-clock-20260930.md)); runtime failure ownership separately qualified ([evidence](evidence/runtime-read-20260930.md)); bounded fiber-sum attention/KV qualified on clean081f567 ([evidence](evidence/device-fiber-20260930.md)); all five fiber pooling profiles qualified on clean691cb31 ([evidence](evidence/device-fiber-pool-20260930.md)); event attention/GQA/window qualified on cleanb668f2f ([evidence](evidence/device-event-attention-20260930.md)); shared device key-axis tiling qualified on clean7b03614 ([evidence](evidence/device-key-tiling-20260930.md)); shared forward budgets/serial CANN workspace reuse qualified on clean523b323 ([evidence](evidence/device-memory-20260930.md)); event node-time batching qualified on clean26aa09f ([evidence](evidence/device-event-batch-20260930.md)); fiber node-time batching qualified on cleanc83aec3 ([evidence](evidence/device-fiber-batch-20260930.md)); normalized Aggregate qualified on cleanf1b7168 ([evidence](evidence/device-aggregate-20260930.md)); identity/EMA/Add HARD device state-chain VJP qualified on clean3e2d54d (216 CPU FP32/FP64 autograd cases,four real device tapes,all34 component cells; [evidence](evidence/device-state-vjp-20261001.md), [contract](resident-state-vjp.md)); identity/tanh Full VJP qualified on clean3285b13 (96 CPU FP32/FP64 local autograd cases,2 actual tapes,all35 component cells; [evidence](evidence/device-full-vjp-20261001.md), [contract](resident-full-vjp.md)); HARD sum/broadcast identity/EMA/Add+tanh graph backward qualified on clean37430e1 (98 independent CPU FP32/FP64 whole-graph autograd windows,64 device link windows,all37 cells,separate profile; [evidence](evidence/device-graph-vjp-20261001.md), [contract](resident-graph-vjp.md)); alias-owner reduction qualified on clean1ef23f3 (36 CPU Streaming autograd graph/root cases,all38 cells,separate profile; [evidence](evidence/device-parameter-vjp-20261001.md), [contract](resident-parameter-vjp.md)); internal packed SGD/AdamW plus forward publication qualified on clean3b31ee2 (32 optimizer trajectories/256 updates and18 actual graph trajectories/72 continued windows,CPU FP32/FP64,all40 cells,separate profile; [evidence](evidence/device-training-step-20261001.md), [contract](resident-optimizer.md)); internal retained-window backward qualified on cleanc2423f0 (26 trajectories/104 windows,CPU FP32/FP64,after-close snapshots,actual pending/state gradient links,shared-owner accumulation,all41 cells,separate profile; [evidence](evidence/device-retained-20261001.md), [contract](resident-retained.md)); public C++ training owner qualified on clean591e907 (18 trajectories/288 windows/72 updates,CPU FP32/FP64,SGD/AdamW,retained windows,complete-cut in-memory optimizer restore,lifecycle guards,all42 component cells and installed consumer; [evidence](evidence/public-resident-training-20261001.md), [contract](resident-training.md)); Python client/disk lifecycle qualified on clean0f363b8 (40 NPU cases including real loss cotangents,three families,both schedules/optimizers,new-process checkpoint suffix;76 CPU interface tests;separate one-case profile; [evidence](evidence/python-resident-training-20261001.md));LH/SwiGLU adjoints and public C++/Python training qualified on clean5143de3 (90 local configurations/270 replays,46 trajectories/736 windows/184 updates,54 Python device tests,all44 component cells; explicit AdamW/control numerical conditions and retained strict failures; [evidence](evidence/resident-extra-full-20261001.md)); normalized Aggregate adjoints/public C++ and Python training qualified on clean0ba9fc6 (all46 component cells,39 configurations/117 replays,57 trajectories/912 windows/228 updates,106 Python cases,strict Aggregate controls,separate profile; [evidence](evidence/resident-aggregate-training-20261001.md), [contract](resident-aggregate-vjp.md)); HST/SOFTP Emit/control/Read adjoints qualified on clean4b8ced4 (all48 component cells,36 configurations/108 replays,98 trajectories/1568 windows/392 updates,154 Python cases,strict control comparisons,separate profile; [evidence](evidence/resident-control-training-20261001.md), [contract](resident-control-vjp.md)); event and same-fiber attention/KV/bias adjoints jointly qualified on clean66a6ca5 (all52 component cells,196 Python NPU cases,installed C++ training client; event66 roots/8 trajectories and fiber172 roots/20 trajectories,CPU FP32/FP64; [evidence](evidence/resident-attention-training-20261001.md), [event contract](resident-event-vjp.md), [fiber contract](resident-fiber-vjp.md)); FP16 Full/LH/sum components qualified on clean466b4c3 ([evidence](evidence/resident-fp16-components-20261001.md)); FP16 state/Read/SwiGLU/emission/normalized Aggregate components qualified on clean9f010c9 (all62 standalone cells,196 Python NPU cases,four traces; [evidence](evidence/resident-fp16-forward-components-20261001.md)); FP16 HARD inference qualified on clean8b05c04 (eight affected standalone cells,215 Python cases,separate trace; [evidence](evidence/resident-fp16-inference-20261001.md)); FP32 master optimizer/FP16 publication components qualified on clean9a84432 (six standalone cells,215 Python cases,separate trace; [evidence](evidence/resident-fp16-master-publication-20261001.md)); FP16 state/basic Full VJP components qualified on cleand2a1afc (six standalone cells,215 Python cases,two separate traces; [evidence](evidence/resident-fp16-basic-vjp-20261001.md)); FP16 normalized Aggregate/LH/SwiGLU VJP components qualified on clean5ce5346 (six standalone cells,215 Python cases,two separate traces; [evidence](evidence/resident-fp16-extended-vjp-20261001.md)); FP16 packed attention VJP qualified on cleanf20c2cc (four standalone cells,61 affected Python cases,separate trace; [evidence](evidence/resident-fp16-attention-vjp-20261001.md)); FP16 HST/SOFTP inference and local control adjoints qualified on clean3e33973 (five standalone cells,97 Python tests,two traces; [evidence](evidence/resident-fp16-control-20261001.md)); FP16 base graph/retained VJP qualified on clean25e996c (122 windows and42 retained trajectories per dtype,7 standalone cells,97 Python tests,two traces; [evidence](evidence/resident-fp16-graph-reverse-20261001.md)); FP16 normalized Aggregate/LH/SwiGLU retained graph VJP qualified on cleandb9e985 (110 trajectories/440 windows per dtype,6 cells,separate trace; [evidence](evidence/resident-fp16-extended-graph-20261001.md)); FP16 event/fiber cache whole-graph/retained reverse qualified on cleanf26f3b0 (66 event and152 fiber/mixed trajectories per dtype,10 standalone cells,separate two-case trace; [evidence](evidence/resident-fp16-cache-graph-20261001.md)); public FP16 training qualified on clean0095048 (83 trajectories/1328 windows/332 updates,5 standalone cells,138 Python cases,separate profile; [evidence](evidence/resident-fp16-training-20261001.md));reusable two-device loop packets qualified on clean5c3662b (6 cells,2-device profile; [evidence](evidence/device-peer-control-20261001.md));internal remote Full inference qualified on cleana8fa371 (FP32/FP16,120 configurations/600 windows per dtype,8 standalone cells,96 Python single-device regressions,separate two-device profile; [evidence](evidence/device-peer-full-20261001.md));compact Full parameter shards/device packing and memory/locality placement qualified on clean2617a11 (ten terminal jobs, one/two/three-device FP32/FP16 checks,96 Python regressions,separate three-device profile; [evidence](evidence/device-full-shards-20261001.md));retained cross-card Full VJP qualified on clean5591319 (FP32/FP16 each50 trajectories/200 windows,two/three-device placement and one-owner checks,96 Python regressions,separate three-device trace identifying120 Bool ScatterUpdate AiCPU tasks; [evidence](evidence/device-full-reverse-20261001.md));Full reverse metadata fusion qualified on cleanfe68065 (same FP32/FP16 retained gate,four fixed jobs,three-card trace:120 AiCPU Bool scatters replaced by40 device vector merge tasks,no AiCPU observed; [evidence](evidence/device-full-reverse-merge-20261001.md));canonical device alias reduction,global atomic SGD/AdamW and packed alias publication qualified on cleana365e2f (each dtype40 continued training trajectories/640 windows/160 updates,retained VJP/optimizer gates,1/3-card checks,five Python regressions,nine terminal jobs,three-card trace without observed AiCPU; [evidence](evidence/device-canonical-owners-20261001.md));compact state/Read/KV forward qualified on clean62935e9 (each dtype120 configurations/600 windows and28 transaction windows,1/3-owner checks,small public training regressions,nine terminal jobs,two-device trace without observed AiCPU; [evidence](evidence/device-state-owners-20261001.md));compact retained state/Read/KV reverse and canonical internal training qualified on clean49541be (each dtype50 VJP trajectories/200 windows and40 training trajectories/640 windows/160 updates,1/3-owner subsets,device completion chain,separate two-card trace without observed AiCPU; [evidence](evidence/device-state-reverse-20261002.md)); public multi-device C++/Python-owned FP32/FP16 training and portable repartition qualified on clean0d7c45e (ten terminal jobs, each dtype32 trajectories/512 windows/128 updates, four owner transitions,41 Python cases,installed client,separate two-card profile without observed AiCPU; [evidence](evidence/public-sharded-training-20261002.md)); scale consumers pending; immutable attention snapshots qualified on clean38858d0: Python16/consumer24, native160trajectories/2560windows/640updates, D512 same-lease allocator -275,262,464bytes/card, separate FP16 profile ([evidence](evidence/resident-attention-snapshots-20261002.md)); shared-storage accounting qualified on clean4dd8368 ([evidence](evidence/consumer-training-storage-20261002.md)); phase-scoped shared reverse gather qualified on clean1757b90 (native160trajectories/2560windows,Python16/consumer24, same-lease coordinator allocator -28,367,872bytes, separate FP16 trace; [evidence](evidence/resident-reverse-gathers-20261002.md)); vector optimizer finite checks qualified on cleanbb40cff (504 boundary probes,consumer24,restore32, isolated three-process momentum6.778×/AdamW7.007×; [evidence](evidence/resident-optimizer-finite-20261002.md), no graph-speed claim); ordered per-window canonical reduction/projection-adjoint reuse qualified on clean233bf01 (native176trajectories/2944windows,Python16/consumer32,D512 allocator -294249472/-294096896bytes per card,separate FP16 profile; [evidence](evidence/resident-window-reduction-20261003.md)); Attention parameter-adjoint reuse qualified on clean6b9224c (native160trajectories/2560windows,Python16/consumer32,D512 -268961792bytes/card,separate FP16 profile; [evidence](evidence/resident-attention-adjoints-20261003.md)); original Add/Attention B512 resident training now qualified under F6 below; CPU/mixed scale and formal comparisons remain pending |
| F5 | Complete CPU/NPU streaming/prefill matrix: PDG LibTorch, TimedDAG/Settle LibTorch and PyTorch; fine switches and five presets, multi-device/locality/FP32/FP16 | host Read/control/selection placement qualified on clean d412541 (8,952 CPU tests,10 CTests,26 NPU public cases,C++ NPU121 schedules/363 updates and separate profile; [evidence](evidence/execution-placement-20261001.md)); optional public resident C++/Python-owned FP32 HARD inference qualified on clean622dbb2 (8,954 CPU tests,33 standalone component cells,25 Python cases,installed consumer and separate profile; [evidence](evidence/public-resident-20261001.md)); declared single-device FP32 public training profiles qualified on clean66a6ca5 ([evidence](evidence/resident-attention-training-20261001.md)); FP16 HARD resident inference qualified on clean8b05c04 ([evidence](evidence/resident-fp16-inference-20261001.md)); public FP16 training qualified on clean0095048 ([evidence](evidence/resident-fp16-training-20261001.md));public multi-device training/installed client qualified on clean0d7c45e ([evidence](evidence/public-sharded-training-20261002.md)); CPU/mixed continuous consumers qualified on cleanfe2d886,CPU120/NPU18 directed cases ([evidence](evidence/online-consumers-20261002.md)); HARD resident slot-affine adjoints qualified on clean96c75f8 ([evidence](evidence/resident-emission-training-20261002.md)); actual resident FP32 consumers qualified on clean0d61cb9 (CPU59,resident19,mixed18,no skips,separate two-card actual Attention profile without observed AiCPU; [evidence](evidence/online-resident-consumers-20261002.md)); compact physical projection banks/VJPs and canonical publication qualified on cleanacb84f3 ([evidence](evidence/resident-projection-shards-20261002.md)); actual FP32/FP16 consumers with bounded head workspace qualified on cleand178b86 ([evidence](evidence/resident-consumer-head-20261002.md)); complete-consumer phase allocator observations qualified on cleanfdfc748 (CPU4/NPU7 and128-node actual two-card Attention training calibration; [evidence](evidence/consumer-memory-20261002.md)); transactional optimizer recomputation qualified on clean81f4736 (eight jobs,28 actual consumer cases and102MiB/card reduction on D128 calibration; [evidence](evidence/resident-optimizer-recompute-20261002.md)); resident complete-consumer memory admission qualified on clean0b1a5aa (eight jobs,CPU8/NPU19,forced physical splitting and D128 calibration; [evidence](evidence/consumer-capacity-20261002.md)); native worker/packed transport consumer controls qualified on clean2222d9d (CPU11/NPU10,no skips; [evidence](evidence/consumer-host-execution-20261002.md)); eager multi-device ownership/complete-region control/remappable checkpoints qualified on clean55c3960 (CPU310/NPU87,no skips,standalone CPU24/NPU36 configurations,separate two-device profile; [evidence](evidence/eager-payload-owners-20261003.md)); actual multi-owner consumer integration qualified on cleane5d91d7 (CPU73/NPU40,no skips,fresh installed clients,separate actual Attention profile; [evidence](evidence/eager-consumer-owners-20261003.md)); packed cross-device transport qualified on cleana785d43 (CPU352+64,NPU119+40,standalone CPU24/NPU36,separate actual trace; [evidence](evidence/eager-packed-transfer-20261003.md)); eager admission qualified on cleanc686096 (CPU70,NPU12+40,original-width CPU2/NPU7 calibrations,separate profile; [evidence](evidence/eager-consumer-capacity-20261003.md)); B512 CPU/mixed execution and formal performance remain pending |
| F6 | Representative then full-size inference/training, aggressive-safe chunking preferred; CPU + screened mixed + resident, both schedules, three fresh repeats for recommendations and separate profiling | representative TimedDAG/LibTorch/prefill five-preset screen qualified on clean0b1a5aa:72 fresh processes,FP32 exact event/output/cut checks,three repetitions and four separate Attention profiles ([evidence](evidence/representative-preset-screen-20261002.md)); bounded host-policy comparison qualified on clean2222d9d:28 pilot +36 confirmation processes and one selected-policy profile;CPU16 packed wins inference,resident wins training ([evidence](evidence/representative-host-policy-20261002.md)); PDG/LibTorch prefill+streaming qualified on clean80dae6e:40pilot+72confirmation processes ([evidence](evidence/representative-pdg-matrix-20261002.md)); TimedDAG/LibTorch streaming, Settle/LibTorch both and TimedDAG/Python both qualified on clean80dae6e:100pilot+180confirmation processes ([evidence](evidence/representative-family-matrix-20261002.md)); Settle/Python both qualified on clean80dae6e:40pilot+72confirmation processes ([evidence](evidence/representative-settle-python-20261002.md)); all ten required representative submatrices complete; original17.521B Attention8-NPU FP32 TimedDAG/prefill inference executed with full B512/two windows ([evidence](evidence/original-wide-inference-20261002.md)); original9.468B Add8-NPU FP32 TimedDAG/prefill inference also passed ([evidence](evidence/original-wide-add-inference-20261002.md)); original-width AddB2 complete training and ten-card profiling audited ([evidence](evidence/original-width-add-training-20261002.md)); originalB512 Add and Attention complete training now passed below; formal comparisons pending; historical CPU Attention supplementary; ten-card B4/physicalB2 Add pilot failed post-run allocator calibration, B512 not entered ([retained failure](evidence/original-width-add-b2-refusal-20261002.json)) |
| F7 | Clean immutable builds/gates, portable packet/commands, recorded target-pending CUDA/version cells, reviewed minimal records/profile/evidence/support claims and no live task jobs | in progress; fresh CUDA-linked aarch64 core/installed client and187 CPU checks plus seven no-Git relocated entries qualified on clean2c04005 ([evidence](evidence/eager-cuda-host-20261003.md)); GPU/x86_64 device gates target-pending; final integrated audit and active-job closure remain open |

F5 eager FP16 consumers are qualified on clean7b1fae5: CPU61+55,NPU49,eleven
fresh-process memory calibrations and a separate actual consumer trace
([evidence](evidence/eager-fp16-consumers-20261003.md)). FP16 payload gradients
and FP32 loss/masters/slots are distinct from resident FP32 adjoints. Full-size
formal comparisons remain open under F6.

F6 original Attention B512 complete training is now qualified on clean29effae:
17.521B parameters,11 cards,physicalB1×512,two connected windows,one complete
FP32 SGD update5695.490452595s;outputs12288,cut408,finite loss,all allocator/context
peaks within unchanged estimates/pools. The separately declared9000s feasibility
budget passes; original3000s refusal,1.15 coefficient and pilot owner/chunk/capacity
configuration remain intact. Terminal source/build/packet/lease audit passed
([evidence](evidence/original-b512-attention-training-20261003.md)). Cold feasibility
only; formal CPU/mixed/resident matrix and final integration acceptance remain open.


F4/F5 explicit device gradient accumulation is qualified on clean830904b:
FP32/FP16,16trajectories/384windows/48updates,Python21,no skips; separate trace
without observed AiCPU ([evidence](evidence/resident-accumulation-20261002.md)).
It explicitly detaches between backward groups and performs one final optimizer
update. Independent device continuation switching is qualified on cleanc96ebcd:
FP32/FP16,32trajectories/768windows/96updates including accumulation regression,
Python30,no skips, independent trace0AiCPU ([evidence](evidence/resident-contexts-20261002.md)).
Resident consumer sample slicing is qualified on clean75543a7: CPU15/NPU45,
no skips, same-dtype and independent-CPU checks, separate two-device trace0AiCPU
and fixed-shape allocator peak -18.0% ([evidence](evidence/resident-sample-chunks-20261002.md)).
Automatic sample admission is qualified on clean3c3b4e7 (CPU9/NPU18;
[evidence](evidence/resident-auto-samples-20261002.md)); dense live KV remains pending. Optional compact
saved continuations are qualified on cleanf8cc052: Python18, four standalone
FP32/FP16 cells(768windows/96updates), separate trace0AiCPU and representative
saved-tensor bytes -98.09% ([evidence](evidence/compact-continuations-20261002.md)).
Per-device simultaneous-live compact pool admission is qualified on clean48e44b0:
CPU17,NPU64,four native cells768windows/96updates,independent0AiCPU trace; same-shape
whole training allocator peak -5.39% ([evidence](evidence/resident-context-pool-20261002.md)).
Live/retained memory and scale remain pending; automatic sample admission
has the separate qualification above.
Add/Attention gradient admission is qualified on cleanc3ed0f2 (CPU10/NPU9):
model inventory removes fictitious Add QKV/O charges, unchanged margins
([evidence](evidence/consumer-add-capacity-20261002.md)).

F4/F5/F6 shared training-storage accounting is qualified on clean4dd8368:
CPU17/NPU25 and D512 two-card Attention calibration passed; immutable Attention
copies and private accumulation are charged by actual lifetime, with unchanged
API limits/margins. Actual peaks are unchanged; no allocation or throughput gain is
claimed ([evidence](evidence/consumer-training-storage-20261002.md)). Original-width
training still needs a real storage/compute improvement.

F4/F5/F6 physical parameter-gradient lifetime accounting is qualified on clean789e1a5:
CPU23/NPU25 and D512 calibration passed; aggressive multi-device consumers charge
reused projection/Attention adjoints once, conservative/legacy paths stay per-window.
All safety/API budgets stay unchanged. Actual peaks and results are unchanged;
no new runtime/profile or scale claim ([evidence](evidence/consumer-gradient-lifetime-20261003.md)).

F6 original-width Add on clean789e1a5 passed one nine-card B4/physicalB2 complete
FP32 SGD update:21.179s, peak47,394,238,464bytes; events/outputs/cut and tolerant
loss match prior B4. All allocator/context checks pass. Unchanged B512 projection
3117.612s>3000s prevents B512 execution; no formal speed claim
([evidence](evidence/original-width-add-gradient-lifetime-20261003.md)).

F4/F5 private event/full-fiber Attention bank borrowing is qualified on cleanb5e6345:
eight terminal jobs,native160trajectories/2560windows/640updates,Python16,
actualconsumer32,no skips. Private owner lifetime/source/mode/node-map guards;
default/subset snapshots and dynamic records stay owned. Unchanged API budgets
and consumer admission. Same-lease D512 two-card peaks each -275262464bytes,
loss/statistics unchanged; separate FP16 profile53176ops,zero observed AiCPU.
Original dev01 ownership failure retained. No full-size Attention training or
formal throughput claim ([evidence](evidence/resident-attention-borrow-20261003.md)).

F5 bounded static memory-aware Full/state placement is qualified on cleanc38b72e:
CPU29,NPU36 (32 actual independent CPU-referenced complete candidates,three
capacity refusals,one interface check),D512 forced-owner calibration and separate
FP16 profile88089ops without observed AiCPU. At most2N moves/4096 trials;
nonempty explicit maps,conservative mode,canonical owners,all logical capacities
and margins remain. Old empty-tuple/underprovisioned-journal failures and one
build-helper failure are retained. Original Attention training and F6 remain
pending ([evidence](evidence/consumer-memory-balance-20261003.md)).

F5 fixed joint owner-map CLI replay is qualified on clean29effae: three terminal
jobs,CPU45/NPU22,no skips;twenty actual independent CPU-referenced candidates
(twelve explicit maps,eight automatic regressions) plus two invalid-map refusals.
Offline/Python/standalone parity,one/two devices,three families,both schedules,
FP32/FP16,complete updates/continuation. Existing envelopes,cuts,margins and
backend bytes unchanged. This enables comparable batch-geometry pilots; it is
not original Attention training or throughput evidence
([evidence](evidence/consumer-owner-map-20261003.md)).

F5 aggressive sharded Attention KV-journal lifetime accounting is qualified on
cleanaf137df: four terminal jobs, CPU38/NPU17, no skips; sixteen actual two-card
CPU-referenced complete-training candidates and one pre-allocation refusal.
Two live FP32 journal banks plus one retained bank/window replace the repeated
legacy allowance; runtime bytes, capacities, other storage bounds and margins
remain. Fixed-map/chunk D512 estimates decrease3784294400bytes/card while actual
peaks6347777024/5545201152,loss,statistics and continuation match the prior run.
No allocation or speed gain, original B512 Attention training or CUDA claim
([evidence](evidence/consumer-journal-capacity-20261003.md)).

F5 consumer private-bank liveness accounting is qualified on clean219719d:
four terminal jobs,CPU26,NPU25(no skips),24 actual CPU-referenced candidates and
pre-allocation refusal; unchanged b5e6345 backend. Aggressive multi-device declared
consumers avoid duplicate retained-copy charges; conservative/legacy copies,
dynamic storage/API limits/margins unchanged. D512 two-card estimates each
-568389140bytes; actual peaks7233247232/6386085888 unchanged with exact loss/
statistics/continuation. No allocation or speed gain claimed; original Attention
training still needs placement/scale work
([evidence](evidence/consumer-private-bank-capacity-20261003.md)).

F4/F5 private frozen projection banks are qualified on clean4467493: eight terminal
jobs,native160trajectories/2560windows/640updates,Python16,consumer32; aggressive
sharded training borrows only its private banks under the existing publication
barrier. Default/conservative/legacy snapshots remain independent; retained API
budgets and consumer estimates unchanged. Same-lease D512 peaks decrease
296,274,432bytes/card; separate FP16 trace53176ops/zero observed AiCPU
([evidence](evidence/resident-projection-borrow-20261003.md)). Original-size remains open.

F6 original-width Add on clean4467493 passed one unchanged nine-card B4/physicalB2
complete update20.770s,peak43,197,837,312bytes. All memory/work/loss checks pass;
B512 projection3057.317s>3000s, not executed. Optional sample-work/optimizer timing
is qualified on clean26176de: CPU13/NPU8,40 actual CLI candidates and four terminal
jobs; [evidence](evidence/consumer-phase-timing-20261003.md).
It defaults off and adds one synchronized training boundary when enabled. The
next original-width cost diagnostic must preserve original refusals and cost cap
([evidence](evidence/original-width-add-projection-borrow-20261003.md)).

F6 synchronized original-width Add cost diagnostic on clean26176de passed:
B4/physicalB2,9cards,two connected windows,one complete SGD update20.661979177s;
sample19.105143379s + optimizer1.556835798s. Old total projection3041.443335s
remains refused; phase projection2814.067467s admits a separate bounded B512
execution under unchanged3000s/1.15/memory checks. B512 not executed in this
pilot; [evidence](evidence/original-width-add-phase-diagnostic-20261003.md).

F6 original Add B512 complete training passed on clean26176de:9cards,
D2048/B512/T12/V50304,9.468B parameters,physicalB2×256,two connected windows,
one FP32 SGD update. Actual2170.549862239s<=3000s (sample2168.898689171s,
optimizer1.651173068s); outputs12288/events1183429/cut408,loss30.50836181640625,
all allocator/context gates passed. Terminal audited exit0; historical refusals
unchanged. Cold phase diagnostic with limited concurrent development on other
cards; not formal throughput or full-size CPU parity. Attention B512 training is
qualified separately above; eager mixed scale and full comparison/repeats/profiles remain pending
([evidence](evidence/original-b512-add-training-20261003.md)).

F6 original-width Attention fixed-map B4/physicalB1 on clean29effae completed
one11-card FP32 SGD update:50.418376377s,two windows,17.521B parameters,
outputs96/events9256/cut408,all allocator/context gates passed. Phase forecast
7036.453031774s>3000s with unchanged1.15 and original capacities;B512 was not
executed in that pilot. Preserve the refusal; subsequent separately budgeted B512
completion is recorded above. Neither run establishes formal throughput or F6 closure
([evidence](evidence/original-width-attention-owner-diagnostic-20261003.md)).

F4/F5/F6 immutable Full snapshots are qualified on cleanf360489: native144
trajectories/2432windows/560updates,Python16/consumer32, same-lease D512 allocator
-273408bytes/card and a separate FP16 trace with53182ops/zero observed AiCPU
([evidence](evidence/resident-full-snapshots-20261002.md)). This is a small real
storage improvement; wide fixtures have small LH Full banks. Consumer admission
is unchanged; dominant projection/Attention reverse storage and scale remain open.

F5 eager physical sample chunks are qualified on cleane6cc52b: CPU60/NPU38,
whole-batch loss/gradient/update equivalence and separate medium memory observations
([evidence](evidence/consumer-sample-chunks-20261002.md)). Explicit maximum only;
mixed multi-device subsequently qualified on cleane5d91d7. Automatic
[eager admission](eager-consumer-capacity.md) is implemented and has passed68
directed CPU development checks and CPU/NPU installed-client builds; clean
qualification, allocator calibration and original-scale mixed completion remain
pending. Resident slicing uses one parameter/optimizer owner and a single
whole-batch update. [Portable eager CUDA commands](eager-target-validation.md)
remain target-machine-pending.

F4/F5 optional training diagnostics are qualified on cleanca26b47: 51 public tests,
four standalone cells, same-shape allocator reduction 16.26 MiB. Required VJP
journals remain; no throughput or full-size claim
([evidence](evidence/resident-training-records-20261002.md)).

F4/F6 append-only fiber KV staging is qualified on clean80dae6e (CPU4,public NPU33,
four standalone component cells,128MiB lower actual representative training peak;
[evidence](evidence/resident-fiber-append-20261002.md)). Logical capacities and
retained semantics remain unchanged; full-size storage/placement still pending.

F4 compact state/Read/KV forward is verified at62935e9; [evidence](evidence/device-state-owners-20261001.md).
Compact retained state/cache reverse and canonical publication into these banks
are qualified at49541be; [evidence](evidence/device-state-reverse-20261002.md).
Bounded device completion chains avoid concatenating retained windows into one
persistent task buffer. Public multi-device training/session/checkpoint/client is
qualified at0d7c45e ([contract](resident-sharded-training.md),
[evidence](evidence/public-sharded-training-20261002.md)); scale consumers and
complete performance matrix remain pending.
The scale model uses per-edge slot-affine projections. HARD resident slot-affine
adjoints and canonical publication are qualified on clean96c75f8
([evidence](evidence/resident-emission-training-20261002.md)). Compact projection
owners are qualified on cleanacb84f3 ([evidence](evidence/resident-projection-shards-20261002.md)). Actual FP32 public consumers are qualified on clean0d61cb9
([evidence](evidence/online-resident-consumers-20261002.md)); full-size performance
remains pending ([consumer contract](online-consumers.md)). Do not substitute a broadcast model.

F4/F5 public multi-device inference is qualified on clean7329c71
([evidence](evidence/public-sharded-inference-20261002.md)): CPU74,library47,
actual consumers27 and a separate two-card inference trace. Full/state/KV
placement,one/three-card complete-cut restore and Python-owned FP32/FP16 are
covered; actual consumer FP16 is separately qualified on cleand178b86
([evidence](evidence/resident-consumer-head-20261002.md)). Compact projections are separately qualified
on cleanacb84f3; total-memory admission and F6 remain pending.

F4 bounded device-loop canonical contribution/publication streaming is qualified
on clean7e375f5 ([evidence](evidence/resident-owner-stream-20261002.md)). It preserves reverse-window/alias
addition order, bounds both packet endpoints, and releases obsolete optimizer gradient
storage. This does not replace total per-device memory admission or F6.
Per-phase per-device packet reuse is qualified on cleana72868f: seven native
component cells,33 consumer checks,explicit2→3restore32trajectories,representative
allocator -256MiB/card and separate FP16profile
([evidence](evidence/resident-shared-packets-20261002.md)).

F4/F5 local reverse owner/query/key budget splitting is qualified on clean106cbeb
([evidence](evidence/resident-reverse-budget-20261002.md)): CPU1,public cache83,
actual consumers21,32 FP32/FP16 trajectories and a separate actual D32 two-card
profile. Total-memory admission and the F6 performance matrix remain pending.

Apply canonical online greedy algorithms; large blocks when legal, single-action
fallback and natural streaming degeneration otherwise. Compress logical-time
recursion into block computation where possible; do not claim universal constant
stage count or identify one stage with one operator/kernel. Never use an advance
numerical trace, input-specific scheduler or static capture as a substitute.
Computational and scheduling kernels can use different backend mechanisms.

Both conservative and aggressive-safe chunking remain configurable. Formal performance
prioritizes larger batches within calibrated safe budgets, persistent allocation
accounting and headroom, splitting before over-budget work. Profiling is part of
the implementation/verification loop, including batching itself, actual kernels,
host decisions/sync, transfers, padding and memory; instrumented timings stay separate.
No discarded events, hidden KV truncation, dtype changes or gradient-boundary changes.

Prioritize general algorithm/independent correctness, complete public/device/multi-card
training paths, staged full-size comparisons, then final records and supplementary
historical baselines. Preserve current development code and failures. Every long job
has a concrete question, bounds/stops and failure response; waiting is not the main
work. Keep formal heavy timing uncontended; independent development work may proceed
without contaminating it. Use available devices within the resource budget, never
stop unrelated jobs. Commit tested implementation, qualify clean source, commit
evidence separately; push tested commits under the renewed user authorization. Do not call a failed finite capacity assessment delivery.

Exact named C++ initialization is qualified on cleanbe380db:CPU25/NPU30
consumer checks plus three independent CPU processes with full-array byte
comparisons. Projection/QKV/head generation improves6.25–6.51×; no whole-model
or complete-step speedup claim ([evidence](evidence/exact-initializer-20261002.md)).

Valid-prefix retained journals under aggressive policy are qualified on clean0fbc1b2:
CPU17/NPU44,64 standalone FP32/FP16 trajectories/1,024windows/256updates,
separate two-card profile without observed AiCPU. Actual representative Attention
allocator peak1.740→1.356GiB(-22.10%),identical loss/events/output/cut; full consumer
admission still charges dense retained envelopes
([evidence](evidence/resident-retained-journals-20261002.md)).

Update-scoped immutable attention snapshots are qualified on clean38858d0 for
single/sharded training owners: all nine fixed-source jobs, Python16/native128
trajectories/2,048windows/512updates, actual consumers24 and2-to-3-device restore.
Actual forward-bank guards cover fresh grouped fiber gathers; dynamic KV/log-bias/
journals stay per-window. The same-lease D512 allocator comparison saved262.51MiB
per card; a separate FP16 profile observed no AiCPU. Complete-consumer admission
has not been reduced ([evidence](evidence/resident-attention-snapshots-20261002.md)).

F4/F5 private numeric gradient accumulation reuse is qualified on cleanb3a6a24:
all8 jobs passed;50 native boundary cases,32 trajectories/768windows/96updates,
Python14/actual consumer24 and a separate FP16 profile without observed AiCPU.
Old flags and public exports remain independent; the same-lease D512 whole-process
allocator peaks were unchanged. No original-size/throughput claim follows.
[Evidence](evidence/resident-private-accumulation-20261002.md).

F5/F6 aggressive operator chunk selection is qualified on cleane82f971:CPU13,
NPU17 independent comparisons/refusal and forced-splitting FP16 profile; unchanged
memory envelopes/margins and conservative policy. Nonlimiting operator maxima can
remain larger; original-size throughput remains to be measured.
[Evidence](evidence/consumer-chunk-selection-20261002.md).

F6 original-width Add after that planner: ten-card queue timeout retained;
one nine-card B4/physicalB1 FP32 SGD update passed,27.300s and41.097GiB maximum
allocator. Logical work matches the prior B4 run and loss passes existing FP32
tolerance. Original B512 static memory admission passes, but the4018.543s cost
projection exceeds the unchanged3000s gate; no B512 execution or formal speed
claim ([evidence](evidence/original-width-add-chunk-selection-20261002.md)).

## Six implementation classes

All required cells target CPU FP64/FP32, inference and first-order training.
Generic/specialized schedules must be independent; local kernels may be shared.
The final gate covers values, identities, every state slot, histories, Aggregate
contributions, Read/Next/comparison/Full/Emit, routes, pending, occurrence ledger,
isolated roots and input/parameter/initial-state VJPs, None/zero connectivity,
shared/unused owners, optimizer updates and continuation. HST uses its declared
surrogate VJP. Empty/ragged/parallel-edge/reset/delay/chunk regressions remain.

| Class | Actual implementation and existing gate | Accepted scope / evidence |
| --- | --- | --- |
| PDG generic | `reference.py`, `cpp/src/stream.cpp`, cursor; CSR/CSC, feedback, lazy state, packed and node workers; [streaming](evidence/m1-streaming.md), [cursor](evidence/native-cursor.md), [isolated roots](evidence/isolated-autograd.md), [transport](evidence/packed-transport.md) | verified: [final7741-test gate](evidence/foundation-final.md), options and bounded performance |
| PDG specialized | `specialized.py`, `cpp/src/specialized.cpp`: independent self-loop/ring propagation; `test_specialized.py`, `test_isolated_schedules.py` | ring verified; [stage2 gate](evidence/foundation-stage2.md) |
| TimedDAG generic | `frontier.py`, native planner/frontier/block; streaming is restricted PDG; [frontier](evidence/m3-frontier.md), Attention/SSM packed sequences | verified: migrated options, real prefill, [unified entry](foundation-benchmarks.md) and final gate |
| TimedDAG specialized | independent Python/native chain/diamond, including shared middle region; [specializations](evidence/m4-settle-specialized.md) | diamond/paired-region verified; [stage2 gate](evidence/foundation-stage2.md) |
| SettleGraph generic | `settle.py`: independent Python region-major; independent native `settle.h` frontend using frontier, with encoded streaming comparison; [encoding](evidence/m4-settle-specialized.md), [ports](evidence/local-ports.md), [origins](evidence/source-origins.md) | native `settle.h` [verified](evidence/native-settle-frontend.md) for rank-aligned profile |
| SettleGraph specialized | independent Python `settle_chain`, analytic formulas; Attention/SSM isolated roots | layered scalar schedule verified; [stage2 gate](evidence/foundation-stage2.md) |

## Stage gates and required units

| Stage/unit | Required delivery and verification | Status |
| --- | --- | --- |
| S1 | audit implementation/module/training/option matrices; freeze benchmarks, resources and stops; remove superseded tuning priority | verified by [scope audit](evidence/foundation-scope-audit.md) |
| S2.1 | standalone C++ SettleGraph spec, structural encoding, model alias mapping, inputs and complete-boundary projection; no Python dependency; standalone FP32/64 forward/VJP, negative validation and Python/native parity | verified: [native frontend](evidence/native-settle-frontend.md); 230 directed tests and clean C++-only FP64/FP32 gate |
| S2.2 | independent ring and diamond Python/C++ schedules, Python layered SettleGraph; full trace/isolated VJP/initial state/cuts and representative modules | verified; 7167-test [clean stage2 gate](evidence/foundation-stage2.md) |
| S2.3 | explicit Settle → TimedDAG → PDG clock, node/region/edge/source/port, state/history/message/pending/ledger mapping and cut restrictions; source-aware roots | verified for declared profile; [stage2 gate](evidence/foundation-stage2.md), contract in settle-embedding.md |
| S3.1 | actual module step/batch/sequence capabilities and counters, Attention/GQA/window, distinct same-fiber, Linear/Delta/Gated Delta/SSM, FFN/SwiGLU, Agg/Emit | verified including separately named ungated DeltaRule;736 directed tests, S3/S4 and final7741-test gate passed; [capability table](execution-capabilities.md); fallback and new-schedule gates passed |
| S3.2 | migrate packed_sources and batch_next/reset to legal frontier/encoded Settle blocks; compact/region parallel/deferred-release applicability with explicit errors; single-option, interaction and historical failure tests | verified:1098 directed CPU tests, [S3/S4 gate](evidence/foundation-stage34.md) and final gate |
| S3.3 | small deterministic model-style adapter composing layout, norm, explicit position/RoPE/mask/cache; CPU formula, chunk and VJP anchors | verified: tiny explicit RMSNorm/RoPE/GQA/SwiGLU adapter; independent formula/cache/chunk/VJP and final gate passed |
| S3 gate | nontrivial time batches and counted causal fallbacks; independent simple path, node parallel, packed/unpacked, default/option parity; no backward-speed claim from scalar replay | directed option/module gates passed; 7611-test [S3/S4 clean gate](evidence/foundation-stage34.md) passed |
| S4.1 | six-class isolated training roots, shared/unused owners, None vs connected zero, multi-step SGD/momentum/AdamW + decoupled decay, eps=1e-5; detached/chunk/partial buffers | verified extension;274 directed tests plus54 corrected multi-window Settle tests passed; [S3/S4 clean gate](evidence/foundation-stage34.md) passed |
| S4.2 | interrupt/save/new-process restore/continue vs uninterrupted trajectory for promised single-graph, two-clock application and native value scopes | verified:36 fresh subprocesses cover both dtypes, SGD/momentum and AdamW, single/application/native schemas; final payload/identity audit passed |
| S4 gate | transactional malformed-input rejection and publication failure; graph/model/owner/alias/optimizer/group identity | [Python values](evidence/checkpoint-ownership.md), [publication](evidence/checkpoint-io.md), [native format](evidence/cpp-native-checkpoint.md), [application](evidence/token-checkpoint-coordinates.md) verified; retain, do not rebuild formats |
| S5.1 | unified smoke/non-smoke entry and fixed suite below; explicit nograd-forward/grad-forward/backward/optimizer/train-step implementations and bounds | verified unified graph runner;130 directed runner/CLI tests and final gate passed; [entry/modes](foundation-benchmarks.md); LH/PDG portable kit remains inference-only |
| S5.2 | three graph families measured; both large shape presets counted and evaluated under resource preflight, timeout and process reaping; report achieved size and failures | all108 medium runs completed; [medium evidence](evidence/foundation-medium.md); [large assessment](evidence/foundation-large.md) reviewed:7 complete/5 timeout/1 unlaunched across new runs; wide PDG/LH reused, corrected16.608B historical narrow count and frozen8.497B supplement; target-scale success not claimed |
| S5.3 | bounded optional tuning, if justified; all defaults evidence-based, at least three independent repeats for gain claims; export/rebuild/smoke from new directory | closed:zero new candidates, existing negative findings/defaults retained; fresh relocated rebuild/all17 smoke variants passed |
| S6 | freeze clean implementation commit, independent read-only worktree, complete CPU gates + standalone/relocation, source/build/binary/result/exit audit; evidence commit and final matrix | verified: [final gate](evidence/foundation-final.md), clean81a1b26,7741 tests,17 relocated smoke,36 fresh-process payloads and separate no-Python standalone build; all task units terminal |

S6 has finished this acceptance version: all required correctness cells passed,
fixed performance assessment is honestly closed, and no task live job remains.
The final local commit/status is in STATUS. No extra platform/model/tuning work follows.

## Frozen performance suite

Machine-readable definitions: `benchmarks/foundation-v1.json`. Twelve logical
medium/small configurations only. Shapes/workloads cannot be multiplied as
variants; baseline/option/schedule variants keep the same logical workload.
P01/P02 deliberately form one fixed-touched-work topology-size comparison.
Use smoke D16/B4/V257/6 steps/warmup2 for portable LH/PDG; legal small chain,
diamond and layered equivalents for the other graphs. Smoke has no speed claim.

| ID | Workload, width/batch/sequence, static body nodes | Main comparison |
| --- | --- | --- |
| P01 | sparse PDG ring, D128/B8/T128/N128, 4 touched nodes | Python/native, scalar/packed |
| P02 | same touched subgraph plus dormant nodes, D128/B8/T128/N8192 | compare P01, sparse allocation and topology cost |
| P03 | source transport and region/node work, D512/B8/T128/N32, same-fiber attention | serial/workers, Agg/Emit/Next and opt-ins |
| T01 | TimedDAG diamond, D128/B8/T128/N4, event attention | streaming/frontier/independent diamond |
| T02 | TimedDAG four regions, D128/B32/T128/N16, SSM | legal prefill/causal fallback |
| S01 | Settle four layers, D128/B8/T128/N8, event GQA/window128 | generic/encoded/specialized/prefill |
| S02 | Settle chain, D512/B8/T128/N3, SSM/SwiGLU | step/packed sequence |
| A01 | TimedDAG chain, D128/B8/T2048/N2, ragged event GQA | long cache, padding, causal mask |
| A02 | TimedDAG diamond, D128/B8/T512/N4, same-fiber attention | exact/single, CSR/layout/cache |
| M01 | TimedDAG chain, D128/B8/T128/N3, Linear attention | step/chunk/sequence |
| M02 | TimedDAG chain, D128/B8/T128/N3, Gated Delta | step/chunk and counted fallback |
| TR01 | legal diamond shared by three families, D128/B8/T128/N4, SSM/SwiGLU, window16 | grad-forward, backward/replay, optimizer, complete step |

Large presets are limited to wide D2048/B512/nominal1:32 and narrow
D128/B512/nominal1:64. Wide reference is actual ~17.27B Attention, LH 224 leaves
+8 hubs per cortex; inet/onet states counted separately. Narrow target ~8–9B
and tens of thousands of nodes requires exact owner/work counts and staged
resource estimates before allocation. DAG/Settle use legal ranked topologies;
report parameter/active/KV/matrix-work differences. Existing wide evidence and
narrow failure are reusable; no mechanical repeat without a new question.

Every record includes workload/profile, shape/dtype, actual activation, owner
counts, touched nodes/edges, logical Agg/Upd/Next/Full, actual packing/kernels,
padding/matrix work, reset/warmup/detach/window, CPU/thread pools/NUMA, wall
latency/throughput/distribution, RSS/cgroup, correctness anchor, source/build/
binary hashes and terminal status. ms/sample-token divides measured wall time
by batch and measured effective steps; also publish batch-step latency.

## Resource and exploration limits

Recompute from affinity ∩ cpuset and all ancestor CPU quotas; use approximately
half effective CPUs in aggregate. Memory bound is half min(host total, current
MemAvailable, cgroup limits), across the whole task. Include parameters, grads,
optimizer, KV and transient/build costs before large allocation; measure actual
RSS/cgroup (RLIMIT_AS alone is not memory accounting). Stage-one observation:
320 usable logical/physical CPUs, no finite quota, budget160; memory budget about
755 GiB at audit, dynamic. Eight NUMA nodes; choose affinity from discovered IDs.
Correctness uses ATen/OpenMP/BLAS1, build2. Record effective pools, not just env.
No heavy job overlaps formal timing; large variants run sequentially. Durable
background.slice/Nice10 jobs have bounded timeouts and complete child reaping.

A new bounded performance increment considers at most four bottlenecks and two
main candidates each. The frozen acceptance selected none; separately authorized
Full and Aggregate follow-ups are recorded below. Small correctness/cost probe
first, at least three independent repeats plus spread for speed claims. No benefit/regression/instability closes
an experiment; keep conservative defaults. Implementation/correctness work is
not capped by those exploration counts. Existing exact/single, fiber and packed
transport trials are closed evidence, not an invitation to restart wide search.

## Explicit extensions after acceptance

Optional Ascend starts with real single-card CPU-parity operators/forward/VJP,
maximum eight available cards; separate Python TorchNPU and C++ qualification.
CUDA, x86 execution on a different host, arbitrary open-weight imports, other
architectures, higher-order AD, full controller/RNG/data-cursor resume, arbitrary
LH configs and general proof for all graphs/modules are outside this gate.
Stable node locality and batched persistent KV remain optional bounded candidates,
not blockers or existing capabilities. LH original C++ is inference-only authority
for the bounded exact mapping; scale comparisons permit independent weights.

## Completed follow-up: Python TorchNPU boundary

Authorized 2026-09-25. The shared Python graph/model/checkpoint path now has an
explicit `cpu|cuda|npu|auto` runtime boundary with lazy TorchNPU loading,
logical-device resolution, synchronization, CPU-portable checkpoints and
deliberate optimizer-state placement. A clean aarch64 Ascend 910 smoke from
commit `7811418` passed PDG streaming, TimedDAG frontier/diamond, generic and
layered/chain Settle, one first-order backward, isolated Linear/Aggregate VJPs,
an optimizer step and a fresh checkpoint handoff. The exact command, runtime
manifest and comparisons are in [Python TorchNPU evidence](evidence/npu-python-20260925.md).

This verifies only the recorded local eager FP32 Python scope. The local
TorchNPU stack rejects the FP64 graph matmul path, so Python NPU FP64 is
unsupported by design. The smoke is a correctness/parity result, not a
throughput, distributed, all-module or host-fallback audit; optimized operator
traces were unclaimed at that revision. C++ NPU was unsupported because the installed
wheel's `libtorch_npu.so` is classified as `python-wheel-runtime` and lacks the
standalone SDK/CMake/public ABI required for a matching LibTorch build. The
historical report retains the Python cell and blocked C++ attempt. The authorized
2026-09-28 accelerator extension above supersedes that implementation boundary.

## User-requested follow-up: Add scale comparison

2026-09-24 completed: configurable Add/grad-forward/outer-timer comparison;
472 directed CPU FP64/FP32 tests,4 fresh native smoke runs and12 D2048/B512/56-core
runs (3 repeats per engine/mode), all passed. [Reviewed results](evidence/add-scale-comparison.md)
retain the exact9,468,020,899 count, overlapping nograd repeat ranges and15.18x
PDG/LH grad-forward median latency. [Contract](add-scale-comparison.md).
No backward-throughput or arbitrary model/backend claim follows.

The subsequent Full/Aggregate follow-ups below implement isolated packed VJPs
while retaining scalar replay, None/connected-zero and ownership. Remaining
extensions include separating startup/steady-state limits in parallel large
sparse runs, costly Linear/Delta scans and one concrete model adapter. Existing
CPU/memory/Ascend bounds,
independent anchors and repeat requirements apply across all graph families.
This recommendation does not reopen completed acceptance or queue another sweep.

## Completed follow-up: PDG grad-forward

2026-09-24: optional native Full batched affine VJP, default replay retained.
[Evidence](evidence/full-batched-autograd.md), [contract](full-batched-autograd.md).
Clean690-test selected FP64/FP32 gate,4 smoke/2 probes/4 wide runs all passed.
D2048/B512 optimized median10.49906ms/sample-token (3 repeats), versus current
same-binary replay110.43917 (1 repeat) and retained replay108.23288 (3 repeats).
No backward-throughput or all-model/platform claim. Implementation reusable by
legal frontier/Settle calls with bounded correctness; no speed claim there.

The subsequent follow-up below supplies Aggregate/state/Read attribution and
small actual complete-training timing. Do not repeat this Full sweep without
a new question. Earlier resource bounds, independent anchors, honest failure records and three-repeat gain checks apply.

## Completed follow-up: Aggregate VJP and remaining grad-forward costs

2026-09-24: Full-batched diagnostics selected Aggregate for the wide Add case;
Attention small state replay remains a separate candidate. Optional native
`aggregate_autograd=replay|batched` is independent of Full; default replay stays.
Source scaling/reduction VJPs are batched, while normalization preserves per-event
Jacobians and near-zero AdamW behavior. [Contract](aggregate-batched-autograd.md),
[reviewed evidence](evidence/aggregate-batched-autograd.md).

Clean 552-test selected CPU FP64/FP32 gate passed after an independent build.
All 25 runs passed: 4 smoke, 2 cost probes, 12 small actual training, 6 wide Add
forwards and 1 separate diagnostic. No tolerance changes. Complete training
owner values/final gradients/losses match across all six small replay/batched
pairs. Source/archive/build/packet and terminal-record audits passed.

D2048/B512/56 CPUs: replay versus batched Aggregate median 11.61919 versus
8.55906 ms/sample-token, with Full batched throughout and 3 processes per policy.
Latency -26.34%, throughput +35.75%; peak RSS 77.43–78.76 versus 73.54–74.09 GiB.
Small D128/B16/T16, median total time across 4 training windows: Add 1.44187→1.04816s,
Attention 5.47745→4.98471s. These small timings do not certify wide-model training.
No new LH or wide no-grad performance result. Portable kit and complete commands
are linked from the evidence; all jobs are terminal, no task is queued.

Next choices, only for a separately requested bounded increment: wide Add now
spends most coordinator time in Full (2.77939s versus Update 0.59177s in a separate
diagnostic), with scalar Full/Aggregate replay zero. Profile within that phase
before choosing another large-Add kernel. For Attention, state/KV VJPs have the
stronger small-probe signal; retain replay and test backward/optimizer as well.
State/Read replay remains. Shared native TimedDAG/Settle paths have bounded
correctness coverage, not a new throughput claim. Resource/repeat limits above
apply across graph families; acceptance stays closed.


## Completed extension: cross-family streaming and prefill policies

Authorized 2026-09-24. This finite extension reuses the completed PDG performance
policies in TimedDAG and SettleGraph, for streaming AND legal prefill. It does
not reopen the original six-stage acceptance or remove any semantic roots.
Defaults remain conservative. STATUS owns the current handoff, this section
records the finite delivery. Historical foundation-v1 stays frozen.

| Unit | Required result | Status |
| --- | --- | --- |
| E1 | Versioned option/capability and benchmark definitions: separate scheduler, kernel, model-layout and application-head policies; requested/resolved/fallback records; reject unsupported explicit requests | verified: clean qualification and [cross-family audit](evidence/cross-family-qualification.md) |
| E2 | Python independent chain/diamond and Settle layered/chain block schedules; retain scalar schedules; Python isolated Full/Aggregate batched VJPs; native streaming/frontier/specialized/native-Settle uptake and actual sequence/fallback counters | verified: clean qualification and [cross-family performance](evidence/cross-family-performance.md) |
| E3 | CPU FP64/FP32 full-observable, isolated VJP, aliases/None/zero, optimizer/checkpoint/cut/detach/snapshot and prefill-to-streaming policy-switch tests; targeted combinations plus individual policies | verified: 8,577-test CPU gate, 60 relocated smoke variants, and [cross-family audit](evidence/cross-family-qualification.md) |
| E4 | Fixed bounded smoke/medium/large assessment below; separate prefill, streaming, transition, forward/backward/optimizer/whole-step timings and actual paths; explicit terminal failures/limits | verified: [terminal performance assessment](evidence/cross-family-performance.md); one narrow Settle target timeout retained |
| E5 | Frozen clean independent build/gates, relocated export/rebuild/smoke, source/binary/result audit, reviewed evidence and coherent commits; no live task job | verified: [qualification and audit](evidence/cross-family-qualification.md); unit inactive/dead |

### Extended-timeout follow-up

The original narrow Settle target timeout is retained as a failed bounded
record. On 2026-09-25 a separate durable run from the same frozen source and
build increased only that run's bound from 900 to 1800 seconds. Both the
resource stage and the 46,912-node target completed; the target took
1034.865380 seconds, used 8,496,773,056 parameters and peaked at about 51.82
GiB combined RSS. The follow-up output is recorded in
[cross-family performance](evidence/cross-family-performance.md). This is a
bounded confirmation for the exact workload, not a blanket large-scale or
accelerator claim.

Coverage: PDG generic/specialized are regression baselines. TimedDAG generic
Python/native: streaming, frontier, legal blocks and transition. TimedDAG
specialized Python/native: independent chain/diamond step and block schedules.
Settle generic Python/native: region/block prefill, encoded streaming, chunk
continuation and the native frontend. Settle specialized at least Python:
independent scalar and layered/chain block schedules. Python node workers are
not required; native core stays independent of Python.

Policies include workers/packed/prefill, Full/Aggregate replay|batched,
packed_sources/batch_next, compact_events/parallel_regions/defer_state_release,
applicable exact|single attention packing, scalar|CSR pooling, cloned|owned KV,
event|head layout and input|linear projection layout. Head workers belong to
application DenseLinear, absent in graph-only workloads. A missing implementation
is not semantic N/A. Known causal gates (selected adoption, clear, custom Next,
no exact state sequence) retain observable fallbacks. State/Read replay remains
separately counted; removing all replay is outside this extension.

Fixed measurement budget: at most 12 logical workloads, at most 6 formal
variants each, 3 independent processes for speed claims. Smoke D16/B4/T6;
medium D128/D512, B8/B32, T128/T512, ragged attention T2048 where needed.
Large wide D2048/B512/nominal1:32, Add ~9.468B or Attention ~17B; narrow
D128/B512/nominal1:64 ~8-9B is assessed in resource stages. Count actual
parameters, active work, topology/encoding boundaries and explain differences;
do not repeat known timeouts without a new policy/workload question.

Dynamic aggregate half-effective-CPU/half-memory limits apply to every family;
no fixed256GiB cap. Ascend at most8 cards only for an explicit later backend
extension; this gate requires CPU FP32/FP64. Existing durable-job, frozen-build,
reset, repeats, failure retention and timing denominator rules apply.

Completion requires E1-E5, real sequence batching where legal, independent
reference/specialized schedules, passing mandatory correctness cells and an
honestly terminal fixed performance assessment. Every option need not speed up.
This extension is complete. The bounded narrow Settle target timeout is part of
the delivery evidence; it leaves no live task and does not establish a general
target-scale success claim.

F4/F5 compact projection banks/adjoints are qualified on cleanacb84f3
([evidence](evidence/resident-projection-shards-20261002.md)):31 library checks,
48 standalone trajectories,27 actual-consumer checks and a separate two-card
training trace. Actual resident consumer FP16 and budgeted packed head/loss are qualified on
cleand178b86 ([evidence](evidence/resident-consumer-head-20261002.md)):10 CPU,59 NPU
checks,D2048/V50304 head calibration and a separate FP16 training trace. Total per-device admission and F6 remain pending.

Resident window capacity observations are qualified on clean475d4af,34 NPU
consumer cases across both clients/precisions; existing queue peaks and per-window
event/stage/output maxima are recorded without numerical Result exports
([evidence](evidence/resident-window-peaks-20261002.md)). Full-size training remains open.
