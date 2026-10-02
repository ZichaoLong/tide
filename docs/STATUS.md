# Current handoff

Updated 2026-10-02. **ACTIVE: user authorized continued execution and pushes.**
No pause instruction. No subagents. Reference repositories and ObsidianVault are
read-only. Repository /home/zlong/llm/graph-execution-foundation resolves to
/var/tmp/zlong-graph-execution-foundation/repository; branch graph-execution-foundation.
[execution-flows.md](execution-flows.md) is the contract; [ROADMAP F1–F7](ROADMAP.md)
is the only backlog. Overall task remains incomplete.

## Contract and working policy

Every candidate independently consumes common inputs/parameters/initial state;
never CPU reference events/routes/results/gradients. General online greedy accepts
legal topology/input including positive-delay PDG feedback, with natural streaming
fallback. Preserve int64, stable order, parallel-edge identity, missing/zero
messages and None/zero gradients. Performance: PDG LibTorch; TimedDAG/Settle
LibTorch+PyTorch; CPU/NPU × streaming/prefill × inference/complete training.
Python resident is a C++/CANN client, not an independent PyTorch scheduler.
Five presets and fine switches remain; FP32 main, FP16 separate. Training means
loss/VJP/optimizer/continuation/throughput, not downstream model convergence.
Contract outranks run-ml-experiments; minimal existing records, Trackio nonblocking.
Only affected gates, not unchanged8,954 CPU checks. Implementation commit →
immutable qualification → evidence commit; push each. Never edit live-job inputs.

## Completed; do not repeat

Latest implementation2222d9db25a234eb0c739baf11bdccfffa2c5f5e is committed/pushed.
Native consumer --workers/--packed-sources/--batch-next qualified CPU11/NPU10,
zero skips,16 complete-training comparisons plus resident CLI/budget regressions.
Four fixed-source jobs passed:build-host-execution-{cpu,npu}-clean01 and
host-execution-{cpu,npu}-clean01. Core/resident ABI and kernels unchanged.
CPU fresh consumer build; NPU11 source/header/options-verified reused objects,
fresh link/loader, byte-verified qualified dependencies. Report/audit:
[evidence](evidence/consumer-host-execution-20261002.md).

Earlier complete-consumer capacity admission0b1a5aa passed eight jobs,CPU8/NPU19,
24 cross-language plans,eight forced-splitting FP32/FP16 training cases,D128
calibration. See consumer-capacity.md and evidence/consumer-capacity-20261002.md.
Shape estimates concern simultaneous lifetimes and physical chunks; they are
not allocator quotas or arbitrary module/vendor guarantees.
Earlier public sharded training/FP16/continuation/checkpoint/compact banks/
canonical publication/optimizer recomputation remain qualified, indexed by F4/F5.

Representative five-preset evidence78190e3 is pushed:source0b1a5aa,72 fresh
processes +four Attention profiles,TimedDAG/LibTorch/prefill,Add/Attention,
inference/training,five FP32 presets +separate resident FP16,CPU workers1/ATen1.
Mixed-a wins among mixed presets. No observed AiCPU in the four traces. FP16
changes events/loss; it is a separate precision trajectory,not FP32 parity.

Bounded host-policy pilot and confirmation are finished on clean2222d9d:
host-policy-pilot01(28cells),host-policy-confirm0{1,2,3}(36fresh processes),
host-policy-profile01 all PASSED. No further worker search/repeats needed.
Pilot CPU(default1,packed1/4/16) and mixed-a(default1,packed1/4) selects CPU16
packed and mixed-a4 packed in all4groups. Packed enables both switches.
Confirmations each1continued warmup+3measured steps,2windows/64tokens per step,
FP32,ATen/BLAS1,one leased NPU,rotated order,own heavy measurements serial.
Event/output/cut checks exact,loss<1e-5. Median seconds per complete step:
- Add inference CPU0.131925,mixed1.168905,resident0.158360 (resident0.833×CPU throughput).
- Add training CPU0.560248,mixed3.173613,resident0.361768 (1.549×).
- Attention inference CPU0.253683,mixed2.473379,resident0.296540 (0.855×).
- Attention training CPU1.180089,mixed5.185804,resident1.047770 (1.126×).
All four short-process comparisons including startup favor CPU. Selected mixed
profile:275217tasks,265753host launches,16502sync copies,no observed AiCPU;
no trace-loss/capacity warning. Task sum is not wall time. Not full-size/F6-wide.
Report:evidence/representative-host-policy-20261002.{md,json}.
Audits passed:
python TASK/launchers/host_execution_evidence.py 2222d9db25a234eb0c739baf11bdccfffa2c5f5e
python TASK/launchers/host_policy_evidence.py 2222d9db25a234eb0c739baf11bdccfffa2c5f5e
No active build/test/profile job remains. Evidence docs grouped in next commit.

## Current source, builds and inputs

TASK=/mi/data2T/zlong/tide-execution-flows.
Frozen source TASK/sources/host-execution-clean01(clean2222d9d).
Current standalone consumers:
TASK/builds/host-execution-{cpu,npu}-clean01/consumer/tidegraph-online-bench.
Core builds placement-{cpu,npu,npu-python}-clean01; resident backends
optimizer-recompute-clean01 / optimizer-recompute-python-clean01.
Independent standalone/Python-owned runtimes must remain separate.
Representative packets TASK/inputs/screening-representative-{add,attention}01:
128nodes/544edges,D128/B8/T4/V257,clear=true,8,995,632/17,384,240parameters.
Resident queue/arrivals4096,outputs128,trace8192,KVtrace65536,forward8GiB,
retained8GiB,backward128GiB,optimizer4GiB,head256MiB,aggressive.
These local ceilings are not summed HBM allocations.

## Next priority and real remaining gaps

1. Commit/push audited host-controls and host-policy evidence. Then investigate
   actual resident allocation lifetimes on a qualified small/medium shape before
   changing KV/journal storage or total admission. TorchNPU memory.py exposes
   _record_memory_history/_snapshot; availability and useful C++ attribution are
   not yet tested. If used, bound the trace and run separately from throughput.
2. Complete remaining representative family/language/schedule cells and generic
   scale capacity/placement, then full-size CPU+screened mixed+resident comparisons.
3. F7 final qualification/portable packet; CUDA and other CANN tuples still need
   real target tests. No blanket rebuild or repeated full-suite loop.

Full-size input packets already exist,not executed:
TASK/inputs/fullsize-{add,attention}01/workload.json.
Add workload hash116b5d52db98b10a5e3a8cbe51fbc8aa0cce74aad8355ce1ea7f03dc83862a6b;
Attention88bb4b6ff11529606ad670956bab6590761bc8ff48c49b6e87dde40fe3da813f.
Original wide480nodes/2208edges,D2048/B512/T12/V50304,
Add9,468,053,696 /Attention17,521,117,376parameters. Never reduce logical B512
and claim original wide. CPU/mixed currently have one payload device;17B FP32
cannot fit one64GiB NPU. Generic mixed multi-device placement remains open.
Eager/mixed FP16 training explicitly rejects an unqualified master path.

Static observations,not full-size allocation measurements:
PackedFiberAttention preallocates batch×local_attention_nodes×kv_rows, with
retained/VJP banks multiplying storage. Coordinator DeviceJournal allocates
trace×(5*D+2) event payloads and additional D-wide journals under one trace
capacity. Conservative estimates also over-reserve,so rejection alone is not
proof that actual HBM cannot fit. Investigate compact live storage,separate
capacities,accurate simultaneous allocation accounting and safe physical
splitting/backpressure. Keep arbitrary legal topology/input,complete attention
normalization,no truncation,continuation and explicit real capacity errors.

## Environment and preserved history

Use public module libtorch-npu/2.10.0-cann9.0.0 and Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
The user-authorized /opt stack supersedes the dated personal guide. Default
shell Python cannot run Torch tests. TASK_QUEUE_ENABLE=0,
TORCH_DEVICE_BACKEND_AUTOLOAD=0; preserve module PYTHONPATH,prepend source/python.
SoC Ascend910_9392,16logical chips×64GiB; leased/remapped devices only.
freeze_run.py uses background.slice/Nice10; builds2workers; bounded queue/run.
Runtime cwd env -C RUN. Check disk before heavy writes; last data215GiB/root14GiB.
Atomic handoff writes use scripts.durable_records.replace_text.
Nonblocking TASK/online-measurement.lock is distinct from historical timing.lock.

historical-cpu-attention-01 remains deliberately SIGSTOP,pause.json outweighs
running status; it holds memory/TASK/timing.lock. Do not resume/kill/clean it.
Historical Add CPU78.793172/NPU4 47.932888ms/token is1.6438× faster throughput;
it does not qualify new online flows. Earlier24 dirty files are preserved on
pushed archive/restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37,
sha256 inventory TASK/restricted-flow-archive.json. No new approval is required.
