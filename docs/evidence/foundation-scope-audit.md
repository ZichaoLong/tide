# Foundation acceptance scope audit, 2026-09-23

Audited clean source: `be0b08d15ec590201b18f5f181e6a91f8fa195b5` on graph-execution-foundation.
No uncommitted changes at entry; seven local commits ahead of origin. Startup
status/log/status.py and all durable records were inspected. No starting/running
record or live project workload was found. Previous packed transport unit was
inactive/dead, MainPID0, Result=success, ExecMainStatus0; its immutable evidence
remains [packed transport](packed-transport.md). No new wide run was started.

Stage 1 is an audit/freeze acceptance, not new implementation qualification.
[ROADMAP](../ROADMAP.md) maps all six classes and required units to code, directed
gates and finite existing evidence. [Capabilities](../execution-capabilities.md)
separates existing frontier fiber support from Streaming-only transport/scheduler
switches. `benchmarks/foundation-v1.json` freezes twelve medium/small logical
configurations and two large shape families; entry points are not yet claimed.

Read-only code inspection confirms no native SettleGraph constructor/encoder:
`settle.py::embed` builds the graph, `native.py` converts that graph, native
frontier executes it. Native checkpoint is already independently qualified and
is not the next implementation task. Only self_loop/chain independent native
schedules exist; diamond/ring/layered extension is required. Attention/SSM
prefill is real and already counted in block_prepare.cpp. RoPE/model composition
and process-boundary training resume remain concrete acceptance gaps.

Raw local records: `artifacts/foundation-audit-20260923/` contains resources.json,
portability-audit.json, cpu-doctor.json, each written with durable_records and
read back. CPU operator probe exited0 (FP32 matmul/ReLU/sum, actual=expected152).
Static audit exited0, zero errors and12 warnings: runtime adapter CUDA guards,
Torch dependency range and explicit device rejection tests/CLI. These are not
backend failures. No CUDA/NPU verification claim follows. Existing project
portability contract is retained, not replaced with a template.

Resource observation: 320 available CPUs,320 physical cores,8 NUMA nodes,
no finite inherited CPU quota; aggregate default160 CPUs. Effective memory
budget at observation 755.186 GiB, derived from half the minimum
of host total, available and cgroup limits. Refresh before large jobs. Disk had
~27GiB free. No cleanup was needed. Correctness uses2 build workers, ATen/BLAS1;
no hardcoded CPU set or fixed256GiB memory cap. Source files inspected for size:
core implementation files below500 lines; runtime adapter359 is the largest.

Prior evidence is reused only for its recorded source/profiles. Latest full
regression6897 passes does not make every newly required acceptance cell passed.
S2 starts with independent native SettleGraph structure/model encoding and VJPs;
S6 alone closes the full acceptance version.
