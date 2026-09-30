# Python resident training and disk-resume qualification

Exact implementation **0f363b89253cf887ecc62c5d6d08835a5823fefd**, Python-owned
TorchNPU2.10/CANN9.0.0 on aarch64 Ascend910_9392. The clean backend build,
CPU interface gate,NPU gate and separate profile all terminated with exit0.
[The JSON audit](python-resident-training-20261001.json) records exact source,
core,binary,loader,logs and raw profile CSV identities. The plugin does not
link the standalone LibTorchNPU SDK into the Python runtime.

The CPU interface gate passed76 tests with37 explicit optional-NPU skips.
The required device job passed all40 tests with no skips: the25 existing resident
inference cases and15 training/configuration cases. Training covers all three
graph families,both schedules,SGD/AdamW,real NPU loss/autograd cotangents,independent
CPU parameter/input VJPs,None/zero,shared owners,disk restoration and a fresh
process continuing optimizer updates. Saved parameters are actual device-updated
values rather than the unchanged construction template.

The independent PDG greedy AdamW checker profile observed1,569 AI_VECTOR_CORE,
14 AI_CORE and42 MIX_AIV tasks,including12 optimizer,66 graph-reverse,6 window-
bridge and12 ready-pack records. No AiCPU task or host-fallback diagnostic was
observed. This includes construction,CPU oracle and NPU loss cotangents; it is
not a throughput measurement,and task-time sums are not wall time.

This qualifies the [Python client and disk contract](../resident-training.md)
for the existing single-NPU FP32 HARD,sum/broadcast,identity/EMA/Add state and
identity/tanh Full profile. It is a Python client of C++/CANN,not an independent
pure-PyTorch resident scheduler. Other adjoints,FP16,multiple devices and the
representative/full-size performance matrix remain separate required work.
