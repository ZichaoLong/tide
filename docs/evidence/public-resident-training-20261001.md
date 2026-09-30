# Public resident C++ training qualification

Exact implementation **591e90756311cc776026de08c20136f99e1411df**, standalone
LibTorch2.10/CANN9.0.0 on aarch64 Ascend910_9392. The accompanying
[JSON audit](public-resident-training-20261001.json) records source, core,
binary, loader, logs and profiling CSV hashes. All five recorded jobs passed:
clean build, all42 device component cells, separate profile, installed consumer
build and installed consumer execution. The build also passed four CTests.

The public owner passed18 independent CPU FP32/FP64 trajectories,288 windows
and72 actual optimizer updates. Coverage includes both schedules,SGD/AdamW,
retained roots and continued state,large int64 coordinates,width257,parameter
aliases,empty trainable registries,None/zero gradients,checkpoint restoration,
capacity refusals and failure lifetime. The consumer uses installed public
headers from a prefix containing spaces and passes three inference windows,
three training windows,retained backward and optimizer restoration.

The actual online forward,reverse,optimizer and parameter-publication chain
uses the candidate's own device state and tapes. No CPU reference trajectory
is passed to the candidate. The caller's original model remains a construction
template; checkpoints export the owner's actual updated weights and slots.

The checker profile recorded107,345 AI_VECTOR_CORE,1,857 AI_CORE and1,673 MIX_AIV
tasks,including304 optimizer records. No AiCPU tasks or host-fallback diagnostic
was observed. Construction,exports and assertions are included in this profile;
task-time sums are not wall time or a training-throughput measurement.

This qualifies the [public training contract](../resident-training.md) for
single-NPU FP32 HARD,sum/broadcast,identity/EMA/Add state and identity/tanh Full.
Python/disk integration,other adjoints,FP16,peer progression and performance
are separate gates. This subset does not complete ROADMAP F4–F7. Earlier
development build failures remain in their original durable records.
