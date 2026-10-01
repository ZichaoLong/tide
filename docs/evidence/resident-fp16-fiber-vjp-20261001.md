# FP16 local same-fiber attention adjoints

Implementation: **f2ec4f161c86c0af6bfaf9ec2ff9755b1cb4d0ea**.
[Contract](../resident-fiber-vjp.md), [audit](resident-fp16-fiber-vjp-20261001.json).
All six fixed-clean-source jobs passed/exit0 on LibTorch-NPU2.10.0/CANN9.0.0,
Ascend910_9392. Selected physical devices map to logical `npu:0`.

The component preserves half QKV matmul/bias, query scaling before QK, completed
query output and final pooling rounding. Mean pooling sums before division.
Forward KV/log-bias is built independently on NPU; the CPU FP32/FP64 oracle
recomputes from public inputs. Cotangents, cache/source/parameter adjoints and
owner reductions stay FP32. No public half graph-training guard was removed.

| Gate | Result |
| --- | --- |
| Local fiber VJP, each dtype |37 configurations/74 replays: five pooling modes,seven root modes,width1/4/257,multihead,complete-fiber visibility |
| Dynamic input and refusal checks |same captured program with different source/cache lengths,None/zero,poisoned padding;bounded tick count and tensor budget |
| FP32 event/fiber training |66/172 root cases,8/20 optimizer trajectories,CPU FP32/FP64 |
| Python-owned client |61 passed,zero skips: precision,event-training,fiber-training |
| Separate half profile |15017 AI_VECTOR_CORE,1405 AI_CORE,170 MIX_AIV tasks;no observed AiCPU/logged CPU fallback |

Half roots are multiplied by256 to make small derivatives visible to the
absolute tolerance. Half uses rtol2e-3/atol2e-5; original FP32 remains
rtol2e-5/atol2e-6. Connection bits and padding checks are exact. Development
build/gate dev01, root-amplified checker dev02 and FP32 regression all passed.
No failed fiber development run was discarded or renamed.

Standalone qualification rebuilt the checker and reused source-byte-matched
terminal host/kernel archives. Python qualification rebuilt its local fiber
host object and relinked the client against matching CANN archives. Audit
checks source/core/archive/object/loader identities, case logs and profile CSVs;
this is not a from-scratch vendor build. The unchanged portable core's8,954
CPU checks were not rerun.

Artifacts: `TASK=/mi/data2T/zlong/tide-execution-flows`;
`build-low-precision-fiber-vjp-{clean01,python-clean01}` and
`low-precision-fiber-vjp-{components,regression,python,profile}-clean01`.
The four runtime jobs used physical9/1/3/9; the second use of9 followed its
release. Original status/log/result files remain under their unique run paths.

```bash
python "$TASK/launchers/precision_fiber_vjp_evidence.py" f2ec4f161c86c0af6bfaf9ec2ff9755b1cb4d0ea
```

This evidence covers a local fiber component and affected FP32 clients.
It does not certify half fiber-cache/graph/retained-window training, public
FP16 training/checkpoints, peers or throughput. No full-size speed ratio changed.
