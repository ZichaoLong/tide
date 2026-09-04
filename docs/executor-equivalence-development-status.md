# 执行器等价性开发验证状态

> 本文记录一次可复现的开发验证结果和后续交接事项。它不定义或修改 SettleGraph 语义，也不把 development evidence 提升为资格验证。计算含义仍以[实验语义、命名与数学符号](experiment-semantics-and-naming.md)为准；资格门槛仍以[core-v1 资格计划](core-v1-qualification-plan.md)为准。

## 1. 当前结论

固定 \(K\) 的 `core-v1` 子集已经有一条通用 packed executor，以及 `single-layer.v1` 和 `hb-line.v1` 两条拓扑特化 executor。对预先固定的 development candidate corpus，修复后的 clean exact commit 已完成受控 run：60/60 tests 通过，execution receipt 闭合，运行前后的 commit 和源码指纹一致。在记录的 output、最终 state、balance、route、trace、live autograd 元数据和 VJP 目标上，没有发现 eager、packed 和适用的拓扑特化路径之间的差异。

这是一项强度较高的开发证据，结论仅限于当前支持子集、当前 CPU FP64/FP32 调用域和记录的测试场景；它不是 `C00`--`C12` qualification，也不是性能资格。

本次受控验证绑定的实现提交为：

```text
5712e66f1cf51a85360e6507839c2fe443aa81ae
fix: harden executor equivalence semantics
```

最初的 packed/特化实现提交是 `107a05218e7ba95ec80fbcf78f788cff83775d17`。修复提交没有修改主语义文档 `docs/experiment-semantics-and-naming.md`。实现边界和资格边界的说明位于[实现与等价性验证计划](settlegraph-implementation-plan.md)和[core-v1 资格计划](core-v1-qualification-plan.md)。

## 2. 已完成的可执行验证

受控 runner `scripts/run_executor_equivalence.py` 已在上述 clean exact commit 上完成。它在运行前后检查 commit、工作区指纹和源码指纹；任何 dirty source、运行中源码变化、缺失测试、skip、expected failure 或覆盖回执不闭合都会使运行失败。

本次成功记录位于工作区的 `runs/20260904T070450Z-executor-equivalence-5712e66/`：

- [运行 manifest](../runs/20260904T070450Z-executor-equivalence-5712e66/run.json)：`completed`，`exact_commit=true`，运行前后工作树均 clean，源码指纹未改变；
- [终态摘要](../runs/20260904T070450Z-executor-equivalence-5712e66/summary.json)：60/60 tests 通过、退出码为零、`qualification=false`；
- [实际执行回执](../runs/20260904T070450Z-executor-equivalence-5712e66/artifacts/execution-receipt.json)：9298 个执行事件，1680 个 forward cells、176 个 VJP case/dtype groups 和 4 个 lifecycle scenarios 与期望集合精确闭合；
- [原始测试日志](../runs/20260904T070450Z-executor-equivalence-5712e66/stdout.log)。

初始实现提交 `107a052` 的 45/45 历史 run 仍保留在 `runs/20260903T143738Z-executor-equivalence-107a052/`，但当前实现的结论以上述 60-test 修复后 run 为准。

candidate corpus identity 为：

```text
8497fccea52a958373ae5963c433a0f8420874005c88639ebd9e35d51fec6111
```

packed、single-layer 和 HB 的静态支持分区 identity 为：

```text
49fb9c797f40546f29534bbaf1fac5c4b04669b4990239d4af22d3154fa4c703
```

### 2.1 通用 packed executor

通用路径 `tide.generic-packed.torch.v1` 直接使用同一个 `SettleGraph` 参数 owner，不复制参数，也不调用 eager scheduler。它按 region 对完整 \([B,T]\) 段处理固定 \(K\) 的 `core-v1` DAG，并支持 N、SD、BO，content/pre/post，EMA、Gated DeltaNet、窗口 Attention、forced-active singleton、state、route、trace、full prefill 和 decode。

对 256 个预先固定的 legal candidates，在 FP64 和 FP32 下，逐一比较 token-major eager 与 packed 的：

- full prefill；
- `T=3` 的两个非空 two-chunk splits；
- 逐 Token decode；
- output、最终 state、balance sufficient statistics、完整 canonical trace、route 和 trace invariant。

每个 chunk 调用和每步 decode 的 trace 都先直接与 eager 对照，随后规范合并，并再次与 full-prefill trace 对照。实际 packed forward 覆盖为 `256 × 2 × 3 = 1536` case/dtype/mode cells。

64 个预标记 VJP candidates 在两种 dtype 下从同一次保留计算图依次隔离查询 output、可微 balance loss、每个最终 state owner/component、每个 region 的 `soft_sum`、每个可微 trace region event 的 logits/probabilities、组合目标以及末尾重复 output。每项均比较 hidden 与所有具名参数的数值 VJP，以及 `None`/Tensor 连通性。实际 packed VJP 覆盖为 128 个 case/dtype groups 和 7390 个 objective queries；single-layer 与 HB 分别另有 16 和 32 个，因此三条路径合计 7438 个。

此外，FP64 定向回归覆盖了 EMA、Gated DeltaNet、窗口 Attention 的可微外部初态，Attention 的 keys/values 分开检查，以及跨 chunks 保留计算图的 public-result 和 trace-state objectives。一个多候选反例逐个检查 node-event logit root，曾发现 eager `stack` 的 connected-zero 与 packed 的 `None` 结构不一致；该差异已修复并作为回归保留。

后续审查又发现多类公开 autograd 元数据或连通性差异。空调用前状态下，receiver 首次 Observe 以前的 `NodeEventTrace.state_before` 在 eager 中是不可微首状态，packed 却因整段 causal scan 的 Tensor 拼接而带有 `requires_grad=true`。分组计算结果写入同一 dense Tensor 后，identity/hard 的 node compute、Emit、edge payload、parent message 或 terminal message 也会继承其他 lane 的可微性；同类污染还能传播到下游 Aggregate，在 \(K=1\) 时使主 output 错误带图，或让混合 frozen/trainable formula group 中实际 active 的冻结 lane 继承 inactive 可训练 lane 的图。single-layer 特化的整调用 output-score 堆叠则会让单个 output event 对其他事件使用的 score 参数产生 connected-zero，而 eager 中为 `None`。此外，packed 的 no-detach 状态发布曾可能切断本次 batch 未出现的 dormant sequence 状态图。它们的数值可以完全相同，原 runner 查询的主训练 VJP 也不一定触及这些根，因此必须按公开 occurrence 和调用边界分别检查。

`5712e66` 实现在首次 Observe 前直接复用调用入口 state；入口没有该 owner 时，按语义重建零 Tensor 或空 Attention 首状态。它在 `record_trace=True` 时保留未混入 dense storage 的每个 NodeCompute/Emit occurrence，并用这些 occurrence 还原 edge、parent、terminal 和 output trace；公开结果边界再按单个 observable 的真实 source liveness 去除 dense lane 带来的假阳性图，而不切断真实来源。未实际 Observe 的 owner 和本次 batch 外的 dormant sequence 在 no-detach 分支原样 carry 入口 state。single-layer output event 也按该事件实际 terminal messages 单独重建，不共享整调用的 score 图。新增回归递归比较整个 live `ExecutionResult` 的全部 Tensor shape 和 `requires_grad`，并用 isolated VJP 检查事件局部 score、跨 chunk state-before/state-for-compute roots 和 batch 外状态 leaves。`record_trace=False` 的训练/性能路径不保留诊断 occurrence，但主 output、发布状态和 source-liveness 边界仍受相同公开语义约束。

source-liveness resolver 必须存活到反向实际选定公开 objective，但不应与内部 autograd graph 形成永久强引用环。调用含有语义可微的公开结果时，当前由公开 result boundary 唯一持续强持有该次调用的 connectivity tracker，内部 selective input/stack autograd context 只保存弱引用；结果仍存活，或派生 loss 等 Tensor 仍引用其图时，普通 backward 和 `retain_graph=True` 的重复 isolated VJP 均可取到同一 tracker。公开结果及所有仍引用该图的派生 Tensor 都释放后，tracker、resolver 和 region runtimes 可以一并回收；没有语义可微公开结果的调用不需要建立这一结果边界。定向回归还要求 tracker 若异常提前释放必须显式失败，不能静默退回 dense 连通性。

selective parameter boundary 还需要把参与 stack 的非叶 Tensor，例如可学习 EMA decay 的 sigmoid，精确追溯到它们的叶参数。当前遍历在访问期间强持有 autograd node 的 Python wrapper；若只记录 wrapper 的瞬时整数 id，CPython 可在 wrapper 释放后复用该 id，从而静默跳过另一个上游分支。定向回归覆盖 direct leaf、sigmoid、reduction、16 叶分支表达式和 shared subgraph 的精确叶集。

prefill 与 decode 省略参数时的默认 autograd detach，以及 `detach_at_end=False` 的跨边界 VJP，均有定向验证：两种配置保持相同 forward state carry 和数值结果，前者对前一 chunk 输入返回 `None`，后者保留有限非零梯度。eager、packed 和 HB 特化路径还直接比较该 no-detach VJP 的三方数值。Python 入口只接受精确的 `bool`，不会把 `None`、整数或 Tensor 按 truthiness 当成配置。这些是两种显式可比的训练配置，不是强制所有训练都在 chunk 边界断图。当前入口也在公式执行前拒绝空 batch 和 FP16/BF16；这表示调用域明确，而不是低精度或空 batch 已获资格。

### 2.2 特化拓扑 executor

对静态支持集合，当前测试执行 token-major eager、通用 packed、特化 executor 三方的 FP64/FP32 full prefill、`T=3` 的两个非空 two-chunk splits、逐 Token decode，以及另行 full-prefill VJP：

| 特化路径 | 已验证的静态适用 candidates | 实际 forward cells | 实际 VJP case/dtype groups | 证据含义 |
| --- | ---: | ---: | ---: | --- |
| `single-layer.v1` | 8 | 48 | 16 | 无状态 N/content 单层拓扑的独立平铺 Tensor 调度 |
| `hb-line.v1` | 16 | 96 | 32 | fully expanded HB Plan 的独立 Line/barrier 调度 |

两条路径都复用同一个参数 owner，构造前静态拒绝不适用 Plan，且不回退到通用 eager scheduler。全部 24 个支持 case 在 FP64 下另递归比较 live 公开 Tensor 的 `requires_grad`；一个同时含 EMA、GDN 和 Attention 的 HB case 把 17 个调用方初始状态 leaves 纳入完整结果与 VJP 三方比较。对有状态代表场景，另行检查 reset、row reorder、empty tail，以及 mask/位置/状态所有权负例和晚期 empty-terminal 失败回滚。

HB 的结论需要准确限定：它是独立的拓扑调度和 barrier oracle，不是独立的局部公式 oracle，也不是高性能 HB kernel。因而三方一致能增强对拓扑顺序、barrier、state commit 和 trace 的证据，但不能单独排除三条路径共同错误地实现同一个局部公式。

### 2.3 当前 run 的覆盖闭合

runner 不从静态 support 数量推断通过覆盖。测试在完整比较、finite 检查和 VJP 连通性检查均成功后才写 execution receipt；随后用期望 key set 校验无缺失、无重复、无额外 cell。

| 项目 | 实际数量 |
| --- | ---: |
| 测试（`5712e66` 当前 run） | 60/60 通过 |
| forward cells | 1680 |
| 其中 packed / single-layer / HB | 1536 / 48 / 96 |
| VJP case/dtype groups | 176 |
| 其中 packed / single-layer / HB | 128 / 16 / 32 |
| VJP objective queries | 7438 |
| lifecycle scenarios | 4 |

当前 required discovery 包含 candidate corpus 4 个、core executor equivalence 14 个、packed 25 个和 specialized 17 个测试。60 个 test ID 唯一，runner 钉住的 29 个 required semantic IDs 全部存在。2026-09-04 的受控运行耗时 5353.196 秒，无 failure、error、skip、expected failure 或 unexpected success。

受控 runner 的终态摘要记录 execution receipt 校验为 `passed`，receipt SHA-256 为 `a5131bf1eadfb5f1b0020d4d0cf8ac087c098826f81679452dfd1c2c5c7125dc`。其 `qualification=false` 是终态事实，不是待后续解释的注记。上述计数也不表示每一类轴都有同等强度的 VJP 覆盖：64 个 marked case 的现有选择与 `d_model` 轴相关，实际全部为 `d_model=2`，且只含 affine/SwiGLU、Hard-ST/soft-probability、linear/MLP Score 和 mean output Aggregate。更广的公式、shape 与 topology 覆盖主要来自 256-case forward，而不是一个正交的 isolated-root VJP 设计。

### 2.4 探索性 CPU 性能记录

非资格 benchmark `runs/20260904T013946Z-packed-cpu-prefill-decode-exploratory-107a052/` 在 aarch64 CPU、FP32、单线程和 `record_trace=false` 下完成了 4 个有状态拓扑、2 个上下文长度和 prefill/decode 两个阶段的 16 个 workload。16/16 的 output、state 和 balance correctness gate 均通过。相对 token-major eager：

- packed full prefill 的 workload 中位/最低加速为 2.0125×/1.3640×；
- packed 单 Token stateful decode 的 workload 中位/最低加速为 0.3489×/0.3154×，即当前实现明显更慢。

这不是 `X07`：workload、进程数、warmup/sample 数、峰值内存和 profiler 均不符合资格计划。运行期间相关源码 fingerprint 没有改变，但工作区出现 README/状态文档改动，最终 `exact_commit=false`；而且它执行的是本次 trace 元数据和未 Observe state carry 修复之前的 `107a052`。trace occurrence 的保留与还原只在 `record_trace=True` 时启用，但该旧记录仍不能作为当前实现的正式性能证据。

后续在本机 aarch64、Torch `2.10.0+cpu` 环境中，针对 `5712e66` 实现做过一次未持久化的非资格 RSS 生命周期烟测。输入是 `corpus.011.mixed-regions.base-r2` 的 CPU FP64 fixture，形状为 \([B,T,d]=[2,3,3]\)；每轮在 grad-enabled、`record_trace=False` 下执行 prefill，删除结果并运行 Python GC，每十轮再调用一次 `malloc_trim`，随后从 `/proc/self/status` 读取 `VmRSS`。修复前，RSS 在 100 次调用中从 241.0 MiB 持续升至 458.6 MiB，100/100 个 tracker 弱引用仍存活；`torch.no_grad()` 对照预热后稳定。定位到的强环是 tracker 到 resolver 捕获的 runtimes，再经 Tensor autograd context 回到 tracker。改为上述单一强 owner 后，`5712e66` 的 60 次同类调用在预热增长后稳定于约 282.9 MiB，每次删除结果并 GC 后的采样点均没有存活 tracker。该烟测只表示在该进程与该 case 中未再观察到已定位的强环，不是峰值内存、吞吐或 `X07` 资格证据。

source-liveness 的当前正确性实现本身仍是性能风险。在 grad-enabled 前向中，公开 Tensor occurrence 为确定精确 `requires_grad` 元数据，会各自触发一次基于已发现 route 的完整反向语义遍历；该遍历还含 device 标量读取和 CPU list 转换。一个 `B=2,T=3` 的有状态代表 case 在 `record_trace=False` 下就会为 output、公开 state components 和 balance regions 重复触发多次遍历。现有 `python_*_hot_loops` profile 字段没有统计这项成本，因此其零值不能证明该路径没有 Python 控制流或 host synchronization。正式宣称 packed 高性能前，必须把这项前向成本移出热路径或合并，并用真实 profiler、同步计时和峰值内存证据验证。

## 3. 这项结果证明什么，以及不证明什么

### 可以据此陈述

- 在 clean exact commit `5712e66` 上，对 256 个固定 \(K\) `core-v1` development candidates，通用 packed 的两种 dtype full prefill、\(T=3\) 的两种非空 two-chunk splits 和逐 Token decode，与 token-major eager 在记录的可观测量上等价；
- 64 个标记 case 的两 dtype isolated VJP 在值、输入/参数梯度和 `None` 连通性上通过；
- 对静态适用的 8 个 single-layer 和 16 个 HB candidates，eager—packed—specialized 三方 forward、VJP 和 live autograd 元数据一致；定向回归另覆盖 HB 外部可微初态、默认 detach 和显式 no-detach；
- 当前定向回归关闭了已知的首次 Observe 前 state、跨 lane/Token 的 node/edge/parent/terminal provenance、下游 Aggregate 与 \(K=1\) output、混合 frozen/trainable formula lane、single-layer event-local output score，以及 batch 外 dormant sequence no-detach state 图反例；
- selective parameter 的非叶 Tensor 叶源追溯已关闭 autograd node wrapper id 复用造成的漏分支反例；
- 当前生命周期回归和 RSS smoke 没有再观察到已定位的 connectivity tracker 强引用环。

### 不能据此陈述

- `C00`--`C12` 的任何一个完整 qualification gate 已通过；
- `X07` CPU performance qualification 已通过，或 packed decode 已被证明加速；
- packed decode 是独立算法、独立 kernel 或独立 oracle；当前 `decode` 只是同一 packed prefill 路径的 \(T=1\) wrapper；
- HB 是优化 kernel 或独立局部公式实现；
- `python_*_hot_loops=0` 的 profile 字段已经通过 callback/profiler 实测；当前非 trace 路径仍有位置校验、StateStore 打包/发布和下一位置更新中的 Python 循环及标量同步；
- 当前 source-liveness 边界已经满足高性能资格；grad-enabled 前向仍可能按公开 Tensor occurrence 重复执行完整语义遍历与 host synchronization，且现有 profile counters 不覆盖该成本；
- 所有公开 trace Tensor 都已经做了全语料 isolated-root VJP；正式隔离目标主要覆盖 output、balance、最终 state、region `soft_sum` 和 region logits/probabilities，其余 trace roots 只有定向检查；
- FP16/BF16、CUDA、当前代码的 NPU、x86_64、Qwen/Base、KV cache、LM loss、训练、完整 checkpoint、selector-history、可学习首状态、多 site 或 input-K packed 已验证；
- 所有合法 Plan、所有 chunk split、所有并发 backward 或所有未来局部公式都已得到等价性证明。

“packed 高性能”在当前阶段只描述实现路线：它不调用 eager scheduler，主要公式使用 Tensor 批处理或编译递推。它已经具备强开发级语义等价性证据、一项探索性 prefill 加速观察和一次定向 RSS 生命周期烟测，但还没有 callback/profile、峰值内存和冻结 workload 所要求的性能资格证据；现有探索记录未观察到 decode 性能收益，当前提交尚无 decode 加速证据。

## 4. 后续工作的建议顺序

后续应先把开发证据转化为资格证据，再展开性能、设备与训练实验：

1. 冻结并物化 qualification corpus/bundles，补齐独立 golden、canonicalization、负例和 failure envelope 记录；
2. 完成 `C00`--`C06`，重点是冻结的 objective/cotangent records、32 个 finite-difference、Hard-ST 局部 oracle，以及全部要求的 chunk 场景；
3. 完成 `C07`--`C12`：optimizer、所有 chunk split、lifecycle/concurrency/rollback、negative 和 fresh-process checkpoint；
4. 正确性 gates 完成后，按资格计划冻结的 workload、warmup/sample/process 规则运行 `X07`，记录长序列 prefill 与 stateful decode 的延迟、吞吐、峰值内存和 profile；
5. 再分别完成当前 commit 的 NPU/CUDA packed 证据、fallback closure、低精度能力，以及 Qwen/Base 和正式训练实验。

当前开发 runner 不能通过增加更多随机 candidates 自动升级为 qualification。升级需要资格计划规定的冻结 corpus identity、独立工件、完整 required cells 和可追溯 terminal evidence。
