# 从 Base checkpoint 到 SettleGraph + BO 实验及后续扩展路线

> 本文是当前实现完成 executor 等价性开发验证后的实验与工程推进计划。它不定义新的模型语义，也不修改既有语义。SettleGraph、BO、SD、placement、loss 和实验条件的计算含义以[实验语义、命名与数学符号](experiment-semantics-and-naming.md)为准；当前软件能力以[实现与等价性验证计划](settlegraph-implementation-plan.md)和[执行器等价性开发验证状态](executor-equivalence-development-status.md)为准；正式资格门槛以[core-v1 资格计划](core-v1-qualification-plan.md)为准。
>
> 本文的用途是让后续工作从一个明确基准继续：先用 checkpoint 实验判断 SettleGraph，尤其是 BO，是否具有可复现的训练或推理价值；只为实际实验路径做必要的特化加速；确认正面信号后，再建设面向超大、超稀疏、多卡 DAG 的通用高性能执行器。

## 1. 已确定的优先级

后续工作分成三层，按以下顺序投入：

1. **实验有效性主线**：从真实 Base checkpoint 出发，接通推理、训练、恢复和评测，优先检验 SettleGraph + BO 能否取得可复现的正面结果。
2. **实验所需的特化加速**：如果 profiler 证明当前执行路径阻塞第 1 层实验，则优化该实验实际使用的 Plan、状态和 placement；每个优化版本继续以主语义和 token-major eager reference 为实现内语义基准，并与现有 packed 路径差分。packed 是已有强开发等价证据的对照路径，不是独立的局部公式 oracle。
3. **通用超大 DAG 性能工程**：只有在某类 SettleGraph/BO 配置已表现出稳定价值、并积累真实路由与通信记录后，才建设节点跨卡放置、设备端调度和大规模 packed 执行。

这一区分避免把“模型是否值得训练”和“任意大图是否能极致加速”混成一个问题。第 1 层允许使用尚未达到最终性能资格、但已经通过所用调用域正确性验证的实现。在完整资格门之前，这些运行必须标为 engineering bring-up 或 development pilot，不能冒用正式 capability/scientific cell 身份；任何公开的模型效果或性能结论仍需对应的冻结证据。

“希望获得 BO 的正面结果”是研究方向，不是允许事后改变判据的通过条件。每轮正式比较必须在看到结果前冻结主要指标、对照、预算、seed、停止规则和允许的调参范围；未达到判据的结果同样保存。

## 2. 术语与首轮研究假设

### 2.1 本文使用的几个概念

- **Base checkpoint**：一个已训练 Base LLM 的不可变权重、配置和 tokenizer 身份。`init-from` 表示从它建立新的实验轨迹，不恢复旧实验的 optimizer、数据位置或随机数轨迹。
- **实验垂直切片**：一个从 checkpoint 装载开始，贯通 Base block 接入、SettleGraph 执行、logits、loss、backward、optimizer、checkpoint、fresh-process resume、prefill 和 decode 的最小完整配置。
- **BO**：对每个局部 region，所有 reached receivers 都 Observe 并提交状态，只有 active receivers 执行完整 NodeCompute 并发送消息。
- **SD**：只有 active receivers Observe、提交状态、执行完整计算并发送。profile 是规范 Plan 字段，因此 BO 与 SD 使用 hash 不同的两份 Plan；直接比较应保持除 profile 外的拓扑与公式字段、状态容量、active budget、NodeCompute、Emit、Aggregate、训练 Token 和 optimizer 匹配。
- **函数等价起点**：SettleGraph 初始化后满足其输出等于输入，即残差为零，使接入后的模型在声明调用域内与原 Base 模型具有相同 forward 和 LM loss。
- **正面实验结果**：在预先声明的主要指标和预算下，相对主要对照出现多 seed 可复现的改善，同时没有被稳定性、路由坍缩或系统成本否定。它不是单次 run 的最低 loss，也不是从许多事后切片中挑出的最好数字。
- \(R\) 与 \(K\)：\(R\) 是一个 region 中的固定 receiver 数，\(K\) 是该 region 每次 settlement 的固定 active 数上限。

### 2.2 首轮需要检验而不是预设成立的假设

1. Base + SettleGraph 能从函数等价的 checkpoint 起点稳定离开中性初始化，而不破坏 Base 的已有能力。
2. 在固定相同 \(K\)、匹配 active NodeCompute 预算且各条件使用自然路由的主比较中，BO 让未 active 但 reached 的 receiver 保留内容历史，因此能比 SD 提供可复现价值；固定 route replay 只用于隔离 Observe 机制的诊断。
3. BO 的状态被后续 selector 或 NodeCompute 因果地使用，而不只是数值发生变化。
4. SettleGraph 带来的质量或适应收益足以覆盖 selector、状态、packing 和通信成本。
5. 当总节点数增长而每 Token 只激活少数节点时，可达容量能够增长，且信用分配、节点饥饿和路由集中仍可控制。

前四项构成当前主线。第五项在小规模信号成立后才进入大规模验证。

## 3. 可作为参考基准的当前实现

### 3.1 固定基准

当前可执行实现基准是：

```text
5712e66f1cf51a85360e6507839c2fe443aa81ae
fix: harden executor equivalence semantics
```

状态记录提交是：

```text
0977001031832cd85c31e5f2a5391e52456ecdd0
docs: record controlled executor equivalence run
```

受控 development run 在 `5712e66` 的 clean exact commit 上完成，60/60 tests 通过；它比较了固定 \(K\) `core-v1` 的 eager、通用 packed 以及适用的单层/HB 特化路径。覆盖包括 CPU FP64/FP32 的 full prefill、两种非空 `T=3` chunk 切分、逐 Token decode、output、state、balance、route、完整 trace、公开 Tensor 的 `requires_grad`，以及记录目标上的 VJP 数值和 `None` 连通性。详细边界见[执行器等价性开发验证状态](executor-equivalence-development-status.md)。

主语义文档始终是计算含义的权威来源。后续实现不得用更快路径替换或删除 token-major eager reference；可获得独立 golden 时还必须一并保留。若新的 Base 接入、设备 kernel、特化 executor 或分布式调度改变了可观测结果，应先按 correctness 问题处理，不能用性能收益解释差异。packed 与 eager 一致是强开发证据，但不能单独排除二者共享的局部公式错误。

### 3.2 已有能力与使用边界

| 范围 | 当前可以复用的内容 | 当前不能由此推断的内容 |
| --- | --- | --- |
| 通用执行 | `tide.generic-packed.torch.v1` 实现当前固定 \(K\) `core-v1` 范围的 N、SD、BO、content/pre/post、EMA、Gated DeltaNet、窗口 Attention、prefill、chunk 和 SettleGraph decode，并已有所述 development coverage | 这不是完整 `core-v1` 资格；decode 只是 SettleGraph 的 \(T=1\) prefill 调用，不是 Base/Qwen KV-cache decode。尚无低精度、目标 NPU 或正式性能资格；不支持 selector-history 和模型内 adaptive K |
| 拓扑特化 | `single-layer.v1` 与 `hb-line.v1` 已与 eager、packed 做开发级三方等价验证 | `single-layer.v1` 只接受无状态 N/content；不能直接承担 stateful BO 首轮实验。`hb-line.v1` 只对其静态支持谓词接受的 fully expanded HB Plan 提供 BO 的独立拓扑调度 oracle，不代表任意 HB/BO Plan，也不是高性能 HB kernel |
| 跨调用梯度 | 省略参数时默认在调用结束 detach；显式 `detach_at_end=False` 保留跨 chunk/decode 的状态图，两种模式均已有定向验证 | checkpoint 不能保存一个仍需继续反向的现场 autograd 图；训练必须声明截断边界或保存可重放前缀 |
| placement | POST、PARBLK、PARATTN、PARMLP 的 Tensor 方程和 identity 退化已有单元测试 | 尚未接入真实 Qwen block，也未覆盖真实 causal mask、position IDs、KV cache、logits、LM loss 和 Base 参数梯度 |
| checkpoint | SettleGraph 参数、Plan/binding、receiver state、Adam/AdamW、CPU RNG 和基础 continuation 已有版本化 CPU checkpoint v1 | 还不是完整 Base 训练 checkpoint；scheduler、AMP scaler、backend RNG、sampler/data cursor、累积中梯度和未归约统计窗口尚未闭合 |
| 后端 | 有 CPU/CUDA/NPU runtime 解析边界；旧 eager-reference 内容在本机 Ascend 上有一次 FP32 定向 attempt | 当前 packed/特化提交尚未在本机 NPU 重新形成 clean exact-commit 证据；CUDA 仍未验证，FP16/BF16 不在当前 executor 调用域 |
| 分布式 | 语义允许物理实现改变布局和节点放置，只要恢复相同可观测值 | 仓库当前没有 DDP/FSDP、HCCL DAG 调度、跨卡 state ownership 或节点并行实现 |

因此，当前代码可以作为下一阶段的 **SettleGraph 局部实现参考与执行器差分基准**，但不能直接作为真实 Base checkpoint 的端到端训练程序。

### 3.3 端到端实验尚缺的软件边界

当前源码中没有真实 Qwen/Transformers model adapter 或训练 runner，项目包也没有声明 Base 模型与训练依赖。第一个垂直切片需要新增并验证：

- 经结构审计、符合当前 dense decoder block 方程的 Qwen-family 具体型号及其 block 内 placement 接线，以及 causal mask、position IDs/RoPE 和 Base residual 边界；不能从家族名推断 Qwen3 MoE 或任意 remote-code variant 已受支持；
- KV cache 的 prefill/decode、batch reorder、sequence release/reset 和 SettleGraph state 生命周期；
- LM head、next-token shift、LM target mask、LM loss、balance loss 和 Base/SettleGraph 参数梯度；
- 确定性数据管线、评测、LoRA target modules、稳定参数键、rank/alpha/dropout、optimizer 分组、checkpoint 规则和训练命令行边界；
- Base + SettleGraph + 可选 LoRA 的联合 checkpoint 与完整 resume；
- 当前提交的 NPU packed 验证、fallback closure，以及后续才需要的 distributed execution。

这些是当前实现缺口，不是新的模型语义。适配器应把 Base 和 SettleGraph 已有公式原样接起来，而不是在集成代码中隐式重新定义它们。

## 4. 第一个必要里程碑：checkpoint 端到端垂直切片

### 4.1 开始编码前先冻结的输入

新会话首先只读盘点本机资产和环境，并为首个切片写出一份短配置记录。至少固定：

- Base 模型家族与精确 architecture class、不可变 model revision、模型实现/library revision、config、权重文件及其 hash，并锁定 attention backend 与 remote-code 状态；
- tokenizer revision、特殊 Token 配置及其 hash；
- 用于 plumbing/overfit 的确定性小数据，以及用于首轮 pilot 的训练和验证数据 identity；
- 插入的 block/site、placement、完整规范化 Plan 及 Plan hash；
- Base、SettleGraph、LoRA 中分别可训练和冻结的稳定参数键；LoRA 另固定 target modules、rank、alpha、dropout、optimizer 分组和 checkpoint 规则；
- dtype、设备、executor、编译选项、seed、确定性设置；
- full prefill、chunk、decode 的序列 ID、position、mask、state carry 和 detach 规则；
- 主要指标、对照、训练 Token/step 预算、停止条件和成功阈值。

若本机存在多个 checkpoint 或数据集，不应仅因文件最近或模型最大就静默选择。先选择能最快完成完整闭环的最小代表模型；模型选择会改变研究结论时，再由用户确认。

### 4.2 推荐的首个切片

首个切片以减少同时变化的因素为目标：

- **Base**：本机已有、经结构审计且受当前 dense decoder 语义覆盖的最小代表性 Qwen-family checkpoint；先完成 plumbing，再扩到更大 checkpoint。型号名本身不构成支持证据，MoE 或需要未审计 remote code 的变体不默认进入首轮切片。
- **site**：先只插入一个 site，并明确其 block index。多个 sites 在单 site 的参数、状态和 cache ownership 闭合后再增加。
- **placement**：首轮科学条件默认优先 PARMLP，因为原 dense MLP 保持 always-on，SettleGraph 作为并行稀疏残差加入。若适配器开发先用 POST 验证 block 边界，这只是 integration smoke，不自动替代 PARMLP 的科学比较。
- **Plan**：先用一个单层、单 region、stateful 的小图隔离 BO。建议从 \(R=8,K=1\) 或同等规模开始，状态先用公式最简单的 EMA；首轮信号稳定后再扩到 \(R=32,K=1\)、多层一般 DAG 或 HB-Lattice。
- **profile/timing**：首先比较 BO/pre 与 SD/pre；两份 Plan 除 profile 外匹配，并能保持相同 selector 输入定义。当前仓库尚无 active-set 注入接口；若后续实现并验证仅供诊断的 route-replay 能力，可在诊断 batch 上 replay 同一 active route。随后再比较 BO/post 与 BO/pre/content，单独判断 proposal 参与选择的价值。
- **NodeCompute/Emit/Aggregate**：选用当前 eager/packed 已有 development differential coverage 的组合；首个可训练 selector 路径优先使用已有开发差分覆盖的 Hard-ST 配置，但独立 Hard-ST oracle 仍按资格计划补齐。图输出先使用 mean Aggregate。任何替换一次只改变一个实验坐标。
- **executor**：CPU 小样本以 token-major eager reference 和独立 golden 为基准，设备训练优先尝试通用 packed。当前 `single-layer.v1` 不接受 stateful BO；只有 profiler 证明通用 packed 阻塞实验时，才新增该 BO Plan 的特化版本。
- **dtype**：CPU FP64 用于小 fixture/梯度诊断，CPU 和 NPU FP32 用于首个端到端闭环。BF16/FP16 在 accumulation role、数值门槛和真实设备差分完成前不进入首轮科学结论。

\(R=8,K=1\) 是行为诊断点，不是最终 1/32 稀疏目标。

### 4.3 初始化与梯度边界

首个 checkpoint 起点必须分别验证：

1. Base-only replay 能加载原 checkpoint，并对照 pinned upstream/reference 实现或由其冻结的独立 golden，复现固定输入的 logits、LM loss、mask、position IDs、next-token shift 和 KV-cache continuation；只让同一 adapter 的 prefill、chunk 和 decode 彼此自洽不够。若切片将训练 Base 或 LoRA，还要冻结共同逻辑参数键、标量目标和 tolerance，对照选定 Base/LoRA 参数梯度及一个 optimizer step；
2. 插入中性初始化 SettleGraph 后，图基值 \(b_{\mathcal G}=h^{\mathrm{in}}\)、图残差 \(\Delta_{\mathcal G}=0\)，最终 logits 与 LM loss 在声明容差内保持不变；这项 identity 比较先使用 `BAL-NONE`，不把可能非零的路由辅助项混入 Base LM loss；
3. 虽然 forward 中性，每个受查逻辑参数键仍符合预先冻结的 `None`、connected-zero 或 nonzero 梯度分类，并且声明应更新的键在相应 optimizer step 后按预期离开初始化；不能要求所有 selector 参数首步都非零，因为 identity 起点下 Hard-ST 主任务局部导数中的 \(g-h\) 可以为零，其中 \(h\) 是 receiver 的残差输入，\(g\) 是对应完整计算输出；均匀路由下 balance 梯度也可以为零。selector 路径另用非中性输入/参数 fixture 或显式辅助目标验证；
4. Base 冻结时没有 Base 参数被更新；Base/LoRA 解冻时，参数组和梯度键集与声明一致。

若一个训练样本被切成多个 chunks：

- 需要跨 chunk 信用分配时显式使用 `detach_at_end=False`，并验证后块目标对前块 input/state 的梯度；
- 采用 truncated BPTT 时在声明边界使用 detach，不能把默认值当成未记录的训练策略；
- 默认只在 optimizer step 已完成、参数梯度已清空且 LM/balance 统计窗口闭合的边界保存 checkpoint。若明确支持梯度累积中途保存，则还必须保存已经累积的参数梯度，并保存或可确定性重建未归约 LM/balance 统计继续求 VJP 所需的输入、状态、RNG 与前缀；普通 Tensor checkpoint 不能恢复现场 autograd 图。

### 4.4 推进 gates

| development gate | 必须完成的证据 | 通过后才允许 |
| --- | --- | --- |
| `EXP-G0` 资产锁定 | checkpoint/tokenizer/data/Plan hashes，设备与软件栈清单，训练和评测预算 | 编写真实 Base 适配器 |
| `EXP-G1` Base replay | 对照 pinned upstream/reference 或其冻结 golden，比较 Base-only 的 logits、loss、KV continuation、mask、position IDs 和 next-token shift；训练范围另比选定 Base/LoRA 逻辑键的梯度与一步更新；再验证 prefill/chunk/decode 自洽 | 插入 SettleGraph |
| `EXP-G2` 中性接入 | 所用 placement 的 hidden 边界、identity output/logits/loss、参数 owner 和多次调用 state 行为通过 | 训练新增参数 |
| `EXP-G3` executor/设备闭环 | CPU eager/packed 对 NPU eager/packed FP32 的 forward、state、route、loss、关键梯度和 optimizer step 通过；profiler 没有未声明 CPU fallback | 使用 NPU 做短训练 |
| `EXP-G4` 训练机制 | 确定性小样本 overfit；各逻辑参数键的 `None`/connected-zero/nonzero 与更新分类；非中性 selector 路径；detach/no-detach；fresh-process 对不中断 reference 的下一步差分通过 | 启动首轮 development pilot |
| `EXP-G5` development pilot | 预注册矩阵、主要估计目标、checkpoint 选择、至少三个 seed、失败 run、多重比较和统计规则，保存完整原始曲线与干预指标 | 按冻结协议尝试复现信号 |
| `EXP-G6` 可复现信号 | 按冻结聚合与不确定性规则，主要指标达到预定改善，seed 方向稳定，状态因果干预和路由健康检查支持解释 | 扩大数据、checkpoint、sites、LoRA 或 continued pretraining |
| `EXP-G7` 工程决策 | 真实 route/state 记录与 profiler 确认下一瓶颈 | 优化已选拓扑；只在已满足第 9.1 节时投入通用多卡大图 |

这些 `EXP-*` 是针对实际实验切片的 development gates，不应冒用 `C00`--`C12` 或 `X01`--`X08` 的正式资格 cell ID。bring-up 和 development pilot 可以逐切片前进。`C*`/`X*` 及其冻结工件证明相应软件 capability；正式模型效果结论还必须满足本文预注册的数据、外部基线、对称调参预算、seed 与统计协议，不能仅凭 capability cells 推出。

任一 gate 失败时，应停在当前层修复、收缩条件或记录该方向失败；不能用扩大训练、增加卡数或切换更大 checkpoint 掩盖接入、梯度、恢复或路由问题。

## 5. 首轮训练与推理实验

### 5.1 训练能力阶梯

用于干净配对的每一级都从同一 Base checkpoint 身份开始，不能把上一级挑出的最好中间权重当成另一条件的原始 checkpoint：

1. **SettleGraph-only**：冻结 Base，只训练 SettleGraph。先做很小数据 overfit，回答中性初始化能否稳定离开、各参数键是否符合预先声明的梯度/更新分类；selector 的可导路径另由非中性 fixture 隔离，不以 identity 首步必须非零作为判据。
2. **SettleGraph + Base LoRA**：Base 主权重冻结，训练 SettleGraph 与明示的 LoRA 参数。它是资源受限情况下的首个实用微调条件。
3. **continued pretraining 或全量微调**：只有前两级出现可信信号、完整 checkpoint resume 可用且设备容量允许时进入。此时要把 Base 与 SettleGraph 的学习率、参数组和梯度裁剪分别记录。

允许研究显式注册的 staged curriculum，例如先训练 SettleGraph 再联合 LoRA；但它必须作为具有自身 `init-from` 链和预算的独立条件，不能冒充同一起点的配对实验。

每一级都应同时运行 prefill 和 stateful decode smoke。训练成功不能替代 KV cache、sequence reorder、reset 和多轮 state continuation 的推理检查。

### 5.2 最小科学对照矩阵

| 条件 | 主要作用 | 必须匹配或记录的量 |
| --- | --- | --- |
| Base-only replay | 验证冻结起点与评测接线，不作为训练后的充分对照 | checkpoint 与独立 reference/golden、数据和评测配置；不执行训练 |
| Base continuation / Base + LoRA | 与 BO 共享训练过程的主要 Base 对照 | SettleGraph + LoRA 对 Base + 相同 LoRA；SettleGraph 全量/CPT 对 Base continuation。匹配数据顺序、训练 Token、验证时点、seed、停止规则和对称调参预算 |
| Base + 匹配 Dense residual | 判断收益是否只是增加参数 | function-equivalent 初始化；在看结果前二选一固定 parameter-matched 或 active-FLOP-matched 规则，并匹配 placement 与训练预算 |
| SettleGraph SD/pre | BO 的直接内部对照 | 与 BO Plan 除 profile 外匹配；state、active \(K\)、NodeCompute、Emit、Aggregate、optimizer 相同 |
| SettleGraph BO/pre | 首个主要条件 | 同 SD/pre；自然 route 是主结果，只有待实现并验证诊断接口后才允许 replay route |
| BO state intervention | 判断状态是否真的被读取 | 正常运行对照 freeze、clear、shuffle、no-read、reset；一次只做一种干预 |
| BO/post | 判断当前 proposal 参与选择是否额外有用 | 对照 BO/pre/content；不能伪装成与 SD 具有完全相同 selector 输入 |
| BO/pre + `BAL-NONE` | 隔离 balance loss 的影响 | 对照使用预先固定系数的 `BAL-AVAIL-SOFT`，其余条件不变 |
| Flat MoE | 面向外部稀疏基线 | 明确 expert、Top-K、gate、capacity、drop/reroute、shared path 和辅助 loss |

首个 pilot 不必一次运行整张表。工程最低闭环是 Base-only replay、SD/pre、BO/pre 和至少一种 state intervention；形成训练收益判断前，必须加入匹配的 Base continuation 或 Base + LoRA。Dense residual 与 Flat MoE 在开始声称结构优势前补齐。

SettleGraph-only 相对冻结 Base 的比较衡量“增加一个可训练模块”的整体收益，不是对称训练过程的 Base 对照；归因于 BO 结构时仍需要 SD、匹配 Dense residual 和适用的外部稀疏基线。

所有主要条件使用相同的数据顺序、训练 Token、验证时点和 seed 集。首轮实验建立独立、预先冻结的 seed 域；资格 toy 条件中的 `17/23/47` 不作为默认实验 seed。调参预算也要对称：若 BO 获得额外学习率或 balance 系数搜索，主要对照应获得预先声明的等价预算。

第一次正式矩阵运行前还要冻结主要估计目标，即哪两个条件在什么评测量、预算和时点上的差值，以及它的汇总方法：使用最后一个预定 checkpoint，还是只按验证集选择 checkpoint；跨 seed 使用何种中心量、离散度或置信区间；OOM、nonfinite、提前退出和缺失评测如何计入；允许几次基础设施重试；同时检验多个任务、时间点或 BO 变体时如何控制或明确披露多重比较。所有条件使用同一 checkpoint-selection 规则，不能用 test set 或事后最好切片选择主结果。

主结果使用各条件自然产生的 route；route replay 只用于因果诊断，不替代模型实际路由。BO 与 SD 可以精确匹配 active NodeCompute，但 BO 还会对所有 reached receivers 执行状态 Update/commit；因此不能预设两者总 FLOPs、显存或 wall time 相等，必须实测并与质量收益一起报告。

每个 state intervention 都要固定被干预的 site、receiver/状态分量或 Read 路径、发生在 Observe/commit/read/reset 的哪个边界、持续的 Token/sequence 范围，以及恢复规则。shuffle 还要固定 sequence/node/component 轴与 seed；clear/reset 要区分单序列、单 site 和全局范围；no-read 要记录替代值。没有这些坐标，干预 run 不能互相比较。

### 5.3 正面结果的最低解释要求

任务相关的主要质量阈值必须在选定数据集后数值化。无论任务为何，至少同时检查：

- train/validation LM loss、perplexity 或任务主要指标的未平滑原始序列；
- 与 Base、SD 和匹配容量基线相比的均值、逐 seed 方向和最差 seed；
- reached、Observe、active、send 次数，soft mass、hard share、熵、active-set 变化和节点饥饿；
- 状态变化量、Read 输出、write-to-read 延迟，以及 freeze/clear/shuffle/no-read/reset 对 output/loss 的影响；
- 每个参数组的 gradient norm、nonfinite、实际更新量和零梯度比例；
- 训练 tokens/s、prefill tokens/s、decode latency、峰值 HBM、host memory 和通信量；
- 失败、重启、OOM、fallback 和被跳过 step。

只有状态干预改变了后续行为，才能说状态被模型使用。只有路由和状态指标健康且主要质量改善可复现，才能把结果解释为 BO 的正面信号。若质量改善来自明显更大的 active compute 或训练预算，只能报告该实际条件，不能归因于 BO 语义。

## 6. Checkpoint、恢复与运行记录

### 6.1 `init-from` 与 `resume`

两个入口互斥，并在创建最终 output directory 前完成 checkpoint identity、schema 和兼容性验证。

- `init-from` 装载 Base checkpoint 以及明确声明的 SettleGraph 初始化，从新的 optimizer、receiver state、数据位置和 RNG 轨迹开始。
- `resume` 恢复同一训练 run。完整恢复至少需要模型与 SettleGraph 参数、每个 `sequence_id` 的 receiver state 和 next position、optimizer、scheduler、AMP scaler、global step、Token 数、sampler/data cursor、CPU/backend RNG、distributed rank/world-size 约束，以及与保存边界相符的梯度累积和未归约统计状态。若契约允许在 live prefill/decode session 中断，还必须保存 Base KV cache、row—sequence 映射及 cache/state ownership；否则应明确只在没有 live KV session 的训练安全点保存。
- CPU checkpoint 到 NPU 可以作为 portable handoff；除非 backend、软件栈、数据顺序和确定性设置完全匹配并已有证据，不承诺跨 backend 的逐 step 同轨迹恢复。
- 保存后必须退出进程，再由 fresh process 加载并执行下一步；同一进程内 save/load 不能单独作为 resume 证据。

当前 checkpoint v1 只接受独立 `SettleGraph` owner，可复用其 Plan、SettleGraph 参数 owner、状态和 optimizer 校验逻辑，但不能直接装载 Base + SettleGraph + LoRA 联合模型；真实 Base 训练 checkpoint 需要扩展上述缺失范围。扩展时不得放松现有 no-partial-commit、hash、shape、dtype 和 rollback 约束。

默认 checkpoint 只允许落在 optimizer step 与完整 LM/balance 统计窗口共同闭合的边界，此时 gradient-accumulation microstep 为零且参数梯度已清空。若 schema 明确允许在 accumulation 中途保存，则必须保存已累积参数梯度；对尚未 backward 的 LM 或 balance 充分统计，还必须保存能在新进程中确定性重放并重建 VJP 的输入、前置状态、mask、RNG 和窗口位置，而不能只保存已经 detach 的统计数值。两种策略必须二选一并由 fresh-process 下一步测试验证。

同栈 exact resume 的“下一步通过”指与不中断 reference 使用相同 next-batch identity，并比较 output/loss、route、receiver state/position、选定参数梯度、optimizer 后参数、scheduler/scaler、data cursor 和 RNG continuation；只证明新进程能够再执行一步不够。跨 backend portable handoff 使用另行声明的数值与轨迹 envelope，不冒充 exact resume。

### 6.2 每个 durable run 的最低记录

本地 `runs/` 记录是证据源；Trackio 只作为可从原始记录重建的可视化投影。每次运行至少保存：

- run ID、命令、开始/结束时间、status、退出码；
- 仓库 commit、dirty 状态、运行前后源码 fingerprint；
- host architecture、CPU/NUMA、实际可见设备数量与拓扑、实际分配的 device 精确 SKU/逻辑索引、Torch/TorchNPU/CANN/driver/HCCL 版本；
- Base checkpoint、tokenizer、data、Plan、初始化和 checkpoint hashes；
- executor、placement、请求及 resolved backend、`resolution_reason`、dtype、autocast/reduction、attention backend、编译/自定义 kernel、fallback visibility、seed 和确定性设置；
- 完整训练配置、可训练参数清单、detach/truncated-BPTT 策略；
- 原始逐 step metrics、评测输出、峰值内存、profiler、stdout/stderr；
- terminal summary、主要判据是否通过及所有偏离预注册配置的说明。

长实验应在 clean exact commit 上启动。调试 run 可以是 dirty，但必须明确标为 exploratory，不能与正式结果合并。

## 7. 如何利用本机加速卡

### 7.1 首轮扩展顺序

1. CPU 小 fixture 建立 eager reference 和独立 golden；
2. 单张 NPU 完成当前 commit 的 FP32 forward/backward/optimizer/checkpoint 与 fallback 闭环；
3. 单卡完成小样本 overfit 和短 pilot；
4. 若模型可单卡容纳，优先使用数据并行扩到 2、4、8，再到实际可见数量允许的目标上限 16 卡，因为它不改变单个 SettleGraph 的逻辑节点所有权语义；
5. 若模型或 optimizer 不能单卡容纳，再引入参数/optimizer sharding，并为 checkpoint 与数值行为建立独立证据；
6. 只有需要让同一个 SettleGraph 的节点跨卡放置时，才进入节点并行和 DAG 通信设计。

若只读盘点确认本机确有 16 张可用卡，“充分利用 16 张卡”首先意味着提高可靠实验吞吐，而不是从第一天就让单个图横跨 16 卡。数据并行能够更早回答 BO 是否有效；节点并行解决的是随后扩大总容量的问题。数据并行时，同一 live sequence 应稳定归属一个 rank；若在 ranks 之间迁移或 reorder，必须连同 Base KV cache、receiver state 和下一 Token position 一起转移并验证。

数据并行不能改变统计窗口语义。对每个 site-region 和 receiver \(v\)，先跨所有 chunks 与 ranks 合并 \(N,P_v,A_v,F_v,Q\)：\(N\) 是纳入路由统计的非空候选事件数，\(P_v\) 是 receiver 可用时 selector probability 的和，\(A_v\) 是 availability 基准份额的和，\(F_v\) 是 hard active 份额的和，\(Q\) 是候选数至少为 2 的竞争事件数。\(P_v\) 的 collective 必须保留 selector probability 的梯度，其他量 stop-gradient；随后才在完整窗口构造 `BAL-AVAIL-SOFT`。不能逐 rank 构造非线性 balance loss 后简单平均。LM loss 同样先全局合并负对数似然总和与 target 数再归约。完整公式见[实现与等价性验证计划第 6.3 节](settlegraph-implementation-plan.md#63-状态chunk-与随机数)。

动态路由还可能使各 rank 的 unused 参数集合不同。任一 rank 与参数连通时，全局归约结果必须在所有 replicas 上表现为 Tensor gradient；只有所有 ranks 都不连通时才保持全局 `None`，否则 Adam/AdamW 与 weight decay 行为可能改变。扩卡比较还要冻结 global batch、gradient accumulation、数据采样和学习率缩放规则，不能把它们的变化归因于卡数。

### 7.2 设备正确性优先于低精度和规模

当前共享 executor 明确只接受 FP32/FP64，而 NPU 首轮使用 FP32。BF16/FP16 不能只通过替换 execution binding 加入：必须先闭合各 Tensor 的 dtype role、accumulation/reduction dtype、autocast 与 rounding 规则及逐公式 tolerance，并进入资格计划第 13 节定义的 Schema v2 扩展，使用新的 schema/canonicalizer identity。之后才分别验证 output、state、route、loss、gradient、optimizer、checkpoint、真实设备算子和 fallback。不能因为 Base 模型能用 BF16，就假定 SettleGraph 的 selector、递推状态和 Top-K 边界也已获得同等能力。

每次扩卡都先运行固定 smoke 和短 continuation，再运行长训练。任一卡出现 fallback、HCCL timeout、OOM 或不一致时保存完整失败记录；不通过缩短日志或自动跳过失败 rank 得到“成功”终态。

## 8. 与第 1 块并行的特化性能工作

特化性能工作只有两个启动理由：

1. 实际 BO 垂直切片已经正确，但 profiler 证明 SettleGraph executor 是训练或推理的主要瓶颈；
2. 某个更大 pilot 需要当前实现无法承受的显存或延迟，而该配置已经有足够科学价值继续投入。

优化顺序是：

1. 先保存目标 workload、正确性输入和 profiler；
2. 区分 Base、packing、selector、state Update/Read、NodeCompute、scatter、状态发布和跨卡通信耗时；
3. 只优化占主要时间或内存的部分；
4. 用同一 parameter owner 和同一输入，逐项比较 eager、通用 packed 与新特化路径；
5. 再运行端到端吞吐和质量回归。

当前已知的重点风险包括：grad-enabled packed 前向为了恢复每个公开 Tensor occurrence 的精确 source liveness，会重复遍历语义连接并包含 device 标量读取/CPU 转换；现有 profile counters 没有覆盖这项成本。StateStore 打包/发布、position 校验和部分下一位置更新也仍有 host-side mapping；当前 stable Top-K 的 pairwise rank 还会形成随 region 宽度近似二次增长的中间量。首轮设备 profiler 必须观察真实 callback、host synchronization 和 kernel timeline，不能以 `python_*_hot_loops=0` 的静态字段代替。

当前 decode 是长度为 1 的 packed prefill 路径复用，已有正确性证据，但旧 CPU 探索记录没有观察到加速。低延迟 decode 需要单独的 workload 和调度/kernel 设计，不应从 prefill 吞吐外推。此外，同一结果图上不同 roots 的并发 backward 尚未声明或验证；在其生命周期和线程安全契约闭合前，实验 runner 不得隐式依赖该用法。

候选优化包括把 route 与 source-liveness 信息在一次设备端/编译遍历中复用、把同构 node/region 工作按 shape 和算子分组、将因果 state scan 放入编译图或设备 kernel、为固定 BO Plan 生成静态 schedule，以及把非 trace 训练路径与诊断 trace 路径分离。任何优化都不得改变 BO 的 Observe/commit 时序、Top-K、state owner 或消息结算。

## 9. 通用超大 DAG 与多卡性能路线

### 9.1 启动条件

通用引擎开始大规模实现前，至少满足：

- 一个 SettleGraph + BO 配置在预注册 pilot 中产生可复现正面信号；
- 已确定值得扩展的 topology、state、selector timing、NodeCompute 和 active \(K\)；
- 已保存真实 run 的 route、节点利用率、state 大小、算子 shape、通信候选和 prefill/decode 分布；
- profiler 证明增加节点数或跨卡放置是下一主要限制，而不是 Base、数据管线或尚未优化的小图实现；
- 当前 token-major eager reference 与独立 golden 能为计划采用的物理分片生成小规模对照，packed 继续作为强开发差分路径。

### 9.2 物理实现需要解决的问题

- 固定 node 到 device 的放置，并区分 Plan 中决定稳定键和状态含义的唯一逻辑 owner、实际保存与更新 Tensor 的 authoritative physical shard，以及数据并行的同步 replicas；checkpoint 必须能恢复三者映射；
- 按 device、算子类型、shape 和拓扑阶段形成设备端工作队列，避免 Python 对每个 node/Token 循环；
- region Top-K、消息 fan-out/fan-in、Line/barrier 和 terminal Aggregate 的确定性恢复；
- 卡内 grouped/fused kernel 与跨卡点对点或 all-to-all 的选择；
- active 节点不均衡、热点卡、空队列和 straggler；
- sequence reorder、reset、checkpoint 和失败回滚下的分布式 state ownership；
- prefill 的大批量吞吐与 decode 的小批量低延迟分别优化。

物理上的分层路由只有在精确恢复同一候选集与全局 Top-K 时，才能仍称为同一实验条件。若先选卡再选卡内节点导致 active set 改变，它是新的模型语义/实验条件，不能隐藏在执行器优化名下。同样，语义边当前携带完整 \(d_{\mathrm{model}}\) hidden；有损压缩若被引入，必须成为明示的新条件。

复制式数据并行可提高首轮实验吞吐，但不减少每卡权重存储，稠密 gradient all-reduce 也不会自动获得 1/32 通信收益。FSDP/ZeRO 可减少部分常驻量，但可能为整个参数组 all-gather。node/region-owner sharding 是让节点权重长期留在 owner 卡上的直接方案，但不是唯一方案；expert-aware sharding、tensor parallel 或按需参数调入也可减少每卡常驻量，只是会形成不同的 collective、带宽和延迟模型。所有方案都必须计入 hidden/state 路由和反向通信。

为避免把同一 payload 向同一卡上的多个 target 重复计数，令 \(\mathcal H\) 是所有可复用的 hidden 产生事件，包括图入口注入、node Emit，以及 terminal Aggregate 返回 Base 的边界事件。对 \(u\in\mathcal H\)，\(s(u)\) 是 payload 所在源设备，\(D(u)\) 是所有需要该 payload 的不同目标设备集合；\(b_h\) 是 hidden 每个元素的字节数，\(d_{\mathrm{model}}\) 是 hidden 宽度。假设同一 payload 在每个目标设备只传一次且不压缩，一个 reuse-aware 的逻辑前向 payload 基线为

$$
V_{\mathrm{hidden,logical}}
=b_h d_{\mathrm{model}}
\sum_{u\in\mathcal H}
\left|\{d\in D(u):d\ne s(u)\}\right|.
$$

这不是物理链路字节的充分预测：同一远端卡内 fan-out 可复用，跨卡 multicast、batching、协议与实际互联路由会改变源端发送次数和逐链路字节。真实运行必须另记应用 payload、各链路字节和消息数；反向 hidden gradient、Top-K/logit 协调、CLOSED/mask/index 控制、balance 统计归约、参数 collective 和 barrier 空闲也要分列。固定逻辑拓扑不会自动变成低物理通信。

### 9.3 1/32 激活不能直接推出 32 倍总容量

令 \(\mathcal Q\) 是所有已执行且 candidate 集非空的 Token-region settlement 事件，包括不调用 selector 的 forced singleton。对 \(q\in\mathcal Q\)，\(\mathcal C_q\) 和 \(\mathcal A_q\) 分别是 reached candidates 和 active receivers，\(\Phi_{q,v}\) 是 receiver \(v\) 在该事件上昂贵 NodeCompute 的实际或声明 FLOPs；窗口 Attention 等计算可随有效历史长度变化，不能无条件写成与 \(q\) 无关的常数。至少应分开报告按节点计数和按昂贵计算加权的激活比：

$$
\alpha_{\mathrm{count}}
=\frac{\sum_{q\in\mathcal Q}|\mathcal A_q|}
{\sum_{q\in\mathcal Q}|\mathcal C_q|},
\qquad
\alpha_{\mathrm{flop}}
=\frac{\sum_{q\in\mathcal Q}\sum_{v\in\mathcal A_q}\Phi_{q,v}}
{\sum_{q\in\mathcal Q}\sum_{v\in\mathcal C_q}\Phi_{q,v}}.
$$

若 \(\mathcal Q\) 为空，这两个比例记为不适用而不是零。“1/32 激活”必须声明指的是哪一个量。即使每个宽度 32 的普通 region 都做 Top-1，forced singleton、不同 NodeCompute 大小与可达候选集变化，也可能使全模型的两种比例不是 1/32。每 Token 总计算还近似为

$$
F_{\mathrm{token}}
=F_{\mathrm{Base}}
+F_{\mathrm{aggregate/selector/route}}
+F_{\mathrm{state}}
+F_{\mathrm{active\ NodeCompute}}.
$$

其中 \(F_{\mathrm{Base}}\) 是 Base 模型成本，\(F_{\mathrm{aggregate/selector/route}}\) 是消息聚合、候选打分和路由成本，\(F_{\mathrm{state}}\) 是 Observe、Update 和 commit 的状态成本，最后一项是 active receivers 的昂贵计算。最后一项始终直接随 active NodeCompute 稀疏化；BO 的状态成本仍随 reached 集合发生，SD 的 Observe/Update/commit 则随 active 集合稀疏。selector 仍处理 reached candidates，Base、图结算和通信也不会自动按 1/32 缩小。

参数激活还要区分唯一存储量和累计访问量。令 \(P_{\mathrm{NC,all}}\) 为统计范围内全部 receiver NodeCompute 参数量，\(P_{\mathrm{NC,touched}}(t)\) 为 Token \(t\) 至少调用一次的 NodeCompute 唯一参数存储量，则

$$
r_{\mathrm{param}}(t)
=\frac{P_{\mathrm{NC,touched}}(t)}{P_{\mathrm{NC,all}}}.
$$

这个定义不含 Base、selector 或状态参数；若报告全模型比例，必须重新声明相容的分子和分母。带宽代理则应另报按调用次数累计的参数元素访问量 \(\sum_{c\in\operatorname{calls}(t)}P_{\mathrm{NC}}(c)\)；同一参数被多次调用会重复计数，因此不能与唯一存储量互换。

容量受最拥挤 rank 的 peak working set 限制，而不是全局逻辑参数和一个“常驻内存”数字。令 \(r\) 表示 rank，\(c\) 表示权重、梯度或 optimizer 的 dtype/量化等 storage class；\(P_{\mathrm{all},r}^{(c)}\) 与 \(P_{\mathrm{train},r}^{(c)}\) 分别是该 rank 上属于 \(c\) 的常驻参数和需要训练状态的参数元素数，\(b_w^{(c)},b_g^{(c)},b_m^{(c)},b_1^{(c)},b_2^{(c)}\) 分别是权重、梯度、可选 master weight 和两个 optimizer moment 的每元素字节数。再令 \(M_{\mathrm{act},r},M_{\mathrm{KV},r},M_{\mathrm{state},r},M_{\mathrm{buf},r}\) 为该 rank 的峰值 activation、Base KV cache、receiver state 和通信/临时 buffer，则估算式是

$$
M_{\mathrm{peak},r}
\approx\sum_c P_{\mathrm{all},r}^{(c)}b_w^{(c)}
+\sum_c P_{\mathrm{train},r}^{(c)}
\left(b_g^{(c)}+b_m^{(c)}+b_1^{(c)}+b_2^{(c)}\right)
+M_{\mathrm{act},r}+M_{\mathrm{KV},r}+M_{\mathrm{state},r}+M_{\mathrm{buf},r},
\qquad
M_{\mathrm{peak,max}}=\max_r M_{\mathrm{peak},r}.
$$

\(M_{\mathrm{act},r}\) 和 \(M_{\mathrm{buf},r}\) 不是常驻项；它们被保留是因为容量实验关心峰值。具体 optimizer 少于或多于两个 moments 时应替换相应项，不能硬套 Adam 模型。\(r_{\mathrm{param}}\) 与 \(\alpha_{\mathrm{count}}\) 或 \(\alpha_{\mathrm{flop}}\) 也不必相等。稀疏激活不会自动减少上式的全部权重，也不会自动减少全量训练时的梯度与 optimizer state。参数/optimizer sharding、offload、量化和节点放置会改变每卡内存，但必须分别计入通信与带宽成本。训练足够久时，短期未 active 的节点仍可能在后续被访问，checkpoint 和 optimizer 容量不能按当前 batch 的 1/32 估算。receiver state 的最坏容量也取决于 live sequences 与跨时间被触及的全部 nodes，而不只是当前 Token 的 active set。

因此以下四类容量实验必须分开：

| 模式 | 主要峰值项 | 主要吞吐限制 |
| --- | --- | --- |
| 全量训练 | 全部权重、梯度、master/moments、activation、状态 | HBM、collective/消息通信、backward 和 optimizer |
| LoRA FT | Base 权重、SettleGraph 全量训练状态、LoRA 训练状态、activation | Base forward/backward 激活、图状态、通信 |
| `eval`/`no_grad` 纯 prefill 推理 | 权重、KV、receiver state、整段临时 Tensor | 算子吞吐、内存带宽、packing、跨卡批量消息 |
| `eval`/`no_grad` 纯 decode 推理 | 权重、增长的 KV、receiver state、小批量 buffer | kernel launch/调度、同步、跨卡延迟和负载不均 |

grad-enabled prefill 属于训练工作负载，其 activation、source-liveness 和反向内存不能从 `no_grad` 推理结果外推。纯 CPU 还应作为独立 binding 测量。它可能凭主存容纳更多权重，但稀疏权重访问、NUMA 和内存带宽可能使 decode 很慢；不能把“能装下”当成“有实用吞吐”。

### 9.4 最大规模与吞吐研究方法

后续不能先写一个“16 卡可训练多少 B 参数”的单点答案，也不能在确认实际可见设备前假定 16 卡均可用。应按以下顺序形成容量 envelope：

1. 记录实际可见设备数、每张卡可用 HBM、主机 RAM、NUMA、卡间拓扑，并实测大块带宽、小消息点对点延迟和 collective 延迟；decode 往往受后两者支配；
2. 对每种模式固定 dtype、optimizer、sharding/offload、batch、sequence length、state 和 KV 规则；
3. 用上述内存模型计算理论上界，并预留框架、allocator、碎片和通信 buffer 余量；
4. 用逐级增大的真实稳态 probe 找到可运行上界；训练 probe 至少包含若干次 forward、backward、optimizer step、`zero_grad` 和 checkpoint save/load，因为 Adam 类状态常在首次 step 才初始化；推理 probe 覆盖 warmup、KV/state 增长与释放；
5. 在 1、2、4、8，直至实际可见数量允许的目标上限 16 卡分别做 strong scaling 与 weak scaling，同时覆盖均衡 route 和对抗性偏斜 route；
6. 对纯 prefill 推理分别报告完整 prefill latency 与 tokens/s；只有完整自回归请求才报告首 Token 延迟（time to first token，TTFT），并明确其是否包含排队、tokenization 和调度。对 stateful decode 报告按 batch/并发分组的单 Token p50/p95/p99，对全量和 LoRA 训练分别报告 forward/backward/optimizer 耗时；
7. 每项同时保存峰值 HBM/RAM、allocated/reserved memory、通信 buffer、链路字节、collective 次数、每卡 max/mean 负载、kernel batch size、host/device timeline 和 fallback；
8. 分别报告总参数、每 Token 的 \(P_{\mathrm{NC,touched}}\)、按调用累计参数访问量、\(\alpha_{\mathrm{count}}\)、\(\alpha_{\mathrm{flop}}\)、实际 active FLOPs 和端到端吞吐，不用稀疏比代替这些量；
9. 在最小和接近上界规模上都运行语义对照，防止只在小图正确、扩容后改变路由或状态。

## 10. 新会话的直接起点

新会话应按下面的顺序继续，不从通用性能重构开始：

1. 完整阅读本文，以及主语义文档中 placement、BO/SD、配对实验、loss 和实验记录章节；保持主语义文档不变。
2. 确认工作树、分支和 `5712e66` 基准仍可定位；审视任何新的未提交交接，不能默认其正确。
3. 只读盘点本机实际可见的加速卡数量、可用内存/拓扑/软件栈（扩展目标上限为 16 卡），以及本地 Base checkpoints、tokenizers 和候选数据；不要在未确认 identity 前启动下载或长任务。
4. 基于实际资产，物化第 4.2 节的首个垂直切片配置；若 checkpoint 或数据选择会实质改变研究目标，再向用户确认。
5. 接通真实 Base replay 和一个 site 的 placement，先完成 Base-only 与函数等价初始化测试。
6. 接通 prefill/decode、KV cache、LM loss、balance loss、参数组和完整 checkpoint resume；为每个新边界添加 CPU 小 fixture。
7. 在当前机器上完成单 NPU FP32 eager/packed 差分与 fallback profiler，再做小样本 overfit。
8. 所用执行路径稳定后，在 clean exact commit 上运行持久化的三 seed pilot。
9. 根据质量、状态干预和 profiler 决定：继续扩大 BO 实验、只优化该拓扑，或停止该配置。没有正面信号时，不启动通用超大 DAG 加速。

首轮交付不是“16 卡上最大的模型”，而是一个可以从真实 checkpoint 新建、训练、退出、恢复、prefill、decode，并能与 Base/SD 对照的 SettleGraph + BO 条件。这个闭环成立后，模型效果、特化性能和通用扩容才有共同的可靠起点。
