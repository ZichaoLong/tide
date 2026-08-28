# TIDE Checkpoint 生长实验协议

> 状态：近期实验政策与验收口径。
>
> 项目目标和当前主线见 [README](../README.md)；结构与机制的详细含义见 [实验语义、命名与数学符号](experiment-semantics-and-naming.md)。

## 1. 总原则

1. 工作流 A 与 B 共享数据、训练配方、correctness oracle、成本口径、checkpoint 谱系和实验账本。
2. 探索实验可以同时引入一组有共同理由的机制，用来寻找完整候选的正面信号；任何单项因果结论都要补直接反事实。
3. 总参数、active parameters、实际 FLOPs、训练 Token、优化器和数据尽量匹配。无法同时匹配时，分别报告 capacity-matched、compute-matched、resource-matched 和 quality-matched 结果。
4. Correctness、机制使用、训练质量、容量扩展和系统性能使用彼此独立的证据门。
5. Selector 的模型语义不能依赖 batch 组成、chunk 切分或实时设备负载。
6. 每个实验完整记录静态拓扑、门控范围、传播 profile、状态生命周期、selector、预算、merge/backbone 和物理放置。
7. 下一项机制由已观察问题牵引，不按预设阶段机械推进。

实验分为三类：

- **探索实验**：允许使用最小但完整的机制包，只回答是否出现正面存在性信号。
- **诊断实验**：针对已观察问题做 knockout、paired counterfactual 或局部修改。
- **确认实验**：冻结结构和训练配方，用新 seed、数据切片、硬件或规模复现结论。

## 2. 两条并行工作流

### 2.1 工作流 A：dense / flat MoE 基线

工作流 A 建立三层参照：

1. 原生 dense checkpoint 的 correctness 与 continued-training oracle。
2. 成熟 flat MoE reference recipe 或兼容原生实现。
3. 从 dense checkpoint 中性生长的 flat MoE matched control。

Head-Wise MoE 可以在这些基线可靠后作为可选后续，用来检验局部因子化 MoE 的质量、通信和 kernel 利用率。它不是工作流 B 的启动前置条件。

#### 原生 checkpoint 必须验证

- state-dict 参数逐项覆盖；
- logits 与逐层主要 artifact 对齐；
- 单 Token `decode` 与多 Token `prefill` 对齐；
- 任意 chunk continuation state 对齐；
- 训练 loss 和主要参数梯度对齐；
- save/reload 与固定种子可重复。

#### Flat MoE 强基线

成熟强基线应包含合理调优的 router、load balancing、shared expert、expert packing 和训练超参数。Checkpoint-grown matched control 至少比较：

- 零 residual 或零输出投影，初始化函数等于原模型；
- clone-and-split，并经代数验证保持旧模块贡献；
- token-local Top-K 与立即 weighted merge；
- 无 expert 私有跨 Token 状态；
- 可配置 shared expert 或 always-on residual path。

如果 TIDE 只优于尚未调好的 checkpoint-grown MoE，不能据此声称优于 flat MoE 强基线。

### 2.2 工作流 B：寻找可训练、可扩展的 TIDE 候选

工作流 B 的核心任务，是从 checkpoint 中性生长出至少一个符合 TIDE 结构方向、能够稳定训练并显示容量扩展价值的成功候选。

当前默认稳定骨架是：

- 完整保留的 checkpoint / always-on backbone；
- 中性 residual growth；
- fan-in 和 fan-out 具有统一上界的局部分支；
- 明确 active/message budget 的稀疏昂贵计算；
- 在声明位置使用 fan-in 有界的 fixed merge，或回到稳定 region/backbone 接口。

工作流 B 把 `broadcast-observe`（BO）作为主要验证轴。第一轮候选会围绕下面两条可能作用路径设计：

面向以后的一般 Graph，BO 是主要候选 profile；N 和 SD 分别作为从无状态 MoE 与 selected dispatch 出发的 matched controls。这个定位不预先代表 BO 已被证明更优。

```text
当次 Observe / Proposal：
receiver 看到当前消息并更新摘要
-> proposal 或 selector 输入改变
-> 当前 active set 或输出改变
```

```text
跨 Token 延迟记忆：
未执行昂贵计算的 receiver 收到消息
-> 私有状态发生变化
-> 以后激活时读出该状态
-> 输出或 loss 可测地改变
```

私有状态、交叉汇聚和 backbone reinjection 是与 BO 并列需要组合、替换或消融的重要候选。第一轮探索可以把其中若干机制放进同一个完整候选；随后必须用匹配的 `selected-dispatch`、状态 knockout 和交叉边开关建立归因。

浅层 BO 实验可以先判断传播与状态机制是否有 learning value；要支持 TIDE 的容量扩展主张，后续候选必须进入有界度多跳递归或等价的空间 scaling，而不能永远停留在一跳广播。

## 3. 问题驱动的闭环

一次完整循环是：

```text
观察
-> 定位失败类型
-> 选择针对性机制
-> paired counterfactual
-> 新 seed / 数据 / 规模确认
-> 沉淀或否定 know-how
```

| 观察到的问题 | 优先测量 | 可尝试的候选机制 |
| --- | --- | --- |
| receiver 长期不激活 | receive/update/active/gradient 覆盖、语义分数 | recovery bias、shadow activation、局部 quota |
| 状态持续写入但没有作用 | read sensitivity、write-to-read 延迟、knockout | 加强 readout、局部辅助目标、缩短读出距离 |
| 下游历史覆盖不足且确认伤害质量 | 消息覆盖、历史重叠、记忆任务、causal knockout | BO、提高局部 K、私有状态、backbone reinjection、交叉汇聚 |
| route/path 快速漂移 | route churn、节点输入漂移、merge 前后跳变量 | 缩短 merge 距离、慢化 selector、强化公共接口 |
| 长路径梯度不足 | 分层梯度覆盖、动态 hop、控制寿命 | 更频繁收拢、局部辅助 loss、缩短递归寿命 |
| Observe / Update 成本过高 | 投影、写带宽、fan-out、packing | 压缩状态、降低 fan-out、降低写入维度或频率 |
| mixer 重新形成全局通信 | collective 范围、物理距离、等待 | region-local、低秩、tree/hierarchical 或邻居 mixer |

## 4. 必要的直接反事实

首次探索不要求穷举整个设计空间，但对应结论出现前必须补足相应对照。

| 对照 | 回答的问题 |
| --- | --- |
| 原 checkpoint continued pretraining | 新结构是否优于继续训练原模型 |
| 等参数 dense 扩展 | 收益是否只来自更多参数 |
| 等 active-FLOPs flat MoE | 局部候选是否优于成熟稀疏基线 |
| 相同拓扑和 replay route 的 `selected-dispatch` | BO 的消息可达和未选更新是否有额外价值 |
| BO selector 读取 vs 忽略 post-Update state | 更新后的状态改变本次选择是否有独立作用 |
| BO 但冻结未激活 state 或禁止延迟读出 | 收益是否来自未激活期间积累的状态 |
| state clear / shuffle / no-read / reset | 已写状态是否按预期影响以后输出 |
| fixed/hash route | Learned selector 是否真正有价值 |
| matched Leaf-Gated 配置 | 内部 receiver 门控是否有独立作用 |
| 无状态 FFN 路径 vs 有状态 Attention/SSM receiver | 收益来自条件计算，还是私有序列记忆 |
| 相同多父拓扑关闭 vs 开启交叉边 | 扩大局部历史来源是否改善质量 |

最干净的 BO 对照让两种传播 profile 使用相同 active set，必要时直接 replay route，只改变未选 receiver 是否收到并更新。静态拓扑、状态容量、昂贵模块、active budget、merge、训练 Token 和物理放置保持不变。

如果 BO selector 读取 post-Update state，`selected-dispatch` 在选择前没有同一个输入。此时要增加一个 selector 忽略 post-Update state 的 BO 对照，把“状态改变本次选择”和“未激活状态以后被读出”分开。

## 5. 五道证据门

### 5.1 Correctness gate

- 原生 checkpoint 的参数、logits、cache/state、训练 loss 和主要梯度对齐。
- 函数保持生长的初始化时刻，旧模型输出和旧状态轨迹对齐。
- `prefill`、逐 Token `decode` 和任意 chunk continuation artifact equality。
- 不同 batch、合法 chunk 和物理调度不改变单序列 reference semantics。
- 空消息、多上游聚合、selector 和 merge 顺序具有确定语义。
- 拓扑、传播 profile、状态和 selector 配置可保存、恢复并重放。

“从 checkpoint 无损生长”只要求初始化时保持旧模型可观察行为。新增状态可以在后台形成轨迹，但在中性初始化期间不能影响旧输出。零输出接口还必须说明新分支怎样获得梯度。

### 5.2 Mechanism-use gate

完整候选 loss 下降并不能证明 BO、私有状态或交叉汇聚被实际使用。至少要建立一条可重复的因果链：

```text
Observe / Update 或交叉消息发生
-> selector、状态或昂贵分支实际读取它
-> blocking / perturbing 该路径改变输出或 loss
```

至少记录：

- receive、update、active、read、emit 覆盖率；
- 状态变化量与 write-to-read 延迟；
- selector 和 readout 对状态的敏感度；
- freeze、clear、shuffle、no-read、reset 的输出或 loss 差异；
- 交叉边开关前后的局部历史来源与行为变化。

### 5.3 Training gate

- train/validation loss、perplexity 和下游质量；
- route churn、active-set overlap 和节点输入漂移；
- 每个节点的消息、更新、激活、发送和有效梯度次数；
- 梯度范数、覆盖率和长期未更新参数比例；
- 私有状态稳定性、历史覆盖与 read 延迟；
- backbone、递归、交叉边和 merge 频率的消融；
- 多 seed 或新数据切片复现。

消息均衡、激活均衡和梯度均衡不是一回事，必须分别记录。

### 5.4 Scaling gate

- 扩大潜在节点、总参数、递归深度或空间直径时，单节点 fan-in、fan-out、参数、状态和候选处理继续具有统一上界。
- 分开报告总容量、最大单节点成本、每 Token 到达节点、昂贵激活、Emit 边、Observe / Update 和状态容量。
- 质量或能力随潜在容量增长，而不是只在固定小模型上超过弱基线。
- 深度增加时，下游历史覆盖、route churn、梯度和状态利用没有失控。
- 收益不依赖不断增大的 fan-out、全局 router、全局 mixer 或近似稠密的广播。

只有小规模 learning value，不能证明 TIDE 的容量扩展目标。

### 5.5 System gate

- 总参数、active parameters 和实际 FLOPs；
- 消息、Update、selector、状态读写、昂贵计算、packing 和 merge 的分项成本；
- 训练吞吐、`prefill` 吞吐和 `decode` latency；
- 峰值显存、私有状态和 optimizer state；
- 跨设备字节、邻接距离、collective 范围和等待；
- grouped GEMM 利用率、设备负载分布和尾延迟。

系统收益至少要求：

```text
被跳过的昂贵计算与远程通信
>
局部消息 + Observe/Update + selector + 状态读写 + packing + merge
```

## 6. BO 结论的四个层次

关于 BO 的结论必须逐级说明，不能互相替代：

1. **机制运行并被使用**：Observe / Proposal 或未激活状态写入对模型行为有因果作用。
2. **具有 learning value**：相对 matched `selected-dispatch`，质量、样本效率或稳定性可复现改善。
3. **具有 scaling value**：潜在容量扩大时仍有质量收益，消息、状态和昂贵激活保持可控。
4. **具有 system value**：新增传播与状态成本小于跳过的工作，并带来端到端收益。

工作流 B 把 BO 作为主要验证轴，但不会用“消息送达”代替以上任何一层证据。

## 7. 近期软件边界

近期实现建立少量稳定抽象，不先完成一般 Graph runtime：

| 抽象 | 职责 |
| --- | --- |
| `CheckpointAdapter` | 原生装载、状态映射和 equality oracle |
| `GraphBranchBoundary` | GraphBranch 与 checkpoint backbone 的外部接口及唯一 merge |
| `GraphInputPort` / `GraphOutputPort` | 所有 GraphBranch 拓扑共用的唯一入口端点与终端聚合端点 |
| `HBLatticePlan` | 保存已展开的边界端口、Lines、节点、边、regions 和镜像直通 |
| `HBLatticeExecutionConfig` | 配置 propagation profile、node template/state、selector、Emit、消息聚合和训练期均衡 |
| `TopologyBuilder` | 由规则树、逐坐标混合或空间 Graph 生成 Plan |
| `WavefrontExecutor` | 严格逐 Line 结算受限 HB-Lattice |
| `MessageProjection` | 固定、有界 receiver slots 和消息形状 |
| `ReceiverCell` | 实现单个 receiver node 的稳定输入、轻量读出、状态提交和完整输出契约 |
| `ReceiverNodeTemplate` | 组合状态模块、昂贵计算、归一化和 residual；当前默认是 Pre-Norm 双 residual |
| `PropagationProfile` | 切换 `selected-dispatch` / BO 并产生各类 mask |
| `Selector` | 在一个 Line 的固定有界区域内选择 reached nodes；它不是拓扑发散点 |
| `ReceiverState` | 保存节点私有状态，并实现 Update 与供 selector / node compute 使用的局部读出 |
| `EmitPolicy` | 把 active 节点的完整输出变成发往固定 children 的消息 |
| `AggregatePort` / `MessageAggregate` | 统一处理 receiver 输入与 GraphBranch 输出的局部消息聚合 |
| `BoundaryMerge` | 处理 GraphBranch 与 checkpoint backbone 的外部 residual merge |
| `BalancePolicy` | 仅在训练时根据 routing events 产生辅助均衡 loss |
| `RouteArtifact` | 记录每 Token 和每节点的 receive/update/active/read/emit |
| `ExperimentLedger` | 保存谱系、配置、成本、数据、checkpoint 和指标 |

同一 `HBLatticePlan` 必须能配合不同的 `HBLatticeExecutionConfig` 切换传播 profile、保存和恢复状态、执行 knockout，并分别统计轻量更新和昂贵计算。近期需要受限的 HB-Lattice 波前执行器，但不需要一般 event IR、任意 DAG 调度器、跨设备 allocator 或有环 Graph executor。

## 8. 首个可交付成果

1. 选定开放权重 checkpoint、训练数据、框架和目标硬件。
2. 完成原生 equality oracle、continued-training 校准与 save/reload 测试。
3. 建立统一 `ReceiverCell` contract、`ReceiverNodeTemplate`、传播 profile、状态和 instrumentation 接口。
4. 在工作流 A 建立 dense 与成熟 flat MoE 强基线。
5. 在工作流 B 实现一个保留 always-on backbone、具有有界局部连接、BO、可实际读出的私有状态、稀疏昂贵激活和 fixed merge 的首轮完整候选。
6. 同时保留 matched `selected-dispatch`、状态 knockout 和交叉边开关。
7. 完成短程训练并输出 correctness、机制使用、质量、路由、梯度、历史覆盖、状态利用和分项系统成本。

这个交付的成功标准是实验可重放、候选语义完整、问题可观测、主要验证轴能够配对比较。它不要求首轮结果已经证明 TIDE 有效。
