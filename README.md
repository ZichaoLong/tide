# Tide 实验仓库启动文档

> 状态：项目启动草案
>
> 日期：2026-08-10
>
> 目标读者：即将开始实现、训练与验证 Tide 的研究和工程人员
>
> 上游研究总入口：[ObsidianVault / TIDE](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/README.md)

## 1. 本仓库要回答什么问题

Tide 暂沿用全称 `Token Inference Decentralized Engine`。本仓库研究的是模型架构、训练语义和可执行实验，不把名称中的 `Engine` 限定为某一个既有 runtime。

Tide 的长期目标是研究一种自回归神经网络架构，使它同时具有：

1. **局部通信**：一个计算节点只与固定、数量有界的邻近节点通信，而不是让每个 Token 都可以被动态发送到任意全局节点。
2. **结构稀疏与激活稀疏**：静态 Graph 本身不是全连接；一次输入只执行全部潜在昂贵计算中的一小部分。
3. **可训练性**：动态选择不会使路径漂移、节点饥饿和长距离信用分配严重到无法稳定训练。
4. **`prefill = decode`**：逐 Token `decode` 与任意合法 chunk 的 `prefill` 实现相同的单序列 reference semantics。
5. **实际系统收益**：被跳过的工作足够昂贵，能够覆盖 selector、状态更新、packing、通信和负载不均衡的成本。

这五项存在天然张力。被跳过的模块必须足够大，稀疏执行才有收益；但单个模块的语义贡献又需要足够平滑、重叠或及时 merge，动态换路才不至于使训练失稳。Tide 当前最核心、也最可证伪的研究假设是：

> 在当前或未来的模型规模与硬件上，存在一种有实际意义的中间粒度：node 内部仍使用高效稠密 kernel，node 之间采用固定局部连接和动态稀疏激活，并且模型质量、训练稳定性与端到端性能可以同时成立。

本仓库首先推进可逐项归因的实验，不从最一般的 Graph 开始实现。近期主线是：

```text
原生 dense checkpoint
-> 标准 MoE 对照
-> Leaf-Gated Tide（叶级门控 Tide）
-> Receiver-Gated Tide（接收者门控 Tide）
```

HB-Line、HB-Lattice、一般空间 DAG 和一般动态 Graph 是远期研究方向。本仓库会为它们保留接口意识，但不会让远期表达力拖累第一批实验。

## 2. 从成熟基线开始

### 2.1 原生 dense checkpoint

近期实现从一个 pre-norm、decoder-only、开放权重 Transformer 开始。省略 dropout、位置编码和 cache 细节，一个标准 block 可以直白写成：

$$
h'=h+\operatorname{Attention}(\operatorname{Norm}(h)),
$$

$$
h^+=h'+\operatorname{FFN}(\operatorname{Norm}(h')).
$$

residual stream 是始终存在的公共接口；Attention 和 FFN 是在该接口上增加的计算分支。若新增 residual 分支的末端贡献初始化为零，扩展模型可以在初始点精确保持原 checkpoint 的函数。原生模型还已经提供成熟的 causal `prefill/decode`、训练配方和稠密 kernel，因此它是所有增量实验的 correctness 与质量基线。

第一步不是重新发明 Transformer runtime，而是完整装载 checkpoint，并对齐参数、logits、KV cache、任意 chunk continuation、训练 loss 和主要梯度。

### 2.2 先进开放权重模型提供的现实基线

以下结构口径截至 2026-08-10，来自官方仓库、模型卡或技术报告。Qwen3.8-Max 在该日期尚未公开完整 checkpoint 配置，因此只记录发布页已经披露的内容。

| 模型 | 总参数 / 激活参数 | 与 Tide 直接相关的结构事实 |
| --- | --- | --- |
| GLM-5.2 | 744B / 40B | 规则深度主干；256 个 routed experts，每 Token 选择 8 个，并有 shared expert；DSA 与 IndexShare |
| Kimi K3 | 2.8T / 104B | 93 层；896 个 experts，每 Token 选择 16 个，并有 2 个 shared experts；KDA、Gated MLA、AttnRes、Stable LatentMoE |
| DeepSeek-V4-Pro | 1.6T / 49B | 61 层；384 个 experts，每 Token 选择 6 个，并有 shared expert；混合 attention 主干 |
| DeepSeek-V4-Flash | 284B / 13B | 43 层；256 个 experts，每 Token 选择 6 个，并有 shared expert |
| Qwen3.8-Max | 2.4T / 95B | 发布口径为 Gated DeltaNet/full-attention 混合主干与 sparse MoE；完整配置待公开 |

这些模型的共同点不是采用了一般稀疏 Graph，而是：

1. 绝大多数计算仍组织为规则的深度 block stack。
2. MoE 在一个局部子层内选择少数专家并立即 merge 回共同 hidden/residual stream。
3. 下一层重新开放完整专家候选集合；上一次路由不会在拓扑上持续限制未来可达节点。
4. shared expert、dense 层、残差主干、负载均衡和成熟的 expert-parallel runtime 共同降低训练与部署风险。

MoE 已经经验性证明了一种重要平衡：总参数容量可以随专家数增加，而每 Token 只执行少数昂贵专家。但它仍有路由漂移、selected-only feedback、专家饥饿、负载不均衡和 all-to-all 通信成本。Tide 希望用层级局部候选和固定局部通信替代部分全局 dispatch，但这会放弃 MoE 的两项重要优势：每层重新开放全体候选，以及一次选择立即结束其显式路径身份。

因此，标准 MoE 必须成为 Tide 的主要对照组，而不是只与 dense Transformer 比较。

### 2.3 当前增量路径

从上述基线出发，当前实验只逐级增加一种主要能力：

| 阶段 | 相对上一阶段新增的能力 | 首要问题 |
| --- | --- | --- |
| 原生 dense checkpoint | 无；建立完整 oracle | 能否精确复现原模型 |
| 标准 MoE 对照 | 平铺候选中的 token-local Top-K 与立即 merge | 条件计算的成熟基线表现如何 |
| Leaf-Gated Tide | 末级兄弟集合局部选择、层次化递归结构、内部 node 常亮 | 局部候选和长短固定路径是否有收益 |
| Receiver-Gated Tide | 部分内部 node 也可不激活，因而一次选择可以裁剪后续子树 | 更强结构稀疏是否值得额外训练难度 |
| 可选状态化变体 | `broadcast-observe`、延迟私有状态或逐序列历史负载进入模型语义 | 跨 Token 潜伏语义能否稳定训练和高效执行 |

最后一行不是 Receiver-Gated Tide 的专属阶段。Leaf-Gated Tide 也可以让所有叶子收到信号并更新私有状态，只让被选叶子执行昂贵计算；反过来，Receiver-Gated Tide 也可以使用接近 MoE 的 `selected-dispatch`，不为未选接收者更新状态。门控层级、传播方式和状态化程度必须分别配置与消融。

第 3 节先说明两条路线如何分工；第 4、5 节再定义当前核心分支结构和三档近期架构。

## 3. 两条研究路线

两条路线服务于同一个总体目标，但承担不同职责：Graph 收缩线负责探索和约束设计空间，checkpoint 生长线负责形成可复现、可归因的真实实验。当前实验优先级明确偏向后者，但前者仍持续为后者提供设计边界。

### 3.1 Graph 收缩线：从一般机制空间提取约束

Graph 收缩线来源于 Tide/LH 最早对“局部通信 + 超稀疏”的一般 Graph 设想。它从表达力较强、机制相互混合的候选空间出发，逐步追问：哪些语义可以明确定义，哪些依赖可以并行，哪些 selector 或状态副作用会破坏 `prefill = decode`，以及哪些结构可能被收缩成可训练、可实现的模型。

可以用下列方向概括这一思考过程：

```text
一般 Graph / LH mechanism pool
-> 有限、依赖完整的 logical event DAG
-> 显式 allocator 的一般空间 DAG
-> HB-Lattice / HB-Sliced / HB-Line 等局部结构候选
-> 有界递归、固定 merge 的可实验分支族
```

这不是近期代码必须依次实现的工程流水线。Graph 收缩线当前主要产生四类结果：

1. **语义约束**：明确 reference transition、状态所有权、消息依赖、chunk 边界和 `prefill = decode` 的前提。
2. **反例与下界**：识别隐藏反向控制依赖、不可组合跨 Token selector 和 pointer-chasing 式路由，避免在注定无法通用低-span `prefill` 的机制上过早投入实现。
3. **设计坐标**：区分固定局部邻接、动态激活、allocator、fixed merge、路径寿命、私有状态和物理调度，使架构变化可以逐项讨论。
4. **候选收缩**：把过于一般的 Graph 自由度逐步压缩成固定 DAG、有界递归、局部 selector 和及时 merge 等可实验结构。

因此，它不是本仓库近期的实验主线，却直接帮助判断“哪些自由度值得进入实验”和“实验失败可能来自哪个结构因素”。本仓库不会先实现一般 Graph runtime；只在具体实验需要且理论边界清楚时，引入 Graph 收缩线得到的某项能力。

正式定义、正向定理与一般空间 DAG 见上游 [Tide 数学基础](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/tide-mathematical-foundations.md)；不可组合自适应路由的反向边界见 [Adaptive routing prefill lower bound](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/adaptive-routing-prefill-lower-bound.md)。

### 3.2 Checkpoint 生长线：本仓库近期实验主线

Checkpoint 生长线从已有预训练 Transformer/Mamba 出发：

```text
原生预训练模型
-> 完全等价装载
-> 函数保持的 residual branch
-> 标准 MoE 对照
-> 末级叶子受门控的 Leaf-Gated Tide
-> 有界递归分支
-> 内部节点也可受门控的 Receiver-Gated Tide
-> 可选的空间化与结构变异
```

这条路线负责复用完整 checkpoint 和成熟训练配方，每次只增加一种主要结构自由度，使质量、稳定性和性能变化可以被归因。后期可以删除节点、替换 kernel 或形成不再兼容原结构的后代模型，但必须保留清楚的 checkpoint 谱系。

更完整的 checkpoint 生长、递归 fixed merge 和训练风险讨论见上游 [Tide 模型架构与训练](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/tide-model-architecture-and-training.md)。

### 3.3 两条路线如何交换结果

Graph 收缩线可以用 correctness、复杂度或训练风险否决某些过强的 growth operator，也可以提出新的局部结构候选；checkpoint 生长线则用真实训练和系统实验检验这些约束是否有意义、是否过强，以及某项 Graph 自由度是否真正带来收益。

递归 fixed merge 分支是当前最可能的交界面，但两条路线不必最终得到同一个架构。它们可以只共享 selector scope、fixed merge、状态所有权和 chunk correctness 等契约，也可以长期保持“理论设计线”和“可部署实验线”的分工。

## 4. 当前核心结构：层次化递归并行分支

### 4.1 为什么从固定 merge 分支开始

考虑一个父模块，其输入和输出空间相同。父模块包含 always-on 主分支 $B_0$ 和有限个候选 residual 分支 $B_1,\ldots,B_N$。selector 为输入 $x$ 选择集合 $A(x)\subseteq\{1,\ldots,N\}$，固定 merge 为：

$$
T(x)
=
B_0(x)
+
\sum_{j\in A(x)}g_j(x)B_j(x).
$$

“固定”表示分支的入口、出口、merge 位置和 merge 算子在模型结构中预先确定；动态变化的只有激活集合和可选权重。短分支不能越过 merge 提前修改外层状态，长分支也不能在 merge 后追赶并改写同一个输出。

该结构具有四个近期优势：

1. 原 checkpoint 主路径可以完整保留。
2. 新分支可以零初始化或 clone-and-split，在初始点保持原函数。
3. 每个父模块只连接自己的有限子分支，逻辑 fan-out 有界。
4. 一次动态选择在固定 merge 处结束，控制寿命可知。

### 4.2 递归分支怎样自然表达长短路径

一个候选分支可以是：

- 单个 Attention、FFN、SSM 或 Linear Attention 模块。
- `Attention -> FFN` 等有限串联模块。
- 另一个满足单入口、单出口、固定 merge 契约的递归分支模块。

因此，同一个父模块可以具有不同串行深度的并行路径：

```text
短/长并行：

             +-> Attention --------------------+
input -------+                                  +-> fixed merge -> output
             +-> Attention -> FFN --------------+

等长并行：

             +-> Attention -> FFN --------------+
input -------+                                  +-> fixed merge -> output
             +-> Attention -> FFN --------------+
```

若其中一个 `Attention -> FFN` 分支内部再次使用同样结构，就得到层次化递归。长短路径来自模块串联深度不同，不需要引入跨 Token 的特殊 delay edge。物理执行仍在 fixed merge 等待本次所有已激活分支完成，因此这是静态有限 DAG 中的不等长路径，而不是“晚到分支在以后 Token 修改旧输出”。

### 4.3 为什么它具有局部通信倾向

每个 selector 只管理同一父模块的兄弟分支；每个子分支只与父入口、内部子节点和父 merge 通信。若递归最大深度、每级 fan-out 和 Top-K 都有固定上界，则逻辑连接度保持有界，分支不需要访问全局任意 expert。

这只证明**逻辑局部性**。要得到物理局部通信，还必须把父模块及其活跃子树放置在同一设备、相邻设备或固定 region，并测量真实通信字节和链路距离。若逻辑邻居被放在远端设备，层次化命名不会自动产生系统收益。

### 4.4 共享 selector

同一父模块分出的兄弟分支共享一套 selector、预算和 selector state。分支不能各自独立宣布自己激活，否则无法严格控制 Top-K、负载和总 active FLOPs。

第一版 selector 应为 token-local：

$$
A_t=R_\theta(h_t),
$$

它只依赖当前 Token 在当前父模块的表示和静态参数，不依赖物理 batch、实时设备负载或当前 chunk 的切分方式。历史负载进入 selector 的方案后置。

### 4.5 传播、状态更新、激活和发送是不同决策

标准 MoE 通常把“选中 expert、把 Token dispatch 给它、执行 expert、立即 merge”组织成一个紧密阶段。Tide 候选架构不必把下面四件事绑定在一起：

1. **消息可见性**：哪些静态下游可以看到当前信号。
2. **状态更新**：看到信号的 node 是否更新私有状态。
3. **昂贵激活**：哪些 node 执行 Attention/FFN/SSM readout 等重计算。
4. **继续发送**：哪些 node 产生输出并沿固定边继续传播或进入 fixed merge。

当前至少保留两种传播 profile：

| Profile | 消息与状态语义 | 与标准 MoE 的关系 |
| --- | --- | --- |
| `selected-dispatch` | 先选 active children；只有被选 children 收到输入、更新状态并执行 | 最接近普通 MoE dispatch，成本最低、状态语义最简单 |
| `broadcast-observe` | 已激活 sender 向全部静态 children 发送；每个 receiver 都更新状态，但只有 active receivers 执行重计算并继续发送 | 把“看到并记住”与“本次执行并传播”分开，状态成本和延迟信用更高 |

`broadcast-observe` 的最小流程是：

```text
active sender
-> broadcast to every declared static successor
-> every actual receiver observes and updates its state
-> one shared sibling selector chooses the active receivers
-> active receivers run expensive compute and emit
-> inactive receivers retain the updated state but emit nothing
```

该 profile 不只适用于内部 node 受门控的 Receiver-Gated Tide。它同样可以用于 Leaf-Gated Tide：例如一个 always-on 父模块下面有若干并行的 `Attention -> FFN` 叶支路，所有叶子都接收当前信号并写入各自 K/V 或其他轻量状态，但本次只有被选叶子执行 Attention readout、FFN 和 fixed-merge 输出。这样，一个暂时未激活的叶子以后仍可能利用此前收到的信号。

这里仍有两个独立选择：

- selector 可以读取 receiver 更新后的语义状态，也可以完全忽略该状态，只根据当前内容选择。
- receiver 可以维护私有持久状态，也可以让 `Update` 退化为无状态摘要、短期统计或恒等操作。

因此，“总是发送、收到即更新、但不一定激活”是一个可配置的消息/状态 profile，不是所有 Tide 模型必须采用的公理。它可能缓解未选 node 完全看不到数据的问题，但会增加状态写入、显存、跨 Token 依赖和延迟信用分配；必须与 `selected-dispatch` 做直接消融。

### 4.6 Selector 的候选输入与决策规则

令同一父模块的候选 child 集合为 $J$。对 $v\in J$ 和 Token 位置 $t$，可以分别构造：

- 当前内容摘要 $c_{v,t}$。
- 更新后的语义状态摘要 $q_{v,t}^{+}$，例如 KV/SSM/accumulator 的低维读出。
- 逐序列历史负载状态 $\ell_{v,t}$，例如激活计数、指数移动平均、距上次激活的间隔、恢复量或预算余额。
- 静态 node 信息 $\lambda_v$，例如 node id、层级、固定容量和模块类型。

一个一般 score 接口可以写成：

$$
s_{v,t}
=
G_\theta(c_{v,t},q_{v,t}^{+},\ell_{v,t},\lambda_v),
\qquad v\in J.
$$

具体 selector 不必读取全部四类输入。当前值得分别实验的 profile 是：

| Selector profile | 实际读取 | 主要作用与风险 |
| --- | --- | --- |
| 固定或 hash route | $t,\lambda_v$ | 最简单的稀疏与负载对照；不根据语义适配 |
| content-only | $c_{v,t},\lambda_v$ | 最接近标准 MoE router；是建议的第一版 learned selector |
| semantic-state-aware | $c_{v,t},q_{v,t}^{+},\lambda_v$ | 可依据累计信号、KV/SSM 状态或阈值决定激活；引入跨 Token 语义依赖 |
| load-aware | $c_{v,t},\ell_{v,t},\lambda_v$ | 可抑制长期过强或沉默 node；同一内容会随历史负载改变路由 |
| state-and-load-aware | 全部四类输入 | 同时考虑内容、局部记忆和恢复/预算；自由度最高，归因和 `prefill` 最困难 |

历史负载也有不同介入强度：可以只在语义分数接近时作为 tie-breaker，可以作为小幅 additive bias，可以在候选集内实施 quota，也可以直接禁止尚未恢复的 node。它一旦改变模型输出，就必须是逐序列隔离、可重放的 reference state；实时设备负载只能改变物理调度，不能进入这里的 $\ell_{v,t}$。

score 之后的决策规则也可以独立变化：

- soft weighted mixture，适合训练早期或作为 teacher。
- hard Top-K，真正跳过重计算，但有 selected-only feedback。
- 先按语义取得较大候选集，再按负载选最终 active set；较有利于保留语义匹配。
- 先按负载/恢复状态排除不可用 node，再在剩余集合按语义选择；负载约束更强，但可能牺牲语义匹配。
- 累计信号超过局部阈值才激活；适合与 `broadcast-observe` 配合，但会增加状态递推和信用距离。

这些 profile 目前只是候选设计空间，不是已证明的优势。分离消息可见性、状态更新和昂贵激活，确实提供了标准 MoE 一次性 dispatch 之外的模型选择；同时也增加了必须通过 correctness、训练和系统实验逐项排除的失败方式。

## 5. 三档近期架构

`Leaf-Gated` 与 `Receiver-Gated` 描述的是**哪些拓扑层级允许被门控**，不是“保守程度”的价值判断，也不决定使用 `selected-dispatch` 还是 `broadcast-observe`。

为避免名称暗中绑定其他机制，每个具体模型至少用下面五个坐标描述：

$$
\mathcal C
=
(\text{门控范围},\ \text{传播 profile},\ \text{状态生命周期},
\ \text{selector 输入},\ \text{selector 决策}).
$$

例如，第一版实验配置是：

```text
(Leaf-Gated, selected-dispatch, no-delayed-private-state, content-only, hard-Top-K)
```

而“所有叶子都看到并记录当前信号，但只激活少数叶子”可以写成：

```text
(Leaf-Gated, broadcast-observe, persistent-private-state, content-only, hard-Top-K)
```

第二个配置仍然是 Leaf-Gated Tide，因为受门控的拓扑层级没有变化；它只是采用了不同的传播与状态语义。若 selector 还读取历史负载，则第四项独立改为 `load-aware` 或 `state-and-load-aware`；若从 hard Top-K 改为 soft mixture，则只改变第五项。

### 5.1 标准 MoE 对照

标准 MoE 在一个子层内完成：

```text
common hidden
-> router over every expert in this MoE sublayer
-> Top-K experts
-> weighted merge
-> common hidden
```

它的关键性质是：

- expert 一般只在一个子层内保持独立路径身份。
- merge 后只保留专家输出造成的数值语义影响。
- 下一层重新面对完整 expert 集合。
- 标准 FFN experts 通常没有独立、可延迟读出的单序列持久状态。

MoE 的选择仍会改变当前 hidden，因而可能影响下一层路由，也可以通过后续 Attention 影响未来 Token。立即 merge 删除的不是语义影响，而是持续存在的显式控制路径身份。

### 5.2 Leaf-Gated Tide：只有末级叶子受门控

Leaf-Gated Tide 满足：

1. 原模型主路径和所有非叶 node 始终激活。
2. 只有递归结构最末一级的叶 node 可以被 selector 跳过。
3. 叶输出在父模块出口立即 fixed merge。
4. merge 后下一段重新开放完整候选集合。
5. 一次叶选择不限制更远层的拓扑可达集合。

```text
always-on internal hierarchy
-> sparse terminal leaves
-> immediate fixed merge
-> common representation
```

它与标准 MoE 的主要区别可以集中在：

- 候选集合按父子层级组织，而不是一个平铺全局 expert 集合。
- selector 只在各末级兄弟集合内局部选择；内部父节点仍全部常亮，不在本级裁剪激活子树。
- 分支可具有不同的有限串行深度。
- 逻辑拓扑更适合映射为固定局部通信。

Leaf-Gated Tide 可以采用第 4.5 节的任一传播 profile：

- 使用 `selected-dispatch` 时，selector 先选叶子，只有被选叶子接收输入、更新状态、执行并进入 fixed merge。这是第一版实验采用的最小语义。
- 使用 `broadcast-observe` 时，父模块把信号发给全部静态叶子；所有叶子执行预先声明的轻量状态更新，但只有被选叶子执行昂贵计算、产生输出并进入 fixed merge。

例如，若末级有四条并行 `Attention -> FFN` 叶支路，`broadcast-observe` 可以让四条支路都写入各自的 K/V 或其他私有状态，而只让 Top-K 支路执行 Attention readout、FFN 和输出投影。该机制是否值得其状态写入、显存和训练成本，不由 Leaf-Gated 定义保证，必须与 `selected-dispatch` 直接比较。

#### 与 MoE 信用距离近似等价的必要限定

“只有叶子稀疏”本身不足以推出信用分配与标准 MoE 等价。还必须让比较双方具有相同的状态语义，并至少满足：

1. 叶分支在本次 fixed merge 后不继续保留显式路径身份。
2. 叶分支没有以后 Token 才读出的私有持久状态，或者 MoE 对照也具有完全相同的状态更新与读出规则。
3. 下一段候选集合不受本次叶选择的拓扑限制。

在这些条件下，二者新增的**路由专属控制寿命**都在局部 merge 处结束；最终 loss 到早期 expert/leaf 的普通深层梯度仍然可能很长，但这与一般深网络相同。

若末级叶子在未激活时仍写入私有 KV/SSM 状态，并在若干 Token 后激活读出，那么即使所有非叶 node 常亮，也已经新增一条标准无状态 MoE 没有的跨 Token 状态信用链。该变体应以完整五元配置单独记录和实验，不能归入“MoE 信用距离等价”的最小版本。

### 5.3 Receiver-Gated Tide：内部接收者也可受门控

Receiver-Gated Tide 只比 Leaf-Gated Tide 多改变一个坐标：部分非叶 node 也允许不执行昂贵计算和不继续发送，因此一次局部选择可以裁剪更深的后续子树。固定空间结构仍是有限 DAG；每个父模块仍只管理自己的有限兄弟集合，而不是把 Token 动态发送到任意全局 node。

对一个父模块的静态 child 集合 $J$，两种传播 profile 给出不同的局部流程。

`selected-dispatch` 先选择、后发送：

```text
parent output and pre-existing child summaries
-> shared selector chooses A_t subset of J
-> only children in A_t receive and update
-> those children run expensive compute and continue
```

`broadcast-observe` 先发送和更新、后选择昂贵激活：

```text
active parent broadcasts to every child in J
-> every child receives and applies its declared Update
-> shared selector chooses A_t subset of J
-> only children in A_t run expensive compute and continue
```

第二种流程允许 selector 读取各 child 更新后的语义状态摘要；也可以让 selector 忽略这些摘要，只使用当前内容。两种流程都可以使用或不使用逐序列历史负载。因而，内部 node 受门控并不推出“收到即更新”，`broadcast-observe` 也不推出 state-aware 或 load-aware selector。

若采用 `broadcast-observe`，对 node $v$ 在 Token 位置 $t$，令 $M_{v,t}$ 为实际收到的消息集合，$S_{v,t}$ 为其单序列私有状态，则轻量更新和昂贵计算分别写成：

$$
S_{v,t}^{+}=U_v(S_{v,t},M_{v,t}),
$$

$$
a_{v,t}\in\{0,1\},
\qquad
a_{v,t}=1
\Longrightarrow
y_{v,t}=F_v(S_{v,t}^{+},M_{v,t}).
$$

只有 $a_{v,t}=1$ 的 node 产生 $y_{v,t}$ 并继续发送；$a_{v,t}=0$ 的 node 只保留 $S_{v,t}^{+}$。这里 $U_v$ 可以是 KV 写入、SSM accumulator 更新、有限窗口更新、轻量摘要更新，也可以退化为无持久效果的恒等操作。它不是必须执行完整 Attention/FFN 的另一个名称。

若 $M_{v,t}$ 为空，状态是保持、执行 decay 还是进行其他空步转移，必须由模型配置预先声明；物理 runtime 不能根据设备是否繁忙临时改变该规则。多个上游消息的聚合顺序或并列聚合算子也必须属于确定的 reference semantics。

模型记录至少区分三类对象：semantic state 包含 KV、SSM accumulator 或其他神经记忆；load state 包含真正进入 selector 语义的逐序列计数、恢复量或预算；runtime record 包含设备队列、显存占用和实时硬件负载。前两者都可以选择不启用，但第三类绝不能作为改变模型输出的隐式 selector 输入。

#### `broadcast-observe` 缓解了什么

无论它用于 Leaf-Gated 还是 Receiver-Gated Tide，实际收到消息的 node 即使本次未执行昂贵分支，也可以看到数据并积累私有状态。这可能缓解 selected-only data exposure，并允许早期信号以后重新影响输出或 selector；但它不会自动解决所有饥饿：

| 问题 | 收到即更新是否自动解决 |
| --- | --- |
| 状态/数据饥饿 | 对实际收到消息的 node 有所缓解 |
| 激活饥饿 | 不一定；node 仍可能长期不执行昂贵计算 |
| 梯度饥饿 | 不一定；延迟状态若未影响 loss 或被 detach，仍无有效梯度 |
| 语义饥饿 | 不一定；输入分布频繁漂移仍可能无法形成专门化 |
| 没有活跃前驱导致的无消息 | 不解决 |

`Observe / Update` 也不是免费操作。若 $U_v$ 接近完整 Attention/FFN 的成本，跳过 $F_v$ 可能没有系统收益。实验必须把消息传输、状态写入、昂贵激活和继续发送分别计费。

#### 两类新增信用距离

Receiver-Gated 的**内部层级门控**可以让一次离散选择在多个 node 之间持续限制后续可达子树，直到固定 merge；即使 node 完全无持久状态，这条控制信用距离也可能比立即 merge 的 MoE 更长。

与此正交，任一门控范围都可以引入**延迟状态读写**。设 Token 位置 $t_A<t_B$，位置 $t_A$ 的信号写入 node $v$ 的状态，而该状态直到位置 $t_B$ 才参与昂贵计算：

$$
\text{write at }t_A
\to
S_{v,t_A}^{+}
\to
S_{v,t_A+1}
\to\cdots\to
S_{v,t_B}
\to
\text{read at }t_B
\to
\mathcal L.
$$

这条链在 `Leaf-Gated + broadcast-observe + persistent-private-state` 中已经存在，不需要内部 node 受门控。训练必须判断早期写入、后期读出和中间状态更新各自对 loss 的贡献；若采用截断反向传播或跨 chunk detach，早期写入甚至可能收不到后续梯度。

Receiver-Gated 与延迟私有状态同时启用时，控制距离、状态写入到读出的跨 Token 距离以及多次状态混合后的归因难度会叠加。实验不能同时加入这两个自由度后再把变化笼统归因于“Receiver-Gated”。

固定空间 DAG 仍允许按拓扑序为每个 node 收集完整 chunk inbox，并让每个 node 对整个 chunk 调用一次或固定次数；因此空间遍历次数可以不随 chunk 长度增长。但如果 $U_v$ 或 selector 是任意逐 Token 状态递推，node 内部仍可能顺序处理 Token。只有当该递推具有 scan、causal bulk 或其他已验证的组合结构时，才能进一步得到 Transformer/Mamba 级的 node 内 Token 并行。correctness、空间常数遍历和 node 内低 span 是三项不同结论。

### 5.4 三种信用距离必须分开

| 距离 | 含义 | 标准 MoE | Leaf-Gated Tide | Receiver-Gated Tide |
| --- | --- | --- | --- | --- |
| 数值语义距离 | 中间数值贡献到最终 loss 经过的深层计算 | 可以很长 | 可以很长 | 可以很长 |
| 控制寿命 | 离散选择保持独立路径身份并限制未来候选的深度 | 通常一个子层 | 在 fixed merge 条件下近似一个局部段 | 可跨多个 node/切片 |
| 状态读写延迟 | 私有状态写入到以后 Token 读出的间隔 | 标准 FFN MoE 通常没有 | 取决于传播 profile 与状态生命周期 | 同样取决于传播 profile 与状态生命周期 |

always-on backbone 可以为模型保留短而连续的主梯度路径，但不能保证稀疏分支本身获得足够梯度。固定 merge 可以限制控制寿命，但不会删除已经写入 node state 的潜在语义影响。

## 6. 粗略实验设计

### 6.1 总原则

1. 每一阶段只增加一个主要自由度。
2. 总参数、active parameters、active FLOPs、训练 Token、优化器和数据尽量匹配。
3. 每个新设计同时与源 dense checkpoint、dense 扩展和标准 MoE 比较。
4. correctness、训练质量与系统性能使用三套独立 gate。
5. selector 的模型语义不得依赖 batch 组成、chunk 切分或实时设备负载。
6. 第一批实验优先在 FFN 分支上进行；Attention/SSM 分支和私有持久状态后置。
7. 每个实验完整记录门控范围、传播 profile、状态生命周期、selector 输入和 selector 决策；不得只写一个 Tide 类名。

FFN-first 的原因是标准 MoE 对照成熟、FFN 没有独立跨 Token cache、active FLOPs 更容易匹配。确认层次化 selector 与递归分支本身有效后，再把叶模块扩展到 Attention、`Attention -> FFN`、SSM 或 Linear Attention。

### 6.2 阶段 E0：原生 checkpoint 基线

选择一个 pre-norm、decoder-only、开放权重且规模足以快速重复实验的模型。首先实现或包装原生模型，不增加任何 Tide 分支。

必须验证：

- state-dict 参数逐项覆盖。
- logits 与逐层主要 artifact 对齐。
- 单 Token `decode` 与多 Token `prefill` 对齐。
- 任意 chunk 切分的 continuation state 对齐。
- 训练 loss 和主要参数梯度对齐。
- 固定随机种子下结果可重复。

如果 E0 不能稳定复现，后续所有架构比较都没有可信基线。

### 6.3 阶段 E1：标准 MoE 对照

在一个局部 FFN 子层或新增 residual branch 上建立 flat MoE：

- 固定数量 experts。
- token-local Top-K router。
- 立即 weighted merge。
- 无 expert 私有跨 Token state。
- 可配置 shared expert 或 always-on residual branch。

至少包含两种初始化：

1. 零 residual/零输出投影，初始函数等于原模型。
2. clone-and-split，把旧模块贡献拆给多个副本并保持总和。

E1 的作用不是证明 MoE 最优，而是建立成熟条件计算对照，测量普通 routing drift、负载、梯度覆盖和系统开销。

### 6.4 阶段 E2：Leaf-Gated Tide

保持 E1 的 expert 数、active 数、active FLOPs 和 merge 位置，主要改变：

- 把平铺 experts 组织成有界 fan-out 的层次化兄弟分支。
- 内部父模块始终执行；每个末级父模块只用一套共享 selector 在自己的叶子子节点中选择。
- 所有非叶 node 常亮，只有末级叶子稀疏。
- 每个局部分支在固定出口立即 merge。
- 固定使用等深 FFN 叶子、`selected-dispatch`、content-only selector 和无延迟私有状态。

这一阶段应回答：

1. 层次化局部候选是否能达到 flat MoE 相同或更好的质量/计算折中？
2. 分级 selector 是否更容易或更难负载均衡？
3. 逻辑局部性是否能映射成真实通信收益？
4. route churn 与边界跳变量是否高于标准 MoE？

### 6.5 阶段 E3：Leaf-Gated 的结构与传播消融

E3 仍保持所有非叶 node 常亮，只研究 Leaf-Gated 内部的自由度，并分成先后两个子阶段。

**E3a：有界递归与长短路径。** 在保持 `selected-dispatch`、content-only 和无延迟私有状态不变时，逐项加入：

1. 一层递归。
2. 两层递归。
3. 同深度并行分支。
4. 不同深度的短/长分支。
5. 不同但无跨 Token 私有状态的 atomic/serial 分支 grammar。

每次只改变一种分支 grammar，并记录最大 fan-out、最大递归深度、Top-K、最长串行路径、固定 merge 位置和每 Token 实际激活子树。

**E3b：传播与叶状态。** 固定一项已验证的拓扑和 content-only selector，再比较：

| 配置 | 目的 |
| --- | --- |
| `selected-dispatch` + 无延迟私有状态 | E2/E3a 语义基线 |
| `broadcast-observe` + 无延迟读出 | 单独测量广播、轻量更新和 packing 成本；未选更新不改变以后输出 |
| `selected-dispatch` + 持久私有状态 | 测量 selected-only state exposure |
| `broadcast-observe` + 持久私有状态 | 测量所有叶子都看到数据能否改善训练，及其延迟信用成本 |

`Attention -> FFN`、SSM 或 Linear Attention 叶支路应在 E3b 才进入，因为它们的 KV/accumulator 生命周期必须与传播 profile 一起定义。以四条并行 `Attention -> FFN` 叶支路为例，可以直接比较“只有 Top-K 叶子写 K/V”与“四条叶子都写 K/V、只有 Top-K 做 readout 和 FFN”。

E3b 仍不让 selector 读取更新后的语义状态或历史负载；否则无法区分传播/状态语义和 selector 变化各自的影响。

### 6.6 阶段 E4：Receiver-Gated Tide

在 E2/E3 已经给出正面证据后，才把允许门控的范围从末级叶子扩展到部分非叶 node。首先使用：

```text
(Receiver-Gated, selected-dispatch, no-delayed-private-state, content-only, hard-Top-K)
```

它与 E3 中拓扑、active FLOPs 和 selector 规模匹配的 Leaf-Gated 配置直接比较，以单独测量“内部选择裁剪后续子树”的收益和控制信用距离。

随后才在同一 Receiver-Gated 拓扑上复用 E3b 已定义的传播/状态组合：

- **E4a：**`selected-dispatch` 与无延迟私有状态；隔离非叶门控本身。
- **E4b：**`broadcast-observe`，但更新不在以后 Token 读出；隔离内部广播与更新成本。
- **E4c：**持久 KV/SSM/accumulator；仅在对应 Leaf-Gated 状态变体已有可解释结果后加入。

每个 E4 变体都必须有传播 profile、状态生命周期、selector 输入和 selector 决策完全相同的 Leaf-Gated 对照。E4c 还需要明确 BPTT、detach、chunk boundary 和状态生命周期策略。

### 6.7 阶段 E5：语义状态与历史负载进入 selector

最后才改变 selector 输入。固定已验证的门控范围、传播 profile 和状态生命周期，以 content-only 为共同基线，分别加入：

1. 更新后 semantic-state summary，但不读取历史负载。
2. 历史负载只作为语义分数近似并列时的 tie-breaker。
3. 小幅 load bias、局部 quota 或 recovery eligibility。
4. semantic state 与 load state 联合输入。
5. 累计信号阈值或其他显式状态递推。

soft mixture、hard Top-K、语义候选后负载筛选和负载准入后语义筛选也应分别记录，不能统称为“stateful selector”。若状态递推可表示为 scan 或 causal bulk，应优先使用该实现；只有实验证据足够时才接受 node 内任意逐 Token sequential selector。

物理设备负载只能改变调度和放置，不能改变模型路由 artifact。

## 7. 对照组与消融矩阵

每个主要实验至少保留：

| 对照 | 作用 |
| --- | --- |
| 原 checkpoint continued pretraining | 判断新增结构是否优于继续训练原模型 |
| 等参数 dense 扩展 | 判断收益是否只来自更多参数 |
| 等 active-FLOPs flat MoE | 判断层次化局部结构是否优于成熟稀疏基线 |
| fixed/hash route | 判断 learned selector 是否真正有价值 |
| Leaf-Gated Tide | 作为 Receiver-Gated Tide 的直接对照 |

建议逐项消融以下变量，而不是一次全部组合：

- flat vs hierarchical candidate set。
- equal-depth vs mixed-depth branches。
- leaf-only gating vs internal-node gating。
- `selected-dispatch` vs `broadcast-observe`。
- 无延迟状态 vs selected-only persistent state vs receive-always persistent state。
- content-only vs semantic-state-aware vs load-aware selector 输入。
- soft mixture vs hard Top-K vs 两级候选/负载决策规则。
- shared expert/backbone 比例。
- fixed merge 间隔和最大控制寿命。

每个实验表至少把下列坐标列成独立字段：branch grammar、门控范围、传播 profile、状态更新函数与生命周期、selector 输入、selector 决策规则。若两个配置同时改变其中多个字段，它只能作为组合实验，不能替代单轴消融。

## 8. 验收指标

### 8.1 Correctness gate

- 初始 checkpoint 参数、logits、cache/state 与梯度对齐。
- `prefill`、逐 Token `decode` 和任意 chunk continuation artifact equality。
- 不同 batch 组合不改变单序列输出。
- 不同物理调度不改变 route artifact 和模型状态。
- hard/soft selector 在训练与推理模式下的语义明确。
- 传播 profile、状态生命周期和 selector 输入均写入 checkpoint/config，恢复后不发生隐式变化。

### 8.2 训练 gate

- train/validation loss、perplexity 和下游质量。
- route churn：同一输入跨 checkpoint 的 active-set 变化。
- active-set overlap 与分支输出跳变量。
- 每个 node 的消息接收次数、状态更新次数、昂贵激活次数、继续发送次数和有效梯度次数；五者不得合并成一个“使用次数”。
- 梯度范数、梯度覆盖率和长期未更新参数比例。
- node 输入分布漂移与状态数值稳定性。
- 状态写入到读出的 Token 延迟分布。
- 按 selector input profile 分组的负载分布、route churn 和状态读写延迟。
- 短路径、长路径和 always-on backbone 的贡献消融。

激活均衡不等于训练均衡。节点即使被均匀激活，也可能没有有效梯度或没有稳定语义分布。

### 8.3 系统 gate

- 总参数、active parameters 和实际 active FLOPs。
- 消息投递、`Observe / Update`、selector、昂贵激活、继续发送、packing 与 merge 的单独成本。
- 端到端训练吞吐、`prefill` 吞吐和 `decode` latency。
- 峰值显存、KV/SSM state 大小和 optimizer state。
- 跨设备通信字节、通信邻接距离和 collective 时间。
- grouped GEMM/packed kernel 的 tile 利用率。
- 每设备工作量分布、尾部延迟和空闲比例。

局部 Graph 只有在物理放置也局部、并且跳过工作大于控制与状态成本时，才获得真实系统收益。

## 9. 第一阶段建议的软件边界

初始实现应优先建立少量稳定抽象，而不是把一般 Graph runtime 一次做完：

```text
CheckpointAdapter
    原生模型装载、状态映射和 equality oracle

BranchModule
    单入口、单出口；可为 atomic、serial 或 recursive

PropagationProfile
    明确 selected-dispatch 或 broadcast-observe，并产生 receiver mask

SiblingSelector
    为同一父模块输出 active child ids 和 weights

StateUpdater
    声明 semantic/load state 的更新函数、持久范围和空输入行为

FixedMerge
    固定槽位与固定 merge 算子

NodeState
    semantic state 与 load state；不得混入物理 runtime state

RouteArtifact
    记录每个 Token、父模块和递归层级的 receiver、update、active 与 emit mask

ExperimentLedger
    记录模型谱系、配置、参数/FLOPs、数据、checkpoint 和指标
```

建议的仓库布局可以从下面开始，具体代码框架在新开发线程中确定：

```text
tide/
├── README.md
├── pyproject.toml
├── configs/
├── src/tide/
│   ├── checkpoint/
│   ├── branches/
│   ├── selectors/
│   ├── states/
│   ├── models/
│   └── instrumentation/
├── tests/
│   ├── equality/
│   ├── routing/
│   └── continuation/
├── experiments/
└── docs/
```

第一版不需要实现 HB-Line executor、一般 event IR、跨设备 allocator 或有环 Graph。

## 10. 第一个可交付里程碑

第一个里程碑不是“训练出完整 Tide”，而是建立可信实验底座：

1. 选定一个小型 pre-norm decoder-only 开放权重 checkpoint。
2. 完成 E0 原生复现与 equality tests。
3. 实现统一 `BranchModule + PropagationProfile + SiblingSelector + StateUpdater + FixedMerge` 接口；第一版只启用 `selected-dispatch` 和无延迟状态。
4. 实现一个 FFN flat MoE 对照。
5. 实现一个只有末级叶子受门控、无私有延迟状态的 Leaf-Gated Tide。
6. 在同一训练数据和 active-FLOPs 下完成短程 continued-pretraining 对比。
7. 输出 correctness、质量、路由、梯度覆盖和系统成本报告。

只有该里程碑通过，才进入递归长短路径和 Receiver-Gated Tide。

## 11. 远期展望

HB-Line/HB-Lattice 可以把递归或局部分支映射到重复空间切片，使 node 参数和状态长期驻留在局部设备，并通过有界邻接形成更一般的局部计算介质。一般空间 DAG 进一步允许不等长路径、显式 allocator 和拓扑序 chunk 执行；一般 Graph 还可能包含反馈、动态 event DAG 和更强状态机制。

这些方向需要额外解决：

- 路径持续限制未来可达集合。
- 跨 Token 控制链与低-span `prefill` 的冲突。
- 边界在途消息和状态 continuation。
- 长路径信用分配与路径分布漂移。
- 局部设备放置、通信和负载热点。

近期代码不为这些远期目标预先加入复杂机制；当实验给出需要它们的证据时再扩展。

## 12. 当前不能主张的结论

- 局部通信 Graph 已经优于 Transformer、Mamba 或 MoE。
- 人脑结构证明了 Tide 可训练或高效。
- 收到即更新自动消除了节点饥饿和信用分配问题。
- 只有叶子稀疏就无条件与 MoE 具有相同信用距离。
- 固定空间 DAG 自动得到 Transformer/Mamba 级 node 内 Token 并行。
- 任意 stateful selector 都可以获得高性能 chunk `prefill`。
- 两条研究路线必然汇合。

## 13. 新开发线程的起点

新线程应先完成以下决策，不要直接实现一般 Graph：

1. 选择首个 checkpoint、训练框架与目标硬件。
2. 明确 E0 equality oracle 的逐项比较对象。
3. 选定 FFN branch 的函数保持初始化方式。
4. 定义 flat MoE 与 Leaf-Gated Tide 的 matched-compute 配置。
5. 定义 `BranchModule`、`PropagationProfile`、`SiblingSelector`、`StateUpdater`、`FixedMerge` 和 `RouteArtifact` 的最小接口。
6. 建立第一批单元测试和最小 continued-pretraining 实验。

后续任何架构自由度，都应说明它解决了哪一个已观察问题，以及用哪个对照和指标证伪。

## 14. 历史动机与研究背景

本节不属于理解近期实验的前置知识。它说明 Tide 为何曾从一般 Graph 出发、为何后来收缩到当前增量路线，以及人脑调查为哪些设计提供了启发。不了解 LH 的读者可以跳过本节；需要细节时再阅读第 15 节链接的上游笔记。

### 14.1 从 LH 到 Tide

Tide 的早期动机来自 LH。LH 尝试把模型组织为局部连接的空间 Graph，并通过局部 selector 形成很强的时间和结构稀疏：

- 已激活节点向局部下游发送信号。
- 下游节点接收、记忆或更新状态。
- 局部节点竞争有限激活预算。
- 历史激活次数、恢复状态或阈值影响后续激活。
- 主路径或 hub 可以保持常亮。

这类机制适合 streaming/decode：一个 Token 进入后，信号在 Graph 中经过多轮传播，节点和边可以并行工作。但早期 LH selector 会读取一个区域内多个节点的当前状态，又反过来影响同一区域内多个节点；不同 Token 的信号还会在 Graph 内交错。结果是显式空间 Graph 即使无环，也会被 selector 引入隐式的跨 Token 控制依赖链。

如果 Token $t+1$ 的路由必须等待 Token $t$ 在很晚的 Graph 阶段产生控制结果，而这种依赖对每个 Token 重复出现，那么控制链长度会随 chunk 长度增长。除非该递推具有 scan、bulk 或其他可组合代数结构，否则不能期望获得 Transformer/Mamba 意义上的高性能 chunk `prefill`。

因此，Tide 不以完整复刻 LH 为目标。LH 的价值是提供“局部通信 + 超稀疏”的历史动机，暴露 selector、状态副作用和动态路径可能带来的困难，并提供可供裁剪、替换和重新组合的机制集合。当前实验保留局部性、层级和稀疏激活作为近期结构约束；把状态积累、历史负载和累计阈值拆成可选机制，并逐项后置验证会形成未收缩跨 Token 控制链的组合。

### 14.2 从 `prefill = decode` 得到的约束

编译器、ISA、乱序执行和 dataflow 的共同经验是：性能优化必须在明确 reference semantics 的前提下重排、融合或并行计算。对任意一次有限 chunk 执行，数据、状态和控制依赖应能展开成一个有限、依赖完整的 logical event DAG；实现可以改变物理调度，但不能改变该 DAG 所定义的可观察结果。

这给 Tide 两条直接约束：

- 固定空间 Graph 可以是 DAG，节点可以按拓扑序处理整个 chunk；动态物理调度不应进入模型语义。
- 任意 pointer-chasing 式、不可组合的自适应路由链，不存在对所有实例都有效的通用低深度 exact `prefill` 加速。

近期实验因此优先使用固定 merge、有界递归和 token-local selector。状态化 selector 与长期不收拢路径后置。

### 14.3 人脑调查提供的启发与边界

人脑不是 Tide 的实现模板，也没有已知的 Transformer 式高性能 `prefill`。但神经解剖与神经生理调查提供了若干有价值的结构倾向：

| 观察 | 对 Tide 的启发 | 不能推出什么 |
| --- | --- | --- |
| 神经连接高度局部且结构稀疏，同时存在少量长程投射 | 研究有界度局部 Graph、层级连接和多尺度 backbone | 局部 Graph 一定比当前加速卡上的稠密模型更快 |
| 信号广泛发散、汇聚，并存在大量反馈 | 允许并行分支、固定 merge 和多种长短计算路径 | 应在第一版复制有环、跨 Token 反馈 |
| 神经元持续积累输入，只在部分时间产生 spike | 把 `Observe / Update` 与昂贵 `Activate / Emit` 分开 | 粗粒度计算 node 等价于单神经元 |
| 局部抑制、恢复和稳态机制限制长期过强或沉默 | 研究局部预算、恢复量和慢速负载状态 | 简单按历史次数轮换就能形成有用专门化 |
| 大尺度功能网络多表现为增益和耦合变化，而非整个脑区严格开关 | 保留 always-on backbone、重叠分支和短 merge | hard selector 必然不可用 |
| 学习依赖局部可塑性、调质、重放和多时间尺度机制 | 关注局部辅助信号和延迟信用 | 人脑已经解决了数字网络中的精确全局信用分配 |

最重要的边界是：微观 spike 稀疏和平滑的群体表示，并不自动迁移到粗粒度 node 的 0/1 激活。一个 Tide node 可能包含完整 Attention、FFN 或 SSM，开关它造成的语义跳变远大于单个神经元。增加 node 数量也不会自动得到人脑式平滑性；还需要重叠表示、residual、固定 merge、短控制寿命和合适的训练机制。

## 15. 上游研究笔记与外部参考

本启动文档整理自 ObsidianVault 中更完整的研究笔记；正式理论成熟后再选择性复制到本仓库。后续 agent 若需要一般定义、证明、历史或实现背景，应从下列 GitHub 文档读取，而不是依赖本机绝对路径：

- [Tide 研究线总入口](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/README.md)：战略路线、文档地图、当前主张边界与阅读顺序。
- [Tide 模型架构与训练](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/tide-model-architecture-and-training.md)：checkpoint 生长、递归分支、HB-Sliced/HB-Line、selector 与训练稳定性。
- [Tide 数学基础](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/tide-mathematical-foundations.md)：`StepTransition`、`prefill = decode`、logical event DAG、一般空间 DAG 与函数保持生长。
- [Adaptive routing prefill lower bound](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/adaptive-routing-prefill-lower-bound.md)：不可组合自适应路由链的反向复杂度边界。
- [Tide 背景、历史谱系与参考](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/tide-background-history-and-references.md)：ISA/编译器/dataflow 谱系和完整人脑传播调查。
- [Tide runtime 验证与状态](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/tide-runtime-validation-and-status.md)：runtime contract、LH 映射、artifact equality 与工程状态。
- [Tide 统计力学与信息动力学](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/tide-statistical-mechanics-and-information-dynamics.md)：粗粒化、路径相关性和统计力学类比及其边界。

先进模型与 MoE：

- [GLM-5.2 official repository](https://github.com/zai-org/GLM-5)
- [Kimi K3 official repository and report](https://github.com/MoonshotAI/Kimi-K3)
- [DeepSeek transparency center](https://www.deepseek.com/transparency/)
- [DeepSeek-V4 technical report](https://arxiv.org/abs/2606.19348)
- [Qwen3.8 official release blog](https://qwen.ai/blog?id=qwen3.8)
- [StableMoE](https://arxiv.org/abs/2204.08396)
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066)
- [Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2408.15664)
- [OLMoE](https://arxiv.org/abs/2409.02060)

人脑结构、动态网络与信用分配：

- [The Neocortical Circuit: Themes and Variations](https://doi.org/10.1038/nn.3917)
- [Distributed Hierarchical Processing in the Primate Cerebral Cortex](https://doi.org/10.1093/cercor/1.1.1-a)
- [Brain Networks and Cognitive Architectures](https://doi.org/10.1016/j.neuron.2015.09.027)
- [Neuromodulatory Influences on Integration and Segregation in the Brain](https://doi.org/10.1016/j.tics.2019.04.002)
- [Eligibility Traces and Plasticity on Behavioral Time Scales](https://doi.org/10.3389/fncir.2018.00053)
- [Backpropagation and the Brain](https://doi.org/10.1038/s41583-020-0277-3)
- [Homeostatic Synaptic Plasticity](https://doi.org/10.1101/cshperspect.a005736)
