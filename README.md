# TIDE Checkpoint 生长实验仓库启动文档

> 状态：项目启动草案
>
> 日期：2026-08-20
>
> 目标读者：即将开始实现、训练与验证 Tide 的研究和工程人员
>
> 仓库定位：以 checkpoint 生长线为当前实验主线；本文同时保存一眼可读的研究逻辑纲要和可执行的实验政策，不要求读者先跳转到外部笔记才能理解为什么做这些实验。
>
> 当前实验入口：本 README 是 `fractal-latcarf` 分支当前工作流、配置坐标、验收 gate 与首个交付的概念入口；ObsidianVault 只沉淀可泛化的研究动机、正式理论和跨实验结论，不镜像这里的易变政策。
>
> 上游研究总入口：[ObsidianVault / TIDE](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/README.md)

快速阅读：只想一眼看懂研究理由，读第 1.1、1.2 节；想检查完整推导，继续读第 2.3 节；准备实现和训练，直接进入第 6 至第 10 节。

## 1. 研究逻辑总览

TIDE 的完整项目表述是：`TIDE: A Topology-Invariant Degree-bounded Expansion Architecture for Autoregressive Token Inference`。

中文表述是：**TIDE：面向自回归 Token 推理的拓扑固定、度有界容量扩展架构。**

本仓库以 checkpoint 生长线为当前主线，研究 TIDE Architecture、训练语义和可执行实验。

Tide 当前更具体的结构要求是：一个模型实例的空间图拓扑在执行中保持不变，单个节点的资源成本不随模型总容量增长，同时整个模型可达、可利用的容量能够继续增长。这里的“拓扑固定”只表示节点和边不随 Token 或 selector 临时改变，不表示节点的度天然有界；扩大模型时仍可以在同一规则下加入更多节点和边。

本文把一个能够接收上游消息、并拥有自身参数或状态的下游模块称为 `receiver`；把 Attention readout、FFN、大型 SSM 更新等主体计算称为“昂贵计算”，以区别于较轻的消息接收和状态更新。

> **当前状态：** Flat MoE 已经提供了可靠的外部正面证据；本仓库自身尚无可靠训练结果。`broadcast-observe`、私有状态、层次递归和多次局部选择是否有效，都仍需实验回答。

### 1.1 一眼看懂的两条逻辑链

第一条回答：**为什么 Tide 不能依靠一步平铺访问不断增长的容量？**

```text
Flat MoE 已证明：大量参数可以与每 Token 少量计算同时存在
    ↓
Tide 还要求同时保持三件事：
空间图拓扑在执行中固定、单节点成本不随总容量增长、可达总容量继续增长
    ↓
在入口数量和节点连接度都有上界时，一步只能接触固定数量的节点；
要让越来越多的容量保持可达，就必须经过更多跳或扩展到更大的空间范围
    ↓
如果每个 Token 同时还只能执行极少模块，
就需要在传播过程中做多次局部选择，并限制总激活量
    ↓
多级选择会带来更复杂的路径漂移和长路径信用分配
    ↓
为验证这种更难训练的结构，先从已有 checkpoint 出发，保留始终开启的原模型主干，
并在固定位置及时收拢分支，
寻找一个能够训练的完整候选，再根据失败逐步调整结构
```

这里由目标直接推出的是“有界度下的多跳扩展”，不是某一种特定层次结构。规则层次递归只是当前最容易从 checkpoint 生长、也最容易控制发散和收拢的工程起点；line、lattice、mesh、多尺度 backbone 和其他局部 DAG 都可以表达同一个基本方向。

这条推理把入口、router、广播和 merge 都计入成本；不能把一步访问全局容量所需的服务当作 Graph 外部的免费能力。

第二条回答：**为什么优先验证 `broadcast-observe`（下文简称 BO）？**

```text
多级 selected-dispatch 只把消息发给当次选中的下游
    ↓
经过多次分叉后，每个深层分支的私有 KV/SSM 可能只见过一小部分历史，
从而形成“分支记忆变薄”的问题
    ↓
broadcast-observe 把“收到并更新状态”与“执行昂贵计算”分开：
active sender 的所有固定直接下游都收到消息，只有少数下游做昂贵计算并继续传播
    ↓
私有状态可以保留本次没有激活时收到的信息；
必要时还可以让不同分支交叉会聚，互相借用已经处理过的信息
    ↓
最终验证：这些机制是否真的改善学习和规模扩展，
以及收益是否大于新增的状态、通信和调度成本
```

“分支记忆变薄”目前是待验证的核心猜测，不是既有结论。当前 hidden 可能已经整合了完整上下文；BO 也只让 active sender 的直接下游都看到消息，不能自动让所有深层节点看到全部历史。工作流 B 的目标正是判断这个问题是否真实、BO 是否有用，而不是预先宣布 BO 是唯一答案。

### 1.2 当前研究选择

- **三个原始要求**：空间图拓扑在执行中固定；单节点参数、状态、连接接口、候选处理和通信成本具有不随总容量增长的上界；模型可达、可利用的总容量能够继续增长。
- **由此得到的结构方向**：固定拓扑本身不推出度有界，但在每条连接都需要实际资源的成本口径下，单节点成本有界会限制节点度；有界度与容量增长再共同要求多跳、逐级或空间化扩展。
- **额外的稀疏目标**：若每 Token 还只能执行少量昂贵模块，就需要多次局部选择，以及明确的消息、传播深度和总激活预算。
- **近期工程起点**：从已有 checkpoint 生长规则的层次递归，用始终开启的原模型主干（always-on backbone）保住原模型能力，用固定收拢（fixed merge）及时结束分支。
- **工作流 B 的核心赌注**：一个 active sender 的所有固定直接下游都先收到消息并更新私有状态，再只选择少数下游做昂贵计算。
- **按需加入的结构**：多父交叉会聚、不等长路径，以及更一般的 line、lattice、mesh 或局部 DAG。它们可以从首个完整候选开始使用，也可以由具体失败牵引。
- **Head-Wise MoE 的位置**：它主要研究如何把 flat MoE 拆成多个较小的组内 MoE，当前放在工作流 A 作为可选后续，不要求与工作流 B 强行合并。

### 1.3 本仓库要回答的问题

Tide 的长期目标是研究一种自回归神经网络架构，使它同时具有：

1. **固定空间拓扑**：对一个已经确定的模型，节点和边不随 Token、状态或 selector 改写；动态变化的是这次输入激活哪些固定节点和边。
2. **单节点成本有界、总容量可扩展**：所有节点，包括入口、selector、router 和 merge，其资源成本都具有不随模型总容量增长的上界；与此同时，可达且能实际贡献的总容量可以继续增长。
3. **结构稀疏与激活稀疏**：静态 Graph 本身不是全连接；一次输入只执行全部潜在昂贵计算中的一小部分。
4. **可训练性**：动态选择不会使路径漂移、节点饥饿和长距离信用分配严重到无法稳定训练。
5. **`prefill = decode`**：逐 Token `decode` 与任意合法 chunk 的 `prefill` 实现相同的单序列 reference semantics。
6. **实际系统收益**：被跳过的工作足够昂贵，能够覆盖 selector、状态更新、packing、通信和负载不均衡的成本。

这些目标存在天然张力。被跳过的模块必须足够大，稀疏执行才有收益；但单个模块的语义贡献又需要足够平滑、重叠或及时 merge，动态换路才不至于使训练失稳。Tide 当前最核心、也最可证伪的总假设是：

> 在当前或未来的模型规模与硬件上，存在一种有实际意义的中间粒度：node 内部仍使用高效稠密 kernel，node 之间沿固定空间拓扑进行动态稀疏激活；模型扩容时单节点成本保持有界，而模型质量、训练稳定性与端到端性能可以同时成立。

本仓库不从最一般的 Graph 开始实现，也不再预设一条必须依次通过的架构阶段链。Checkpoint 生长线近期并行推进两个实验工作流：

```text
工作流 A：dense checkpoint / flat MoE 基线与校准
工作流 B：从 checkpoint 中性生长，正面验证 broadcast-observe 局部计算介质
```

工作流 A 提供 correctness oracle、成熟训练配方和强稀疏对照；工作流 B 从一开始就允许联合使用有界度的多跳扩展、私有状态、局部 selector、always-on backbone 与 fixed merge，寻找一个有成功可能的完整候选。具体结构随后由实验观察和失败诊断继续演化，而不是由预先写死的阶段编号决定。

HB-Line、HB-Lattice、一般空间 DAG 和一般动态 Graph 仍是重要研究方向。本仓库会为它们保留接口意识，并优先摸索可由 checkpoint 生长得到的受控局部 DAG，而不会先实现一般 Graph runtime。

## 2. 从成熟基线开始

### 2.1 原生 dense checkpoint

近期实现从一个 pre-norm、decoder-only、开放权重 Transformer 开始。省略 dropout、位置编码和 cache 细节，记 $\mathcal N$、$\mathcal A$、$\mathcal F$ 分别表示 Norm、Attention 和 FFN，一个标准 block 可以直白写成：

$$
h'=h+\mathcal A(\mathcal N(h)),
$$

$$
h^+=h'+\mathcal F(\mathcal N(h')).
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
3. 下一层面对该层自己的完整 expert 候选集合，不会因上一层选择而在拓扑上失去候选。
4. shared expert、dense 层、残差主干、负载均衡和成熟的 expert-parallel runtime 共同降低训练与部署风险。

MoE 已经经验性证明了一种重要平衡：总参数容量可以随专家数增加，而每 Token 只执行少数昂贵专家。标准 flat MoE 的模型拓扑通常也是固定的；它与 Tide 当前目标的差别不在于“拓扑是否固定”，而在于它把本层全部专家作为一步可选的全局候选集合。专家数增长时，这个一步候选集合也会增长；常见实现中的候选打分、router 接口或全局 dispatch 范围会随之扩大。Tide 要研究的是：能否改为沿固定空间图逐步接触更多容量，使单节点成本不随总容量增长。这样做也会放弃 MoE 的两项重要优势：每层都面对该层自己的完整候选集合，以及一次选择立即结束其显式路径身份。此外，MoE 仍有路由漂移、selected-only feedback、专家饥饿、负载不均衡和 all-to-all 通信成本。

因此，标准 MoE 必须成为 Tide 的主要对照组，而不是只与 dense Transformer 比较。

### 2.3 从目标到工作流 B：当前推理

下面记录的是本仓库设计实验时采用的起始逻辑，而不是已经得到的结论。其中，第 1 条来自成熟模型的经验；第 2、3 条由 Tide 的目标牵引；第 4 条是对训练困难的预期；第 5 至 7 条是优先采用的工程办法；第 8 条以后是围绕 `broadcast-observe` 的核心待验证假设。

#### 从 MoE 到逐级局部扩展

1. **Flat MoE 是已经得到正面验证的起点。** 它证明了很大的潜在参数量可以和每 Token 少量昂贵计算同时成立。

2. **固定空间拓扑、单节点成本有界和总容量增长，共同要求多跳扩展。**

3. **如果还要求超稀疏，就需要在逐级展开时反复做局部选择。**

#### 预期困难与起步护栏

4. **多级局部选择可能比 flat MoE 更难训练。** Flat MoE 也有路由漂移、专家饥饿和信用分配问题，但一次选择通常很快 merge。Tide 中较早的选择可能继续影响后续多步路径；如果模块还有私有状态，一次写入也可能在以后才产生作用。因此，路径变化和信用分配问题可能被放大。不过，MoE 的成功也说明稀疏离散选择并非天然不可训练。

5. **固定收拢（fixed merge）是最直接的缓解手段。** 它让分支在已知位置重新回到共同接口，避免一次局部选择无限期决定后续可达路径。

6. **始终开启的原模型主干（always-on backbone）是从 checkpoint 生长时的稳定锚点。** 原模型主路径始终保留，新增模块通过初始中性的 residual 接口接入。这样可以从一个已经可用的模型出发，而不是一开始就把全部能力押在尚未训练好的稀疏路径上。它只能保证稳定起点，不能保证新增机制会被模型真正使用。

7. **局部 selector 有很大的设计空间。** 局部候选较少，可能比一次全局 $N$ 选 $M$ 更容易；但选择需要重复发生，早期错误也可能影响更远的下游，因此也可能更难。负载均衡、历史激活、receiver 自身信号和 shared expert 等已有 MoE 经验都可作为参考，具体设计则由实验问题决定。

#### 为什么关注分支记忆与 broadcast-observe

8. **多级 `selected-dispatch` 可能让下游模块拥有的局部历史越来越少。** 一个模块若只在被选中时收到消息，它自己的 KV、SSM 或其他状态就只记录路由到它的那部分历史。本文把这个可能的问题简称为“上下文记忆稀释”。这里的“稀释”指模块自己的历史覆盖变少，不表示当前 hidden 被机械地切成了几份。它是否真的损害模型，是工作流 B 要验证的假设。

9. **纯 FFN 递归不会遇到同一种私有状态问题。** FFN 每次处理当前 hidden，本身没有跨 Token 的私有 KV 或 SSM 状态。纯 FFN 路径仍可能增加参数容量、条件深度和非线性计算，因此应保留为对照；但它不能充分检验“局部有状态计算介质”是否有价值，当前优先级可以较低。

10. **如果局部历史不足确实成为问题，首先有两类改善思路。**
    - 第一类是私有状态与 Update/Compute 分离：模块收到消息时可以先更新状态，是否执行昂贵计算另行决定。
    - 第二类是分支间交叉会聚：不同路径在最终 fixed merge 之前互相交换、借用已经处理过的局部信息。

11. **第一类思路最直接地引出了 `broadcast-observe`。** 已激活的上游把消息发给全部固定下游；所有下游都可以接收并更新自己的状态，但只有少量下游执行昂贵计算并继续发送。私有状态并非只能用 BO 实现，但如果希望所有直接下游都持续获得写入机会，BO 就是最直接的传播方式，也是工作流 B 主动选择验证的核心机制。BO 还允许先更新状态，再让 selector 根据更新后的状态决定本次激活，这是 `selected-dispatch` 没有的设计空间。

12. **BO 首先缓解的是一跳范围内的历史缺失。** 一个 active parent 的所有直接 children 都能看到当前消息，不再因为这一跳没有被选中而完全错过它。但 BO 不会自动绕过更早没有激活的祖先，也不保证写入的状态一定有用。

13. **交叉会聚可能进一步扩大不同分支之间的信息交换。** 多个父节点可以把各自处理过的信息送到同一局部节点；反复加入这类连接，会使规则递归逐渐接近 lattice 或 line 一类局部 DAG。它可能更充分地缓解分支隔离，但不能免费保证每个末端都获得全部无损历史。

14. **首个候选可以优先在完整的 `Attention/SSM -> FFN` 之后借出消息。** 这样较贴近已有 block 的接口和 checkpoint，也保持“先读取上下文、再处理信息”的常见结构。是否应该在 Attention/SSM 后、FFN 前更早交换信息，可以作为后续对照，不必在起步时展开全部接口组合。

15. **会聚对训练的影响可能是双向的。** Fixed merge 明确地结束旧路径，因而有助于限制一次选择影响多远。分支间的交叉会聚还可能提供更多、更短的信息和梯度路径，从而缓解对单一路由的依赖；但它也会增加动态输入组合和多路归因的复杂度。因此，“交叉会聚能够缓解路径漂移和信用分配”是值得验证的正面假设，不是已有结论。

16. **多父节点会增加消息聚合和状态更新的接口复杂度。** 而 BO 相比于 `selected-dispatch` 与这种多父 DAG 的结合更自然。

#### 工作流 B 要验证什么

17. **工作流 B 的核心任务，是找到一个包含 BO 的完整成功候选。** 当前候选至少应包含：

    - 执行期间不变的空间拓扑；
    - 数量具有统一上界的直接下游；
    - 所有直接下游都能 Observe / Update；
    - 可在以后真正读出的私有状态；
    - 只激活少量昂贵计算的局部 selector 和总预算；
    - always-on checkpoint backbone；
    - fan-in 具有统一上界的 fixed merge。

有界度的多跳扩展可以从首个候选开始加入，也可以先用浅层结构验证 BO；但若要证明 Tide 的容量扩展主张，后续必须进入多跳或空间 scaling 实验。多父交叉会聚是可选增强项，可以预先用于一个完整候选，也可以在实测到分支历史不足后再引入。

工作流 B 最终需要依次回答四个直白问题：这个完整候选能否稳定训练；BO 和私有状态是否真的被使用；它是否优于匹配的 `selected-dispatch` 对照；容量扩大以后，质量收益是否仍大于新增的计算、状态和通信成本。具体证据门见第 8 节。

## 3. 与上游 Graph 收缩线的关系

在 TIDE 总体研究中，Graph 收缩与 checkpoint 生长是两个出发方向：前者负责探索和约束设计空间，后者负责形成可复现、可归因的真实实验。它们不同于第 6 节所说的两个并行实验工作流；工作流 A/B 都属于本仓库承载的 checkpoint 生长方向。当前实验优先级明确偏向后者，但 Graph 收缩仍持续为实验提供设计边界。

### 3.1 Graph 收缩线：从一般机制空间提取约束

Graph 收缩线来源于 Tide/LH 最早对“局部通信 + 超稀疏”的一般 Graph 设想。它从表达力较强、机制相互混合的候选空间出发，逐步追问：哪些语义可以明确定义，哪些依赖可以并行，哪些 selector 或状态副作用会破坏 `prefill = decode`，以及哪些结构可能被收缩成可训练、可实现的模型。

可以用下列方向概括这一思考过程：

```text
一般 Graph / LH mechanism pool
-> 有限、依赖完整的 logical event DAG
-> 显式 allocator 的一般空间 DAG
-> HB-Lattice / HB-Sliced / HB-Line 等局部结构候选
-> 有界度的多跳扩展、固定 merge 的可实验分支族
```

这不是近期代码必须依次实现的工程流水线。Graph 收缩线当前主要产生四类结果：

1. **语义约束**：明确 reference transition、状态所有权、消息依赖、chunk 边界和 `prefill = decode` 的前提。
2. **反例与下界**：识别隐藏反向控制依赖、不可组合跨 Token selector 和 pointer-chasing 式路由，避免在注定无法通用低-span `prefill` 的机制上过早投入实现。
3. **设计坐标**：区分固定空间拓扑、节点度、动态激活、allocator、fixed merge、路径寿命、私有状态和物理调度，使架构变化可以逐项讨论。
4. **候选收缩**：把过于一般的 Graph 自由度逐步压缩成固定 DAG、有界度的多跳扩展、局部 selector 和及时 merge 等可实验结构。

因此，它不是本仓库近期的实验主线，却直接帮助判断“哪些自由度值得进入实验”和“实验失败可能来自哪个结构因素”。本仓库不会先实现一般 Graph runtime；只在具体实验需要且理论边界清楚时，引入 Graph 收缩线得到的某项能力。

正式定义、正向定理与一般空间 DAG 见上游 [Tide 数学基础](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/tide-mathematical-foundations.md)；不可组合自适应路由的反向边界见 [Adaptive routing prefill lower bound](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/adaptive-routing-prefill-lower-bound.md)。

### 3.2 Checkpoint 生长线：本仓库近期实验主线

Checkpoint 生长线从已有预训练 Transformer/Mamba 出发：

```text
原生预训练模型
-> 完全等价装载
├── 工作流 A：dense / flat MoE 基线与校准
│     ├── dense continued-pretraining
│     ├── 成熟 flat MoE reference recipe / 原生实现复现
│     ├── 函数保持的 checkpoint-grown flat MoE
│     └── 可选后续：Head-Wise MoE
└── 工作流 B：函数保持的 residual growth
      -> broadcast-observe 局部计算介质候选
         ├── fan-in / fan-out 统一有界的递归分支
         ├── 私有状态、局部 selector 与稀疏昂贵计算
         ├── always-on backbone 与 fixed merge
         └── 按观察需要引入长短路径、空间化和结构变异
```

这条路线负责复用完整 checkpoint 和成熟训练配方。探索实验允许把一组有共同设计理由的机制组成一个完整候选，以优先寻找正面存在性信号；任何关于单项机制的因果结论，仍需要只改变关键因素的直接反事实。后期可以删除节点、替换 kernel 或形成不再兼容原结构的后代模型，但必须保留清楚的 checkpoint 谱系。

更完整的 checkpoint 生长、递归 fixed merge 和训练风险讨论见上游 [Tide 模型架构与训练](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/tide-model-architecture-and-training.md)。

### 3.3 Graph 收缩与 checkpoint 生长如何交换结果

Graph 收缩线可以用 correctness、复杂度或训练风险否决某些过强的 growth operator，也可以提出新的局部结构候选；checkpoint 生长线则用真实训练和系统实验检验这些约束是否有意义、是否过强，以及某项 Graph 自由度是否真正带来收益。

递归 fixed merge 分支是当前最可能的交界面，但两条路线不必最终得到同一个架构。它们可以只共享 selector scope、fixed merge、状态所有权和 chunk correctness 等契约，也可以长期保持“理论设计线”和“可部署实验线”的分工。

## 4. 当前候选骨架：层次化递归并行分支

### 4.1 为什么采用固定 merge 分支

考虑一个父模块，其输入和输出空间相同。父模块包含 always-on 主分支 $B_0$ 和 $b$ 个候选 residual 分支 $B_1,\ldots,B_b$。跨模型规模设定一个统一上界 $b_{\max}$，始终要求 $b\leq b_{\max}$，而不是随着总容量增加不断扩大同一个父模块。先用无状态简写只描述当前 Token 的昂贵分支输出与 merge：selector 为输入 $x$ 选择集合 $A(x)\subseteq\lbrace 1,\ldots,b\rbrace$，固定 merge 为：

$$
T(x)=B_0(x)+\sum_{j\in A(x)}g_j(x)B_j(x).
$$

在有状态 `broadcast-observe` 中，Receive / Update 可以先执行，激活集合与分支 readout 更一般地写成 $A_t=R_\theta(x_t,\{S_{j,t}^{+}\})$ 和 $B_j(x_t,S_{j,t}^{+})$。上式不试图省略这段状态转移，只定义各 active 输出如何回到固定接口。

“固定”表示分支的入口、出口、merge 位置和 merge 算子在模型结构中预先确定；动态变化的只有激活集合和可选权重。短分支不能越过 merge 提前修改外层状态，长分支也不能在 merge 后追赶并改写同一个输出。

该结构具有四个近期优势：

1. 原 checkpoint 主路径可以完整保留。
2. 新分支可以通过零输出或显式代数等价的 clone、缩放与 merge 构造，在初始点保持原函数；任意复制后重组不能自动声称等价。
3. 每个父模块最多连接 $b_{\max}$ 个子分支；这个上界不随模型总容量增长，因此 selector 和 merge 都不会因扩容变成越来越大的中心节点。
4. 当前 Token 的显式输出分支身份在固定 merge 处结束，因而本段控制寿命可知；已经写入的 private state 仍可能跨 Token 保留语义影响。

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

任何包含传播边的非平凡 Graph 都由具有长度的传播路径构成，很多可扩展 Graph 也会表现出某种逐级展开、空间分区或多尺度组织；不同路径是否等长、如何形成以及何时收拢，则不一定来自当前这种规则递归。规则的层次化递归因此不是一般 Graph 的唯一来源或形式，却是一个值得先行摸索的受控候选：它可以从现有 checkpoint 的稳定接口逐级生长，在 fan-in 和 fan-out 具有统一上界时显式改变传播距离，并把不等长局部 DAG、按需计算和 merge 频率放进同一个可实验框架。在本仓库的推进语境中，它既是通向更一般局部 Graph/DAG 的重要桥梁，也是进入 line、lattice、mesh 和多尺度空间结构之前的证据与工程前置。

### 4.3 固定拓扑、度有界和容量扩展怎样同时成立

对一个已经确定的模型，父子边、分支入口和 merge 位置都保持不变；selector 只决定本次激活哪些固定节点和边，不会临时改写拓扑。每个 selector 最多管理 $b_{\max}$ 个兄弟分支，每个 merge 也最多收拢 $b_{\max}$ 个直接分支，因此节点度、局部候选处理和直接 merge 成本具有统一上界。这个度上界来自每个局部接口的规模上界，不是由“拓扑固定”自动得到的。

跨模型规模扩展时，每级 fan-in、fan-out 和局部 Top-K 继续保持有界，但递归深度或空间直径可以增长。若连接度和最大深度都永远固定，从一个固定入口可达的节点总数也会存在固定上限，无法支持总容量持续增长。因此，每个具体模型只需是有限 Graph；不能把“所有规模使用同一个固定深度”误写成 Tide 的长期要求。

这只证明**逻辑局部性**。要得到物理局部通信，还必须把父模块及其活跃子树放置在同一设备、相邻设备或固定 region，并测量真实通信字节和链路距离。若逻辑邻居被放在远端设备，层次化命名不会自动产生系统收益。

### 4.4 共享 selector

同一父模块分出的兄弟分支共享一套逻辑预算和 selector state。实现可以是共享网络直接打分，也可以由各 receiver 产生 local proposal / eligibility，再由有界的 sibling arbitration 形成 active set；这里的“共享 selector”指局部预算语义统一，不要求一个全模型中心路由器。若分支完全独立地宣布激活，则必须另有可验证的预算协议，否则无法严格控制 Top-K、负载和总 active FLOPs。

最简单的 selector 可以是 token-local：

$$
A_t=R_\theta(h_t),
$$

它只依赖当前 Token 在当前父模块的表示和静态参数。`broadcast-observe` 主假设也允许 selector 读取 receiver 更新后的语义状态和逐序列历史激活信息；是否引入这些输入由当前观察和待解决问题决定，不再绑定到预设的串行顺序。无论采用哪种输入，模型语义都不得依赖物理 batch、实时设备负载或当前 chunk 的切分方式。

### 4.5 传播、状态更新、激活和发送是不同决策

标准 MoE 通常把“选中 expert、把 Token dispatch 给它、执行 expert、立即 merge”组织成一个紧密阶段。Tide 候选架构不必把下面四件事绑定在一起：

1. **消息可见性**：哪些静态下游可以看到当前信号。
2. **状态更新**：看到信号的 node 是否更新私有状态。
3. **昂贵激活**：哪些 node 执行 Attention/FFN/SSM readout 等重计算。
4. **继续发送**：哪些 node 产生输出并沿固定边继续传播或进入 fixed merge。

当前至少保留两种传播 profile：

| Profile | 消息与状态语义 | 与标准 MoE 的关系 |
| --- | --- | --- |
| `broadcast-observe` | 已激活 sender 向全部静态 children 发送；每个 receiver 执行声明的 Observe / Update，Update 可以持久也可以无持久效果；只有 active receivers 执行重计算并继续发送 | 把消息可见性和状态写入机会，与本次昂贵执行和传播分开 |
| `selected-dispatch` | 先选 active children；只有被选 children 收到输入、更新状态并执行 | 最接近普通 MoE dispatch，通常具有更低的消息投递和未选状态更新成本；主要作为成熟基线和直接反事实 |

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

因此，“总是发送、收到即更新、但不一定激活”仍是待实验验证的消息/状态 profile，而不是已经成立的公理；但它比 `selected-dispatch` 更直接表达了本仓库想研究的有状态局部计算介质，因而成为当前正面探索的主假设。它可能缓解未选 node 完全看不到数据的问题，也会增加状态写入、显存、跨 Token 依赖和延迟信用分配。

正面探索可以把 `broadcast-observe` 与有界度的多跳扩展、持久状态和 fixed merge 组成一个 coherent bundle；关于 `broadcast-observe` 本身的因果结论，则仍需在相同拓扑、状态容量、昂贵模块和 active budget 下，与 `selected-dispatch` 做直接反事实。还要注意：只有 active sender 的直接静态后继收到消息；若某个上游从未激活，更远处节点不会因“broadcast”而自动看到全局数据。

两种 profile 都可以扩展到多父局部 DAG。`selected-dispatch` 可以让每个 parent 只向自己选中的 children 发送，receiver 再聚合实际收到的零到多条消息；`broadcast-observe` 则让每个 active parent 沿全部声明的局部出边发送。多父语义必须额外定义 inbox 完整条件、消息顺序或结合归约、空消息、重复消息、state commit 和跨父预算仲裁。BO 在这里是本仓库主动选择的消息可见性假设，不是多父拓扑强迫出的唯一传播方式。

### 4.6 Selector 的候选输入与决策规则

令同一父模块的候选 child 集合为 $J$。对 $v\in J$ 和 Token 位置 $t$，可以分别构造：

- 当前内容摘要 $c_{v,t}$。
- 更新后的语义状态摘要 $q_{v,t}^{+}$，例如 KV/SSM/accumulator 的低维读出。
- 逐序列历史负载状态 $\ell_{v,t}$，例如激活计数、指数移动平均、距上次激活的间隔、恢复量或预算余额。
- 静态 node 信息 $\lambda_v$，例如 node id、层级、固定容量和模块类型。

一个一般 score 接口可以写成：

$$
s_{v,t}=G_\theta(c_{v,t},q_{v,t}^{+},\ell_{v,t},\lambda_v),
\qquad v\in J.
$$

具体 selector 不必读取全部四类输入。当前值得分别实验的 profile 是：

| Selector profile | 实际读取 | 主要作用与风险 |
| --- | --- | --- |
| 固定或 hash route | $t,\lambda_v$ | 最简单的稀疏与负载对照；不根据语义适配 |
| content-only | $c_{v,t},\lambda_v$ | 最接近标准 MoE router；是最简单的 learned selector 参照 |
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

## 5. 设计坐标与候选结构，不是固定阶段

`Leaf-Gated` 与 `Receiver-Gated` 描述的是**哪些拓扑层级允许被门控**，不是“保守程度”的价值判断，也不决定使用 `selected-dispatch` 还是 `broadcast-observe`。

为避免名称暗中绑定其他机制，每个具体模型至少用下面五个坐标描述：

$$
\mathcal C=(\text{门控范围},\ \text{传播 profile},\ \text{状态生命周期},
\ \text{selector 输入},\ \text{selector 决策}).
$$

例如，最接近标准 MoE 的反事实配置是：

```text
(Leaf-Gated, selected-dispatch, no-delayed-private-state, content-only, hard-Top-K)
```

本仓库当前希望正面探索的配置可以写成：

```text
(Leaf-Gated or Receiver-Gated,
 broadcast-observe,
 persistent-private-state,
 semantic-state-aware or state-and-load-aware,
 sparse activation)
```

这里没有把门控范围、selector 输入或离散规则提前写死；它们由首个完整候选的需要和后续观察决定。`Leaf-Gated`、`Receiver-Gated`、`selected-dispatch`、`broadcast-observe` 仍然是重要的精确术语和配置轴，但不再代表必须按顺序经过的开发阶段。

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
- 下一层面对该层自己的完整 expert 候选集合，不因本层选择而在拓扑上失去候选。
- 标准 FFN experts 通常没有独立、可延迟读出的单序列持久状态。

MoE 的选择仍会改变当前 hidden，因而可能影响下一层路由，也可以通过后续 Attention 影响未来 Token。立即 merge 删除的不是语义影响，而是持续存在的显式控制路径身份。

### 5.2 Leaf-Gated Tide：只有末级叶子受门控

Leaf-Gated Tide 满足：

1. 原模型主路径和所有非叶 node 始终激活。
2. 只有递归结构最末一级的叶 node 可以被 selector 跳过。
3. 叶输出在父模块出口立即 fixed merge。
4. merge 后下一段面对该段自己的完整候选集合，不因本次叶选择而失去候选。
5. 一次叶选择不限制更远层的拓扑可达集合。

```text
always-on internal hierarchy
-> sparse terminal leaves
-> immediate fixed merge
-> common representation
```

它与标准 MoE 的主要区别可以集中在：

- 候选集合按父子层级组织，而不是把全部 expert 作为一步可选的平铺集合。
- selector 只在各末级兄弟集合内局部选择；内部父节点仍全部常亮，不在本级裁剪激活子树。
- 分支可具有不同的有限串行深度。
- 逻辑拓扑更适合映射为固定空间图上的局部通信。

Leaf-Gated Tide 可以采用第 4.5 节的任一传播 profile：

- 使用 `selected-dispatch` 时，selector 先选叶子，只有被选叶子接收输入、更新状态、执行并进入 fixed merge。这是接近标准 MoE 的直接反事实。
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
a_{v,t}\in\lbrace 0,1\rbrace,
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

#### 历史暴露变薄不等于当前表示被机械切碎

`selected-dispatch` 直接造成的是 receiver exposure 的差异：某个节点的私有状态只包含实际路由到它的消息，多个节点因而可能保存不同的历史子集。它不自动意味着一个 active child 只拿到 parent hidden 的一部分；parent hidden 仍可能已经整合了完整前缀中的任务相关信息。

因此需要分别测量消息/Token 覆盖、私有状态历史、当前表示中的任务相关信息，以及这些差异对质量的影响。BO 能保证 active parent 的全部直接 children 获得本次 Observe / Update 机会，但不能让它们绕过未激活祖先，也不能保证状态写入以后一定有用。选择性历史也可能形成有价值的专门化，而不只有负面结果。

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

### 5.5 Head-Wise MoE：工作流 A 的可选后续

Head-Wise MoE 把 hidden 按 Attention Head 或 Head Group 切开，每组拥有自己的小型 FFN expert pool，并在组内独立选择少数 experts，最后把各组输出 concat 或通过 mixer 重新混合。它主要想验证：能否把一个大的 flat MoE 拆成多个较小、可以随 Head/Group 本地放置的 MoE，从而减少 expert-parallel dispatch/combine 通信。

```text
完整 hidden
-> 按 Head / Group 切片
-> 每组在自己的 expert pool 内做 Top-K
-> 各组输出 concat / mixer
-> 完整 hidden
```

它与 Tide 有一层启发性关系：二者都把一次较大范围的选择拆成较小范围的局部选择，也都有规则的发散—收拢结构。但原始 Head-Wise MoE 中，每个 group 对每个 Token 都会参与，稀疏只发生在组内 expert 选择；它没有验证工作流 B 最关心的三件事：

- 未激活的下游是否仍应收到消息并更新状态；
- 私有状态是否会在以后被读出并产生作用；
- 多级局部传播和递归选择能否训练。

因此，Head-Wise MoE 当前放在工作流 A，作为 flat MoE 基线可靠之后的可选后续，而不是工作流 B 的必需骨架或直接反事实。它主要比较 flat MoE 与局部因子化 MoE，并检查：切分粒度是否损害全局语义、私有 expert pool 是否形成有效专门化、小 GEMM 是否高效、concat/mixer 是否恢复跨组表达，以及理论上的通信减少能否变成端到端收益。

实验至少需要对齐总参数、active FLOPs、训练 Token 和硬件资源，并记录 Group 数、每组宽度、每组 expert 数与 Top-K、mixer 成本、实际通信和吞吐。Head-Wise 的名称本身不保证等资源，也不保证通信收益一定成立。

未来工作流 B 如果恰好使用 Head/Group 来组织 receivers，那只是 BO 候选的一种实现方式，需要重新定义 group 级 Receive、Update、Activate 和 Emit，并在相同 receiver 拓扑上比较 `selected-dispatch` 与 `broadcast-observe`。这个新候选不能反过来算作原始 Head-Wise MoE 已经合入 Tide。

## 6. 两条并行实验工作流与问题驱动闭环

### 6.1 总原则

1. 两条工作流共享数据、训练配方、correctness oracle、成本口径、checkpoint 谱系和实验账本；dense equality 就绪后可以并行推进，不要求 flat MoE 先全部完成。
2. 探索实验可以同时引入一组有共同设计理由的机制，以寻找“存在一个可工作的完整候选”的正面信号；任何单项因果主张都必须补直接反事实。
3. 总参数、active parameters、actual FLOPs、训练 Token、优化器和数据尽量匹配；无法同时匹配时，分别报告 capacity-matched、compute-matched、resource-matched 和 quality-matched 结果。
4. correctness、机制使用、训练质量、容量扩展与系统性能使用彼此独立的证据门。
5. selector 的模型语义不得依赖 batch 组成、chunk 切分或实时设备负载。
6. 每个实验完整记录 branch grammar、门控范围、传播 profile、状态更新与生命周期、selector 输入与决策、active budget、merge/backbone 和物理放置；不得只写一个 Tide 类名。
7. 下一项机制由已观察问题牵引。selector history、辅助 loss、递归深度、长短路径和 mixer 形式是可按需使用的干预工具，不是预先排好的隐性阶段。

实验分为三类：

- **探索实验**：允许使用一个最小但完整的机制包，只主张候选是否出现正面存在性信号。
- **诊断实验**：针对已经观察到的问题做直接 knockout、paired counterfactual 或局部修改。
- **确认实验**：冻结结构和训练配方，用新 seed、数据切片、硬件或规模复现结论。

### 6.2 工作流 A：dense / flat MoE 基线与可选 Head-Wise 后续

工作流 A 有两个基础对象：原生 dense correctness 与 continued-training oracle，以及成熟 flat MoE 强基线。在这两项可靠以后，可以继续做 Head-Wise MoE，验证局部因子化的 MoE 是否具有独立的质量或系统价值。Head-Wise 是可选后续，不是工作流 B 启动前必须完成的阶段。

#### 6.2.1 原生 checkpoint 与 dense 训练校准

选择一个 pre-norm、decoder-only、开放权重且规模足以快速重复实验的模型。首先实现或包装原生模型，不增加任何 Tide 分支。

必须验证：

- state-dict 参数逐项覆盖。
- logits 与逐层主要 artifact 对齐。
- 单 Token `decode` 与多 Token `prefill` 对齐。
- 任意 chunk 切分的 continuation state 对齐。
- 训练 loss 和主要参数梯度对齐。
- 固定随机种子下结果可重复。

如果原生 checkpoint 不能稳定复现，后续所有架构比较都没有可信基线。该 correctness 工作是两条工作流共同依赖的基础设施，不表示工作流 B 必须等待所有 MoE 实验结束。

#### 6.2.2 flat MoE 强基线

flat MoE 强基线首先应复现一套成熟 reference recipe 或兼容的原生 MoE 实现，包括 router、load balancing、shared expert、expert packing 和训练超参数，而不把“从 dense checkpoint 中性生长”强加为唯一接入方式。它用于回答 Tide 是否真的优于一个经过合理调优的成熟稀疏模型。

另外，在一个局部 FFN 子层或新增 residual branch 上建立 checkpoint-grown flat MoE matched control：

- 固定数量 experts。
- token-local Top-K router。
- 立即 weighted merge。
- 无 expert 私有跨 Token state。
- 可配置 shared expert 或 always-on residual branch。

该 matched control 至少包含两种初始化：

1. 零 residual/零输出投影，初始函数等于原模型。
2. clone-and-split，把旧模块贡献拆给多个副本并保持总和。

flat MoE 的作用不是证明 MoE 最优，而是建立成熟条件计算对照，测量普通 routing drift、负载、梯度覆盖和系统开销。若 Tide 只优于尚未调好的 checkpoint-grown MoE，不能据此声称优于 flat MoE 强基线。成熟 reference、matched control 与 dense 校准共同为工作流 B 提供参考区间。

#### 6.2.3 Head-Wise MoE（可选后续）

在 flat MoE 基线可靠后，可以按第 5.5 节实现 Head-Wise MoE，并扫描 Head/Group 切分粒度、每组私有 expert pool、组内 Top-K 和 concat/mixer。主要对照是等容量、等 active FLOPs 和等资源的 flat MoE；主要结果是训练质量、路由专门化、小 GEMM 利用率、实际通信量和端到端吞吐。

这个实验不承担工作流 B 的 `selected-dispatch` control。工作流 B 应在自己的 receiver 拓扑上保留完全匹配的 `selected-dispatch` 开关。

### 6.3 工作流 B：`broadcast-observe` 完整候选

工作流 B 不从无状态 `selected-dispatch` 的空壳开始，而是直接构造能够真实检验核心命题的最小完整候选。当前首个完整候选把下列机制作为设计契约；这表示它们共同组成要寻找的正面候选，不表示它们已经被证明对所有 Tide 架构必要：

- 完整保留的 checkpoint / always-on backbone。
- 通过中性 residual 接口生长、fan-in 和 fan-out 具有统一上界的分支。
- active sender 沿固定空间拓扑向全部直接 children 发送；所有实际 receivers 执行声明的轻量 Observe / Update。
- 能在未执行昂贵计算时写入、并在以后真正读出的私有 KV、SSM 或 summary state。
- 读取当前内容、语义状态和可选逐序列激活历史的局部 selector。
- 少数 receivers 执行昂贵 Attention、FFN、SSM readout 或私有 MoE，并继续发送。
- 在声明的位置通过 fan-in 有界的 fixed merge 逐级收拢，或者回到稳定的 region/backbone 接口。

一层或多层受控递归、Leaf-Gated/Receiver-Gated、多父交叉会聚和不等长路径，是这个核心候选可以从一开始联合采用的结构坐标，不要求等待更小阶段依次通过。首个最小配对实验也可以立即 merge，以干净地测量 BO；但若最终要支持“固定空间拓扑下，单节点成本保持有界而可达总容量继续增长”的 TIDE 主张，后续 scaling 实验必须进入多跳递归或等价的空间扩展，而不能永远停留在一跳 BO。

第一轮组合实验回答的是：**是否存在一个包含 `broadcast-observe` 的可训练、可稀疏、可从 checkpoint 生长的完整候选**。它不会单独证明每个部件都必要；出现正面信号或具体失败后，再用第 6.4 节的诊断工具建立归因。

`broadcast-observe` 可能通过两条不同路径产生作用，必须分别观测和归因。

**当次 Observe / Proposal 路径：**

```text
every receiver observes the current message
-> its proposal or post-Update summary participates in local selection
-> the current active set or output changes
-> blocking that input changes quality or loss in a reproducible way
```

**跨 Token 延迟记忆路径：**

```text
inactive receiver receives a message
-> its private state changes
-> a later activation reads that state
-> the delayed read measurably affects output or loss
```

第一条不要求跨 Token 持久状态，第二条才验证未激活期间的模块记忆。只有消息送达、却既没有影响当次 proposal/selection，也没有持久写入和以后读出，只能验证传播语义与系统成本，不能证明 learning value。

### 6.4 问题驱动的诊断与干预工具

下面的机制不再按先后阶段排列。它们既可以因共同设计理由进入首个完整候选，也可以在观察到相应失败形态后作为干预；一旦要归因于某项新增或调整，就应与修改前候选组成配对实验。

**递归、长短路径与收拢。** 可以调整：

1. 一层递归。
2. 两层递归。
3. 同深度并行分支。
4. 不同深度的短/长分支。
5. 不同但无跨 Token 私有状态的 atomic/serial 分支 grammar。

记录最大 fan-out、最大递归深度、局部 Top-K、静态与实际路径长度分布、最长串行路径、fixed merge 位置和每 Token 实际激活子树。规则递归可以作为首个完整候选的组成部分；其深度、长短路径和 merge 频率则按 route drift、信用距离、局部混合和系统尾延迟继续调整。

**历史暴露与多父局部会聚。** 当 route artifact 显示深层 receiver 的消息覆盖过低、私有状态高度路径化，且相应 knockout 或记忆任务表明确认这会伤害质量时，可以加入有界多父边、周期性局部会聚、backbone reinjection 或 merge 后重新发散。优先把两个概念分开记录：

- `cross-coupling` 让分支交换摘要但继续保持各自路径身份，主要扩大局部 source coverage，也可能扩大输入漂移和归因歧义。
- `fixed convergence/reset` 把多路信号收回共同接口并结束旧路径身份，主要限制控制寿命，也可能提供更短或冗余的梯度路径。

首个交叉接口可以优先放在完整 `Attention/SSM -> FFN` receiver 的 Emit 之后，以保持旧 block 边界；Attention/SSM 后、FFN 前和低秩 summary 则作为可比较坐标。所有方案必须继续满足 fan-in/fan-out 统一有界、有限 DAG 顺序和显式 active/message budget，并分别统计新增消息、等待和状态成本。同一多父拓扑仍应能够切换 `selected-dispatch` / `broadcast-observe`，避免把拓扑变化和传播 profile 混成一个因果结论。

**传播与状态反事实。** 对同一拓扑、昂贵模块和 active budget，至少保留：

| 配置 | 目的 |
| --- | --- |
| matched/replay route 的 `selected-dispatch` + 对应状态容量 | content/pre-state selector 下 `broadcast-observe` 的直接传播反事实 |
| `broadcast-observe` + selector 忽略 post-Update state | 隔离“更新后状态改变本次选择”的作用 |
| `broadcast-observe` + selector 不读 post-Update state + Update 不影响当前输出 + 禁止延迟读出 | 单独测量广播、轻量更新、状态写入和 packing 的系统成本 |
| `selected-dispatch` + 持久私有状态 | 测量 selected-only state exposure |
| `broadcast-observe` + 持久私有状态 | 测量所有叶子都看到数据能否改善训练，及其延迟信用成本 |

Attention、SSM 或 Linear Attention 支路的 KV/accumulator 生命周期必须与传播 profile 一起定义。以四条并行 `Attention -> FFN` 支路为例，可以直接比较“只有 Top-K 支路写 K/V”与“四条支路都写 K/V、只有 Top-K 做 readout 和 FFN”。还必须使用 state freeze、clear、shuffle、no-read 或跨 chunk reset 等 knockout，证明未激活期间写入的状态后来确实被使用。

**观察—定位—干预。** 机制由真实失败牵引，可以从下表开始，但每个干预仍需直接反事实：

| 观察到的问题 | 优先定位 | 可尝试的干预 |
| --- | --- | --- |
| receiver 长期不激活 | receive/update/active/gradient 覆盖与语义分数 | history/recovery bias、shadow activation、局部 quota |
| state 持续写入但没有作用 | read sensitivity、write-to-read 延迟和 knockout | 加强 readout、局部辅助目标、缩短读出距离 |
| 深层 receiver 的路径历史暴露不足且确认伤害质量 | receiver exposure、状态历史重叠、记忆任务与 causal knockout | BO、提高局部 $K$、backbone reinjection、有界多父交叉或周期会聚 |
| route/path 分布快速漂移 | route churn、node 输入漂移、merge 前后跳变量 | 缩短 merge 距离、慢化 selector 更新、强化 always-on 接口 |
| 长路径梯度不足 | 分层梯度覆盖、动态 hop 与控制寿命 | 更频繁收拢、局部辅助 loss、缩短递归寿命 |
| Observe / Update 成本过高 | 状态投影、写带宽、fan-out 和 packing 分项 | 压缩状态、降低 fan-out、降低写入维度或频率 |
| mixer 重新形成全局通信 | collective 范围、物理距离与等待时间 | region-local、低秩、tree/hierarchical 或邻居 mixer |

一次完整循环写成：`观察 -> 定位失败类型 -> 针对性干预 -> paired counterfactual -> 新 seed / 数据 / 规模确认 -> 沉淀或否定 know-how`。

### 6.5 门控范围：Leaf-Gated 与 Receiver-Gated

Leaf-Gated 与 Receiver-Gated 是候选结构轴。工作流 B 可以为了真实检验局部传播介质，从一开始就让部分非叶 receiver 只 Observe / Update 而不继续发送；也可以在浅层候选中先只门控叶子。选择取决于当前待验证命题，不再要求前者必须等待后者通过。

例如，用来隔离“内部 receiver 门控”本身的反事实配置可以是：

```text
(Receiver-Gated, selected-dispatch, no-delayed-private-state, content-only, hard-Top-K)
```

若要单独测量“内部选择裁剪后续子树”的收益和控制信用距离，则需要传播 profile、状态生命周期、selector 输入、active FLOPs 和 selector 规模匹配的 Leaf-Gated 直接反事实。

同一 Receiver-Gated 拓扑可以按观察需要使用：

- `selected-dispatch` 与无延迟私有状态：隔离非叶门控本身。
- `broadcast-observe`，selector 不读 post-Update state、Update 不影响当前输出且禁止延迟读出：隔离内部广播与更新成本。
- 持久 KV/SSM/accumulator：检验未激活 receiver 的记忆是否以后被使用。

持久状态候选需要明确 BPTT、detach、chunk boundary 和状态生命周期策略。组合候选不必在探索前穷举所有对照，但在声称内部 receiver 门控或 `broadcast-observe` 具有独立贡献前，必须补相应的 matched counterfactual。

### 6.6 Selector 状态是按问题引入的工具

selector 可以从首个完整候选开始读取必要的语义状态；历史负载、恢复量和阈值则在出现激活饥饿、长期偏置或预算问题时按需引入。可选输入与规则包括：

1. 更新后 semantic-state summary，但不读取历史负载。
2. 历史负载只作为语义分数近似并列时的 tie-breaker。
3. 小幅 load bias、局部 quota 或 recovery eligibility。
4. semantic state 与 load state 联合输入。
5. 累计信号阈值或其他显式状态递推。

soft mixture、hard Top-K、语义候选后负载筛选和负载准入后语义筛选也应分别记录，不能统称为“stateful selector”。若状态递推可表示为 scan 或 causal bulk，应优先使用该实现；只有实验证据足够时才接受 node 内任意逐 Token sequential selector。

物理设备负载只能改变调度和放置，不能改变模型路由 artifact。

## 7. 必要对照、直接反事实与实验坐标

组合实验用于发现有效候选；每一个因果结论则需要只改变关键因素的直接反事实。这里不要求在首次训练前穷举整个笛卡尔积，也不再把任何对照写成工作流 B 的前置阶段。

共享基线和主要直接反事实包括：

| 对照 | 主要回答的问题 |
| --- | --- |
| 原 checkpoint continued pretraining | 新结构是否优于继续训练原模型 |
| 等参数 dense 扩展 | 收益是否只来自更多参数 |
| 等 active-FLOPs flat MoE | 局部层次候选是否优于成熟稀疏基线 |
| 相同局部拓扑、相同或 replay route 的 `selected-dispatch` | `broadcast-observe` 的消息可达与未选更新是否有额外 learning value |
| `broadcast-observe` 内 selector 读取 vs 忽略 post-Update state | 更新后状态参与本次选择是否有独立作用 |
| `broadcast-observe`，但未激活 state 冻结或禁止延迟读出 | 收益是否来自未激活期间积累并在以后读出的记忆 |
| state clear / shuffle / no-read / reset knockout | 已写入状态是否以预期的时间、节点和内容关系影响输出 |
| fixed/hash route | learned selector 是否真正有价值 |
| matched Leaf-Gated 配置 | 内部 receiver 门控和后续子树裁剪是否有独立贡献 |
| flat MoE vs Head-Wise MoE（工作流 A 可选） | 输入切片、私有 expert pools 和跨组 mixer 是否有独立质量或系统收益 |
| matched stateless FFN 路径 vs 有状态 Attention/SSM receiver | 收益来自条件计算深度/容量，还是私有序列记忆 |
| 相同多父拓扑、关闭 vs 开启交叉边或周期会聚 | 扩大局部 source coverage 是否改善记忆和质量，以及是否加剧输入漂移 |

其中，最干净的传播反事实让两种 profile 都根据 parent/current content 或 pre-Update state 产生相同 active set，必要时直接 replay route artifact，只改变未选 receiver 是否收到并更新。静态拓扑、状态容量、昂贵模块、active budget、merge、训练 Token 与物理放置应保持不变。若 `broadcast-observe` selector 读取 receiver 的 post-Update state，`selected-dispatch` 在选择前并不存在同一输入，此时不能把两者称为单轴 matched-selector 对照；应另加一个同为 `broadcast-observe`、但 selector 忽略 post-Update state 的反事实，把“更新后状态改变本次选择”与“未激活记忆在以后读出”分开。

需要显式记录、并在相应结论前做单轴比较的坐标包括：

- flat vs hierarchical candidate set。
- equal-depth vs mixed-depth branches。
- leaf-only gating vs internal-node gating。
- `selected-dispatch` vs `broadcast-observe`。
- stateless、selected-only persistent state 与 receive-always persistent state。
- content-only、semantic-state-aware 与 load-aware selector 输入。
- soft mixture、hard Top-K 与两级语义/负载决策规则。
- shared expert / always-on backbone 比例。
- fixed merge 间隔、范围和最大控制寿命。
- 单父 vs 多父、cross-coupling vs fixed convergence，以及消息接入点。

每个工作流 B 实验表至少把 branch grammar、门控范围、传播 profile、状态更新与生命周期、selector 输入与决策、active budget、merge/backbone 和物理放置列成独立字段。Head-Wise 实验另行记录 Head/Group 切分、私有 expert pool 和 mixer。一个配置同时改变多个字段时，它可以支持“完整候选有效”的存在性结论，但不能单独支持任一部件的因果结论。

## 8. 验收指标

### 8.1 Correctness gate

- 原生 checkpoint 的 state-dict 覆盖、logits、原有 cache/state、训练 loss 和主要 backbone 梯度对齐。
- 函数保持生长的初始化时刻，原模型的可观察输出和原有状态轨迹对齐。
- `prefill`、逐 Token `decode` 和任意 chunk continuation artifact equality。
- 不同 batch 组合、合法 chunk 切分和物理调度不改变单序列 reference semantics。
- hard/soft selector、空消息转移、多上游消息聚合和 merge 顺序在训练与推理模式下语义明确。
- 传播 profile、拓扑、状态生命周期、selector 输入与决策均写入 checkpoint/config；fresh save/reload 后 route 和 state 可重放。

“从 checkpoint 无损生长”只表示初始化时保持旧模型的可观察行为。新增 KV/SSM/private state 在旧模型中没有对应物；它可以在后台形成自己的轨迹，但在中性初始化期间不得影响旧输出。zero-output residual gate 还可能使新分支内部暂时没有梯度，因此函数保持方式必须同时记录分支的梯度开启或生长日程。

### 8.2 Mechanism-use gate

完整候选 loss 下降并不能证明 `broadcast-observe` 或私有记忆被实际使用。至少要针对候选实际启用的作用路径，建立并记录对应因果链：

```text
current observe/proposal
-> current selector or active output changes
-> blocking the post-Update input changes output or loss
```

和/或：

```text
inactive receiver receives a message
-> its state changes
-> a later activation reads the changed state
-> blocking or perturbing that write/read changes output or loss
```

对应指标至少包括：receiver proposal 对当前 route 的敏感度、未激活 receiver 的 receive/update 覆盖率、状态变化量、write-to-read 延迟、后续 readout 对状态的敏感度，以及 freeze、clear、shuffle、no-read、reset 等 knockout 的输出或 loss 差异。若 selector 读取 post-Update 或持久状态，还要分别测量“状态改变选择”和“状态改变昂贵分支输出”两条作用路径。

关于 `broadcast-observe` 的结论分为四层，不应互相替代：

1. **机制确实运行并被使用**：当次 Observe / Proposal 或未激活期间的状态写入，对模型行为具有可重复的因果作用；具体是哪一条必须说明。
2. **具有 learning value**：在 matched `selected-dispatch` 反事实下，质量、样本效率或稳定性出现可复现改善。
3. **具有 scaling value**：潜在容量和局部空间扩大时，质量或能力继续增长，而昂贵激活、消息和状态成本仍保持稀疏可控。
4. **具有系统 value**：广播、轻量更新、selector、状态读写和 merge 的额外成本，小于跳过的昂贵计算与通信，并带来端到端收益。

### 8.3 训练 gate

- train/validation loss、perplexity 和下游质量。
- route churn：同一输入跨 checkpoint 的 active-set 变化。
- active-set overlap、分支输出跳变量和局部候选分布漂移。
- 每个 node 的消息接收次数、状态更新次数、昂贵激活次数、继续发送次数和有效梯度次数；五者不得合并成一个“使用次数”。
- 梯度范数、梯度覆盖率和长期未更新参数比例。
- node 输入分布漂移、状态数值稳定性和 write-to-read 延迟分布。
- 按 selector input profile 和递归层级分组的负载、route churn、状态利用与梯度覆盖。
- always-on backbone、局部递归、长短路径和 merge 频率的贡献消融。
- 多个 seed、数据切片或后续规模上的复现情况。

路径相关指标必须分开记录：静态拓扑路径长度、某个 Token 的动态传播 hop 数、实际执行的昂贵模块数，以及写状态到以后读出的 Token 距离。它们都可以被口语称为“路径长”，但对应不同成本和信用问题。

激活均衡不等于训练均衡。节点即使被均匀激活，也可能没有有效梯度、没有读出其状态，或没有稳定语义分布。

### 8.4 Scaling gate

- 对每个模型实例保持固定空间拓扑；跨规模增加潜在节点数、总参数量、递归深度或空间直径时，入口、普通 receiver、selector、router 和 merge 的入度/出度与资源成本都要保持统一有界。
- 分开报告总容量、最大单节点参数与状态、最大邻居数、局部候选处理量、每 Token 实际到达的节点、昂贵激活数、Emit 边数、Observe / Update 次数和状态容量；不能只用 active parameters 代表全部成本。
- 验证质量、能力、样本效率或可保留知识是否随潜在容量增长，而不是只在固定小模型上超过一个弱基线。
- 记录深度增加时的 receiver exposure、route churn、梯度覆盖、状态利用、最长控制寿命和 write-to-read 延迟，判断局部 selector 的早期错误与历史碎片化是否累积。
- 与 capacity/compute/resource-matched dense 和 flat MoE 比较，并检查收益是否依赖不断增大的 fan-out、全局 mixer 或近似稠密的 Observe / Update。

只有 correctness、mechanism-use 和小规模 learning value 成立，还不能证明 Tide 的“极大容量 + 超稀疏”目标。Scaling gate 要回答的是：增加潜在容量时，模型是否继续获得收益，单节点成本是否仍有统一上界，以及每 Token 的总工作、传播跳数和物理通信范围怎样随规模变化。任何方案都不能把增长的成本隐藏到全局 router、广播、collective 或 merge 中。

### 8.5 系统 gate

- 总参数、active parameters 和实际 active FLOPs，按 node、expert 和 backbone 分项统计。
- 消息投递、`Observe / Update`、selector、状态读写、昂贵激活、继续发送、packing 与 merge/mixer 的单独成本。
- 端到端训练吞吐、`prefill` 吞吐和 `decode` latency。
- 峰值显存、KV/SSM/private state 大小和 optimizer state。
- 跨设备通信字节、静态/动态通信邻接距离、collective 范围和等待时间。
- grouped GEMM/packed kernel 的 tile 利用率。
- 每设备工作量分布、尾部延迟和空闲比例。

系统收益至少要求：

```text
被跳过的昂贵计算与远程通信
>
局部广播 + Observe/Update + selector + 状态读写 + packing + merge
```

逻辑局部 Graph 只有在物理放置和 mixer 范围也受到约束、上述不等式在端到端测量中成立时，才获得真实系统收益。

## 9. 近期实验的软件边界

近期实现应优先建立少量稳定抽象，使两个工作流和关键反事实使用同一套执行语义，而不是把一般 Graph runtime 一次做完：

```text
CheckpointAdapter
    原生模型装载、状态映射和 equality oracle

BranchModule
    单入口、单出口；可为 atomic、serial 或 recursive

MessageProjection
    把 sender 输出映射到固定、有界的 receiver slots；声明局部拓扑和消息形状

ReceiverCell
    分离 Observe、Update、ExpensiveCompute 与 Emit；可承载 Attention、SSM 或 FFN branch

PropagationProfile
    执行 selected-dispatch 或 broadcast-observe，并产生 receive、update、active 与 emit mask

SiblingSelector
    为同一父模块输出 active child ids 和 weights

StateUpdater
    声明 semantic/load state 的更新函数、持久范围和空输入行为

FixedMerge
    固定槽位、merge 范围与 merge 算子

ReceiverState
    receiver-private semantic state；支持 reset/save/reload

SelectorState
    sibling selector 共享的逐序列 load/history state；与 ReceiverState 分开保存

RouteArtifact
    记录每个 Token、父模块、receiver 和递归层级的 receive、update、active、read 与 emit artifact

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

这套底座必须能够在同一 receiver 拓扑上切换两种 propagation profile，保存和恢复持久状态，执行 state knockout，并分开统计轻量更新和昂贵计算。工作流 B 的 `selected-dispatch` 对照与 BO 候选使用同一种 `ReceiverCell` 布局和执行器，只切换声明清楚的传播与状态 profile。

函数保持接入需要显式实现 identity-compatible 或 residual-isolated 接口。若采用零 gate 或零输出投影，还要验证哪些新参数在起点能够获得梯度，并定义可复现的 gate 开启方式。Head-Wise 若从 dense checkpoint 生长，也需要单独说明如何拆分旧权重、设置 mixer 或保持新分支初始中性；Head/Group 切分本身并不自动保持原函数。

近期底座不需要实现 HB-Line executor、一般 event IR、跨设备 allocator 或有环 Graph。

## 10. 首个可交付成果

首个交付不是“训练出完整 Tide”，也不是完成一串串行架构阶段，而是让两条工作流在可信共同底座上同时启动：

1. 选定一个适合快速重复实验的 pre-norm decoder-only 开放权重 checkpoint、训练数据、框架和目标硬件。
2. 完成原生模型的 equality oracle、continued-pretraining 校准和 fresh save/reload 测试。
3. 实现第 9 节的统一接口，使相同局部拓扑能够切换 `selected-dispatch` / `broadcast-observe`、持久/无延迟状态和各类 knockout。
4. 在工作流 A 中建立 dense continued-pretraining 和成熟 FFN flat MoE 强基线；Head-Wise MoE 在基线可靠后作为可选后续。
5. 在工作流 B 中实现一个最小但机制完整的候选：保留 always-on checkpoint，通过中性 residual 接口加入拓扑固定且入度/出度具有统一上界的 receivers、`broadcast-observe`、可延迟读出的私有状态、局部稀疏昂贵激活和声明清楚的 fixed merge；Leaf/Receiver 门控和一层或多层递归由具体命题说明。
6. 为该候选同时保留 matched `selected-dispatch` 开关，以及 inactive-state freeze/clear/shuffle/no-read/reset 等直接反事实。
7. 完成短程训练并输出 correctness、mechanism-use、质量、路由、梯度、状态利用、路径分布和分项系统成本报告。

这个交付的成功标准是实验可重放、候选语义完整、问题可观测且关键反事实可运行；它不要求首轮结果已经证明 Tide 有效。随后根据实际观察定位问题、引入干预、做配对反事实并复现，而不是等待某个预设小阶段通过后才允许使用递归、状态或 Receiver-Gated。

## 11. 远期展望

非平凡的一般 Graph/DAG 天然由具有长度的传播路径构成，实际可扩展结构也很可能出现不同传播距离、逐级展开、空间分区或多尺度组织。它们的长短路径不必来自当前这种规则层次递归，也未必采用相同的 merge 方式；因此不能把当前结构称为一般 Graph 的唯一必经形式。

不过，规则层次化递归仍是值得先行摸索的代表性台阶。它能够从 checkpoint 的稳定接口逐级生长，在 fan-in 和 fan-out 具有统一上界时控制局部候选、传播深度和收拢频率，并自然形成可测的不等长局部 DAG 与按需计算。这既可能带来更高性能上限或更低平均成本，也能为走向更一般局部 Graph/DAG 积累关于路径长度、空间扩展、信用分配和物理放置的经验；前两项收益目前仍是假设，而不是既有结论。按照第 2.3 节的证据推进关系，应先利用该受控结构建立正面信号和失败诊断能力，再承担 line、lattice、mesh、多尺度 backbone 与一般局部 DAG 逐步增加的拓扑和系统自由度。

HB-Line/HB-Lattice 可以进一步把递归或局部分支映射到重复空间切片，使 node 参数和状态长期驻留在局部设备，并通过度有界的固定空间拓扑形成更一般的局部计算介质。一般空间 DAG 可以使用其他来源的不等长路径、显式 allocator 和拓扑序 chunk 执行；含反馈的一般 Graph 还会引入动态 event DAG、边界在途消息和更强状态机制。

这些方向需要额外解决：

- 路径持续限制未来可达集合。
- 跨 Token 控制链与低-span `prefill` 的冲突。
- 边界在途消息和状态 continuation。
- 不同路径长度下的信用分配与路径分布漂移。
- 局部设备放置、通信和负载热点。

近期软件不预先实现一个包罗万象的一般 Graph runtime，但首个完整候选可以联合使用当前命题所需的递归、状态、selector 和 fixed merge。后续结构按“观察问题—定位—干预—配对反事实—确认”的闭环演化，并在局部候选不足以回答问题时逐步引入空间化和更一般拓扑。

## 12. 当前不能主张的结论

- 局部通信 Graph 已经优于 Transformer、Mamba 或 MoE。
- `broadcast-observe` 已经具有 learning、scaling 或端到端系统收益。
- 收到消息、状态发生变化，就等于该状态已被以后有效读出。
- 路径相关的 receiver 历史暴露变薄，必然等于当前 hidden 丢失上下文，或必然伤害任务质量。
- 多父交叉会聚必然恢复全部无损记忆、必然缓解路径漂移，或者在语义上只能采用 `broadcast-observe`。
- 一个联合使用递归、私有状态、selector 和 fixed merge 的候选成功，就分别证明了这些部件都必要。
- 规则层次化递归是一般 Graph 的唯一扩展方式，或它已经解决路径漂移和长路径信用分配。
- fixed merge、always-on backbone 或某一种 mixer 是所有局部 Graph 必需或最优的收拢方式。
- Head-Wise MoE 已经优于 flat MoE、已经消除了端到端通信瓶颈，或者已经自然合入工作流 B。
- 人脑结构证明了 Tide 可训练或高效。
- 收到即更新自动消除了节点饥饿和信用分配问题。
- 只有叶子稀疏就无条件与 MoE 具有相同信用距离。
- 逻辑邻接局部会自动带来物理通信局部和更低延迟。
- 固定空间 DAG 自动得到 Transformer/Mamba 级 node 内 Token 并行。
- 任意 stateful selector 都可以获得高性能 chunk `prefill`。
- 两条研究路线必然汇合。

当前仓库仍处于实验启动期，没有足以支持上述结论的可靠训练结果。README 中出现的收益描述均是待检验命题或设计理由。

## 13. 新开发线程的起点

新线程应先完成以下决策，不要直接实现一般 Graph：

1. 选择首个 checkpoint、训练框架、共享训练数据与目标硬件，并定义两条工作流共同使用的成本口径。
2. 明确原生 equality oracle 和函数保持生长的逐项比较对象，包括 logits、原 cache/state、prefill/decode、chunk continuation、主要梯度和 save/reload。
3. 定义 `broadcast-observe` reference contract：固定空间拓扑、直接邻居范围、receive/update/active/read/emit 的顺序、空消息规则、状态生命周期和 merge 范围。
4. 选定工作流 B 的首个完整候选，包括 branch grammar、receiver 拓扑、门控范围、私有状态、selector 输入与决策、active budget、递归深度和收拢方式。
5. 在同一 receiver 拓扑上定义工作流 B 的 `selected-dispatch` control、inactive-state knockout 和必要的 Leaf/Receiver 直接反事实。
6. 定义 dense、flat MoE，以及可选 Head-Wise MoE 的 capacity/compute/resource-matched 配置。
7. 确定 `BranchModule`、`MessageProjection`、`ReceiverCell`、`PropagationProfile`、`SiblingSelector`、`StateUpdater`、`ReceiverState`、`SelectorState`、`FixedMerge`、`RouteArtifact` 和 `ExperimentLedger` 的最小接口。
8. 建立第一批单元测试、短程 continued-pretraining 协议，以及“正面信号、失败定位、确认复现”各自允许主张什么的报告模板。

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

因此，Tide 不以完整复刻 LH 为目标。LH 的价值是提供“局部通信 + 超稀疏”的历史动机，暴露 selector、状态副作用和动态路径可能带来的困难，并提供可供裁剪、替换和重新组合的机制集合。当前实验保留局部性、层级和稀疏激活作为近期结构约束，并把 `broadcast-observe + persistent state + later readout` 提升为需要正面验证的核心候选。历史负载、恢复量和累计阈值可以为解决饥饿或预算问题进入 selector，但不因 LH 曾包含它们就默认全部启用；任何改变输出的跨 Token 状态都必须承担 reference semantics、可重放、训练归因和 `prefill` span 的验证义务。

### 14.2 从 `prefill = decode` 得到的约束

编译器、ISA、乱序执行和 dataflow 的共同经验是：性能优化必须在明确 reference semantics 的前提下重排、融合或并行计算。对任意一次有限 chunk 执行，数据、状态和控制依赖应能展开成一个有限、依赖完整的 logical event DAG；实现可以改变物理调度，但不能改变该 DAG 所定义的可观察结果。

这给 Tide 两条直接约束：

- 固定空间 Graph 可以是 DAG，节点可以按拓扑序处理整个 chunk；动态物理调度不应进入模型语义。
- 任意 pointer-chasing 式、不可组合的自适应路由链，不存在对所有实例都有效的通用低深度 exact `prefill` 加速。

近期实验因此优先使用 fixed merge、有界度的多跳扩展和只协调固定直接邻居的 selector。selector 可以按完整候选需要读取当前内容、语义状态和逐序列历史激活信息；只要这些输入会改变路由，就必须作为模型 state 明确定义并验证 chunk 等价性。长期不收拢路径和不可组合的逐 Token 控制递推不是固定排在后面的“阶段”，而是只有在能够说明其必要性、语义和 span 代价时才引入的高风险自由度。

### 14.3 人脑调查提供的启发与边界

人脑不是 Tide 的实现模板，也没有已知的 Transformer 式高性能 `prefill`。但神经解剖与神经生理调查提供了若干有价值的结构倾向：

| 观察 | 对 Tide 的启发 | 不能推出什么 |
| --- | --- | --- |
| 神经连接高度局部且结构稀疏，同时存在少量长程投射 | 研究有界度局部 Graph、层级连接和多尺度 backbone | 局部 Graph 一定比当前加速卡上的稠密模型更快 |
| 信号广泛发散、汇聚，并存在大量反馈 | 允许并行分支、固定 merge 和多种长短计算路径 | 应直接复制有环、跨 Token 反馈 |
| 神经元持续积累输入，只在部分时间产生 spike | 把 `Observe / Update` 与昂贵 `Activate / Emit` 分开 | 粗粒度计算 node 等价于单神经元 |
| 局部抑制、恢复和稳态机制限制长期过强或沉默 | 研究局部预算、恢复量和慢速负载状态 | 简单按历史次数轮换就能形成有用专门化 |
| 大尺度功能网络多表现为增益和耦合变化，而非整个脑区严格开关 | 保留 always-on backbone、重叠分支和短 merge | hard selector 必然不可用 |
| 学习依赖局部可塑性、调质、重放和多时间尺度机制 | 关注局部辅助信号和延迟信用 | 人脑已经解决了数字网络中的精确全局信用分配 |

最重要的边界是：微观 spike 稀疏和平滑的群体表示，并不自动迁移到粗粒度 node 的 0/1 激活。一个 Tide node 可能包含完整 Attention、FFN 或 SSM，开关它造成的语义跳变远大于单个神经元。增加 node 数量也不会自动得到人脑式平滑性；还需要重叠表示、residual、固定 merge、短控制寿命和合适的训练机制。

## 15. 上游研究笔记与外部参考

本启动文档在仓库内自足保存 checkpoint 生长线的逻辑纲要、当前实验选择和执行边界；读者理解第 1 至第 13 节不需要先打开 ObsidianVault。一般定义、完整证明、长期历史和跨实验研究结论仍由 ObsidianVault 维护，后续 agent 需要这些背景时应从下列 GitHub 文档读取，而不是依赖本机绝对路径：

- [TIDE 研究线总入口](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/README.md)：正式命名、对象边界、战略路线、文档地图、当前主张边界与阅读顺序。
- [TIDE Architecture / Network：模型架构与训练](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/tide-model-architecture-and-training.md)：checkpoint 生长、递归分支、HB-Sliced/HB-Line、selector 与训练稳定性。
- [TIDE 数学基础](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/tide-mathematical-foundations.md)：`StepTransition`、`prefill = decode`、logical event DAG、一般空间 DAG 与函数保持生长。
- [Adaptive routing prefill lower bound](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/adaptive-routing-prefill-lower-bound.md)：不可组合自适应路由链的反向复杂度边界。
- [TIDE 背景、历史谱系与参考](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/tide-background-history-and-references.md)：ISA/编译器/dataflow 谱系和完整人脑传播调查。
- [TIDE Engine：runtime 验证与状态](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/tide-runtime-validation-and-status.md)：Engine/runtime contract、LH 映射、artifact equality 与工程状态。
- [TIDE 统计力学与信息动力学](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/tide-statistical-mechanics-and-information-dynamics.md)：粗粒化、路径相关性和统计力学类比及其边界。

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
