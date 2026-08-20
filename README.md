# Tide 实验仓库启动文档

> 状态：项目启动草案
>
> 日期：2026-08-20
>
> 目标读者：即将开始实现、训练与验证 Tide 的研究和工程人员
>
> 上游研究总入口：[ObsidianVault / TIDE](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/README.md)
>
> 当前 checkpoint 生长实验契约：[ObsidianVault @ c3548d1](https://github.com/ZichaoLong/ObsidianVault/blob/c3548d18330a84bba95243bc8852861c9561554e/20-tide-decentralized-neural-network/tide-checkpoint-growth-experiment-contract.md)

## 1. 研究逻辑总览

Tide 暂沿用全称 `Token Inference Decentralized Engine`。本仓库研究的是模型架构、训练语义和可执行实验，不把名称中的 `Engine` 限定为某一个既有 runtime。本文把能够接收局部上游消息、并拥有自身参数或状态的下游模块称为 `receiver`；把 Attention readout、FFN、大型 SSM 更新等主体计算称为“昂贵计算”，以区别于较轻的消息观察和状态写入。为使以下推导既可读又不把猜测写成事实，本文使用五类判断：

| 类型 | 在本文中的含义 |
| --- | --- |
| **已有证据** | 来自成熟模型、公开规模实验或本仓库可形式检查的性质 |
| **条件性刚需** | 在明确接受一组目标约束以后才能推出；不表示具体实现已经有效 |
| **工程护栏** | 为降低 checkpoint 生长和训练风险主动采用的约束；不声称是所有 Tide Graph 的唯一解 |
| **研究赌注** | 有明确动机、值得优先正面验证，但当前没有本仓库可靠实验结论的机制假设 |
| **可选干预** | 在完整候选中可以预先采用，或在观察到相应问题后引入的结构工具 |

> **当前证据状态：** 本仓库尚未产生足以证明 Tide learning value、scaling value 或系统收益的可靠训练结果。Flat MoE 的现实成功、checkpoint 函数保持和若干有界图性质可以作为已有依据；`broadcast-observe`、私有状态、多次局部选择和交叉会聚的净收益仍是待验证内容。

### 1.1 一眼看懂的两条逻辑链

第一条是从终极目标到可推进结构的**目标约束链**：

```text
【已有证据】Flat MoE 已证明：很大的潜在参数容量
             可以与每 Token 少量昂贵激活同时成立
        ↓ 但它没有验证固定有界的局部传播和长期本地状态
【目标约束】容量继续随节点数增长 + 连接度不随总规模增长
             + 不依赖全局 router + 输入自适应 + 全局昂贵计算超稀疏
        ↓
【条件性刚需】有界度、多跳、逐级的局部容量扩展
             + 多次有界局部选择或等价的分布式路由
             + 对活跃前沿、昂贵计算和传播深度的显式预算
        ↓
【新增风险】早期选择持续影响更长路径，带来路径输入漂移、
             更长控制信用链，以及私有状态可能引入的延迟信用链
        ↓
【工程护栏】从正面 checkpoint 生长，保留 always-on backbone，
             使用有界局部 selector、函数保持接口和 fixed merge
```

严格推出的是“有界度的多跳局部扩展”，不限定它必须是一棵规则树。规则层次化递归是本仓库优先采用的工程前置：它离已有 checkpoint 最近，便于控制发散、收拢和证据变量；随后可以逐步进入 line、lattice、mesh、多尺度 backbone 和其他局部 DAG。

第二条是解释为何重点研究 `broadcast-observe` 的**机制假设链**：

```text
【结构事实】多级 selected-dispatch 使固定 receiver 只看到
             与自身路径前缀相符的消息历史
        ↓
【待验证风险】receiver 的历史暴露变薄、私有状态彼此碎片化，
             可能削弱局部上下文记忆和后续处理
        ↓
【研究赌注 A】broadcast-observe + 私有状态 + Update/Compute 分离：
               直接 children 即使本次不做昂贵计算，也获得观察和写入机会
【研究赌注 B】有界的多父局部会聚或交叉传播：
               不同路径交换已经处理过的局部摘要
        ↓
【最终待验证】这些机制能否产生真实 learning/scaling value，
             并覆盖新增状态、通信、selector 和调度成本
```

这里的“历史暴露变薄”不表示每次分叉都会把当前 hidden 中的信息机械地除掉一部分。一个 child 仍可能收到已经整合完整前缀的 parent hidden；真正由结构直接造成的是某个 receiver 的私有状态只记录了它实际收到的 route-conditioned 历史。该差异是否伤害任务相关记忆，以及 BO 或交叉会聚是否值得成本，才是实验问题。

两条链不能互相替代：第一条说明为何最终需要探索多跳局部扩展；第二条说明为何本仓库主动把 BO 选为首要正面假设，但没有证明 BO 是实现 TIDE 目标的唯一必要机制。

### 1.2 当前组件在逻辑中的位置

| 组件 | 当前定位 |
| --- | --- |
| 统一有界的局部入度/出度、多跳扩展 | 在上述终极目标约束下的条件性刚需 |
| 多次局部选择与全局 active budget（每 Token 的总激活预算） | 在“无全局 router、输入自适应、昂贵计算超稀疏”前提下的条件性刚需 |
| 规则层次化递归 | 从 checkpoint 进入 line/lattice/mesh 和一般局部 DAG 的首选工程与证据前置，不是唯一数学形式 |
| `broadcast-observe` | 工作流 B 的定义性核心和当前首要研究赌注，不是已证明的唯一解 |
| 私有持久状态与 later readout（以后真正读出） | 验证“未激活期间积累的局部记忆以后有用”这一命题的必要部件 |
| post-Update selector（先更新 receiver 状态、再选择昂贵计算） | 只在验证“当次 Observe 改变当次选择”时需要；BO 也可以使用 pre-Update/content-only selector |
| always-on backbone、fixed merge（固定位置收拢） | 首个 checkpoint-growth 完整候选的强工程契约；分别保留稳定主路径、限制显式控制寿命 |
| 多父交叉会聚、不等长路径和更一般空间化 | 可进入完整候选，也可由实际失败牵引的可选干预 |
| 纯 FFN 递归 | 不充分检验私有序列记忆，但仍是区分条件计算收益与状态收益的重要对照 |

### 1.3 本仓库要回答的问题

Tide 的长期目标是研究一种自回归神经网络架构，使它同时具有：

1. **局部通信**：一个计算节点只与固定、数量有界的邻近节点通信，而不是让每个 Token 都可以被动态发送到任意全局节点。
2. **结构稀疏与激活稀疏**：静态 Graph 本身不是全连接；一次输入只执行全部潜在昂贵计算中的一小部分。
3. **可训练性**：动态选择不会使路径漂移、节点饥饿和长距离信用分配严重到无法稳定训练。
4. **`prefill = decode`**：逐 Token `decode` 与任意合法 chunk 的 `prefill` 实现相同的单序列 reference semantics。
5. **实际系统收益**：被跳过的工作足够昂贵，能够覆盖 selector、状态更新、packing、通信和负载不均衡的成本。

这五项存在天然张力。被跳过的模块必须足够大，稀疏执行才有收益；但单个模块的语义贡献又需要足够平滑、重叠或及时 merge，动态换路才不至于使训练失稳。Tide 当前最核心、也最可证伪的总假设是：

> 在当前或未来的模型规模与硬件上，存在一种有实际意义的中间粒度：node 内部仍使用高效稠密 kernel，node 之间采用固定局部连接和动态稀疏激活，并且模型质量、训练稳定性与端到端性能可以同时成立。

本仓库不从最一般的 Graph 开始实现，也不再预设一条必须依次通过的架构阶段链。Checkpoint 生长线近期并行推进两个实验工作流：

```text
工作流 A：dense checkpoint / flat MoE 基线与校准
工作流 B：从 checkpoint 中性生长，正面验证 broadcast-observe 局部计算介质
```

工作流 A 提供 correctness oracle、成熟训练配方和强稀疏对照；工作流 B 从一开始就允许联合使用有界递归、私有状态、局部 selector、always-on backbone 与 fixed merge，寻找一个有成功可能的完整候选。具体结构随后由实验观察和失败诊断继续演化，而不是由预先写死的阶段编号决定。

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

MoE 已经经验性证明了一种重要平衡：总参数容量可以随专家数增加，而每 Token 只执行少数昂贵专家。但它仍有路由漂移、selected-only feedback、专家饥饿、负载不均衡和 all-to-all 通信成本。Tide 希望用层级局部候选和固定局部通信替代部分全局 dispatch，但这会放弃 MoE 的两项重要优势：每层都面对该层自己的完整候选集合，以及一次选择立即结束其显式路径身份。

因此，标准 MoE 必须成为 Tide 的主要对照组，而不是只与 dense Transformer 比较。

### 2.3 输入性逻辑链：从目标约束到工作流 B

本节把第 1 节的两条短链展开为一份用于产生实验候选的输入性推导。它允许使用外部经验、条件推导和有方向的工程直觉，但每一步都说明自己属于“已有证据、条件性刚需、工程护栏还是研究赌注”。它的作用是解释为什么值得做当前实验，不是提前宣布实验结论。

#### 2.3.1 从 MoE 现实基线到有界多跳扩展

**1. Flat MoE 给出了重要的正面起点【已有证据】。** 标准 flat MoE 已经经验性验证，大量潜在参数可以与每 Token 少量昂贵激活同时规模化成立。它因此是 Tide 的强基线，而不是需要先被否定的旧方案。它没有直接验证的是固定有界的局部邻接、多跳局部传播和 receiver 私有状态；分组、复制、静态放置或局部 router 可以让 MoE 更局部，但也会改变标准 flat MoE 的候选范围、路由和成本结构。

**2. 有界度、多跳、逐级的局部扩展是条件性刚需【条件性刚需】。** 若希望潜在容量随模块数持续增长，同时固定入口数量、要求节点入度和出度具有不随总规模增长的统一上界，不允许一个全局 router 直接访问任意远端节点，并希望输入仍有机会自适应访问随规模增长的潜在容量，那么不断增长的候选容量只能通过某种多跳、逐级或空间局部的方式到达。若最大 fan-out 为 $\Delta$，半径 $D$ 内至多经过 $1+\Delta+\cdots+\Delta^D$ 个可达槽位；当 $\Delta>1$ 时约为 $O(\Delta^D)$，当 $\Delta=1$ 时则随 $D$ 线性增长。容量继续扩展时，传播深度、空间直径或入口数量中至少一项必须增长。

严格推出的是“有界度的多跳局部扩展”，不是某一种规则树。Line、lattice、mesh、多尺度 backbone 和其他局部 DAG 都可以满足这一结构方向。本文仍把规则层次化递归视为 checkpoint 生长线的工程与证据前置，因为它能在保留原 backbone、接口和 fixed merge 的情况下，逐级验证多跳传播、路径长度、稀疏预算和信用分配，再放开更多拓扑自由度：

```text
规则层次化递归与局部 fixed merge
-> line / 重复空间切片
-> lattice / mesh
-> 多尺度 backbone 与更一般局部 DAG
```

这不是严格的数学包含序，也不要求未来机械地依次实现每一项。越向后，邻接、放置、路径长度、merge、状态所有权和调度的自由度越大，结构通常也离已有 checkpoint 更远，正面验证、失败归因和工程推进都更困难。规则递归的价值正在于用一个较规则的局部 DAG 先摸清这些问题。

**3. 输入自适应的超稀疏执行需要多次局部决定和显式总预算【条件性刚需】。** 在“不使用全局 router、候选只能局部访问、选择又要随输入变化”的前提下，一次全局 $N$ 选 $M$ 会自然分解为多次有界局部选择，或语义等价的分布式路由协议。这里的 selector 可以共享参数，也可以由 receiver proposal 和局部 sibling arbitration 组成，不要求每个物理节点拥有完全独立的网络。

局部候选变少可以降低单次打分和协调范围，但不会让选择困难消失：早期决定可能级联，远端候选的信息也不再一次可见。静态 fan-out 有界更不自动等于全局超稀疏；还必须显式限制每层或每 region 的 active budget、每 Token 昂贵计算总量、Emit 边数和最大传播深度，并把 Observe / Update 成本一并入账。

#### 2.3.2 相对 flat MoE 新增的训练风险与近期护栏

**4. 多级选择可能放大路径漂移和信用分配困难【风险判断】。** Flat MoE 已经存在 hard Top-K、selected-only feedback、router drift 和专家饥饿，但通常在一个子层内立即 merge。Tide 若让早期选择继续限制后续若干节点的可达集合，就延长了显式控制路径的寿命；若再加入私有状态，还会出现一次写入在以后 Token 才被读出的延迟信用链。这些结构差异使风险更值得警惕，但并不预先证明模型一定不可训练。成熟 MoE 的正面结果也提供了间接信心：离散稀疏选择本身并非不可克服。

**5. Fixed merge 是限制控制寿命的直接办法【工程护栏】。** 固定收拢点使旧分支身份在已知位置结束，并恢复共同接口。它可以限制一次选择继续约束未来候选的距离；若 merge 使用归一化、小 residual 或稳定槽位，还可能减小换路造成的表示跳变。它不会自动降低 route churn，也不会删除已经写入 private state 的跨 Token 影响。

**6. Always-on backbone 是 checkpoint 生长的稳定锚点【工程护栏】。** 保留原 checkpoint 主路径、让新增分支以中性 residual 方式接入，可以在起点保持原函数，并保证至少存在一条不被 selector 切断的前向和反向拓扑路径。它不保证新分支具有足够梯度，也可能掩盖新机制没有被真正使用，因此必须配合 mechanism-use 检验。Fixed merge 在一般结构上并不要求 backbone；二者在 checkpoint 生长中组合使用，是因为一个提供稳定锚点，另一个限制局部控制寿命。

**7. 局部 selector 既可能更容易，也可能更困难【研究赌注】。** 候选集合变小可以降低单次打分、Top-K 和局部通信协调成本，并允许参考成熟 MoE 的语义路由、负载均衡、shared expert 和通信优化经验；但多级决定会带来串行控制、早期错误累积、局部视野和更长归因链。当前不预设哪一方向占优。Selector 可以读取当前内容、pre/post-Update semantic state 和可选的逐序列历史激活状态；任何改变模型输出的历史状态都必须成为可重放的 reference state，而不能偷用实时设备负载。

#### 2.3.3 路径相关历史暴露：结构事实与待验证伤害

**8. 多级 selected-dispatch 会造成路径相关的历史暴露变薄【结构事实 + 研究赌注】。** 一个固定 receiver 的私有 KV/SSM/summary state，只能记录实际路由到该节点的消息；不同 receiver 因而可能保存不同的历史子集。建议把三个容易混用的问题分开：

- **历史暴露变薄**：一个 receiver 实际看过多少 Token 或父消息。
- **私有状态碎片化**：不同 receiver 保存了哪些不同的 route-conditioned 历史。
- **表示压缩或任务相关信息损失**：当前 fixed-width hidden 是否仍保留完成任务所需的前缀信息。

前两项由传播结构直接造成；第三项及其质量伤害需要实验。一个 active child 通常收到完整的 parent hidden，而不是因为有 $B$ 个 siblings 就只拿到 $1/B$ 的信息。选择性历史也可能促进专门化，所以这里要验证的是“路径相关私有历史是否成为瓶颈”，而不是预设所有分叉都会丢失上下文。

在一个简化的均匀 $B$ 叉、每级选择 $K$ 个 child 的树中，深度 $d$ 的固定节点在 `selected-dispatch` 下收到消息的比例约为：

$$
\left(\frac{K}{B}\right)^d.
$$

这个量刻画 receiver exposure，不刻画当前 hidden 的信息量。

**9. 纯 FFN 递归不会产生同一种私有历史碎片化，但不能据此淘汰【重要对照】。** Stateless FFN 每次变换当前 parent hidden，没有 branch-private 的跨 Token KV/SSM 历史，因此不直接承担上述状态暴露问题；它也不能重新读取当前 hidden 之外的序列历史。不过，纯 FFN 仍可增加条件深度、参数容量和非线性计算，并非没有意义的重复加工。`Attention/SSM -> FFN` 的常见配比是很强的 checkpoint 工程先验，不是架构定理。纯 FFN 路径不必成为第三条主线，但应保留为区分“条件计算收益”和“私有状态收益”的结构对照。

**10. 若历史暴露确实形成瓶颈，有两类优先补救方向【研究赌注】。** 第一类是私有状态与 Update/Compute 分离：receiver 即使本次不执行昂贵 readout/FFN，也可以保存已经到达的消息。第二类是有界的多父局部会聚或交叉传播：不同路径在 fixed merge 前后交换经过处理的摘要，扩大 receiver 的局部 source coverage。Shared regional memory、周期性 backbone reinjection、merge 后重新发散、较大 $K$、soft routing 和状态同步也都是替代或互补机制，因此历史暴露问题即使成立，也不会只剩一种解法。

#### 2.3.4 为什么仍把 broadcast-observe 选为工作流 B 核心

**11. 私有状态与 Update/Compute 分离不在一般意义上推出 BO，但工作流 B 主动采用一个更强 profile【项目选择】。** `selected-dispatch` 也可以让选中节点维护持久状态，模型也可以使用共享区域状态或周期写入。只有增加“active sender 的全部固定直接 receivers，无论本次是否做昂贵计算，都应获得观察和私有状态写入机会”这一要求时，`broadcast-observe` 才按定义成为必要的传播 profile。本仓库选择正面验证这一更强机制，因为它直接表达“消息可见性与昂贵激活分离”的局部计算介质。

BO 有两条必须分开归因的作用路径：当次 Observe / Update 可以改变 receiver proposal 或当前 selector；未激活时的写入也可以在以后真正 readout。前者不要求跨 Token 持久状态，后者才检验延迟模块记忆。若从旧 checkpoint 或 selected 模型迁移，新增状态必须先不被输出读取或通过零 residual 隔离，才能声称初始函数保持。

**12. BO 直接改善的是当前 active parent 下一跳的 exposure【结构性质】。** 在上面的均匀模型中，若 active parent 向全部 $B$ 个 children 发送、只有 $K$ 个执行昂贵计算并继续 Emit，则深度 $d$ 的固定 receiver 收到消息的比例约为：

$$
\left(\frac{K}{B}\right)^{d-1}.
$$

因此 BO 消除了本节点这一层 child selection 造成的 exposure loss；它没有绕过从未激活的祖先，也不保证写入内容有用。若 inactive receiver 继续廉价转发到更深处，覆盖会进一步增加，但消息和 Update 成本也可能迅速接近稠密传播。

**13. 多父交叉会聚可能更广泛地恢复局部 source coverage，但不保证完整无损记忆【研究赌注】。** 有界 fan-in、有限消息宽度、有限传播 hop 和稀疏 Emit 决定了末端不能免费获得所有原始历史；若要求无损保留任意多路输入，相应带宽或状态容量也必须增长。交叉边可以把递归树逐步变成 lattice/line-like 的局部 DAG，但不会仅因“多次会聚”就自动成为具有特定切片、邻接和状态语义的 HB-Line。

**14. 优先在完整 `Attention/SSM -> FFN` 之后借出消息，是合理的首个接口选择【工程偏好】。** 这样一个 receiver 对应一个完整旧 block，较容易复用 checkpoint、保持清楚的 Update/readout/compute/emit 边界，也避免连续堆叠多个 memory readout 而没有中间信息处理。它携带的是经过上下文读取和 FFN 加工后的表示；在 Attention/SSM 后、FFN 前接入则更早交换 memory readout。两者优劣尚无结论，可以把后者和低秩摘要接口作为对照，而不把“Attention 与 FFN 必须 1:1”写成定理。

#### 2.3.5 会聚的双向作用与多父语义

**15. “会聚缓解路径漂移和信用分配”只对一部分会聚形式方向明确【双向假设】。** 需要区分：

- **交叉耦合（cross-coupling）**：不同分支交换消息，但各自的路径身份继续存在。它扩大 receptive field，也可能把一路漂移传播给更多节点，增加动态 fan-in、输入组合变化和归因歧义。
- **固定会聚（fixed convergence/reset）**：多路分支进入共同、规范化接口，旧路径身份在此终止。它可以缩短控制寿命、提供冗余或更短的梯度路径，并在归一化或小 residual 条件下减小换路跳变。

因此，“会聚可能缓解路径漂移和信用分配”的直觉对第二类更成立：fixed convergence 有理由缓解控制寿命和单一路由敏感性；普通 cross-coupling 的净效果则必须实测。会聚本身通常不降低 selector 的 route churn，私有状态的跨 Token 写读链也不会因一次数值 merge 自动消失。

**16. 多父局部 DAG 与 `selected-dispatch` 可以兼容【接口事实】。** 每个 parent 可以只向自己选中的 children dispatch，receiver 再确定性聚合实际收到的零到多条消息。它需要额外声明 inbox 何时完整、消息顺序或结合归约、重复消息、state commit、空消息和多个父节点之间的预算仲裁。BO 与固定局部多父接口可能更自然，也更符合本仓库希望检验的消息可见性，但不是由“多父节点”在数学上强迫出来的；相同多父拓扑上的 selected control 仍应保留为反事实。

#### 2.3.6 工作流 B 的最终落点

**17. 工作流 B 的核心任务是寻找一个 BO 完整成功候选，而不是预先证明 BO 唯一必要【最终待验证】。** 当前首个候选的设计契约是：

```text
完整保留的 checkpoint / always-on backbone
+ 固定、有界的局部 receivers
+ broadcast-observe
+ 可延迟读出的 receiver-private state
+ 局部 selector 与显式 active budget
+ 少量 receiver 的昂贵 Compute / Emit
+ 声明清楚且有界的 fixed merge
```

规则递归可以从首个候选开始联合使用，并且最终必须进入多跳容量扩展的 scaling 验证；多父交叉会聚则可以预先进入一个有充分设计理由的候选，也可以在实测到路径历史隔离后作为干预。一个组合候选成功只证明“存在这个组合可以工作”，不单独证明每个部件必要。

最终主张按五层递进：先证明 reference semantics 和 checkpoint 生长正确；再证明 Observe/proposal 或未激活状态写入确实被使用；再证明相对 matched `selected-dispatch` 和成熟 flat MoE 具有 learning value；然后证明容量增长时昂贵激活、消息与状态成本仍保持稀疏可控；最后才证明物理通信和端到端时间、能耗具有系统收益。

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
├── 工作流 A：dense / flat MoE 基线与校准
│     ├── dense continued-pretraining
│     ├── 成熟 flat MoE reference recipe / 原生实现复现
│     └── 函数保持的 checkpoint-grown flat / Group-receiver 对照
└── 工作流 B：函数保持的 residual growth
      -> broadcast-observe 局部计算介质候选
         ├── 有界局部 fan-out 与递归分支
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

考虑一个父模块，其输入和输出空间相同。父模块包含 always-on 主分支 $B_0$ 和有限个候选 residual 分支 $B_1,\ldots,B_N$。先用无状态简写只描述当前 Token 的昂贵分支输出与 merge：selector 为输入 $x$ 选择集合 $A(x)\subseteq\lbrace 1,\ldots,N\rbrace$，固定 merge 为：

$$
T(x)=B_0(x)+\sum_{j\in A(x)}g_j(x)B_j(x).
$$

在有状态 `broadcast-observe` 中，Receive / Update 可以先执行，激活集合与分支 readout 更一般地写成 $A_t=R_\theta(x_t,\{S_{j,t}^{+}\})$ 和 $B_j(x_t,S_{j,t}^{+})$。上式不试图省略这段状态转移，只定义各 active 输出如何回到固定接口。

“固定”表示分支的入口、出口、merge 位置和 merge 算子在模型结构中预先确定；动态变化的只有激活集合和可选权重。短分支不能越过 merge 提前修改外层状态，长分支也不能在 merge 后追赶并改写同一个输出。

该结构具有四个近期优势：

1. 原 checkpoint 主路径可以完整保留。
2. 新分支可以通过零输出或显式代数等价的 clone、缩放与 merge 构造，在初始点保持原函数；任意复制后重组不能自动声称等价。
3. 每个父模块只连接自己的有限子分支，逻辑 fan-out 有界。
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

任何包含传播边的非平凡 Graph 都由具有长度的传播路径构成，很多可扩展 Graph 也会表现出某种逐级展开、空间分区或多尺度组织；不同路径是否等长、如何形成以及何时收拢，则不一定来自当前这种规则递归。规则的层次化递归因此不是一般 Graph 的唯一来源或形式，却是一个值得先行摸索的受控候选：它可以从现有 checkpoint 的稳定接口逐级生长，在有界 fan-out 下显式改变传播距离，并把不等长局部 DAG、按需计算和 merge 频率放进同一个可实验框架。在本仓库的推进语境中，它既是通向更一般局部 Graph/DAG 的重要桥梁，也是进入 line、lattice、mesh 和多尺度空间结构之前的证据与工程前置。

### 4.3 为什么它具有局部通信倾向

每个 selector 只管理同一父模块的兄弟分支；每个子分支只与父入口、内部子节点和父 merge 通信。若递归最大深度、每级 fan-out 和 Top-K 都有固定上界，则逻辑连接度保持有界，分支不需要访问全局任意 expert。

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

正面探索可以把 `broadcast-observe` 与有界递归、持久状态和 fixed merge 组成一个 coherent bundle；关于 `broadcast-observe` 本身的因果结论，则仍需在相同拓扑、状态容量、昂贵模块和 active budget 下，与 `selected-dispatch` 做直接反事实。还要注意：只有 active sender 的直接静态后继收到消息；若某个上游从未激活，更远处节点不会因“broadcast”而自动看到全局数据。

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

- 候选集合按父子层级组织，而不是一个平铺全局 expert 集合。
- selector 只在各末级兄弟集合内局部选择；内部父节点仍全部常亮，不在本级裁剪激活子树。
- 分支可具有不同的有限串行深度。
- 逻辑拓扑更适合映射为固定局部通信。

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

### 5.5 Head/Group-wise：一种规则的局部发散—收拢骨架

这里需要区分三种容易被统称为 Head-wise 的设计：

| 设计 | 发散与候选范围 | 状态 | 收拢 | 与 `broadcast-observe` 的关系 |
| --- | --- | --- | --- | --- |
| 原始 Group-wise FFN MoE | 将 hidden channel 分组；所有 groups 参与，每组从私有 FFN expert pool 中选择 $K_e$ 个 experts | FFN 没有天然持久状态 | group 内 expert merge，随后 concat / mixer | expert 级仍是 `selected-dispatch`；group 级没有稀疏门控，不等于 `broadcast-observe` |
| Attention head-group | 将 Attention heads 组织为固定 groups | 可以定义或复用与 group 关联的 K/V；私有程度取决于 MHA、GQA、MLA 等参数化 | 固定 head slots 与输出投影 | 普通 Attention 默认所有 groups 都 readout；receiver 化以后才成为候选 BO 载体 |
| Tide Group-receiver cell | 固定、有界的 group receivers；可由 Attention、SSM、FFN branch 或其组合实现 | receiver 私有 semantic state；共享 selector 可另有 sibling-level load/history state | fixed merge、region mixer 或分层收拢 | 明确分离 group 级 Observe / Update、昂贵激活和 Emit |

原始 Head-wise MoE 提议更准确地说是一种新的 Group-wise FFN 算子：它同时改变输入因子化、每组算子宽度、全局 active expert 数和跨组表达耦合，并不是把同一个全局 `N 选 M` 原样拆成几个等价的局部选择。它提供的关键启发，是固定的 group layout、私有候选池、输出 slots 和 mixer 所组成的规则发散—收拢骨架。若每个 group 对每个 Token 都执行自己的 expert Top-K，则 group 层面并不稀疏，也没有验证“未激活 group 仍能观察并积累状态”。

在该骨架上再增加 group 级 selector，得到一个新的 **Group-receiver BO** 设计。需要把 group 级预算 $K_g$ 与 active group 内部可选的 expert 级预算 $K_e$ 分开记录：

```text
parent produces message
-> fixed projection/slice into G receiver slots
-> every receiver observes, updates private semantic state, and produces a proposal
-> one shared local selector, optionally with sibling-level history, chooses K_g active groups
-> only active groups run expensive Attention / SSM readout / FFN
   or run K_e experts from a private local pool
-> active groups emit and may continue recursively; inactive groups stop after Update
-> fixed slots -> region mixer / fixed merge -> stable backbone or next region
```

在这个组合中：

- Head/Group layout 决定如何发散、receiver 与局部候选如何分区，以及参数和 private semantic state 放在哪里。
- `broadcast-observe` 决定所有固定 receivers 都能看到消息，并可在没有执行昂贵计算时更新 private state。
- shared local selector 只在有界 sibling/neighbor 集合中分配 $K_g$ 个昂贵激活和继续传播预算；历史负载若启用，属于这个共享 selector 的逐序列 state，而不是每个 receiver 的 private semantic state。
- fixed merge 或 mixer 决定信号在哪里收拢，并为 checkpoint 生长提供稳定接口和较短的显式控制路径。
- 层次化递归使潜在容量可以在单次局部候选规模有界时继续增长，并容纳不同传播深度的路径。

Attention head-group 只有在明确 group 与 K/V 的所有权后才能充当独立有状态 receiver；GQA/MQA/MLA 中的共享 K/V 不能自动视为 group-private state。K/V projection、写入带宽和显存都属于 Observe / Update 成本，不能预先假定为轻量。

为使 Head/Group-wise 真正进入两条工作流，而不只是名义上的可选项，本仓库建立一对共享骨架的配置：

1. **工作流 A：Group-receiver selected control。** 使用 group-level `selected-dispatch`，只有 active groups 收到消息并执行同一 Update；递归拓扑和 merge 位置与配对的 B 配置一致。
2. **工作流 B：Group-receiver BO candidate。** 在同一骨架上使用 `broadcast-observe`，让 inactive groups 也收到消息和执行同一 Update；其他执行语义与配对的 A 配置一致。

这两个配置必须保持 group 数、input slice/projection、固定 slots、昂贵算子类型与宽度、Update/readout 函数、state 形状与生命周期、$K_g$、递归拓扑、mixer/merge 位置与范围和物理放置一致，才能作为传播与状态 profile 的直接反事实；核心差异只是 inactive receiver 是否 Receive / Update。最简单的配对版本中两边都立即 merge；带额外递归的 BO 可以作为完整组合候选，但要归因于传播 profile 时，必须同步建立相同递归拓扑的 selected control。核心 matched pair 的 selector 应只读取 parent/current content 或 pre-Update state，必要时 replay 同一 route；读取 post-Update proposal/state 的 BO 变体在组内另做“读取 vs 忽略”对照。若一边使用 FFN、另一边改用 Attention/SSM，或同时改变 group layout 与 mixer，它们只能是两个组合候选，不能称为 matched pair。原始“所有 groups 都执行、每组内部 expert Top-K”的 Group-wise FFN MoE 仍可在工作流 A 中作为辅助结构对照，用于单独判断因子化、私有 expert pool 和 mixer 的作用。

Head/Group-wise 只描述逻辑结构，不自动保证物理局部。receiver、私有参数和状态还需要静态共置，parent 只能向固定邻近 region 发送，selector 只能协调局部候选。全维 dense mixer 可以在 checkpoint 生长初期作为表达恢复接口；但若每次收拢都依赖跨全部设备的全局 collective，它只能证明局部分支或稀疏计算有效，不能证明完整的去中心化通信有效。后续可以比较 region 内 dense mixer、低秩连接、tree/hierarchical mixer、邻居 mixer 和低频全局 merge。

只有当研究问题独立变成“即使 `broadcast-observe`、持久状态和递归传播无效，原始 Group-wise FFN MoE 是否仍能在质量或系统效率上替代 flat MoE”时，才把它升级为第三条并行线。所有参数与 FLOPs 对照都必须按 group 数、expert width、$K_g$、$K_e$、router、state、mixer 和 Observe / Update 成本重新计算；Head-wise 的命名本身不保证等参数、等 active FLOPs 或通信清零。

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

### 6.2 工作流 A：dense / flat MoE 基线与 Head/Group 校准

工作流 A 内部包含三个相互服务的对象：原生 dense correctness 与 continued-training oracle、成熟 flat MoE 强基线，以及和工作流 B 共用骨架的 Group-receiver selected control。它们属于同一基线与校准工作流，不是工作流 B 之前必须串行完成的三个阶段。

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

#### 6.2.3 Group-receiver selected control

按照第 5.5 节的共同骨架，工作流 A 同步建立只有 active groups Receive / Update 的 group-level `selected-dispatch` control。它与工作流 B 的 Group-receiver BO candidate 共享 group layout、input slice/projection、固定 slots、昂贵算子、Update/readout、state 生命周期、$K_g$、递归拓扑、mixer/merge 和物理放置，并使用相同或 replay route 来建立最干净的传播反事实；最简单版本两边都立即 merge。

原始 Group-wise FFN MoE 也可以作为辅助结构对照复现，但它让所有 groups 参与、只在 group 内做 expert Top-K，不能替代上述 group-level matched control。

### 6.3 工作流 B：`broadcast-observe` 完整候选

工作流 B 不从无状态 `selected-dispatch` 的空壳开始，而是直接构造能够真实检验核心命题的最小完整候选。当前首个完整候选把下列机制作为设计契约；这表示它们共同组成要寻找的正面候选，不表示它们已经被证明对所有 Tide 架构必要：

- 完整保留的 checkpoint / always-on backbone。
- 通过中性 residual 接口生长的有界 fan-out 分支。
- 至少一组与工作流 A 共享布局和昂贵算子的 Group-receiver BO candidate，形成 Head/Group-wise 的跨工作流直接反事实。
- active sender 向全部固定局部 children 发送；所有实际 receivers 执行声明的轻量 Observe / Update。
- 能在未执行昂贵计算时写入、并在以后真正读出的私有 KV、SSM 或 summary state。
- 读取当前内容、语义状态和可选逐序列激活历史的局部 selector。
- 少数 receivers 执行昂贵 Attention、FFN、SSM readout 或私有 MoE，并继续发送。
- 在声明的有界深度内进行 fixed merge，或者回到稳定的 region/backbone 接口。

一层或多层受控递归、Leaf-Gated/Receiver-Gated、多父交叉会聚和不等长路径，是这个核心候选可以从一开始联合采用的结构坐标，不要求等待更小阶段依次通过。首个最小配对实验也可以立即 merge，以干净地测量 BO；但若最终要支持“容量在有界局部连接下继续扩展”的 TIDE 主张，后续 scaling 实验必须进入多跳递归或等价的空间扩展，而不能永远停留在一跳 BO。

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

首个交叉接口可以优先放在完整 `Attention/SSM -> FFN` receiver 的 Emit 之后，以保持旧 block 边界；Attention/SSM 后、FFN 前和低秩 summary 则作为可比较坐标。所有方案必须继续满足统一有界 fan-in/fan-out、有限 DAG 顺序和显式 active/message budget，并分别统计新增消息、等待和状态成本。同一多父拓扑仍应能够切换 `selected-dispatch` / `broadcast-observe`，避免把拓扑变化和传播 profile 混成一个因果结论。

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
| ungrouped vs 原始 Group-wise FFN MoE | 输入因子化、私有 expert pools 和跨组 mixer 是否有结构收益 |
| matched Group-receiver `selected-dispatch` vs `broadcast-observe` | 相同 group 骨架下传播与未选状态写入是否有额外价值 |
| 相同 Group-receiver、不同 mixer/merge 范围 | 收拢方式与通信中心化风险的独立作用 |
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
- Head/Group 划分、局部 expert pool 和 mixer 范围。
- shared expert / always-on backbone 比例。
- fixed merge 间隔、范围和最大控制寿命。
- 单父 vs 多父、cross-coupling vs fixed convergence，以及消息接入点。

每个实验表至少把 branch grammar、门控范围、传播 profile、状态更新与生命周期、selector 输入与决策、Head/Group 结构、active budget、merge/backbone 和物理放置列成独立字段。一个配置同时改变多个字段时，它可以支持“完整候选有效”的存在性结论，但不能单独支持任一部件的因果结论。

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
- 按 selector input profile、Head/Group 和递归层级分组的负载、route churn、状态利用与梯度覆盖。
- always-on backbone、局部递归、长短路径和 merge 频率的贡献消融。
- 多个 seed、数据切片或后续规模上的复现情况。

路径相关指标必须分开记录：静态拓扑路径长度、某个 Token 的动态传播 hop 数、实际执行的昂贵模块数，以及写状态到以后读出的 Token 距离。它们都可以被口语称为“路径长”，但对应不同成本和信用问题。

激活均衡不等于训练均衡。节点即使被均匀激活，也可能没有有效梯度、没有读出其状态，或没有稳定语义分布。

### 8.4 Scaling gate

- 在节点入度/出度统一有界的前提下，逐步增加潜在节点数、总参数量、递归深度或空间直径。
- 分开报告总容量、每 Token 实际到达的节点、昂贵激活数、Emit 边数、Observe / Update 次数和状态容量；不能只用 active parameters 代表全部成本。
- 验证质量、能力、样本效率或可保留知识是否随潜在容量增长，而不是只在固定小模型上超过一个弱基线。
- 记录深度增加时的 receiver exposure、route churn、梯度覆盖、状态利用、最长控制寿命和 write-to-read 延迟，判断局部 selector 的早期错误与历史碎片化是否累积。
- 与 capacity/compute/resource-matched dense 和 flat MoE 比较，并检查收益是否依赖不断增大的 fan-out、全局 mixer 或近似稠密的 Observe / Update。

只有 correctness、mechanism-use 和小规模 learning value 成立，还不能证明 Tide 的“极大容量 + 超稀疏”目标。Scaling gate 要回答的是：增加潜在容量时，模型是否继续获得收益，同时单 Token 的昂贵工作和物理通信半径仍按预期受控。

### 8.5 系统 gate

- 总参数、active parameters 和实际 active FLOPs，按 node、Head/Group、expert 和 backbone 分项统计。
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
    分离 Observe、Update、ExpensiveCompute 与 Emit；可承载 Head/Group、Attention、SSM 或 FFN branch

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
    记录每个 Token、父模块、Head/Group 和递归层级的 receive、update、active、read 与 emit artifact

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

这套底座必须能够在同一拓扑上切换两种 propagation profile，保存和恢复持久状态，执行 state knockout，并分开统计轻量更新和昂贵计算。两条工作流共享的 Group-receiver 配对配置使用同一种 `ReceiverCell` 规则布局和执行器，只切换声明清楚的传播与状态 profile。

函数保持接入需要显式实现 identity-compatible 或 residual-isolated 接口。任意 Head/Group 切分并不天然保持原函数；需要复制/拆分旧权重、恒等 mixer，或者先让新分支对输出严格中性。若采用零 gate 或零输出投影，还要验证哪些新参数在起点能够获得梯度，并定义可复现的 gate 开启方式。

近期底座不需要实现 HB-Line executor、一般 event IR、跨设备 allocator 或有环 Graph。

## 10. 首个可交付成果

首个交付不是“训练出完整 Tide”，也不是完成一串串行架构阶段，而是让两条工作流在可信共同底座上同时启动：

1. 选定一个适合快速重复实验的 pre-norm decoder-only 开放权重 checkpoint、训练数据、框架和目标硬件。
2. 完成原生模型的 equality oracle、continued-pretraining 校准和 fresh save/reload 测试。
3. 实现第 9 节的统一接口，使相同局部拓扑能够切换 `selected-dispatch` / `broadcast-observe`、持久/无延迟状态和各类 knockout。
4. 在工作流 A 中建立 dense continued-pretraining、成熟 FFN flat MoE 强基线和第 5.5 节定义的 Group-receiver selected control；原始 Group-wise FFN MoE 可作为辅助结构对照。
5. 在工作流 B 中实现一个最小但机制完整的候选：保留 always-on checkpoint，通过中性 residual 接口加入固定有界 receivers、`broadcast-observe`、可延迟读出的私有状态、局部稀疏昂贵激活和声明清楚的 fixed merge；其中至少一组使用与工作流 A control 完全匹配的 Group-receiver 骨架，其他候选的 Leaf/Receiver 门控和一层或多层递归由具体命题说明。
6. 为该候选同时保留 matched `selected-dispatch` 开关，以及 inactive-state freeze/clear/shuffle/no-read/reset 等直接反事实。
7. 完成短程训练并输出 correctness、mechanism-use、质量、路由、梯度、状态利用、路径分布和分项系统成本报告。

这个交付的成功标准是实验可重放、候选语义完整、问题可观测且关键反事实可运行；它不要求首轮结果已经证明 Tide 有效。随后根据实际观察定位问题、引入干预、做配对反事实并复现，而不是等待某个预设小阶段通过后才允许使用递归、状态或 Receiver-Gated。

## 11. 远期展望

非平凡的一般 Graph/DAG 天然由具有长度的传播路径构成，实际可扩展结构也很可能出现不同传播距离、逐级展开、空间分区或多尺度组织。它们的长短路径不必来自当前这种规则层次递归，也未必采用相同的 merge 方式；因此不能把当前结构称为一般 Graph 的唯一必经形式。

不过，规则层次化递归仍是值得先行摸索的代表性台阶。它能够从 checkpoint 的稳定接口逐级生长，在有界 fan-out 下控制局部候选、传播深度和收拢频率，并自然形成可测的不等长局部 DAG 与按需计算。这既可能带来更高性能上限或更低平均成本，也能为走向更一般局部 Graph/DAG 积累关于路径长度、空间扩展、信用分配和物理放置的经验；前两项收益目前仍是假设，而不是既有结论。按照第 2.3 节的证据推进关系，应先利用该受控结构建立正面信号和失败诊断能力，再承担 line、lattice、mesh、多尺度 backbone 与一般局部 DAG 逐步增加的拓扑和系统自由度。

HB-Line/HB-Lattice 可以进一步把递归或局部分支映射到重复空间切片，使 node 参数和状态长期驻留在局部设备，并通过有界邻接形成更一般的局部计算介质。一般空间 DAG 可以使用其他来源的不等长路径、显式 allocator 和拓扑序 chunk 执行；含反馈的一般 Graph 还会引入动态 event DAG、边界在途消息和更强状态机制。

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
- Head/Group-wise 已经成功融合进 Tide，或仅凭分组和全维 mixer 就得到了去中心化通信。
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
3. 定义 `broadcast-observe` reference contract：固定局部邻接、receive/update/active/read/emit 的顺序、空消息规则、状态生命周期和 merge 范围。
4. 选定工作流 B 的首个完整候选，包括 branch grammar、Group-receiver 布局、门控范围、私有状态、selector 输入与决策、active budget、递归深度和收拢方式；另有非 Group 候选时也用同一字段记录。
5. 定义共享 group layout 的工作流 A `selected-dispatch` control、inactive-state knockout 和必要的 Leaf/Receiver 直接反事实。
6. 定义 dense、flat MoE，以及可选 Group-wise FFN MoE 的 capacity/compute/resource-matched 配置。
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

近期实验因此优先使用 fixed merge、有界递归和只协调固定局部候选的 selector。selector 可以按完整候选需要读取当前内容、语义状态和逐序列历史激活信息；只要这些输入会改变路由，就必须作为模型 state 明确定义并验证 chunk 等价性。长期不收拢路径和不可组合的逐 Token 控制递推不是固定排在后面的“阶段”，而是只有在能够说明其必要性、语义和 span 代价时才引入的高风险自由度。

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
