# TIDE Checkpoint 生长实验仓库

> 状态：项目启动草案
>
> 日期：2026-08-20
>
> 目标读者：即将开始实现、训练与验证 TIDE 的研究和工程人员
>
> 当前实验入口：本 README 说明 `fractal-latcarf` 分支为什么做这些实验、当前真正要完成什么，以及怎样判断结果。详细候选语义和实验协议放在 `docs/`；可泛化的理论、历史和跨实验结论由 ObsidianVault 维护。
>
> 上游研究总入口：[ObsidianVault / TIDE](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/README.md)

快速阅读：想一眼看懂研究理由，读第 1.2 节；想逐条检查完整推理，读第 2.3 节；想了解当前任务，读第 3 节；准备实现时读第 5、6 节和 [实验协议](docs/experiment-protocol.md)；实际采用的每日设置与结果见 [实验记录](experiments/README.md)。

## 1. 项目目标与当前状态

TIDE 的完整项目表述是：`TIDE: A Topology-Invariant Degree-bounded Expansion Architecture for Autoregressive Token Inference`。

中文表述是：**TIDE：面向自回归 Token 推理的拓扑固定、度有界容量扩展架构。**

本仓库以 checkpoint 生长线为近期主线，研究 TIDE Architecture、训练语义和可执行实验。

TIDE 当前有三个最底层的结构要求：

1. **固定空间拓扑**：对一个已经确定的模型，节点和边不随 Token、状态或 selector 临时改变。
2. **单节点成本有界**：入口、receiver、selector、router 和 merge 的参数、状态、连接、候选处理与通信成本，不随模型总容量一直增长。
3. **可达容量增长**：扩大模型时，更多潜在容量仍然能够被输入到达并实际利用。

固定拓扑本身不表示度有界。但每条直接连接都会占用接口、状态、候选处理或通信资源；在这个成本口径下，单节点成本有统一上界会进一步限制节点度。有界度与可达容量增长共同要求多跳、逐级、层次化或空间化扩展，而不是一步平铺访问全部容量。

如果每个 Token 还只能执行少量昂贵模块，就要在逐级传播中做局部选择，并对消息、传播深度和总激活量设置明确预算。

本文把接收上游消息并拥有自身参数或状态的下游模块称为 `receiver`；把 Attention readout、FFN、大型 SSM 更新等主体计算称为“昂贵计算”，以区别于消息投影和轻量状态更新。

> **当前状态：** Flat MoE 已经提供可靠的外部正面证据；本仓库自身尚无可靠训练结果。层次递归、私有状态、`broadcast-observe`（BO）、交叉汇聚和多次局部选择是否有效，都需要实验回答。

### 1.1 本仓库最终要回答什么

TIDE 长期希望同时满足：

1. 固定空间拓扑与有界单节点成本；
2. 可达、可利用的总容量持续增长；
3. 静态 Graph 和每 Token 昂贵激活都保持稀疏；
4. 动态选择仍能稳定训练，不造成无法控制的路径漂移、节点饥饿和信用分配问题；
5. `prefill` 与逐 Token `decode` 具有同一单序列 reference semantics；
6. 跳过的工作足以覆盖 selector、状态、packing、通信和负载不均衡的成本。

当前最核心、也最可证伪的总假设是：

> 存在一种有实际意义的计算粒度：节点内部继续使用高效稠密 kernel，节点之间沿固定空间拓扑稀疏激活；扩大模型时单节点成本保持有界，同时模型质量、训练稳定性与端到端性能能够成立。

### 1.2 一眼看懂的两条逻辑链

第一条回答：**为什么 TIDE 需要有界度多跳扩展？**

```text
[已有证据]
Flat MoE 证明：大容量可以与每 Token 少量昂贵计算同时存在
    ↓
[原始要求]
固定空间拓扑 + 单节点成本有界 + 可达容量增长
    ↓ 每条直接连接都会占用实际节点资源
[结构推论]
节点度必须有统一上界；
入口数量和传播跳数都固定时，可达节点数也有上界
    ↓
[扩展要求]
容量继续增长，需要更多传播层级或更大的空间直径
    ↓ 如果每 Token 还要保持超稀疏
[额外要求]
传播过程中反复做局部选择，并限制消息、深度和总激活量
    ↓
[近期工程起点]
从 checkpoint 中性生长，使用 always-on backbone 和 fixed merge
```

第二条回答：**为什么 BO、私有状态和交叉汇聚值得重点验证？**

```text
[潜在问题]
多级 selected-dispatch 可能使越靠下游的有状态节点，
接收到的上下文历史越来越少
    ↓
[待验证假设]
下游历史覆盖稀释是否真实存在，并且是否伤害学习？
    ↓ 如果答案是肯定的
[重要候选]
私有状态与 Update/Compute 分离
+ broadcast-observe
+ 分支间交叉汇聚或 backbone reinjection
    ↓
[稳定骨架]
与 always-on backbone、fixed merge 组合
    ↓
[工作流 B]
以 BO 为主要验证轴，
寻找可训练、可扩展的成功候选，并通过配对实验建立归因
```

第一条把“由目标得到的结构推论”和“近期采用的工程起点”分开；第二条记录的是“待验证风险 -> 候选机制 -> 实验任务”，不把 BO 写成由 TIDE 目标严格推出的结论。本节供快速阅读，第 2.3 节保留同一逻辑的完整版本；核心逻辑变化时，以第 2.3 节为完整记录，并同步更新这里的摘要。

## 2. 从成熟基线到当前推理

### 2.1 原生 dense checkpoint

近期从 pre-norm、decoder-only、开放权重模型开始。原生 checkpoint 已经提供成熟的 residual 接口、causal `prefill/decode`、训练配方和稠密 kernel，是所有增量实验的 correctness 与质量基线。

新增 residual 分支可以在输出端初始化为零，使扩展模型在起点保持原函数。但“函数保持”只描述初始输出；还必须检查新分支能否获得梯度，以及何时、怎样离开中性状态。

第一步不是重新发明 Transformer runtime，而是完整装载 checkpoint，并对齐参数、logits、KV cache、任意 chunk continuation、训练 loss、主要梯度和 save/reload。

### 2.2 Flat MoE 提供的现实证据

成熟 MoE 已经证明：很大的潜在参数容量可以与每 Token 少量昂贵计算同时存在。它通常在一个局部子层内从完整专家池做一次选择，并立即 merge 回共同 hidden；下一层重新面对自己的完整候选集合。

标准 flat MoE 的模型拓扑也可以固定。它与 TIDE 的关键差别不是“是否固定”，而是 flat MoE 通常把随规模增长的专家池作为一步可选的全局候选集合，router、候选处理或 dispatch 范围可能随专家数扩大。

TIDE 要研究的是：能否沿固定空间图逐步接触更多容量，使单节点成本不随总容量增长。这样做会引入 flat MoE 较少面对的多级选择、路径持续、下游历史覆盖和更长信用链。因此，flat MoE 必须是主要强基线，而不是被当作错误起点。

### 2.3 从目标到工作流 B：当前推理

本节保留本仓库最接近原始研究输入的完整逻辑。它记录为什么选择这些实验，不把候选机制写成已有结论。其中，第 1 条来自成熟模型经验；第 2、3 条由 TIDE 目标推出；第 4 条是训练风险预期；第 5 至 7 条是当前稳定办法；第 8 条以后是围绕下游历史覆盖问题提出的候选假设。

#### 从 MoE 到逐级局部扩展

1. **Flat MoE 是已经得到正面验证的起点。** 它证明了很大的潜在参数容量可以和每 Token 少量昂贵计算同时成立。它没有验证的是：在单节点成本保持有界时，模型能否沿固定空间图通过多跳传播利用持续增长的容量。

2. **固定空间拓扑、单节点成本有界和可达容量增长，共同要求多跳扩展。** 固定拓扑只说明执行期间不临时改边，本身不限制一个节点有多少邻居。但每条直接连接都占用实际资源，因此单节点成本上界会限制节点的入度和出度。在入口数量和节点度都有统一上界时，固定跳数内能到达的节点数也有上界；要让更多容量保持可达，传播深度或空间直径就必须增长。

3. **如果还要求超稀疏，就需要在逐级展开时反复做局部选择。** 每个 selector 只面对少量直接候选，但全模型仍要限制每 Token 执行多少昂贵模块、发送多少消息和传播多深。否则局部 fan-out 虽然有界，总工作仍可能快速增长。

#### 预期困难与当前稳定骨架

4. **多级局部选择可能比 flat MoE 更难训练。** Flat MoE 的一次选择通常很快 merge；TIDE 中较早的选择可能继续限制后续多步路径。如果模块还有私有状态，一次写入也可能在以后才产生作用。路径漂移和信用分配问题因而可能被放大。

5. **固定收拢（fixed merge）是当前最直接的控制手段。** 它让分支在已知位置回到共同接口，避免一次局部选择无限期决定未来可达路径。它能限制显式控制寿命，但不会删除已经写入私有状态的跨 Token 影响。

6. **始终开启的原模型主干（always-on backbone）是 checkpoint 生长的稳定锚点。** 原模型路径始终保留，新增模块通过初始中性的 residual 接口接入。这样可以从可用模型出发，但不能保证新增分支会被真正使用。

7. **局部 selector 有很大设计空间。** 局部候选较少，可能比一次全局 $N$ 选 $M$ 更容易；但选择会重复发生，早期错误可能影响更深路径。Content、receiver state、历史激活和局部预算都可以成为候选输入，具体使用什么应由待验证问题决定。

#### 为什么关注下游历史覆盖

8. **多级 `selected-dispatch` 可能造成下游历史覆盖稀释。** 分支路径越往下游，有状态节点可能只接收到越来越少的历史 Token 或消息，因而其私有状态覆盖的上下文历史越来越窄。本文把它称为“分支路径下游的上下文历史覆盖稀释”，简称“下游历史覆盖稀释”。这里减少的是 receiver 自身接触到的历史范围，不表示当前 hidden 被机械切碎，也不表示质量一定下降。

9. **这个问题只是一项待验证假设。** Parent hidden 可能已经整合完整前缀中的任务相关信息；选择性的私有历史也可能帮助专门化。实验必须同时测量消息覆盖、私有状态历史、当前表示和任务质量，不能只看到路由不同就宣布上下文丢失。

10. **如果下游历史覆盖稀释真实存在并且有害，有几类重要候选机制。** 私有状态与 Update/Compute 分离可以保存收到的信息；BO 可以让未被选中做昂贵计算的固定直接下游仍获得 Observe / Update 机会；分支间交叉汇聚可以交换不同路径已经处理过的信息；backbone reinjection 可以重新提供稳定公共上下文。

11. **BO 是工作流 B 的主要验证轴。** Active sender 向全部固定直接 children 发送；所有实际 receivers 执行声明的轻量 Observe / Update，只有少数 receivers 做昂贵计算并继续传播。它还允许 selector 读取更新后的 receiver state。第一批候选会重点检验这种“先收到和更新，再稀疏计算”的局部介质是否有 learning、scaling 和系统价值。

12. **BO 首先改变的是一跳范围内的消息和写入机会。** 它不会绕过更早没有激活的祖先，也不保证状态一定在以后被读出，更不保证所有深层节点得到完整无损历史。因此必须与相同拓扑、相同 route 或 replay route 的 `selected-dispatch` 做直接对照。

13. **私有状态必须通过以后真正读出才产生记忆价值。** 收到消息和状态数值发生变化，只证明 Update 执行了。还需要用 state freeze、clear、shuffle、no-read、reset 等实验确认某次写入在以后改变了 selector、输出或 loss。

14. **交叉汇聚可能扩大不同分支之间的信息来源。** 多个父节点可以把各自处理过的信息送到同一局部节点；也可以在 fixed merge 前增加有界 cross-coupling，或在 merge 后重新发散。它可能缓解分支隔离，也可能增加输入组合、归因、等待和通信成本。

15. **Fixed merge 和交叉汇聚不是同一个概念。** Fixed merge 主要结束旧路径身份、限制控制寿命；cross-coupling 主要让分支交换信息，同时仍可保留各自身份。两者可以组合，但实验记录和因果结论必须分开。

16. **多父节点会增加消息聚合和状态更新的接口复杂度。** BO 与固定局部接收方式结合得比较自然；`selected-dispatch` 仍然可以在同一多父拓扑上工作，因此应保留为传播反事实。

#### 工作流 B 要验证什么

17. **工作流 B 的核心任务，是找到至少一个可训练、可扩展的成功候选。** 当前用 always-on backbone 和 fixed merge 作为稳定骨架，把私有状态、BO、交叉汇聚、backbone reinjection、门控范围和 selector 作为可以组合、替换和消融的设计轴。其中 BO 是第一批实验的主要验证轴。

第一轮探索要回答：在固定局部拓扑和明确预算下，能否从 checkpoint 中性生长出一个稳定训练、真实使用新增机制并优于匹配对照的完整候选。浅层实验可以先判断机制是否有 learning value；要支持 TIDE 的长期主张，还必须进一步进入有界度多跳或空间 scaling，验证容量增长时单节点成本、总工作和系统成本仍然可控。

更详细的候选语义见 [TIDE 候选设计空间](docs/candidate-design-space.md)。

## 3. 当前核心任务

Checkpoint 生长线近期并行推进两个工作流：

```text
工作流 A：建立 dense / flat MoE 的可信基线与成本参照
工作流 B：从 checkpoint 生长并寻找可训练、可扩展的 TIDE 候选
```

两条工作流共享数据、训练配方、correctness oracle、成本口径和实验账本。Dense equality 就绪后可以并行推进，不要求工作流 A 的所有后续实验完成后才启动 B。

### 3.1 工作流 A：建立可信强基线

工作流 A 包含：

1. 原生 checkpoint 的参数、logits、cache/state、`prefill/decode`、训练和 save/reload 对齐；
2. 成熟 flat MoE reference recipe 或兼容原生实现；
3. 从 dense checkpoint 中性生长的 flat MoE matched control；
4. 基线可靠后的可选 Head-Wise MoE。

它提供 correctness oracle、成熟训练区间和系统成本参照。TIDE 不能只与未调好的 checkpoint-grown MoE 比较。

### 3.2 工作流 B：寻找成功候选

工作流 B 不把 BO 当作由结构目标自动推出的结论，但把它作为主要验证轴。当前候选空间分成四层：

| 类别 | 内容 |
| --- | --- |
| 必须面对的 TIDE 约束 | 固定空间拓扑、单节点成本有界、可达容量增长、稀疏预算 |
| 当前稳定骨架 | checkpoint 中性生长、always-on backbone、fixed merge |
| 主要候选机制 | 私有状态、BO、交叉汇聚、backbone reinjection |
| 直接反事实 | `selected-dispatch`、无延迟状态、关闭交叉边、flat MoE |

探索阶段允许把有共同理由的机制组成完整候选，先寻找正面存在性信号；任何关于单项机制的结论，都必须补只改变关键因素的直接反事实。

第一批 BO 实验分别观察两条作用路径：

```text
当前作用：Observe / Update 改变 proposal、selector 或本次输出
延迟作用：未激活期间写入私有状态，以后被读出并改变输出
```

### 3.3 问题驱动的实验闭环

后续不按固定阶段机械推进，而是使用下面的闭环：

```text
观察问题
-> 定位是覆盖、状态、路由、梯度还是系统成本问题
-> 选择针对性候选机制
-> 做 paired counterfactual
-> 用新 seed、数据或规模确认
```

例如，只有在下游历史覆盖不足被测量到，并通过记忆任务或 causal knockout 确认它伤害质量时，才有依据增加 BO 强度、交叉边、backbone reinjection 或更大的局部 K。详细诊断表见 [实验协议](docs/experiment-protocol.md#3-问题驱动的闭环)。

## 4. 候选技术地图

README 只保留每项技术的角色，具体 reference semantics、公式和配置坐标见 [候选设计空间](docs/candidate-design-space.md)。

| 技术 | 当前作用 | 主要风险 |
| --- | --- | --- |
| 有界度递归或局部 DAG | 通过多跳接触增长的容量 | 传播深度、控制寿命和物理通信增长 |
| Always-on backbone | 保留 checkpoint 能力和公共梯度路径 | 新分支可能长期不被使用 |
| Fixed merge | 在已知位置结束显式路径身份 | 不会消除已写入私有状态的延迟影响 |
| 私有状态 | 保存 receiver 收到的局部历史 | 显存、顺序依赖和延迟信用 |
| `broadcast-observe` | 分开消息/状态更新与昂贵激活 | 广播、写状态和调度可能过贵 |
| 交叉汇聚 | 扩大分支间的局部历史来源 | 多路归因、等待和通信复杂度 |
| Backbone reinjection | 重新提供稳定公共表示 | 可能削弱局部路径的独立价值 |
| Local selector | 在固定小候选集内控制稀疏激活 | 路由漂移、饥饿和状态语义复杂度 |

每个具体候选至少要记录：静态拓扑、最大 fan-in/fan-out、门控范围、传播 profile、状态生命周期、selector 输入与决策、active/message budget、backbone/merge、交叉边和物理放置。

## 5. 实验方法与成功标准

### 5.1 探索、诊断和确认分开

- **探索实验**：允许完整机制包，只回答是否出现正面信号。
- **诊断实验**：使用 knockout 或配对反事实定位原因。
- **确认实验**：冻结方案，用新 seed、数据、硬件或规模复现。

一个组合候选成功，可以证明“存在一个组合有效”；不能单独证明其中 BO、私有状态、交叉汇聚或 fixed merge 各自必要。

### 5.2 五道证据门

| 证据门 | 必须回答的问题 |
| --- | --- |
| Correctness | checkpoint 是否等价装载；`prefill/decode`、chunk、save/reload 是否一致 |
| Mechanism use | 消息、状态或交叉路径是否真的被 selector/readout 使用 |
| Training | 候选能否稳定训练并超过匹配基线 |
| Scaling | 容量增长时质量是否继续受益，单节点和总工作是否仍可控 |
| System | 新增消息、状态、selector 和 merge 成本是否小于跳过的工作 |

对于 BO，还要把结论分成四层：机制确实被使用、具有 learning value、具有 scaling value、具有 system value。前一层不能代替后一层。

### 5.3 最重要的直接对照

- 原 checkpoint continued pretraining；
- 等参数 dense 扩展和等 active-FLOPs flat MoE；
- 相同拓扑、相同或 replay route 的 `selected-dispatch`；
- BO selector 读取与忽略 post-Update state；
- 未激活 state freeze / clear / shuffle / no-read / reset；
- 无状态 FFN 路径与有状态 Attention/SSM receiver；
- 相同多父拓扑关闭与开启交叉边。

完整指标和对照定义见 [实验协议](docs/experiment-protocol.md)。

## 6. 近期交付与软件边界

近期实现少量稳定抽象，不先完成一般 Graph runtime：

- `CheckpointAdapter`：原生装载、状态映射和 equality oracle；
- `BranchModule`：atomic、serial 或 recursive 分支；
- `MessageProjection`：固定、有界的 receiver slots；
- `ReceiverCell`：分离 Observe、Update、ExpensiveCompute 与 Emit；
- `PropagationProfile`：切换 `selected-dispatch` / BO；
- `SiblingSelector`：在有界兄弟集合内选择；
- `ReceiverState` / `SelectorState`：分开保存语义与 selector 状态；
- `FixedMerge`：声明固定槽位、范围和 merge；
- `RouteArtifact` / `ExperimentLedger`：保存行为、成本与实验谱系。

首个交付包括：

1. 选定 checkpoint、数据、框架和目标硬件；
2. 完成原生 equality、continued-training 和 save/reload；
3. 建立统一 receiver、传播、状态和 instrumentation 接口；
4. 完成 dense 与成熟 flat MoE 强基线；
5. 实现一个保留 always-on backbone、具有有界局部连接、BO、可实际读出的私有状态、稀疏昂贵激活和 fixed merge 的首轮完整候选；
6. 同时保留 matched `selected-dispatch`、状态 knockout 和交叉边开关；
7. 输出 correctness、机制使用、质量、路由、梯度、历史覆盖、状态利用和分项系统成本报告。

这个交付首先要求实验可重放、候选语义完整、问题可观测、主要验证轴能够配对比较；不要求首轮结果已经证明 TIDE 有效。

首轮 v0 已有可运行的 PyTorch reference implementation，位于 [`src/tide`](src/tide)。当天实际采用的模型、公式、命令、运行状态和结果统一记入 [每日实验记录](experiments/README.md)，避免把易变的运行细节继续堆进 README。

近期底座不需要 HB-Line executor、一般 event IR、跨设备 allocator 或有环 Graph。

## 7. 当前不能主张的结论

- 局部通信 Graph 已经优于 Transformer、Mamba 或 MoE。
- BO 已经具有 learning、scaling 或端到端系统收益。
- 收到消息或状态发生变化，就等于该状态已经被以后有效读出。
- 下游历史覆盖稀释必然等于当前 hidden 丢失上下文，或必然伤害任务质量。
- 多父交叉汇聚必然恢复完整历史、缓解路径漂移，或只能采用 BO。
- 一个组合候选成功，就分别证明其中所有部件都必要。
- 规则递归是一般 Graph 的唯一扩展方式。
- Fixed merge、always-on backbone 或某种 mixer 对所有局部 Graph 都必要或最优。
- 逻辑邻接局部会自动带来物理通信局部和更低延迟。
- 固定空间 DAG 自动得到 Transformer/Mamba 级 node 内 Token 并行。
- 任意 stateful selector 都能获得高性能 chunk `prefill`。
- 人脑结构证明了 TIDE 可训练或高效。

当前仓库仍处于实验启动期。README 中的正面描述都是待检验命题或设计理由。

## 8. 背景、理论与上游文档

TIDE 的早期动机来自 LH 对“局部通信 + 超稀疏”的一般 Graph 设想。后续研究发现，任意跨 Token、不可组合的自适应控制链会妨碍高性能 exact chunk `prefill`，因此近期实验优先使用固定局部 DAG、明确状态、有限控制寿命和可验证的 `prefill = decode` 语义。

Graph 收缩线继续负责定义、证明、反例和更一般拓扑；本仓库的 checkpoint 生长线负责真实训练、归因和系统实验。两条路线可以交换约束和候选，不要求最终收敛到同一种架构。

上游文档：

- [TIDE 研究线总入口](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/README.md)：正式命名、战略路线和文档地图。
- [TIDE Architecture / Network：模型架构与训练](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/tide-model-architecture-and-training.md)：checkpoint 生长、结构候选和训练风险。
- [TIDE 数学基础](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/tide-mathematical-foundations.md)：`StepTransition`、`prefill = decode`、logical event DAG 与函数保持生长。
- [Adaptive routing prefill lower bound](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/adaptive-routing-prefill-lower-bound.md)：不可组合自适应路由的反向边界。
- [TIDE 背景、历史谱系与参考](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/tide-background-history-and-references.md)：LH、ISA/dataflow 和脑科学背景。
- [TIDE Engine：runtime 验证与状态](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/tide-runtime-validation-and-status.md)：runtime contract、artifact equality 与工程状态。

仓库内细节：

- [TIDE 实验语义、命名与数学符号](docs/experiment-semantics-and-naming.md)
- [TIDE 候选设计空间](docs/candidate-design-space.md)
- [TIDE Checkpoint 生长实验协议](docs/experiment-protocol.md)
- [TIDE 每日实验设置与结果](experiments/README.md)
