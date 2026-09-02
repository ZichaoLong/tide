# TIDE：固定拓扑上的有界局部容量扩展

> **TIDE: A Topology-Invariant Degree-bounded Expansion Architecture for Autoregressive Token Inference**
>
> **TIDE：面向自回归 Token 推理的拓扑固定、度有界容量扩展架构**

本仓库研究 TIDE 的神经网络架构与 reference semantics。更上层的研究背景和长期设想见 [ObsidianVault / TIDE](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/README.md)。

## 1. 研究目标

TIDE 从三个结构要求出发：

1. **固定空间拓扑**：模型确定后，底层节点和边不随 Token 临时改变；每个 Token 的 active 子图可以变化。
2. **单节点成本有界**：每个节点的参数、状态、连接、候选处理、通信和计算都有与总容量无关的上界。
3. **可达容量增长**：模型扩容后，输入仍能沿固定局部连接到达并利用更多潜在容量。

固定拓扑本身不表示度有界；但每条连接都会占用节点资源，因此单节点成本有界会进一步约束局部连接度。在入口数和局部度都有界时，一步平铺不能持续覆盖增长的总容量，模型需要通过多跳、逐级、层次化或空间化传播接触更多节点。

如果每个 Token 还只能执行少量昂贵模块，就需要在传播途中反复进行固定局部选择，并约束传播深度、消息数和 active 计算量。

## 2. 当前语义对象：SettleGraph

本仓库把一个具有单输入、单输出和固定空间拓扑的 Single-Settlement Graph 简称为 **SettleGraph**。它可以插入 decoder-only Base LLM 的不同位置，并保持 base checkpoint 的 always-on 路径。

对每个 Token：

1. SettleGraph 接收一个 \(d_{\mathrm{model}}\) 维 hidden。
2. hidden 沿固定 DAG 的边传播；每条边最终恰好结算为完整数据或关闭。
3. receiver node 聚合实际到达的父消息，维护可选私有状态，并提供轻量 selector 读出。
4. 每个固定局部 region 的 selector 只在已经 reached 的 nodes 中选择少量 active nodes。
5. propagation profile 决定哪些 reached nodes Observe/commit；active nodes 执行完整计算并发送。
6. 终端 active 消息被聚合成一个 \(d_{\mathrm{model}}\) 维输出，再按 placement 合回 Base LLM。

三种核心 propagation profiles 是：

| Profile | 状态提交 | 完整计算与发送 |
| --- | --- | --- |
| **N** | 无状态 | active nodes |
| **SD** | active nodes | active nodes |
| **BO** | 全部 reached nodes | active nodes |

selector 可以读取当前内容、更新前状态或更新后 proposal，分别形成 content-only、pre-update 和 post-update 三种时序。BO 对三者都天然兼容；SD 不使用 post-update 选择。

本文还定义两种拓扑实例：

- **单层并列 receivers**：用于隔离验证 selector、状态、N/SD/BO、Emit 和聚合；
- **HB-Lattice**：在同一公共语义上增加规则化 Lines、波前 barrier 和扩展—平台—收拢拓扑。

完整数学定义、执行顺序、loss 和命名见 [实验语义、命名与数学符号](docs/experiment-semantics-and-naming.md)。该文档是“模型实际怎样计算”的权威来源。

## 3. 主要实验问题

后续实验需要逐步回答：

- 新增 SettleGraph 能否从函数等价的 checkpoint 起点稳定离开中性初始化；
- 私有状态是否被后续 selector 或 NodeCompute 真正读取，而不只是发生数值变化；
- 在匹配 Plan、active route 和计算预算时，BO 相比 SD 是否具有可复现价值；
- content、pre-update 和 post-update selector 分别带来什么影响；
- 多父聚合、局部交叉边和多层传播能否缓解下游历史覆盖不足；
- 增加可达容量和传播深度时，训练稳定性、信用分配、节点饥饿和路径集中是否可控；
- 被跳过的昂贵计算能否覆盖 selector、状态、消息、packing 和通信成本。

主要外部基线包括原生 Dense checkpoint、匹配参数或计算量的 Dense 扩展，以及实现细节明确的 Flat MoE。SettleGraph 内部则使用 N/SD/BO、状态 knockout、selector 时序、边开关和 route replay 建立可归因对照。

## 4. 当前仓库状态

当前仓库尚无符合上述新语义的软件实现或实验结果。现阶段的权威产物只有语义规范；文档中的接口名称表示数学角色或实验条件，不表示同名软件模块已经存在。

下一阶段可以按以下顺序推进：

1. 核验并冻结语义文档；
2. 定义可验证的展开 Plan 表示；
3. 实现纯 PyTorch/CPU reference interpreter；
4. 先通过单层实例验证 placement、N/SD/BO、状态时序和梯度；
5. 冻结首个自包含实验条件；
6. 再实现 HB-Lattice Builder、规则调度和高性能后端。

具体软件组织、机器可读 manifest、实验晋级规则和运行基础设施将在实现阶段另行定义，不写入神经网络语义规范。

## 5. 术语

- **TIDE Architecture / TIDE Network**：模型结构与 reference semantics；
- **TIDE Model**：按某个 TIDE 架构训练得到的具体模型；
- **TIDE Engine**：训练或推理 TIDE Model 的 runtime；
- **SettleGraph**：本文当前定义的固定 DAG 单次结算语义；
- **Plan**：经过静态校验、完全展开的 SettleGraph 图描述；
- **receiver node**：图中的有状态或无状态计算节点；
- **region**：共享一次局部选择的固定 receiver 集合。

## 6. 当前不能主张的结论

目前不能声称：

- TIDE 已经优于 Dense Transformer、Mamba 或 Flat MoE；
- BO 已经具有 learning、scaling 或系统收益；
- 状态发生变化就等于状态已被模型有效使用；
- 多父交叉汇聚必然改善下游历史覆盖或模型质量；
- HB-Lattice 是唯一或最优的容量扩展拓扑；
- 固定逻辑邻接会自动转化为更低的物理通信成本。
