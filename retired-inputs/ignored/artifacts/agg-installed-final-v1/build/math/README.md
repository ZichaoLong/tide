---
type: index
status: active
tags:
  - tide
  - mathematics
  - semantic-anchor
---

# TIDE：数学教材与语义总览

TIDE 研究固定消息图上的有状态计算：节点聚合收到的值、形成候选状态，区域共同选择完整计算，结果沿正时延边传播。节点的私有状态与区域的选择历史连接不同时间的作用。

本目录是数学与语义上游。教材面向第一次接触 TIDE 的数学读者，给出对象、函数、例子与证明；共同名称和返回值族为下游实现提供直接对应。计算表示与系统术语见各篇附录。

阅读分为语义与语义应用：语义内部包括上层正典与局部教材，应用则选择拓扑与局部候选，形成完整实例。[研究意图与探索路径](memos/design/design-requirements-and-architecture.md)说明为什么探索这些方向、不同机制如何共同支持意图；[应用目录](instances/README.md)连接具体候选及其形成理由。

## 语义

### 正典教材

图的层级为

$$
\mathrm{SettleGraph}\subseteq\mathrm{TimedDAG}\subseteq\mathrm{PositiveDelayGraph}.
$$

三个层级沿用同一组局部函数。分块教材研究这些函数的联合求值，是另一条阅读方向。

| 教材 | 主要问题 | 前置 |
|---|---|---|
| [SettleGraph：单次结算图](settlegraph-learning-note.md) | 严格分层与到达对齐下的单次结算、有状态序列 | 集合、函数与归纳法 |
| [带区域选择的 TimedDAG](timed-dag-region-selector-learning-note.md) | 一般到达时间下的纤维、规范作用、关闭与切面继续 | 集合、函数与归纳法 |
| [正时延图的有限切面语义](positive-delay-graph-finite-cut-learning-note.md) | 空间有环时的有限记录、区间组合与强连通分量 | TimedDAG |
| [TimedDAG 的分块预填充](timed-dag-chunk-prefill-learning-note.md) | 因果状态块、完整输出时间批与求值阶段 | TimedDAG |

SettleGraph 与 TimedDAG 都可独立作为起点。读完 TimedDAG 后，按问题选择有环图或分块求值续篇。[符号与函数索引](reference/semantic-interface-index.md)用于查阅共同名称、类型与定义位置。

### 带权消息的局部架构

局部函数教材位于 local-semantics/。带权消息支线选定正典允许的一组局部空间与函数；共同架构给出组合条件，模块教材定义状态、读取、评分与发送。

| 文档 | 定义与证明的分工 |
|---|---|
| [共同局部架构](local-semantics/weighted-message/architecture-learning-note.md) | 来源残差与同刻局部自注意力两种接收、完整计算过程、模块组合与数值性质，精确继续与联合求值 |
| [区域评分与发送分配](local-semantics/weighted-message/region-allocation-learning-note.md) | 描述、共同编码与候选自注意力评分、指定总量、区域阈值、节点增益及角色 quad／Shannon；可行性、求解、导数和各候选的温度极限 |
| [带权 KV](local-semantics/weighted-message/kv-memory-learning-note.md) | KV 写入、描述与内容读取、位置及局部状态性质 |
| [基础循环记忆](local-semantics/weighted-message/recurrent-memory/affine-delta-learning-note.md) | affine／delta 的完整局部定义，循环候选共用的仿射复合 |
| [固定逻辑时间短卷积](local-semantics/weighted-message/recurrent-memory/causal-convolution-learning-note.md) | 带权历史槽、当前读取、空档与弱事件；供完整循环候选复用的部件 |
| [Gated DeltaNet](local-semantics/weighted-message/recurrent-memory/gated-delta-learning-note.md) | 多头 Delta、短卷积、保留与写入门、输出归一化及完整局部绑定 |
| [Mamba-2／SSD](local-semantics/weighted-message/recurrent-memory/ssd-learning-note.md) | 分组状态、输入步长、标量转移、短卷积、直通与输出门的完整绑定 |

recurrent-memory/ 汇集三份循环记忆候选教材及共用的短卷积部件。三份教材分别定义可独立绑定的状态与读取；接口和组合条件由共同架构给出，仿射复合的推导集中在[基础循环 §4.1](local-semantics/weighted-message/recurrent-memory/affine-delta-learning-note.md#41-仿射复合)。

接收候选由 agg_mode 选择。[局部自注意力](local-semantics/weighted-message/architecture-learning-note.md#23-同刻消息的局部自注意力)在同一节点的到达消息间进行带强度先验的多头读取，逐条加工后仍交付 $(r^{\mathrm{in}},u)$；它与历史 KV 读取和区域 Attention 评分分别定义。完整参考绑定见 [HB §1.1](instances/hb-multiscale-weighted-message.md#11-接收自注意力的完整替换)，规模表分别计入来源残差、两种轻量与加宽配置。

第一次阅读可先读共同架构 §1—5，理解一次接收到发送的过程；区域分配 §1—2 随后算出三个节点的发送强度，KV §1—5.1 继续完成同一例子的记忆与内容计算。走完这次计算后，可回到区域分配 §3—4 学习各部件的具体构造与替换。

学习矩阵记忆可先读基础循环。完整 Gated DeltaNet 与 SSD 使用短卷积部件，再定义自己的更新、描述与内容；它们填入同一套架构函数。

## 语义应用

### 具体拓扑与完整实例

topologies/ 定义节点、区域、边、端口和时间。instances/ 引用拓扑与局部函数，给定初态、参数及输入读出，形成完整模型，并推导传播与计算量。

[应用目录](instances/README.md)区分拓扑、局部组合、参数归属与教学用途；[主干候选的形成与取舍](instances/backbone-design.md)保存从参考 LH、异时输出到 HB 各分支的理由、取舍和替代方案。拓扑与局部组合相互约束，具体候选按其意图选择，已有目录不穷尽设计空间。

[三档整体参数绑定](instances/parameter-profiles.md)统一规定新 FC／HB／LH 实例的 independent、region_shared、layer_shared，覆盖 Agg、记忆、评分和 Full，含 KV 基座例外、私有状态及训练通信。它说明如何绑定原教材参数；旧教学实例继续保留自己的归属。

| 图 | 拓扑定义 | 完整绑定 |
|---|---|---|
| 全连接分层 | [每层一个区域、相邻层全连接](topologies/fully-connected-layered.md) | [KV、稠密 Attention 与混合记忆实例](instances/fully-connected-layered-weighted-message.md) |
| 原 HB-Lattice | [多区域展开收拢与稀疏邻域](topologies/hb-lattice.md) | [KV、发送规则比较与混合记忆实例](instances/hb-lattice-weighted-message.md) |
| HB 多跨度图 | [普通通路、可选 hub、三种分区与时间绑定](topologies/hb-multiscale.md) | [K／S／X 三组主绑定、共享与继续](instances/hb-multiscale-weighted-message.md) |
| HB-Exchange 位交换 | [展开、分阶段交换与收拢](topologies/hb-exchange.md) | [Settle 三种空间图的共同主绑定](instances/hb-settle-weighted-message.md) |
| LH-DAG／LH-PDG | [双侧层次、局部网格与可选反馈](topologies/lh-exploration.md) | [复用主局部候选的探索绑定](instances/lh-exploration-weighted-message.md) |

选择一种拓扑和相容的局部函数即可形成新的绑定。单位强度传播、输入连续性、软硬极限与联合求值分别使用各自的前提。

原有全连接与 HB-Lattice 两份实例先给出完整的 KV 基础绑定，再从小图上的两个连续输入位置展开计算：[全连接例子](instances/fully-connected-layered-weighted-message.md#2-两层图上的两次输入)展示接收、状态、选择及下一层聚合；[HB 例子](instances/hb-lattice-weighted-message.md#2-三层图上的分叉汇合与记忆)展示多区域分叉、稀疏邻域与汇合。随后分别讨论传播性质、可替换的局部绑定和计算量。

[显式残差 MoE 教学绑定](instances/explicit-residual-moe.md)另从通用 TimedDAG 直接构造：每个专家单独占节点，基线旁路与专家贡献在合并节点汇合；先无状态，再加入本地激活计数。它与上述带权 KV 实例独立，供 Transformer 到 Tide 的教学过渡使用。

## 研究与实现

| 入口 | 用途 |
|---|---|
| [研究意图与探索路径](memos/design/design-requirements-and-architecture.md) | 意图与机制的多对多关系、语义与应用的支撑、启发和后续探索；保留计算例子与有界通信推导 |
| [首次展示讲述脚本](presentations/design-introduction/script.md) | 从 Transformer 残差到显式 MoE，再展开局部网络、历史与接口；数值源和附页各有明确归属 |
| [展示材料](presentations/design-introduction/README.md) | 可逐步推进的离线 HTML、PDF、画面预览与讲者备注 |
| [教学图集](figures/README.md) | 节点与区域、P/S/U/F、带权消息与局部模块图，以及定义出处与绘图源文件 |
| [主干比较绑定与尺寸阶梯](instances/backbone-comparison.md) | FC／HB 的统一局部绑定、独立／共享、二次幂预算、合理维度及可复算规模表 |
| [主干拓扑与规模面板](explorers/backbone-atlas/README.md) | 三种 HB 空间图与窄图、完整邻接、端点矩阵、公式、时间，以及 FC／HB 主干规模筛选 |
| [主干参考构造与计数](reference/backbones/README.md) | 面板与数学核对共用的构图、潜在传播和参数形状代码 |
| [教学与数学核对程序](examples/README.md) | 各程序的覆盖范围、运行入口及与公共参考代码的关系 |
| [局部架构设计总览](memos/design/weighted-message-design.md) | 理解整体设计空间、信息与梯度依赖、候选取舍及外部证据 |
| [可选研究问题与对照](memos/training/weighted-message-research-questions.md) | 按需选择目标、比较和观测；优先级由研究者安排 |
| [局部架构实现核对](reference/weighted-message-implementation-checks.md) | 核对函数值、反向规则、数值边界、组合与切面继续 |
| [研究备忘](memos/README.md) | 设计分析、学习问题、数学补充与外部背景 |
| [学习资源](resources/learning-resources.md) | 按问题选读的书目和练习 |
| [教学参考程序](examples/positive_delay_graph_reference.py) | 正时延图有限切面、分段继续与规范记录的可执行例子 |
| [显式残差 MoE 核对](examples/explicit_residual_moe_checks.py) | 专家、旁路和合并的时间求值；两轮状态、直接公式与工作量核对 |
| [区域分配公式核对](examples/region_allocation_checks.py) | 二次分配的独立求解比较、正文算例、光滑分支导数与零温反例 |
| [角色分配核对](examples/role_bounded_allocation_checks.py) | 独立活动集、全输入可行性、零／缺席与光滑分支导数 |
| [紧凑图的相位与合作](examples/hb_compact_checks.py) | 窄图、首层节点覆盖、独立交换轮数和更深平台 |
| [主干参数核对](examples/backbone_capacity_checks.py) | 二次幂预算、逐头适配器、独立参数闭式计数和活跃代理 |
| [区域自注意力核对](examples/regional_attention_checks.py) | 编码到评分与控制、候选记忆间影响、重排及零／弱边界 |
| [接收自注意力核对](examples/incoming_attention_checks.py) | 手算、多头、零／弱消息、消息间依赖和三档参数增量 |
| [HB-Settle 构图与交换核对](examples/hb_settle_checks.py) | 原图字地址一致性、18 组交换配置、每轮位覆盖、度数与原周期条件 |
| [HB／LH 构图与时间核对](examples/hb_lh_checks.py) | 深度与尺寸、分区前提、hub 删除、对齐时间、反馈有限继续 |
| [循环记忆公式核对](examples/recurrent_memory_checks.py) | 短卷积、Delta／SSD 恒等式、弱事件和分段继续的有限数值检查 |

reference/ 提供名称检索、实现核对及共用的主干参考计算；explorers/ 保存交互面板维护源；memos/ 提供研究参考；examples/ 保存教学与数学核对程序，resources/ 汇集学习材料。版本、下游采用和修订历史集中在 records/，入口为[版本与采用记录](records/semantic-versions-and-adoption.md)。

## 维护与生成

仓库内的[研究维护技能](skills/tide-research-maintenance/SKILL.md)组织意图、机制、语义和实例变更；[数学写作技能](skills/tide-math-writing/SKILL.md)用于教材与数学正文，[展示写作技能](skills/tide-presentation-writing/SKILL.md)用于讲述、图示和展示。按任务读取相应文件，无需全局安装。具体意图与配置仍由上述设计文档、教材和实例维护。

本线的教材、图集、实例和展示之间的同步职责见[全仓维护地图](../ARTIFACTS.md#各条工作线的维护地图)。定义变化后先核对受影响的图形与展示维护源，再按组构建；构建成功不能替代语义核对。图集与展示的具体编辑方式见各自 README。
