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

## 正典教材

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

## 带权消息的局部架构

局部函数教材位于 local-semantics/。带权消息支线选定正典允许的一组局部空间与函数；共同架构给出组合条件，模块教材定义状态、读取、评分与发送。

| 文档 | 定义与证明的分工 |
|---|---|
| [共同局部架构](local-semantics/weighted-message/architecture-learning-note.md) | 完整计算过程、模块接口与组合，局部条件如何支持整网性质，精确继续与联合求值 |
| [区域评分与发送分配](local-semantics/weighted-message/region-allocation-learning-note.md) | 描述与评分、指定总量、区域阈值和节点增益；发送候选、求解与导数，以及各候选的温度极限 |
| [带权 KV](local-semantics/weighted-message/kv-memory-learning-note.md) | KV 写入、描述与内容读取、位置及局部状态性质 |
| [基础循环记忆](local-semantics/weighted-message/recurrent-memory/affine-delta-learning-note.md) | affine／delta 的完整局部定义，循环候选共用的仿射复合 |
| [固定逻辑时间短卷积](local-semantics/weighted-message/recurrent-memory/causal-convolution-learning-note.md) | 带权历史槽、当前读取、空档与弱事件；供完整循环候选复用的部件 |
| [Gated DeltaNet](local-semantics/weighted-message/recurrent-memory/gated-delta-learning-note.md) | 多头 Delta、短卷积、保留与写入门、输出归一化及完整局部绑定 |
| [Mamba-2／SSD](local-semantics/weighted-message/recurrent-memory/ssd-learning-note.md) | 分组状态、输入步长、标量转移、短卷积、直通与输出门的完整绑定 |

recurrent-memory/ 汇集三份循环记忆候选教材及共用的短卷积部件。三份教材分别定义可独立绑定的状态与读取；接口和组合条件由共同架构给出，仿射复合的推导集中在[基础循环 §4.1](local-semantics/weighted-message/recurrent-memory/affine-delta-learning-note.md#41-仿射复合)。

第一次阅读可先读共同架构 §1—5，理解一次接收到发送的过程；区域分配 §1—2 随后算出三个节点的发送强度，KV §1—5.1 继续完成同一例子的记忆与内容计算。走完这次计算后，可回到区域分配 §3—4 学习各部件的具体构造与替换。

学习矩阵记忆可先读基础循环。完整 Gated DeltaNet 与 SSD 使用短卷积部件，再定义自己的更新、描述与内容；它们填入同一套架构函数。

## 具体拓扑与完整实例

topologies/ 定义节点、区域、边、端口和时间。instances/ 引用拓扑与局部函数，给定初态、参数及输入读出，形成完整模型，并推导传播与计算量。

| 图 | 拓扑定义 | 完整绑定 |
|---|---|---|
| 全连接分层 | [每层一个区域、相邻层全连接](topologies/fully-connected-layered.md) | [KV、稠密 Attention 与混合记忆实例](instances/fully-connected-layered-weighted-message.md) |
| HB-Lattice | [多区域展开收拢与稀疏邻域](topologies/hb-lattice.md) | [KV、发送规则比较与混合记忆实例](instances/hb-lattice-weighted-message.md) |

选择一种拓扑和相容的局部函数即可形成新的绑定。单位强度传播、输入连续性、软硬极限与联合求值分别使用各自的前提。

两份实例先给出完整的 KV 基础绑定，再从小图上的两个连续输入位置展开计算：[全连接例子](instances/fully-connected-layered-weighted-message.md#2-两层图上的两次输入)展示接收、状态、选择及下一层聚合；[HB 例子](instances/hb-lattice-weighted-message.md#2-三层图上的分叉汇合与记忆)展示多区域分叉、稀疏邻域与汇合。随后分别讨论传播性质、可替换的局部绑定和计算量。

## 研究与实现

| 入口 | 用途 |
|---|---|
| [局部架构设计总览](memos/design/weighted-message-design.md) | 理解整体设计空间、信息与梯度依赖、候选取舍及外部证据 |
| [可选研究问题与对照](memos/training/weighted-message-research-questions.md) | 按需选择目标、比较和观测；优先级由研究者安排 |
| [局部架构实现核对](reference/weighted-message-implementation-checks.md) | 核对函数值、反向规则、数值边界、组合与切面继续 |
| [研究备忘](memos/README.md) | 设计分析、学习问题、数学补充与外部背景 |
| [学习资源](resources/learning-resources.md) | 按问题选读的书目和练习 |
| [教学参考程序](examples/positive_delay_graph_reference.py) | 正时延图有限切面、分段继续与规范记录的可执行例子 |
| [区域分配公式核对](examples/region_allocation_checks.py) | 二次分配的独立求解比较、正文算例、光滑分支导数与零温反例 |
| [循环记忆公式核对](examples/recurrent_memory_checks.py) | 短卷积、Delta／SSD 恒等式、弱事件和分段继续的有限数值检查 |

reference/ 提供名称检索与实现核对；memos/ 提供研究参考；examples/ 保存教学程序，resources/ 汇集学习材料。版本、下游采用和修订历史集中在 records/，入口为[版本与采用记录](records/semantic-versions-and-adoption.md)。
