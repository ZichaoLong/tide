---
type: semantic-application-index
status: active
tags: [tide, topology, instance]
---

# 拓扑、局部组合与实例

本目录组织 Tide 的语义应用。topologies/ 规定节点、区域、边、端口与时间；完整实例选择相容的局部函数、初态、参数归属和输入读出。四份正典与局部教材提供所用定义，入口见[语义总览](../README.md#语义)。

从“为什么探索”开始，可先读[研究意图与探索路径](../memos/design/design-requirements-and-architecture.md)；从已有图或配置接续，可读[主干候选的形成与取舍](backbone-design.md)。后者保存异时输出、分工合作、相位覆盖、Hub、规模与共享如何推动当前候选，也保留替代选择和未解问题。

## 拓扑与时间

空间连接、区域划分和时间绑定分别说明。Lattice、Exchange、Multiscale 标识不同连接构造；Timed／Settle 标识相应时间绑定，同一空间图可以有多种绑定。

| 图族 | 定义与采用方式 | 当前实例入口 |
|---|---|---|
| FC 分层 | [每层一区、相邻层全连接](../topologies/fully-connected-layered.md)，Settle | [教学实例](fully-connected-layered-weighted-message.md)；新对照采用[统一比较绑定](backbone-comparison.md) |
| HB-Lattice | [区域展开收拢与邻域](../topologies/hb-lattice.md)，原 Settle 构造 | [教学实例](hb-lattice-weighted-message.md)；新对照采用 [Settle 共同绑定](hb-settle-weighted-message.md) |
| HB-Exchange | [展开、位交换与收拢](../topologies/hb-exchange.md)，面向 Settle 的中途合作 | [Settle 三图共同绑定](hb-settle-weighted-message.md) |
| HB-Multiscale | [多跨度连接与可选 Hub](../topologies/hb-multiscale.md)，分别绑定 Timed／Settle | [三组主绑定](hb-multiscale-weighted-message.md)，Settle 对照另见[共同绑定](hb-settle-weighted-message.md) |
| LH-DAG／LH-PDG | [双侧层次与区内连接](../topologies/lh-exploration.md)，PDG 另加正时延反馈 | [一般语义探索绑定](lh-exploration-weighted-message.md) |

HB 主候选采用严格区域分层，并在实例中核对局部块求值条件；LH 保留区内消息及反馈探索。每种结构优先回应的意图不同。空间图的潜在可达性、分配后的实际传播和任务效果分别判断。

## 局部组合与参数归属

[三组主绑定](hb-multiscale-weighted-message.md#2-两类记忆与三个主候选)把当前实现范围收敛为 K（全 KV／sum）、S（全 SSD／state）、X（普通节点 KV、Hub SSD）。聚合、描述、评分、分配、状态采用和 Full 在该实例中一并绑定。无 Hub 时 X 与 K 重合；已有的其他局部候选仍可在具体问题需要时使用。

描述与评分采用[共同编码和区域候选关系](hb-multiscale-weighted-message.md#12-共同编码与区域候选关系)，约束主绑定显式使用 received 下界；区域中间表示只用于评分。

[三档整体参数绑定](parameter-profiles.md)说明 independent、region_shared、layer_shared 如何作用于 Agg、记忆、描述与评分、Full 和区域头。共享范围与局部算法是不同选择，两个共享档使用基座加节点适配，运行状态始终私有。原 FC／HB 教学实例保留各自参数定义，新比较实例使用统一的三档归属。

接收另有 source_residual 与 incoming_self_attention 两个函数候选。[接收自注意力的完整替换](hb-multiscale-weighted-message.md#11-接收自注意力的完整替换)给定来源残差、两种轻量自注意力和加宽自注意力四个尺寸配置，分别明确宽度、头数、尺度与三档参数归属，可用于 FC、三类 HB 和 LH，分别与 K／S／X 组合。FC／HB 生成规模表已展开这四个配置并计入完整参数；计数和准备阶段成本见[比较绑定 §5.1](backbone-comparison.md#51-接收自注意力的参数与准备成本)。候选目录按这些维度组织，实验无需穷尽笛卡尔积。

[主干比较绑定与尺寸阶梯](backbone-comparison.md)进一步规定内容维度、区内节点数、二次幂预算与 quad／Shannon／pool 对照，并给出参数计数口径。[交互面板](../explorers/backbone-atlas/README.md)展示三类 HB 的连接与时间，以及 FC／HB 的规模筛选。生成表覆盖已选组合，后续实验按需选取；LH、部分 Hub 放置与其他分区另有定义，未全部纳入规模表。

## 教学、比较与接续

第一次理解完整计算，可用[全连接两次输入](fully-connected-layered-weighted-message.md#2-两层图上的两次输入)或 [HB 分叉汇合](hb-lattice-weighted-message.md#2-三层图上的分叉汇合与记忆)。[显式残差 MoE](explicit-residual-moe.md)用于 Transformer 到 Tide 的教学过渡；常规 Transformer／MoE 的实验对照由下游安排。

新增候选可以改变拓扑，也可以只重组局部候选、分区、时间或参数。已有语义足够时，在应用文档给出完整绑定及相应性质的条件；新函数或依赖则需要相应语义定义。候选的形成理由回到[主干记录](backbone-design.md)，局部机制的取舍见[带权消息设计空间](../memos/design/weighted-message-design.md)。新增意图、替代机制和具体实验均可逐步推进，无须穷举所有维度。

编辑连接、公式或规模绑定后，按[维护地图](../../ARTIFACTS.md#各条工作线的维护地图)核对受影响的参考代码、图形和展示，再构建对应组。面板数据与参数表由维护源生成。
