---
type: research-memo-index
status: active-index
tags:
  - tide
  - index
---

# Tide 研究备忘

备忘保存研究意图、机制分析、训练问题、补充数学与外部背景。这里按文档用途组织，内容可以涉及语义，也可以涉及语义的应用。正典与局部教材的阅读路线见[语义入口](../README.md#语义)；拓扑、局部组合和完整绑定见[应用目录](../instances/README.md)。

design/ 讨论意图、机制与启发，training/ 讨论学习与观测，mathematics/ 给出声明条件下的推导，background/ 汇集外部文献。HB／LH 的具体构造缘由、取舍和扩展移到实例目录，与使用它们的候选相邻维护。

## 设计与研究

| 文档 | 职责 |
|---|---|
| [研究意图与探索路径](design/design-requirements-and-architecture.md) | 保存想获得的能力、意图与机制的多对多关系，以及上层语义、局部语义和应用之间的支撑 |
| [主干候选的形成与取舍](../instances/backbone-design.md) | 应用层记录：意图如何推动 HB／LH、交换、Hub、时间与规模选择；保留替代方案及接续问题 |
| [局部架构设计总览](design/weighted-message-design.md) | 设计能力、依赖路径、候选取舍、外部证据与可修正的判断 |
| [脑科学启发的模型问题](design/neuroscience-inspired-model-questions.md) | 从生物观察保留能力设想、机制想像与可逐步推进的研究问题 |
| [路由统计与物理类比](design/routing-statistics-and-physical-analogies.md) | 路径相关、路由熵与宏观描述的候选问题 |
| [学习风险与诊断](training/sparse-routing-risks-and-diagnostics.md) | 跨架构的路径漂移、信用距离、状态负担、饥饿与干预机制 |
| [可选研究问题与对照](training/weighted-message-research-questions.md) | 按问题取用的训练目标、对照、观测和结果解释 |

研究意图贯穿语义与应用，一项意图可以由多种机制共同支持，后续也可以替换其中的机制。局部设计总览展开带权支线的选择，主干记录说明具体组合的来由；[展示脚本](../presentations/design-introduction/script.md)安排首次听众的理解顺序。诊断和研究问题供实验按需使用，推进优先级由研究者安排。[实现核对](../reference/weighted-message-implementation-checks.md)单独检查所实现的函数与数学定义是否一致。

## 数学补充

| 文档 | 对象与结论 |
|---|---|
| [函数保持生长](mathematics/function-preserving-growth.md) | 状态投影与输出保持、中性残差、有限 DAG 细化及固定汇合的分段等价 |
| [来源、聚合与读出](mathematics/provenance-and-readout.md) | 保持指定观察的聚合与状态压缩、保守来源集合及读出时间 |
| [自适应路由下界](mathematics/adaptive-routing-prefill-lower-bound.md) | 确定性精确黑盒查询模型的轮数下界、完整证明及向具体模型转用的条件 |

这些材料在各自声明的模型内给出结论。局部架构的函数与组合定理仍由[局部教材](../local-semantics/weighted-message/architecture-learning-note.md)承担。

## 外部背景

- [人脑信号传播调查](background/neuroscience-survey.md)：解剖、生理与文献事实。
- [编译器与 dataflow](background/compiler-and-dataflow.md)：程序表示、验证、通信模型与逻辑进展。
- [学习资源](../resources/learning-resources.md)：按问题选读的书目与练习。

## 版本与追溯入口

- [版本差异与下游采用](../records/semantic-versions-and-adoption.md)：当前迁移要点、采用声明与引用方式。
- [文档修订记录](../records/history/document-revisions.md)：带日期的编辑过程与旧章节线索。
- [内容迁移索引](../records/history/migration-map.md)：旧文件的现行落点。
- [有界通信来源](../records/history/bounded-communication-origins.md)：直接通信范围、输入扩散与输出收拢的旧文依据及迁移、删除线索。
- [LH 历史](../records/history/lh-design-origins.md)：早期实现与语义设计来源。
