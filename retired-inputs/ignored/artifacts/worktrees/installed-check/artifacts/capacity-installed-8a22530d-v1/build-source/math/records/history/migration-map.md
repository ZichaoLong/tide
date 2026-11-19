---
type: migration-index
status: reference
as-of: 2026-11-16
tags:
  - tide
  - history
---

# 旧内容重写与追溯索引

## 2026-11-16 研究意图与语义应用

将跨层研究缘由与具体候选记录分开维护，入口按语义与语义应用组织。四份正典和全部局部教材保持；这次整理说明意图、语义支撑、拓扑和局部组合的关系，不新增数学条件。

| 原位置或职责 | 当前维护入口 |
|---|---|
| `memos/design/design-requirements-and-architecture.md` 的总体需求说明 | 同路径的[研究意图与探索路径](../../memos/design/design-requirements-and-architecture.md)，保留两次计算与可达范围推导 |
| `memos/design/hb-and-lh.md` | [instances/backbone-design.md](../../instances/backbone-design.md)，主干候选的形成、取舍与替代选择 |
| 拓扑、局部组合与参数绑定的分散入口 | [instances/README.md](../../instances/README.md)，语义应用的阅读路线 |

局部机制、脑科学与物理启发继续位于 memos/design/；memos 按研究文档用途组织，其内容可以跨越语义与应用。面板、生成报表和展示同步引用，生成物仍由维护源重建。

## 2026-11-15 局部教材与三档参数绑定

整理前版本为 [636d755](https://github.com/ZichaoLong/ObsidianVault/commit/636d755)。角色分配合入原区域教材，参数归属说明迁到实例层，原局部函数继续按绑定使用。

| 原位置 | 当前维护入口 |
|---|---|
| `local-semantics/weighted-message/role-bounded-allocation.md` | [区域教材 §4.1.5](../../local-semantics/weighted-message/region-allocation-learning-note.md#415-带角色上下界的二次分配)，求解与导数在 §6.5，连续性在 §8.3，配置在 S.4 |
| `local-semantics/weighted-message/parameter-profiles.md` | [instances/parameter-profiles.md](../../instances/parameter-profiles.md)，三档整体参数绑定 |

旧 shared 对应 layer_shared，原计数保持；新增 region_shared 按实际区域拆分共享组。既有 MoE 教学实例与展示维持原位置和内容。

## 2026-11-15 主干面板与参考计算

整理前版本为 [85dccd9](https://github.com/ZichaoLong/ObsidianVault/commit/85dccd9)。本次调整目录职责和当前引用，四份正典、连接规则、置换种子、配置标识、参数计数与时间绑定保持。

| 原位置 | 当前维护入口 |
|---|---|
| `explorers/hb-timed/` | [主干拓扑与规模面板](../../explorers/backbone-atlas/README.md)，包含三类 HB 的图与时间、FC／HB 规模筛选 |
| 旧面板的 `topology.py` | [reference/backbones/multiscale.py](../../reference/backbones/multiscale.py) |
| 旧面板的 `settle_topology.py` | [reference/backbones/settle_graphs.py](../../reference/backbones/settle_graphs.py) |
| 旧面板的 `capacity.py` | [公共计数](../../reference/backbones/capacity.py)与[面板报表输出](../../explorers/backbone-atlas/capacity_reports.py)分离 |
| `memos/design/hb-timed-and-lh.md` | [主干候选的形成与取舍](../../instances/backbone-design.md) |
| `topologies/hb-settle-exchange.md` | [HB-Exchange 位交换图](../../topologies/hb-exchange.md)，继续采用原 Settle 时间绑定 |
| `presentations/design-introduction-script.md` | [展示目录内的讲述脚本](../../presentations/design-introduction/script.md) |

构建组由 `hb-topology-atlas` 改为 `backbone-atlas`，CI 名称同步。旧机器的忽略产物不会随 Git 迁移，按[升级说明](../../explorers/backbone-atlas/README.md#从旧目录升级)将旧目录移至仓库外保存，再重建新组；仓库内不保留第二套展示源。历史提交身份与核验快照仍按原版本解释。

## 较早的整理背景

2026-09-14 的整理将 20-tide 定位为数学教材与语义上游；下表记录 2026-09-20 的非正典目录整理及 2026-10-20 的后续归属。内容对应按主题追溯。整理前完整版本可从提交 **133d638** 追溯；更早一次大整合前的材料见 **d27819f**。

例如在仓库根目录执行：

~~~bash
git show 133d638:20-tide-decentralized-neural-network/tide-mathematical-foundations.md
~~~

## 内容去向

| 原文件／部分 | 现行落点 |
|---|---|
| README、current-mainline 核心定位与学习路线 | [[../../README]]、[[../../reference/semantic-interface-index]] |
| 旧数学基础中的图、事件与切面定义 | 正典教材；不保留第二套核心定义 |
| transition、状态嵌入、生长与节点细化 | [[../../memos/mathematics/function-preserving-growth]] |
| kernel 分块、scan 与有限组合 | 图级条件见 [[../../timed-dag-chunk-prefill-learning-note|正典分块预填充教材]]；旧局部推导从 Git 历史追溯 |
| owner、数值依赖与读出归因 | [[../../memos/mathematics/provenance-and-readout]] |
| 自适应路由下界 | [[../../memos/mathematics/adaptive-routing-prefill-lower-bound]] |
| selector、记忆、恢复与负载 | [[../../memos/design/weighted-message-design]] |
| 训练困难、可归因比较与诊断 | [[../../memos/training/sparse-routing-risks-and-diagnostics]] |
| 执行能力、成本与物理映射 | 当前实现、设备映射和 profiling 由独立实验仓库维护；旧稿从 Git 历史追溯 |
| LH 早期实现与语义设计来源 | [[lh-design-origins]]，不作为当前能力声明 |
| 脑科学调查与数字模型启发 | [[../../memos/background/neuroscience-survey]]、[[../../memos/design/neuroscience-inspired-model-questions]] |
| 编译器与 dataflow | [[../../memos/background/compiler-and-dataflow]] |
| 统计力学与信息动力学 | [[../../memos/design/routing-statistics-and-physical-analogies]] |
| SCC 旧迁移索引、学习路线 | [[../../resources/learning-resources]] |

## 退出的规则

- 旧文“唯一正式入口”“七份核心职责文件”等权威声明退出。
- 空输入时保存坐标自动逐时更新、从非候选中激活等旧规则不沿用。时间衰减重写为统一逻辑时间上的有效状态解码，不产生自主激活。
- 零时延不再列为研究候选；解释正时延为何重要的反例仍可用于教学。
- Full 结果回写未进入核心；边界说明见 [[../../timed-dag-region-selector-learning-note#134-让完整输出改变后续状态或选择|TimedDAG §13.4]]。
- region 隐含共同读写 KV／SSM 的处方删除；参数共享与明确 owner 的普通消息机制保留。
- 任意 DAG 分区收缩仍是 DAG 的泛化删除；合法分块必须检查真实跨块依赖。
- “stateful 但无负载历史就是 token-local”、全项目强制生长阶梯、已完成语义仍称待定义等表述删除。
- 旧单位时延窗口边界、显式 allocator 与固定 phase 只按原受限范围追溯，不覆盖一般正时延教材。

旧架构图、toy 代码与运行时计划不再作为当前目录内容；需要追溯时使用 Git 历史。当前正时延图的教学参考程序由[实现核对文档](../../reference/weighted-message-implementation-checks.md)定位。

书籍目录保持为本地阅读资源，不纳入本次数学语义提交；不根据文件名判断或删除 PDF。

## 2026-10-20 路径与内容归属

本次整理前版本为 [2e196c1](https://github.com/ZichaoLong/ObsidianVault/commit/2e196c1)。四份正典教材逐字保持。

| 原路径或内容 | 当前落点 |
|---|---|
| `weighted-kv-hb-lattice-example.md` | [HB-Lattice 实例](../../instances/hb-lattice-weighted-message.md)，图与时间构造可组合不同记忆 |
| `weighted-kv-implementation-checks.md` | [带权消息实现核对](../../reference/weighted-message-implementation-checks.md)，覆盖整条局部支线 |
| `memos/architecture/selector-and-memory.md` | [带权消息设计总览](../../memos/design/weighted-message-design.md)，整体候选与依赖分析 |
| `memos/learning-systems/weighted-kv-experiment-plan.md` | [可选研究问题](../../memos/training/weighted-message-research-questions.md)，按需问题与对照 |
| `semantics-anchor.md` 的共同定义汇总 | [接口索引](../../reference/semantic-interface-index.md)；原路径当时保留为入口，现已删除 |
| KV 与循环教材的整网性质 | [共同架构 §6](../../local-semantics/weighted-message/architecture-learning-note.md#6-数值性质的组合)及附录 A；两种记忆各证局部条件 |
| 版本页的详细修订表 | [文档修订记录](document-revisions.md) |

前次整理保留 KV 与分层实例的正文路径；下节记录随后完成的目录分层。历史章节号按对应提交解释。

学习资源的早期书目来自 `133d638:20-tide-decentralized-neural-network/appendices/tide-directed-learning-roadmap.md`。

## 2026-10-20 目录分层与拓扑拆分

整理前版本为 [e4f1cda](https://github.com/ZichaoLong/ObsidianVault/commit/e4f1cda)。四份正典逐字保持。局部函数、拓扑、完整实例、检索核对和版本记录分别归入对应目录；设计问题与外部文献分开归类。

| 原正文路径 | 现行正文路径 |
|---|---|
| `local-architecture-learning-note.md` | [local-semantics/weighted-message/architecture-learning-note.md](../../local-semantics/weighted-message/architecture-learning-note.md) |
| `region-allocation-learning-note.md` | [local-semantics/weighted-message/region-allocation-learning-note.md](../../local-semantics/weighted-message/region-allocation-learning-note.md) |
| `weighted-kv-sparse-activation-example.md` | [local-semantics/weighted-message/kv-memory-learning-note.md](../../local-semantics/weighted-message/kv-memory-learning-note.md) |
| `recurrent-memory-learning-note.md` | [local-semantics/weighted-message/recurrent-memory/affine-delta-learning-note.md](../../local-semantics/weighted-message/recurrent-memory/affine-delta-learning-note.md) |
| `weighted-kv-layered-example.md` | [instances/fully-connected-layered-weighted-message.md](../../instances/fully-connected-layered-weighted-message.md) |
| `hb-lattice-example.md` | [instances/hb-lattice-weighted-message.md](../../instances/hb-lattice-weighted-message.md) |
| `semantic-interface-index.md` | [reference/semantic-interface-index.md](../../reference/semantic-interface-index.md) |
| `local-architecture-implementation-checks.md` | [reference/weighted-message-implementation-checks.md](../../reference/weighted-message-implementation-checks.md) |
| `semantic-versions.md` | [records/semantic-versions-and-adoption.md](../semantic-versions-and-adoption.md) |
| `memos/history/document-revisions.md` | [records/history/document-revisions.md](document-revisions.md) |
| `memos/history/migration-map.md` | [records/history/migration-map.md](migration-map.md) |
| `memos/history/lh-history.md` | [records/history/lh-design-origins.md](lh-design-origins.md) |
| `memos/architecture/local-architecture-design.md` | [memos/design/weighted-message-design.md](../../memos/design/weighted-message-design.md) |
| `memos/learning-systems/local-architecture-research-questions.md` | [memos/training/weighted-message-research-questions.md](../../memos/training/weighted-message-research-questions.md) |
| `memos/learning-systems/learning-risks-and-diagnostics.md` | [memos/training/sparse-routing-risks-and-diagnostics.md](../../memos/training/sparse-routing-risks-and-diagnostics.md) |
| `memos/background/neuroscience-model-ideas.md` | [memos/design/neuroscience-inspired-model-questions.md](../../memos/design/neuroscience-inspired-model-questions.md) |
| `memos/background/statistical-mechanics.md` | [memos/design/routing-statistics-and-physical-analogies.md](../../memos/design/routing-statistics-and-physical-analogies.md) |

[全连接分层拓扑](../../topologies/fully-connected-layered.md)与 [HB-Lattice 拓扑](../../topologies/hb-lattice.md)分别接收两份原实例中的节点、区域、边、端口与时间定义。位置编码、记忆配置、输入强度、传播和实际工作量保留在完整实例中。两篇实例的混合记忆绑定移到 §3，联合求值及成本覆盖各自声明的 KV 与循环模块。

该次整理曾为正典引用保留根目录的 `weighted-kv-sparse-activation-example.md`、`weighted-kv-layered-example.md` 与 `semantics-anchor.md`，并保留数学备忘和教学程序的正典链接。随后按下节清除向下引用和转向入口。

## 2026-10-20 移除转向入口

以 [e6f5513](https://github.com/ZichaoLong/ObsidianVault/commit/e6f5513) 为基线，四份正典清除对局部语义、实例、索引、备忘和程序的引用，数学内容保持。三个旧入口直接删除：

| 已删除入口 | 当前正文 |
|---|---|
| `semantics-anchor.md` | [数学符号与函数对应索引](../../reference/semantic-interface-index.md) |
| `weighted-kv-sparse-activation-example.md` | [带权 KV 教材](../../local-semantics/weighted-message/kv-memory-learning-note.md) |
| `weighted-kv-layered-example.md` | [全连接分层的带权消息实例](../../instances/fully-connected-layered-weighted-message.md) |

仓库内入口已直接引用上表正文。外部旧链接按表迁移；正典的逐篇改动和引用引入时间见[修订记录](document-revisions.md#2026-10-20定义顺序与正典引用边界)。

## 2026-10-20 循环记忆候选归类

整理前版本为 [254c870](https://github.com/ZichaoLong/ObsidianVault/commit/254c870)。下表路径均相对于 local-semantics/weighted-message/；原位置不保留转向页。

| 原路径 | 现行教材 |
|---|---|
| `recurrent-memory-learning-note.md` | [recurrent-memory/affine-delta-learning-note.md](../../local-semantics/weighted-message/recurrent-memory/affine-delta-learning-note.md) |
| `gated-delta-memory-learning-note.md` | [recurrent-memory/gated-delta-learning-note.md](../../local-semantics/weighted-message/recurrent-memory/gated-delta-learning-note.md) |
| `ssd-memory-learning-note.md` | [recurrent-memory/ssd-learning-note.md](../../local-semantics/weighted-message/recurrent-memory/ssd-learning-note.md) |
| `causal-convolution-learning-note.md` | [recurrent-memory/causal-convolution-learning-note.md](../../local-semantics/weighted-message/recurrent-memory/causal-convolution-learning-note.md) |

本次目录整理及共同推导的正文调整见[修订记录](document-revisions.md#2026-10-20循环记忆目录与共同推导)。

## 背景与补充推导的来源

以下材料的早期正文可从 Git `133d638` 追溯。现行文档保留各自声明的论证或文献范围；目录整理不构成新的外部研究核验。

| 材料 | 早期来源 |
|---|---|
| 函数保持生长 | `tide-mathematical-foundations.md` 第五部分；`tide-model-architecture-and-training.md` 第一部分 |
| 来源、聚合与读出 | `tide-mathematical-foundations.md` 第一部分 §2、§3.6 与第三部分 |
| 编译器与 dataflow | `tide-background-history-and-references.md` 第一部分 |
| 人脑信号传播调查 | 同上第二部分，文献范围为 2026-07-22 |
| 脑科学启发的模型问题 | 架构旧文第三部分 §14 |
| 路由统计与物理类比 | `tide-statistical-mechanics-and-information-dynamics.md` |
