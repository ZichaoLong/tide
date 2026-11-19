---
type: semantic-version-record
status: active
semantic-version: tide-core-5
as-of: 2026-11-17
tags:
  - tide
  - semantic-version
---

# 语义版本差异与下游采用

本页记录影响采用的语义差异、配置迁移与有来源的采用声明。数学定义见[接口索引](../reference/semantic-interface-index.md)所列教材；编辑历程见[文档修订记录](history/document-revisions.md)。

## 核心语义

| 版本 | 变化 | 采用时核对 |
|---|---|---|
| tide-core-3 | 聚合输入保留入口、父边、终端或端口的来源标签 | 保留来源；裸值聚合由忽略标签表达 |
| tide-core-4 | SettleGraph 激活容量取上界，允许少选与空选择；缺席终端输出为 $\bot$ | 空激活、未激活候选的状态作用、缺席输出与编码恢复 |
| tide-core-5 | SettleGraph 沿用 TimedDAG 的独立 $X_v$、原子集合、局部签名及逐边／逐端口返回族；以严格分层和时间对齐给出单次结算 | 保留平行边与一般端口；广播取 $\operatorname{Full}=\Delta\circ\operatorname{EmitValue}$；旧记录按明确边界映射比较 |

tide-core-5 保持 TimedDAG 与 PositiveDelayGraph 的局部函数、参数顺序和作用结果。局部特化可独立演进；核心版本与采用的不可变提交共同定位定义。

2026-10-21 的教材勘误补全[分块教材 §8.3](../timed-dag-chunk-prefill-learning-note.md#83-自回归输出的结构检查)的自回归检查：采用固定输出坐标、各端口允许前缀，以及包含消息存在性依赖的保守来源集合。只追溯实际已发送消息不能独立作为自回归证据。核心递归、局部函数与继续状态保持，核心版本仍为 tide-core-5。

## 局部支线的采用要点

下表只记录变化及采用影响。完整函数与当前参数范围由对应教材定义；章节重排、编号和检查过程见[修订记录](history/document-revisions.md)，旧路径见[迁移索引](history/migration-map.md)。

| 修订 | 变化与兼容性 | 采用时核对 |
|---|---|---|
| 2026-10-14 非空约束分配 | 数值语义改变：constrained 的总发送下限为 $\min(1,\sum_v\mu_v)$；配对硬函数全低分时保留最高分候选 | 全低分、单候选及低接收分支；定义见[区域分配 §4.1.3](../local-semantics/weighted-message/region-allocation-learning-note.md#413-节点增益与固定阈值分配) |
| 2026-10-17 稠密 Attention | 新增无求根发送规则，既有约束函数保持 | 按各自发送上限、总量与零边界实现；见[Attention 份额分配](../local-semantics/weighted-message/region-allocation-learning-note.md#42-通过-attention-份额分配) |
| 2026-10-19 模块组合 | 分开记忆族、记忆描述读取、区域评分与发送；增加基础循环、received 权重、总量缩放及 Read 模式组合 | 既有默认数值保持；新增描述字段及旧键的对应见下方迁移表 |
| 2026-10-20 有限温度条件勘误 | 函数保持；旧 $t_{(4)}$ 条件不足，改用第二名评分 $t_{(2)}$ | 更新采用的充分条件，见[§7.3.2](../local-semantics/weighted-message/region-allocation-learning-note.md#732-正温度的精确性)与附录 A.2 |
| 2026-10-20 反向与零消息勘误 | 函数保持；区分 quad 的固定／松弛总和导数，以及固定总强度 Attention 的两种基权重 | 核对零消息删除、softmax 等值范围与边界反向，见[实现核对 §3](../reference/weighted-message-implementation-checks.md#3-区域分配) |
| 2026-10-20 名称与精确性补充 | 配置名称按下表迁移；补入满足条件的有限正温度精确运行结论 | 旧函数与默认值保持；结论按[架构 §6.3](../local-semantics/weighted-message/architecture-learning-note.md#63-局部分配极限与整网极限)的条件使用 |
| 2026-10-20 完整循环候选 | 新增 gated_delta、ssd 与固定逻辑时间短卷积；原 affine／delta／kv 保持 | 新状态与参数按 [Gated DeltaNet](../local-semantics/weighted-message/recurrent-memory/gated-delta-learning-note.md#附录-s配置原模型对应与工作量)、[SSD](../local-semantics/weighted-message/recurrent-memory/ssd-learning-note.md#附录-s配置原模型对应与工作量)附录绑定；state 描述为短期摘要加多头矩阵查询 |
| 2026-10-20 求值对应澄清 | 函数保持；区分求根区间点与实际响应点，明确卷积抽头的逻辑时间单位 | 反向在实际响应点求值；原序列混合器对应须满足抽头与接收间距条件，见[实现核对](../reference/weighted-message-implementation-checks.md) |
| 2026-10-22 区域控制，基线 [6d3b6e3](https://github.com/ZichaoLong/ObsidianVault/commit/6d3b6e3) | 新增 regional_total／regional_threshold、区域控制描述及头；旧 constrained 与默认数值保持 | 两者采用接收下界和最大倍率上限，分别控制实际总量与阈值；完整参数及配对函数见[区域分配附录 S.1](../local-semantics/weighted-message/region-allocation-learning-note.md#s1-配置名称) |
| 2026-10-22 配对函数配置，基线 [0ad390e](https://github.com/ZichaoLong/ObsidianVault/commit/0ad390e) | 为已定义函数增加 regional_total_zero、regional_threshold_hard 名称 | 保留各自描述与区域头；容量和不适用参数按[配置表](../local-semantics/weighted-message/region-allocation-learning-note.md#s1-配置名称)核对 |
| 2026-10-22 分组证据，基线 [3bdee09](https://github.com/ZichaoLong/ObsidianVault/commit/3bdee09) | 新增 region_evidence=grouped；默认 cat 保留原函数，描述类型保持 | 证据与评分分别绑定；原 match 编码参数的名称对应见下段；定义见[§3.4](../local-semantics/weighted-message/region-allocation-learning-note.md#34-区域证据) |
| 2026-10-22 内容 key，基线 [fefbf7b](https://github.com/ZichaoLong/ObsidianVault/commit/fefbf7b) | 新增 region_key=content；默认 content_stats 保留原 key，描述类型保持 | content 另给独立编码参数；定义见[§3.6.2](../local-semantics/weighted-message/region-allocation-learning-note.md#362-记忆-key)，统计通路与反向见[实现核对 §3.6](../reference/weighted-message-implementation-checks.md#36-区域-attention-的记忆-key) |

2026-11-15 的基线 [1f1c542](https://github.com/ZichaoLong/ObsidianVault/commit/1f1c542) 新增 [HB 多跨度拓扑](../topologies/hb-multiscale.md)、[LH-DAG／PDG](../topologies/lh-exploration.md)及 [role_bounded_quad](../local-semantics/weighted-message/region-allocation-learning-note.md#415-带角色上下界的二次分配)；[5da2bb4](https://github.com/ZichaoLong/ObsidianVault/commit/5da2bb4) 增加 [Settle 位交换图](../topologies/hb-exchange.md)、三图共同绑定及两种完整参数配置。原 HB、旧发送函数与默认值保持，核心版本仍为 tide-core-5。新主实例收敛为 KV／SSD／混合三组，新增角色边界、参数共享和时间绑定须显式采用。当前多跨度图的时间候选收敛为 Timed 与 Settle，Settle 保持 $D>M$ 条件。这些提交不声明下游已经实现或采用。

本轮复核收紧 role_bounded_quad 的描述空间：上下界必须由固定角色函数生成，保证 SelStep 在整个声明定义域上可行；原合法描述的数值函数保持。多跨度相位定理显式限定单位边时延，LH 层次深度改记 $d_{\mathrm{tree}}$，与载荷宽度 $d_{\mathrm{model}}$ 分开。HB 多跨度与位交换的带 hub 主图默认省去末层 H，`terminal_hubs=True` 可恢复上述基线的图；group／singleton 中保留节点与端口输出保持，mixed 中可能改变共同分配，采用时须记录此开关。四份正典和 Settle 的 $D>M$ 条件保持。

分组编码统一名称时，原 match 的 $B^{\mathrm{enc},g},C^{\mathrm{enc},g},b^{\mathrm{enc},g},d^{\mathrm{enc},g}$ 分别对应 $B^{\mathrm{score},g},C^{\mathrm{score},g},b^{\mathrm{score},g},a^{\mathrm{score},g}$；数值保持。区域证据和内容 key 使用各自参数。更换函数会改变后续状态；保留旧缓存继续与新模型从初态运行是两种比较，按[架构附录 S.1](../local-semantics/weighted-message/architecture-learning-note.md#s1-配置对应)声明。

完整循环模块的普通序列对应固定到各自附录中的源码提交与开关。局部候选的上述增补均未据此声明下游已经支持，也未据此声明训练或设备性能结果；采用证据见本页下方。

## 局部名称迁移（2026-10-20）

以下名称替换保留原来的数值函数、参数值及默认选择；核心版本仍为 tide-core-5。

| 旧名称 | 当前名称 | 所指函数或参数 |
|---|---|---|
| selection_mode=soft | selection_mode=constrained | 约束分配，包含稠密与可稀疏的正则族 |
| selection_mode=direct | selection_mode=direct_attention | 直接 Attention |
| selection_mode=attention | selection_mode=unit_attention | 固定总强度 Attention |
| selection_mode=pooled | selection_mode=pooled_attention | 总量缩放 Attention |
| selection_mode=dense | selection_mode=proportional | 按比例分配 |
| selection_mode=hard | selection_mode=hard_selection | 硬选择 |
| budget_family | regularizer | 约束分配的正则函数 |
| tau_soft | allocation_temperature | 发送分配的温度 |
| 局部接收强度 $r,r_v$ | $r^{\mathrm{in}},r_v^{\mathrm{in}}$ | 与正典的节点秩 $r_v$ 分开 |
| 局部激活集合 $A$ | $\mathcal A$ | 与图的边集 $A$ 分开 |
| 分配温度 $\tau_{\mathrm{soft}}$ | $\tau_{\mathrm{sel}}$ | 与正典读取模式 $\tau_j$ 分开 |

下游旧配置按表映射；新旧键同时出现时检查数值一致或拒绝冲突。函数完整定义及参数适用范围见[区域分配附录 S.1](../local-semantics/weighted-message/region-allocation-learning-note.md#s1-配置名称)。有限正温度推论只覆盖满足条件的 quad 与固定 $\alpha>1$ 的 ent；它补充已定义函数的性质，未改变任何分配规则。

## 局部配置迁移（2026-10-19）

下表保留该次修订的名称；其中的发送模式再按上节迁移为当前名称。

| 既有名称或类型 | 该次修订的对应 | 数值与采用说明 |
|---|---|---|
| memory=sum／qry | memory_family=kv，memory_read=sum／qry | 区分状态类型与描述读取，保留原读取函数 |
| attention_base=uniform／reception | selection_mode=attention，allocation_base=uniform／mu | 旧 reception 使用的是 $\mu$，包括增益；数值保持 |
| direct 未声明基权重 | allocation_base=mu | 保留原直接 Attention；received 是使用 $w$ 的新增分支 |
| att 未声明 query 权重 | query_weight=mu | 保留原 query；received 为新增分支 |
| KV 固定 Read$^+$ | read_mode=+ 为默认，可另取 -／0 | 既有默认数值保持；模式改变属于新组合 |
| $D^{\mathrm{ind}}=(\mu,s)$、$D^{\mathrm{att}}=(\mu,x,K,s)$ | 分别在首位加入 $w$ | 支持 query 与份额读取接收量。只采用 mu 分支的旧表示可保留；适配时 $w$ 从本次聚合取得，不能一般由 $\mu$ 反推 |

共同载荷与六个正典签名保持，局部描述空间的具体绑定扩充。下游兼容解析若同时收到新旧键，须检查对应一致或拒绝冲突；不可把旧 reception 解释成 received。正温度连续性新增覆盖 pooled、dense 及 received 分支，硬化定理仍仅比较约束分配与原硬函数。循环模块采用自己的状态投影与时间规则。

## 2026-11-15：角色分配归入教材与三档参数绑定

角色二次分配的定义、可行性、导数及配置已整合到[区域教材](../local-semantics/weighted-message/region-allocation-learning-note.md#415-带角色上下界的二次分配)，候选函数保持。实例共用的[参数绑定](../instances/parameter-profiles.md)迁到 instances/：旧 shared 改名 layer_shared，新增 region_shared，independent 保持。两个共享档的模块范围、适配规则及私有状态相同，区域档把共享组按实际区域拆分；HB 优先区域档，LH 默认沿用层级档。FC 两个共享档等价。计数、CSV、面板和检查同步；四份正典、旧教学绑定及 MoE 展示保持。本次不声明下游已实现或取得实验效果。

## 2026-11-17：同刻消息的局部自注意力接收

2026-11-17 新增[同刻局部自注意力接收](../local-semantics/weighted-message/architecture-learning-note.md#23-同刻消息的局部自注意力)，配置为 agg_mode=incoming_self_attention。原接收明确命名 source_residual，未另指定的旧绑定继续使用原函数。新候选按接收强度形成 Key／Value 先验和最终 Query 汇聚，逐条残差与来源非线性后返回 $(r^{\mathrm{in}},u)$；四个合并投影与新来源嵌入按三档参数绑定。四份正典及 tide-core-5 保持，接收候选由采用提交定位。

新增候选可组合 K／S／X，完整参考尺寸及接口在 [HB §1.1](../instances/hb-multiscale-weighted-message.md#11-接收自注意力的完整替换)，数值性质在共同架构附录 A.1。采用时核对零质量屏蔽、外层强度权重、全部正强度读取集、合并投影的适配粒度和准备阶段成本；改变 Agg 后须重新说明状态采用。现有 CSV 明确标记 source_residual，自注意力参数增量可另复算；此修订不声明下游实现或训练效果。

## 2026-11-17：接收尺寸的实例绑定

[HB 接收配置](../instances/hb-multiscale-weighted-message.md#11-接收自注意力的完整替换)明确区分 agg_mode 的函数选择与 agg_profile 的尺寸规则，增加 $d_A=p_a=d$ 的加宽档，保留来源残差及 $d_A=64$／128、$p_a=d/8$ 的轻量档。仅扩充实例绑定，四份正典、局部教材和核心版本保持；KV sum／qry 未变。

FC／HB 规模表与面板分别列出四档，参数已包含 MLP、新旧来源嵌入和适配器随形状的变化。采用新 CSV 时以 agg_profile 参与行身份，并直接使用该行总参；原 incoming_attention_increment 仅用于 MLP 宽度不变的注意力增量比较。准备成本仍按实际消息数计量，不按 Full 的预算比例折减。

## 2026-11-17：共同编码、区域关系与主分配迁移

相对基线 [a7490fa](https://github.com/ZichaoLong/ObsidianVault/commit/a7490fa)，区域教材新增 local_score=encoded、region_score=self_attention、可选 region_evidence=encoded，以及 role_bounded_shannon。完整描述空间、参数形状、零／弱边界、Shannon 求解与分支导数均在[区域教材](../local-semantics/weighted-message/region-allocation-learning-note.md)；共同架构补充接口引用与选择工作。四份正典和 tide-core-5 保持，原函数名及默认含义保留。

新 FC 比较、HB 多跨度／Settle 和 LH 主绑定从 cat＋att/content_stats 改为共同编码＋候选 self_attention，并显式从 lower_base=mu 改为 received，保留 interval、固定增益 2、阈值 0、温度 1。节点编码宽度为 d/2；区域块每头 32 维、FFN 宽度 d、一层。三个主分配为 quad、Shannon、pool；group 分区只替换 D 区，H 区保持角色 quad。LH 的 mixed Shannon 与拆区 pool 分别按实例绑定，固定通过区域省去评分网络。

新主计数替换旧评分和 query 参数，沿用 independent／region_shared／layer_shared 的归属。CSV 按 allocation_profile 分为 676,200 行，页面去重保存 225,400 份参数配置；角色、区域尺寸和分配身份均可查询。Shannon/pool 在正接收假设下的代理为全部参数，quad 继续按 k/n 估算。旧 FC／HB 教学实例、记忆函数、四档接收尺寸、拓扑与时间绑定保持。

这是新模型绑定，不能仅重命名字段后将旧权重和缓存视为等价状态。下游需采用新参数、重算或另行声明状态转换，并记录所用提交；这里未声明设备实现、训练质量或吞吐已验证。研究意图与机制取舍分别补入既有设计文档；[研究维护技能](../skills/tide-research-maintenance/SKILL.md)只组织后续同步职责，不把当前配置冻结为通用规则。

## 下游采用记录

采用声明与符合性验证分别记录；本表保留有日期和来源的信息。

| 下游 | 采用声明 | 依据与范围 | 符合性证据 |
|---|---|---|---|
| [graph-execution-foundation，2026-10-28](https://github.com/ZichaoLong/tide/tree/2c46aa7d0b77afe7a7b82ed428466e36f5162de4) | tide-core-5／上游 0770593；一般 SettleGraph、五记忆与十种发送等已实现 | ascend 当前提交 2c46aa7；固定快照、代码和运行记录的定向审视见[审视记录](reviews/2026-10-28-ascend-implementation-review.md) | 19 份上游快照与固定 Git 对象一致；抽查 9 个任务的 19 份留存文件，新增 30 项 CPU 检查通过。CPU／NPU 有分范围证据与保留失败；CUDA 仅构建，尚无任务质量相当时的稀疏收益证据 |
| graph-execution-foundation，2026-10-10 历史声明 | 当时为 tide-core-3；包含若干局部模块特化，带权 KV 特例尚未实现 | 2026-10-10 维护者说明（当日记录）；该状态已由上行更新 | 保留原日期和来源；当时的文档修订未审计实现 |
| fractal-latcarf | 具体采用范围由实验记录声明 | SettleGraph 实验平台，维护模块、模型接入和训练设置 | 以该平台的版本对应和实验报告为准 |

## 下游引用

每份采用声明记录：

| 字段 | 内容 |
|---|---|
| upstream repository | https://github.com/ZichaoLong/ObsidianVault |
| upstream revision | 采用的不可变提交 |
| semantic version | 核心语义版本 |
| adopted family | PositiveDelayGraph、TimedDAG 或 SettleGraph，以及附加条件 |
| local choices | 局部函数、尺寸、初态、输入日程、聚合、发送、状态和读出 |
| local extensions | 改变公共函数类型或依赖关系的内容 |
| implementation mapping | 下游维护的定义、配置与代码、测试的双向对应入口；按固定实现版本定位 |
| validation | 比较对象、投影、数值容差、固定提交和验证结果 |

逐项对应与运行证据按[实现核对 §1.3](../reference/weighted-message-implementation-checks.md#13-从定义到验证证据)记录，采用声明链接其入口。上游采用页保留有日期和来源的声明及下游入口；具体支持范围、代码路径、测试和运行结果由下游维护。

离线副本保留来源与提交，从上游单向更新。上游保留一般定义、证明与教学参考程序；新增核对要求不改变既有函数，也不更新下游已采用或已验证的范围。

## 来源

SettleGraph 的早期定义参考了 fractal-latcarf 的实验语义文档。早期材料与来源见[迁移索引](history/migration-map.md)、[LH 历史](history/lh-design-origins.md)和[文档修订记录](history/document-revisions.md)。TIDE 的历史全称为 Topology-Invariant Degree-bounded Expansion Architecture for Autoregressive Token Inference。
