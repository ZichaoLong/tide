# Tide 教学与数学核对程序

这些程序给出有限算例、独立公式核对和图结构见证。它们不代表完整 KV／SSD 网络实现、训练结果或设备吞吐证据。所有程序均可从仓库根目录用 `python3 20-tide-decentralized-neural-network/examples/<文件名>` 运行，使用 Python 标准库。

| 程序 | 核对范围 |
|---|---|
| [positive_delay_graph_reference.py](positive_delay_graph_reference.py) | 正时延图的有限切面、规范记录与分段继续 |
| [explicit_residual_moe_checks.py](explicit_residual_moe_checks.py) | 专家贡献、显式残差合并、两轮局部状态与工作量；展示也读取其轨迹 |
| [region_allocation_checks.py](region_allocation_checks.py) | 区域发送公式、独立活动集求解、导数与零温边界 |
| [role_bounded_allocation_checks.py](role_bounded_allocation_checks.py) | 角色 quad／Shannon、received 下界、独立活动面、全输入可行性、缺席及光滑分支导数 |
| [recurrent_memory_checks.py](recurrent_memory_checks.py) | 短卷积、Delta／SSD 公式、弱事件与分段继续 |
| [hb_lh_checks.py](hb_lh_checks.py) | HB 构图与时间、分区、hub 删除；LH-DAG／PDG 教学构造和标量继续 |
| [hb_settle_checks.py](hb_settle_checks.py) | Lattice 字地址、Exchange 位覆盖、度数与 Settle 对齐 |
| [hb_compact_checks.py](hb_compact_checks.py) | 紧凑图相位、来源覆盖、独立交换轮数及深度扩展 |
| [backbone_capacity_checks.py](backbone_capacity_checks.py) | 二次幂预算、逐头适配器、四档接收的完整参数与闭式计数、加宽时的适配秩变化及活跃代理 |
| [regional_attention_checks.py](regional_attention_checks.py) | 共同编码、跨候选记忆影响、完整分配、多头、重排与零／弱边界 |
| [incoming_attention_checks.py](incoming_attention_checks.py) | 接收自注意力的单／多头手算、顺序与强度缩放、零／弱消息、逐消息相互作用及三档参数增量 |

HB 构造和参数计数从[公共参考代码](../reference/backbones/README.md)导入；核对程序和[主干面板](../explorers/backbone-atlas/README.md)分别消费同一份构造。公式与实现的对应入口见[实现核对](../reference/weighted-message-implementation-checks.md)，文档、链接和产物检查工具位于 `tools/` 及仓库根目录的 `tools/`。
