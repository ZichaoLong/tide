# 主干构造与参数计数参考

本目录提供面板和数学核对共用的 Python 标准库代码。正式定义分别由 [topologies/](../../topologies/) 与[主干比较绑定](../../instances/backbone-comparison.md)承担；这里构造端点、计算潜在传播和参数形状，不执行整网神经计算或训练。

| 维护源 | 对应定义与职责 |
|---|---|
| [multiscale.py](multiscale.py) | [HB-Multiscale](../../topologies/hb-multiscale.md)：宽／窄轮廓、可选 hub、端点构造、时延支持与潜在来源覆盖 |
| [settle_graphs.py](settle_graphs.py) | [HB-Lattice](../../topologies/hb-lattice.md) 与 [HB-Exchange](../../topologies/hb-exchange.md)：展开、交换与收拢的端点构造 |
| [capacity.py](capacity.py) | FC 与三类 HB 的空间轮廓与时间绑定登记、节点／来源类统计、独立／区域共享／层级共享参数计数及预算活跃代理 |

入口程序将 `reference/` 加入 Python 模块搜索路径，按 `from backbones import multiscale, settle_graphs, capacity` 导入。包内使用相对导入；公共代码不依赖面板目录、HTML 或图形库，也不写生成物。`settle_graphs.py` 与多跨度图共用固定置换规则，种子 `hb-timed-discussion-20261114-v1` 保留其定义身份。

`capacity.PROFILES` 登记空间轮廓，`BOUND_PROFILES` 将其接到主时间绑定；`count` 继续接收空间标识，返回精确标量计数。多跨度 Timed／Settle 共用计数，在生成表和面板中分别列出。导出的 CSV 由面板报表代码换算成 M，内嵌 JSON 保留原始计数。

`count(profile, n, hubs, d, memory, sharing, rank, agg_profile)` 直接返回所选接收配置的完整参数数目，默认 `agg_profile='source_residual'`，保留调用签名；当前计数已迁移到 encoded/self_attention 主绑定。`AGG_PROFILES` 登记来源残差、`incoming_light_64`、`incoming_light_128` 和 `incoming_wide`；`aggregation_config(d, agg_profile)` 解析具体的函数、MLP 隐藏宽度、注意力宽度及头数。完整尺寸规则见[接收自注意力绑定](../../instances/hb-multiscale-weighted-message.md#11-接收自注意力的完整替换)。计数包括逐条 MLP、两种来源嵌入和适配器随形状变化的大小。生成表为每个已有组合分别列出四档。

`regional_config(d)` 给出共同编码与区域尺寸，`regional_inventory(d)` 列出区域四投影、FFN 和读出参数；每份为 2d²+2d。节点清单只计新编码，不叠加旧 cat/key。`ALLOCATION_PROFILES` 为 quad／shannon／pool，分配不增加参数；CSV 分行，页面复用同一份参数数据。`active` 对 quad 使用 k/n，对 Shannon/pool 使用稠密代理，pool 没有数值 k。

`incoming_attention_increment(profile, n, hubs, d, sharing, rank, attention_width)` 只计算新增四个注意力矩阵与来源嵌入，适用于来源 MLP 宽度不变的对照；不能仅将它加到来源残差计数来得到加宽档。四个矩阵按合并投影分别适配，头数不改变参数数目。准备工作按实际到达数计算，不能由参数增量代替。

[教学与核对程序](../../examples/README.md)检查构造及公式；[主干面板](../../explorers/backbone-atlas/README.md)负责视图、CSV 和 Markdown 范围表。FC 目前通过节点和来源类统计参与计数，尚未提供用于面板绘图的逐边构造；LH 的教学构造仍在 HB／LH 核对程序中。

修改本目录后，从仓库根目录查影响并核对定义、例子和界面公式，再重建：

```bash
python3 tools/check_artifacts.py --affected 20-tide-decentralized-neural-network/reference/backbones
python3 20-tide-decentralized-neural-network/examples/hb_lh_checks.py
python3 20-tide-decentralized-neural-network/examples/hb_settle_checks.py
python3 20-tide-decentralized-neural-network/examples/hb_compact_checks.py
python3 20-tide-decentralized-neural-network/examples/backbone_capacity_checks.py
python3 20-tide-decentralized-neural-network/examples/incoming_attention_checks.py
python3 tools/build.py --group backbone-atlas
python3 tools/check_artifacts.py --require-built --group backbone-atlas
```

生成登记与人工核对来源见[维护地图](../../../ARTIFACTS.md)。构图、计数与数学正文需要同步审阅，构建通过本身不能证明语义一致。
