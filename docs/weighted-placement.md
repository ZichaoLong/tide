# 带权 Describe / Sending 的执行位置

固定 [architecture §3](upstream/tide-core-5/local-semantics/weighted-message/architecture-learning-note.md)
及 [allocation §2–6](upstream/tide-core-5/local-semantics/weighted-message/region-allocation-learning-note.md)。
本项扩展 [ExecutionPlacement](execution-placement.md)，不改变图身份、参数绑定、checkpoint v5
或完整 raw Continuation。实际资格与硬件范围见 STATUS；源码存在不表示设备通过。

| 位置 | 精确执行域与返回 |
| --- | --- |
| payload owner | Aggregate、Upd、Next、Full 和私有 KV/循环/卷积状态；消息仍为 `(c,G)` |
| read | 对实际绑定 Read^0 / Read^- / Read^+ 快照执行完整 Describe，包括 Mem、current、score/evidence/key；只移动临时参数/状态/内容视图 |
| control | 完整区域候选的区域 query/Attention score、context/head 和总量/阈值边界 |
| selection | 十种发送的分配/排序/归一化与正支持；精确解 VJP 身份保持，与 legacy HST 无关 |
| events | 本适配器的拓扑、就绪、完整候选和推进仍由 host 在线调度；resident 另行实现/验证 |

Python 使用 `place_model(graph, model, placement)`；native 使用
`Native(graph, model, placement=..., packed=True)`；独立 C++ 使用
`tide::place_model` 后构建独立 Streaming/Greedy。保留全部参数叶身份和别名，
不添加 state_dict 前缀、不修改共享模型或私有状态。重新放置从原始局部程序建立视图。
只识别固定带权 Describe/Sending 和已有 builtin，不隐式替换任意自定义数学程序。

`scoring_dtype=profile|payload|float32|float64` 显式控制带权 Describe 的临时浮点计算。
Read 字段、控制头和分配使用该描述量精度；最终 controls 转回区域 payload 精度，
再按既有 owner 规则传向各节点。NPU 的 Read/control/selection 不接受 FP64；
NPU payload 配 CPU FP64 的全部评分路径可显式请求，实际设备资格单列。
所有时间、逐条 KV/卷积戳始终 int64，不经 scoring dtype 转换。
CPU FP32/FP64是参考构建域；FP16的显式评分/算术和master边界见
[eager精度契约](weighted-precision.md)，其资格与本页原始FP32/FP64门禁分别记录。

正支持在分配所在设备由实际数值结果确定；正值向 payload 转换若下溢为零，明确拒绝，
不悄悄丢弃消息或伪造活动集合。分配器自身的数学支持/下溢契约仍见
[allocation](weighted-allocation.md) 与 [sending](weighted-sending.md)。
改变精度可能改变近似平票/分支；资格继续严格检查路由，不放宽整数/消息存在性。

相同形状、来源 dtype/device 的参数/状态/内容按最多 8 MiB 的行组 stack、cast/copy、unbind，
单个超预算 Tensor 作为不可分割的一行。实际别名只转移一次；独立同值根不合并。
原始 Tensor 版本与每个输出根的 None/连接零由独立 copy VJP 保存；反向只打包实际使用的行。
Read/Region 随后使用已有真正数值 batch 和独立字段/候选 VJP。所有工作计入完整执行成本，
本实现未声称跨所有不同形状使用一次传输或一个厂商 kernel。

| 数学/边界 | 实现 | 独立检查 |
| --- | --- | --- |
| 结构化 Read 与快照 | [Python](../python/tidegraph/weighted/placement.py)、[C++](../cpp/src/weighted_placement_read.cpp) | [placement tests](../tests/test_weighted_placement.py)：五种记忆、0/-/+、反馈/重复边、完整字段/状态根、保留切点 |
| 区域 control/selection | [Python Sending](../python/tidegraph/weighted/programs.py)、[batch](../python/tidegraph/weighted/region_batch.py)、[C++ placement](../cpp/src/weighted_placement_region.cpp)、[step](../cpp/src/weighted_region.cpp) | 十种发送、异构 Attention/共享参数、三步 AdamW、pending 恢复和第四次更新 |
| typed 打包/一阶转换 | [Python](../python/tidegraph/weighted/placement_tensors.py)、[C++](../cpp/src/weighted_placement_copy.cpp) | literal Read^0 FP64 公式；独立行根、有限组容量、超过 2^54 的 metadata、原始版本；[standalone 解析 VJP](../cpp/test/weighted_placement.cpp) |

同 dtype CPU 只验证代码路径与局部等价，不能认证跨设备转移；显式 CPU FP64 scoring
另验证真实 cast 及其反向。mixed-a/b/c 真机、resident、跨卡和性能均保持独立证据边界。
