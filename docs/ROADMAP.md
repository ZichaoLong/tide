# 当前路线图与验收范围

固定采用 `0770593838a37c790a568226930567ed60f99b83` 的 core-5 与带权局部定义。
主体交付已记录；2026-10-28审查发现的三项缺口已实现修复、开发门禁442项通过，固定源码资格见下方待办。
[STATUS](STATUS.md)是唯一当前交接，[execution-flows](execution-flows.md)是持续契约。
已完成阶段的逐次提交/原失败由Git与[证据索引](evidence/README.md)保留，不恢复旧任务。

| 阶段 | 本轮交付 | 独立证据入口 |
| --- | --- | --- |
| 0 路径、工具链与采用 | 当前模块/私有环境、最小CPU开发链路；19份core-5原文及旧5份独立锁定；本机aarch64，x86_64不外推 | [开发](development.md)、[来源](upstream.json)、[语义对应](semantic-map.md) |
| 1 公共表示与一般图 | PositiveDelayGraph、非对齐TimedDAG、一般SettleGraph；同秩独立区域、端口日程/返回、原样嵌入；SrcClass、多字段Read、int64元数据/精确解码 | [Settle](settle-embedding.md)、[结构化表示](structured-records.md) |
| 2 首个完整组合 | kv/sum/+、cat/ind、regional_total/context/cat/quad；FC/HB及反馈/非对齐/一般端口；完整前后向、多步optimizer、续接/checkpoint | [首组合](evidence/core5-weighted-kv-20261027.md)、[standalone训练](evidence/core5-native-training-20261027.md) |
| 3 正式候选 | affine/delta/gated_delta/ssd、固定逻辑时间短卷积、异构绑定；十种发送、quad/bern/固定alpha ent、全部适用评分/证据/key及来源共享 | [循环](weighted-recurrent.md)、[分配](weighted-allocation.md)、[发送](weighted-sending.md)、[评分](weighted-scoring.md)、[来源](weighted-source-binding.md) |
| 4 联合/packed/VJP | 在线greedy prefill；KV ragged投影、循环仿射扫描、原始时间卷积；Aggregate/State/Read/Full/Region实际数值批和批量VJP；独立根、参数共享、合法计算共享 | [packing](weighted-packed.md)、[区域批](weighted-region-batch.md)、[计算共享](weighted-compute-sharing.md)、[总对应](semantic-map.md) |
| 5 执行位置与训练 | CPU、mixed、FP32/FP16 resident；完整窗口/保留图VJP、canonical master/SGD/AdamW、精确恢复；样本副本、单样本节点放置及共同发布 | [mixed](evidence/core5-device-20261027.md)、[half完整训练](evidence/core5-resident-half-training-20261028.md)、[样本副本](evidence/core5-resident-weighted-peers-20261028.md)、[放置前向](evidence/core5-weighted-node-placement-20261028.md)、[放置反向/训练](evidence/core5-weighted-node-training-20261028.md) |
| 6 有限选型与追溯 | 24配置单次流程、完整调用/profile/工作量、四个新进程启动和分阶段内存；固定定义→配置/符号→独立测试→固定源码证据 | [有限性能](evidence/core5-weighted-performance-20261028.md)、[启动/内存](evidence/core5-weighted-observation-20261028.md)、[能力范围](execution-capabilities.md) |

每行是已实现能力与限定运行范围，不是所有拓扑/配置/尺度/平台的笛卡尔资格。
完整返回等价、指定投影和不同函数性质各自有前提；精确解VJP、迭代导数与硬函数指定反向保持独立身份。
当前图/checkpoint版本由[semantics](semantics.md)统一维护；下游拥有数据、任务头、loss和训练配方。

## 当前修复待办

- 已实现ClockedState递归捕获/分组与独立VJP/更新回归；等待固定源码资格。
- 已实现quad scalar/batch正支持下溢拒绝与Python/C++闭式回归；等待固定源码资格。
- 已实现resident参数bank分配前准入与Full输出预留，CPU边界通过；设备组件构建/真机资格待完成。
- 提交实现后固定源码验证，并独立提交证据。

## 剩余验证与限制

| 事项 | 当前状态与下一步条件 |
| --- | --- |
| CUDA真机 | 源码、aarch64构建、sm80/90/100常驻核及安装loader已交付；实际GPU前向、反向、optimizer、驻留、跨卡、容量/恢复与性能未验。有硬件后按[设备门禁](device-control.md)与各模块复现命令分层执行，不以CPU/编译替代 |
| x86_64目标 | 源码保留可移植入口；本轮工具链与设备证据来自aarch64。新目标须独立构建及资格，不能继承本机结果 |
| eager FP16 GDN默认AdamW | CPU Python/native严格见证约.001984，strict xfail仍保留。显式SGD、master和resident不同精度边界的通过不解除它；见[精度](weighted-precision.md) |
| 默认异步双NPU eager VJP | CANN invalid event仍保留；显式launch blocking路径通过，不宣称异步已修复；见[mixed设备报告](evidence/core5-device-20261027.md) |
| standalone SDK FP32 affine | fc/affine第三轮evidence差3.09944e-6，mixed-a同见证，原严格失败不改容差；见[mixed设备报告](evidence/core5-device-20261027.md) |
| FP32反馈PDG性能严格对照 | ceae7bf两项maxabs4.11272e-6/2.71201e-6；24配置中的这两项不计等价加速，FP64提升诊断不解除原失败；见[性能报告](evidence/core5-weighted-performance-20261028.md) |
| 容量/数值定义域 | 有限在途队列、各分区预算和正支持下溢等公开拒绝继续生效；限制不能静默截断状态、改算法或改变活动集。大规模容量与跨卡吞吐未由本轮小型资格证明 |

上述严格失败保留可复现材料，不能改称不支持或通过。本机其他已声明能力已完成相关资格，
不因缺少GPU而自动排队、重测历史矩阵或搜索训练效果。新增模型/尺度性能由下游选型问题驱动；
本组小中规模优先CPU的有限结论已经给出。
