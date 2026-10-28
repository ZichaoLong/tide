# 当前路线图与验收范围

固定采用 `0770593838a37c790a568226930567ed60f99b83` 的 core-5 与带权局部定义。
既有主体交付及三项审查修复见[固定源码资格](evidence/core5-review-fixes-20261028.md)。
用户进一步授权处理已知限制、迁移历史性能经验并补齐参数匹配性能设施；以下A—E为当前待办。
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

## 当前实施增量

| 增量 | 交付与验收 | 状态 |
| --- | --- | --- |
| A | 当前固定源码限制复现、首次分歧定位；LH/fractal/core-3经验索引，文档纠偏 | 已记录；纯Torch异步小反例仍失败，见数值隔离证据 |
| B | 五记忆/评分/来源共享静态参数清单；参数预算匹配；可扩展拓扑、二进制材料、容量及正确性/性能记录 | 实现及开发测试通过，固定提交资格随后 |
| C | 有证据支持的队列复制、参数存储及程序构造优化；完整根/状态/恢复与前后成本 | ready副本/冻结bank复用已实现；设备资格与成本对照进行中 |
| D | 低秩/共享计算、进一步合批；独立公式/VJP与在线调度资格，收益与负结果都记录 | KV共享适配低秩投影、初始化消冗已实现；测量随后 |
| E | 同拓扑和同主体参数预算有限对照；宽2048/窄节点、全活动/稀疏，CPU/mixed/resident选型与最终资格 | 已选定有限案例；实施提交后运行 |

经验记录只保存适用条件、源码、验证及测量证据，不另建任务台账。长期硬件待验不阻塞本机工作。

## 已完成的审查修复

实现`1d19c59`：ClockedState嵌套参数/独立VJP、quad正支持下溢拒绝、resident参数bank分配前准入及Full预留。
CPU442项、NPU两dtype四组件（含单设备/双卡训练与恢复）、CUDA三架构构建/CPU31项通过。
首次NPU环境初始化失败和早期取消保持原终态；[报告](evidence/core5-review-fixes-20261028.md)分列来源、独立反例及复现入口。
旧任务全部终态；本轮按新问题建立新冻结源码与有限任务，不恢复历史队列。

## 剩余验证与限制

| 事项 | 当前状态与下一步条件 |
| --- | --- |
| CUDA真机 | 源码、aarch64构建、sm80/90/100常驻核及安装loader已交付；实际GPU前向、反向、optimizer、驻留、跨卡、容量/恢复与性能未验。有硬件后按[设备门禁](device-control.md)与各模块复现命令分层执行，不以CPU/编译替代 |
| x86_64目标 | 源码保留可移植入口；本轮工具链与设备证据来自aarch64。新目标须独立构建及资格，不能继承本机结果 |
| eager FP16 GDN默认AdamW | CPU Python/native严格见证约.001984，strict xfail仍保留。显式SGD、master和resident不同精度边界的通过不解除它；见[精度](weighted-precision.md) |
| 默认异步双NPU eager VJP | 当前Torch2.12/CANN9.1纯Torch双卡反向也重现invalid event；显式launch blocking通过，不宣称异步已修复；见[mixed设备报告](evidence/core5-device-20261027.md) |
| standalone SDK FP32 affine | fc/affine第三轮evidence差3.09944e-6，mixed-a同见证，原严格失败不改容差；见[mixed设备报告](evidence/core5-device-20261027.md) |
| FP32反馈PDG性能严格对照 | ceae7bf两项maxabs4.11272e-6/2.71201e-6；24配置中的这两项不计等价加速，FP64提升诊断不解除原失败；见[性能报告](evidence/core5-weighted-performance-20261028.md) |
| 容量/数值定义域 | 有限在途队列、各分区预算和正支持下溢等公开拒绝继续生效；限制不能静默截断状态、改算法或改变活动集。大规模容量与跨卡吞吐未由本轮小型资格证明 |

上述严格失败保持原记录；A增量在当前源码复现并尝试修复，不改输入/容差来制造通过。
E增量补充用户明确要求的参数匹配选型；既有小型CPU优先结论不外推大型模型。
缺少GPU不阻塞CPU/NPU；不搜索训练效果，不恢复历史实验矩阵。
