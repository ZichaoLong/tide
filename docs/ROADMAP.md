# 当前路线图与验收范围

固定采用 `0770593838a37c790a568226930567ed60f99b83` 的 core-5 与带权局部定义。
既有主体交付及三项审查修复见[固定源码资格](evidence/core5-review-fixes-20261028.md)。
用户进一步授权的限制调查、历史性能经验迁移、参数匹配设施及私有矩阵合批已交付实现和有限验证；
数值修复、实际收益、负结果及未解除的限制分别列出，不等于所有数值域或全部低成本优化已经完成。
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

## 本轮增量的交付状态

| 增量 | 交付与验收 | 状态 |
| --- | --- | --- |
| A | 固定源码复现、首次分歧定位；LH/fractal/core-3经验索引 | 已交付；FP16 GDN及纯Torch异步小反例仍严格失败，D64梯度/参数见证保留；quad共同偏移已有独立闭式修复和后端资格 |
| B | 五族/评分/来源共享参数清单、预算匹配、拓扑与二进制材料 | 已实现并通过相关固定CPU资格；实际92.62亿参数材料加载，执行拒绝单列 |
| C | ready复制、冻结bank、canonical发布去重及成本 | 完整CPU/NPU相关资格通过；NPU真实allocated小降，同材料单次端到端反而约慢8.5% |
| D | 合法共享KV低秩投影、初始化消冗、共享组内同形私有投影真实 BMM/独立 VJP | 已交付；07e56ca CPU574项/2 strict xfail、NPU相关单/双卡训练恢复及CUDA构建通过；最大真实私有批32，单次CPU约持平、NPU mixed稍慢。跨不同输入/基座的全面合批及窗口反向构造仍有空间 |
| E | 同拓扑/同主体参数有限CPU/mixed/resident对照、profile | 已执行并保留全部终态；小型通过用例倾向CPU，大D/部分D64严格失败，无法给出大型选型胜者 |

[本轮固定证据](evidence/core5-performance-transfer-20261028.md)：实现2da3f2c、诊断/设施0503593；
CPU376 passed/2 strict xfail、设施14项、native24图×4更新/精确恢复，NPU四组件×两dtype通过，CUDA同源构建/CPU核验。
参数匹配、完整正确性失败、单次性能、实际存储与profile各自解释。没有放宽容差或用FP64通过替换FP32失败。
[经验记录](performance-lessons.md)同时保存正、负结果；长期硬件待验不阻塞本机工作。
[后续私有矩阵批/quad证据](evidence/core5-private-projection-20261028.md)固定07e56ca；
原D2048材料仍严格残差拒绝，没有大型吞吐结论。原报告保持原终态，后续结果不覆盖它。

## 用户新批准的闭环增量（进行中）

| 顺序 | 工作与当前事实 | 尚需闭环 |
| --- | --- | --- |
| F1 实际失败 | wide2048和matched2048原输入均已捕获，独立120位Decimal定位为远处最大评分导致局部响应相消；首个失败行成为可复现测试数据 | 原完整材料修复后核对，不能由局部通过宣布大图通过 |
| F2 共同求解器 | cap质量分位原点/局部根区间、约束分支消去大共同阈值；Python/C++、设备源码、独立闭式/实际行/VJP检查已实现 | 固定源码CPU/NPU/CUDA构建资格；GPU实机仍待硬件 |
| F3 完整训练差异 | D64分支/free集合一致但连续误差传播；wide2048修复后窗口可执行，完整keys仍严格不一致；半精度GDN局部L2/scaled-RMS未发现公式错误 | 核对/改进KV实际投影和其他可证实算术边界，保留无法消除的严格见证 |
| F4 有限选型 | 失败快照/独立Decimal/实际成功调用观察；benchmark在独立核对前释放候选，推理不复制未更新参数 | 同材料和core-3主体匹配有限测量，完整通过才称等价加速；经验/负结果继续记录 |

本阶段已获实现、有限实验、提交和普通推送授权；阶段提交后继续。捕获job终态成功仅表示获得实际失败资料，
内部两次benchmark保持failed。新求解开发432项、二进制材料5项和设备源码CPU适配37项通过；原wide/D64完整对照仍failed，
尚不形成大型吞吐资格。正式资格先固定实现提交，详见STATUS。

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
| 新D64 KV完整训练 | 480节点CPU两种调度、Python及HB64 CPU/resident/mixed存在严格浮点差异；两步CPU诊断还见参数梯度和SGD参数差异，FP64显式提升两步通过。先核对算术边界/条件敏感性，不能只比较输出或据此宣称新优化无影响 |
| quad大D数值域 | 共同偏移的实际响应/残差/VJP修复已通过CPU独立闭式、NPU和CUDA构建资格；原wide2048和matched2048仍残差拒绝。下一步先提取失败分配的实际输入，区分大分差/求解和传播条件，不能据同一错误字串认定同因或放宽守卫 |
| 后续性能优化 | 共享base/输入组已有真实私有矩阵批；ranked64只覆盖34/479次State，仍22501次mm及大量索引/copy，State最大批1。研究跨不同输入的同形运算、小批设备提交成本与窗口反向复用；完整梯度/恢复和端到端成本一起验证 |
| 容量/数值定义域 | 有限在途队列、各分区预算和正支持下溢等公开拒绝继续生效；限制不能静默截断状态、改算法或改变活动集。本轮约92.62亿参数CPU推理材料构造/warmup已实际执行，但未通过测量窗；大规模训练/设备容量及跨卡吞吐未获资格 |

上述严格失败保持原记录。后续优先顺序：实际失败输入的数值响应/一阶训练边界 → 扩大同形私有模块合批与反向复用 →
在可获得硬件上完成CUDA/x86_64资格。每项形成独立可审查增量，先定向测试，固定提交后补必要资格和一次有限测量。
不自动恢复历史队列、不重跑所有开关矩阵、不以参数匹配等同函数或效果匹配。公共任务头、loss配方和模型效果仍由下游负责。
