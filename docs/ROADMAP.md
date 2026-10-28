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
| B | 五族/评分/来源共享参数清单、预算匹配、拓扑与二进制材料 | 已实现并通过相关固定CPU资格；实际92.62亿参数材料已完成有限推理窗，完整状态核对失败单列 |
| C | ready复制、冻结bank、canonical发布去重及成本 | 完整CPU/NPU相关资格通过；NPU真实allocated小降，同材料单次端到端反而约慢8.5% |
| D | 合法共享KV低秩投影、初始化消冗、共享组内同形私有投影真实 BMM/独立 VJP | 已交付；07e56ca CPU574项/2 strict xfail、NPU相关单/双卡训练恢复及CUDA构建通过；最大真实私有批32，单次CPU约持平、NPU mixed稍慢。跨不同输入/基座的全面合批及窗口反向构造仍有空间 |
| E | 同拓扑/同主体参数有限CPU/mixed/resident对照、profile | 已执行并保留全部终态；小型通过用例倾向CPU，大D/部分D64严格失败，无法给出大型选型胜者 |

[本轮固定证据](evidence/core5-performance-transfer-20261028.md)：实现2da3f2c、诊断/设施0503593；
CPU376 passed/2 strict xfail、设施14项、native24图×4更新/精确恢复，NPU四组件×两dtype通过，CUDA同源构建/CPU核验。
参数匹配、完整正确性失败、单次性能、实际存储与profile各自解释。没有放宽容差或用FP64通过替换FP32失败。
[经验记录](performance-lessons.md)同时保存正、负结果；长期硬件待验不阻塞本机工作。
[后续私有矩阵批/quad证据](evidence/core5-private-projection-20261028.md)固定07e56ca；
07e56ca的原D2048材料仍严格残差拒绝；该历史报告保持原终态。474f9af后续已执行同一测量窗，
但完整状态核对仍失败，仍无大型等价吞吐结论；后续结果不覆盖原报告。

## 本次数值与低秩增量的闭环结果

| 顺序 | 已交付及终态 | 尚未解除的边界 |
| --- | --- | --- |
| F1 实际失败 | 两份原D2048失败行提交到tests/data；120位Decimal定位局部相消；公开批量AllocationFailure/失败快照与独立诊断可复现 | scalar、前置域拒绝与常驻失败不承诺同一快照；捕获不是部分成功返回 |
| F2 共同求解器 | cap质量分位局部根区间、约束分支共同阈值消去；固定474f9af CPU746 passed/2 strict xfail、NPU九项及CUDA构建通过 | 任意数值域不由有限检查认证；原正支持/残差/区间塌缩拒绝保持，GPU实机待验 |
| F3 完整训练差异 | 所有eager KV行采用低秩式，独立dense/Jacobian与多步optimizer/恢复通过；30组实际分配局部误差≤8.33111e-8，进入求解器前评分已不同；显式FP64两步诊断通过 | D64 FP32完整返回/梯度/SGD仍严格不等价，D2048完整keys也失败；不能全部归因于求根，resident组合矩阵/tape未优化 |
| F4 有限选型与设施 | 同材料CPU medium近持平、NPU mixed累计单次约−2%；完整核对均通过。匹配92.62亿参数B1推理窗已执行；medium profile独立完成。b62f2db阶段持久化固定6项通过 | 两个大图严格失败不计等价吞吐；单次小收益不证明稳定收益；没有大设备完整训练/跨卡性能结论 |

[474f9af数值/低秩及b62f2db设施证据](evidence/core5-local-allocation-factorized-kv-20261028.md)保存
精确来源、独立检查、终态、原失败和复现命令。CPU study总体failed（5项通过/2项严格失败），
NPU study passed（2项通过）；各项不互相冒充。原材料拒绝解除与完整等价验收分开。
本轮实施、定向资格和约定的有限测量已终态；总体大图严格正确性仍未通过。
本阶段的提交/普通推送已获授权；不自动恢复旧队列或重复性能矩阵，当前交接见STATUS。

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
| eager FP32/FP16 GDN默认AdamW | 新扩大CPU门禁补记FP32两项失败：约1.411885e-6，旧1850434同样复现，原测试保持failed；FP16约.001984的strict xfail仍保留。显式SGD、master和resident不同精度边界的通过不解除它；见[精度](weighted-precision.md) |
| 默认异步双NPU eager VJP | 当前Torch2.12/CANN9.1纯Torch双卡反向也重现invalid event；显式launch blocking通过，不宣称异步已修复；见[mixed设备报告](evidence/core5-device-20261027.md) |
| standalone SDK FP32 affine | fc/affine第三轮evidence差3.09944e-6，mixed-a同见证，原严格失败不改容差；见[mixed设备报告](evidence/core5-device-20261027.md) |
| FP32反馈PDG性能严格对照 | ceae7bf两项maxabs4.11272e-6/2.71201e-6；24配置中的这两项不计等价加速，FP64提升诊断不解除原失败；见[性能报告](evidence/core5-weighted-performance-20261028.md) |
| 新D64 KV完整训练 | 480节点CPU两种调度、Python及HB64 CPU/resident/mixed存在严格浮点差异；两步CPU诊断还见参数梯度和SGD参数差异。1f1ec57的显式FP64两步完整记录/optimizer及换owner下一窗口精确恢复通过；四种接口/调度训练测量通过。原FP32资格未解除 |
| quad大D数值域 | 已捕获并修复两份原失败行的局部相消，原完整材料已越过残差拒绝并执行窗口；但keys严格核对失败。30组实际分配与独立Decimal已分离求解误差/输入差异；1f1ec57同材料D2048显式FP64推理完整核对通过，不能回填FP32；其他数值域未获资格，正支持下溢/区间塌缩继续明确拒绝 |
| 后续性能优化 | 全部eager KV低秩已实施；私有BMM仍限共享base/输入组，ranked64只覆盖34/479次State、State最大批1。medium新profile有1394次mm/288次bmm，但不是大图统计。跨不同输入的同形运算、小批设备提交、resident组合矩阵与窗口反向复用仍有空间；完整梯度/恢复和端到端成本一起验证 |
| 容量/数值定义域 | 有限在途队列、各分区预算和正支持下溢等公开拒绝继续生效；限制不能静默截断状态、改算法或改变活动集。约92.62亿主体参数CPU FP64推理在1f1ec57完整通过，151.416s/窗，候选测量后RSS峰值约90.35GiB；原FP32失败仍保留，大规模训练/设备容量及跨卡吞吐未获资格 |

上述严格失败保持原记录。1f1ec57已完成共享准备稳定顺序、真实optimizer核对与精确恢复的定向资格，
并取得五项显式FP64同材料/参数匹配测量。CPU扩大门禁为572通过/2个原FP32 GDN失败；
NPU eager KV及CUDA构建通过，均见[本阶段证据](evidence/core5-stable-preparation-precision-20261028.md)。

后续优先顺序：
1. 对新补记的FP32 GDN默认AdamW边界验证normalized L2/scaled-RMS一阶收缩：动态Python三步探针通过，
   还需公共Python/C++实现、独立Jacobian、原两项失败及多步恢复/后端资格；不凭探针回写原失败。
2. D64/D2048 FP32的完整浮点一致性仍待改进；已有row/reduction/BMM/局部VJP探针未解决完整训练，未采用。
   已验证的显式FP64是独立配置，不是隐式替换。
3. 扩大跨不同输入/基座的同形私有模块合批、降低设备小批提交及resident窗口反向构造成本；
   每项先核对独立根/共享/恢复，再据一次有限完整计时决定收益。
4. 在可获得硬件上完成CUDA真机和x86_64资格；异步双NPU原最小反例与显式blocking通过分列。

不自动恢复旧队列、不重跑所有开关矩阵，不以参数量匹配等同函数、模型效果或完整训练工作量匹配。
公共任务头、loss配方和模型效果仍由下游负责。当前具体任务/源码/作业只见STATUS。
