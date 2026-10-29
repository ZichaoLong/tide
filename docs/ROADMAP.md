# 当前路线图与验收范围

数学固定`0770593838a37c790a568226930567ed60f99b83`的core-5与带权局部定义。
[STATUS](STATUS.md)是唯一当前交接；[execution-flows](execution-flows.md)是持续契约。
用户授权逐项补齐非GPU边界，GPU延后；每项保留原输入、严格标准、有限预算和停止/重启条件。
任务终态、相关资格通过和限制解除是不同结论。完成记录由Git与[证据索引](evidence/README.md)保存。

## 已有公共能力与入口

| 范围 | 契约和证据入口 |
| --- | --- |
| PDG反馈、非对齐TimedDAG、一般SettleGraph，同秩区域/独立端口/原样嵌入 | [总对应](semantic-map.md)、[Settle](settle-embedding.md)、[结构化表示](structured-records.md) |
| 五记忆、原始时间短卷积、十种发送、适用的评分/证据/key、来源分类与共享 | [KV](weighted-kv-sharing.md)、[循环](weighted-recurrent.md)、[分配](weighted-allocation.md)、[发送](weighted-sending.md)、[评分](weighted-scoring.md)、[来源](weighted-source-binding.md) |
| 在线streaming/prefill、实际矩阵批与独立VJP根 | [packing](weighted-packed.md)、[区域批](weighted-region-batch.md)、[计算共享](weighted-compute-sharing.md)、[f03e9be资格](evidence/core5-independent-projection-20261029.md) |
| CPU、混合、常驻FP32/FP16、单/双NPU训练和精确续接 | [能力矩阵](execution-capabilities.md)、[常驻训练](resident-weighted-training.md)、[节点训练](evidence/core5-weighted-node-training-20261028.md) |
| 参数清单、core-3参数量匹配拓扑/同材料、有限完整训练计时和诊断 | [工作负载](weighted-workloads.md)、[精度](weighted-precision.md)、[历史经验](performance-lessons.md) |

上述入口说明已实现能力和限定资格，不构成所有配置/拓扑/尺度/平台的笛卡尔资格。
下游拥有数据、任务头、loss设计、训练配方和效果搜索；基座验证机制与数学对应。

## 逐项推进与停止条件

| 项目 | 当前证据与本轮剩余动作 | 停止或重启条件 |
| --- | --- | --- |
| D64 FP32完整训练 | e0d6e27单行私有VJP消冗后原两步native/Python仍有记录、梯度、参数和SGD槽差异；两调度自身恢复精确。见[eager证据](evidence/core5-eager-boundary-20261029.md) | 本候选数值修复停止；不重复九项分段精度/外层汇合探针。重启须有覆盖局部算术、中间伴随与共享贡献的具体新方案及独立根证据 |
| D2048 FP32 | [实际首写入](evidence/core5-boundary-isolation-20261029.md)：来源+私有批链194坐标失败；标量FP32对局部FP64也有289坐标失败，连续局部FP64后cast为0。原quad残差拒绝已修复，原全图keys失败未解除 | 没有公共整段算术/VJP方案前停止9B全图重跑；局部更准确不保证与旧标量FP32相同。显式全CPU FP64已有独立原窗通过证据 |
| 反馈PDG FP32 | 原W32/B4四窗记录/梯度/SGD槽仍失败，参数通过；reference/streaming/prefill三者自身恢复均精确。显式FP64四窗独立通过 | 单行VJP方案到此停止；新方案须覆盖反馈递归与跨窗更新，不能仅核对最终参数 |
| FP16 GDN/default AdamW | [归一化VJP内部FP64](evidence/core5-boundary-isolation-20261029.md)仍第二步差.00198364；两个原strict xfail保留。FP32对应见证已由ad6d1e7解除 | 不改epsilon/loss scale/输入或容差；仅提升归一化反向候选停止。重启需新的完整局部伴随/舍入方案 |
| NPU运行时初始化顺序 | 固定资格暴露CPU autograd后才注册NPU触发device_ready_queues_断言；独立标量进程已确认late失败/early通过，独立NPU资格14项+eager6组通过 | 应用先resolve_device再首次autograd；已启动CPU引擎的晚注册路径保留限制，等待运行时修复。当前门禁分进程，不改数学实现 |
| 默认异步双NPU eager VJP | CPU桥完整delta-pdg仍EE1017，未完成4更新/恢复；小桥接通过不足以认证。显式blocking是另一个配置 | 同环境同事件方案停止；重启须运行时修复或新的事件所有权方案，先反例再完整图/独立根/恢复 |
| 单行私有VJP | Python/C++直接建图、轻量一阶/版本守卫已交付；固定e0d6e27 CPU517 passed/2原strict xfail。NPU相关开发已通过；[15c3e65 NPU固定资格](evidence/core5-resident-boundary-20261029.md)通过，[有限完整对照](evidence/core5-boundary-performance-20261029.md)已终态，CPU未见收益 | 以实际端到端成本决定下一步；不从减少私有图数直接推断收益 |
| 异形字段packing与小提交 | 按device/dtype合并真实字段、8MiB分组、按shape/offset恢复；int64独立，None/零/版本守卫保留。mixed-a/c完整核对通过、单次1.505→1.477/2.183→1.848s，allocator峰值不变；小算子/scalar仍显著 | 若小算子/提交仍主导，须据新profile提出具体边界；无真实可合并域不盲目扩大batch |
| resident组合矩阵/窗口反向 | KV改为B N(u)+U(V N(u))，保存实际B/V/U tapes；主机构造/seed/run/bridge/reduce分别计量，[固定资格](evidence/core5-resident-boundary-20261029.md)通过，medium构造928ms、执行/等待1094ms、完整窗2794ms，实际分段已取得 | 完整adjoint绑定journal/root/参数地址，不能按shape跨窗口缓存。下一候选为显式模板重绑，先独立验证绑定/版本/跨窗，不能缓存旧窗口对象；不跳过版本/独立根 |
| 较大NPU训练/跨卡 | D64匹配9076975参数，单/双NPU显式blocking自身磁盘恢复精确，独立CPU均在node89 keys严格失败；resident在Aggregate scratch构造拒绝 | 本轮各一次已终态，不复跑同候选；新增8c48f49按物理入边/端口缩小resident局部槽，CPU9项通过，设备资格中；新候选仅重测受影响resident |
| 诊断与训练观测设施 | scalar/native实际分配观察已通过；常驻20–22失败部分行、实际梯度/SGD槽/canonical共享owner解码及磁盘换owner检查[固定资格](evidence/core5-resident-boundary-20261029.md)通过 | 不补造未计算残差，不以参考重算结果。fake布局测试必须有真实公共owner消费者作独立检查 |
| 容量/正支持 | 显式正int64容量选项、静态Settle上界与分区预算；原正支持下溢/区间塌缩仍拒绝 | 保留表示下界、同时在途容量与实际预算拒绝；无静默裁剪/截断/改活动集合。不以保守fixture预设作为引擎上限 |
| CUDA/x86_64 | GPU依用户延后；本机aarch64无x86目标。当前新增resident/C++路径不继承旧CUDA构建报告 | GPU待用户恢复且取得硬件；x86待目标环境。新源码构建、前向、反向、optimizer、常驻、跨卡及恢复各自验收 |

原失败记录和严格断言不变；新增错误与旧失败分别保留。本轮在构建/资格全部终态后串行执行有限性能，
不重复旧矩阵，不把参数量匹配等同函数、工作量或模型效果匹配，也不宣称低成本优化已经全部完成。
当前具体作业和接续命令只见STATUS；新结果只更新本表、相关契约、证据和performance-lessons。
