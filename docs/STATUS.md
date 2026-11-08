# 当前交接 — 第四批多阶段能力保持链运行中（2026-11-08）

观测时间：北京时间2026-11-08 17:22:04。用户已授权连续推进[第四批固定研究表](studies/fourth-batch.md)。原生基线、信息诊断、NPU梯度机制、两候选新进程恢复和batch校准均已执行；TIDE seed11已训练到日志第80/128步，训练loss 0.499010。最新验证在第64步，相对MSE 0.583060（初始1.0）。这是中途学习信号，尚未判定终态拟合或非零接入后的能力保持。[当前交接证据](evidence/fourth-batch-handoff.json)。

| 轴/阶段 | 已观测状态 | 边界/后继 |
| --- | --- | --- |
| 执行 | `s4-capability`运行中，物理NPU14→logical npu:0；`s4-capability-completion`已于17:14:15提交，CPU等待原服务终态，尚未请求NPU | 两者实际cgroup均在background.slice；协调器不并行开展模型计算 |
| 原生基线 | 查表val64准确率90.625%；末词复制68.75% | 80%门槛仅查表通过；复制保留失败，不换提示或只挑答对样本；独立test尚未检查 |
| 信息诊断 | 查表局部MLP原维ridge相对MSE0.014904，PCA16为0.228786 | 本线性诊断下压缩有损，不证明所有16维非线性模型上限 |
| 严格数值 | 原生完整/缓存单例max_abs 5.43594e-5，atol1e-5/rtol1e-4严格失败；初态α0逐位保持 | 有限超差及未知原因保留，历史Core5严格失败不解除；不是模型质量差 |
| 离散/机制 | 上述缓存例argmax相同；两候选NPU外部MSE导数、内部分支/输入梯度及唯一optimizer参数检查通过 | 只覆盖实际检查范围，不认证全部离散路径 |
| 恢复 | 两候选新进程3步B2模型/AdamW/标准化/数据游标/CPU+NPU+Python RNG及运行身份精确比较通过 | tensor/值相同，不要求序列化字节相同；没有128步完整独立等价检查 |
| 学习/保持 | 共同B128、seed11 TIDE训练中；近参数MLP对照及固定10%混合尚未完成 | TIDE拟合及保持门槛通过才追加两候选seed22；最后统一独立test，不自动扩层/调alpha/加步 |

调度校准暴露一个实验侧遗漏：B128预计128步约27.35min，超过已冻结单次20min限制。原活动作业不改动、不打断；完成协调器已实际提交，只有原候选明确因该时限失败且服务无后代时才从完整checkpoint续接。每个TIDE候选最多一次续接，目标仍为128次逻辑更新；未发布步骤可能重算，原失败、重算下界/未知量与实际耗时分别保留。[完成策略及提交观测](evidence/fourth-batch-budget-completion.json)。

模型源码固定`1ec3f737cf4767dd3f807b5892e0dc2b7c60e29a`（`.local/source-fourth-batch`）；协调器单独固定`4502d073e78876bca2abe88947bda9aeb7dbdc98`（`.local/source-fourth-completion`）。续接模型worker仍执行原模型源码/config/cache。两快照hash已复核；数学0770593、基座85d161d、安装锁248个公共包文件及私有环境不变。配置`configs/studies/fourth/b-preserve.json`，已选B128/128步/AdamW。

预算仍为1NPU/4CPU线程/24GiB、所有尝试已分配计算≤2h，共同截止**2026-11-09 05:00:01北京时间**，新增磁盘≤4GiB；启动账号约165/800GiB。完成链扣除原服务完整耗时和30秒余量，不重新获得2h或12h。原MLP在10%混合时仍执行，不宣称节省计算。

Lens小记录及baseline/information/mechanisms/batch-calibration已读回verified；正式影子训练记录中。项目`tide-learning-lab`，组TL-B4-BASE/INFO/CHECK/SHADOW/PRESERVE；私有入口与SSH方法在`.local/LENS.md`。项目JSON/JSONL权威，R14–R17仍awaiting，终态后按各轴人工审阅，不凭退出码生成科学支持。

第三批结论不变：HB三seed平均dense/quad99.805%/99.040%，稀疏Full/消息约少45%；B四候选200步均未达旧25%门槛。旧新符号任务负面结果不定义新B路线成败。[第三批结论](studies/third-batch-results.md)。A本批不新增训练。当前作业、查看、停止和恢复方法见[RUNNING](RUNNING.md)。
