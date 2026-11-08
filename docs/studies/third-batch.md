# 第三批：12小时内的HB重复、扩大评估和预训练旁路比较

2026-11-08用户已接受建议并授权执行。本表在提交前固定；执行状态见[STATUS](../STATUS.md)，第二批[结果](second-batch-results.md)和两个NPU队列超时原样保留。本轮不增加图规模、任务、固定稀疏候选、替换层数或下一批。

| 研究组与问题 | 候选、对照及唯一改变量 | 指标与验收 | 固定配置、预算和停止条件 |
| --- | --- | --- | --- |
| **TL-A2-REPEAT：从零训练完整TIDE**，HB稠密/可学习稀疏的检索和实际工作折衷是否跨seed存在？ | 补seed22/33各一对dense Shannon/quad；复用第二批seed11终态，共3对。四个新模型均从同seed初态训练，固定16节点/4区域、width16、KV/qry、regional_total/k2、temp.25 | 原validation128/50%门槛、test256和去历史干预保留；训练CE/梯度/区域活动/Full/消息/candidate/Upd；终态固定400步，不按最佳验证点选择 | `configs/studies/third/hb-{dense,sparse}-s{22,33}.json`。CPU FP32，每作业1线程/8GiB，B32、400更新/12,800样本、AdamW .001/eps1e-6/decay.01/clip1；同seed data_seed=1000+seed。四作业并行，各含清理≤3h，合计4线程/32GiB |
| **TL-A2-REPEAT：独立质量与推理计时**，小测试集是否掩盖稳定差别？ | 对上述6个固定400步checkpoint做同一新测试集；再独立重复推理。新评估不改变训练、候选或读出 | 2,048个唯一test映射，逐例预测/CE，逐seed配对正确差、准确率和CE差；公共工作总计。计时保留7次原始值、中位数/范围/标准差；不自动判非劣或速度优越 | `hb-evaluation.json`；CPU1线程/4GiB，等4个HB作业终态≤10,860秒，之后计算≤1h；缺失/失败模型单列，已有独立checkpoint仍可评估。共同12h截止时间始终生效 |
| **TL-B3：预训练模型生长**，冻结骨干是否让TIDE旁路形成可预测贡献？ | 同缓存Qwen0.5B：冻结+head、冻结+TIDE、冻结+MLP；另保留原骨干全参数继续训练+head。四个候选seed11、同数据和更新暴露，TIDE/MLP零输出投影。冻结策略与分支类型为声明的改变量 | 初始行为/续接、梯度进入/冻结参数不被optimizer更新；适用的完整恢复。原40步/25%门槛在40步独立记录，不以200步结果覆盖；终态val128、达原门槛后test256。40/200步分支on/off逐例干预；与重训MLP/head对照分别解释 | `b-stage.json`、`b-{full,head,mlp,tide}.json`。1NPU/CPU4线程/24GiB，FP32/B32/SGD .01/clip1，各200步/6,400样本，非40步checkpoint恢复。准入≤600秒，冻结/全骨干恢复各≤750秒，随后四候选计算与记录共用剩余预算；整个已分配阶段≤3h |
| **TL-S2：共享实现观测**，主线程时间消耗与详细观测开销在哪里？ | 重试未执行的20ms主线程采样；1次warmup后从相同模型/optimizer/RNG对比observe off/on | 同一步终态模型/optimizer逐位比较、实际形状/工作与同步总耗时；既有严格失败保留 | `npu-sampling.json`；独立1NPU/CPU4线程/8GiB，分配后含记录与清理≤15min；只做已有3步，不把Python栈样本说成NPU kernel时间或可靠速度提升 |

所有配置位于`configs/studies/third/`。任务仍为8 keys/8 values、4关联、delay2、独立episode/full BPTT。FC长度专题不在本批。A和B共同任务与hash映射划分不变，但不同预训练背景、架构、参数、更新量和设备成本禁止把两条路线直接排名。

## 固定数据与计时口径

`research/data/hb-test2048-v1.json`由`build_hb_holdout.py`在读入新训练结果前生成。SHA256为`31baa50f378ff609f08c6d0f7f56cad748774c15c8cc2ef7cce3c510bf3a92ec`。原hash规则仍为train80%/val10%/test10%，同一映射所有排列/查询同split。固定seed20261108、确定性抽样顺序，取前2,048个不同test映射，每映射一个query；排除原seed20261107/step0/test256的全部映射。标签、唯一性、split和排除规则有独立检查。此评估扩展了样本量；不是另一个任务，也不作为选配方的验证集。

HB正式推理计时使用新集前64个例子、B32、CPU FP32/1线程/no_grad/trace off。每模型warmup两轮；7轮，每seed稠密→稀疏与稀疏→稠密交替。窗口含已备好CPU输入的完整前向、公共内部计数，排除加载、loss/argmax、JSONL、Lens；输出在窗口外校验重复逐位一致。质量评估和构建先完成，计时不与本批HB训练重叠。主机非独占，记录负载；不同训练后模型的实际路由/工作不同，因此不是同模型实现等价benchmark。

B采用本地已固定的`Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`：缓存完整、原tokenizer、eager栈和初始接入已有证据，无新下载。符号codebook和分类头是实验侧新增，不能将检索准确率称作语言质量。B先独立验证冻结TIDE准入及3步/B2新进程精确恢复，全骨干控制另做恢复；只阻塞依赖失败门槛的路径。冻结head/MLP复用冻结参数/optimizer检查的共有机制，不声明它们获得了独立TIDE数值验收。

## Batch、并行与五类验收

采用重新读取的`run-ml-experiments/references/measurement.md`。HB已有B32/B128校准和400步学习证据，B已有B8/B32校准；本次为配对重复和固定暴露比较保留B32，而非沿用B2 smoke默认。独立样本batch=每设备microbatch=每次更新global batch=32，accumulation=1；不将节点行数或7个时间步算作独立样本，不声称全局最优batch。HB四seed/candidate作业并行，B四候选顺序共用一张卡；采样可在另一真实空闲卡独立运行。NPU请求显式npu_count=1，按优先级0/FIFO、两次采样和整组锁分配，无共享占卡或抢占。

执行、严格数值、离散语义、完整恢复、学习效果分别记录。沿用数学`0770593838a37c790a568226930567ed60f99b83`、基座`85d161d5c9148eac2b164f89a0ba4204b4331898`和实际安装锁。core5原atol1e-6/rtol1e-5、growth检查原atol1e-5/rtol1e-4及失败不变；有限超差经过既有有界机制排查后可做标记探索，未知原因仍未知。新diagnostic只读观测先检查模型/optimizer/RNG轨迹及新进程恢复；B用适用NPU检查。没有把所有A限定CPU FP64，也没有每步跑CPU参考的要求。

非有限输出/梯度、非法配置、因果或共享梯度/状态错误、适用精确恢复失败立即停止相应路径。质量门槛失败是负面学习结果，不改问题或自动加步。执行完成不能改写严格失败；成功恢复小轨迹不认证400/200步完整恢复。新模型间质量差与同模型实现数值差分开解释。HB依然缺少固定稀疏对照，不能把所有收益归因于可学习路由。

## 总预算、记录与交接

第一次统一提交写入不可变`.local/third-batch-budget.json`，各作业取得同一个12h绝对deadline，包含队列、准入、运行、评估和清理；服务截止另预留30秒清理。不是每候选训练12h。B优先提交，采样独立随后提交；两项NPU最多等8h，单卡计算上限分别3h/15min，不自动重排。CPU依赖等待也在后台服务内，不能从依赖结束重获12h。

峰值约CPU12个计算线程/RAM64GiB，最多2张真正空闲NPU；CPU评估等待器额外请求1线程/4GiB但等待时不加载模型，正式计算在HB训练退出后。cgroup v1生命周期隔离有效，CPUQuota/MemoryMax仅请求，实际应用线程与服务时限有效。账号启动165/800GiB、文件系统约907GiB可用；新增磁盘≤20GiB。B成功候选和成功恢复检查的临时大checkpoint按已有策略退役，保留hash/结果/逐例干预；失败checkpoint保留，临时峰值估计低于20GiB。HB终态checkpoint保留。

使用只读源码归档和原私有环境；不得修改活动输入或基座。项目JSON/JSONL为权威，Lens项目`tide-learning-lab`以study/run ID关联。正式提交前验证小型当前事件进入同一私有实例；私有入口只存`.local/LENS.md`。研究图新增真实研究组和计划/排队/运行状态；结果节点等待人工审阅，不凭exit0产生科学支持或迁移边。

本批终态再交付每组一页结论、配置/依赖与恢复证据、Lens对应、更新研究图和有依据的下一批选项，交由用户验收；不自动扩张后续规模。
