# v5既有候选的CPU并行安排

2026-11-09用户授权以实际资源和完成时间选择CPU/NPU；NPU小幅更快也不要求等待NPU。本次继续原批次，截止北京时间22:49:13、24累计计算worker小时、最多6个计算worker/24计算线程/144GiB保持。活动FC16、B FC8和旧v4的源码、输入、设备及自动后续不变；不重复其尚未开始的候选。

11:41资源采样：320逻辑CPU，2秒整体忙碌37.6%，可用内存约1.64TiB，账号磁盘169/800GiB。此为共享资源瞬时观察，不是独占benchmark。FC16已有CPU1约20.86秒/步、CPU4约32.17、共享NPU约80.20的记录；不能外推B和Transformer。新增独立worker优先用CPU，避免同一小图增加线程反而变慢。

| 独立作业 | 研究问题和控制 | 固定配置与预算 | 验收/停止条件 |
| --- | --- | --- | --- |
| `lm-a-transformer-cpu-v5` | 补齐原容量对照；没有增加候选或seed | 原2block/width96/head4/FFN512，670592参数；CPU1/FP32/B8/L32/seed11；2048步、2048日程，留64/128前缀；最多1800秒 | CPU机制、3步精确恢复、校准后训练及独立test/推理；任何硬失败停止本候选 |
| `lm-a-hb16-sparse-v5` | 完成原链因dense失败而未运行的quad候选；从零初始化，不继承dense | 原HB16/width32，CPU1/FP32/B8/L32/seed11；128步、512日程前缀，留64/128；最多10800秒 | 独立机制/微型恢复/校准/验证/test/推理；完整配对结论仍受dense失败限制 |
| `lm-b-cpu-cal-v5` | B的冻结前缀+TIDE+语言head路径是否值得放到CPU | 原FC8 dense；CPU1与CPU4各B8/L32、1预热+2计时更新；同缓存、完整alpha1 CE；总1200秒，每项最多480秒 | 只记录训练步时、评价预览与内存；不认证学习或跨后端严格等价，不迁移活动B |

原浅宽FC4在NPU训练56步后、HB16 dense在CPU训练86步后均因`positive allocation batch support underflowed in requested dtype`硬失败。原阈值、checkpoint和失败保留；CPU不保证绕过该问题。HB40依赖等待已因HB16失败终止，未训练，不以资源空闲绕过学习前提。

Transformer原NPU校准约0.0178秒/步；CPU校准将独立记录。后端变化单列，不能把2048步结果当作与128步TIDE等曝光、等计算；128步前缀也保留不同LR日程的限制。稠密/稀疏是独立初始化的比较，不是Shannon训练后切换quad。

扩展预算监督`lm-depth-budget-parallel-v5`包含所有原v5费用和本次三个作业，先启动监督再提交计算。原监督保持不动；新增监督使用新的冻结配置，不修改活动守护进程输入。两者均只停止各自明确列出的本项目作业。

Lens项目仍为`tide-learning-lab`，学习run前缀为`tl-lm-a-transformer-cpu-v5-`（TL-A-CAPACITY）和`tl-lm-a-hb16-sparse-v5-`（TL-A-TOPO）。B技术校准仅保留项目记录，不新增学习曲线。实际源码、状态、Lens投影和预算见`docs/evidence/language-depth-handoff.json`。

通用操作：`python scripts/job.py status artifacts/jobs/JOB_ID`、`logs artifacts/jobs/JOB_ID --lines 50`、`stop artifacts/jobs/JOB_ID`。恢复使用原冻结源码和配置、另一个作业ID及输出、显式checkpoint；新增恢复必须纳入同一预算监督。改变CPU/NPU不叫精确恢复。
