# 当前交接 — 第三批后台执行中（2026-11-08 09:34）

已按用户批准的[12小时研究表](studies/third-batch.md)实际提交7个后台服务，核对均进入`background.slice`。**共同截止北京时间2026-11-08 21:29:11**，计算、排队、检查和清理均有界；各组更短上限仍有效。现在可交接，无需本会话等待全部长任务。[固定源码、实时状态/日志、停止及恢复方法](RUNNING.md)、[实际启动证据](evidence/third-batch-launch.json)。

| 组 / 作业 | 最新已核对状态 | 剩余工作与上限 |
| --- | --- | --- |
| HB seed22/33各dense/quad，`s3-hb-*-s*` | 4个CPU训练running，已完成16次以上更新，JSONL已有学习/工作记录 | 各B32/400步；4线程/32GiB并行，各含清理≤3h |
| `s3-hb-evaluation` | 后台服务running，应用waiting_dependency | 四训练终态后，复用seed11，2048独立映射+独立重复推理计时；CPU1线程/4GiB计算≤1h |
| `s3-growth` / TL-B3 | 已获物理NPU1；冻结准入及冻结/全骨干精确恢复均通过，正式200步比较已启动 | full/head/MLP/TIDE各200步，保留40步诊断；全部分配后≤3h |
| `s3-npu-sampling` / TL-S2 | **completed**，NPU3已释放，无存活后代，Lens verified/readback_matches | 原同一步off/on模型与optimizer逐位一致；单次同步步19.59/19.13秒，不作可靠性能结论 |

冻结准入的小输入检查：初始task/LM行为相同、前缀续接通过原阈值，梯度在第二步进入旁路，冻结骨干未获梯度/更新且不进入optimizer。此处数值通过仅限该检查；不解除历史CPU/NPU或quad严格失败。另行完成的冻结/全骨干恢复在3步/B2新进程比较中，模型、SGD、CPU/NPU/Python RNG及身份均逐位一致；正式学习仍未得出结论。采样不是独立CPU参考、完整应用恢复或学习验证。

运行源码固定`1c6cb44197cb603cdab1e89b1be1a96bf692c789`，只读`.local/source-third-batch`，树hash `284807226729141d002e77c01d65c687ef6bc5652cb7864a434a2596cce021fa`，已普通推送。工作目录之后仅更新交接和研究图，不修改活动作业读取的归档。数学0770593、基座85d161d和私有环境不变；[准备证据](evidence/third-batch-preflight.json)复核248个基座文件和依赖version/RECORD。新增诊断的CPU小型FC/HB模型/AdamW/RNG轨迹、新进程恢复、非有限失败、时限与后代清理检查通过。

第二批结果和失败保留：[总报告](studies/second-batch-results.md)、[终态证据](evidence/second-batch-terminal.json)。HB seed11 test99.609%/98.438%，Full/消息少42.61%/43.41%，单seed和非独占CPU不支持稳定速度或质量非劣声明；FC delay8有退化，原B1旁路预测贡献未证实。第二批两项NPU超时没有执行，不被本批新尝试改写。

项目JSON/JSONL为权威，Lens沿用原认证实例和项目`tide-learning-lab`，私有入口/SSH转发见`.local/LENS.md`，按TL-A2-REPEAT/TL-B3/TL-S2筛选。当前训练持续投影，终态后逐一核对projection。研究图新增三个实际组，执行状态与待审阅结果分开，无自动科学支持边。

资源：启动账号165/800GiB、新增磁盘≤20GiB；峰值计算约12CPU线程/64GiB、最多2NPU，现在采样卡已释放。cgroup v1确认生命周期隔离，CPU/内存配额是请求；线程及服务时限有效。后续唤醒时审阅终态、每组一页结论和下一批选项，本次不自动扩大规模或延长预算。
