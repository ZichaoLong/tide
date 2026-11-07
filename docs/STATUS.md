# 当前交接 — 首批终态，待用户验收（2026-11-08）

当前无活动或排队的本项目训练作业。最后训练于北京时间11月8日00:34左右结束；全部训练exit0、实际cgroup与已知后代均为空。历史严格审计失败、主动停止测试和profile超时分别保留，不能把全部终态说成全部通过。[首批总报告与建议](studies/first-batch-results.md)、[生命周期复核](evidence/first-batch-lifecycle.json)。本次状态检查没有新开实验。

## 已审阅结果

| 研究 | 实际结果 | 判断 |
| --- | --- | --- |
| FC CPU64，3seed×400步 | 独立test97.27%/99.22%/99.61%；移除历史后接近机会水平 | 当前短程关联检索学习成立，不外推长延迟/其他任务 |
| FC NPU32，80步 | val24.22%，query-only10.94%，移除历史10.16% | 有历史利用信号，未达50%门槛；不能继承CPU400步效果 |
| HB CPU32，稠密/quad各80步 | val31.25%/35.16%；Full少37.63%、消息少39.75%、总用时少3.55% | 两者未达50%，不启动本批正式重复；单seed不足以证明质量或速度优势 |
| Qwen生长，原模型/TIDE/MLP各40步 | val12.50%/17.97%/12.50%；关闭TIDE分支accuracy不变 | 接入成立，实际预测贡献未证实，未达25%门槛，不重复 |

各组一页结论：[A0](studies/TL-A0.md)、[A1](studies/TL-A1.md)、[A1-NPU](studies/TL-A1-NPU.md)、[A2](studies/TL-A2.md)、[B0](studies/TL-B0.md)、[B1](studies/TL-B1.md)、[C](studies/TL-C.md)。新终态结果、曲线、Full/消息总计、Lens对应与checkpoint hash见[evidence](evidence/first-batch-terminal.json)。只有CPU64三seed确认使用了最终独立test；其余先导的test仍未检查。

## 独立检查与测量边界

执行完成、严格数值、离散语义、恢复和学习效果分开记录。CPU64 FC小型独立检查通过；NPU FC、CPU HB quad、历史FP32 AdamW及原Qwen CPU cache严格失败继续保留，原阈值未改，有限误差原因仍未知。实际尺寸离散/机制与相应同栈新进程完整恢复在各自范围通过；跨后端、不同物理映射或完整长轨迹恢复未认证。四个新增80步终态checkpoint核对step/hash一致，不升级为恢复通过。[启动审计](startup-audit.md)、[NPU](evidence/a1-npu-admission.json)、[HB](evidence/a2-hb-admission.json)。

发现冻结训练循环的记录节奏偏差：配置log_every=5但每一步都采集trace/梯度，计时含每步观测开销。源码和旧结果未改写，修复列入后续准备。[测量说明](measurement-notes.md)。HB候选/状态更新仅减少2.07%，不能把Full减少37.63%解释成总计算减少同等比例；NPU profile超时保留，尚未得到算子级耗时归因。

## 身份、资源与查看

数学0770593、公共基座85d161d；依赖身份见dependencies四份锁。CPU确认/B源码0d7ec5b，NPU0494d20，HB训练aff03ee、等待驱动e712831；所有只读快照保留。实际配置和查看/停止/恢复说明见[RUNNING](RUNNING.md)，本批没有待恢复的中断训练。

Lens项目`tide-learning-lab`，终态run均已读回verified，现有认证服务active/running；查看入口与转发方法仅在`.local/LENS.md`，没有部署新服务或改访问策略。项目记录权威，Lens为查看副本。

账号约165GiB/硬800GiB；项目runs约63MiB、artifacts约11MiB，未超磁盘预算。cgroup v1的CPU/内存控制器未按服务委派，CPUQuota/MemoryMax仅为请求；应用线程和服务超时实际生效。当前本项目不占NPU，预算与RSS/HBM观测仍分列，不将配置值称作实际硬限制。

## 下一决定

首批交用户验收，不自动增加规模、任务或候选。建议先修正记录节奏并做少量FC检索机制评估；若继续主问题，保持当前HB架构/任务，仅安排一对较长训练检验是否训练不足，再依据原50%门槛决定多seed。B冻结骨干对照和轻量NPU采样是后续可选项，预算与停止条件见总报告；均未提交。
