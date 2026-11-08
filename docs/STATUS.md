# 当前交接 — 第二批已终态，待验收（2026-11-08）

当前无本项目活动或排队作业。第二批五个作业全部终态：FC机制评估和两个HB训练completed/exit0；NPU采样与冻结骨干B均排队超时，实际workload未启动。最后训练于北京时间04:39:56结束，所有cgroup与已知后代现为空。本次只审阅已有记录，没有新开实验。[总报告与建议](studies/second-batch-results.md)、[原记录hash/曲线/终态证据](evidence/second-batch-terminal.json)。

## 已审阅结论

| 组 | 结果 | 结论边界 |
| --- | --- | --- |
| FC三个旧checkpoint配对评估 | 换query98.44%–100%；delay4为97.66%–100%，delay8为78.91%–92.97% | 支持当前模型随query检索；更长空档有退化。同128个validation映射，非新独立test |
| HB400步，CPU FP32/seed11/B32 | dense/quad validation100%/96.875%；独立test99.609%/98.438%；移除历史10.938%/13.281% | 均达到原50%门槛；当前小HB检索学习成立，不能由单seed主张质量相当或优越 |
| HB工作与时间 | Full少42.61%、消息少43.41%、候选/状态更新少3.70%；本次总用时少10.26% | 一对并行非独占CPU、不同实际工作量；不作可靠加速声明 |
| NPU采样 / 冻结骨干B | 约306.70/610.39秒等卡后超时，未获得设备 | 性能观察及冻结策略准入/恢复/学习都没有新证据；不是模型效果失败 |

一页结论：[FC机制](studies/TL-A1-MECH.md)、[HB400步](studies/TL-A2-LONG.md)、[采样未执行](studies/TL-S1.md)、[冻结B未执行](studies/TL-B2.md)。三个实际运行的Lens投影均verified/readback_matches。两项NPU任务没有模型run，不伪造成功或空曲线。

## 检查与固定输入

执行、严格数值、离散语义、恢复和学习分列。原HB quad/NPU/FP32等严格失败和未知原因保持，未改阈值；CPU FP32学习成立不解除这些失败。相关小型机制与恢复复用；新日志节奏下小FC/HB quad的模型/AdamW/RNG逐值一致及新进程恢复通过。400步终态checkpoint step/hash核对一致，仅为存储完整性，不称新的长轨迹恢复通过。

日志节奏已实际修复：HB各81个详细观测、319个未详细观测步骤，工作总计覆盖400步。区域/梯度只在81个观测点检查，不声称所有样本/节点没有饥饿。80步val准确率和CE与首批同值，不升级为完整数值等价。[测量说明](measurement-notes.md)、[准备证据](evidence/second-batch-preflight.json)。

只读源码94bf712，目录`.local/source-second-batch`，树hash e4fb0dfa6fc74c2c9c8b6a30d43215532321d4e2621d45f6abb0d6e3012133a0；本次已复核未变化。数学0770593、公共基座85d161d与原私有环境固定，consumer wheel保留，实际执行实验包由冻结源码明确覆盖。配置、作业、日志、停止/恢复方法见[RUNNING](RUNNING.md)。

账号约165/800GiB，runs约72MiB、artifacts约11MiB；未用到B大checkpoint预算。cgroup v1只验证生命周期隔离，CPU/内存配额是请求，应用线程及服务时限实际约束。Lens服务active/running，项目`tide-learning-lab`，私有入口与SSH转发只在`.local/LENS.md`。

## 下一决定

建议本批验收后仅增加HB seed22/33两对（CPU4线程/RAM32GiB/并行≤2h），并为两项未执行NPU工作单列较长、有界的等待窗口，原计算预算仍≤9/45分钟。没有提交这些工作；不自动追加更多任务、图规模、替换层数或固定稀疏组合。[具体依据及资源](studies/second-batch-results.md)。首批80步失败、B1贡献未证实以及两次队列超时全部保留。
