# v12进一步缩减为四卡

2026-11-16用户要求“缩减到4张卡”。本轮仅更改调度，沿用原38链、128Mi目标、
科学源码62e32602、模型/数据/batch/LR、数值阈值及磁盘边界。
[固定16条行政续段](../../configs/studies/language/hb-v12/four-card.json) ·
[暂停/保存审计](../evidence/hb-v12-four-card-transition.json)。

| 项目 | 当前安排 |
| --- | --- |
| 物理卡 | 仅4、9、12、14；四槽及原全寿命设备锁 |
| 保留原服务 | share8-lr3-q3双seed、base128-q3-s22、residual-q3-s22 |
| 本次退出 | 卡6、11上的share8-d128-q3双seed；完整更新边界保存后退出 |
| 后续 | 2条本次暂停续段，加14个原六卡等待器，全部改为`v12-four-*` |
| 截止 | 保持此前已延长的2026-11-19 23:08:07+08 |
| 限额 | 64GiB新增目标/96GiB停止，硬配额余量至少200GiB；不增加科学规模 |

先停止14个尚未分配设备的六卡等待器，再向两个模型进程单独发送SIGTERM，
由原训练循环保存边界状态并正常返回。seed11/22分别保存54.3594/77.1422Mi；
已解码及核对optimizer、progress、source/config/data身份与SHA，服务无存活后代。
外层failed/训练段budget_truncated/stop_signal15保留，明确归类为用户资源暂停。
此前六条行政父状态原样复用，不重新从原失败前位置计算。

新任务复用六卡续训编排，仅参数化固定卡集合、四槽、独立run前缀与输入计划。
科学阶段仍从`.local/source-hb-v12-repair-v2`运行；四条活动原服务的源码/配置未改。
8条待恢复训练仍先执行完整batch两步连续对1+1新进程恢复；8条原未启动修复沿原迁移/前缀门槛。
探针不增加科学曝光；通过后续至128Mi并完成原确认/诊断，不自动重试失败。

按全部16条排队任务都从零训练128Mi、原形状校准速度除以1.5，并每链额外1h估算，
四槽约67.3h；调整时剩余墙钟约77.1h。实际会复用已保存曝光。
这是调度估计，实际共享负载和检查耗时另记，当前无需再次延长截止。
本次约74.03GiB已超过64GiB目标，仍低于96GiB停止；不提高磁盘限额。

只读状态：`python scripts/review_hb_v12_six_card.py --policy four-card`。
单作业/停止：`python scripts/job.py status artifacts/jobs/v12-four-<配置>`、
`python scripts/job.py stop artifacts/jobs/v12-four-<配置>`，停止后核对后代退出。
训练日志`runs/v12-four-<配置>/train.log`，外层日志`artifacts/jobs/v12-four-<配置>/job.log`。
恢复始终用实际完整latest.pt、新job/output、同源准入和恢复证据，不重跑已提交批量入口。

Lens项目`tide-learning-lab`，新主段`tl-v12-four-main-*`，原研究组/seed保持；
prefix/confirm/diag同用`tl-v12-four-`。四条保留段仍为`tl-v12-repair-main-*`。
此前14个`v12-six-*`从未启动模型或产生训练曲线，其停止不计科学失败。
私有入口`.local/LENS.md`保持；新硬件采样只覆盖固定20个剩余服务，全部终态后结束。
提交/排队、实际恢复、学习结果分别报告，资源调整不升级质量结论。

18:02交接：16条已提交并确认waiting_resources，4条原服务继续；
新增编排冻结79259e55，实际身份/策略/父状态见[交接证据](../evidence/hb-v12-four-card-handoff.json)。
新CPU采样服务`v12-four-hardware`只观察固定20个剩余作业，不占用NPU训练槽。
