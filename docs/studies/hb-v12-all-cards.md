# v12剩余任务放开全部可用卡

2026-11-16 22:06用户允许有剩余任务时使用剩下的所有卡。本轮仅扩并行度，
沿用原38链、128Mi终点、科学源码62e32602、配置/种子、数据与原检查阈值。
[固定12条重新调度清单](../../configs/studies/language/hb-v12/all-cards.json) ·
[启动前审计](../evidence/hb-v12-all-cards-transition.json) · [检查点占用规划](../evidence/hb-v12-all-cards-storage.json)。

审计时已有22条完成原终点流程，另4条实际训练、12条未分配设备。
四条在4/9/12/14上的`v12-four-*`训练及其固定输入保持；只停止12个未启动等待器，
从同一具名科学父状态重排为`v12-all-*`。已完成结果及旧数值失败不重跑。

| 项目 | 安排 |
| --- | --- |
| 物理卡候选 | 全部0–15，不预先固定一个小子集 |
| 并发上限 | 同一`tide-direct-v10`命名空间16槽，物理卡锁全寿命持有 |
| 共享准入 | 原两次健康/AICore/HBM采样；沿用各形状校准余量，不因其他项目在卡上就一概拒绝 |
| 当前忙卡 | 1/13/15剩余HBM不足；3有其他进程、仅部分形状余量合格。实际分配由队列决定 |
| 恢复临时状态 | 仅在4条旧冻结作业存活时错峰；这些旧作业终态后自动取消临时限制 |
| 截止/磁盘 | 截止保持11月19日23:08:07+08；用户已放开项目上限与固定余量，仅账号总硬限800GiB |

本机约1.7TiB可用主机内存、320逻辑CPU。审计时v12约77.03GiB、账号硬余量347.50GiB。
12份实际父状态均有optimizer且SHA匹配，曝光32–77.14Mi；无需从零重新训练。
其新增最终完整状态约6.21GiB、剩余128Mi里程碑模型张量约2.08GiB；
全部同时生成三份恢复状态约18.62GiB，会挤占96GiB界内的写入余量。
后续用户已明确允许磁盘增长，只要求账号总占用不超过800GiB。
新worker取消项目96GiB停止和固定200GiB余量，按实测父状态的四倍估算该链写入所需空间；
提交时按剩余12链完整状态/权重/临时写入估计约34GiB检查账号空间，不能把它误作新项目上限。
原4条活动worker的冻结输入不能修改，它们仍带旧96GiB检查。因此只在这些旧worker存活时，
以两个恢复名额（约5.36GiB三份探针峰值）和90GiB临时写入投影保护旧执行；
旧worker全部确认终止后自动取消这层兼容限制。正式训练及确认/诊断不持有恢复名额。
此过程不改变模型或batch。不同并行度下吞吐含共享负载，不声称独占加速。

新增编排作为单独冻结输入，所有科学脚本仍从`.local/source-hb-v12-repair-v2`执行。
每条沿用原准入、恢复、先导（适用时）、128Mi、独立确认及诊断；新失败不自动重试。
临时检查点仍须比较通过后按SHA回执退役，未改变原失败父和行政父的保留策略。

只读：`python scripts/review_hb_v12_six_card.py --policy all-cards`。
`preparation_admission`明确区分等待临时磁盘/名额、已取得名额、已释放；
服务running或取得NPU不等于已经进入主训练。
单任务状态/停止：`python scripts/job.py status artifacts/jobs/v12-all-<配置>`、
`python scripts/job.py stop artifacts/jobs/v12-all-<配置>`；停止后核对后代退出。
阶段日志`runs/v12-all-<配置>/train.log`，外层`artifacts/jobs/v12-all-<配置>/job.log`。
恢复始终新job/output、同源完整latest.pt与准入/恢复证据，不重跑批量入口。

Lens项目`tide-learning-lab`，新段`tl-v12-all-main-*`，prefix/confirm/diag同前缀；
保留四条仍为`tl-v12-four-main-*`。原study/seed保持，私有入口`.local/LENS.md`不变。
硬件采样`v12-all-hardware`覆盖固定16个未完成服务，记录AICore/AIVector/AICPU及HBM。
放开设备和作业完成均不自动升级科学结论；完整配对质量等待审阅。
