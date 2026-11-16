# v12六卡调度与延长截止

2026-11-16用户要求当前任务缩减至6张卡以内，并允许延长执行时间。
这是原38链的资源调整，不增加候选、种子、模型参数、训练曝光或第二次数值修复。
配置表：[固定14条行政续段](../../configs/studies/language/hb-v12/six-card.json)。

| 项目 | 调整 |
| --- | --- |
| 物理卡集合 | 固定4、6、9、11、12、14；六槽且全寿命物理锁 |
| 本项目退出的卡 | 0、2、5、7、10、15；不表示其他用户也已退出 |
| 截止 | 原11月17日23:08:07+08延至11月19日23:08:07+08，增加48h |
| 继续原服务 | base128-q3-s22、residual-q3-s22、share8-d128-q3双seed、share8-lr3-q3双seed |
| 新服务 | 6条更新边界保存后的续训，加8条原尚未分配设备的修复重排 |
| 不变 | 38链、128Mi终点、batch/LR/数据/阈值、确认与诊断；无累计计算时上限 |
| 磁盘 | 64GiB目标、96GiB停止、800GiB硬配额至少200GiB余量；调整时72.82GiB、351.89GiB余量 |

先停止旧提交控制器及8个等待器，再只向6个训练进程发送SIGTERM；
训练在完整更新边界保存model/Adam/RNG/游标/控制器状态，未强杀模型进程。
六个`latest.pt`已完整解码，progress与终态记录一致，SHA与checkpoint记录一致，
实际可恢复曝光20.851–57.462Mi。已核对对应服务无存活后代。
[迁移审阅](../evidence/hb-v12-six-card-transition.json)保留进程身份、原始状态及检查点。

原worker只接受完整终点，因而这六个服务外层显示failed、训练段显示budget_truncated/stop_signal15。
保留这些原始记录，并在当前审阅中明确标记行政暂停；它们不是新增可行性失败。
原38条科学失败及数值阈值从未改写。

科学源码仍为`62e32602`及`.local/source-hb-v12-repair-v2`，模型子进程cwd、
PYTHONPATH、全部阶段脚本均取该冻结目录。新增调度脚本作为独立冻结且校验SHA的输入；
没有模型源码迁移，也不修改六条保留作业的输入。单改槽数无法约束持有旧高编号槽的worker，
因此同时固定上述六张物理卡，并复用`tide-direct-v10`槽及设备锁。

六条续训先在实际新卡上检查两步连续对1+1新进程恢复，通过后从保存的科学父状态继续原128Mi，
再完成原确认和诊断；恢复探针不作为训练父轨迹。八个原未启动修复沿原迁移/恢复/前缀流程执行。
每条新服务只提交一次；数值、恢复或资源失败均停止该链，不自动重试。
临时恢复checkpoint仅在比较成功后按SHA回执退役；行政暂停父及原科学失败父保留。

任务使用`v12-six-<配置>`，Lens主段为`tl-v12-six-main-<配置>`，原study组与seed不变；
前缀、确认和诊断分别为`tl-v12-six-prefix-*`、`tl-v12-six-confirm-*`、`tl-v12-six-diag-*`。
六条保留作业仍使用`v12-repair-*`/`tl-v12-repair-*`，旧曲线不覆盖。
项目仍为`tide-learning-lab`，私有入口仍在本机`.local/LENS.md`。

只读状态：`python scripts/review_hb_v12_six_card.py`；
单作业：`python scripts/job.py status artifacts/jobs/v12-six-<配置>`；
日志：`artifacts/jobs/v12-six-<配置>/job.log`、`runs/v12-six-<配置>/train.log`。
停止：`python scripts/job.py stop artifacts/jobs/<JOB>`并核对后代退出。
恢复一律新job/output、实际完整latest.pt和同源准入/恢复记录，不能重跑已提交的批量脚本。
新的硬件采样覆盖固定20个剩余服务，全部终态后停止；记录AICore/AIVector/AICPU及HBM，
保持共享占用和低频采样的解释限制。

资源约束检查通过不等于NPU续训已开始；实际提交/排队见交接证据和实时状态。
CPU/NPU局部分配修复资格已通过、Lens verified；原完整模型有限数值超差仍单列，
目前8条修复链已完成，语言质量及干预收益尚未在本资源调整轮审阅。
