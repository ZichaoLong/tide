# v10 pool局部VJP修复与原对照补齐

2026-11-15用户批准执行[定位后的建议](direct-v10-gradient-results.md)。本表授权来源为该次“同意，请执行”；
实施新源码、回归与新进程恢复检查，64个完整更新的稳定性先导，通过后直接补齐三个pool到原128M目标。
不增加模型、数据池、seed或曝光档；不改活动源码，不覆盖原failed，不改变原控制器的256M成组延伸决定。
固定数学0770593、环境/安装74e7bbd不变；仍为独立直接实现，不接入基座新优化。

|配置|参数|接入检查|恢复/先导|正式目标|终点评价|
|---|---:|---|---|---|---|
|d128 R64 pool seed11|171,459,232|真实算子见证CPU FP64/NPU FP32，整网小型因果/别名/VJP|训练权重上的完整global64两步，连续与新进程1+1精确比较；64步先导|原累计128Mi有效目标|131K validation与独立262K confirmation|
|d128 R64 pool seed22|171,459,232|同上|同上|同上|同上|
|d128 R28 pool seed22|75,541,048|同上|同上|同上|同上|

三条独立单卡链并行。沿用实测选定的L256/global64，R64 physical B8、R28 B16；
梯度累积与原LR/AdamW/256M衰减日程不变。64步先导停在完整update边界，不增加token截断点；
其真实更新计入后续累计曝光，恢复检查的重复计算不计入学习轨迹。

实现选项仅增加`implementation.aggregate_vjp="ordered-v1"`。旧配置默认native，原冻结0ea05d61不改。
迁移从诊断回放得到的失败前完整快照开始，避免重复数百步定位；父快照的source/config/data/install/设备身份严格校验，
model/optimizer/RNG/cursor数值保持，源码身份显式改变。记录原run、原错误、诊断父SHA与新identity；
61个旧共同记录点一致不升级为不可取得的旧中间权重逐字节一致。
新源码恢复检查比较model、Adam、progress、最后梯度hash，以及CPU/NPU/Python RNG。
只在比较通过后退役本链自己产生的三份临时恢复payload，保留hash/比较结果/退役收据；原科学和失败checkpoint不删。

上限仍为原48h截止2026-11-16 21:51:33 +08:00、全局12个NPU worker、每服务4CPU线程/72GiB主机内存；
两次健康/负载/HBM检查与同一12slot/设备锁，全部background.slice。无累计CPU/NPU算时上限。
本修复链新增盘限24GiB，整个v10仍64GiB目标/96GiB停止线、账号800GiB硬配额至少200GiB余量。
算子检查≤300s、整网准入≤600s、迁移≤600s、每恢复段/比较≤300s、先导≤900s；
训练和确认服从共同剩余截止。非有限、非法支持/可行性、身份或精确恢复失败立即阻止该链后继。
有限严格超差沿原阈值记录，符合已有机制检查后才进入有限学习探索；不要求全部训练CPU参考。
先导只验收执行/状态/有限性和训练信号，不要求短程loss必降，也不把它当长期稳定或质量证据。

Lens项目仍`tide-learning-lab`，先导与正式续训同属`TL-CONTROL-v10`，确认属`TL-CONFIRMATION-v10`。
名称分别`tl-v10-repair-pilot-<原ID>`、`tl-v10-repair-<原ID>`、`tl-v10-repair-confirm-<原ID>`。
长续训前要求先导投影verified/readback_matches；公共仓库不写私有实例信息。
原`tl-v10-main-...`维持failed，不与修复链合并成一次无中断原实现训练。

实际配置由`configs/studies/language/direct-v10/repair.json`有限manifest和原固定配置解析，仅变更上述一个选项；解析结果写入新的只读输入目录并登记SHA。
作业`v10-repair-<原ID>`，输出`runs/v10-repair-<原ID>/`，worker状态`runs/v10-repair-<原ID>-worker/state.json`。
`python scripts/job.py status artifacts/jobs/v10-repair-<原ID>`查看生命周期，
`tail -n 30 runs/v10-repair-<原ID>/train.log`查看正式学习；阶段日志与结果在同级目录。
停止用`systemctl --user stop tide-lab-v10-repair-<原ID>.service`，再核对所有后代。
恢复仅使用该新固定源码和相同解析配置，另建身份/输出，从本链最新完整checkpoint接续；
若资格未完成，从对应未通过门槛继续，不能绕过检查。新源码SHA及实际状态在STATUS补充。

只读汇总：`python scripts/review_v10_repair.py`。原20条状态仍用`python scripts/review_direct_v10.py`，不把修复链计作新seed。
