# v9配额告警误判与原批恢复

2026-11-12下午状态检查发现：原控制器于15:19:45北京时间因 `quota -w` 返回1失败，
随后停止四个主worker。原现场见[15:43只读记录](../evidence/direct-v9-status-20261112-pm.json)，
`runs/v9-campaign/state.json`和原job.log保持失败身份。不是模型非有限、分配错误或授权时限耗尽。

当时主58条中46完成、4被外部停止、8未启动；72筛选及19batch校准完成。
四条停止训练的原结果字段为 `budget_truncated`，其实际原因是上述控制器故障，
不能解释成已耗尽48小时或模型学习负面。全部旧服务及后代已经退出。

## 原因与有界修正

账号用量刚超过300GiB软配额，仍处于约6天宽限期；硬配额800GiB，余量约500GiB。
最后成功的原监测记录本批42.44GiB，低于64GiB界限，硬余量501.24GiB。
Linux quota在软额度告警时返回1；控制器用check_output把有效告警当成查询失败。

新监控用 `quota -w -p` 解码完整的用量、软/硬配额及epoch宽限期。
只在返回0/1且完整行可验证时继续；未知/过期宽限、不可解析输出、实际查询错误、文件硬配额
继续失败，原64GiB本批边界和200GiB硬余量保持。没有将软配额告警隐藏成“未超配额”。
四项定向回归覆盖软告警、低硬余量、过期/未知宽限/错误输出，以及不重跑完成项和必须有恢复点。

## 恢复范围与身份

这是已授权v9的故障恢复，截止仍为 **2026-11-13 21:36:26 +08:00**，不重置48小时预算。
新控制器使用独立固定源码；所有模型、worker、配方、配置字节、数据和安装继续使用原
`6e7bd49eb5e35c15445826cac3a9b6ddda75aff4` 与 `.local/source-v9-execution`。
控制器与训练源码分别记录，不把控制器修正变成模型更新。原失败文件不覆盖。

|恢复轨迹|原恢复点有效目标|动作|
|---|---:|---|
|r32-h3-q10-s11|12,480,574|原latest.pt含Adam/RNG/数据进度，续至16M|
|r32-h3-q10-s22|12,431,695|同上|
|r64-h2-bern-s11|15,509|同上|
|r64-h2-bern-s22|64,052|同上|

另有8条原表未启动项：TF-FC16/32各seed11、TF-FC64双seed、R32 quad入口树双seed、
R64 Bern入口树双seed。各自新进程准入/恢复检查后按原配置训练，不新增候选或种子。
12个独立单case worker可并行取得资源，减少原静态分片等待；仍最多12个NPU worker。

新controller记录：`runs/v9-recovery-20261112/state.json`，job为`v9-recovery-controller`。
新主运行：`runs/v9-recover-main-<id>`；Lens run ID `tl-v9-recover-main-<id>`，原study组不变。
控制器先合并46条原完成结果与12条恢复/首启结果，再按原有限条件决定延伸，最多六条中档256M
及六条同形状活动64M；延伸身份 `v9-recover-extend-*`。没有新增独立confirmation/test评价。
活动延伸判据保持原控制器的最后训练update比例，最终科学审阅另用完整validation活动及其分母，
不能把触发延伸视为独立验证或改变原目标带。

四个恢复点先在CPU检查文件hash、checkpoint类型、原源码身份、optimizer及RNG存在、已保存目标数。
实际恢复成功还需新NPU训练进度越过原恢复点；这仍不证明整条轨迹的独立精确重放。
严格数值失败、离散/机制、恢复、学习与性能继续分列。

## 操作

```sh
python scripts/job.py status artifacts/jobs/v9-recovery-controller
python scripts/review_direct_v9.py
```

原控制器state将继续显示failed，应同时查看新恢复controller，而非覆盖其历史终态。
停止全批仍为 `systemctl --user stop 'tide-lab-v9-*.service'`，随后核对后代清空。
活动训练和新controller均读取自己的固定快照，不编辑它们。
恢复的实际提交/运行状态以job记录与STATUS为准；本页的方案不是启动成功证明。
