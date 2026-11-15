# v11执行与查看

研究表：[capacity-utilization-v11](capacity-utilization-v11.md)；
已完成诊断与唯一HB干预：[选择依据](capacity-v11-diagnosis-and-intervention.md)。
本轮从2026-11-15 17:43:08至**11月17日17:43:08（北京时间）**，共享48h。
最多12个NPU worker共用`tide-direct-v10`槽位，64GiB新增目标/96GiB停止，硬配额余量>=200GiB。
其他账户/公共环境/基座未改。研究结果需后续人审，不由控制器自动生成科学支持。

|工作|固定源码|作业与Lens对应|范围|
|---|---|---|---|
|首轮MoE控制器|ecb1b1f2，`.local/source-capacity-v11-moe-control`|`v11-moe-controller`|运行两条small E64 balance.01；六条初次延伸未产生新更新，保持原记录|
|真实MoE延伸|e3b2ab38，`.local/source-capacity-v11-extensions`|`v11-extend-r2-large-{dense,e20,e64}-…-s{11,22}`|原128→256Mi；端点配置显式迁移、训练实现逐文件同一性、完整batch新进程恢复|
|两条平衡系数控制|8bd33dd1，`.local/source-moe-execution-v2`|`v11-moe-balance-small-e64-lr0.003-b0.01-s{11,22}`|独立从零128Mi，对照原balance.001及E20/.01|
|16项诊断/新数据|36dec66a，`.local/source-capacity-v11-diagnostics`|`v11-diag-*`，数据重提`v11-confirmation-data-r2`|诊断全部完成；数据首次因当前训练环境没有tokenizers启动失败，使用原私有数据环境成功；不改变任何环境|
|四条HB干预|159190c3，`.local/source-capacity-v11-hb`|`v11-hb-hb-d64-r64-q3-bias{0.001,0.01}-s{11,22}`|原50.324M、约3%目标，8Mi先导→受训恢复→128Mi，单一负载偏置家族|
|新确认控制器|159190c3，同上|`v11-fresh-controller`；子任务`v11-fresh-*`|固定28个旧父/控制/新终点，最多4个评价并发，占同一12槽；未达曝光或失败端点记ineligible|

每项准确源码树SHA、实际依赖、配置SHA、物理卡和cgroup见其`artifacts/jobs/<JOB>/job.json`、
`queue-state.json`、`runs/<JOB>-worker/state.json`。安装仍为私有`.venv-positive-floor-v1`，
Torch2.12.0+cpu/TorchNPU2.12.0.post3/CANN9.2；基座74e7bbd、数学0770593。
数据准备使用原`.venv`中的tokenizers0.22.2；它不承担本轮模型训练。
训练数据`runs/v9-data-r3`原锁40b7c0eb…未改；新确认锁`d9be5ec8…`在`runs/v11-confirmation-data`。

只读查看（会显示运行/排队、当前阶段、实际曝光、设备与失败，默认不写任何文件）：

```sh
python scripts/review_capacity_v11.py
python scripts/job.py status artifacts/jobs/<JOB>
python scripts/job.py logs artifacts/jobs/<JOB> --lines 80
```

worker的子阶段stdout在`runs/<JOB>/<STAGE>.log`，训练指标/进度在
`runs/<JOB>/{pilot,train}/metrics.jsonl`、`progress.json`；终点分别查看`result.json`和`validation-*.json`。
恢复验收在`recovery-compare/recovery.json`（MoE）或`trained-compare/recovery.json`（HB）。
强制数值准入在`check/checks.json`；有限探索不能覆盖这些原阈值失败。
HB先导Gate及峰值/吞吐记录在`pilot-gate.json`。失败链不自动换配方或放宽阈值。

停止单项（会向作业发终止信号，允许训练保存，随后验证无存活后代）：

```sh
python scripts/job.py stop artifacts/jobs/<JOB>
```

停止整批时，先停止`v11-fresh-controller`、`v11-moe-controller`，再停止只读状态表中仍活动的
本批12条worker及已提交的`v11-fresh-*`评价服务。控制器与独立子服务是不同systemd单元；
不能只停止控制器就假设所有训练也停止。不要终止其他批/账户的服务。

恢复必须使用**新JOB、新输出目录及原固定源码**，不得覆盖原尝试；先确认共享截止、
磁盘余量、checkpoint identity、config/data/source/install与检查/恢复身份一致。
可复制对应control目录的worker计划到新的私有目录，保留原config，并设置
`reuse_admission`指向原`check/result.json`、`reuse_recovery`指向该配置已通过的恢复记录，
`train_arguments`中加入`--resume <latest.pt>`及原有限`--until-tokens`、新的`--run-id`。
使用同一固定源码的`scripts/campaign_worker.py`经`scripts/job.py submit`提交，
保留resource policy/queue helper、background.slice、截止及资源约束；失败检查不得绕过。
HB正式续训父为通过Gate的`pilot/latest.pt`，验证探针不是训练父。
本轮截止后不自动恢复，不增加原目标曝光。

Lens沿用既有安装/账号和访问策略，私有入口及SSH方法仅在`.local/LENS.md`。
项目`tide-learning-lab`，精确显示名以`projection.json`的destination_name为准。
统一运行ID通常是`tl-<JOB>`；HB先导是`tl-v11-hb-pilot-<CONFIG_ID>`，正式是`tl-v11-hb-<CONFIG_ID>`。
分组：`TL-CAPACITY-DIAG-v11`、`TL-CAPACITY-EXPOSURE-v11`、`TL-CAPACITY-BALANCE-v11`、
`TL-CAPACITY-HB-v11`、`TL-CAPACITY-CONFIRM-v11`。fresh与旧seen确认由ID明确区别。
小型录制已verified/readback_matches；运行中投影状态recording不是独立学习验收。
