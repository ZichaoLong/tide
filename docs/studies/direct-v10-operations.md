# v10：查看、停止与恢复

本轮研究表见[固定配置/预算](language-capacity-activity-v10.md)，
[校准和参数清单](../evidence/direct-v10-selection.json)。独立直接实现，不接入新基座。
主配置目录`configs/studies/language/direct-v10/table/main/`，固定执行源码目录`.local/source-v10-execution`；
实际commit **0ea05d610df82df5b55663cbcd2fa4bcc92c3a6d**，tree SHA
`b42a2533e88677e6c7e1fa298d2ba94b122bcbb8f873b851cc5e8ba517eef798`，
见该目录`.tide-source.json`及`artifacts/jobs/v10-controller/job.json`。
本轮私有环境`.venv-positive-floor-v1`保持原安装74e7bbd，数学0770593；运行identity记录实际Torch/TorchNPU/CANN来源。

共同截止**2026-11-16 21:51:33北京时间**；最多12个单卡worker，每worker至多4CPU线程/72GiB主机内存。
根据两次健康/低负载/显存观测共享选卡，持有设备及并行slot锁，全部`background.slice`。
每case所需空闲HBM按实测peak×1.15+8192MiB固定；最长任务先请求，短任务结束立即释放资源，
没有固定分片尾部空等。无累计CPU/NPU算时上限；64GiB新增目标、96GiB停止上限、账号至少200GiB硬余量。

## 查看

在本仓库目录执行，不会启动训练：

```sh
python scripts/review_direct_v10.py
python scripts/job.py status artifacts/jobs/v10-controller
tail -n 30 artifacts/jobs/v10-controller/job.log
```

控制器状态`runs/v10-controller/state.json`列出所有已接收服务、阶段、实际设备及成组延伸决策。
每个配置ID（例如`hb-d128-r64-q3-s11`）对应：

- 服务`tide-lab-v10-main-<ID>.service`，生命周期`artifacts/jobs/v10-main-<ID>/`。
- 数据/梯度准入`runs/v10-main-<ID>/check/`，恢复`recovery.json`，
  曲线/检查点`train/`，文本日志`train.log`；worker状态为`runs/v10-main-<ID>-worker/state.json`。
- 条件续训使用`v10-extend-<ID>`，128M父checkpoint保留，记录SHA及恢复谱系。
- 独立确认使用`v10-confirm-main-<ID>`与`v10-confirm-extend-<ID>`；不是新训练seed。

```sh
python scripts/job.py status artifacts/jobs/v10-main-hb-d128-r64-q3-s11
tail -n 30 runs/v10-main-hb-d128-r64-q3-s11/train.log
python scripts/observe_v9_resources.py --campaign v10 --samples 8 --interval 4 --output docs/evidence/direct-v10-resources.json
```

最后一条只采样已有服务及设备，记录AICore/AIVector/AICPU、HBM带宽和主机CPU；
设备共享计数不能归因到单进程，不能相加成忙碌比例。trace与无插桩完整步计时分开。

## Lens

沿用已安装实例及`tide-learning-lab`项目，查看入口在本机`.local/LENS.md`。
不部署或改变访问策略。项目文件为权威，Lens只是查看投影。

|组名|内容|正式run名称|
|---|---|---|
|TL-CAPACITY-v10|R28约10%、R64约3%；d64/128×双seed共8条|tl-v10-main-hb-d…-q10/q3-s…|
|TL-ACTIVE-v10|3.756M/9.668M小稠密，作为近活跃预算的最低总容量端点，共4条|tl-v10-main-hb-d…-dense-match-s…|
|TL-CONTROL-v10|d128同形状pool4条、调参TF4条|tl-v10-main-hb-…-pool-s… / tl-v10-main-tf-…|
|TL-CONFIRMATION-v10|全部合格128M及实际256M终点的262K独立评价|tl-v10-confirm-main/extend-…|
|TL-CALIBRATION-v10|已完成的10项HB活动、12项TF配方先导|tl-v10-cal-…|
|TL-PERF-v10|融合梯度探针、完整步/microbatch与控制峰值|tl-v10-fusion-probe / tl-v10-benchmark-… / tl-v10-cal-controls|

容量组的8条同时参与活跃预算比较，不重复训练；延伸把`main`替换为`extend`，组名不变。
只读投影核对见各`projection.json`的`status`和`readback_matches`。

## 停止与恢复

先停控制器，阻止后续提交，再停止本批worker与评价服务，并核对后代：

```sh
systemctl --user stop tide-lab-v10-controller.service
systemctl --user stop 'tide-lab-v10-main-*.service' 'tide-lab-v10-extend-*.service' 'tide-lab-v10-confirm-*.service'
systemctl --user stop tide-lab-v10-admission-aligned.service
python scripts/review_direct_v10.py
```

每个`job.py status`应确认`termination_verified_now`且无存活后代；只有包装器退出不算停止完成。
不要改冻结源码、注册输入、原记录、锁文件或活动checkpoint。

恢复需要新作业名和新输出目录，复用原固定源码/配置/环境/数据、原准入与恢复凭据；
`campaign_stage.py train --resume <原train/latest.pt>`恢复model/optimizer/RNG/cursor/活动控制器。
只有`latest.pt`带optimizer；8/32/128M的`tokens-*.pt`主要为权重，不承诺可训练恢复。
先核对原服务终态、checkpoint身份及原共同截止仍有余量，再将带以下参数的一项worker plan交给原`job.py submit`：

```text
reuse_admission = 原主轨迹/check/result.json
reuse_recovery  = 原主轨迹/recovery.json
train_arguments = [--resume, 原轨迹/train/latest.pt,
                   --until-tokens, 原阶段终点,
                   --run-id, 新tl-v10-resume-身份,
                   --recording-group, 原研究组]
```

当前控制器不自动恢复失败任务，不悄悄延长48h或覆盖原失败。后续唤醒本会话可按上述条件建立新身份，
恢复记录必须补入交接；不得把权重可加载称为长轨迹精确重放。小型三更新新进程检查与正式实际续训分别记录。

## 单独的检查口径修正

`v10-admission-aligned`使用固定6125c751，对四个quad主形状补做同warmup设置的CPU/NPU检查。
它遵守同一12slot锁，目前可排队，最晚11月15日01:51:33结束，不读取或修改任何活动权重。
0ea05d61原准入CPU参照未同步`set_progress`，其大幅差异属于不同活动设置的无效等价比较，
原failed保留，不解释成普通浮点误差；新检查的有限超差仍使用原阈值。具体见[补证说明](direct-v10-optimization.md)。
本项只更新检查资格，不是新训练候选或正式轨迹的源码替换。
