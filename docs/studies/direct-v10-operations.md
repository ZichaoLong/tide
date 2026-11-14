# 2026-11-15修复续训的操作入口

新增三条pool修复，原20条和原失败不改。固定源码d089dd0556798df55a0440ef471ffe254f8054e5，
目录`.local/source-v10-repair-v1`；原安装/数学和共享截止2026-11-16 21:51:33 +08:00保持。
逐链身份、完整启动argv和只读输入SHA位于`artifacts/jobs/v10-repair-<ID>/job.json`。
ID为`hb-d128-r64-pool-s11`、`hb-d128-r64-pool-s22`、`hb-d128-r28-pool-s22`。
解析配置/worker plan/资源策略在`runs/v10-repair-control/<ID>/`，都为只读固定输入。

```bash
python scripts/review_v10_repair.py
python scripts/job.py status artifacts/jobs/v10-repair-hb-d128-r64-pool-s11
tail -n 30 runs/v10-repair-hb-d128-r64-pool-s11/train.log
```

正式日志仅在门槛通过后出现；此前查看同目录`check.log`、`migration.log`、`recovery-*.log`、`pilot.log`。
状态在`runs/v10-repair-<ID>-worker/state.json`，原20条仍用`python scripts/review_direct_v10.py`。
停止任一链：`python scripts/job.py stop artifacts/jobs/v10-repair-<ID>`，它核对完整cgroup和已知后代。
只停止该修复服务不会改动原控制器或其他作业。

恢复采用**新job/output**，环境和固定d089dd05源码不变。已过门槛时使用该链最新完整
`train/latest.pt`（正式尚未保存则`pilot/latest.pt`），`check/result.json`和
`recovery-compare/recovery.json`；复用同一解析配置，原128Mi目标和256Mi LR日程。
按现有`campaign_worker.py`单case plan设置`reuse_admission`、`reuse_recovery`，
`train_arguments`显式包含`--resume <该checkpoint>`、`--until-tokens 134217728`和新run ID，
再由固定源码的`job.py submit`提交；使用原resource.json、同一data-lock、4线程/72GiB/剩余共同截止，
并把全部上述文件登记为`--input`。原argv可从job.json取出，不能覆盖原output。
门槛未完成时不得绕过恢复/先导；截止后继续需要用户另行授权。临时恢复payload已通过比较后才退役，不可用作训练父。

Lens项目`tide-learning-lab`，组`TL-CONTROL-v10`，三条正式run为`tl-v10-repair-<ID>`；
先导`tl-v10-repair-pilot-<ID>`、独立确认`tl-v10-repair-confirm-<ID>`（后者TL-CONFIRMATION-v10）。
私有访问方式仍在`.local/LENS.md`。先导verified/readback_matches是长续训门槛，确认不参与训练决策。
局部参考、整网严格数值、离散机制、迁移/精确恢复、短程和长期学习分别记录。

以下为原v10操作说明。

# v10：查看、停止与恢复

2026-11-15三项pool故障诊断已终态，见[根因、候选和后续建议](direct-v10-gradient-results.md)。
只读`python scripts/review_v10_gradients.py`可查看实际回放/单步干预证据；原三个Lens run仍是failed。
诊断的`failure-checkpoint.pt`用于重现原失败；原源码恢复仍会遇到该错误，候选源码迁移和新进程恢复尚未验收。
没有自动恢复正式训练；活动作业仍读原固定源码。

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
