# MoE补充批：查看、停止、恢复

在仓库根目录执行。固定预算在`configs/studies/language/moe-controls/budget.json`，
共同截止2026-11-17 09:05:08 +08；旧v10仍是原截止，以下命令只针对MoE。
至多8个MoE单卡worker，共享12slot；每服务4CPU线程/72GiB上限，64GiB新盘目标/96GiB停止，账号硬余量200GiB。

当前控制器`moe-controller-r2`的固定来源为`.local/source-moe-execution-v2`；完整commit/文件树身份见
`artifacts/jobs/moe-controller-r2/job.json`与`.tide-source.json`。
四个继承dense8Mi前缀及其32Mi续训使用`2f96d745`和`.local/source-moe-execution-v1`，新作业不改旧输入。
原`moe-controller`与12次零阈值准入失败保留，不再提交；四dense正常完成后才关闭该控制器。
配置表90个文件含30筛选配方与60个**条件候选**seed文件，实际只选至多12条正式轨迹，不表示全提交。

## 状态与日志

```bash
python scripts/review_moe_controls.py
python scripts/review_moe_controls.py --json
python scripts/job.py status artifacts/jobs/moe-controller-r2
python scripts/job.py logs artifacts/jobs/moe-controller-r2 --lines 30
```

每条ID和物理卡号在只读审阅输出中。单条服务例如：

```bash
python scripts/job.py status artifacts/jobs/moe-r2-screen8-small-e20-lr0.0003-b0.001-s33
python scripts/job.py logs artifacts/jobs/moe-r2-screen8-small-e20-lr0.0003-b0.001-s33 --lines 30
tail -n 30 runs/moe-r2-screen8-small-e20-lr0.0003-b0.001-s33/train.log
```

`check/`、`recovery.json`、`train/progress.json`和`train/result.json`分别是准入、恢复、进度与训练终态。
服务completed可能含内层失败，须用审阅器。筛选选择见控制器`selection8/selection32`，确认只在选择冻结后读取。
作业启动后不编辑其source/config/plan/父checkpoint。初始准入12次失败与后继尝试是不同运行身份。

## Lens

项目`tide-learning-lab`，沿用已有认证实例，私有地址/转发说明见`.local/LENS.md`。

|组|前缀|含义|
|---|---|---|
|TL-MOE-CAL|`tl-moe-batch-`、`tl-moe-sort2-`|完整步物理batch/排序校准|
|TL-MOE-CAL|`tl-moe-screen8-`、`tl-moe-r2-screen8-`、`tl-moe-r2-screen32-`|seed33配方筛选；旧前缀四dense继承|
|TL-MOE-CONTROL|`tl-moe-r2-main128-`|新seed11/22正式MoE及dense|
|TL-MOE-CONFIRM|`tl-moe-r2-confirm-`|独立262K确认|

运行尾部的`projection.json`含实际Lens显示名与run ID；运行中recording、结束后verified/readback_matches分开。
若small dense最终选择LR.001并满足精确配方匹配，可直接复用旧`tl-v10-main-tf-d192-l6-s11/s22`，
保持原身份，不计新增独立重复。配置ID、源数据、seed和目标token可直接关联研究图。
质量看task NLL，aux/balance、aux/z_loss、train/objective另看；专家选择率、各层负载/冷专家/熵、
物理expert行数、额外统计遍历、吞吐和显存另记。

## 停止

停止整批时先关闭控制器，避免再次提交；它会停止自己提交的活动子服务。

```bash
python scripts/job.py stop artifacts/jobs/moe-controller-r2
python scripts/review_moe_controls.py
```

对仍存活的MoE单条服务执行`python scripts/job.py stop artifacts/jobs/<JOB>`，核对MainPID=0、
control group及已记录后代均消失。继承的dense前缀已终态；原v10作业不在本控制器children列表。
不要使用按Python进程名批量kill，也不要停止旧v10控制器。

## 恢复

训练的`latest.pt`包含模型、Adam、RNG、有效token/数据游标与严格source/config/data/install身份。
正式复原必须使用原来源、原config、原环境、原admission/recovery文件与新作业/输出ID，
给`campaign_stage.py train`传入`--resume <old/train/latest.pt>`，其余如原worker的train argv。
实际完整argv、资源policy和父SHA均在对应worker plan及job.json，不猜测私有状态。
可从原`runs/moe-controller-r2/<phase>/<ID>-worker.json`复制一个新plan，只修改输出/作业ID，
填`reuse_admission`、`reuse_recovery`以及train_arguments内的`--resume`；再用同来源的`job.py submit`
与`campaign_worker.py --plan <NEW_PLAN>`提交，仍受同一截止/8worker/磁盘约束。
若控制器仍管理该案例，先协调其剩余提交，避免重复推进；控制器输出目录本身不能原地覆盖重启。

8/32Mi校准优化器在所有晋级完成后按哈希回执退休，只剩科学权重时不再承诺精确续训；
见`calibration-optimizer-retirement.json`。正式128Mi优化器保留。当前仅宣称三次B4/L32同栈新进程恢复，
不宣称未执行的长轨迹逐位恢复或跨源码精确恢复。
