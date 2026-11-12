# direct-v7：查看、停止与恢复

2026-11-12：十条主轨迹的4M终点保留，其余筛选及中间权重已按授权退役。
部分原续训父checkpoint现在只保留谱系/hash和观测；恢复前查[可用性清单](checkpoint-retirement-20261112.md)，不直接重跑原父checkpoint命令。

项目记录为权威；当前快照见[STATUS](../STATUS.md)，研究范围见[固定表](direct-language-v7.md)。
训练源码固定`d30559c8bce74d5b6ea392bb18f53e8ba3b90cf1`，只读目录`.local/source-v7-main`。
校准源码3455c7a，解析计数首次bc58bef、修正b463f80，全部另有冻结目录；未改活动训练代码。
数学0770593、安装74e7bbd、`.venv-positive-floor-v1`保持；实际安装文件核对在每个阶段identity中。

## 查看

从仓库根目录运行：

```sh
python scripts/review_direct_v7.py
python scripts/job.py status artifacts/jobs/v7-controller
python scripts/job.py status artifacts/jobs/v7-main-w0
python scripts/job.py logs artifacts/jobs/v7-main-w0 --lines 40
```

`review_direct_v7.py`只读runs/jobs，显示训练、严格超差、恢复和Lens各轴；
只有显式`--output docs/evidence/direct-v7-observations.json`才原子更新审阅快照。
控制选择、每阶段所有case、具体配置及输出在`runs/v7-campaign/state.json`。
服务ID：筛选`v7-screen-w0..3`，主训练`v7-main-w0..3`，监督`v7-budget-v2`；
条件满足时才有`v7-residual-w0..1`。`v7-budget`是已退役的校准期监督。
解析参照`v7-count-controls`保留原seed22身份错误，使用`v7-count-controls-r2`的修正结果。

每case的`train.log`在`runs/v7-<phase>-<config-id>/`，`train/`内有progress/result、
逐块validation、inference、metrics.jsonl、checkpoint、projection.json。
不能只看外层服务退出码；真正完成须达到目标token、完整验证、检查/恢复记录，并核对无存活后代。

Lens复用已有私有实例，访问/SSH方法在本机`.local/LENS.md`；服务配置未改变。
在项目`tide-learning-lab`选择：

| group | run ID | 含义 |
| --- | --- | --- |
| TL-DIRECT-RECIPE-v7 | `tl-v7-cal-cpu/npu` | 短吞吐校准，无质量结论 |
| TL-DIRECT-RECIPE-v7 | `tl-v7-screen-<config-id>` | 12格262K配方筛选 |
| TL-DIRECT-SCALE-v7 | `tl-v7-main-<config-id>` | 5模型×seed11/22，至4M；部分seed11承接screen |
| TL-DIRECT-SCALE-v7 | `tl-v7-count-controls-r2-s11/s22` | 同4M目标解析词频/二元参照 |
| TL-DIRECT-STRUCTURE-v7 | `tl-v7-residual-<config-id>` | 仅条件触发才存在，不存在不能当完成 |

`<config-id>`如`hb88-fan_in-raw-s22`、`tf28-balanced-s11`、`fc16-fan_in-raw-s11`。
完整逐运行映射在[审阅快照](../evidence/direct-v7-observations.json)，不在公共仓库写私有实例URL。

## 停止

本批仍活动时可统一停止监督，让它停止所有本批活动服务：

```sh
python scripts/job.py stop artifacts/jobs/v7-budget-v2
python scripts/review_direct_v7.py
```

确认`artifacts/jobs/v7-*/job.json`各自状态中的MainPID、完整cgroup与记录后代均已清空。
单独停一个worker用`python scripts/job.py stop artifacts/jobs/v7-main-w0`。
控制器和worker是独立服务；只停controller不代表worker已停。
终态后无须再次stop；不要按通用python进程名停止他人任务。

## 恢复与复现

每条轨迹保存262K、1M、4M里程碑，以及latest/previous。精确续训只接受相同
config SHA、训练源码tree/commit、数据lock、安装、device/threads/dtype与完整optimizer/RNG/cursor身份。
已通过的资格记录与父checkpoint均以实际hash记录；三步恢复通过不认证整条长轨迹独立重放。

需要恢复时，依据`runs/v7-campaign/v7-<phase>-wN-plan.json`复制一个**新**worker plan，
只选待恢复case，设置新的case/worker/job输出，保留原config；复用该case的admission/recovery，
`train_arguments`使用`--resume <旧train/latest.pt>`、新`--run-id`，目标仍为原4M。
用冻结源码的`job.py submit`与原job.json相同env、共享资源policy、data-lock、deadline及输入hash方式提交。
将父checkpoint及资格记录加入`--input`。只在本批剩余预算内恢复，不重置预算或自动变成8M/新候选。
本批终态后新训练须先对齐下一批；复看报告无需计算。
