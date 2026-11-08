# 后台作业、终态与恢复

2026-11-08终态复核：五个作业均已结束、无存活后代。FC评估与两个HB400步训练completed/exit0；NPU采样与冻结骨干B等卡超时，workload未启动。当前没有活动或排队作业。下面保留固定输入、预算和历史操作方法，不能把命令示例当作已提交的新任务。[第二批结论](studies/second-batch-results.md)、[终态证据](evidence/second-batch-terminal.json)。

固定源码：`94bf712f8bd2e4007f0f56d1fea4799c99833dcd`，只读目录`.local/source-second-batch`，树SHA256 `e4fb0dfa6fc74c2c9c8b6a30d43215532321d4e2621d45f6abb0d6e3012133a0`。数学/公共基座及`.venv`不变；旧consumer wheel原样保留，新实验包显式从冻结目录导入，实际身份见各manifest。不要修改快照、环境或被锁定的原FC checkpoint/config。工作区文档更新不影响这些作业。

| 作业ID | 内容/固定配置（快照内） | 预算与停止 |
| --- | --- | --- |
| s2-fc-mechanisms | 三个旧step400 checkpoint；`configs/studies/second/fc-mechanisms.json` | CPU1线程/4GiB；服务19.5min+最多30s清理；无训练 |
| s2-hb-dense | seed11/B32/400步；`configs/studies/second/hb-dense-matrix.json`及`hb-dense.json` | CPU1线程/8GiB；服务119.5min+最多30s清理；训练110min |
| s2-hb-sparse | 与dense配对；`configs/studies/second/hb-sparse-matrix.json`及`hb-sparse.json` | 同上；两组并行，总CPU2线程/16GiB |
| s2-npu-sampling | 原FC/B32；`configs/studies/second/npu-sampling.json` | 单NPU/CPU4线程/8GiB；等卡≤5min，命令≤9min；服务14.5min+最多30s清理 |
| s2-frozen-growth | 冻结准入→完整恢复→三候选各40步；`configs/studies/second/b-stage.json`及引用配置 | 单NPU/CPU4线程/24GiB；等卡≤10min，阶段≤45min；服务59.5min+最多30s清理 |

CPU/内存控制器未逐服务委派，表中为请求预算，不能称硬隔离。应用线程和服务超时生效。NPU只由共享队列选择真实空闲卡，随后逻辑`npu:0`；没有共享其他作业占用的卡，没有自动重试。峰值请求11 CPU线程/52GiB/最多2张独立NPU，新增磁盘≤20GiB。

在项目根目录执行，其他任务替换ID：

```sh
python scripts/job.py status artifacts/jobs/s2-hb-dense
python scripts/job.py logs artifacts/jobs/s2-hb-dense --lines 60
python scripts/job.py stop artifacts/jobs/s2-hb-dense
```

stop停止整个服务、队列等待和所有后代，并核对终止。单元名为`tide-lab-ID.service`；实际启动命令、输入hash、资源、来源、cgroup和终态保存在`artifacts/jobs/ID/{job,state,submission}.json`，无需重启计算查看。

权威结果位置：

- FC：`runs/s2-fc-mechanisms/result.json`，含配对逐例预测及输入hash；`metrics.jsonl`提供各seed/条件的指标。
- HB：`runs/s2-hb-{dense,sparse}/tl-a2-long-{dense,sparse}-s11/`中的state/result/checkpoint/metrics。日志步为1、5、10…400，工作总量每步累计；阶段batch.json记录内层退出。
- NPU采样：`runs/s2-npu-sampling/result.json`。若还在队列，目录尚未生成，以`artifacts/jobs/s2-npu-sampling/queue-state.json`为准。
- B：`runs/s2-frozen-growth/stage.json`及ingress/resume/learning记录；学习run名为`tl-b2-{none,tide,mlp}-s11-frozen`。队列中未生成stage不是失败。

Lens沿既有项目`tide-learning-lab`；小记录、FC评估及两个HB训练均已读回verified。采样和冻结B因未取得设备没有模型run。训练通过独立记录器跟随JSONL；FC评估在完成后投影。各目录projection.json给出具体destination_name/run_id，私有入口及SSH转发见`.local/LENS.md`。记录器失败与执行、学习分开，不更改既有服务。

恢复仅针对已发布checkpoint的中断训练，须先确认旧服务与后代终止。HB每40步发布，检查点包含model/AdamW/RNG/数据游标。保持原配置和源码，使用新作业ID与输出，例如：

```sh
python scripts/job.py submit \
  --name s2-hb-dense-resume-01 --source .local/source-second-batch \
  --output artifacts/jobs/s2-hb-dense-resume-01 \
  --env-script "$PWD/.local/source-second-batch/scripts/env.sh" \
  --env TIDE_LAB_VENV="$PWD/.venv" --device cpu \
  --timeout 7170 --wait-seconds 0 --threads 1 --memory-gib 8 \
  --input runs/s2-hb-dense/tl-a2-long-dense-s11/checkpoint.pt \
  -- python -m tide_lab.train \
  --config "$PWD/.local/source-second-batch/configs/studies/second/hb-dense.json" \
  --resume "$PWD/runs/s2-hb-dense/tl-a2-long-dense-s11/checkpoint.pt" \
  --output "$PWD/runs/s2-hb-dense-resume-01"
```

这是恢复方法，不是已额外授权的一份完整训练预算。已用时间计入本批，恢复前确定剩余上限；没有checkpoint不能声称续训。直接train入口写权威JSONL，新的Lens投影须用独立record命令并保持同一项目。B成功后按计划退役大checkpoint保留hash；中断保留。NPU精确恢复要求同一实际映射及完整identity，跨映射迁移未认证。FC评估/采样是有限一次任务，无optimizer恢复，不自动重试。

# 首批后台作业与恢复

2026-11-08终态复核：表中全部训练已完成，当前无活动或排队作业；以下保留原固定输入、预算与历史操作方法。新的训练/恢复须另用新ID，不能把本页示例当作正在执行的任务。结论见[首批总报告](studies/first-batch-results.md)。

项目根目录执行下列命令。旧CPU确认作业固定`.local/source-first-a`中的`0d7ec5bdaa34556df9c93886e95f55fa5c9876ab`；新增NPU阶段固定`.local/source-a-npu`中的`0494d208e2a561ac66da9d1fa3d72672af6809e8`（树SHA256 `adc5f9eabc01e106b36055b017f3d1391f19d4a1079ee9b213cb3420c24a34fe`）。不要改这些快照、`.venv`、`.local/configs-*`或NPU输出内的resolved配置；普通工作区修改用于下一快照。模型/任务/训练包源码与原固定wheel相同，新阶段仅增加驱动、检查、配置与记录。安装身份见dependencies，站点私有查看说明见`.local/LENS.md`。

| 作业ID | 实际内容 | 固定配置 | 时间上限与资源预算 |
| --- | --- | --- | --- |
| confirm-a-s11 | query-only + FC/KV，seed11 | configs/studies/confirm-a-s11.json | 各400步/B32，CPU1线程/8GiB，组内75min，服务80min |
| confirm-a-s22 | 同上，seed22 | configs/studies/confirm-a-s22.json | 同上 |
| confirm-a-s33 | 同上，seed33 | configs/studies/confirm-a-s33.json | 同上 |
| a1-npu-explore | 独立NPU数值/离散/机制、精确恢复→batch校准→同预算query/FC探索 | configs/studies/a1-npu-stage.json；实际选择另存runs/a1-npu-explore/resolved-*.json | 1NPU/CPU4线程/24GiB；设备≤3h、等待≤30min、服务≤3.5h；校准≤20min，学习≤2h/各80步 |
| a2-hb-prepare（已完成） | HB配对初值、各自数值/离散/机制/恢复，共同B32/B128校准 | configs/studies/a2-hb-prepare.json；源码aff03ee | CPU1线程/RAM预算8GiB；准备20min/服务22min |
| a2-hb-dense-pilot / a2-hb-sparse-pilot | 已完成，B32各80步，两个CPU作业曾并行 | runs/a2-hb-prepare/resolved-hb-{dense,sparse}.json；生成后只读 | 各CPU1线程/RAM预算8GiB，训练40min、等待20min、服务70min |
| pilot-b（已完成） | 原模型、TIDE、MLP均终态；不启动B正式重复 | configs/studies/pilot-b.json | 各40步/B32，单NPU+CPU4线程/32GiB，组内50min；等待≤15min，服务70min |

新NPU阶段已确认进入background.slice，并通过共享队列取得物理1→逻辑npu:0；后续运行仍由队列选择，不能把当前物理ID写入模型。与三个CPU确认并行时，上限为1NPU/CPU配额7线程/RAM48GiB，低于首批64GiB。已完成B先导期间历史请求峰值为56GiB；这些均不是实际内存峰值或独占CPU承诺。B校准allocated峰值约3.8GiB，不等于驱动及其他HBM占用的完整峰值。最新状态以STATUS及job记录为准。

```sh
python scripts/job.py status artifacts/jobs/confirm-a-s11
python scripts/job.py status artifacts/jobs/a1-npu-explore
python scripts/job.py logs artifacts/jobs/a1-npu-explore --lines 60
# 停止整个服务及后代，包括等待队列和记录器；命令随后核对终止
python scripts/job.py stop artifacts/jobs/a1-npu-explore
```

其他seed替换作业ID即可。可直接查看`runs/confirm-a-s11/batch.json`与各子目录的`state.json`、`result.json`、`metrics.jsonl`；B对应`runs/pilot-b/`。新NPU阶段进度与五个验收轴见`runs/a1-npu-explore/stage.json`；子阶段日志为同目录的`mechanisms.log`、`resume.log`、`calibrate-b*.log`、`profile.log`、`learning.log`，训练曲线位于`learning/tl-a1-npu-{query,fc-dense}-s11-explore/metrics.jsonl`。尚未生成的文件表示尚未执行到该阶段。

systemd单元名为`tide-lab-ID.service`。即使终态单元被清理，`artifacts/jobs/ID/state.json`仍保存内层退出和后代检查；单元不存在不等于任务通过。已用真实CPU服务（含sleep孙进程）验证stop后无存活后代；新驱动另用实际CPU子进程测试了超时/忽略TERM/父进程exit0时清理后代。子阶段失败或超时的退出及终止检查保留在stage.json，不把外层成功当作内部通过。

Lens项目为`tide-learning-lab`，按`tl-a1-fc-dense-s11-confirm`、`tl-b1-tide-s11-pilot`等run前缀查找；独立记录器会添加projection后缀。每个终态run的`projection.json`保存destination_name/destination_run_id及读回核对。项目JSONL权威，展示投影不是exactly-once交付承诺。私有实例入口不写入公开仓库。

新NPU校准前缀为`tl-a1-npu-cal-b32-s11`、`tl-a1-npu-cal-b128-s11`及条件成立时的b256；学习前缀为`tl-a1-npu-query-s11-explore`和`tl-a1-npu-fc-dense-s11-explore`。计划不会预先创造不存在的Lens运行。接入补验只写项目权威检查记录，Lens查看训练曲线。

## 完整恢复

先停止旧作业并确认无后代，再使用最后发布的`checkpoint.pt`。A每40步发布，B每20步发布；未发布步骤可能丢失，不能假定任何当前步都可恢复。没有checkpoint时只能从初态开始，另记尝试。完成的B先导按预声明策略删除大checkpoint并保留hash；失败/中断不会自动删除。

恢复使用新作业ID、新输出和相同冻结配置/源码/已装依赖；执行脚本不会把权重加载当完整恢复。示例（A seed11中断后）：

```sh
python scripts/job.py submit \
  --name confirm-a-s11-resume-01 --source .local/source-first-a \
  --output artifacts/jobs/confirm-a-s11-resume-01 \
  --env-script "$PWD/.local/source-first-a/scripts/env.sh" \
  --env TIDE_LAB_VENV="$PWD/.venv" --device cpu \
  --timeout 4800 --wait-seconds 0 --threads 1 --memory-gib 8 \
  --input .local/configs-confirm-a-s11/a1-fc-dense-s11.json \
  --input runs/confirm-a-s11/tl-a1-fc-dense-s11-confirm/checkpoint.pt \
  -- python -m tide_lab.train \
  --config "$PWD/.local/configs-confirm-a-s11/a1-fc-dense-s11.json" \
  --resume "$PWD/runs/confirm-a-s11/tl-a1-fc-dense-s11-confirm/checkpoint.pt" \
  --output "$PWD/runs/tl-a1-fc-dense-s11-resume-01"
```

上例直接写权威记录；需要Lens时使用同样声明的独立`trackio-lens record --follow`查看新run，不追加旧projection。B恢复同理，需队列helper和固定asset路径（原job.json中完整记录）。现有严格恢复要求runtime identity相同，包括物理可见映射；不同映射会明确拒绝，不假称逐值续训。新进程CPU/NPU同栈恢复已有通过证据，跨设备映射恢复未检查。`--init-from`仅恢复模型权重、重新开始optimizer/数据进度，是不同实验。

首批终态已审阅：HB两者未达50%，B未达25%，均不启动本批正式重复；执行完成不解除严格失败。下一批规模/任务/候选待用户验收决定。当前没有需要恢复的中断训练。

新增NPU阶段不自动从半途重跑。若学习中断，停止并核对无后代后，使用同一`.local/source-a-npu`、原只读`resolved-fc-dense.json`和该run的`checkpoint.pt`提交新NPU作业，命令仍是`python -m tide_lab.train --config ABS_CONFIG --resume ABS_CHECKPOINT --output NEW_OUTPUT`，另设新run ID和剩余有界预算；保留原stage失败/中断。NPU checkpoint每20步发布。该B2恢复检查覆盖同栈完整AdamW/model/数据步/RNG，不覆盖不同物理映射或新增batch的逐值资格；当前恢复代码仍严格核对身份，不因探索而绕过。

本机实际资源控制与请求不同：cgroup v1未为用户服务委派CPU/内存控制器，CPUQuota/MemoryMax字段仅为请求；应用线程数和服务超时有效。详见STATUS及`artifacts/jobs/a1-npu-explore/resource-policy-observation.json`，不能把配置值当已生效的RAM硬限制。


## HB依赖等待与固定输入

HB模型与训练包固定`aff03eee27960113694c7c3d3b7a5c4cc82f8434`，只读`.local/source-hb-pilot`（树SHA256 `73e4a05c941dd2eb071ea816931765958df750f006fa106e90e8e92b26d710e4`）；等待驱动是只读`.local/source-hb-dependent/scripts/run_hb_after_preparation.py`，源码`e712831bd65c54cd8b361baac2cc8c47e7b2d5f7`，文件hash另列在各job.json的inputs。两个服务的cwd/PYTHONPATH仍指向aff03ee的训练源码，没有修改准备作业的活动输入。

```sh
python scripts/job.py status artifacts/jobs/a2-hb-prepare
python scripts/job.py status artifacts/jobs/a2-hb-dense-pilot
python scripts/job.py logs artifacts/jobs/a2-hb-dense-pilot --lines 60
# 如需取消一个先导，整个等待/训练/记录器一同停止并检查后代
python scripts/job.py stop artifacts/jobs/a2-hb-dense-pilot
```

sparse替换dense即可。准备状态在`runs/a2-hb-prepare/preparation.json`，每个先导的`runs/a2-hb-{dense,sparse}-pilot/dependency.json`区分waiting_dependency/admitted/failed。systemd的active不表示已经训练。准入后实际结果位于`learning/tl-a2-hb-{dense,sparse}-s11-pilot/`，Lens同此前缀查找，projection.json核对目标身份；尚未生成的run不当作已有结果。

HB恢复使用新job/output，保留`.local/source-hb-pilot`、原resolved配置及该run的checkpoint，按上文直接`python -m tide_lab.train --config ABS_CONFIG --resume ABS_CHECKPOINT --output NEW_OUTPUT`，限定剩余原预算；原等待驱动不会自动续训或再开正式重复。当前工作区的status/stop查看器支持实际cgroup v1全服务后代检查，活动作业仍保持旧冻结源码；正常子阶段退出另有自己的进程组清理记录。
