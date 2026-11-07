# 首批后台作业与恢复

项目根目录执行下列命令。旧CPU确认作业固定`.local/source-first-a`中的`0d7ec5bdaa34556df9c93886e95f55fa5c9876ab`；新增NPU阶段固定`.local/source-a-npu`中的`0494d208e2a561ac66da9d1fa3d72672af6809e8`（树SHA256 `adc5f9eabc01e106b36055b017f3d1391f19d4a1079ee9b213cb3420c24a34fe`）。不要改这些快照、`.venv`、`.local/configs-*`或NPU输出内的resolved配置；普通工作区修改用于下一快照。模型/任务/训练包源码与原固定wheel相同，新阶段仅增加驱动、检查、配置与记录。安装身份见dependencies，站点私有查看说明见`.local/LENS.md`。

| 作业ID | 实际内容 | 固定配置 | 时间上限与资源预算 |
| --- | --- | --- | --- |
| confirm-a-s11 | query-only + FC/KV，seed11 | configs/studies/confirm-a-s11.json | 各400步/B32，CPU1线程/8GiB，组内75min，服务80min |
| confirm-a-s22 | 同上，seed22 | configs/studies/confirm-a-s22.json | 同上 |
| confirm-a-s33 | 同上，seed33 | configs/studies/confirm-a-s33.json | 同上 |
| a1-npu-explore | 独立NPU数值/离散/机制、精确恢复→batch校准→同预算query/FC探索 | configs/studies/a1-npu-stage.json；实际选择另存runs/a1-npu-explore/resolved-*.json | 1NPU/CPU4线程/24GiB；设备≤3h、等待≤30min、服务≤3.5h；校准≤20min，学习≤2h/各80步 |
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

首批终态后唤醒接续：先查表中服务及batch/stage记录，审阅完整结果、精确恢复/数值边界、Lens投影；再更新每组结论和研究图。HB仍依赖A确认结果；B先导未过预设门槛，本轮不启动B正式重复，不因外层exit0自动扩展。下一批规模/任务/候选待用户验收决定。

新增NPU阶段不自动从半途重跑。若学习中断，停止并核对无后代后，使用同一`.local/source-a-npu`、原只读`resolved-fc-dense.json`和该run的`checkpoint.pt`提交新NPU作业，命令仍是`python -m tide_lab.train --config ABS_CONFIG --resume ABS_CHECKPOINT --output NEW_OUTPUT`，另设新run ID和剩余有界预算；保留原stage失败/中断。NPU checkpoint每20步发布。该B2恢复检查覆盖同栈完整AdamW/model/数据步/RNG，不覆盖不同物理映射或新增batch的逐值资格；当前恢复代码仍严格核对身份，不因探索而绕过。

本机实际资源控制与请求不同：cgroup v1未为用户服务委派CPU/内存控制器，CPUQuota/MemoryMax字段仅为请求；应用线程数和服务超时有效。详见STATUS及`artifacts/jobs/a1-npu-explore/resource-policy-observation.json`，不能把配置值当已生效的RAM硬限制。
