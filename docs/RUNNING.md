# 首批后台作业与恢复

项目根目录执行下列命令。活动实现始终为`.local/source-first-a`中的`0d7ec5bdaa34556df9c93886e95f55fa5c9876ab`，不要改它、`.venv`或`.local/configs-*`；普通工作区可编辑并提交供下一快照使用。安装身份见dependencies，站点私有查看说明见`.local/LENS.md`。

| 作业ID | 实际内容 | 固定配置 | 硬预算 |
| --- | --- | --- | --- |
| confirm-a-s11 | query-only + FC/KV，seed11 | configs/studies/confirm-a-s11.json | 各400步/B32，CPU1线程/8GiB，组内75min，服务80min |
| confirm-a-s22 | 同上，seed22 | configs/studies/confirm-a-s22.json | 同上 |
| confirm-a-s33 | 同上，seed33 | configs/studies/confirm-a-s33.json | 同上 |
| pilot-b（已完成） | 原模型、TIDE、MLP均终态；不启动B正式重复 | configs/studies/pilot-b.json | 各40步/B32，单NPU+CPU4线程/32GiB，组内50min；等待≤15min，服务70min |

当前活动仅三个CPU确认服务（3线程/24GiB上限）。已完成B先导阶段的最大同时请求为1张NPU、7个CPU线程配额、56GiB RAM上限；不是实际内存峰值或独占CPU承诺。B校准allocated峰值约3.8GiB，不等于驱动及其他HBM占用的完整峰值。队列记录每次真实物理→逻辑映射；程序只使用npu:0。训练输出、日志和记录位于不同目录；不写入固定源码。

```sh
python scripts/job.py status artifacts/jobs/confirm-a-s11
python scripts/job.py status artifacts/jobs/pilot-b
python scripts/job.py logs artifacts/jobs/pilot-b --lines 60
# 停止整个服务及后代，包括等待队列和记录器；命令随后核对终止
python scripts/job.py stop artifacts/jobs/pilot-b
```

其他seed替换作业ID即可。可直接查看`runs/confirm-a-s11/batch.json`与各子目录的`state.json`、`result.json`、`metrics.jsonl`；B对应`runs/pilot-b/`。systemd单元名是`tide-lab-ID.service`。即使终态单元被清理，`artifacts/jobs/ID/state.json`仍保存内层退出和后代检查；单元不存在不等于任务通过。已用真实CPU服务（含sleep孙进程）验证stop后无存活后代。

Lens项目为`tide-learning-lab`，按`tl-a1-fc-dense-s11-confirm`、`tl-b1-tide-s11-pilot`等run前缀查找；独立记录器会添加projection后缀。每个终态run的`projection.json`保存destination_name/destination_run_id及读回核对。项目JSONL权威，展示投影不是exactly-once交付承诺。私有实例入口不写入公开仓库。

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

首批终态后唤醒接续：先查这四个服务及batch记录，审阅完整结果、精确恢复/数值边界、Lens投影；再更新每组结论和研究图。HB仍依赖A确认结果；B先导未过预设门槛，本轮不启动B正式重复，不因外层exit0自动扩展。下一批规模/任务/候选待用户验收决定。
