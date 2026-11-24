# 实验仓导航

## 查状态

在仓库根目录运行 `python scripts/status.py`。两个分支使用同一个只读入口，`--json` 输出当前分支、提交、工作树修改、采样起止时间、交接原文和 `runtime`。本仓的 `runtime.kind` 按下表解释。记录中的门禁、选方和完成计数来自调度与验收文件。详细任务、心跳和观测错误可沿 `runtime.next_check` 查看。[状态入口](../scripts/status.py) · [实验状态采样](../scripts/status_runtime.py) · [调度报告](../pilotlab/report3.py)

| 采样结果 | 可以读出什么 |
|---|---|
| `live` | 调度控制器的 PID 和启动标识与本机活进程吻合，PID 同时匹配 systemd 主进程并属于其控制组。指针登记了 InvocationID 时也须吻合。runq 作业另外核对独立 worker 服务名、PID 和启动时钟，按物理卡列出与驱动 PID 的交集。 |
| `snapshot_only` | 有调度记录。服务、进程身份或采样期间的指针变化未能把它绑定到当前作业。查看 systemd 的实际状态、控制组和观测错误。 |
| `unavailable` | 这份检出缺少可用的启动指针或调度记录。最近一次已审阅的采样在交接中，并附时间与证据。 |

## 接续同一批

先读当前交接的“定位与接续入口”和当前待办。在原工作树从 `.local/pilot-r3/launch.json` 取 `unit`、`plan_path` 与 `plan_sha256`、`prepared_path`、`budget_ledger`、`deadline_unix`、`output` 和 `current_review`。核对计划与冻结源码/数据身份、总账的已消费量、绝对截止、最近发布的检查点、实际首个后继更新及阶段门禁。更换实现时在新目录验收，再将新的采用记录绑定到正式作业。[当前交接](STATUS.md) · [待办](ROADMAP.md) · [活动量和恢复验收](evidence/pilot-r3-r26-activity-readiness.json)

干净检出可以定位当时采用的源码、计划摘要、检查点哈希和首步。r26 正式采样的 `plan` 指向整批调度计划，四项首步核验的 `plan` 指向抽查计划。两份记录通过同一 `launch.sha256` 和 `unit` 对接，再按任务 ID 核对检查点哈希与首个后继步。运行中的账本、数据、完整检查点、冻结安装与日志在原工作树。可按交接给出的路径和哈希找回这些输入，再按当次授权恢复。同一批内保留已消费预算和原截止。[r26 正式采样](evidence/pilot-r3-main-r26-handoff.json) · [四项首步核验](evidence/pilot-r3-r26-resume-first-steps.json)

## 自动补位与手动派发

在本机任意目录可运行 `~/.local/bin/runq`，仓库入口是 `scripts/runq`。自动池默认占用最多 8 张物理 NPU，目标为 40 个 CPU worker。一个账号队列协调所有提交者。CPU 的派发独立于 NPU 驱动采样。调度器、各物理卡租约和各 worker 分别运行在 `background.slice` 的独立用户服务里。调整自动目标或重启调度器时，已启动的 worker 持续执行。[命令入口](../scripts/runq) · [派发实现](../pilotlab/runq.py)

```bash
~/.local/bin/runq status
~/.local/bin/runq status --state queued
~/.local/bin/runq show "$job_id"
~/.local/bin/runq pin "$job_id" "npu:$physical_id"
~/.local/bin/runq pin "$another_job_id" cpu
~/.local/bin/runq limits --cards 8 --cpu 40
```

`pin` 接受队列数字 ID 或完整任务键，指定排队任务的目标。要让它回到自动池，用 `pin "$job_id" auto`。手动任务可以超过自动卡数或 CPU 数，卡上的手动请求优先取得下一块满足其峰值显存的空间。若该卡只有本队列的任务，自动任务也可利用剩余显存。有外部进程时，手动请求绑定现场进程身份。给该请求加 `--allow-auto-share` 可在它运行期间允许自动任务与同一批外部进程共享。授权按该次请求生效，提交时的进程身份有变时可重新核对并用 `pin` 更新。最后一个手动任务退出后，已借用该卡的 worker 继续完成。有空出的常规卡名额时，无外部进程的卡可转入自动池。[准入与优先顺序](../pilotlab/runq_place.py)

也可以提交其他项目的独立命令。在该项目的工作目录和已激活环境中，设置唯一键、峰值显存与主存估计（单位 MiB）。以下命令中的路径和资源量由提交者按实际任务替换。已领取的 NPU 在程序内映射为 `npu:0`。通用程序可读取 `RUNQ_DEVICE`，物理卡号可读 `RUNQ_PHYSICAL_DEVICE`。

```bash
~/.local/bin/runq submit --key "$job_key" --target "npu:$physical_id" \
  --hbm-mib "$peak_hbm" --ram-mib "$peak_ram" --timeout "$seconds" -- "$python" "$worker" --device npu:0
~/.local/bin/runq submit --key "$cpu_key" --target cpu \
  --ram-mib "$peak_ram" --timeout "$seconds" -- "$python" "$worker" --device cpu
```

自动提交使用 `--target auto --backend npu|cpu|either`。同一键与同一份命令和环境重复提交会返回同一 ID。任务列表、等待原因、提交/派发/启动时间、PID、终态都保存在 `~/.local/state/tide-runq/`，可用 `show` 核对。实验 worker 的首个真实优化步从其 `steps.jsonl` 记录。新卡需要等待账号协作队列与两次驱动采样。已有租约上的派发由唤醒通知触发。指定的主存与显存为峰值，新增预订会扣除已运行进程尚未使用的峰值余量。Stage A 优先采用同一条冻结轨迹已发表的主存峰值。CPU 真正完成首个优化步后，调度器结合该进程的内核高水位、检查点写入开销以及任务的冷启动估算更新预记，所有 worker 共用原批次 1 TB 上限。[设备测量](../pilotlab/runq_devices.py) · [进程与设备锁](../pilotlab/runq_runtime.py)

接续 Stage A 时，队列沿用冻结的任务和检查点。未绑定的成对种子共用一次后端决定，已有轨迹保持原后端。收尾跨设备迁移需逐条核验检查点兼容性与预计节时。单条通用任务可用 `runq cancel ID` 停止。调度器升级用 `runq restart`，它加载新的调度源码快照。实验依赖控制器的 unit、冻结计划与账本记录在本机启动指针中。[Stage A 接口](../pilotlab/runq_campaign.py) · [启动指针说明](#接续同一批)

## 恢复已归档的检查点

2026-11-23 清理的 R1、R2、R3 冷检查点保存在本账号的私有 SFS 目录。恢复清单按原仓库相对路径记录大小、SHA-256、权限和修改时间。原轮次的任务、结果及终态审计仍按原路径引用这些文件。需要重跑某轮的全文件审计时，先在仓库根目录按该轮恢复，再运行原审计命令。[清理和恢复实测](evidence/pilot-r3-r30-storage-cleanup.json) · [恢复程序](../scripts/restore_checkpoint_archive.py)

下面的命令将一个已抽查的检查点放回原路径，要求本机有 Python 3.9+ 和 `zstd`。

```bash
archive=/mi/sfs_turbo/zlong/tide-learning-lab-archive/storage-20261123-r30
(cd "$archive" && sha256sum -c recovery-manifest.sha256)
checkpoint=.local/pilot-r3/campaign-r09/round/tasks/exchange-d128-k-base-4aac2dcb10-a2-2026112001-r2/attempt-0/checkpoints/step-000002300-40220a07fa.pt
python scripts/restore_checkpoint_archive.py \
  --manifest "$archive/recovery-manifest.json" --under "$checkpoint" --root "$PWD"
```

`--under` 也接受轮次目录，如 `.local/pilot-r3/campaign-r09/round`。使用 `--verify-only` 可逐文件解压并核对哈希。用 `--root` 指定空目录可做隔离恢复。目标已有相同内容时跳过，遇到不同内容时保留现场并报错。私有目录自带恢复程序。若整个归档搬到新目录，将 `archive` 指向新位置，恢复时加 `--store local="$archive" --store sfs="$archive"`。

另有 89 个重复文件在清理时改用硬链接，共用保留文件的修改时间。归档目录内的 `dedup-events.jsonl` 保存了每个原路径、内容哈希及去重前的纳秒修改时间 `prior_mtime_ns`。

## 定位和修改

| 问题 | 源码和最近的回归入口 |
|---|---|
| 调度、进程与现场报告 | [调度器](../pilotlab/scheduler3.py) · [只读审计](../pilotlab/audit3.py) · [报告用例](../tests/test_pilot_report3.py) |
| 选方、活动量与门禁 | [选方](../pilotlab/round3.py) · [门禁](../pilotlab/gate3.py) · [零步接续用例](../tests/test_pilot_activity_resume3.py) |
| 跨源码恢复、首步与训练凭据 | [来源核验](../pilotlab/source_migration3.py) · [恢复适配](../pilotlab/resume_adapter3.py) · [接续审阅](../pilotlab/stage_a_continued3.py) |
| 算子、梯度、支持集或模型状态 | 从故障样本和独立预期到本批基座的实现、回归、下游验收和新批采用。[r24 完整案例](requests/pilot-r3-r24-quad.md) · [基座数值问题索引](../../graph-execution-foundation/docs/numerical-failure-modes.md) |

本批采用基座维护的 `tide-instances-collaboration-v2`。同机两份工作树按 `tide-learning-lab` 和 `graph-execution-foundation` 同级放置即可沿相对链接互查。单独检出实验仓也可按记录中的提交读固定版契约与实现。源码、测试、计划和审阅记录保留在原路径。审阅 JSON 的绝对运行路径用于辨认原机输入，搬迁后以记录的哈希核对原件。[协作契约](../../graph-execution-foundation/docs/downstream-contract.md) · [本批固定版本](https://github.com/ZichaoLong/tide/blob/6e14a6cc90e3ea1972611ee98b9de9bfea3b14fe/docs/downstream-contract.md)

实验回归按用例绑定的基座运行。r26 活动量回归可在实验环境下将 `PYTHONPATH` 指向 `6e14a6cc` 的 `src`。针对早期 quad 选方证据迁移的 `test_pilot_quad_transfer_probe3.py` 检查当时的 `allocation.py` 契约，可使用转移记录中 `new_prepared` 对应的冻结基座。干净检出可直接运行 `python -m pytest -q tests/test_worktree_status.py tests/test_r24_quad_evidence.py` 核验状态入口与 r24 证据链。[迁移记录](evidence/pilot-r3-stage-a-transfer.json) · [迁移探针回归](../tests/test_pilot_quad_transfer_probe3.py)

## 开新周期

从用户本次指定的语义来源、基座、研究问题和授权开始。选定数学定义后可按算子挑选数值反例和独立参考，在批量训练前验证快速路径。数据、配方、资源、预算和门禁按这次任务确定。历史按用户明确指定的范围查询。[新周期工作规则](../AGENTS.md#新周期工作规则)
