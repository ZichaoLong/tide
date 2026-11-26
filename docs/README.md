# 实验仓导航

## 查状态

在仓库根目录运行 `python scripts/status.py`。两个分支使用同一个只读入口，`--json` 输出当前分支、提交、工作树修改、采样起止时间、交接原文和 `runtime`。本仓的 `runtime.kind` 按下表解释。记录中的门禁、选方和完成计数来自调度与验收文件。详细任务、心跳和观测错误可沿 `runtime.next_check` 查看。[状态入口](../scripts/status.py) · [实验状态采样](../scripts/status_runtime.py) · [调度报告](../pilotlab/report3.py)

| 采样结果 | 可以读出什么 |
|---|---|
| `live` | 调度控制器的 PID 和启动标识与本机活进程吻合，PID 同时匹配 systemd 主进程并属于其控制组。指针登记了 InvocationID 时也须吻合。runq 作业另外核对独立 worker 服务名、PID 和启动时钟，按物理卡列出与驱动 PID 的交集。 |
| `snapshot_only` | 有调度记录。服务、进程身份或采样期间的指针变化未能把它绑定到当前作业。查看 systemd 的实际状态、控制组和观测错误。 |
| `unavailable` | 这份检出缺少可用的启动指针或调度记录。最近一次已审阅的采样在交接中，并附时间与证据。 |

本批启动指针还登记了 44 个主体的检查点接续和 74 个早期主体的源码复核。状态命令的 `runtime.queue_additions` 按冻结的入队编号查询这两组作业。接续计入 Stage A 训练，单步复核单独计数。单项复核结果在其计划目录的 `observations/comparison.json`，其中记录原批次、起点、梯度与后继状态的比较。[当前交接](STATUS.md)

## 接续同一批

先读当前交接的“定位与接续入口”和当前待办。在原工作树从 `.local/pilot-r3/launch.json` 取 `unit`、`plan_path` 与 `plan_sha256`、`prepared_path`、`budget_ledger`、`deadline_unix`、`output` 和 `current_review`。核对计划与冻结源码/数据身份、总账的已消费量、绝对截止、最近发布的检查点、实际首个后继更新及阶段门禁。更换实现时在新目录验收，再将新的采用记录绑定到正式作业。[当前交接](STATUS.md) · [待办](ROADMAP.md) · [活动量和恢复验收](evidence/pilot-r3-r26-activity-readiness.json)

干净检出可以定位当时采用的源码、计划摘要、检查点哈希和首步。r26 正式采样的 `plan` 指向整批调度计划，四项首步核验的 `plan` 指向抽查计划。两份记录通过同一 `launch.sha256` 和 `unit` 对接，再按任务 ID 核对检查点哈希与首个后继步。运行中的账本、数据、完整检查点、冻结安装与日志在原工作树。可按交接给出的路径和哈希找回这些输入，再按当次授权恢复。同一批内保留已消费预算和原截止。[r26 正式采样](evidence/pilot-r3-main-r26-handoff.json) · [四项首步核验](evidence/pilot-r3-r26-resume-first-steps.json)

## 自动补位与手动派发

新提交使用独立维护的账户工具 `~/.local/bin/runq`，源码在 `~/tools/runq`，状态目录是 `~/.local/state/runq`。从 `runq submit --help` 或账户 `use-runq` skill 获取字段与完整示例。每个后端申请主机 RAM 与 CPU 线程，NPU 作业还申请每卡 HBM。开发者可在有足够余量的共享卡上手动试跑，向队列附上读数是可选项。

```bash
~/.local/bin/runq submit --backend cpu --threads 2 --ram-mib 1024 -- ./analyze input
~/.local/bin/runq status --json
~/.local/bin/runq show "$job_id"
~/.local/bin/runq priority "$job_id" 50
~/.local/bin/runq refill "$physical_id" off
```

`pin` 指定排队任务的 CPU 或物理卡，`request` 可在线调整预约。训练输入、评估、检查点与依赖仍由项目记录。按选定任务的完工时间和结果可用时刻决定优先级。

已冻结 Stage A 作业通过 `python scripts/status.py` 和启动指针中的 `runq_root` 查看，其独立工作服务使用原有的项目状态目录。项目队列的低层检查入口是 `python -m pilotlab.runq --root ~/.local/state/tide-runq status`。[项目状态](#查状态)

同轮交接控制器时，新控制器从本轮 `scheduler.json` 和 SQLite 接回全部已派模型，包括已结束待入账的 worker 和已预约待发布命令的请求。接回的模型数可以暂时超过 `pipeline_model_limit`。有新模型名额时，控制器再按冻结计划顺序补位。已有 worker 由各自的服务继续执行。[交接验证](evidence/pilot-r3-pipeline-window-recovery-20261124.json)

项目旧队列的 `limits --cards N --auto-card-ids ID ...` 热调自动使用的物理卡。已有外部进程的卡可用 `auto-share ID ... --until-unix TIMESTAMP` 登记当前进程身份和许可截止时间。后续任务沿用这份许可，出现新的外部进程时队列会等待。`auto-share ID ... --clear` 撤回这些卡的自动共享许可。

### 用阶段 A 的观测申请 NPU 显存

本轮项目控制器将 NPU 尝试逐项记入输出目录的 `hbm-profile.sqlite3`。同一冻结 worker 来源、模型配置、精度、确定性、序列长度、流分块数和 batch size 共用一组观测。种子和阶段可以不同。每次尝试分别保存 PyTorch 分配及缓存池峰值、可取得的初始化峰值、与本 worker 服务、PID 和启动时钟匹配的驱动采样峰值，以及 OOM 当时的池大小和申请量。旧 worker 的训练计数从初评结束后开始。首个训练更新前的驱动采样可补充这段记录，采样间隙仍可能漏峰。[资源记录](../pilotlab/hbm_profile3.py) · [worker 度量](../pilotlab/worker3.py)

缺省预约取 `ceil(1.5 × 估算基数/GiB + 0.5)` GiB。估算基数取各次观测峰值、OOM 当时已用量或缓存池加上未满足申请、已登记的代表任务临时依据中的最大值。可分别查看成功峰值、实际观测峰值和 OOM 估算。项目队列再取任务声明、显式下限与形状预约的最大值。派发时从驱动报告的物理空闲量中扣除在跑作业尚未兑现的预约。余量不足时，已有作业继续执行，容量恢复后自动补位。共享进程后续增长和动态输入仍可能导致实际 OOM。[热调与补跑](../pilotlab/runq_campaign.py) · [物理卡准入](../pilotlab/runq_devices.py)

在仓库根目录可用下列入口。变量 `task` 是冻结的 NPU 任务 JSON，`snapshot` 是本次导出的新文件路径。

```bash
round=.local/pilot-r3/campaign-runq/round
python -m pilotlab.hbm_profile3 --round "$round" estimate --task "$task"
python -m pilotlab.hbm_profile3 --round "$round" snapshot --output "$snapshot"
```

热调系数用 `policy --factor 1.5 --padding-gib 0.5`，运行中的控制器会随驱动采样更新预约。缺少同形状成功样本时，在提交这一组之前用 `trial --task "$task" --basis-gib N` 指定一个正式 NPU 任务，`N` 是加余量前的实测依据。该形状出现带显存读数的成功结果后，其余任务自动放行。失败结果保留原始错误、未满足申请和检查点；该形状继续等待。[代表任务与资源估算](../pilotlab/hbm_profile3.py)

同一输出目录中新建的阶段 B 正式任务会读取匹配的 A 记录。后续批次可向 `python -m pilotlab.prepare_campaign3` 传入 `--hbm-profile "$snapshot"`，计划按路径和 SHA-256 绑定资源表。冻结目录可迁移，跨目录复用时核对项目与基座的全部相对文件名和摘要、数据清单、Python 和依赖版本以及输入形状。对不上时沿用该任务自身的资源依据。正式任务冻结后，重启沿用同一份任务和门禁哈希。[阶段 B 正式任务](../pilotlab/round3.py) · [跨轮冻结入口](../pilotlab/prepare_campaign3.py)

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
