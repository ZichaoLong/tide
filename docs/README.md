# 实验仓导航

贵阳 R3 正式续跑沿用本机冻结的任务、配方和检查点。两份 8 卡作业分别运行 runq，作业编号、继续派发与阶段门禁见云端交接。[贵阳 R3 正式续跑](portable-execution.md#贵阳-r3-正式续跑)

## 查状态

在仓库根目录运行 `python scripts/status.py`。两个分支使用同一个只读入口，`--json` 输出当前分支、提交、工作树修改、采样起止时间、交接原文和 `runtime`。本仓的 `runtime.kind` 按下表解释。记录中的门禁、选方和完成计数来自调度与验收文件。详细任务、心跳和观测错误可沿 `runtime.next_check` 查看。[状态入口](../scripts/status.py) · [实验状态采样](../scripts/status_runtime.py) · [调度报告](../pilotlab/report3.py)

| 采样结果 | 可以读出什么 |
|---|---|
| `modelarts_snapshot` | 控制器最近一次读取的贵阳平台阶段、runq 计数、已选主体和真实首步回执。结合平台采样时间和本机控制器 unit 判断时效。可通过两个作业 ID 直接核实分配状态。[贵阳作业入口](portable-execution.md#贵阳-r3-正式续跑) |
| `live` | 调度控制器的 PID 和启动标识与本机活进程吻合，PID 同时匹配 systemd 主进程并属于其控制组。指针登记了 InvocationID 时也须吻合。runq 作业另外核对独立 worker 服务名、PID 和启动时钟，按物理卡列出与驱动 PID 的交集。 |
| `snapshot_only` | 有调度记录。服务、进程身份或采样期间的指针变化未能把它绑定到当前作业。查看 systemd 的实际状态、控制组和观测错误。 |
| `unavailable` | 这份检出缺少可用的启动指针或调度记录。最近一次已审阅的采样在交接中，并附时间与证据。 |

原本机启动指针登记了 44 个主体的检查点接续和 74 个早期主体的源码复核。本机适配器的 `runtime.queue_additions` 对应这两组历史作业。贵阳适配器返回两份作业各自的 `running_by_backend` 与 `queue_observed_unix`。已选出配方的主体数和对照数分别见 `runtime.selected_subjects` 与 `runtime.selected_controls`。单项复核报告为 `comparison.json`，沿原计划可找到已复用观测的哈希及新尝试。[当前交接](STATUS.md)

## 接续同一批

当前 R3 从贵阳两份作业、华东的尝试账本和科学快照接续。冻结计划、两套 worker 源码、各尝试的断点历史、累计预算和原绝对截止都随该批保留。原本机 `.local/pilot-r3/launch.json` 可定位迁移前使用的计划、预算和审阅记录。在新目录验收实现变更后，将采用记录绑定到后续作业。[贵阳 R3 正式续跑](portable-execution.md#贵阳-r3-正式续跑) · [当前交接](STATUS.md) · [待办](ROADMAP.md)

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

贵阳生产作业 `tide-r3-20261128-1` 已从原检查点完成 NPU 任务 2237 的第 1,313 步和 CPU 任务 2907 的第 1,771 步。本机随后释放 196 份检查点的 266 个原路径，共 17.66 GiB。其中包括 2237 搬卡时另存的硬链接。本次迁移累计释放 509.44 GiB。2026-11-28 20:32，本仓实占 6.83 GiB，账户实占 50.89 GiB。[两项贵阳首步](evidence/pilot-r3-guiyang-first-updates-20261128.json) · [最后一批释放与隔离恢复](evidence/pilot-r3-modelarts-post-first-retirement-20261128.json) · [较早的本机清理](evidence/pilot-r3-modelarts-local-storage-20261128.json)

14 份训练数据和清单（278,720,072 字节）留在原路径，可通过 `make_base` 重建两份冻结基座清单。任务、逐步指标、结果、选方和调度记录继续供原生审核使用。清理后在隔离目录复核了原 281 项可派任务所需的 229 份输入，并从当前登记的 18 个云端请求重建出完全相同的描述符和运行计划。[阶段 A 候选复现](evidence/pilot-r3-guiyang-candidate-recovery-20261128.json) · [清理后复核](evidence/pilot-r3-modelarts-post-first-retirement-20261128.json)

2026-11-28 20:53，华东至贵阳的 14 项 OMS 传输全部完成。两地对同一批 210 份续训输入及数据（19,237,056,320 字节）逐件核对了完整 SHA-256。现有贵阳作业就近读取了其中 167 份，另 43 份复用了与当前 ETag 和版本匹配的完整读取回执。对应的两地索引、逐对象回执以及首步验证后释放本机副本的增量记录可一并查询。[双区终验及 OBS 索引](evidence/pilot-r3-modelarts-regional-completion-20261128.json)

同一批需要续验时，可运行 `python scripts/verify_tide_regional_inputs.py --job tide-r3-20261128-1 --workers 2`。每条 SSH 连接按顺序核验一批对象，`--workers` 指定已有贵阳节点上的低优先级读流数。中断后重跑会对齐目标版本并复用已完成的回执。[区域输入校验工具](../scripts/verify_tide_regional_inputs.py)

华东总索引支持按原路径定位检查点、任务结果、冻结数据、日志及早期私有归档。149,041 个索引项中有 148,745 个可恢复，另 296 个已在原实验中按授权退役，索引记录了它们的原 SHA 和结果来源。用配置好的 `maflow`、Python 3.9+ 和 `zstd` 可在干净客户机上核验、恢复任一存续文件。[华东索引及四类隔离恢复回执](evidence/pilot-r3-modelarts-storage-recovery-20261128.json) · [按原路径恢复](../scripts/restore_modelarts_archive.py)

```bash
saved=.local/pilot-r3/risk-screen-r01/frozen/round/tasks/risk-trained-fc-s-npu/attempt-0/checkpoints/step-000008207-d1c2146be6.pt
index_sha=b65c287a1beb58574ae58e646658eccf002d447d8f3a64746abda8fd8330a808
python scripts/restore_modelarts_archive.py "$saved" --index-sha256 "$index_sha"
python scripts/restore_modelarts_archive.py "$saved" --index-sha256 "$index_sha" --to "$PWD/$saved"
```

第一条命令查询来源、大小与摘要。第二条恢复文件、权限和修改时间。NPU、CPU 两个续训点以及 2237 的交接副本均已用同一入口从华东恢复至空目录，并核对了内容和文件属性。[隔离恢复回执](evidence/pilot-r3-modelarts-post-first-retirement-20261128.json) · [贵阳 R3 正式续跑](portable-execution.md#贵阳-r3-正式续跑)

2026-11-23 归档的 R1、R2、R3 检查点共有 1,830 个原路径、1,818 份不同内容。华东 OBS 沿用原恢复清单及压缩对象结构。清单按原仓库相对路径记录解压后的字节数、SHA-256、权限和纳秒修改时间。原轮次的全文件审计按路径读取这些文件，可以先恢复所需的轮次或单个检查点。[原归档记录](evidence/pilot-r3-r30-storage-cleanup.json) · [恢复程序](../scripts/restore_checkpoint_archive.py)

配好本账户的 `maflow`、Python 3.9+ 和 `zstd` 后，可从任一客户机恢复下列检查点。用 `--root` 指定空目录可做隔离恢复。

```bash
archive=east:/tide-learning-lab/production/r3-20261128/legacy/storage-20261123-r30
checkpoint=.local/pilot-r3/campaign-r09/round/tasks/exchange-d128-k-base-4aac2dcb10-a2-2026112001-r2/attempt-0/checkpoints/step-000002300-40220a07fa.pt
python scripts/restore_checkpoint_archive.py \
  --manifest "$archive/recovery-manifest.json" \
  --manifest-sha256 33df9e2b4d050117636f3e612d7e83635de891af13694a4deef446f1e87a67fa \
  --store local="$archive" --store sfs="$archive" --under "$checkpoint" --root "$PWD"
```

`--under` 也接受轮次目录，如 `.local/pilot-r3/campaign-r09/round`。`--verify-only` 逐文件解压并核对哈希。目标已有相同内容时可重复执行，遇到不同内容时保留现场并报错。本地或 SFS 上的归档可继续用目录作为 `--manifest` 和 `--store` 的位置。

另有 89 个重复文件在清理时改用硬链接，共用保留文件的修改时间。归档目录内的 `dedup-events.jsonl` 保存了每个原路径、内容哈希及去重前的纳秒修改时间 `prior_mtime_ns`。

## 定位和修改

| 问题 | 源码和最近的回归入口 |
|---|---|
| 调度、进程与现场报告 | [调度器](../pilotlab/scheduler3.py) · [只读审计](../pilotlab/audit3.py) · [报告用例](../tests/test_pilot_report3.py) |
| 贵阳两节点派发与结果回传 | [控制器](../scripts/tide_cloud_controller.py) · [结果验收与跨区归档](../scripts/tide_cloud_results.py) · [首次自动生成后继的凭据](evidence/pilot-r3-guiyang-first-successors-20261128.json) |
| 选方、活动量与门禁 | [选方](../pilotlab/round3.py) · [门禁](../pilotlab/gate3.py) · [零步接续用例](../tests/test_pilot_activity_resume3.py) |
| 跨源码恢复、首步与训练凭据 | [来源核验](../pilotlab/source_migration3.py) · [恢复适配](../pilotlab/resume_adapter3.py) · [接续审阅](../pilotlab/stage_a_continued3.py) |
| 算子、梯度、支持集或模型状态 | 从故障样本和独立预期到本批基座的实现、回归、下游验收和新批采用。[r24 完整案例](requests/pilot-r3-r24-quad.md) · [基座数值问题索引](../../graph-execution-foundation/docs/numerical-failure-modes.md) |

本批采用基座维护的 `tide-instances-collaboration-v2`。同机两份工作树按 `tide-learning-lab` 和 `graph-execution-foundation` 同级放置即可沿相对链接互查。单独检出实验仓也可按记录中的提交读固定版契约与实现。源码、测试、计划和审阅记录保留在原路径。审阅 JSON 的绝对运行路径用于辨认原机输入，搬迁后以记录的哈希核对原件。[协作契约](../../graph-execution-foundation/docs/downstream-contract.md) · [本批固定版本](https://github.com/ZichaoLong/tide/blob/6e14a6cc90e3ea1972611ee98b9de9bfea3b14fe/docs/downstream-contract.md)

实验回归按用例绑定的基座运行。r26 活动量回归可在实验环境下将 `PYTHONPATH` 指向 `6e14a6cc` 的 `src`。针对早期 quad 选方证据迁移的 `test_pilot_quad_transfer_probe3.py` 检查当时的 `allocation.py` 契约，可使用转移记录中 `new_prepared` 对应的冻结基座。干净检出可直接运行 `python -m pytest -q tests/test_worktree_status.py tests/test_r24_quad_evidence.py` 核验状态入口与 r24 证据链。[迁移记录](evidence/pilot-r3-stage-a-transfer.json) · [迁移探针回归](../tests/test_pilot_quad_transfer_probe3.py)

## 开新周期

从用户本次指定的语义来源、基座、研究问题和授权开始。选定数学定义后可按算子挑选数值反例和独立参考，在批量训练前验证快速路径。数据、配方、资源、预算和门禁按这次任务确定。历史按用户明确指定的范围查询。[新周期工作规则](../AGENTS.md#新周期工作规则)

## 跨位置执行

从本机、ModelArts 或两处运行冻结 R3 任务时，用项目脚本生成计划，交给 `maflow project run --plan` 提交。接手已有批次先用 `maflow project list` 找项目和批次，再用 `maflow project show`、`collect`、`resume` 查看进度、收取证据和补跑。项目根据原始结果、检查点及尝试链复核完成项。[冻结、跨作业恢复与汇总](portable-execution.md)
