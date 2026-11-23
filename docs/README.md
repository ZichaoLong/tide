# 实验仓导航

## 查状态

在仓库根目录运行 `python scripts/status.py`。两个分支使用同一个只读入口，`--json` 输出当前分支、提交、工作树修改、采样起止时间、交接原文和 `runtime`。本仓的 `runtime.kind` 按下表解释。记录中的门禁、选方和完成计数来自调度与验收文件。详细任务、心跳和观测错误可沿 `runtime.next_check` 查看。[状态入口](../scripts/status.py) · [实验状态采样](../scripts/status_runtime.py) · [调度报告](../pilotlab/report3.py)

| 采样结果 | 可以读出什么 |
|---|---|
| `live` | 调度控制器的 PID 和启动标识与本机活进程吻合，PID 同时匹配 systemd 主进程并属于其控制组。指针登记了 InvocationID 时也须吻合。驱动可读时按物理卡列出控制组成员与驱动 PID 的交集。 |
| `snapshot_only` | 有调度记录。服务、进程身份或采样期间的指针变化未能把它绑定到当前作业。查看 systemd 的实际状态、控制组和观测错误。 |
| `unavailable` | 这份检出缺少可用的启动指针或调度记录。最近一次已审阅的采样在交接中，并附时间与证据。 |

## 接续同一批

先读当前交接的“定位与接续入口”和当前待办。在原工作树从 `.local/pilot-r3/launch.json` 取 `unit`、`plan_path` 与 `plan_sha256`、`prepared_path`、`budget_ledger`、`deadline_unix`、`output` 和 `current_review`。核对计划与冻结源码/数据身份、总账的已消费量、绝对截止、最近发布的检查点、实际首个后继更新及阶段门禁。更换实现时在新目录验收，再将新的采用记录绑定到正式作业。[当前交接](STATUS.md) · [待办](ROADMAP.md) · [活动量和恢复验收](evidence/pilot-r3-r26-activity-readiness.json)

干净检出可以定位当时采用的源码、计划摘要、检查点哈希和首步。r26 正式采样的 `plan` 指向整批调度计划，四项首步核验的 `plan` 指向抽查计划。两份记录通过同一 `launch.sha256` 和 `unit` 对接，再按任务 ID 核对检查点哈希与首个后继步。运行中的账本、数据、完整检查点、冻结安装与日志在原工作树。可按交接给出的路径和哈希找回这些输入，再按当次授权恢复。同一批内保留已消费预算和原截止。[r26 正式采样](evidence/pilot-r3-main-r26-handoff.json) · [四项首步核验](evidence/pilot-r3-r26-resume-first-steps.json)

新计划按账号硬配额剩余量与输出文件系统可用空间的较小值派工，可用字节耗尽时停止准入。冻结源码时按本次复制的大小核对空间。检查点轮换、长期保留量和并发峰值仍写入容量预测。r26 的冻结计划使用两项 24 GiB 限制。同批受控停机后，可在 `pilot-r3-planned-controller-continuation-v1` 接续审阅中指定 `"storage_policy": "hard_quota"`，由新控制器采用零预留，沿用原 worker、总账、截止与 A 阶段范围。[调度准入](../pilotlab/scheduler3.py) · [接续准备](../pilotlab/prepare_stage_a_migration3.py)

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
