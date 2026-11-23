# A 阶段 r26 已越过 A3 选方阻断，B 关闭

以下是北京时间 2026-11-23 11:14:05 的交接采样。查本机当前状态请运行 `python scripts/status.py`。采样时服务 `tide-pilot-r3-campaign-r26-final.service` 正在运行。270 个主体中已有 92 个选定配方。r26 继承 2,735 个完成请求，本修订新完成 15 个，累计 2,750 个。此时调度登记 47 个运行、159 个待派、0 个失败。控制组中有 16 个 CPU worker，NPU 驱动在 8 张卡上列出 23 个本服务进程。另有 74 个早期曝光主体待单独审阅。[正式运行采样](evidence/pilot-r3-main-r26-handoff.json)

## A3 活动量

Exchange/d256/K/capacity 的两个 A3 种子均达到 2,097,152 token。种子 2026112001 零步接续，选方沿原任务、源码、结果和精确恢复的检查点引用前段第 8,192 步的实测样本。种子 2026112002 使用本段第 8,192 步的样本。两者各测 256 个外部 token，平均活动骨干参数 6,092,229，骨干参数总数 6,501,589，活动比例 93.7037%。离线选方、跨源码训练链审阅和 A 门禁分别从原始文件复算。错配种子和哈希均被拒绝。正式选方已写入相同结果。[接续验收](evidence/pilot-r3-r26-activity-readiness.json) · 原机正式选择见轮次输出的 `selected.json`

## 四个恢复点

原审计保留 1 条共享 worker 结果绑定差异。已按任务、PID、进程启动标识、终态心跳、结果和检查点索引核实该 worker 在第 2,688 步按信号停机。另 3 条 `worker_exit_-9` 已绑定最后发布并解码验证的检查点。两条日志比最后心跳多一个完整步，训练从索引指向的检查点重放尾段。Shannon 的阶段起点评估在原冻结环境中从第 1,024 步补测，消耗 16,384 个选择集 token，优化器更新 0 步，已计入总账和主体评估预算。[接续验收](evidence/pilot-r3-r26-activity-readiness.json) · [原终态审计](evidence/pilot-r3-campaign-r25-terminal-audit.json)

| 任务 | 接入步 | 正式首步 | 本次采样最新步 |
| --- | ---: | ---: | ---: |
| Timed/d128/S/mixed9，A2 种子 2001 | 2,688 | 2,689 | 2,932 |
| Timed/d128/X/mixed9，A2 种子 2002 | 6,002 | 6,003 | 6,188 |
| Exchange/d256/K/base，A3 种子 2001 | 7,377 | 7,378 | 7,569 |
| Exchange/d256/K/shannon，A2 种子 2002 | 1,174 | 1,175 | 1,181 |

已观察到 4/4 个任务在正式作业中越过接入步。[四项首步核验](evidence/pilot-r3-r26-resume-first-steps.json)记录了每项的检查点哈希、实际启动凭据和首个优化器更新。各任务的正式参数和步日志见[正式运行采样](evidence/pilot-r3-main-r26-handoff.json)。

## 定位与接续入口

- 在仓库根目录运行 `python scripts/status.py`，加 `--json` 可读采样时间、systemd、控制组、驱动 PID、阶段和阻断。原机主指针为 `.local/pilot-r3/launch.json`，其 `plan_path` 指向冻结计划，`log` 指向控制器日志。逐任务报告见输出的 `scheduler.json` 及命令给出的 `runtime.next_check`。[状态和接续入口](README.md#查状态) · [提交时的计划和作业采样](evidence/pilot-r3-main-r26-handoff.json)
- 实验源码在 `pilotlab/`。选方和门禁见 `round3.py`、`gate3.py`，活动量来源和恢复凭据见 `source_migration3.py`、`resume_adapter3.py`，跨源码审阅见 `stage_a_continued3.py`。回归用例见 [A3 零步接续测试](../tests/test_pilot_activity_resume3.py)。训练基座冻结在 `6e14a6cc`，模型、算子、数据和配方沿用 r25。[当前活动量和恢复验收](evidence/pilot-r3-r26-activity-readiness.json)
- CPU 上限 160 个物理核和 40 个 worker，NPU 上限 8 张卡，每卡最多 16 个共享 worker，总 RSS 上限 1 TB。提交观测的账号硬配额与文件系统可用量取小值为 235.0 GiB，安全余量和并发检查点写入上限各 24 GiB。沿用原设备时数账本和北京时间 2026-12-20 00:48:33 截止。受控停止命令为 `systemctl --user stop tide-pilot-r3-campaign-r26-final.service`。[正式运行采样](evidence/pilot-r3-main-r26-handoff.json)
