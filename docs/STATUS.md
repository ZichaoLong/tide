# A 阶段 r27 已按硬配额派工，B 关闭

北京时间 2026-11-23 14:45:55，服务 `tide-pilot-r3-campaign-r27.service` 正在运行。r27 继承 2,998 个完成请求，本修订已完成 3 个，累计 3,001 个。本次采样登记 55 个运行、193 个待派、0 个失败。270 个主体中有 101 个已选方，另有 74 个早期曝光主体待独立审阅。实时状态可用 `python scripts/status.py` 查询。[正式运行采样](evidence/pilot-r3-main-r27-handoff.json) · [当前待办](ROADMAP.md)

## 切换与恢复

截至受控停止，r26 本修订共完成 263 个请求。原控制组、登记进程和队列租约均已释放。原始终态审计列出一条共享 worker 结果绑定差异。已用该进程的 PID 和启动标识、终态心跳、结果与检查点索引核实其按信号停止。另一条等待分配 NPU 的请求在停机时已有预期结果路径，队列凭据、进程记录及空的尝试目录均证实它尚未取得设备或启动 worker。[原始终态审计](evidence/pilot-r3-campaign-r26-terminal-audit.json) · [逐项接续审阅](evidence/pilot-r3-r27-storage-readiness.json)

新控制器在首轮接回全部 251 个未完成请求。47 个中断任务从信号停机检查点接续，1 个从已发布并校验的检查点第 3,308 步接续，2 个沿用先前绑定的检查点，201 个未启动请求重新入队。沿用原始数据、配方、worker 源码、验收基座 `6e14a6cc` 和已消费预算。本次采样从 31 个中断任务的正式步日志验证了恢复后的首个优化器更新。原 A3 零步接续所需的活动量证据仍可按任务与种子追溯。[首轮准备与正式首步](evidence/pilot-r3-main-r27-handoff.json) · [A3 活动量来源](evidence/pilot-r3-r26-activity-readiness.json)

## 定位与接续入口

- 主指针为 `.local/pilot-r3/launch.json`。核对 `unit`、`plan_path`、`plan_sha256` 和 `current_review`，可找到冻结计划、日志、原始停止记录及各恢复凭据。逐任务明细见 `python .local/pilot-r3/inspect.py --json`。受控停止命令为 `systemctl --user stop tide-pilot-r3-campaign-r27.service`。[状态与接续说明](README.md#接续同一批)
- 实际可用空间取账号硬配额剩余量与输出文件系统可用空间的较小值，固定预留为零。采样时实际可用 199.9 GiB，运行中任务的检查点估算合计 24.39 GiB，驱动在 7 张卡上列出 18 个本服务进程。CPU 上限 160 个物理核与 40 个 worker，NPU 上限 8 张卡，总 RSS 上限 1 TB，绝对截止为北京时间 2026-12-20 00:48:33。[正式运行与资源记录](evidence/pilot-r3-main-r27-handoff.json)
- 调度与接续源码见 `pilotlab/scheduler3.py`、`scheduler_dynamic3.py`、`prepare_stage_a_migration3.py` 和 `cutover_review3.py`。选方与门禁见 `round3.py`、`gate3.py`，模型、梯度和状态修复归所采用的基座。[源码与回归入口](README.md#定位和修改)
