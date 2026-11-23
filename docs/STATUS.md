# A 阶段 r30 已切换到 CPU/NPU 直接派工，B 关闭

北京时间 2026-11-23 19:00，`tide-pilot-r3-campaign-r30.service` 正在运行。累计继承 3,284 个已完成请求。本轮接回的 284 个请求中，调度器登记 87 个运行、197 个待派、0 个失败。当前 270 个主体已有 114 个选定配方，156 个仍在推进。另有 74 个早期曝光主体待独立审阅。后续请求随依赖完成而生成。[正式运行采样](evidence/pilot-r3-main-r30-handoff.json) · [当前待办](ROADMAP.md)

## 实际占用与排队

- CPU 已启动 40/40 个训练进程，每个配置 4 个计算线程。连续 8 秒采样中，作业可用的 160 个物理核平均繁忙率为 84.6%，其中 CPU 训练进程平均消耗约 97.1 核。主机繁忙率包括其他进程。[实际进程与采样口径](evidence/pilot-r3-main-r30-handoff.json)
- NPU 已申请 8 张，取得 7 张，物理 ID 为 2、4、5、7、8、11、14。控制组有 46 个 NPU 训练进程，驱动在 7 张卡上列出其中 41 个 PID。另 9 张卡均有其他作业进程，第 8 个申请排在队首。已取得的共享卡按 48 GiB 上限和在跑任务预计峰值计，最多还可接纳 1.78 GiB，当前最小待派任务需要 2.02 GiB。[队列、每卡显存与驱动采样](evidence/pilot-r3-main-r30-handoff.json)
- 磁盘固定预留为零。采样时按账号硬配额与文件系统可用量计算，可用空间为 127.5 GiB。全批主存上限为 1 TB，账本绝对截止为北京时间 2026-12-20 00:48:33。[资源与账本](evidence/pilot-r3-main-r30-handoff.json)

## 切换与恢复

r29 先派了 30 个大批量 CPU 任务，登记的预计主存合计 994.6 GB，只能再派 1 个 NPU 任务。r30 先为已绑定 NPU 的训练申请卡，再从主存需求较小的任务开始补位。26 个容量校准请求已从训练队列撤下。此前等待设备校准的请求中，38 个已在 CPU、21 个已在 NPU 写出正式优化器更新。[r29 停止审计](evidence/pilot-r3-campaign-r29-terminal-audit.json) · [r30 接续](evidence/pilot-r3-r30-resource-readiness.json) · [正式步记录](evidence/pilot-r3-main-r30-handoff.json)

已核实旧控制组释放。r30 接回 41 个带检查点的请求、20 个从初始检查重试的请求和 223 个未启动请求。已从正式步日志核验 25 个检查点后的首个更新。切换涉及 5 个任务合计重算 6 个尚未写入检查点的更新。正式训练沿用原数据、配方、worker 源码、验收基座 `6e14a6cc`、预算总账和截止时间。[迁移逐项审阅](evidence/pilot-r3-r30-resource-readiness.json) · [冻结计划与恢复凭据](evidence/pilot-r3-main-r30-handoff.json)

## 定位与接续入口

运行 `python scripts/status.py` 查询实时状态，运行 `python .local/pilot-r3/inspect.py --json` 查看逐任务明细。启动指针为 `.local/pilot-r3/launch.json`，其中 `unit`、`plan_path`、`plan_sha256`、`output`、`current_review` 对应服务、冻结计划、输出及接续审阅。受控停止命令为 `systemctl --user stop tide-pilot-r3-campaign-r30.service`。[状态与接续说明](README.md#接续同一批)
