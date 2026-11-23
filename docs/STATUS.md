# A 阶段 r30 已切换到 CPU/NPU 直接派工，B 关闭

北京时间 2026-11-23 22:39:52，`tide-pilot-r3-campaign-r30.service` 持续运行，B 门禁关闭。当前 270 个主体已有 116 个选定配方，154 个待选方。r30 调度器本轮登记 261 个已完成请求、73 个运行、184 个待派。继承的 3,284 个完成请求另计，后续请求随依赖生成。[清理后现场采样](evidence/pilot-r3-r30-storage-cleanup.json) · [本轮继承范围](evidence/pilot-r3-main-r30-handoff.json)

另有 74 个早期主体已完成旧证据盘点。42 个已从原任务复算选方并核验接续链，32 个已定位到首个未完成阶段。全部 74 个在当前基座上的适用性待验证。[逐模型证据](evidence/pilot-r3-deferred-74-review.json) · [审阅顺序](ROADMAP.md)

## 实际占用与排队

- 22:39 调度器登记 40 个 CPU worker、32 个 NPU worker。驱动在物理卡 2、4、5、7、8、11、14 上列出属于该服务的 32 个 PID。19:00 另做过 8 秒 CPU 繁忙率与队列位置采样。[22:39 调度与驱动](evidence/pilot-r3-r30-storage-cleanup.json) · [19:00 利用率与队列](evidence/pilot-r3-main-r30-handoff.json)
- 本地硬配额为 700 GiB，清理后已用 404.16 GiB，可用 295.84 GiB，固定预留为零。私有 SFS 保存了 146.19 GiB 压缩检查点。已用最终清单恢复两个样本，并用各自冻结的 worker 核验内部状态摘要。[空间及恢复凭据](evidence/pilot-r3-r30-storage-cleanup.json) · [恢复命令](README.md#恢复已归档的检查点)
- 全批主存上限为 1 TB，原账本绝对截止为北京时间 2026-12-20 00:48:33。[资源与账本](evidence/pilot-r3-main-r30-handoff.json)

## 切换与恢复

r29 先派了 30 个大批量 CPU 任务，登记的预计主存合计 994.6 GB，只能再派 1 个 NPU 任务。r30 先为已绑定 NPU 的训练申请卡，再从主存需求较小的任务开始补位。26 个容量校准请求已从训练队列撤下。此前等待设备校准的请求中，38 个已在 CPU、21 个已在 NPU 写出正式优化器更新。[r29 停止审计](evidence/pilot-r3-campaign-r29-terminal-audit.json) · [r30 接续](evidence/pilot-r3-r30-resource-readiness.json) · [正式步记录](evidence/pilot-r3-main-r30-handoff.json)

已核实旧控制组释放。r30 接回 41 个带检查点的请求、20 个从初始检查重试的请求和 223 个未启动请求。已从正式步日志核验 25 个检查点后的首个更新。切换涉及 5 个任务合计重算 6 个尚未写入检查点的更新。正式训练沿用原数据、配方、worker 源码、验收基座 `6e14a6cc`、预算总账和截止时间。[迁移逐项审阅](evidence/pilot-r3-r30-resource-readiness.json) · [冻结计划与恢复凭据](evidence/pilot-r3-main-r30-handoff.json)

## 定位与接续入口

运行 `python scripts/status.py` 查询实时状态，运行 `python .local/pilot-r3/inspect.py --json` 查看逐任务明细。启动指针为 `.local/pilot-r3/launch.json`，其中 `unit`、`plan_path`、`plan_sha256`、`output`、`current_review` 对应服务、冻结计划、输出及接续审阅。受控停止命令为 `systemctl --user stop tide-pilot-r3-campaign-r30.service`。[状态与接续说明](README.md#接续同一批)
