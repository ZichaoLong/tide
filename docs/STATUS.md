# Pool 正支持修复已交付

实现提交 `b93eeae7` 提供显式 `pool_support_policy="positive-floor-v1"`。对舍入丢失正发送者的行，存储保留最小正次正规数，按行调整极小预算的舍入余量。反向使用舍入前的概率，旧配置默认 strict。普通区域、批量区域及现有 work 采样均覆盖本策略。[数值约定](semantics.md)

冻结代码通过 46 项 CPU 和 45 项 Ascend 定向检查，Ascend 跳过 1 项 CPU FP64 专项。下游故障批次在两个设备上均完成一次 AdamW 更新，原弱节点被选中，全部 6 条出边保留最小正数。CPU 独立进程复算第 3,780 步获得逐位相同的参数、优化器矩、梯度和私有状态。[下游验收](../../tide-learning-lab/docs/evidence/pilot-r3-r16-foundation-pool-acceptance.json)

北京时间 2026-11-22 00:36:37，下游 A 阶段 r17 运行中，B 关闭。正式故障 A2 已通过第 3,775 步并完成全部 4,096 步，第 3,914 步正式检查点和最终第 4,096 步检查点均通过文件、任务及状态摘要校验。[正式接续与出边实测](../../tide-learning-lab/docs/evidence/pilot-r3-main-r17-handoff.json)

Timed 恒零路由、独立零先验、KV 未写入槽位、quad 常数下界及 Shannon 正支持继续按各自已测设备和精度记录。[路由验收](../../tide-learning-lab/docs/evidence/pilot-r3-r14-foundation-route-acceptance.json) · [Shannon 支持策略](shannon-positive-floor.md)
