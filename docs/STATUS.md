# 区域注意力恒零先验修复已交付

实现提交 `68cf6d40` 在单区域与批量区域打分中传播节点入边的恒零证明。真实入边全部有证明且先验等于零时，区域注意力在归一化前排除该键。独立零先验保留右导数，正权重沿用原计算。[语义约定](semantics.md)

48 项 CPU 定向检查与 21 项 Ascend 定向检查通过。下游正式故障批次在 CPU 和 Ascend 上均通过 AdamW 更新，122 个参数张量与原公式 CPU FP64 参考的最大绝对误差分别约为 6.63e-6 和 3.37e-6。Ascend 从第 2,569 步正式检查点越过第 2,614 步，独立进程载入第 2,618 步并复算第 2,619 步，参数、优化器与梯度满足既有容差，私有状态逐位一致。[下游故障批次与恢复验收](../../tide-learning-lab/docs/evidence/pilot-r3-r21-foundation-regional-prior-acceptance.json)

北京时间 2026-11-22 19:00:50，下游 A 阶段 r22 的正式故障任务已训练至第 2,793 步，B 仍关闭。[正式接续](../../tide-learning-lab/docs/evidence/pilot-r3-main-r22-handoff.json)

Pool 的 `positive-floor-v1` 验收继续对应实现 `b93eeae7`。[Pool 验收](../../tide-learning-lab/docs/evidence/pilot-r3-r16-foundation-pool-acceptance.json)

Timed 恒零路由、独立零先验、KV 未写入槽位、quad 常数下界及 Shannon 正支持继续按各自已测设备和精度记录。[路由验收](../../tide-learning-lab/docs/evidence/pilot-r3-r14-foundation-route-acceptance.json) · [Shannon 支持策略](shannon-positive-floor.md)
