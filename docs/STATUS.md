# Quad 区域分配修复已交付

基座实现 `6e14a6cc` 保留不等式总量的来源、求和尾数，以及舍入前的上下界归属。多节点复核使用原始分数差和剩余自由质量。第 921 步原故障的第二个节点恢复到 0.0000057220458984375，其接收量梯度与解析解一致。[语义](semantics.md) · [实现](architecture.md)

冻结版 152 项 CPU 定向检查通过，Ascend 测试作业 39 项通过。CPU 与 NPU 在各自轨迹中重放完整更新，另起进程载入第 922 步并复算第 923 步。第 921 步每端的 40,960 行均通过独立有理数投影参考，配对整步计时中位变化为 CPU +0.94%、NPU +0.45%。[双端验收、计时及旧结果抽查](../../tide-learning-lab/docs/evidence/pilot-r3-r24-foundation-quad-total-acceptance.json)

北京时间 2026-11-23 09:51:08，下游 A 阶段 r25 已接回原 CPU 任务，训练至第 922 步，B 关闭。[正式接续与进程采样](../../tide-learning-lab/docs/evidence/pilot-r3-main-r25-handoff.json)

此前区域注意力恒零先验及独立零先验右导数的修复对应 `68cf6d40`。[区域注意力验收](../../tide-learning-lab/docs/evidence/pilot-r3-r21-foundation-regional-prior-acceptance.json)

Pool 的 `positive-floor-v1` 验收继续对应实现 `b93eeae7`。[Pool 验收](../../tide-learning-lab/docs/evidence/pilot-r3-r16-foundation-pool-acceptance.json)

Timed 恒零路由、独立零先验、KV 未写入槽位、quad 常数下界及 Shannon 正支持继续按各自已测设备和精度记录。[路由验收](../../tide-learning-lab/docs/evidence/pilot-r3-r14-foundation-route-acceptance.json) · [Shannon 支持策略](shannon-positive-floor.md)
