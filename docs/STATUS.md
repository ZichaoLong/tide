# Timed 的反向范围修复与 Shannon 弱质量策略

基座 `889a2ab3` 修复接收加权均值的恒零质量梯度，并标出已识别的反向范围错误。实验仓两项正式 CPU 接续分别越过第 2959、1760 步。可选策略 `weak-log-v1` 固定在 `e9bd1483`，在这两个实际失败窗口的 CPU 与 Ascend FP32 重放中，均以原始损失尺度完成参数更新。实测窗口包含 S、K 普通训练与 selection 验证。CPU 本次定向回归 144 项通过、3 项保留原 strict xfail，Ascend 6 项专用回归通过。[两种策略的数学与状态契约](shannon-positive-floor.md#weak-positive-log-coordinates) · [实验仓配对结果](../../tide-learning-lab/docs/pilot-r3-weak-mass-comparison.md)

基座实现 `6e14a6cc` 保留不等式总量的来源、求和尾数，以及舍入前的上下界归属。多节点复核使用原始分数差和剩余自由质量。第 921 步原故障的第二个节点恢复到 0.0000057220458984375，其接收量梯度与解析解一致。[语义](semantics.md) · [实现](architecture.md)

冻结版 152 项 CPU 定向检查通过，Ascend 测试作业 39 项通过。CPU 与 NPU 在各自轨迹中重放完整更新，另起进程载入第 922 步并复算第 923 步。第 921 步每端的 40,960 行均通过独立有理数投影参考，配对整步计时中位变化为 CPU +0.94%、NPU +0.45%。[双端验收、计时及旧结果抽查](../../tide-learning-lab/docs/evidence/pilot-r3-r24-foundation-quad-total-acceptance.json)

上述 quad 交付固定在完整提交 `6e14a6cc90e3ea1972611ee98b9de9bfea3b14fe`。本机实例作业记录可用 `python scripts/status.py` 查阅。下游验收、正式采用与实时进度分别沿实验仓的案例和状态入口查询。[r24 从故障到正式采用](../../tide-learning-lab/docs/requests/pilot-r3-r24-quad.md) · [实验仓当前交接](../../tide-learning-lab/docs/STATUS.md)

新实现可从与其数学定义对应的输入和梯度开始复验。CPU 定向回归中仍保留一个弱质量 V 梯度在三条执行路径上的 strict xfail。[数值条目与已知偏差](numerical-failure-modes.md) · [回归清单](../tests/numerical-regressions.txt)

此前区域注意力恒零先验及独立零先验右导数的修复对应 `68cf6d40`。[区域注意力验收](../../tide-learning-lab/docs/evidence/pilot-r3-r21-foundation-regional-prior-acceptance.json)

Pool 的 `positive-floor-v1` 验收继续对应实现 `b93eeae7`。[Pool 验收](../../tide-learning-lab/docs/evidence/pilot-r3-r16-foundation-pool-acceptance.json)

Timed 恒零路由、独立零先验、KV 未写入槽位、quad 常数下界及 Shannon 正支持继续按各自已测设备和精度记录。[路由验收](../../tide-learning-lab/docs/evidence/pilot-r3-r14-foundation-route-acceptance.json) · [Shannon 支持策略](shannon-positive-floor.md)
