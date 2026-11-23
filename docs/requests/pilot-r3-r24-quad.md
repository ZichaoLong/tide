# pilot-r3-r24-quad · 第 921 步的弱余量

本案由实验仓报告并验收，`graph-execution-foundation` 维护区域分配的实现。采用的数学快照为 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`。以下记录区分首次故障、修复交付、消费者验收和正式作业采用。[本批协作契约](../../../graph-execution-foundation/docs/downstream-contract.md)

| 环节 | 可核对的事实 |
|---|---|
| r24 故障 | CPU 正式任务在第 921 步未满足 quad 总量约束。原行两个分数为 `[34.98263168334961, -205.1953582763672]`，接收为 `[0.5-d/2, 0.5+d/2]`，`d=5.7220458984375e-6`。旧实现把第二节点发出的量置零。独立有理数投影给出 `[1-d,d]`，第二项对接收的梯度为 `[-2,0]`。精确 `d=0` 的控制例应取恒零分支。[原始诊断](../evidence/pilot-r3-r24-quad-total-constraint.json) · [独立投影](../../scripts/r24_quad_reference.py) |
| 基座修复 | `6e14a6cc90e3ea1972611ee98b9de9bfea3b14fe` 保留不等式总量的来源、求和尾数和舍入前的自由面。原行、邻近弱正余量和精确零均有闭式期望。基座根目录运行 `TIDE_STABLE_DEVICE=cpu TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q tests/test_quad_total.py`。[实现](../../../graph-execution-foundation/src/tide/quad.py) · [回归](../../../graph-execution-foundation/tests/test_quad_total.py) · [可复用的数值条目](../../../graph-execution-foundation/docs/numerical-failure-modes.md#quad-不等式总量的来源与舍入前的自由面) |
| 实验仓验收 | CPU 从原正式检查点开始，NPU 从便携负载和独立的设备 RNG 开始。两个设备各核对第 921 步 40,960 行，在各自轨迹上保存第 922 步、另起进程复算第 923 步。四组配对整步计时的中位变化为 CPU +0.94%、NPU +0.45%。验收中的 `diagnosis.sha256` 绑定原诊断，`authority.commit` 绑定基座实现。[验收程序](../../scripts/accept_r24_quad.py) · [双端验收与计时](../evidence/pilot-r3-r24-foundation-quad-total-acceptance.json) |
| 正式采用 | r25 的 `acceptance.sha256` 绑定上述验收，`foundation_commit` 绑定同一实现。原任务从正式检查点越过故障步，921、922 步的记录与 CPU 重放相符。r26 的交接继续登记相同的基座提交。[r25 正式采用](../evidence/pilot-r3-main-r25-handoff.json) · [r26 采用记录](../evidence/pilot-r3-main-r26-handoff.json) |

## 在干净检出中核对

实验仓根目录运行 `python -m pytest -q tests/test_r24_quad_evidence.py` 即可用已提交的记录和参考源码核对上述引用。基座的局部回归可以直接从相邻分支运行。单独检出也可用固定提交定位实现与用例。[证据链回归](../../tests/test_r24_quad_evidence.py) · [固定修复](https://github.com/ZichaoLong/tide/tree/6e14a6cc90e3ea1972611ee98b9de9bfea3b14fe)

原诊断的 SHA-256 为 `228289e28063b79218b0ac34b0acb374cade594ee3791ea8969a0b26bdf27c14`，消费者验收为 `b77e783a2f77a12b91e4e73c4226db446b03632ca03725fb9f47ecd4f5b5e2b7`。完整双端更新、恢复和计时复演还需要验收记录列出的原检查点、数据、环境和设备。局部公式、测试和上述绑定均可在没有 `.local/` 的检出中核验。
