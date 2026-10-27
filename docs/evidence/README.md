# 历史验证与测量证据

最新：[core-5 带权循环记忆](core5-recurrent-20261027.md) — `442e8b2`，544 项定向 CPU 资格。

[core-5 带权 KV 首组合](core5-weighted-kv-20261027.md) — `6fb467c`，129 项定向 CPU 资格。

[core-5 结构化表示](core5-structured-20261027.md) — `714d5f9`，212 项定向 CPU 资格。

[core-5 一般 SettleGraph](core5-settle-20261027.md) — `efb9d62`，247 项定向 CPU 资格；不认证尚未完成的带权局部函数或新设备路径。

本目录保留已审阅的实现提交、验证范围、结果、失败和性能限制。
它是历史证据，不是待执行清单；当前任务只由 [STATUS](../STATUS.md) 和
[ROADMAP](../ROADMAP.md) 决定。

2026-10-12 用户确认旧实验默认不再复核，原始构建/运行产物可退役。
报告中的绝对路径、`TASK/...` 和 `artifacts/...` 仅标识当时的来源，
**不承诺文件仍在线可用**。必要记录与退役说明见 [产物保留规则](../artifact-retention.md)。
历史报告保留原文和原失败；原始文件移除不等于原验证未发生，也不构成新源码通过证据。

## 主要结论入口

- 基座：[最终资格](foundation-final.md)、[跨图资格](cross-family-qualification.md)、[公共库](library-foundation.md)。
- eager 加速器：[范围与环境](accelerators-20260928.md)。
- 最新批量实现：[初始 CPU/CUDA 资格](batched-vjp-cuda-local-20261010.md)、[Add 后续](add-batch-followup-20261010.md)。
- NPU：[公共模块](npu-module-batching-20261011.md)、[常驻完整组件](resident-composite-20261011.md)、[stream/精度](precision-and-stream-gates-20261011.md)。
- 性能：[旧全尺寸结论](selection-terminal-20261010.md)、[六组件](npu-batched-components-20261011.md)、[最新完整选型](native-selection-20261011.md)。
- 不能忽略的限制：[原始近似平票路由失败](original-add-route-witness-20261004.md)、最新完整选型中的两项容量拒绝。
- 接续与清理：[固定教材/干净 clone 起步验证及历史产物退役](artifact-retirement-20261012.md)。

查看某个结果时先确认源码、环境、模块、图类别、精度、输入和观察量范围。
跨硬件浮点结果、共享服务器争用与历史计时不能承诺原样重放。
默认基于当前源码创建新的独立验证/测量，不重启旧 job 或补写旧结果。
