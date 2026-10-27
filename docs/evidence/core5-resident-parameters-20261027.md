# 常驻带权参数归约与局部训练资格

源码 `2e335e7a77f255c0418ce46b7f1673165330ccfc`；上游 `0770593`。[契约](../resident-weighted-parameters.md)；[机器记录](core5-resident-parameters-20261027.json)。

CPU 3 项归约测试和 NPU FP32 完整局部训练组合通过。三 bank 的跨角色真实别名归约为 10 个 canonical owners；同值独立叶保持独立。
SGD/Nesterov、AdamW/AMSGrad 各 4 步，对照独立逐行 ATen/autograd 和 NamedOptimizer，检查全部参数梯度、参数值、slots、None/零计数。
第 2 步后换 program/optimizer 恢复，之后两步完整 optimizer 状态按字节相同；全部发布 bank 相符。
513 元素尾块、缺席 poison、动态 root、非有限/溢出拒绝与失败不改 master/slot/发布值通过。

NPU 复用开发构建，完整组件/core/二进制摘要与固定源码一致，保留原 manifest。
CUDA sm80/90/100 新构建及 loader 检查通过；无 GPU 真机证据。
两任务均 clean、exit0、inactive；CPU耗时 0.77 秒。

首次开发检查误传显式空参数组，因此未选择参数并在计数断言失败；冻结树 `18b4665d` 与原失败日志保留。
修正的是检查 fixture 的参数组，未改容差或数值归约。新开发树 `3f166d4c` 通过。

本证据仅认证局部训练组合；完整带权图的状态、retained 反向、checkpoint、跨卡和性能仍需独立验收。
