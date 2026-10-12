# 当前路线图与验收范围

本文件只记录当前待办和完成边界。[STATUS](STATUS.md) 是唯一现场交接，
[执行契约](execution-flows.md) 是持续约束，历史逐阶段过程保存在 Git 和
[证据索引](evidence/README.md)。已关闭阶段不是新实验队列。

## 已完成：tide-core-3 本机交付

| 范围 | 状态与入口 |
| --- | --- |
| 原始六阶段基座 | 完成；[最终验收](evidence/foundation-final.md) |
| 跨图类别、独立特化、公共实验库 | 完成；[跨类资格](evidence/cross-family-qualification.md)、[库资格](evidence/library-foundation.md) |
| eager CPU/CUDA/NPU 扩展 | 源码交付，CPU/NPU 声明范围实测；[加速器资格](evidence/accelerators-20260928.md)；CUDA 实机待办独立保留 |
| G1：State/Read 批量一阶 VJP、Add 组织 | 完成；[实现与资格](evidence/batched-vjp-cuda-local-20261010.md)、[Add 后续](evidence/add-batch-followup-20261010.md) |
| G2：有限 profiling 与选型 | 完成；[六组件](evidence/npu-batched-components-20261011.md)、[完整消费者](evidence/native-selection-20261011.md) |
| G3：CUDA 常驻 | 声明范围源码和本机构建完成；[目标机门禁](device-control.md) 尚待真机 |
| G4：CPU/NPU 相关验证 | 完成；[公共模块](evidence/npu-module-batching-20261011.md)、[常驻 composite](evidence/resident-composite-20261011.md)、[精度/stream 修复](evidence/precision-and-stream-gates-20261011.md) |
| G5：有限建议与目标机交接 | 本机阶段完成；[最终选型](evidence/native-selection-20261011.md) |
| 新线程接续整理 | 完成；[固定教材与语义索引](semantic-map.md)、[干净源码入口](development.md)、[产物退役与起步验证](evidence/artifact-retirement-20261012.md) |

这些结论保留各自提交、环境、模块和输入范围。接口存在不证明任意模型可导入；
源码/编译不证明设备运行；单次观测不证明稳定速度排名。历史失败不改为通过。

## 剩余与后续任务

1. **外部验证**：在 GPU 真机重建干净提交，运行正确性、实际常驻、多卡、续接、
   内存校准和有限性能门禁。其他主机架构/工具链版本也分别验证。
2. **待用户提供**：新版教材与特例。先按 [语义索引](semantic-map.md) 比较变化，
   再决定需要添加局部模块、独立特化、表示扩展还是调度/反向修复；当前不预先实现。
3. **可选优化**：混合提交/同步、小 Tensor、因果批宽、常驻 gather/scatter/通信、
   CUDA 初版串行语义适配等。需要具体 profiling 问题和有限预算，不是自动待跑任务。

## 新任务验收顺序

固定数学与输入域 → 独立小型参考/反例 → 实现与相关回归 → 干净源码必要资格
→ 更新能力与限制 → 仅在需要选型时运行有限性能实验。

新特例仍验证完整观察量、续接、独立根 VJP、优化器与 checkpoint 的适用范围。
原规范内的任意合法拓扑/输入不得因为某个批量契约缺失而被静默换模型；
合法 streaming 后备与明确设备能力拒绝按现有接口分别处理。
