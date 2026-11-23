# 当前待办

1. 使用消费者提供的真实样本扩展 `6e14a6cc` 的机制验证，覆盖 quad 弱余量、区域注意力、pool 正发送量和多分支 Timed 路由。核对恒零边、独立零先验、微小正权重及跨窗状态。用采样行数、补值行数和主机复核开销限定性能结论。[路由语义](semantics.md) · [Shannon 支持策略](shannon-positive-floor.md)
2. 按实际负载扩展 Attention 单支持、近似常量 V、集中分布、弱质量和 BF16 多 key 的数值与性能范围，分别记录设备、执行路径、精度和误差。[Attention 验收](attention-native-support-repair.md)
3. 扩展恢复与状态验证时沿用 Shannon 恢复、CPU FP64 私有 KV 与弱质量见证的原始容差，分别报告源码、设备、精度与有限超差。[数值回归与已知偏差](numerical-failure-modes.md)

消费者的主体迁移、选参和门禁由实验仓跟进。[实验仓待办](../../tide-learning-lab/docs/ROADMAP.md)
