# 当前待办

NPU单支持解析分支及FP32融合V中心化已完成有限资格，见[STATUS](STATUS.md)、[修复记录](attention-native-support-repair.md)。

1. 跟进下游从 `885f7ef9` 冻结的 Shannon、Reception 和 Attention 修复。Shannon 正支持已通过 r09 隔离验收，后续按实际负载测量异常行主机复核的成本。[接口与当前范围](shannon-positive-floor.md)
2. 配合74个旧曝光主体的迁移审阅，明确新源码与原checkpoint、优化器矩和已消费曝光的契约；可加载不等于旧学习结果合格。
3. 根据实际证据扩展数值范围和性能检查。当前独立单支持/近似常量V及集中分布见证通过，不是所有弱质量、幅值或BF16多key的全局保证。
4. 保留原Shannon恢复、CPU FP64私有KV与弱质量见证的有限超差及原阈值；源码、设备、精度和实测范围分别交付。

下游 A 停在 r09，B 关闭。原冻结记录和数学快照用于复核。
