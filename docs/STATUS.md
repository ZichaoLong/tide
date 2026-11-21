# Timed 首层恒零发送量的反向修复已交付

2026-11-21，基座实现 `29c4798f` 在 Multiscale Timed 当前窗口内按外部输入时钟标记首层恒零发送量，沿真实入边延迟一拍传递。可微 pending 与独立零先验保留各自的右导数，证据标记只在路由时使用。无效来源层槽位使用布尔选择。真实先验的 tiled Attention 反向产生非有限值时，错误给出先验索引与 dtype。[语义契约](semantics.md)

冻结实现通过 144 项 CPU 与 65 项单卡 Ascend 定向回归。下游 A2 原失败批次在 CPU 和 Ascend 的 logits 和下一私有状态摘要均与各自原前向一致。123 份参数梯度对 CPU FP64 参考的最大绝对差分别为 2.12 × 10⁻⁷ 和 9.69 × 10⁻⁸。Ascend 从原正式第 2,720 步跑至第 2,732 步，新进程精确加载第 2,731 步。复算后私有状态逐位一致，参数、AdamW 矩和梯度分别按原 FP32 容差通过。[下游发布验收](../../tide-learning-lab/docs/evidence/pilot-r3-r13-foundation-ingress-acceptance.json)

下游正式 r14 已固定本实现，接续原 230 个未完成任务。故障 A2 在物理卡 1 上通过第 2,727 步，正式写出的第 2,761 步检查点已通过文件、状态及任务契约校验。B 关闭。[r14 正式接续](../../tide-learning-lab/docs/evidence/pilot-r3-main-r14-handoff.json)

当前源码也包含 [quad 常数下界修复](../../tide-learning-lab/docs/evidence/pilot-r3-r11-foundation-quad-reception-acceptance.json)、[KV 未写入槽位验收](../../tide-learning-lab/docs/evidence/pilot-r3-r10-foundation-kv-acceptance.json)、[Shannon 支持策略](shannon-positive-floor.md) 和 [Attention 单支持修复](attention-native-support-repair.md)。既有资格按对应的设备、精度、实现及工作负载分别记录。
