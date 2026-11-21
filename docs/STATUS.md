# 当前交接｜quad 常数下界的接收反向已交付

北京时间 2026-11-21。实现提交 `75d2a893` 将普通节点 quad 常数零下界的证明沿真实入边送到接收注意力。下游 Exchange/K A3 的两条零权重入边曾在第 8016 步分别产生约 3.11 × 10³⁹ 的独立先验右导数。它们对实际生产者输入的导数均为零。Timed 的一步延迟、批量区域、重计算与区域放置均传递对应标记。可独立变化的先验仍使用声明的导数。[基座语义](semantics.md) · [下游故障证据](../../tide-learning-lab/docs/evidence/pilot-r3-r11-quad-reception-vjp.json)

冻结实现通过 170 项 CPU 与 84 项单卡 Ascend 定向回归。原故障快照的独立缩放对照与发布实现 logits 逐位一致。NPU 参数梯度最大绝对差为 1.86 × 10⁻⁹。同一 logits 的交叉熵归约存在 FP32 末位浮动，FP64 复核通过。下游 NPU 隔离接续从正式第 7986 步完成到第 8021 步，新进程精确载入第 8020 步并复核下一步的参数、AdamW、梯度、私有状态、数据游标和 RNG。[发布版验收](../../tide-learning-lab/docs/evidence/pilot-r3-r11-foundation-quad-reception-acceptance.json)

下游 r12 已固定该实现并于 14:18 正式接续 A，Exchange/K A3 已越过原故障步。故障任务已通过本阶段终态检查，完成 8192 步、2,097,152 个训练 token。正式第 8192 步检查点已重验文件与状态摘要。B 关闭。既有 KV 未写入修复及 Shannon `positive-floor-v1` 使用同一份基座源码。[下游交接](../../tide-learning-lab/docs/evidence/pilot-r3-main-r12-handoff.json) · [KV 验收](../../tide-learning-lab/docs/evidence/pilot-r3-r10-foundation-kv-acceptance.json) · [Shannon 配方](shannon-positive-floor.md)

## 同一源码中的既有 Attention 修复

北京时间 2026-11-20 18:40。按用户授权，NPU auto/显式npu_fusion的单有效key与空行已使用解析VJP，混合batch保留多key融合。扩展见证发现近似常量V消减后，FP32融合包装增加公共V中心化。质量振幅、真实零质量右导数、连接零梯度及原state/checkpoint格式保留。

详见[当前修复记录](attention-native-support-repair.md)。定向CPU172通过、3原xfail；实际NPU39通过。42个NPU样本完成，含四个旧状态各16步真实接续、首尾8个CPU FP64抽查，均在原门槛内；全部fresh-process恢复通过，最大差5.6904e-7。独立局部公式、同实现FP64诊断、有限恢复与长期学习结论分开。

原Shannon恢复超差、CPU私有KV有限超差、弱质量夹具及首次单支持-only多key反例均保留。当前不声明全部幅值、BF16多key、CUDA或旧选参已获得新资格。
