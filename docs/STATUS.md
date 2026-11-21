# 当前交接｜KV 未写入槽位修复已交付，A 已接续

2026-11-21。基座实现提交 `17c7d675` 将 KV Full 的有效时间窗与实际写入位置取交集。Timed/K A1 的两个未写入槽位曾产生约 −1.21 × 10⁴⁰ 的中间权重导数。零接收的 KV 写入导数为零，通用 Attention 对显式零先验继续使用右导数。[KV 写入语义](semantics.md) · [下游故障现场与 FP64 公式](../../tide-learning-lab/docs/evidence/pilot-r3-r10-kv-no-write.json)

极端空槽和正权重 0.5、1e-30 的两个回归在旧版均复现。修复后定向 CPU 139 项通过、3 项原有预期失败，单卡 Ascend 58 项通过、3 项原有预期失败。下游用同一提交从正式 step 1126 在 NPU 上完成 1127–1140，故障 batch 的输入与目标逐项一致。新进程精确载入 1139，重放 1140 的逻辑状态、数据、loss、梯度范数匹配，参数、AdamW 和梯度通过原 FP32 容差。[定向验收](../../tide-learning-lab/docs/evidence/pilot-r3-r10-foundation-kv-acceptance.json)

下游 r11 自 12:20 在该实现上接续 A，B 关闭。Shannon 主体沿用 `positive-floor-v1`。74 个先前曝光主体仍待独立迁移审阅。[下游交接](../../tide-learning-lab/docs/STATUS.md) · [Shannon 配方与证据](shannon-positive-floor.md)

## 同一源码中的既有 Attention 修复

北京时间 2026-11-20 18:40。按用户授权，NPU auto/显式npu_fusion的单有效key与空行已使用解析VJP，混合batch保留多key融合。扩展见证发现近似常量V消减后，FP32融合包装增加公共V中心化。质量振幅、真实零质量右导数、连接零梯度及原state/checkpoint格式保留。

详见[当前修复记录](attention-native-support-repair.md)。定向CPU172通过、3原xfail；实际NPU39通过。42个NPU样本完成，含四个旧状态各16步真实接续、首尾8个CPU FP64抽查，均在原门槛内；全部fresh-process恢复通过，最大差5.6904e-7。独立局部公式、同实现FP64诊断、有限恢复与长期学习结论分开。

原Shannon恢复超差、CPU私有KV有限超差、弱质量夹具及首次单支持-only多key反例均保留。当前不声明全部幅值、BF16多key、CUDA或旧选参已获得新资格。
