# 当前交接｜Shannon 正支持已通过 r09 隔离验收，A 停止，B 关闭

2026-11-20 夜间。本机 `src/tide` 提供显式 `shannon_support_policy="positive-floor-v1"`。26 个 Python 文件的内容哈希为 `cbf9c33157bd744e240fc4e662862cd147a5ba78861452838c053e8945d1e4f4`，已验收的实现提交为 `885f7ef96be8cc31814e509c687017e6ff922d72`。CPU 和单卡 NPU 在相同故障输入恢复 23 个正发送者，并完成该批及两个真实后继批的 AdamW 更新。CPU 同策略恢复逐位一致，NPU 私有状态逐位一致，其他差异通过原阈值。新策略对异常行使用主机高精度求解，吞吐待测。[基座接口及回归](shannon-positive-floor.md) · [下游验收](../../tide-learning-lab/docs/requests/TL-REQ-007.md)

## 同一源码中的既有 Attention 修复

北京时间 2026-11-20 18:40。按用户授权，NPU auto/显式npu_fusion的单有效key与空行已使用解析VJP，混合batch保留多key融合。扩展见证发现近似常量V消减后，FP32融合包装增加公共V中心化。质量振幅、真实零质量右导数、连接零梯度及原state/checkpoint格式保留。

详见[当前修复记录](attention-native-support-repair.md)。定向CPU172通过、3原xfail；实际NPU39通过。42个NPU样本完成，含四个旧状态各16步真实接续、首尾8个CPU FP64抽查，均在原门槛内；全部fresh-process恢复通过，最大差5.6904e-7。独立局部公式、同实现FP64诊断、有限恢复与长期学习结论分开。

原Shannon恢复超差、CPU私有KV有限超差、弱质量夹具及首次单支持-only多key反例均保留。当前不声明全部幅值、BF16多key、CUDA或旧选参已获得新资格。

下游 r09 已在旧 strict 基座上停止。74 个旧曝光主体仍待迁移审阅。后续实验可从基座提交 `885f7ef9` 冻结源码，并在 Shannon 配方中显式选择正支持策略。[下游当前交接](../../tide-learning-lab/docs/STATUS.md)
