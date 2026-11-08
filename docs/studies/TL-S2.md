# TL-S2 — NPU轻量采样完成，观测状态一致，耗时归因仍有限

2026-11-08主代理审阅。问题是详细活动/梯度观测是否改变训练状态，以及主线程栈能否缩小耗时解释。固定core5 FC/KV、width16/depth2、dense、seed11、B32、NPU FP32，物理NPU3→logical npu:0；同一warmup后恢复模型/AdamW/RNG，对同一步做observe off/on。[配置](../../configs/studies/third/npu-sampling.json)、[完整原始采样](../evidence/third-batch-progress-review.json)。

| 详细观测 | 同步step秒 | loss | 主线程栈样本 | Full / messages |
| --- | --- | --- | --- | --- |
| off | 19.5932 | 2.3461859 | 943 | 1,792 / 3,584 |
| on | 19.1291 | 2.3461859 | 933 | 1,792 / 3,584 |

两次终态模型和optimizer逐位一致，工作计数一致。开关指详细活动/梯度观测，两次都启用20ms栈采样，不能把它当采样器本身的开销A/B。on更快0.464秒只是一对次序固定的单步观察，没有足够重复估计观测开销或声称零开销。

栈样本中backward引擎栈占314/943和303/933，query KV read、packed KV准备等也反复出现；样本包含等待和GIL偏差，既不等于CPU自耗时，也不分解NPU内核时间。有效独立batch仍32；Full/region最大行batch224包含时间维，state batch32/sequence7，不能把224说成224个独立序列。

工作负载completed，记录区间61.32秒不含解释器/设备启动及后置Lens投影；服务含排队和收尾89.55秒。Lens verified/readback_matches，队列锁已释放，cgroup和已知后代为空。未做新的独立CPU数值参考、完整应用恢复或学习验证，原严格失败保持。

建议仅将栈分布作为后续有界分段测量的定位线索。若要优化公共执行路径，应向基座提交当前复现、实际形状和工作计数；不在实验侧复制核心，不从此配对外推HB/NPU、预训练旁路或通用加速。
