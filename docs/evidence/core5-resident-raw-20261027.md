# 常驻原始状态与缓存作用资格

源码 `cb68d9187999edc07559b73eb35d53a45575d23f`，上游 `0770593`。[契约](../resident-weighted-state.md)；[机器记录](core5-resident-raw-20261027.json)。

CPU 14 项独立时间列表/每字段 autograd 检查通过；独立 C++ codec 对五种记忆 schema、FP64/FP32/FP16 做精确字节往返，保留 int64、正负零与实际缓存前缀。
NPU FP32 40 行重放检查完整 raw tuple、全部字段/质量 VJP、None/零、弱/饱和/零/缺席接收、动态长度、原来源改写后的保存版本反向。
大于 2^54 与 int64 MAX 的时间、过期但零接收保留、实时容量与计数拒绝、失败不提交通过。

CPU pytest 1.26 秒。两个资格任务均固定 clean 源码、exit0、inactive。
NPU 复用原开发构建并核验全部组件/core/binary摘要，保留原 manifest；CUDA sm80/90/100 新构建与loader通过，无GPU真机证据。

此前两次失败分别是检查参考图过早释放、标量直接 view(Byte)；保留原冻结树、任务和日志，未改设备数值实现或容差。

本项只认证原始布局与缓存作用。H/Z/W、投影及门的数值记忆公式由后续组件提供，完整 resident 图、retained 训练、checkpoint、跨卡和性能继续单独验收。
