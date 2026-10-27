# 常驻 KV 接收更新资格

源码 `681a7e9f712922d13c615baf1837516cc0060dbe`；上游 `0770593`。[契约](../resident-weighted-kv-update.md)；[机器记录](core5-resident-kv-update-20261027.json)。

CPU24项覆盖正接收/缺席poison、独立字段根和合并溢出，以及已声明raw/RMS边界。
NPU以独立逐行公式/autograd验证72行重放、三个共享bank、五个独立字段根和联合根；完整旧state/content/base/U/V/summary梯度、None/零、弱/零/负/饱和/缺席接收、原始时间窗口、>2^54/MAX int64、保存版本及失败不提交。
零接收不读取非法owner或poison内容；只读weights或summary不会连接无关参数。scratch容量拒绝通过。

CPU 24 项（2.63 秒）、NPU 完整声明组件与 CUDA sm80/90/100 构建/loader通过。资格任务均clean、exit0、inactive。
NPU保留开发manifest且全摘要一致；CUDA无真机证据。

局部组件不认证完整五记忆或在线图、retained训练/checkpoint、跨卡或性能。
