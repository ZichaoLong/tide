# 常驻共享投影资格

源码 `2af401dc0b2912fcf09faac9ab1f31a43aec489a`；上游 `0770593`。[契约](../resident-weighted-projection.md)；[机器记录](core5-resident-projection-20261027.json)。

CPU RMS helper以独立FP64 autograd和联合方向差分验证零输入、127维尾宽、None poison与非有限拒绝。
NPU覆盖有/无RMS及有/无bias四配置、48行重放、三个共享bank、完整输入/矩阵/bias VJP、None/零、动态owner及原来源改写后的保存版本反向。容量拒绝通过。
本层只认证共享投影及其一阶伴随，不替代实际KV/循环记忆与完整图验收。

CPU 4 项（1.03 秒）、NPU 完整声明组件与 CUDA sm80/90/100 构建/loader通过。资格任务均clean、exit0、inactive。
NPU保留开发manifest且全摘要一致；CUDA无真机证据。

局部组件不认证完整五记忆或在线图、retained训练/checkpoint、跨卡或性能。
