# 常驻带权摘要与 Attention 归约资格

源码 `afe7b5e3a1d415a4cbeed67f3654bcd285f4e28d`；上游 `0770593`。[契约](../resident-weighted-pool.md)；[机器记录](core5-resident-pool-20261027.json)。

CPU14项以独立逐行FP64公式/autograd、平滑联合方向差分检查完整返回及VJP；clamp边界保留独立FP32总量，避免高精度重求和选错分支。空/全零、弱/单位/饱和权重、零权重衰减导数、None/零、translation-invariant Attention及2048次CPU适配器重放通过。
NPU完成sum/Attention×3/129维四配置、144行重放，所有私有weights/values/logits根、独立vector/total返回、保存版本和容量拒绝。
开发检查/编译/输出目录失败及相关CPU适配器问题按原树与终态保留；未改输入或放宽容差。

CPU 14 项（1.2 秒）、NPU 完整声明组件与 CUDA sm80/90/100 构建/loader通过。资格任务均clean、exit0、inactive。
NPU保留开发manifest且全摘要一致；CUDA无真机证据。

局部组件不认证完整五记忆或在线图、retained训练/checkpoint、跨卡或性能。
