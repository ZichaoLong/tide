# 常驻完整 Describe 资格

源码 `cd576d24434c5daaa21f57d316a93b9bf1c91f0a`；上游 `0770593`。[契约](../resident-weighted-describe.md)；[机器记录](core5-resident-describe-20261027.json)。

八组适用配置涵盖 cat/match、cat/grouped/fixed evidence、content/content_stats key，以及固定、可训练logit和上下文增益。
独立逐行直接公式与autograd检查所有字段、私有输入、共享参数、None/零、保存版本、动态缺席→出现、容量与非法输入拒绝。
clamp上界相等处用非零cotangent核验声明导数1。静态局部算式组合不替代在线图调度。

CPU 38 项（3.86 秒）、NPU 完整声明组件与 CUDA sm80/90/100 构建/loader通过。资格任务均clean、exit0、inactive。
NPU保留开发manifest且全摘要一致；CUDA无真机证据。

局部组件不认证完整五记忆或在线图、retained训练/checkpoint、跨卡或性能。
