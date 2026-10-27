# 常驻 KV 完整内容与公共残差资格

源码 `386970a9dcd0b826e390be020c2a86ea0c7a3437`；上游 `0770593`。[契约](../resident-weighted-kv-full.md)；[机器记录](core5-resident-kv-full-20261027.json)。

CPU50项覆盖原始时间/实际选择、旋转、窗口、池化、RMS和原有KV手算公式。
NPU公共残差四组零/非零尺度48行，全部私有根及四个共享参数角色；KV Full有限/无限窗口×NoPE/双头不同旋转四配置96行，完整未乘c的G、全部五参数角色/u/raw字段与None/零。
独立公式/autograd、极大age距离、空/零记忆、缺席poison、重复重放、保存前向版本、非法输入和容量拒绝均通过。
图层逐边/端口载荷、在线就绪/推进与完整训练续接继续独立组合，局部Full不替代全图资格。

CPU 50 项（4.28 秒）、NPU 完整声明组件与 CUDA sm80/90/100 构建/loader通过。资格任务均clean、exit0、inactive。
NPU保留开发manifest且全摘要一致；CUDA无真机证据。

局部组件不认证完整五记忆或在线图、retained训练/checkpoint、跨卡或性能。
