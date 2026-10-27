# 常驻 KV 原始时间窗口与 sum 记忆资格

源码 `fea6fa7e60ca909aaee494fc5847cda6a06ed4c1`；上游 `0770593`。[契约](../resident-weighted-kv-sum.md)；[机器记录](core5-resident-kv-sum-20261027.json)。

CPU20项由6项窗口和14项池化组成。窗口使用独立int64时间列表与公式/autograd，覆盖>2^54/MAX、未来/过期、空/零、衰减下溢后的可见性和连接保留。
NPU覆盖有限/无限两种窗口、120行重放、四Mem独立根、所有raw字段None/零与数值、原始输入改写后的保存版本反向，以及非法raw/scratch拒绝。
未将sum同型返回冒充qry、Describe或Full；完整图与训练生命周期继续单独实施。

CPU 20 项（1.95 秒）、NPU 完整声明组件与 CUDA sm80/90/100 构建/loader通过。资格任务均clean、exit0、inactive。
NPU保留开发manifest且全摘要一致；CUDA无真机证据。

局部组件不认证完整五记忆或在线图、retained训练/checkpoint、跨卡或性能。
