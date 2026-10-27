# 常驻四循环 Upd/Mem/Full 固定源码资格

源码 `6e2ecb7f169cd1f7f1040258027d882a532e72f5`；上游 `0770593`。[契约](../resident-weighted-recurrent.md)；[机器记录](core5-resident-recurrent-20261028.json)。

affine、delta、gated_delta、ssd × sum/state × Upd/Mem/Full 共24配置，使用独立教材公式核对完整字段、输入/raw/参数的一阶VJP及保存版本。含原始int64时间、巨大age、零接收原样保留、卷积、两组四头SSD、动态缺席后出现与完整容量拒绝。Describe八配置同时回归。

开发dev至dev4的失败与精确tree保留在机器记录；dev5通过不覆盖旧失败。实现使用标准vendor批量代数及显式快照/VJP；独立公式不调用候选的更新、快照或卷积。

CPU 98 项（4.47 秒）、NPU 完整声明组件与 CUDA sm80/90/100 构建/loader通过。资格任务均clean、exit0、inactive。
NPU保留开发manifest且全摘要一致；CUDA无真机证据。

局部组件不认证完整五记忆或在线图、retained训练/checkpoint、跨卡或性能。
