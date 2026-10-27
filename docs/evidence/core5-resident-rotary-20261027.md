# 常驻相对旋转与极大逻辑年龄数值转换资格

源码 `1eb4b72759716118a92fa70e05268f46a20f2443`；上游 `0770593`。[契约](../resident-weighted-rotary.md)；[机器记录](core5-resident-rotary-20261027.json)。

CPU13项覆盖独立dense旋转/autograd、非负int64最近偶数舍入边界、全正常FP32指数代表相位对独立math.sin/cos及原始时间窗口。
NPU旋转覆盖NoPE/部分/完整可用坐标、两次重放12行、2^54及MAX age、相位逐位检查、完整VJP与保存版本；KV sum重新验证120行及四独立根，加入真正巨大age的短期衰减回归。
巨大相位周期归约在设备执行，厂商Sin/Cos只计算归约后的角度；非有限相位明确拒绝，时间元数据保持int64。
两个先前设备开发失败保留原始树和终态；未改变测试输入或容差。

CPU 13 项（1.93 秒）、NPU 完整声明组件与 CUDA sm80/90/100 构建/loader通过。资格任务均clean、exit0、inactive。
NPU保留开发manifest且全摘要一致；CUDA无真机证据。

局部组件不认证完整五记忆或在线图、retained训练/checkpoint、跨卡或性能。
