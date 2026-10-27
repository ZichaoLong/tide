# 常驻 KV 半精度 raw 更新资格

固定实现 `69bf4f0a05533ef52b282e2db73794c8eb7268ad`，上游 `0770593`。
[机器记录](core5-resident-half-kv-update-20261028.json)保存精确源码、binary/loader及运行摘要。
契约见 [KV接收更新](../resident-weighted-kv-update.md) 和 [raw状态](../resident-weighted-state.md)。

CPU相关26项通过、无skip。Standalone NPU的KV更新和raw操作各FP32/FP16通过。
每dtype KV更新72重放行、三共享bank、五独立状态根，独立逐行公式核验完整低秩/摘要参数链、
内容和旧态VJP。raw另核验五schema、40行重放。CPU codec沿用此前独立资格，本次未重跑。
来源改写后的反向、padding poison、None/零与精确int64均保留原断言。

旧态/结果实际为half；content、参数、算术、保存值与伴随为FP32，最终raw存储只舍入一次。
精确见证中FP32正接收2^-25存储为零权重，但条目、时间戳、计数及质量VJP=1保留。
有限FP32摘要溢出half时error24，已提交owner的浮点位模式和整数元组不变。
原始状态保留不以有效状态投影替代；Cast使用声明的一阶identity VJP。

物理NPU 9→npu:0，LibTorch-NPU2.10/CANN9.0；持久passed/exit0、
systemd inactive/success/0、干净源码及实际binary/loader摘要复核。
CUDA组件build/loader/help通过，80/90/100编译命令单列于机器记录。
本报告不认证完整FP16图、训练/恢复、Python-owned执行、GPU真机或性能。
