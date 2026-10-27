# 常驻 KV 相对位置旋转

固定 [KV §2、3.3、4.1](upstream/tide-core-5/local-semantics/weighted-message/kv-memory-learning-note.md)
由 `weighted_rotary.h` 实现相对旋转 Rot(-age/position_scale)。age为上层原始逻辑时间精确
int64差值；共同origin在相对位置中消去，不把绝对大时间先转浮点再相减。

输入FP32 keys[R,C,d]、int64 ages[R,C]及固定频率列表，最多floor(d/2)项；
旋转最后2*pairs个坐标，前面坐标原样保留。空列表为NoPE恒等，不计算位置相位。
频率和position_scale显式转换为有限FP32，scale必须为正；实际相位非有限明确拒绝。
元数据保存int64，浮点仅用于本层数值相位，不回写时间/窗口/计数。

非负int64 age以显式IEEE最近偶数舍入转换为FP32，避免设备scalar转换丢失高位；
相同转换用于窗口数值衰减，原始时间mask仍完全使用int64。
设备phase打包后使用192位2/pi常数的Payne–Hanek周期归约，把有限FP32相位归约到
约[-pi/4,pi/4]，再调用厂商Sin/Cos并按象限还原。CUDA对应FP32 sinf/cosf，
NPU对应aclnnSin/aclnnCos。此路径不在主机计算真实时间相位。
周期归约解决本机NPU厂商Sin/Cos在巨大有限相位下已观察到的能力缺口；
相位本身、数值转换、归约和厂商算子各自可检查，非有限相位仍明确拒绝。
`append_weighted_rotary_vjp` 使用保存的实际cos/sin执行转置线性映射，独立行根false时
不读取poison cotangent；不对固定配置或整数时间求导。原keys/ages改写不影响反向。

前向错误33、反向非有限34，scratch容量单独拒绝。前向有限检查和容量准入不认证所有
设备精度范围；CPU helper、NPU实际Sin/Cos、CUDA构建/GPU真机分别报告。完整qry/Full
还需其查询投影、加权归约、残差与独立参数根，此层不替代这些函数。

| 对应与边界 | 独立验证 |
| --- | --- |
| int64差值转相位、NoPE/部分旋转、保存线性伴随 | `test_resident_weighted_rotary.py`：独立dense旋转矩阵/autograd，2^24、2^54、MAX int64 age与溢出拒绝 |
| 全有限FP32周期归约 | CPU各正常指数代表值/正负相位对独立math.sin/cos，绝对误差2e-7；设备检查实际巨大age |
| 厂商Sin/Cos及完整返回/VJP | `weighted_rotary_check.cpp`：三个配置、两次重放、相同FP32相位的独立CPU矩阵与autograd、修改来源后反向 |

入口 `scripts/{build,verify}_device_control.py --checks weighted-rotary`。

固定源码资格见[1eb4b72运行证据](evidence/core5-resident-rotary-20261027.md)。
