# 常驻 KV 原始时间窗口与 sum 记忆

固定 [KV §1.3、3.1、3.2、3.4](upstream/tide-core-5/local-semantics/weighted-message/kv-memory-learning-note.md)
对应 `weighted_kv_window.h` 和 `weighted_kv_sum.h`。本层给出完整 sum Mem 四元组，
没有将它称为 qry、Describe 或 Full；这些函数按各自契约继续接入。

`append_weighted_kv_window` 接 typed raw KV 状态、int64 times[R]、bool present[R]。
窗口先按 int64 原始写入时间判定 0<=theta-xi<T，再打包实际 K/V/p、两通道权重及长度。
保留原 raw tuple，不触发过期条目清理。保存int64 ages/mapping与实际衰减系数；
没有压紧事件后重编号、浮点时间戳或接收次数代替逻辑时间。

FP32短期权重为 w*exp(-float(theta-xi)*float(beta_short))，长期权重为w；beta必须能表示为
有限正FP32。仅在确定精确窗口成员后转数值时间。时间衰减下溢零是数值现象，原条目仍可见，
mapping、长度、原始状态与梯度连接均保留；不以数值零删除条目。位置匹配在qry/Full的独立层处理。
缺席行不读其raw/时间；实际行须满足既有raw状态域，包括原始缓存数量、时间顺序和有效浮点字段。

`append_weighted_kv_sum` 将两个通道显式打包，经[带权归约](resident-weighted-pool.md)计算，
返回memory[R,2,d_s]及total[R,2]，依次对应短/长通道；仍具有四个独立求导根。
调用者显式提供 Read+ 的新快照、Read- 的旧快照或 Read0 的绑定初态，不用None伪造初态。
本层不读取当前u、K/V值或任何参数来计算sum结果；打包器验证原始状态不构成数学梯度依赖。

`append_weighted_kv_sum_vjp` 接roots[R,4]=(M_short,M_long,W_short,W_long)，
将所有实际通道贡献汇总到私有raw summaries/weights；keys/values及adapter value保持None。
空/全部过期的indexing仍连接零，零写入权重保留sum的权重导数；统计量单独连接weights。
窗口VJP还提供K/V/p及两个权重通道各自的根，供后续qry/Full组合复用。
所有反向仅用保存的实际mapping/decay/数值，不读取被修改的原状态，也不调用参考补写。

FP32局部常驻域，设备窗口错误31、窗口VJP错误32，底层归约错误及保守组合预算单独保留。
失败结果不可采用为live状态或checkpoint。本层读取不会更新原raw owner。

| 固定定义与边界 | 独立验证入口 |
| --- | --- |
| int64窗口、短长权重、raw字段伴随 | `test_resident_weighted_kv_window.py`：独立时间列表/公式/autograd，>2^54/MAX、未来/过期、衰减下溢与连接 |
| 完整sum四元组、四个根和所有raw字段 | `weighted_kv_sum_check.cpp`：独立逐行公式/autograd，有限/无限窗口、120行重放、空/全零快照 |
| 保存版本与拒绝 | 同check修改原始raw、时间和存在标记后再反向，非法raw拒绝和scratch预算 |
| 归约数值和clamp边界 | `test_resident_weighted_pool.py`；sum/Attention身份独立 |

入口 `scripts/{build,verify}_device_control.py --checks weighted-kv-sum`。
CPU helper、实际NPU与CUDA构建/GPU实机分别记录；局部Mem不认证完整图或retained训练。

固定源码资格见[fea6fa7运行证据](evidence/core5-resident-kv-sum-20261027.md)。
