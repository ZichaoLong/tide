# 常驻固定逻辑时间短卷积

固定 `0770593` [短卷积 §2–3](upstream/tide-core-5/local-semantics/weighted-message/recurrent-memory/causal-convolution-learning-note.md)。
本数值组件读取[原始状态布局](resident-weighted-state.md)，服务 GDN/SSD 的投影后特征；
不把时间槽压紧为接收次数，当前项始终是未加权 z，即使 proposal 缓存已有 theta→w*z。

`weighted_convolution.h::append_weighted_convolution` 接 GDN/SSD raw schema、
int64 time[R]、current[R,F]、int64 owners[R] 和参数 bank：weight[B,K,F]、bias[B,F]。
F 是实际 conv_width；K 和 stride 显式固定，并校验与 raw schema 的 horizon 一致。
owners=-1 表示缺席行，其他行必须指向合法 bank。参数共享不合并当前输入或私有缓存。
当前常驻数值组件限定 FP32；CPU codec 的其他 dtype 不构成本组件低精度资格。

设备计划从实际缓存查找 theta−lag*stride，保留全部原始余数类；负目标和缺失槽为零。
int64 先完成比较与减法，不构造 MAX+1，也不将绝对时间转浮点。
得到 taps[R,K,F] 与实际映射，再由厂商 index_select、multiply、sum、add 完成共享参数读取和整批卷积。
缓存寻址需要设备索引 kernel；乘法和归约没有降成逐元素主机执行。
按全部 tap 归约后加 bias 是实数公式的重组，浮点容差不认证逐位数值顺序一致。

`append_weighted_convolution_vjp` 用保存的实际 tap 值、参数行、时间映射和 metadata。
厂商批量乘法产生 tap/weight 偏导，索引层回写各私有缓存与当前特征，按输入行顺序归约共享 bank。
独立行 roots 决定 None/零；K=1 或原 cache 长度0时缓存根为 None；
K>1 且 cache 非空时，即使当前所有历史 tap 缺失，仍保留矩阵匹配产生的 connected zero。
参数 bank 的 weight/bias 都按其实际使用者保留连接，缺席 poison 不读。
未匹配历史 tap 的临时乘积没有实际缓存梯度用途；该临时值即使溢出也不伪造非有限返回，
仍检查所有真正返回的当前项、缓存项和参数梯度，避免把缺失项当成零值消息。
该组件是普通一阶 VJP，不借用 HST，不声明高阶。原输入/参数被改写后，独立 reverse 仍读取保存版本。

完整旧/新快照查询的结果一致性与梯度连接是不同断言：新缓存的当前条目不参与正 lag，
所以弱写入不改变当前卷积数值；新缓存图可能仍带 connected zero。
standalone 检查将 raw update → 新快照卷积 → 卷积 VJP → raw update VJP 串联，
检查质量根保留且数值为零，不通过丢根简化。

输入/cache/参数/output 非法设置 error27，非有限梯度设置 error28；scratch 在提交前拒绝。
调用方在共享错误决定后采用 raw state，检查覆盖失败不提交。
缓存容量、编解码、零接收保留与正接收写入由原始状态层负责。

| 对应 | 独立检查 |
| --- | --- |
| 原始时间 sparse tap 和未加权当前项 | `test_resident_weighted_convolution.py` 教材 18+10ε、同时间 proposal 弱槽、余数与空档、>2^54/MAX |
| 完整数值与 VJP | 同测试使用独立 dense logical-time signal + grouped conv1d，K=1/2/4、stride=1/3、共享 bank、私有缓存、None/零 |
| 实际设备与保存版本 | `weighted_convolution_check.cpp` 两种K、20行重放、三个bank、完整输入/cache/参数VJP、动态owner/缓存、原来源改写后的反向 |
| 状态组合与拒绝 | 同check串联raw update与SSD新/旧快照、独立根，非有限事务及容量拒绝 |

入口 `verify_device_control.py --checks weighted-convolution --device npu:0`。
本组件不认证完整 GDN/SSD、Describe、在线图生命周期或性能；固定源码 CPU/NPU 与 CUDA 构建/真机证据分别记录。
