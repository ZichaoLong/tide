# 带权 eager 局部函数的 FP16 契约

固定数学定义从 [带权绑定](weighted-message.md)、[循环](weighted-recurrent.md) 和
[配置对应](semantic-map.md)进入；实际资格由 [STATUS](STATUS.md) 与证据索引给出。
下面定义eager数值策略；[CPU资格](evidence/core5-half-20261027.md)与
[有限NPU资格](evidence/core5-half-device-20261027.md)分别记录通过范围及严格失败。
这些结果不表示全部FP16配置都通过，也不替代不同算术边界的[常驻精度契约](resident-weighted-training.md)。

`bind(..., dtype=torch.float16)` 用同一 seeded FP32 参数量化初始化。
payload、参数和浮点原始状态保持 FP16；时间、计数、KV 与卷积戳保持 int64。
区域求解器要求显式 `ExecutionPlacement(scoring_dtype="float32")`（CPU 也可 FP64）；
不把未声明的 half 分配求解器自动替换为其他算法。Read 采用实际快照的 typed 副本，
发送强度转回 payload dtype 时丢失正支持仍明确拒绝。

Architecture §1 RMS、GDN §1 epsilon-L2 和 GDN/SSD §3 scaled-RMS 使用原始 half 输入，
在 FP32 计算范数、缩放及一阶收缩后返回 half；FP32/FP64 公式与运算顺序不变。
设 `n=x/d`，RMS 的输入 VJP 是 `(b-n*mean(b*n))/d`，L2 将 mean 替换为 sum；
有逐通道 rho 时 b 替换为 b*rho，rho 的 VJP 为 `b*n` 并归约广播使用者。
浮点 cast 使用声明的一阶 identity VJP，不对 rounding 阶梯做 gradcheck。
原始参数版本检查、None/连接零与各行根保持；不支持二阶反向。

衰减与位置量先做原始 int64 时间差，再转 FP32；half 不承载逻辑时间或频率乘积。
`numeric_time.py` / C++ `weighted_detail.h` 的 passive decay 在 FP32 求指数后返回
payload dtype，零衰减率不会因大于65504的间隔变成 `0*inf`。
KV 旋转在 FP32 求相位/三角函数后返回 half；半精度 KV Attention 的 softmax、
加权归约与 centered VJP 在 FP32 运算，公开结果与叶梯度仍为原 dtype。
这些内部算术选择不改变窗口、原始时间戳、可见条目或 checkpoint 原始布局。
`test_weighted_half_clocks.py` 包含超过2^54的时间戳、100000步间隔、零速率、
手算旋转和五记忆完整图/VJP；纯Python与native的实际结果分别记录。

Python `weighted/normalization.py` 与独立 C++ `weighted_normalization.cpp` 对应这些公式；
`test_weighted_half_normalization.py` 和 C++ literal Jacobian 覆盖零向量、小量、一般向量、
广播 scale、版本检查和指定一阶边界。完整图的 scalar/native/packed 差异另测，
局部公式通过不能替代优化器与续接证据。

训练使用 [FP32 master/静态 loss scale/checkpoint 契约](precision.md)，独立 leaf 的梯度
仍可能在 half 边界舍入。当前 GDN 某些小梯度经默认 AdamW 放大，严格多步比较仍失败；
这两个 Python/native 案例在测试入口具名 `xfail(strict=True)`，不计作通过；原始失败日志保留。
使用相同图、输入与容差的显式 SGD 配置单独检查四步与恢复。默认 AdamW 不被自动替换，
实际 dtype、算术策略和 optimizer 都由调用者明确选择。
不能靠放宽容差或修改输入抹掉这一见证。上述CPU资格覆盖长逻辑时间、局部公式与四步训练恢复。
设备范围见上列NPU报告；[standalone master/schema2](evidence/core5-native-master-20261027.md)与
[resident半精度全图/训练](evidence/core5-resident-half-training-20261028.md)各有独立固定源码证据，
不能从core-3或其他算术路径推定。常驻报告不解除本页eager GDN的严格失败。

[首次分歧与纯Torch异步隔离](evidence/core5-numerical-isolation-20261028.md)记录近零梯度被AdamW放大，以及当前Torch-NPU/CANN双设备反向的小反例。严格失败与显式blocking通过分别保留。

[D64/D2048增量诊断](evidence/core5-performance-transfer-20261028.md)另记录FP32完整记录、参数梯度及SGD参数的严格差异；
同源FP64两步通过只适用于该CPU诊断。quad共同评分偏移的独立反例在FP64也拒绝，因此更宽dtype不是通用修复。

## FP32 GDN 的补充严格见证

固定1f1ec57扩大CPU资格时，原`test_recurrent_multistep_optimizer_and_restore`
的HB/GDN FP32、默认AdamW在Python/native两条候选均失败；改动前1850434同样复现，
并非本次共享准备顺序修复引入。前向记录及梯度通过原容差，第三次更新
`nodes.5.kernel.conv.weight[0,2]`相差1.411885e-6，超过原参数容差。
第一步两侧梯度约−2.00200247e-8和−2.00375894e-8；默认AdamW的epsilon为1e-8，
梯度的微小差异会影响更新。独立80位Decimal使用实际梯度及旧动量计算，复现更新差异的放大；
这不证明所有上游梯度差异的来源均已定位，也不把FP64或其他optimizer的通过回填给此配置。
原测试和断言保持失败，没有改为xfail或放宽容差。详见
[稳定共享/精度证据](evidence/core5-stable-preparation-precision-20261028.md)。
