# 常驻 KV Full

固定 [KV §2、4.1—4.2](upstream/tide-core-5/local-semantics/weighted-message/kv-memory-learning-note.md)
由 `weighted_kv_full.h` 实现。Full 接上层实际选择得到的 `active[R]`，读取明确传入的
候选状态 raw 快照。返回 `output[R,d]` 为未乘发送强度的 G；逐边/端口 `(c,G)` 由图层展开。
该函数不把 Read 使用的旧态/初态误当成 Full 快照。

配置沿用 `KVConfig` 的 d、ds、ffn_width、heads、window、epsilon、position_scale、
各头 full_frequencies/distance_slopes 与 alpha_att/alpha_ff。Q 参数为 `[B,d,d]`，
out/gate/up/down 的对应及尺寸见[公共残差](resident-weighted-residual.md)。
Read 的频率及 Q/K/V 参数不参与 Full；两者仍可经明确参数别名共享同形根。

`append_weighted_kv_full` 先按 int64 原始时间取窗口，再按实际正权重决定 query
输入/参数使用。Q=N_d(u) 的线性投影，K/V 使用保存的未旋转原始条目；
各头独立配置相对旋转和距离偏置，匹配除 sqrt(d/heads)，正权重上稳定 softmax，
读出乘 min(1,W)。各头读出按原坐标拼接后接 out 与 SwiGLU 残差。
NoPE 可用空配置，部分/完整旋转分别按头宽校验。距离偏置也使用显式精确age到FP32转换。

head 切分/合并、query 与矩阵投影、匹配、softmax、残差均使用批量设备算子；
主机只在构建时遍历固定头配置，不读取运行时选择或时间。空/全零记忆的 query 参数
保持 None，out/FFN 参数仍按实际残差表达式连接。未激活行不读取 poison raw/input/owner。

`append_weighted_kv_full_vjp` 从每行 G 的独立根回传：逐头 Attention/旋转的 key、value、
写入权重贡献汇总到实际 raw 槽；query 与残差共同回传 u；所有共享参数使用者求和。
raw 摘要和适配 value 不被 Full 读取，保持 None；数值零不充当缺席标记。
所有数值和连接使用保存 tape，原参数/输入/状态改写不改变该次反向。
checkpoint 的精确状态仍用原始raw codec，本函数不以有效状态投影代替状态。

显式accelerator；raw存储FP32/FP16，输入、参数、窗口与Full结果G、保存值和所有伴随为FP32。
half raw经设备Cast精确拓宽，VJP声明identity；这是实际量化原始状态上的FP32 Full，
不是half参数/Full输出路径。独立scratch准入覆盖拓宽，组合末尾还核验实际保留张量量。
非法原始状态、非有限相位/数值及容量不足
明确拒绝。完整在线图的选择、队列、输出提交、retained训练和checkpoint仍需上层验收。

`weighted_kv_full_check.cpp` 使用逐条独立CPU公式、dense旋转矩阵和autograd，
覆盖有限/无限窗口×NoPE/两种头旋转共四配置、96重放行、两个头/三个共享bank，
全部五参数角色、u/raw五字段、None/零、极大age、保存版本及容量拒绝。
两dtype各自独立检查，公式直接消费实际量化raw；不将FP32资格自动扩大到FP16。
公共残差检查另覆盖四组零/非零尺度和完整反向。
CPU逻辑时间距离/选择使用检查在 `test_resident_weighted_kv_full.py`；
入口 `scripts/{build,verify}_device_control.py --checks weighted-residual weighted-kv-full`。

固定源码资格见[7d86d9e运行证据](evidence/core5-resident-kv-full-20261027.md)。
