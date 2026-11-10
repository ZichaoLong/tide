# 带权 Upd 的批量一阶反向

固定 `0770593` 的 KV 与四种循环 Upd，数学函数见 [KV](weighted-message.md)、
[循环](weighted-recurrent.md)；数值联合计算与解码见 [weighted-packed](weighted-packed.md)。
这里扩展既有 State VJP 的模块能力，不改变图/checkpoint 身份或 Read/Full 快照。

## 输入、参数和结构分组

Python `state_vjp.py::_weights` 对声明的五种记忆捕获 legacy 字段、extra，以及 kernel
完整注册参数/缓冲区树。`weighted/parameter_tree.py::rebind` 建立私有 module 浅副本树，
重新绑定 tensor leaves，保留同一 tensor 和子 module 的共享关系，不修改活模型。
嵌套卷积权重、KV 共享 base、适配器和所有实际参数均进入版本检查与梯度所有权。
C++ 同名参数已经位于 NodeWeights::extra，由 `state_vjp.cpp` 独立捕获。

一般 `Model(state_programs=...)` 将上述记忆置于 `ClockedState` 内时，能力判定与参数
捕获同时递归到内层，`rebind` 保留整个包装树；不能仅宣布支持而漏掉内层梯度根。
包装的 `batch_autograd_key` 使用与实际 step 相同的局部时间/左状态，区分零/正接收。
这属于一般引擎的显式时钟绑定；`weighted.bind` 的教材原始逻辑时间绑定保持不变。

结构分组包括逻辑时间、左状态时钟/计数、槽形状/dtype/device/求导标志、内容来源，
以及 `batch_autograd_key` 声明的实际零/正接收分支。这个分支来自本次真实输入；
不按梯度数值分组，也不从参考轨迹取得。零接收返回原 tuple，不能和正写入的
stack 图混成一组后把无连接输入变成连接零。

五记忆可通过 `content_autograd_keys` 一次求值整块实际接收质量的正/零分支，
再供每个深度的结构分组使用，避免逐事件标量回读。该能力只描述不依赖左状态的
分支；其他合法程序、时钟包装及状态依赖分支继续使用原 `batch_autograd_key`。
它不预测候选/支持、不预跑轨迹，也不改变逐状态独立根和版本检查。

每组用批量单步结构图，序列沿每个 owner 的实际右状态依次连接；
数值仍取联合候选实际算出的结果。private graph 使用同一完整 Upd 的批量公式，
不调用 scalar Upd、参考执行器或数值重解码。输出包含 value 和所有实际 slots；
int64 时间/缓存身份始终不可求导。

卷积的单步左缓存为空时，H 更新无历史依赖；不能由新 raw 加零 mask 虚构对空
conv_values 的连接。完整新缓存仍按 scalar write 的连接语义计算。这一结构分支
不改变非空缓存中“历史槽存在但本次不命中”所拥有的连接零。

按 scalar 公式保持写入特征、GDN 控制、p_v 摘要的独立归一化分支。
把这些调用公共化会先合并 cotangent 再经过一次 RMS VJP；虽然实数导数相同，
FP32 与分别反传再相加有可见归约差异。这里保留分支求值顺序，仍批量计算每个分支，
不提升 dtype、不放宽容差、不对梯度裁剪。参数共享身份保持独立于这个计算选择。

## 返回、反向与统计

Python `_State` 和 C++ `StateVjp` 按收到的未定义 cotangent **模式**组合反向，
只请求参与行的原始输入根，同时汇总每个共享参数的全部使用者贡献。
各 slot 有独立公共根；H、summary、W、KV、conv cache 之间不以拼接消除 None/零区别。
捕获版本变更明确失败；仅支持声明的一阶 VJP，高阶请求明确拒绝。

声明能力覆盖 KV/affine/delta/gated_delta/ssd 的公共 eager CPU FP64/FP32。
packed 状态计算的 `batched_state_events` 计数增加，`semantic_state_replays` 为零；
这不意味着没有 private batch graph、没有反向成本，也不认证 Read/Full/Aggregate/Region
的批量反向。构图、packing、backward 的成本均需计入完整训练计时。
CUDA/NPU 构建/真机与常驻推进仍分别验收。

## 未返回中间状态的KV整序列反向

默认host prefill在`trace=False`且正写入已有完整KV准备值时，使用
`weighted/kv_sequence_vjp.py`与C++ `weighted_kv_sequence_vjp.cpp`：前向发布候选
实际计算的快照，反向按原始前缀/过期端点，用gather加固定维度归约一次回收各字段。
不使用冲突scatter/原子加；单次gather临时索引与数值按16MiB目标分块，
输入伴随/结果和不可拆单行另计。不同owner状态不合并，参数共享仍汇总全部使用者。
零接收保留原tuple；未来写入为None，过去但已过期的连接为零；clamp在质量1处导数为1。
所有原参数、内容和左状态的版本检查保持，一阶之外拒绝。

`trace=False`没有可供调用者取得的中间State句柄，因此允许组合其内部状态链。
公开输出、窗口右状态/pending以及输入、参数、窗口左状态根的VJP仍遵循同一函数。
`trace=True`继续使用逐状态因果链，支持对返回的每个中间状态单独求导；不能把多输出
Function中彼此独立的输出当作互有导数的中间State。未准备、其他模块及自定义程序
继续原后备。此策略不删减已有返回、不改变图/codec/数学定义；有限归约次序差异照常记录。
实际采用量为`joint_state_vjp_sequences/events`，不能当作硬件kernel数。
公共工厂`eager_runner(..., trace=False)`将相同选择传到Python及native构造器，
省略时仍返回完整事件记录；原生`Native.run`的trace策略依然由构造器拥有。

`test_kv_sequence_vjp`比较无trace的全部输出/末态及其独立根，覆盖Read^0/Read^-/Read^+、
FC/HB/反馈、稀疏/零/超过1的质量、暖左状态、共享参数、版本、SGD和精确恢复。
`test_kv_prefix_roots`继续核对有trace的中间字段根；两种可观察接口分别验收。

## 独立检查

[test_weighted_state_vjp](../tests/test_weighted_state_vjp.py)在独立 scalar schedule 上建立
完整记录预期，并比较每个浮点槽、独立输出、混合槽根和连接零目标。
四个 sample 混合零/正接收，左初态的浮点槽（含空 KV/卷积缓存）都是独立梯度根。
Python 测试禁止调用 scalar Upd；native 路径检查真实能力统计并对照所有根。
同形 key/value/summary 共享验证参数只捕获一次但累加全部使用者；
嵌套卷积参数版本变化及高阶请求拒绝分别测试。
已有公式、方向差分、多步 optimizer 和 checkpoint 测试保持独立职责。
`test_weighted_clocked_vjp.py` 覆盖五记忆与两种参考dtype的每个独立状态根，
并检查包装内参数版本、KV/共享delta三步SGD及None/零连接；它不单独认证native时钟组合。
实际通过范围以固定源码证据为准，开发检查不代替原生资格。

本轮审查修复的固定源码、独立反例与后端范围见[审查修复资格](evidence/core5-review-fixes-20261028.md)。
