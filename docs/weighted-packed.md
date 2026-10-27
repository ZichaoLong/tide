# 带权记忆的精确联合求值

固定 `0770593` [时间批 §2–3](upstream/tide-core-5/timed-dag-chunk-prefill-learning-note.md)、
[KV](upstream/tide-core-5/local-semantics/weighted-message/kv-memory-learning-note.md) 与
[循环 §4](upstream/tide-core-5/local-semantics/weighted-message/recurrent-memory/affine-delta-learning-note.md)。
局部函数、参数和状态布局不变，分别见 [KV](weighted-message.md)、[循环/卷积](weighted-recurrent.md)。

## 输入、实现与解码

输入是调度器已证明完整的同节点/程序驱动序列及每个 sample 的私有左状态。
`PackedSequence` 保存连续聚合内容、实际非空事件 offsets、owner、原始 int64 时间与内容视图。
同一程序/参数的各 sample 可以不同长度、不同时间与不同正接收模式；不存在填充候选。
任何图的 Full 依赖尚未解决时，外层在线调度不得把未知驱动放入此接口。
未采用 CPU 真实轨迹预跑；PDG 反馈可按合法前沿自然退化为单作用。

Python `weighted/packing.py::Writes` 与 C++ `weighted_packed.cpp::Writes` 将正接收写入
编码为 ragged segments。零接收事件仍在完整返回中，保持上一个原始元组：不写时间戳、
不加观察数、不清理过期 KV/卷积缓存。差分先在 int64 求值，再转换为浮点衰减参数。
参数所有权不变；打包只合并同参数程序，节点/sample 状态保持私有。

| 定义与计算 | Python / C++ | 实际计算与返回 |
| --- | --- | --- |
| KV 写入投影 | kv_packed.py / weighted_kv_packed.cpp | 正接收内容一次归一化、批量 KV/summary 投影；每 owner 将左缓存和实际新项连接一次，再按每个原始时间选取实际前缀/窗口 |
| affine/delta §4.2 系数 | recurrent_packed.py / weighted_recurrent_packed.cpp | 批量 key/value/门和实际时间衰减；affine 取标量系数，delta 取右乘矩阵系数 |
| GDN/SSD §2/§4 | gated_packed.py / weighted_gated_packed.cpp | 批量输入投影、固定时间卷积、头/组展开、保留及写入系数，系数生成不读取未知 H |
| 固定时间卷积 | convolution_packed.py / weighted_gated_packed.cpp::convolve | int64 原始时间 searchsorted/gather；历史取实际带权 raw 写入，当前 tap 使用未加权 raw；缓存按逻辑 horizon 解码 |
| 仿射前缀 §4.1 | packing.py::matrix_scan、scan.py / weighted_packed.cpp::matrix_scan、kernel.cpp::affine_scan | 按正写入长度分组，时间×owner×head 批量复合；矩阵顺序为 (A1 A2, B1 A2+B2) |
| 原始返回与右边界 | Writes.decode | 直接选取已计算的 H/Z/W、KV 项与卷积项，恢复时间/观察数/原样零接收；不调用 scalar Upd 或参考递归补写 |

KV 所有正写入共用一个投影 batch；循环对相同正写入长度的 owner 分组扫描。
sequence、独立 step batch 和 packed_sequence 进入这些算子。stats 中 joint 能力消除
`state_scalar_sequence_steps`/`state_scalar_batch_steps`；calls 计联合分组，
不等同于 ATen 算子数、硬件内核数或调度阶段数。实际窗口/快照索引仍有 owner/输出循环。

## 一阶反向、数值与适用域

本阶段数值由联合候选计算；引擎仍用已有逐局部语义图绑定一阶反向，
`semantic_state_replays` 如实计入。不能把联合数值通过称为批量 VJP 已完成。
所有公共独立根、None/连接零、参数共享贡献保持；int64 槽不求导。
联合算子本身是普通 ATen 表达式，但直接对拼接输入求导有单个 packed Tensor 的普通连接语义。
公共引擎通过明确绑定保存每条原始输入的连接身份。

批量投影采用权重矩阵乘输入列批，与 scalar 列向量公式保持同一朝向；
开发诊断定位并修复了行批求值在近零 FP32 梯度上的差异。
前缀重括号与矩阵批量化有普通浮点归约差异，沿用原有 FP64/FP32 容差，
路由/支持/平票与 int64 元数据仍严格比较。没有精度升级或路由宽容。
未定义新的 checkpoint/图版本；保存的仍是候选实际原始状态。
本实现是 eager host 管理 packing/元数据；不构成设备常驻就绪或推进证明。
完整 Read/Full/区域打包、批量 VJP、容量切分与跨卡需各自验收。

## 独立验证

[test_weighted_packed](../tests/test_weighted_packed.py)核验非交换矩阵的显式多项式前缀与 VJP、
基础循环手算、GDN/SSD 独立 conv1d 公式、稀疏原始时间、超过 2^54 的 int64 时钟、
不同正写入模式和 raw 零接收。禁止调用 scalar Upd 的测试约束实际联合路径。
整图使用独立 scalar schedule 比较完整记录、各状态槽/独立输出根、各 sample 私有输入、
跨切点 VJP；原有反馈、非对齐图、优化器/checkpoint 测试继续作为回归。
[standalone KV](../cpp/test/weighted.cpp) 与 [循环](../cpp/test/weighted_recurrent.cpp)
分别调用 C++ packed 接口并检查手算/所有 slots/原始时间。
开发与固定源码资格在 STATUS 和 evidence 分开记录；没有实测的后端不自动通过。
