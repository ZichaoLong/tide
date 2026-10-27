# 常驻带权原始状态与缓存作用

对应固定 `0770593` 的 KV §1、基础循环/GDN/SSD §1–2、固定逻辑时间卷积 §2。
[五记忆契约](weighted-recurrent.md)与[KV契约](weighted-message.md)拥有公式；本层负责
原始表示、零/正接收的结构分支、缓存写入、int64 时钟和精确解码。
矩阵/投影/门的数值求值、Mem/Describe/Full、图生命周期是后续组合层，不能由本组件资格推断。

`weighted_raw_state.h::WeightedRawLayout` 按显式 KVConfig/RecurrentConfig 和容量构造。
每个数值组具有统一 schema，每个节点保持私有行，不因参数共享合并状态。
不同配置可以组成不同组；容量限制同时保存的条目，不限制累计逻辑时间。

| buffer | 实际内容与解码 |
| --- | --- |
| integer: int64[R,4+C] | last_time、observations、cache_length、数学 time、原始逐条时间戳；KV 保留 time=0，基础循环 C=0 |
| floating: dtype[R,F] | value 适配字段，然后按显式字段 offset 布局；定长字段直接放置，变长字段分配 C 个原始槽 |
| KV 浮点字段 | value[d+1]、keys[C,d]、values[C,d]、summaries[C,ds]、weights[C]，内容不预乘权重 |
| 循环浮点字段 | value[d+1]、H[matrices,dm,dk]、Z[2,ds]、W[2]；GDN/SSD另有 conv_values[C,conv_width] |

CPU import/export 要求显式 no-grad 与 detached 数据，保留 FP64/FP32/FP16 请求和实际 dtype；
不把 int64 转为浮点，不替换精度。编码校验字段域/形状/有限性、时间顺序、计数与权重，
在分配前按 byte budget 拒绝。解码只切出实际保存的前缀和定长字段，不运行参考递归或有效投影。
未用 padding 不属于数学状态，可以是 poison；正负零和实际字段原值保留。
`copy_weighted_raw_state` 是显式边界拷贝；持久化格式沿用公共 State/checkpoint，不新建另一数学状态版本。

`append_weighted_raw_update` 接 old、int64 time[R]、mass[R]、bool present[R]、
proposal[R,F] 与 sticky error。old 浮点、mass 和 proposal 必须同时为 FP32 或同时为 FP16。
proposal 的定长字段是数值记忆模块已计算的完整 H/Z/W；
每个 ragged 字段首槽放未加权新特征；KV weights 由质量产生，不读取占位 proposal weights。
FP16 路径将实际量化输入通过设备 Cast 拓宽为 FP32，执行本层缓存作用后，结果一次舍入到
FP16 存储。卷积追加的 `w*raw` 在 FP32 计算；时钟、计数、成员判断始终使用 int64。
保存的 proposal、接收系数及全部伴随明确为 FP32。Cast 声明一阶 identity VJP，
不是浮点舍入阶梯的数学导数，也不声称复现 CPU FP16 叶的逐步舍入梯度。
新增临时缓冲计入 byte budget；FP64 只支持 codec，不扩大设备计算资格。
这是原始状态组件的精度契约；完整记忆数值、图执行和训练的 FP16 支持须另行验证。

设备上完成分支和缓存成员决定。present=false 不读取当前 time/mass/proposal；
mass≤0 原样保留原始 tuple，包括已过期缓存、数学时间和计数。
正接收令 w=min(1,mass)，检查严格递增时间与 observation 溢出；KV 保留 0≤age<T 的条目，
窗口0表示无限历史；卷积保留 age≤(size−1)*stride 的所有余数类，追加 time→w*raw。
基础循环接收数值模块的完整矩阵/摘要更新。所有候选保留正接收，不依赖发送控制。
age 在 int64 上相减；不构造 MAX+1，不用接收次数替代原始时间。

`append_weighted_raw_update_vjp` 有每行/每具名浮点字段独立 root。
old.value 始终按实际根连接；零/缺席接收只连接原字段。正接收固定字段连接 proposal，
缓存旧字段通过实际映射连接，即使全部逐出也保留 indexing 的 connected zero。
KV 新权重与卷积 w*raw 各自回传质量；w=1 使用 clamp_max 的普通一阶边界约定，
w>1 的导数为零但仍连接。此 VJP 只反向结构/缓存作用；数值 H/Z/W 和投影公式另行链式求导。
接收系数、实际 proposal、映射与旧 metadata 保存于 tape，原输入后续被改写不影响独立反向。
只声明一阶；不允许外部修改 tape。

计划先校验所有行，error24 表示状态/输入非法，25 表示实时缓存容量不足，26 表示非有限梯度。
数值输出只在计划成功后产生；`append_weighted_raw_adopt` 在所有调用者共享错误决定后提交，
失败不能改变 live owner。提交按实际 FP32/FP16 位模式复制浮点存储和 int64 元数据，
保留负零及未用 padding 的原始位。该小型设备索引/控制 kernel 不承担矩阵计算；后续矩阵层使用批量 vendor 算子。

| 验证 | 能验证的范围 |
| --- | --- |
| `test_resident_weighted_raw_update.py` | 独立原始时间列表/每字段 autograd；五 schema、弱/零/饱和接收、>2^54/MAX、过期保留、padding poison、None/零、保存来源、容量/非法拒绝；FP32/FP16提交位模式 |
| `weighted_raw_check.cpp` | 三 dtype CPU codec 精确字节往返；显式 FP32/FP16五 schema设备重放、40行完整状态与独立 FP32 字段VJP；原来源改写后的反向；容量/非有限不提交 |
| `verify_device_control.py --checks weighted-raw` | 显式 NPU/CUDA 实际组件检查入口；编译与实际设备证据独立 |

当前开发结果及固定资格按 STATUS/证据索引记录。该层不是整个记忆数值函数，也不构成完整图训练/checkpoint或常驻性能证据。

FP32固定源码资格见[cb68d91运行证据](evidence/core5-resident-raw-20261027.md)。FP32/FP16组件扩展见[2e4d445固定资格](evidence/core5-resident-half-primitives-20261028.md)。
