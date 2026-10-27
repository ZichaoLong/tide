# 常驻 KV 内容检索 Mem

固定 [KV §2、3.1、3.3—3.4](upstream/tide-core-5/local-semantics/weighted-message/kv-memory-learning-note.md)
的 `memory_family=kv,memory_read=qry` 由 `weighted_kv_query.h` 实现。
本层接收上层明确绑定的旧/新/初态 raw 快照，不推断 Read 模式，也不更改保存条目。

`append_weighted_kv_query` 使用 [原始时间窗口](resident-weighted-kv-sum.md) 取得
两通道权重、原始 key/value 和精确 int64 age。Q、K、V 各有独立参数角色
`query,key,value[B,ds,d]`，对应 W^RQ、W^RK、W^RV；`owners[R]` 在设备选择共享 bank，
私有输入 `input[R,d]` 保持独立。Q 投影使用 N_d(u)，K/V 投影不再归一化原始条目。
实际使用的相同 bank 梯度汇总；跨角色参数别名由 canonical 参数归约层统一处理。
raw浮点存储允许显式FP32/FP16，通过窗口层拓宽后读取；input、三个参数bank、所有数值输出
及伴随始终为FP32。这里扩展的是半精度raw快照读取，不声称使用FP16查询参数或FP16 GEMM。
精度边界和Cast一阶identity VJP见[窗口契约](resident-weighted-kv-sum.md)，拓宽成本计入组合预算。

K 投影使用独立 Read 频率的 [相对旋转](resident-weighted-rotary.md)，与 Q 做批量
点积再除 sqrt(ds)。短/长通道共用该匹配值，分别调用带权 Attention，返回
`memory[R,2,ds]` 和 `total[R,2]`；解码为 M_short、M_long、W_short、W_long 四个独立根。
Q/K/V 投影、匹配和 Attention 使用 vendor BMM、sum、softmax、multiply；
设备元数据 kernel 只决定实际输入/参数使用及结构连接。

没有正的长期权重时不读取 owner 或 u；不存在的行也不读取 raw。
数值衰减下溢不改变原始条目或窗口成员；短通道空 Attention 返回常值零，
其 total 仍保留写入权重的 connected zero。非有限参数只在实际使用时拒绝。

`append_weighted_kv_query_vjp` 对四个输出独立收根；两个向量通道对共同匹配和
value 的贡献求和，再经保存的旋转/投影/窗口 tape 回传。只取统计量不连接 Q/K/V
参数、u 或 raw key/value；raw 摘要和适配 value 不被 qry 读取，保持 None。
所有 tape 保存实际计算版本，原输入、参数和 raw 改写后反向仍用前向版本。
反向是声明的一阶数学 VJP，不对整数、配置或依赖结构求导。

显式 FP32数值/FP32或FP16 raw存储的accelerator路径，scratch 单独预算并在提交前拒绝。空记忆、没有查询、
零梯度与不连接分别表示。完整 Describe/Full、在线图、训练及 checkpoint 另行组合。

| 对应 | 独立检查 |
| --- | --- |
| 实际参数/输入使用与四根连接 | `test_resident_weighted_kv_query.py`：poison 缺席/零权重、非法 owner/content；独立 eager Attention 的 None/零依赖，含短通道下溢 |
| 四个 Mem 返回、所有参数/输入/raw 一阶反向 | `weighted_kv_query_check.cpp`：每raw dtype在实际量化raw上逐条独立FP32公式/autograd；两个窗口、三个 bank、120重放行、四独立根、保存版本、极大 age、容量失败 |

入口 `scripts/{build,verify}_device_control.py --checks weighted-kv-query`；
CPU 元数据、NPU 实机、CUDA 构建和 CUDA 真机证据分别记录。

固定源码资格见[8b1be80运行证据](evidence/core5-resident-kv-query-20261027.md)。
新增FP16 raw读取开发中，设备资格尚未完成。
