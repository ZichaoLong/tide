# 带权循环记忆与固定时间卷积

固定来源 `0770593838a37c790a568226930567ed60f99b83`；共同 payload、来源分类、
聚合、描述、区域控制与广播见 [weighted-message](weighted-message.md)。本文绑定
[affine/delta](upstream/tide-core-5/local-semantics/weighted-message/recurrent-memory/affine-delta-learning-note.md)、
[GDN](upstream/tide-core-5/local-semantics/weighted-message/recurrent-memory/gated-delta-learning-note.md)、
[SSD](upstream/tide-core-5/local-semantics/weighted-message/recurrent-memory/ssd-learning-note.md)与
[短卷积](upstream/tide-core-5/local-semantics/weighted-message/recurrent-memory/causal-convolution-learning-note.md)。
它们与旧 matrix-memory/SSM profile 是不同的函数和身份。

## 公共绑定及适用配置

`bind(topology, memories=RecurrentConfig(...), regions=RegionalConfig(...))` 使用
`memory_family=affine|delta|gated_delta|ssd`。memories 可为逐节点配置元组，与 KV 混合，
内容宽度 d 必须一致；各节点自己的摘要、矩阵和 FFN 宽度可以不同。区域描述最终宽度
由区域配置确定。正式评分、证据、key与十种发送共用[评分](weighted-scoring.md)、
[发送](weighted-sending.md)绑定；各组合和后端的运行资格分别记录，不把配置不适用与待验混同。

| 字段 | 数学参数及适用域 |
| --- | --- |
| width / summary_width / key_width / value_width / ffn_width | d / ds / dk / dm / df；SSD 的 dk 即 state_dim，其余多头 dk/dm 即 head_k_dim/head_v_dim |
| memory_read | sum/state；基础循环默认 sum，GDN/SSD 默认 state；qry 仅属 KV |
| lambda_short | 摘要短期率 >0；基础循环同时是短期矩阵率；长期率恒为零 |
| heads / groups | GDN/SSD 的头数；groups 仅属 SSD，必须整除 heads；基础循环固定两种时间尺度 |
| matrix_rates | GDN/SSD 每头非负被动衰减，默认全零；不是输入门，也不是接收次数衰减 |
| conv_size / conv_stride | GDN/SSD 固定逻辑时间抽头和步长；基础循环不使用卷积 |
| epsilon / qk_epsilon / output_epsilon | 输入归一化；GDN 的 L2 常数；GDN/SSD 的输出 RMS 常数，均为正 |
| alpha_mem / alpha_ff | 非负记忆及 FFN 残差尺度 |

`RecurrentConfig` 校验尺寸、率、名称和不适用字段；所有冻结配置进入 graph profile 哈希。
不接受 KV 窗口/RoPE 字段。参数和状态默认私有；普通同形 Parameter 可显式绑定共享，
checkpoint 保留别名拓扑。源 A/B 与来源 embedding 仍按共同区域/类别共享。
参数初始化沿用固定 Generator/seed 和 std=0.15 的有限高斯分布；无隐式学习初态。

FC/HB 的 `bind_layered` 接受逐节点 KV/循环混合。KV 使用既有窗口/位置换算；
循环摘要率为 short_rate_per_position/D，GDN/SSD 的 matrix_rates 按位置率除 D，
conv_stride 绑定 D。基础循环没有卷积。直接 `bind` 的率与步长始终采用原始逻辑时间单位。

## 状态、快照与解码

数学状态初始 s=0，所有矩阵/摘要/权重为零，卷积缓存为空。
`State.slots["time"]: int64[]` 保存实际 s；`State.last_time` 是引擎最近正接收时刻，
初始 -1，发生写入后与 s 一致。observations 是精确 int64 范围正写入计数。
`State.value` 是不可学习的 d+1 宽零适配字段，不承载数学矩阵。

| 模块 | 实际浮点状态与整数元数据 |
| --- | --- |
| affine/delta | H:[2,dm,dk]，Z:[2,ds]，W:[2]，time:int64[]；顺序 short,long |
| GDN/SSD | H:[heads,dm,dk]，Z:[2,ds]，W:[2]，time:int64[]，conv_times:int64[N]，conv_values:[N,conv_width] |
| GDN conv_width | heads*(2*dk+dm)，每头依次 Q,K,V |
| SSD conv_width | heads*dm+2*groups*dk，依次全部 value、组 B、组 C；输出门/dt 不卷积 |

查询先在 int64 上减 s，再取非负差并转换浮点形成指数衰减。基础循环矩阵与摘要各有
short/long；多头矩阵使用各自被动率，摘要仍是 short/long。正接收 w=min(1,r) 后写入
矩阵、Z/W 和 s；零接收返回原对象，包括过期原始缓存、时间、观察计数，不产生遗忘步骤。
所有候选保留 proposal；c=0 仅关闭 Full/发送，不撤销本次接收更新。

Read 0/-/+ 以同一当前聚合与时间分别查询零初态、old、proposal。基础循环 state
分别读取两个矩阵；GDN/SSD 的 state 保留 short 摘要、long 使用多头矩阵查询。
Full 独立查询 proposal；从新缓存取得当前卷积不代表能把 Read^- 的矩阵替换成新矩阵。
只允许相同输入、参数和快照的计算复用，目前没有声称消除了这类重复计算。

状态字段直接编码/解码实际计算结果；[checkpoint v5](semantics.md)保存全部原始槽、
pending、ledger、history、参数别名和 optimizer。加载是显式梯度边界。
`effective` 与 `FixedTimeConv.effective` 只用于指定时刻的性质比较，不能替代原始状态解码。

## 卷积及 Full 的不同函数

固定时间卷积读 `theta-lag*stride`，缺失/负时刻为零，当前项是 bias+omega0*z，
不乘 w。正写入保留 [max(0,theta-(size-1)*stride),theta) 的原始项并追加 theta→w*z；
包含最旧端点，最多 `(size-1)*stride+1` 个槽。零写入不清理、不挪动任何槽。
实现保留所有余数类，稀疏事件不会压紧；int64 最大时刻投影不构造溢出的开区间终点。

GDN 使用 epsilon-L2 key/query、dk^(-1/2) query 缩放、输入决定的保留/Delta 门；
Full 每头 RMS 后乘 SiLU 输出门。SSD 的 B/C 不做 L2，写入系数严格为 d=w*Delta，
不是精确零阶保持系数；输出加入 D*value，先门控，再对全部内部宽度做 RMS。
两者再使用共同的输入残差与 normalized SwiGLU。门和衰减不改变消息存在性；
浮点下溢不触发人为删除原始状态。公开 VJP 为这些实值函数的普通一阶导数，
包含全部门、卷积、归一化、摘要和参数共享贡献；不借用旧 HST。
FP32 L2/scaled-RMS采用normalized解析收缩，前向逐值不变；对应公式、精度边界与
literal Jacobian见[精度契约](weighted-precision.md)，完整训练/恢复独立验收。

## 逐函数对应及独立检查

| 固定定义 | 配置与代码符号 | 预期来源、断言和测试 |
| --- | --- | --- |
| 基础 §1–2 | RecurrentConfig、RecurrentMemory.initial/effective/step、BasicMemory.update_matrix；C++ weighted_recurrent/basic.cpp | `test_recurrent_textbook_scalar_examples` 的 G=2 / 1.5；`test_basic_matrix_equations_general_width_and_vjp` 独立 A/B 公式、多维 H/Z/W、sum/state 和全部参数/输入 VJP |
| 基础 §3 | BasicMemory.mem、BasicContent | 同一独立多维公式核对 Mem 四元组、完整内容与导数；不把 sum/state 称为完整记录等价 |
| GDN §1–3 | GatedDeltaMemory.features/update_matrix/matrix_read、GatedDeltaContent；C++ weighted_gated_delta.cpp | 标量 H=.5、第二步 23/32、G=1+SiLU(1)/sqrt(17)；独立 grouped conv1d 和普通多头 Delta 循环核对每步矩阵、Mem、Full/所有根 |
| SSD §1–3 | SSDMemory.features/update_matrix/matrix_read、SSDContent；C++ weighted_ssd.cpp | 标量 H=1、第二步 1.5、直通/先门后 RMS；独立普通序列公式覆盖 4 头/2 组、卷积分支和全部根 |
| 卷积 §1–3 | ConvCache、FixedTimeConv.forward/write/effective；C++ convolve/conv_write/conv_effective | `test_weighted_convolution.py` 手算 18+10ε、零/弱槽、余数/空档、快照等式、>2^54/MAX 时钟、独立 grouped conv1d/VJP |
| 共同 Read/Next，核对 §1.4 | Describe、readout.request、一般引擎 proposal retention | `test_independent_read_snapshots_and_unselected_state_retention` 用独立摘要与 SiLU 线性构造区分三快照；完整 w/mu/score/evidence、未激活写入与 Full proposal |
| 各模块 §5 | RecurrentMemory.effective、卷积有效投影 | `test_recurrent_weak_zero_and_original_time` 在指定未来时间比较有效状态；原始时间戳不同单独断言，不冒充完整记录等价 |
| 一阶训练 | 同上全部参数/浮点状态；int64 无梯度 | `test_recurrent_local_directional_derivative` 全参数联合方向差分；独立公式逐参数 VJP；时间与 sigmoid/归一化边界按普通 ATen 约定 |
| 各模块 §4 继续 | 既有独立 reference/greedy/frontier/native/ports 与 checkpoint | `test_weighted_recurrent_execution.py` 三 Read、独立状态/输出/描述根、共享参数、每切点 pending/记录、3 步 AdamW+恢复后再更新、FC/HB 异构 |

全部测试使用源码内的确定输入、种子和显式参数；浮点容差与整数/None/路由比较沿用
[semantics](semantics.md)。普通序列对照满足连续固定 stride、单位接收、零被动矩阵衰减，
验证对应教材的 mixer 实数公式；没有执行 FLA/Mamba 上游程序，不宣称其浮点内核或整模型等价。
standalone `cpp/test/weighted_recurrent.cpp` 独立实现教材标量和局部 VJP/时钟/卷积检查。
Python/native、纯 PyTorch 和 standalone 证据分别记录。

标量序列仍提供合法递归后备；[packed联合前缀](weighted-packed.md)、
[State批量VJP](weighted-state-vjp.md)、[mixed](weighted-device-validation.md)及
[resident完整训练](resident-weighted-training.md)已经分别实现并按其声明范围验收。
这些能力不意味着所有后端使用相同批量粒度；当前开发运行与不可变资格从
STATUS和[证据索引](evidence/README.md)进入。
