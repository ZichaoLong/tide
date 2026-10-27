# 带权消息与 KV 公共绑定

固定来源 `0770593838a37c790a568226930567ed60f99b83`；共同定义见
[架构](upstream/tide-core-5/local-semantics/weighted-message/architecture-learning-note.md)、
[KV](upstream/tide-core-5/local-semantics/weighted-message/kv-memory-learning-note.md)和
[区域分配](upstream/tide-core-5/local-semantics/weighted-message/region-allocation-learning-note.md)。
以下是本地明确的函数与表示绑定；[循环/短卷积](weighted-recurrent.md)给出其他记忆的接口、状态与逐函数对应，剩余候选仍见 ROADMAP。

## 调用与配置

公共入口是 `tidegraph.weighted.bind(topology, memories=KVConfig(...),
regions=RegionalConfig(...), ...)`，返回 `Binding(graph, model, ...)`。
输入 topology 的节点/区域归属、边、时延、物理端口被保留，原有局部程序被本支线替换。
模型接入现有 reference、streaming、frontier、greedy 与 ports SettleGraph；
PDG 包含正时延反馈，TimedDAG 可以非对齐；一般端口日程不依赖 FC/HB。

首组合为 `kv/sum/+ / cat/ind / regional_total/context/cat/quad`。
同一 KV 增量也提供 `memory_read=qry`、`read_mode=0/-/+`、fixed 总量控制、
NoPE/各头部分或完整 RoPE、独立 Read 旋转及距离偏置。默认旋转为空、距离偏置为零。
fixed 要显式给 total_fraction，region_evidence 取 None；context 不接受固定比例。
[分配](weighted-allocation.md)提供 quad/bern/ent，ent 另给固定 ent_alpha∈[1,2]。
[其他发送规则](weighted-sending.md)接入十种正式发送函数、固定/上下文节点增益及各自适用字段。
[评分/证据/key](weighted-scoring.md)进一步给出 cat/match、ind/att、cat/grouped 和 content_stats/content 的独立绑定。
配置检查不将未知、不适用或尚未实现的选项换成其他函数。

`KVConfig` 给出内容宽度 d、摘要宽度 ds、适配秩、头数、FFN 宽度、窗口、
归一化 epsilon、短期衰减率和两个残差尺度；`RegionalConfig` 给出 dr/dreg、
gamma_max、k 和正分配温度。全部有界性与形状条件在构造时校验。
窗口 0 表示无穷，其余为原始逻辑时间长度；位置比例也以逻辑时间为单位。
位置差先做 int64 减法，再转换浮点，避免大时钟相邻事件被浮点合并。
所有冻结配置与来源类别映射通过带哈希的 profile 进入图身份。

默认初始化为指定 seed 的独立 CPU Generator，有限高斯参数标准差 0.15；
函数验证使用显式参数覆盖，不依赖随机值实现预期。无隐式可训练初态：KV 初态为空。
KV 区域基座共享；节点适配、摘要、Read 与 Full 参数私有；源 A/B 在区域共享，
embedding 在区域/来源类别共享。参数使用普通 Parameter 身份，允许显式同形绑定，
checkpoint 检查别名图；共享参数不合并私有状态。
[来源绑定](weighted-source-binding.md)另提供 node/source/adapted 模式、显式低秩因子和类别宽度；
当前没有声称消除了共享计算。

`bind_layered` 另外落实 FC/HB 实例 §1/§4.4：来源类别为父节点身份，入口为单独 in 类；
以输入位置窗口 H 和短期率 beta_bar 绑定 T=D*H、beta=beta_bar/D、Pos=(theta-rank)/D。
`layered_inputs` 在各入口广播同一单位强度表示；`readout` 从实际逐端口记录计算
Σm+z/max(1,Σm+)，缺席与内容相消保持区别。W_out 与残差/任务头由下游显式组合。
固定位置 t 的终端时间 Dt+L 小于任一后续位置的输入时间；正时延与正典控制/状态因果性
给出保守依赖集合“全部入口的 ≤t 前缀”，包括控制和消息存在性。实例测试同时比较输出值与存在。
该结构论证不使用某次实际轨迹倒推依赖，不宣称覆盖任意应用的输出前缀策略。

## 表示、状态和解码

| 数学对象 | 实际表示与解码 |
| --- | --- |
| payload `(m,z)` | `Payload`，传输向量 `[m,z]` 宽 d+1；`numerical()` 单独得到 `(m_+,m_+z)` |
| 聚合 `(r,u)` | `Received`，独立类型，向量 `[r,u]`；不是数值投影后的 payload |
| KV 原始状态 | `times: int64[N]`、`keys/values: [N,d]`、`summaries: [N,ds]`、`weights: [N]` |
| State.value | 宽 d+1 的固定零，仅适配通用记录，不承载数学记忆；状态校验拒绝非零 |
| 状态时钟/计数 | last_time 是最近正接收写入时间；observations 是累计正写入数，Python int64 域；不用于窗口或位置 |
| Describe | regional/ind 为 w、mu、score、evidence，节点增益/ind 为前三项；att 另含 x/key:[dsel]；见评分契约 |
| Full | 返回 `[c,G]`，同一载荷广播所有逐边/端口槽，不按出度分强度；各槽身份仍分开 |
| 区域历史 | 数学单点；History.last_time 只记录最后处理时间，不新增学习状态 |

正接收先按逻辑窗口清理再写一条；K/V 和摘要不预乘权重。
零接收返回原状态对象，包括尚未清理的过期项。Read 在查询时遮罩过期项；
`Read^0` 显式生成绑定的空初态，`-` 读旧状态，`+` 读 proposal。
Full 始终读 proposal，所有候选的 Next 都返回 proposal，未激活也保留写入。
这些是局部绑定，通用引擎的其他合法程序仍可使用自己的规则。

物理来源各占一个 SourceDomain 槽；SrcClass 只是参数分类，相同类别的每个原子均参与。
本绑定拒绝旧输入 origin 适配器，直接使用一般物理端口。
通用引擎的输入/聚合/边/输出尺度固定为不可训练的 1，并在执行及 checkpoint 预检中校验；
发送强度只在本支线的聚合或显式数值投影中乘一次。
legacy Emit 接口必须取 hard；此处仅表示不插入额外 Emit 变换，发送反向由所选函数单独声明。

checkpoint v5 保存实际原始状态与 typed tensors、全部 pending、ledger、共享参数和 optimizer；
加载开始新梯度段。有效状态投影不用于 checkpoint 解码，解码不调用参考递归。

## 定义与验收对应

| 固定条目 | 配置/符号 | 独立检查 |
| --- | --- | --- |
| 架构 §1–2 | `Payload/Received`、`WeightedAggregate`、`SourceTransform` | `test_srcclass_multiple_physical_atoms_and_single_strength_projection`：三原子手算、独立 VJP |
| KV §1–3 | `KVConfig/KVMemory.initial,step,mem,current` | `test_kv_write_mass_window_zero_and_hand_full`：未预乘内容、原始时间窗口、零接收原状态；位置测试覆盖 >2**54 |
| 架构 §3、区域 §3.1/3.4 | `Describe.features/step` | `test_read_snapshot_and_full_are_separate`；具名根在执行检查中分别求导 |
| 区域 §2–3.5 | `RegionalTotal.total/step`、`ScalarHead` | `test_textbook_evidence_to_total_complete_run`：独立表达式覆盖内容→证据→头→B→c |
| 区域 §4.1、§6 | `allocate_quad` | [quad/bern/ent 的分配契约与独立求解/方向差分](weighted-allocation.md) |
| KV §4、架构 §4 | `KVContent/WeightedFull` | 手算加权 attention、独立旋转公式、载荷强度仅一次 |
| FC/HB 拓扑 §1–4 | `fully_connected/hb_lattice` | 教材 22 区域/1408 节点/3584 边，HB 同秩；完整运行保留每端口 |
| 架构 §7、Batch/正典继续 | 既有独立调度、checkpoint v5 | `test_weighted_execution.py`：FC/HB、非对齐 DAG、反馈 PDG，全记录/根、多步 AdamW、每切点 pending/恢复 |

局部独立数学检查与调度比较分别承担公式和执行顺序验收。完整图的参数方向差分限定
在不跨支持分支的 FP64 小扰动；硬排序只在明确固定顺序的光滑分支作方向检查，不跨离散边界。
本阶段开发测试域为 CPU FP64/FP32 的 Python、Python/native 与 standalone C++；
[KV 不可变资格](evidence/core5-weighted-kv-20261027.md)独立记录；批量 VJP、mixed/resident 与硬件资格继续独立推进。
通用 prefill 可使用声明的完整驱动序列递归能力，缺少 joint 能力时合法退化并保留计数；
当前标量循环不称为完成 packed 执行。浮点指数下溢可影响数值记忆，原始状态/时间不据此删除；
quad 正支持按实际求得控制判定，严格路由对照不能用数值容差放宽。

C++ 公共局部接口位于 `cpp/include/tide/weighted.h`，分别由 weighted_kv.cpp、
weighted_programs.cpp、weighted_allocation.cpp 实现；native.py 只传 Tensor 和固定配置。
`node_binding/region_binding` 返回实际安装函数的配置副本、来源分类与 Aggregate 系数；
它们不运行张量、不解析 profile 名称、不从参数形状猜配置。节点检查同时验证
State/Describe/Full 配置及 Aggregate epsilon 一致；未安装或混装函数提前拒绝。
Read 的初态/旧态/候选态选择仍属于 Graph::Region。Python 传输适配器为
`weighted_node_binding/weighted_region_binding`；`test_weighted_binding_inspection.py`
逐字段核验五族及副本隔离，C++ `check_weighted` 独立拒绝不一致的 Full。
C++ 使用独立有界二分与响应段校正，Python 使用断点扫描；两者都声明精确解映射 VJP。
standalone `cpp/test/weighted.cpp` 有自己的手算、共享根导数、大时钟及零事件原状态检查。
原生定向开发检查已通过；不可变资格与证据状态见 STATUS。
缓存初态 roots、不同内容/描述宽度、多头及每头独立部分 RoPE 有专门 native 检查。

安装配置检查接口固定资格见 [ce21a83 证据](evidence/core5-weighted-binding-20261028.md)。
