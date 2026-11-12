# LH直接公式与Tide列路径的静态结构审阅

审阅源码固定 **fb83ee91cc34ba49bf00465916d6e7e7a7eacdec**；文件hash和逐区域展开在[机器记录](lh-static-structure-20261112.json)。
本报告回答源码如何组织计算，不把算子trace作为算法的定义。后继优化另记提交，不能将本报告当当前源码的永久描述。

## 材料与计数单位

LH176：22区域×8节点、1058内部边、8输入/8输出端口，B4/L16/d64/H1/FFN256。
7区域KV qry、15区域KV sum；14区域att、8区域ind；全部cat/context/cat/regional_total/quad。
来源采用region参数共享，alpha=1、hidden32，无来源低秩因子。不同节点的KV参数与私有状态仍独立。

计数中的“矩阵乘”指一次源码调用的`linear`、`mm`、`matmul`或`bmm`，包括批维；
不包括transpose/reshape/stack/逐元素操作，不把一次matmul等同一个设备kernel。
归一化另外计数，FP32直接公式一次含square/mean/add/rsqrt/multiply五个数值操作。
没有用总代码行数代替工作量。实际稀疏支持、分组和反向根会影响Tide调用数，因此给出表达式和上界。

## 入口、返回与默认链

直接入口是[scripts/lh_tensor_baseline.py](../../scripts/lh_tensor_baseline.py)的`StaticKV(binding,spec)(x,support,start,left,records)`。
构造只编译静态拓扑/来源绑定；输入为`x[B,L,d]`、物理输入存在性`[P,B,L]`及可选非空raw bank。
返回端口G/c、隐藏投影、raw末态和可选区域观察。每区域内部保留`[N,B,L,...]`，
Full计算所有密集行后mask；它提供声明的数值投影，不承诺生产路径全部None/零连接或通用checkpoint。

公共入口是`SettleColumnPlan.run_ports(layout,ports,left)`；native进入
[weighted_settle_columns.cpp](../../cpp/src/weighted_settle_columns.cpp)的`columns::settle`。
输入端口为实际行ID、c列、G列，返回逐节点typed状态、描述/控制、逐物理边/端口消息列。
默认`EagerExecutor.run_ports`先校验真实Settle证书/能力，再调用**同一公共列执行器**，
额外导入/导出完整Continuation和记录生效配置。它并没有在LH每个token重新调用通用PDG调度器。
一般PDG或不匹配模块才进入已声明的record后备；本材料实跑是`settle-columns-v1`。

## 循环实际在哪里

| 边界 | 直接公式 | fb83 native公共列实现 |
| --- | --- | --- |
| 外层推进 | 22次区域循环；同秩独立区域仍合法顺序提交 | 同样22次区域循环 |
| Aggregate | 58次前驱区域块迭代收集数据；按静态来源编号及参数别名去重后394个编译槽合成区域张量 | 区域内176次接收节点迭代，再遍历各节点物理来源；最多1058+8=1066次来源变换 |
| 来源变换 | 每区域一次整个来源轴RMS、两次整块投影和一次SiLU；相同输入/参数的发送结果可复用 | 每个实际非空来源各做RMS、两次投影、SiLU；即使另一接收者使用同一输入/同一参数也再做 |
| Upd | 每区域node×B×L整块；4次投影 | 逐node准备/检查/构造bank；投影按维度和预算分组，归一化与raw重排仍有node循环 |
| Read/Describe | 每区域整体node轴、一次描述字段计算 | `read_many`先分相同配置和ragged Q/K尺寸；组内投影/Attention合批，但归一化、统计/字段拼接仍逐node |
| 区域分配 | 每区域一次`[B,L,N]`求解；缺席容量为零 | 每区域按实际候选数分组；至多8组/区域。完整控制根经组装配/转置返回node列 |
| Full | 每区域一次node×B×L Attention/FFN，最后mask | 只算活动行；但按实际Q/K尺寸分组，组内小算术仍逐node，部分组只有1个node |
| 返回/续接 | 整区域raw bank，逐端口组装；通用解码在显式decode外部 | 每node/field公开列，另有每sample的Continuation实际raw/history/ledger解码 |
| B/L循环 | 数值主链无B或L的逐事件Python循环；构造/显式decode另列 | 核心矩阵不是逐token执行；B/L上的时钟、ragged索引和codec循环仍存在，不能统称每token一次完整模块 |

`StaticKV._sources`中的物理边/节点循环发生在构造，不属于每次forward。
参数列表/stack仍有成本；本配置来源无adapted因子，不能将低秩因子每次矩阵重建成本隐藏到这里。
对前驱区域收集的58次循环中，首区域为外部输入，其他57次各收集G/c两列。
不压紧逻辑时间；稀疏支持/实际状态由数据决定。

**394是当前直接实现的编译槽数，不是最少的实际输入/参数计算对。**
`_sources`的键包含`kind`和来源编号，因此首区域仍编译8个不同输入端口槽。
本材料的`_aggregate`把同一个`x`广播给这8槽，A/B与class0 embedding也完全共享；
各端口不同的support只在质量归约中使用，不进入来源MLP。若进一步复用这一处已知的
相同`F(x)`，来源变换槽可由394减至387，8条物理输入及各自质量贡献仍须保留。
fb83实际仍计算394槽，下面的546次矩阵调用也是其现行实现计数；387只是这一额外复用
成立时的条件推导，未实现或测量，也不是全局最小工作量证明。直接公式本身仍有可消除工作。

## 不运行模型即可展开的主要计算

本材料直接公式的forward，按上述矩阵乘口径：

| 计算 | 表达式 | 次数 |
| --- | --- | ---: |
| 来源MLP | 22×2 | 44 |
| incidence对质量和加权内容的聚合 | 22×2 | 44 |
| KV写入base、adapter两级、summary | 22×4 | 88 |
| Read sum/qry及current summary | 15×3 + 7×7 | 94 |
| cat score/evidence、att current/key | 22×2 + 14×2 | 72 |
| 区域context head、att query两级 | 22 + 14×2 | 50 |
| Full的Q/QK/PV/out/gate/up/down | 22×7 | 154 |
| 合计 | 上述完整forward矩阵乘 | **546** |

直接公式共有**132次RMS归一化**（22×6）。另外有mask/softmax/旋转/归约/stack/gather及求解器；
546不是全部ATen调用、更不是546个kernel。此版本quad独立求解器每区域有限区间枚举，
不含不定次while迭代；是否执行非饱和子集由实际caps/total决定。FP32/FP64或不同backend不改变该源码矩阵乘展开。

Tide仅来源MLP就为 **2A次矩阵乘、A次RMS/SiLU**，其中A是这次窗口非空物理来源列数，
`0 <= A <= 1066`；上界为**2132次矩阵乘**。相同参数不自动触发计算共享，这是可去掉的重复工作，
不是core-5语义必需。合批后需保持每条物理边仍参与聚合一次，并反向累加所有使用者。

其余Tide矩阵批数由prepare/read/full分组决定，不能静态偷用一次真实轨迹当常量：
相容的完整区域可一组，最坏每node一组。区域求解器的调用数为各区域实际候选数类别数之和，
上界22×8=176，直接公式固定22。字段pullback还按独立根连接/分支分组；未连接和connected-zero不能合并。
这些是明确的循环/分支公式，不是“通用算法天然慢”的解释。

## 哪些差额应消除，哪些是取舍

来源变换重复、逐node小归一化/字段拼列是实现选择，可以通过区域数值块与明确块VJP优化。
相同形状但不同参数可以批量矩阵乘；计算复用只允许相同实际输入、参数及版本。
ragged活动行路径减少Full算术，却可能造成159个小Full组；密集pad路径使用更多算术却只需22个大组。
这应是能力与成本策略，而不该成为各拓扑各写一个执行器。生产路径需独立根、实际raw恢复和完整端口支持，
但这些要求不必强迫每字段每node都建立细碎autograd视图。

运行观察用于核对该解释：同材料packing为Read90组/176node，Full159组/171活动node，区域94组。
CPU仪器统计default/direct为976052/94543次ATen；此数含前反向及检查，不与546个forward矩阵乘混算。
这些数据支持存在来源、形状分组、字段/伴随成本，但不把全部时间差都归到一个函数。
已决定实施来源块和精确计算复用；其余保留具体边界和下一可验替代，不无限重复profile。

复现材料生成与三层入口见[LH基线](../lh-tensor-baseline.md)。固定报告依赖源码/生成器与小型审阅数据，
无需旧trace目录。方法已纳入[优化skill的加法/减法对照](../../.agents/skills/optimize-tide-execution/references/additive-comparisons.md)。
