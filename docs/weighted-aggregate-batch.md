# 带权 Aggregate 的批量计算与一阶 VJP

固定 `0770593` [共同局部架构 §2](upstream/tide-core-5/local-semantics/weighted-message/architecture-learning-note.md)。
参数/源分类/载荷与聚合表示见 [weighted-message](weighted-message.md)；
`SrcClass` 只决定参数，每个物理原子仍独立参与一次。

输入为同一节点/程序的一组已完整、非空的实际纤维；保留每条边/端口的来源身份与顺序。
每个原子一行编码 `(m,z)`，真实 fiber offsets 与源类别是精确整数元数据。
没有填充原子、SourceDomain 同类唯一性假设或数值轨迹预跑。
不同 fiber 可有不同原子数、源类别、缺失端口和零/负数值强度。

Python `weighted/aggregate_batch.py::evaluate` 与独立 C++
`weighted_aggregate.cpp::aggregate_batch` 完成：

1. 按实际 SrcClass 合批源投影 `B SiLU(A N(z)+embedding)`，按原始物理行回填。
   class 的 A/B/embedding 共享规则不变；不先按 class 汇总原子再执行非线性。
2. 每条原子取正部质量并只乘一次到变换后的内容，保留 `(m,z)` 与 `m*z` 的区别。
3. 按实际 fiber 长度合批完整质量和内容求和；正总量按其本身归一化，零总量返回零内容。
4. 解码候选实际计算的 `(r,u)`；贡献 map 保持为空。外层原始 atom/slot 记录保持独立。

alpha=0 时不执行源变换。空 request 列表返回空 batch；单个空纤维仍不属于 Aggregate 定义域。
重复 class 的物理行索引互不冲突，index_copy 不合并原子；源参数的显式别名不改变私有状态。
按列批投影与按真实长度的归约保持 scalar 公式顺序；沿用既有浮点容差和严格离散比较。

`WeightedAggregate.joint_batch` / `AggregateKernel::joint_batch` 声明实际数值批量能力，
`aggregate_scalar_fallback_steps` 为零。每类投影和每种 fiber 长度有实际算子工作，
不能将一次 API batch 计数当成单个硬件 kernel。packed SourceBatch 传输是另一能力。

`aggregate_autograd="batched"` 通过 Python `weighted/aggregate_vjp.py` 和独立 C++
`weighted_aggregate_vjp.cpp` 提供一阶批量 VJP。按实际源类别顺序、原子元数据与
零/正总量分组，每组构造一次真实批量计算图；没有 scalar Aggregate 重放。
零总量分支只计算质量，内容为常量零，源变换参数保持无连接。
正总量分支中零强度原子仍执行公式，保留应有的连接零。

私有模块/NodeWeights 副本捕获所有源变换参数并保持别名。每个物理原子是单独使用，
同一输入 Tensor 复用时由外层 autograd 累加全部贡献。来源 scale 不是这条公式的参数，
不为它虚构连接。反向仅选择有已定义 cotangent 的 fiber 输入；同组其他行不产生伪零。
参数汇总全部使用行的贡献；只声明一阶 VJP，原始张量版本变化与高阶请求明确拒绝。

公共 `AggregateProgram.batched_autograd` / C++ 同名能力允许显式声明实现。
默认 replay 仍可配置，批量策略记 `batched_aggregate_events`，不增加语义重放计数。
输入结构分组在 eager 主机侧完成，标准 ATen 数值计算不等同于常驻就绪/推进或 GPU 资格。

[Python 独立检查](../tests/test_weighted_aggregate_batch.py)：多类逐原子手算、
同类重复、稀疏/不同长度、零总量、alpha=0，并禁止 numeric batch 调用 scalar step。
两 sample 的反馈 PDG/非对齐 DAG 对照独立调度的完整记录、控制、pending、输入/参数根与切点继续。
[standalone C++](../cpp/test/weighted_aggregate.cpp)另用显式 SiLU/归一化手算及重复原子的质量导数。
已有发送/评分/记忆的完整训练与恢复测试继续回归；开发与不可变资格见 STATUS/evidence。

[批量 VJP 检查](../tests/test_weighted_aggregate_vjp.py)含独立加权均值 Jacobian、
含源变换的方向差分、独立 fiber/内容/连接零根、混合 requires_grad、参数别名、
重复输入、版本/高阶拒绝。整图分别覆盖 replay/batched 策略；standalone 独立检查相同边界。

[训练恢复](../tests/test_weighted_aggregate_training.py)显式选择 batched 策略，在 HB 与反馈 PDG
比较三步 AdamW 全部参数/梯度/optimizer slots，并从含 pending 的切面保存、换 owner、
恢复后继续训练。Python 调度与 native 分别测试；不以默认 replay 的训练结果代替本策略。

## 归一化 VJP 的稳定表达

Architecture §2 的前向仍是原始 `u=sum(m_i*z_i)/M`，不改数值投影、活动集合或输入 dtype。
`weighted.mean.weighted_mean` / C++ `weighted_mean` 先计算 `z_i-u`，使用
`bar_m_i = dot(bar_u, z_i-u)/M`、`bar_z_i = (m_i/M)*bar_u`；总量输出的导数另行相加。
单一正质量原子的均值在数学上与质量无关，直接返回该质量根的连接零。零总量由既有
Aggregate 分支处理，不引入源 transform 依赖。只声明一阶，保留版本检查。

该表达避免先分别计算两个较大的商导数再相减。新增双样本 SSD/KV/affine 反馈 PDG
在旧表达下的 FP32 质量根严格失败已保留；没有改变输入、dtype、前向公式或容差。
`test_weighted_mean` 用单原子不变性和独立标量方向差分检查，既有 literal mean Jacobian
与共享/独立根测试继续适用。此数值改进不保证任意浮点条件数下的位级一致。

## 区域窗口的物理来源张量

默认列路径在来源变换后直接交接 `SourceBlocks`：唯一计算结果及 int64 的物理使用→所有者映射。
Python `aggregate_layout.SourceIncidence`、独立 C++ `weighted_aggregate_layout`
再以实际 strength/value Tensor 身份及完整行域建立来源 bank。相同值、SrcClass、底层
data pointer 或仅相同参数均不许可合并；同一个数值所有者的重复边仍以整数重数参与。
此处共享计算与存储，不改变私有状态或消息存在性。

记接收者数为 G、唯一来源所有者数为 S、组内实际行域并集长度为 R。
int64 关联矩阵 `C[G,S]` 保存物理重数；来源质量和内容只打包为 `m[S,R]`、
`z[S,R,D]`，缺失行以独立映射标记。质量取正部之后，前向计算

`M = C @ m`，`U = (C @ flatten(m*z)) / M`，

其中零总量的内容为常量零；每个接收者只发布其真实行域。总量、内容各自保持独立根。
不再复制或保存 `[receiver,row,physical-source,D]` 内容，也不先为每条物理使用创建
Tensor 视图再把它们拼回来。来源 MLP 的真实唯一所有者仍各有自己的输入/参数根。

转置沿同一来源 bank 和关联布局进行。内容伴随为
`bar_z = m * Cᵀ @ (bar_U/M)`；质量输出的伴随为 `Cᵀ @ bar_M`。
均值的质量项逐接收者以 `C[g,s] dot(bar_U[g], z[s]-U[g])/M[g]` 累加，
每次仅保留一个 `[S,R,D]` 中心化临时块；不沿 B×L 循环，也不改用两个大商分别求导后相减。
这种实现选择保留原中心化数值稳定性，反向仍有按接收者的阶段循环，不能据此宣称整个归约
只有两次算子调用。某真实行只有一个来源所有者时（含该所有者的重复别名），其均值对质量
严格不变，返回连接零；判定使用实际行存在性，不能把零质量的不同来源删掉。

零总量内容无连接，显式零 cotangent 仍连接。未观察接收者不向无关来源制造零根；
原始版本均受守卫，只声明一阶 VJP。浮点转置不使用冲突 atomic scatter：
来源内容伴随用矩阵收缩，原始行恢复用唯一逆映射 gather。
不同预算组之间仅在真正的组边界发布稳定顺序的共享使用根；组内按 int64 重数直接归并。
半精度、单一接收者和超出额外 packing 预算的请求沿已有公式后备；不会截断原子或静默换 dtype。

预算按唯一来源数、公共行域、输出、一个中心化临时块、伴随及 int64 索引/关联矩阵重新计算。
ragged 并集可能比单接收者行域大，因此预算切分随真实表示改变；连续存储本身不是合批证据。
当前预算规划对每个pending前缀重建来源布局：第g次扫描此前全部物理来源/行域，并构造
g×S_g关联元数据；总主机工作为这些前缀成本之和，来源/行域扫描可达G²量级。
S_g随g增长时，重复构造稠密关联还会放大成本。LH176每区域8节点限制了本代表的G，
但更宽区域仍合法，不能据此声称宽组规划已优化。后继可缓存每条来源的身份/行域，
增量计算准入尺寸，仅在最终组确定时构造一次关联；此次存储改造不将该待办冒充已实现。
`packing/bucket/aggregate_tensor` 记录实际计算组和接收者数；
`aggregate_source_bank` 记录该组实际唯一来源数，不把重数减少算作消息消失。
矩阵求和和别名重数求值会改变浮点结合顺序，既有严格阈值不修改；有限差异、设备执行与
精确恢复仍独立验收。此边界不更改图/checkpoint版本，checkpoint 解码仍来自实际状态。

独立检查：
[`test_aggregate_tensor`](../tests/test_aggregate_tensor.py)逐原子公式、ragged/重复边、
None/零、冻结、原始版本和方向差分；
[`test_aggregate_source_bank`](../tests/test_aggregate_source_bank.py)唯一 bank 实际保存形状、
不同逻辑行域/相等独立根、来源投影、预算切分及大公共偏移的质量不变性。
后者对敏感质量根使用精确存储输入值上的 FP64 字面商公式作为独立 oracle，不改变候选 dtype。
独立 C++ [检查](../cpp/test/weighted_aggregate_tensor.cpp)
（`tidegraph-aggregate-tensor-check` / CTest `weighted-aggregate-tensor`）
分别验证来源所有权、int64 关联、完整字段和根。一般端口/非空初态、训练与恢复继续独立检查。
