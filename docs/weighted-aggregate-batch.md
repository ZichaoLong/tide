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
