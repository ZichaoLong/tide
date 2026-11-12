# 带权区域：完整候选的批量求值与 VJP

来源：[区域分配 §2–6、附录 S.2](upstream/tide-core-5/local-semantics/weighted-message/region-allocation-learning-note.md)
及 [时间批 §2–4](upstream/tide-core-5/timed-dag-chunk-prefill-learning-note.md)。
固定上游 0770593；图/checkpoint 版本见 [semantics](semantics.md)。本页描述实现契约；
开发/固定源码资格分别见 [STATUS](STATUS.md)，实现存在不代表设备或性能已验收。

## 输入、参数、结构与解码

`SelStep` 的每行是同 sample/region/time 的完整非空候选，含每个节点的具名描述字段。
`w,mu,score` 为候选标量，适用配置另有 evidence、x、key，各自保持声明尺寸。
候选数量按实际 N 分桶，内部布局 B×N（向量字段 B×N×D）；不填充虚构候选。
节点 ID、逻辑时间和历史时钟为 int64，不通过 float 编解码。无候选不调用此层。

配置/适用条件和参数由 [scoring](weighted-scoring.md)、[sending](weighted-sending.md) 与
[allocation](weighted-allocation.md)共同定义，不改变 k/K、正支持或归一化域。
区域 Query/ScalarHead 参数独立注册；显式同 Tensor 别名只建立一个私有参数叶，
反向累加所有使用者。计算合批不合并各 sample 的状态或作用身份。
带权 history 是 singleton，实际输出记录保留本行 time；通用引擎不要求所有程序都如此。

输出为完整 `Selection(active, controls, history)`，controls 覆盖所有真实候选，active 精确取 c>0。
分配接口还返回实际乘子、KKT 分支、free 掩码和残差。解码仅解绑候选真实计算的张量，
不调用参考递归，不以有效状态投影补写记录。完整消息载荷仍由后续 Full 生成 `(c,G)`。

已有多帧组的加速器正支持、显式payload cast后的支持，以及公共批结果有限性检查，
按原device/dtype将detached bool元数据合并取回；浮点controls仍是原Tensor与原VJP根。
CPU与单帧入口保留原逐项路径，单个元数据项也直接读取。该实现不改变数值算术、
分桶、精度、稳定平票、选择域、control传输或任何mixed放置含义。
通用vector/empty controls仍按每control全元素有限性验证；原正支持cast下溢、
非有限值、dtype/device及history拒绝继续生效，全部结果验证后才采用history。
独立literal bool/None与连接零见`test_region_metadata`和standalone `weighted_region_batch`；
完整参数别名、独立控制根、版本与跨窗训练仍由下表既有检查验收。设备收益须另测。

节点ID仅由`regional_total_zero`及硬排序发送消费，按完整候选组一次构造int64张量；
连续分配、Attention和proportional不创建未使用的设备ID数组。配置温度在组内共用一个
不可求导标量。旧行接口用整字段`unbind`；新的默认区域VJP直接消费私有列，
不再创建每行分配输入或重建RegionInput，公开每行根与KKT分支不变。
Python区域VJP复用数值层已计算的active集合；发布只读view不再触发第二轮支持取回。
显式payload cast之后的支持一致性检查仍须执行，不能复用cast之前的结果代替它。
`test_region_batch_publication`检查实际ATen构造/标量读取/反向转置，覆盖2/17帧、
超过2**54的节点ID、稳定平票、部分/零control根和未使用帧。固定审阅依据见
[区域接口审阅](evidence/region-batching-audit-20261110.md)；新后端资格须按实现提交另记。

## 数值函数与反向身份

公共`allocate_columns` / `region_columns`（Python weighted包，C++
`tide/weighted_columns.h`）接收完整R×N字段，R可同时展开sample与逻辑时间。
字段维度按数学描述区分；同组N是真实候选数，不用虚构零候选填充。排序发送消费
原始int64节点ID，坐标及history由调度层拥有。返回controls整列、实际bool活动支持，
分配另保留实际乘子、int64分支、response、free和residual。仅controls可求导；
元数据和求解迭代不进入计算图。标量/逐行bound和temperature分别支持共享或独立列根。

列根是一个Tensor的根；旧逐行独立根通过外层adapter暴露，不能从零cotangent推断未使用。
`AllocationColumns.pullback` / `allocation_columns_vjp`接收独立bool行支持，返回每个输入列
的伴随或None/undefined。adapter按输出字段掩码与实际分支分组，在一次列反向后仅分发
参与行，因此caps没有score根、零分配是常量、普通分支的零贡献仍保持连接。
显式pullback只支持一阶，原输入/数值结果版本必须未改；推理张量没有可保留VJP版本时拒绝。
默认区域VJP的私有图只保存字段列、head及列输出，不再为R个事件建立数学图视图。
旧完整Selection发布仍是边界adapter，尚不能据此声称KV→区域→Full→消息整链已贯穿。

新入口由`allocation_columns.py`、`regional_columns.py`及C++对应`weighted_*_columns.cpp`
拥有；旧数值batch入口适配这些列，标量`step`与Python/C++求解仍独立。
`test_weighted_columns`覆盖手算解/共享温度梯度、独立方向导数、分支支持、版本守卫和
十二类评分/发送组合；同一配置中的KV及四类循环节点使用各自Describe构成异构区域。
现有独立根、保留窗/AdamW/checkpoint和一般PDG/HB测试继续检查默认调用。

区域的候选数分组共享调用级参数视图，实际跨device/dtype只转换一次；selection边界
按所选函数的操作数字段合批并保留真实别名，不逐字段重复复制low/high同一总量。
这只改变[区域放置的数据路径](weighted-placement.md)，不拆分完整候选、改变分配函数
或将不同节点的描述根合并。原始版本、高阶拒绝及None/连接零的检查仍适用。
C++列分配的autograd节点在整个分组未收到cotangent时，检查保存版本后返回全部undefined；
不能将未使用分组当作形状错误或补成零梯度。独立节点输出根与多候选数组合的闭式quad
见证包含这一边界。

| 定义 | Python 符号 | 独立 C++ 符号 | 验证入口 |
| --- | --- | --- | --- |
| Alloc quad/bern/ent | allocation_batch.solve / allocation_rows.allocate_batch | weighted::batch::solve / weighted::allocate_batch | test_weighted_allocation_batch、既有面枚举/Decimal 解 |
| Attention、proportional、零温/硬发送 | sending_batch | weighted::batch::attention/proportional/sorted_send | test_weighted_sending_batch、standalone weighted_region_batch |
| complete SelStep、query/context | region_batch.evaluate | weighted::batch::regions | test_weighted_region_batch |
| 独立 frame/control 根 VJP | region_vjp.evaluate | weighted::batch::region_vjp | 同上独立根/零 cotangent/别名/版本检查 |
| 在线调用、采用、history 续接 | region.evaluate_batch / blocks.evaluate_block | select_event_batches / advance_block_events | test_weighted_region_execution、通用 region 契约/调度/续接 |

Python quad 批量排序断点、熵批量受保护 Newton；C++ 独立批量有界二分及实际响应校正。
真实行的零/caps/slack/lower/upper/equality 分支分别保留；残差和正支持下溢拒绝沿用标量契约。
反向为精确解映射的解析 VJP，独立于有限迭代导数，不使用 HST。
零温沿有序候选扣减 remaining、跨行更新，保留逐次减法顺序，避免前缀和舍入改变消息存在性。
稳定平票使用原始 int64 节点 ID；Attention 零 base 的右导数与全零连接零规则保持。

区域私有图同时保存参数和各行每个描述字段的独立叶。实际分配分支与已定义的 control
cotangent 模式共同决定反向分组，不能根据梯度数值选择路径。
这区分 caps 行的无 score 根、普通行的连接零，以及完全未参与根的其他帧。
零分配输出不声明 requires_grad；原始输入版本检查及只支持一阶的拒绝边界保留。

`allocation_rows`逐行检查shape/dtype/device，数值范围和有限性由实际完整批的`solve`统一检查，
避免Python入口再次逐行同步。Python/C++均一次回读求解器实际int64分支编号，
供requires_grad、返回分支名和反向根选择复用；设备上的branch张量及其版本守卫仍保留。
这不是主机预选路由或重新求解，也不改变exact-map VJP。`test_prefill_local_batches`检查
末行非法数值/尺寸仍拒绝，以及2行到32行的标量读取次数不逐行增长；既有分支/独立根测试仍适用。

## 调度能力与边界

通用 `RegionProgram/RegionKernel` 默认不声明 joint_batch、batched_autograd 或 history_independent。
带权程序声明三项；没有能力的一般程序继续按因果次序逐帧求值。
Python 在全部 Read 已就绪且 history_independent 时可合批跨时间帧；否则只合批当前时间的
独立 sample。C++ 在所有Read/proposal已准备且区域声明history_independent/joint_batch时，也按区域合批跨时间帧；
训练另要求batched_autograd。State/Next采用仍按原因果wave推进；任何能力缺失则保留原wave内合批。
重复owner只有history_independent才允许；每组结果全部校验后才采用。
没有 CPU 真实轨迹预跑，不假定固定层数或输入；可合法退化到 B=1。
显式 parallel_regions 的既有 C++ 路径保持自身调度；该路径不据此宣称区域数值合批。

`region_steps` 继续数实际完整非空帧，`region_batch_calls/max_region_batch` 记录能力调用，
`batched_region_frames` 记录使用 VJP 能力的帧，`region_scalar_calls` 记录逐帧后备。
`region_time_batch_frames`另外记录native跨时间能力覆盖帧数。
这些计数不等同硬件 kernel 数，也不承诺加速。CPU eager 实现不构成常驻就绪/调度证明。

[878a5a4自然HB medium测量/profile](evidence/core5-natural-study-20261029.md)在NPU mixed-c实际
64个region frame使用16次调用，最大batch4；profile有厂商BatchMatMul、sort/分配及实际VJP。
全图仍在KV记录严格核对失败，这只证明该路径确实运行了区域批，不是合格吞吐结论。
profile的Query VJP/索引和提交成本明显，不能仅凭“分配是优化问题”认定其主导端到端成本。

独立证据分层：literal quad 解/Jacobian 与稳定分数发送锚定局部公式；既有面枚举和 Decimal
锚定求解；独立字段根检查连接关系；参考/Python greedy/C++ 调度比较完整记录与消息支持。
多步 AdamW、pending checkpoint、换 owner、保留图切点复用公共训练协议并检查区域能力计数。
硬函数不做普通 gradcheck；精确解 VJP、硬函数分支导数保持各自身份。
