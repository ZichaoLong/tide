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

## 数值函数与反向身份

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

## 调度能力与边界

通用 `RegionProgram/RegionKernel` 默认不声明 joint_batch、batched_autograd 或 history_independent。
带权程序声明三项；没有能力的一般程序继续按因果次序逐帧求值。
Python 在全部 Read 已就绪且 history_independent 时可合批跨时间帧；否则只合批当前时间的
独立 sample。C++ 在已认证因果 wave 内合批同区域的独立 owner，保持跨时间 wave 顺序。
没有 CPU 真实轨迹预跑，不假定固定层数或输入；可合法退化到 B=1。
显式 parallel_regions 的既有 C++ 路径保持自身调度；该路径不据此宣称区域数值合批。

`region_steps` 继续数实际完整非空帧，`region_batch_calls/max_region_batch` 记录能力调用，
`batched_region_frames` 记录使用 VJP 能力的帧，`region_scalar_calls` 记录逐帧后备。
这些计数不等同硬件 kernel 数，也不承诺加速。CPU eager 实现不构成常驻就绪/调度证明。

[e50自然HB medium测量/profile](evidence/core5-natural-study-20261029.md)在NPU mixed-c实际
64个region frame使用16次调用，最大batch4；profile有厂商BatchMatMul、sort/分配及实际VJP。
全图仍在KV记录严格核对失败，这只证明该路径确实运行了区域批，不是合格吞吐结论。
profile的Query VJP/索引和提交成本明显，不能仅凭“分配是优化问题”认定其主导端到端成本。

独立证据分层：literal quad 解/Jacobian 与稳定分数发送锚定局部公式；既有面枚举和 Decimal
锚定求解；独立字段根检查连接关系；参考/Python greedy/C++ 调度比较完整记录与消息支持。
多步 AdamW、pending checkpoint、换 owner、保留图切点复用公共训练协议并检查区域能力计数。
硬函数不做普通 gradcheck；精确解 VJP、硬函数分支导数保持各自身份。
