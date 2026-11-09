# 显式 KV 密集查询执行

`query_execution="dense-v1"` 是同一固定数学函数的另一种批量数值求值次序。
默认 `legacy`、严格容差与旧执行身份保持。调用：
`tidegraph.weighted.execution.eager_runner(..., query_execution="dense-v1")`，
或 `Native(..., packed=True, full_autograd="batched", algorithm="greedy", query_execution="dense-v1")`。
公共 `ExecutionSpec` 记录请求/生效值并产生不同执行身份；模型/图v13/checkpoint v5不变。
需要精确续接的应用必须固定执行身份，不能将跨模式继续等同同模式精确恢复。

## 适用域和独立性

当前选择器要求单payload设备、FP32/FP64、一个host worker、packed prefill，
KV/qry、cat/ind、regional_total、context/cat、quad、strict正支持。
不适用组合显式拒绝。模型参数与数学配置不改变，Python执行视图不修改原模块。
C++调用 `tide::weighted::configure_query_execution(NodeWeights&, mode)`，在placement之前选择。
C++不依赖Python；此接口实现不意味着standalone性能消费者已经接入本选项。

## 数学和批次

固定[架构§3–4](upstream/tide-core-5/local-semantics/weighted-message/architecture-learning-note.md)、
[KV§2–4](upstream/tide-core-5/local-semantics/weighted-message/kv-memory-learning-note.md)
的定义域、参数、Read初态和Full comparison绑定沿用[query](weighted-query-batch.md)。
新路径只移除外层VJP分组中KV缓存轴长度的约束；状态有无、槽名、尾维、
dtype/device/requires_grad和长/短Attention正权重分支保留。
实际KV可见长度与正支持索引仍在记忆阶段分桶，不填充或截短可见集。
密集current、Read query、score/evidence、Full query/output/FFN跨长度合批。
Read的零/非零分支一次收回detached bool元数据；浮点值不替换为host数。
这仍是主机在线调度的eager路径，不声称常驻、融合核或已经消除所有索引开销。

[原一阶边界](weighted-query-vjp.md)按实际被用字段选择行的私有叶，保留
None/连接零、共享参数所有贡献、版本检查和一阶限制。不同输入快照保持独立；
解码仅拆分候选实际张量。新布局允许有限舍入差异，不将这种差异自动认证为正确。

## 对应和验证

| 内容 | Python / C++ | 独立证据入口 |
| --- | --- | --- |
| 显式选择和域校验 | weighted/query_execution.py / weighted_programs.cpp | tests/test_weighted_dense_query.py |
| 分支和异形VJP根 | weighted/query_vjp.py / weighted_query_vjp.cpp | 同文件逐字段/私有状态/共享别名/未使用optimizer |
| Describe与Q | dense_describe.py、query_batch.py / weighted_describe_batch.cpp、weighted_query_kv.cpp | 独立scalar公式；实际投影batch断言 |
| Full与FFN | full_batch.py / weighted_full_batch.cpp | 完整返回和独立Full根 |

实现、构建、CPU、NPU、完整训练/恢复和性能资格分开记录；当前任务状态见
[STATUS](STATUS.md)，未解除限制见[ROADMAP](ROADMAP.md)。
