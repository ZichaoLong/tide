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

## 固定资格与接入

[TL-REQ-004交付及完整命令](evidence/dense-query-tl-req-004-20261110.md)对应实现75a5027、
诊断驱动26ca1b6。CPU FP64/FP32、NPU FP32的Python/native分别检查；
CPU FP32独立evidence根两个字段约1.044e-6的原严格失败保留。
小型FC/HB/反馈PDG完整训练、pending/cut与新owner恢复，以及四个安装消费者的新进程恢复通过；
原FC16/B8/L128只作有限成本观察，不能外推大图严格数值、离散或恢复资格。

- 独立局部公式/字段根与FP64方向差分：`tests/test_weighted_dense_query.py`。
- 独立完整调度、三步AdamW、保留图/精确cut恢复：`tests/test_weighted_dense_training.py`。
- 已安装公共API、新进程3步vs2+1：[`dense_query.py`](../examples/consumer_core5/dense_query.py)。
- 固定源码门禁：`bash scripts/qualify_dense_query.sh NEW_OUTPUT cpu|npu MATCHING_BUILD`。
  显式数值探索收集完整比较后保留xfail；None/离散、非有限及精确恢复错误仍使流程失败。
- 局部舍入诊断：`python scripts/diagnose_dense_rounding.py`；这是同一FP32输入/权重提升FP64，
  不是更换输入分布或默认执行精度。
- 原任务有限成本：`scripts/benchmark_dense_query.py`从固定下游Git对象读原模型/head/loss/输入生成器，
  单配置一预热一步；设备profile单列。它不是下游训练接口。

实验通过公共`eager_runner`、`Native`或配置`execution.query_execution`显式选择；
旧study benchmark命令生成器与standalone benchmark CLI目前拒绝该选项。
本次不改变应用checkpoint周期、存储介质或完整返回范围。跨模式切换应创建新执行身份，
先做下游任务验收；同模式小图精确恢复不代表跨模式恢复或原语言训练已通过。
