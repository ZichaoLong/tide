# KV查询默认策略与显式复现路径

公共`eager_runner`、`Native`和`ExecutionSpec`默认请求`query_execution="auto"`。
满足能力的KV FP32/FP64、strict正支持、单payload设备、单host worker、packed prefill使用
`packed-v2`；其他模块、FP16、streaming、scalar、resident与多设备保持原路径。
选择依据是模块能力和执行配置，不依据拓扑名称、节点数或参数量阈值。
Python/native和standalone benchmark使用同一选择原则；实际资格须分别查看证据。

`legacy`、`dense-v1`、`packed-v2`可显式选择，用于诊断和历史复现。
显式优化请求不满足适用域时拒绝，不静默替换。`legacy`只指定查询实现；
跨时间区域批是另一个通用能力改进，整版本历史复现还须固定原源码。

## 数学、布局与一阶边界

固定[架构§3–4](upstream/tide-core-5/local-semantics/weighted-message/architecture-learning-note.md)、
[KV§2–4](upstream/tide-core-5/local-semantics/weighted-message/kv-memory-learning-note.md)
的参数、Read初态/旧态和Full proposal快照沿用[query契约](weighted-query-batch.md)。
图v13/checkpoint v5及数学版本不变；各路径改变浮点运算组织，不改变严格容差。

| 路径 | 内部组织 | 适用域 |
| --- | --- | --- |
| legacy | 按可见长度/正支持逐行索引，再合批 | 原模块能力 |
| dense-v1 | Q、current、score/evidence、Full/FFN跨长度密集投影；仍有逐行可见性/正支持索引 | KV/qry、cat/ind、regional_total、context/cat、quad、strict |
| packed-v2 | 按原始缓存长度分桶；有真实共享存储时复用[唯一KV数值bank](weighted-snapshot-bank.md)，否则拼接实际槽；批量gather、int64时间可见性、正支持mask和Attention | KV sum/qry、strict；各合法Read/评分/发送条件继续由原配置校验 |

每桶按32MiB估计目标限制合并行数，不复制模型矩阵。**单行可能超过目标**，该值不是
总workspace或进程峰值保证；KV/激活/私有反向图仍需另行预算。没有截短缓存、压紧时间或改变窗口。
可见性在int64相减后判断；过期/填充值在投影前置零，真实零权重不变成消息。
Attention中心取第一个正支持值，避免被忽略条目导致相消或`0*inf`；全空返回保留断连。

`packed-v2`的cat evidence现在使用共享矩阵GEMM和逐行归一化，保持每行独立输入与字段根。
历史逐行GEMV曾用于规避近径向VJP的舍入差异；有界同材料FP64核对后不再以此阻塞合批。
原容差保持：CPU FP32指定evidence根约1.044e-6超差作为具名严格xfail保留，其他根照常核对。
逐行BMM候选也重现该差异，没有继续复制权重批；新组织不提升全模型精度。
原[独立行/字段VJP](weighted-query-vjp.md)保留None/连接零、共享别名全部贡献、版本守卫及一阶限制；
按根是否定义来分组，不按梯度数值选择。解码只拆分实际计算张量，不调用参考补写。

新路径仍是主机在线推进的eager执行。合法批次来自当前就绪前沿，没有CPU预跑轨迹；
[区域跨时间批](weighted-region-batch.md)与查询批独立。阶段数、矩阵批长、设备kernel数分别观察，
不是常驻资格，也不是把全图融合成几个算子的承诺。

## 配置、身份与续接

```python
execute = eager_runner(binding, "native", placement, native_library=build)
# 默认auto；诊断时可显式 query_execution="legacy" / "dense-v1" / "packed-v2"
```

公共`ExecutionSpec.record()`记录请求、策略版本和执行身份；不适用的执行配置记录生效legacy。
实际结果统计`query_legacy_nodes/query_dense-v1_nodes/query_packed-v2_nodes`报告模块选择。
`Native.query_execution`还给出逐节点结果。Python浅执行视图保持原参数owner，不修改原模型。
C++核心入口`configure_query_execution`不依赖Python；standalone benchmark接受`--query-execution`。

数学/图身份不因执行路径变化；应用锁必须固定源码和执行身份。旧材料显式legacy保持原身份；
省略选项的新请求采用auto的新身份。checkpoint能解码不等于跨模式逐值续接，旧实验升级需新运行验收。
同源码/配置/设备的精确恢复独立验证，不能从小图推断原语言任务或所有规模。

## 代码、测试与证据

| 边界 | Python / C++ | 独立检查 |
| --- | --- | --- |
| 策略/适用域 | weighted/query_execution.py、planning/execution.py / weighted_programs.cpp | test_weighted_query_layout、test_weighted_dense_query |
| 原始槽/时间/索引 | query_layout.py / weighted_query_layout.cpp | 大于2**54的时间、异形/空/过期状态、分桶目标 |
| Mem/Full | query_packed.py / weighted_query_packed.cpp | 独立scalar函数逐字段、全部私有根、sum/qry与正支持中心反例 |
| 描述与Q/FFN | dense_describe.py、read_batch.py、full_batch.py / weighted_describe_batch.cpp、weighted_full_batch.cpp | 原evidence舍入见证，别名、版本、一阶及未使用optimizer |
| 完整执行 | eager_runner、Native、standalone Eager | FC/HB/PDG三步AdamW、保留图cut、pending、新owner/新进程恢复；一般端口/历史程序回归 |

[代表回归入口](prefill-regression.md)规定改动触发哪些检查及原FC16成本命令；
[性能经验](performance-lessons.md)记录可复用方案及其边界。
[原dense-v1固定资格](evidence/dense-query-tl-req-004-20261110.md)保留原数值失败和实际成本，
不能重新标记为packed-v2通过。[默认prefill固定资格和成本](evidence/prefill-default-20261110.md)
分列新默认的CPU/NPU三客户端、小图训练/恢复、原任务有限成本及剩余碎片化。
当前工作见[STATUS](STATUS.md)，待办仅见[ROADMAP](ROADMAP.md)。

`test_prefill_local_batches`以独立普通Torch公式核对FP64完整/部分/重复行evidence根和共享参数，
并检查一次Describe的三个共享投影提交；`diagnose_dense_rounding.py --query-execution packed-v2`
复现同FP32材料提升FP64的有限诊断。字段检查与分配分支的批量元数据见
[结构化记录](structured-records.md)和[区域批](weighted-region-batch.md)。

安装消费者：`examples/consumer_core5/dense_query.py --query-execution auto ...`；
旧v1复现必须显式指定`dense-v1`。原任务成本驱动`benchmark_dense_query.py`从固定下游Git对象读模型/head/loss，
默认预热后恢复相同初始权重及optimizer，记录测量前hash；`--warmup-state continue`保留旧计时流程。
两者都是有限成本观察；未检查的数值、离散与恢复不能从耗时推出通过。

固定提交安装资格入口：`bash scripts/qualify_dense_query.sh NEW_OUTPUT cpu|npu MATCHING_BUILD auto`。
省略末项继续检查显式dense-v1；auto门禁另包括新布局、一般调度及配置回归。

独立LibTorch安装消费者：先用`prepare_core5_study.py`生成小型FC计划，再用
`export_weighted_benchmark.py --binary --workload PLAN --native-library MATCHING_BUILD --output FIXTURE`
导出纯静态材料。在匹配SDK环境运行
`python scripts/check_query_sdk.py --binary SDK_BENCHMARK --fixture FIXTURE --device cpu --dtype float64 --output-dir NEW`；
NPU队列内改为`--device npu:0 --dtype float32`。它检查独立根、完整训练、3 vs 2+1新进程恢复及跨策略拒绝。
