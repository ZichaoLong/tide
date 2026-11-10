# Query根合并与实际返回边界资格

数学固定0770593。实现`a553beae90374ccf1e68feee45e011d3d7d35679`；
工作线程观测修复`fb406dd74af2f86a81158865dde3273e79914c09`。
[审阅记录与原文件hash](query-root-boundary-20261110.json)保留逐项身份，不依赖旧运行目录导航。

完整连续、互不重叠的view分区把cotangent装配回一次base根；部分、重叠、strided和空view
保持原独立根。公共outputs/metrics任务省略内部trace，插件与完整诊断保留；
执行身份v4，旧冻结记录继续解码为原策略。精确恢复使用实际选定路径，核对原始状态、
返回、loss、参数、梯度和optimizer，不用另一条traced路径代验。
契约见[query VJP](../weighted-query-vjp.md)、[验证边界](../experiment-validation.md)。

| 固定范围 | 结果 |
| --- | --- |
| a553 CPU FP64/FP32相关门禁 | 249通过，2个原GDN/HB数值xfail |
| a553 NPU FP32相关门禁 | 73通过，2个原xfail，1个新增观测失败；原任务failed保留 |
| fb406dd线程观测补测 | CPU24、NPU2通过 |
| Python/native安装消费者，CPU/NPU各一组 | 独立三步训练、完整根与新进程精确恢复通过 |
| 独立SDK | 3项CTest通过；CPU FP64/NPU FP32各complete与outputs三步/新进程恢复通过 |

新增NPU失败在packing统计键，发生于梯度比较前。Torch反向worker不继承Python ContextVar；
修复只传递仍活跃的observer，调用方新observer优先，已关闭observer不复用。
同原始FP32材料诊断中scalar32、未合并与合并梯度逐位相等，对CPU FP64最大9.61643e-8；
这不是放宽原容差，也不是所有NPU数值差异均为舍入的证明。
SDK complete另验独立记录、参数、SGD和独立根；outputs仅检查实际边界和精确恢复，
不能由自身恢复推断独立数值通过。模型与SDK核心构建内容hash复用记录在JSON，
Python和独立SDK环境分离；fb406dd未改C++。NPU补测实际物理11，有限共享准入且非独占承诺。

原FC16/B8/L128/d128、16节点/4区域、KV qry/quad regional_total、5,092,788可训练参数
（含下游头）、FP32/native/CPU1线程，一次预热后恢复初始参数与optimizer再测一步：
a553完整返回29.7882秒，最小返回22.0680秒；此前413faf6分别32.5641/25.4074秒。
最小返回forward8.2020、backward13.7602秒，候选16384/Full8344/消息25060。
两种返回内容不同，各自独立比较；原任务独立数值/离散/学习未认证。
此报告不含NPU新成本、不认证resident、双卡、大规模或标准Transformer/Mamba性能。

干净源码复现：依[development](../development.md)构建/安装该固定提交，运行
`tests/test_query_roots.py tests/test_selected_boundary_execution.py tests/test_kv_sequence_vjp.py`
及相关dense/component_window门禁；CPU用`--dtype both`，NPU用`--dtype float32`
并设`TIDE_DENSE_DEVICE=npu:0 TIDE_WINDOW_DEVICE=npu:0 TIDE_BUILD_DIR=...`。
fb406dd另跑`tests/test_packing_observation.py`；安装消费者入口
`examples/consumer_core5/dense_query.py --no-trace`，SDK入口`check_query_sdk.py`。
原任务成本由`scripts/benchmark_dense_query.py`读取固定下游Git对象复现，
明确`--trace-records complete|minimal`；下游仓库始终只读。
