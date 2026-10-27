# 带权实现的有限选型测量

本页规定性能消费者的范围；运行结果从[证据索引](evidence/README.md)进入。
测量不扩展数学资格，也不代表模型效果或所有规模的性能。遵守
[执行契约 §6](execution-flows.md)，每配置默认一个独立进程，项目重型计时串行。

`scripts/benchmark_weighted.py`消费公共Python、Python调用native或resident接口。
`weighted_benchmark_cases.py`只构造固定输入、参数和拓扑，不执行参考轨迹。
配置包括非对齐TimedDAG、同秩HB SettleGraph与反馈PDG；KV/sum首组合或完整SSD，
FP32、SGD动量、原始时间窗口与短卷积。HB保留真实同秩区域，重复边逐条发送。
small为内容宽4、1样本、每窗2位置；medium为宽32、4样本、每窗4位置。
这两个有限尺寸用于判断调度与接口成本，不代表大型模型。

比较问题是：在这些窗口上，streaming/prefill、CPU/eager mixed/resident的完整调用成本如何；
规模变化是否足以抵消打包、设备提交与完整返回的成本。PDG只需native性能，Python作为独立正确性参考。
常驻训练、Python适配与standalone LibTorch必须分别记录；本脚本不能冒充standalone进程。

每次进程只做一个warmup窗口和默认三个连续测量窗口；逻辑时间和状态连续推进。
训练在每个optimizer边界显式detach，参数不重置。loss仅为实际输出数值投影cG的平方和乘.001，
用于验证完整训练机制，不是下游训练配方。记录候选更新、活动Full、消息、输出、pending与原始调度统计。
发送稀疏不能直接解释为所有局部计算变少，resident统计同时给出物理局部/Full行数。

计时包含输入构造/上传、在线调度、packing、前向、完整CPU返回、loss、反向、有限性检查、
SGD更新、切断续接梯度和最终同步。不同接口把完整返回变为Python对象的成本会保留在测量里。
fixture、模型/owner/optimizer构造、warmup、资格对照及最后参数导出分列或排除。
不把调用裸kernel或只计forward当完整训练。记录实际配置、源码与输入hash、栈、分配器峰值、
进程生命周期RSS峰值、负载均值和已知争用；不据单次流程声称统计稳定或普遍最快。

测量结束后才执行独立CPU时间递归，核对全部窗口记录、loss和最终optimizer参数值。
它不向候选提供选定序列、批次或容量。严格失败保留为失败，不能在该配置上宣称等价加速。
此有限对照不代替局部手算VJP、独立根、共享参数、checkpoint及完整设备资格。

示例（干净固定源码，匹配本机环境与持久任务规范）：

```sh
python scripts/benchmark_weighted.py --device cpu --implementation python \
  --topology hb --family kv --size small --mode training --output-dir NEW_RUN
python scripts/benchmark_weighted.py --device cpu --implementation native --native-library CPU_CORE \
  --topology dag --family kv --size medium --schedule prefill --output-dir NEW_RUN
python scripts/benchmark_weighted.py --device npu:0 --implementation resident \
  --native-library PYTHON_CORE --resident-library PYTHON_RESIDENT \
  --topology hb --family ssd --size small --mode training --output-dir NEW_RUN
```

`--profile`必须在另一个独立进程/目录运行，只记录一个profile窗口；导出的CPU/CANN/CUDA trace
用于判断实际算子、copy和host scalar成本，其带profiling开销的执行不作为吞吐样本。
`--allow-dirty`只供入口开发检查，输出保留完整dirty清单，不作固定源码资格或选型证据。
没有CUDA真机时只记录待验，不运行CPU代替CUDA结果。

独立LibTorch入口在`tools/weighted_benchmark`。静态fixture由
`export_weighted_benchmark.py`导出实际参数TensorImpl别名、完整图/端口、配置与输入；
它不执行图，不包含路由或作用序列。C++解码重新核对完整图identity，安装公共数学函数，
候选使用自己的Streaming/Greedy或resident程序；运行时不链接Python。
同一fixture hash用于Python/standalone对照，工具链版本与接口成本仍需分别解释。
这一有限benchmark数据格式不代替通用图序列化或checkpoint。

```sh
python scripts/export_weighted_benchmark.py --native-library CPU_CORE \
  --topology dag --family kv --size small --output NEW_FIXTURE_JSON
python scripts/build_weighted_benchmark.py --core-build CORE --output-dir NEW_BUILD
NEW_BUILD/client/weighted-benchmark --fixture NEW_FIXTURE_JSON --device cpu \
  --implementation native --schedule prefill --mode training --output-dir NEW_RUN
```

常驻standalone构建另给`--resident-build STANDALONE_RESIDENT`，运行显式指定
`--device npu:0 --implementation resident`。构建器校验core/backend与源码、binary hash、
无Python依赖的loader和help；这些检查本身没有真机或性能结论。
C++计时后使用另一个独立scalar调度核对全部记录、loss及最终SGD参数，局部函数锚点沿语义索引追溯。
训练计时包含新建独立输入、完整CPU返回、loss、VJP、finite checks、SGD和同步；
结果文件采用fsynced staging及原子替换，存在的运行目录直接拒绝。
