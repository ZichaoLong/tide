# 独立 LibTorch 的完整数值 checkpoint

公共 [GraphCheckpoint](../cpp/include/tide/graph_checkpoint.h) 保存同一模型的全部具名参数/固定值、
真实参数别名、可选 SGD/AdamW 状态，以及完整 `Continuation`。
图和局部程序由调用者重建；加载要求同一 graph identity、参数形状/dtype 和别名布局。
本接口独立于 Python，格式不是 Python checkpoint 的替代解析器。

数学依据为 [PDG §6–8](upstream/tide-core-5/positive-delay-graph-finite-cut-learning-note.md)、
[TD §11](upstream/tide-core-5/timed-dag-region-selector-learning-note.md) 和
[Settle §9](upstream/tide-core-5/settlegraph-learning-note.md)。保留最后处理时间/计数、
每个原始 State slot、区域 History、逐边 pending 和完整输入 ledger。
KV 包括尚未清理的过期条目；卷积包括原始时间槽，不用有效投影代替可逆解码。

格式版本由 [semantics](semantics.md) 维护：`TIDEGCK1` schema 2 是完整图容器，
其中嵌入schema 2的 `TIDENCK1` 参数/优化器记录。之前的配对schema 1仍可读，新保存均为v2。外层和内层均有校验和；
所有时间/计数/索引及 int64 slot 都按整数存储，不经浮点编解码。
参数和浮点状态声明 FP16/FP32/FP64；格式使用 CPU 字节序列，可向重建模型的实际 owner 设备恢复。
[显式FP32 master](native-master-optimizer.md)保存优化器精度策略、scale和全部master/slot；
半精度payload与master量化对应须逐值相等。设备构建和实机证据仍各自验收。

保存先校验当前模型和 continuation，采用 fsync staging、排他发布与目录 fsync，
已存在的目标不覆盖。加载先解码全部记录、校验别名/优化器域、分配目标传输，
在暂存模型上校验权重和精确状态，最后复制到原有参数身份并发布优化器槽。
域/格式/配置拒绝发生在参数变更之前；不承诺硬件故障中途 copy 的回滚。
长度、集合数和 tensor 元素数有显式上限；损坏、重复字段、无效 pending、截断和尾随字节拒绝。

保存不修改正在运行的状态；持久化/加载是显式 autograd 边界，恢复的状态 detached。
模型参数保留目标对象及其 trainable 标志；已有参数 grad buffer 不属于 checkpoint。
它不恢复 retained tape、数据游标、框架 RNG、下游 head/loss 或训练控制器。
优化器更新边界由调用者控制；共享参数仍只有一个 canonical owner，不能按使用次数更新多次。

| 返回/契约 | 实现与独立验证 |
| --- | --- |
| 模型/optimizer 与全切面原子发布 | [API 实现](../cpp/src/graph_checkpoint.cpp)；加载返回实际解码后的 Continuation，不执行参考递归 |
| typed 原始状态、历史、pending、ledger | [完整图 codec](../cpp/src/graph_checkpoint_codec.cpp)、[受限二进制原语](../cpp/src/checkpoint_binary.h)；大于 2^54 的逻辑时间、损坏与别名拒绝 |
| 全图训练/续接 | [独立 C++ 可执行文件](../cpp/test/weighted_graph_training.cpp)与[固定参数/拓扑](../cpp/test/weighted_graph_fixture.cpp)：FC、HB depth=1、非对齐 DAG、PDG 反馈/重复边；五种记忆及异构，三步 AdamW、保留图切点、含 pending 恢复后的下一更新 |
| 独立数学与完整比较 | 各 weighted 公式 standalone 检查锚定函数；全图 Streaming 与 Greedy 独立调度，比较完整 Read fields、状态、控制、逐边/端口记录和全部具名梯度/优化器槽；整数 Tensor 严格相等 |
| 原参数格式兼容 | [原生参数/optimizer 测试](../tests/test_cpp_checkpoint.py)，v1读取兼容、v2 typed值/master和排他发布 |

复现入口为 `tidegraph-weighted-training-check --device=cpu --dtype=float64 --output-dir NEW`，
FP32 另运行独立流程。Python 的 [启动测试](../tests/test_cpp_weighted_training.py)只检查进程/结果；
C++ 可执行文件自行构造和验证模型，不读取 Python 输出、路由或梯度。
实现/资格状态见 [STATUS](STATUS.md)；源码中的检查计划不能自动成为通过证据。
