# 架构与源码导航

先读 [语义索引](semantic-map.md)，再进入本页定位实现。
数学版本由 [upstream.json](upstream.json) 固定，本地契约由 [semantics](semantics.md)
维护；历史验收在 [evidence](evidence/README.md)，不作为当前待办。

## 分层

| 层 | 位置 | 职责 |
| --- | --- | --- |
| core-5配置与实验身份 | `python/tidegraph/planning/`；[候选入口](core5-study.md) | allocation-free配置/shape/共享清单，数学/工作负载/执行/测量身份；公共构造器，CLI仅适配 |
| 配置事实与覆盖证据 | `planning/facts.py`、`scripts/core5_coverage.py`、`record_core5_checks.py` | 全部局部配置分组、逐用例JUnit记录、正确性/性能独立资格与来源查询；不执行参考或补造历史结果 |
| 公共接口 | `python/tidegraph/config.py`, `runtime.py`, `session.py`; `cpp/include/tide/` | 配置、图/模型所有权、执行策略、结果与状态续接 |
| 数学对象与记录 | `python/tidegraph/graph.py`, `records.py`, `history.py`, `coordinates.py`; `cpp/include/tide/types.h`, `cpp/src/graph.cpp` | 图、物理边、逻辑来源、时间、状态、history、消息与输入完整性 |
| 局部计算 | `aggregate.py`, `state_program.py`, `readout.py`, `region.py`, `next.py`, `full.py`; 对应 C++ Kernel 接口 | Agg/Upd/Read/SelStep/Next/Full/Emit，批量能力和局部 VJP |
| 独立 host 调度 | `reference.py`, `streaming.py`, `frontier.py`, `greedy.py`, `specialized.py`, `settle.py`; `cpp/src/stream.cpp`, `frontier.cpp`, `greedy.cpp`, `specialized.cpp`, `settle.cpp` | 完整纤维、就绪、因果顺序、稳定提交、图编码与独立特化 |
| 原生适配 | `python/tidegraph/native*.py`, `cursor.py`; `cpp/src/bind*.cpp` | Python/native 转换与测试适配，不拥有 C++ 核心语义 |
| eager 放置与批量反向 | `placement*.py`, `state_vjp.py`, `read_vjp.py`; `cpp/src/placement*.cpp`, `isolated_*.cpp`, `state_vjp.cpp` | 数据/评分/选择位置、公共根连接、批量组合算子反向 |
| 常驻公共库 | `python/tidegraph/resident*.py`; `tools/device_online/` | 设备队列、就绪/选择、批量消息、窗口、journals、显式 VJP、优化器、跨卡所有权 |
| 带权常驻图 | `python/tidegraph/weighted/resident.py`; `tools/device_online/weighted_flow*` | 五记忆与正式配置的独立在线组合、typed raw state、逐边/端口和精确续接；[资格范围](resident-weighted-flow.md) |
| 带权常驻训练与物理放置 | `tools/device_online/include/tide/resident_weighted_training.h`, `python/tidegraph/weighted/resident_training.py`; `tools/device_online/weighted_window*`, `weighted_training*`, `weighted_flow_peers*` | 保留窗口、独立字段根、原owner局部VJP、canonical master/槽、节点bank发布与样本副本共同更新；[训练](resident-weighted-training.md)、[节点放置](resident-weighted-placement.md) |
| 常驻硬件后端 | `tools/device_online/control_flow.*`, `program.*`, `ascendc/`, `cuda/` | CANN 与 CUDA 独立控制流、算子/内核提交、完成通知及 P2P；共享语义不等于共享驱动实现 |
| 参数与持久化 | `checkpoint*.py`, `token_checkpoint.py`; `cpp/src/parameters.cpp`, `optimizer.cpp`, `checkpoint*.cpp` | 别名归属、值保存/恢复、显式梯度边界；格式版本见 semantics |
| 下游协作 | [协作契约](downstream-contract.md)、`templates/downstream/`、`examples/consumer_core5/` | 固定依赖、请求/交付/验收互链；应用训练/checkpoint在实验侧，基座公共接口不依赖模板 |
| 历史迁移与源码身份 | [迁移规则](history-migration.md)、`scripts/history_map.py`、`verify_history_mapping.py` | 旧→新提交映射、独立逐提交验证、单分支检出；保留历史实测身份 |
| 诊断与产物生命周期 | [复现导航](reproduction.md)、[保留规则](artifact-retention.md) | 原问题→生成器/小输入→诊断代码→独立检查→证据；原始运行目录不是开发依赖 |
| 消费者与验证 | `examples/`, `tools/online_bench/`, `tests/`, `cpp/test/`, `scripts/` | 独立调用、公式/差分测试、构建、资格、有限性能入口 |

局部算子可以在同一语言的多个调度中共享；特化必须拥有独立循环/依赖次序。
独立 Python 调度、C++ 调度和手算公式共同避免“两个入口调用同一个调度”的自证。
常驻源码位于 tools 并不表示只是性能脚本：它构建为公共 resident 库。

## 按问题找文档

| 问题 | 契约 |
| --- | --- |
| 端口、物理边与逻辑来源 | [local-ports](local-ports.md)、[source-domains](source-domains.md) |
| 内容、聚合、状态、读取与 Next | [aggregate](aggregate-programs.md)、[content](content-programs.md)、[state](state-programs.md)、[read](read-programs.md)、[next](next-programs.md) |
| 区域选择、history 与控制 | [region](region-programs.md)、[LH selector](lh-selector.md) |
| Full、Emit、HARD/HST/SOFTP | [full](full-programs.md)、[LH Full](lh-full.md)、[精度](precision.md) |
| EMA/SSM/矩阵记忆/Add/Attention | [模块目录](module-extension-plan.md)、[matrix](matrix-memory.md)、[Add](lazy-add.md)、[attention](attention.md) |
| same-fiber 注意力与 pooling | [fiber attention](fiber-attention.md)、[packing](fiber-packing.md)、[pooling](fiber-pooling.md) |
| host prefill 与流式状态 | [greedy](greedy-scheduler.md)、[frontier](frontier-contract.md)、[cursor](streaming-cursor.md) |
| 三图关系和特化适用域 | [Settle 编码](settle-embedding.md)、[跨类策略](cross-family-policies.md) |
| core-5 带权配置、局部函数与精确时间 | [公共绑定](weighted-message.md)、[五记忆/卷积](weighted-recurrent.md)、[发送](weighted-sending.md)、[评分/证据/key](weighted-scoring.md)、[实际packing](weighted-packed.md) |
| eager 与常驻、窗口与多卡 | [放置](execution-placement.md)、[resident 库](resident-library.md)、[训练](resident-training.md)、[peers](resident-peers.md)、[带权公共训练](resident-weighted-training.md)、[带权样本副本](resident-weighted-peers.md) |
| 批量化的梯度连接 | [packed autograd](packed-autograd.md)、[Full VJP](full-batched-autograd.md)、[Aggregate VJP](aggregate-batched-autograd.md)、[resident graph VJP](resident-graph-vjp.md) |
| 实验调用、loss/更新、容量 | [library](library.md)、[online consumers](online-consumers.md)、[容量](consumer-capacity.md)、[eager 容量](eager-consumer-capacity.md) |
| LH 对照与 token 层 | [compatibility](lh-compatibility.md)、[IOCortex](lh-iocortex.md)、[single graph](lh-single-graph.md)、[token checkpoint](token-application-checkpoint.md) |

## 定位规则

从语义符号找到接口，再用 `rg` 查调用者、校验和测试。`Tensor` 类型本身不证明
shape、dtype、设备、有限性、时间顺序或所有权正确；必须结合模块契约和验证入口。
新图编码要给出观察量投影及适用 cut，新批量路径要声明精确联合求值条件，
新设备模块要同时处理前向、声明的 VJP、容量、生命周期与不支持的请求。

源码与当前测试由 Git 保存；旧 build-manifest、运行路径和机器队列不是开发依赖。
重新构建时生成新源码/二进制身份，不能沿用历史通过数认证新的库。
