# 架构与源码导航

先读 [语义索引](semantic-map.md)，再进入本页定位实现。
数学版本由 [upstream.json](upstream.json) 固定，本地契约由 [semantics](semantics.md)
维护；历史验收在 [evidence](evidence/README.md)，不作为当前待办。

## 分层

[新的resident块边界](resident-block-execution.md)把潜在任务DAG、设备choose/commit、
KV Read/Describe和完整区域数值分开：`tools/device_online/block_dispatch*`、`settle_block_plan.h`
和`block_read_selection*`分别拥有调度、静态lowering和数值/VJP；`block_selection_formula.cpp`
为独立公式锚点。修改同步CPU metadata检查、`device_component_checks`及对应设备check。
这些是新执行器组合件，未成为公共完整图owner；旧resident资格不能自动继承。
[KV消息组合](resident-block-kv-messages.md)由`block_kv_messages`/VJP连接独立Read与proposal
快照、区域控制、Full及未缩放(c,G)；`block_kv_message_parameters`按实际owner汇总跨角色alias。
`block_kv_message_formula`独立锚定数学，`block_kv_messages_check`核对独立根、默认分支和拒绝。
修改须同步设备检查注册、独立公式/容量测试；它尚无Aggregate/Upd、路由或公共训练owner。

LH/HB同材料三层对照由[独立张量基线](lh-tensor-baseline.md)连接
`lh_tensor_baseline/math/state`、`benchmark_lh_tensor`与`test_lh_tensor_baseline`；
直接候选的dense Full/数值投影与公共selected Full/完整continuation职责分别记录。
`check_lh_tensor`/`lh_tensor_check`独立检查真实列的控制、支持、raw终态、端口cG及任务VJP；
修改同步`test_lh_tensor_check`，数值投影与完整独立根/恢复资格不混同。
[REQ-006诊断](req006-diagnostics.md)由`diagnose_req006`和`req006_material/initial/runtime/profile`负责，
固定下游源码/压缩配置，无插桩更新/完整步、trace和计算图统计分别执行。
`req006_initial`原子保存/复用实际prime后模型、optimizer槽和RNG，支持跨进程配对。
修改同步`test_req006_diagnostics`、`test_req006_initial_state`；下游显式Attention TF不是FC融合SDPA对照。

FC16同尺寸成本与诊断入口见[阶段归因](fc16-cost-attribution.md)：可选native线程排他计时、
真实profiler范围与独立张量基线分列；修改时维护benchmark_dense_query及test_stage_profile。
[FC4/FC16三层调查](evidence/fc-three-layer-comparison-20261111.md#把差额落到计算与接口)
把快照layout、Read/Full私有图、bank Attention及sequence伴随连接到实际成本。
该调查后的[查询块实施](evidence/kv-query-blocks-20261111.md)已把隐藏前缀改为bank列及块伴随；
公开末态/trace保持独立根。剩余区域/消息/发布接口成本由ROADMAP跟踪，不把外层阶段少当作内部无碎片。
修改诊断时同步`run_fc_comparison/check_fc_comparison/collect_fc_comparison`及
`test_fc_comparison_diagnostics`；可重复配对/附件由`cost_comparison/execution_diagnostics`与
`test_cost_comparison`负责，区分公共根、任务投影、实际计时边界和不同函数比较。
外层/私有autograd图的节点、实际执行和保存存储由[图统计](autograd-census.md)定位；
不把图统计时间当吞吐，也不叠加Python/native重复计数。
逐文件/函数审阅用[固定源码清单与实际覆盖](evidence/implementation-audit-20261110.md)；
自动清单不代表已审阅，相关工具为`scripts/review_inventory.py`/`review_symbols.py`。
后继按层追踪[调度](evidence/scheduling-interface-audit-20261110.md)、[区域](evidence/region-batching-audit-20261110.md)
和[循环/Describe](evidence/recurrent-interface-audit-20261110.md)的实际审阅；hash manifest不混算重叠覆盖。
[本轮诊断/消息组件审阅](evidence/rewrite-followup-reviews-20261112.json)绑定7d5e05c3实际文件hash；
后继[生产边界审阅](evidence/p1p6-boundary-review-20261112.md)绑定55a3074a的24个实际文件，
覆盖当前默认能力分派、五记忆/区域列、standalone与原始状态交接，并区分实现自审和独立资格。
这些记录不认证未列出的函数；C++自动函数提取尚未完成，也不把所有历史工具重审作为当前交付前置。
新修改按[维护影响表](core5-study.md#修改时同步维护什么)选择相应层，记录实际审阅符号/hash与
独立检查；构建、安装消费者、训练恢复和性能各归其固定证据。

常驻单bank投影的整块GEMM/VJP在`tools/device_online/weighted_projection.cpp`，设备实际使用与有限性守卫
在`ascendc/tide_weighted_projection_shared.cpp`；同源转换提供CUDA源码及CPU语义核检查。
`weighted_saved_tensors.h`负责typed tape在新反向program中的借用计费，组合预留在KV query/Full与Residual。
修改同步[投影契约](resident-weighted-projection.md)、[预算契约](resident-storage-accounting.md)、
projection/query/Full独立设备checks及完整flow/training资格；host张量列与device在线调度是不同路径。
`cann_task_services.cpp`负责CANN服务父子树、通知和停止，`task_services_check.cpp`独立检验嵌套、
兄弟、双根及重复窗口。`scripts/trace_resident_step.py`记录实际公共step的失败阶段；
局部计费、任务队列和流数量不能混为一种资源。固定结果见[共享矩阵与服务资格](evidence/resident-shared-bank-20261112.md)。

默认host prefill的区域SCC窗口分解：`window_plan.py`与`cpp/src/component_{plan,window}.cpp`。
公共选项、实验身份和旧greedy复现规则见[调度契约](greedy-scheduler.md#default-component-windows)；
修改时同步ExecutionOptions/ExecutionSpec、Python/native/standalone分派及`test_component_window*`。
KV整序列反向位于`weighted/kv_sequence_vjp.py`与`weighted_kv_sequence_vjp.cpp`，
由未追踪块的模块能力选择；修改时同步[State VJP契约](weighted-state-vjp.md)、
`test_kv_sequence_vjp`和保留中间根的`test_kv_prefix_roots`，再核对完整训练/恢复与实际成本。
公共实验的按需记录由`experiment/eager.py`与standalone `Eager::options.trace`绑定；
修改时同步`ExecutionSpec`/SDK恢复身份、实际工作计数、`experiment/checkpoint.py`及
`test_selected_boundary_execution`。内置输出任务可省去中间日志，插件不得隐式失去loss所需的根。
Describe的完整列计算由`weighted/dense_describe.py`与`weighted_describe_heads.cpp`负责，
`weighted_describe_batch.cpp`连接Mem并只在发布边界拆行；cat/match、证据、key和gain均接收leading batch轴。
修改同步[查询契约](weighted-dense-query.md)、独立字段根、`test_weighted_describe_columns/capability/training`
及`test_prefill_local_batches`；默认策略身份describe-batch-v2，安装/SDK恢复拒绝旧v1身份。
Describe能力与记忆query策略独立报告，非strict数值策略仍按其客户端支持域拒绝。
循环Read/Full由`weighted/query_batch.py`、`full_batch.py`及C++ `weighted_query_recurrent/weighted_full_batch`
按消费者求值，见[用途契约](weighted-recurrent-query.md)；原始状态和版本守卫完整保留。
修改需核对`test_weighted_recurrent_query_scope`、独立公式及完整训练/恢复，不能用内部None替代checkpoint。
区域VJP的私有列打包在`weighted/private_rows.py`/`weighted_private_rows.h`：
同结构组内打包内部叶，外部行/字段根保持；连续视图区间消除重复复制与反向分发。
修改须同步`test_private_rows`、query/region独立根、原版本守卫和安装消费者恢复。
区域数学的公共列边界在`weighted/allocation_columns.py`、`regional_columns.py`与
`cpp/include/tide/weighted_columns.h` / `weighted_*_columns.cpp`；见[区域契约](weighted-region-batch.md)。
默认`region_vjp`直接用这些列和结构行支持做伴随，不再在私有图重建逐事件RegionInput。
修改同步`test_weighted_columns`、既有区域独立根/训练恢复，以及allocation observer。
查询的ragged快照保留独立内部叶；query/region VJP按[只读发布契约](tensor-publication.md)
暴露detach视图，消费者修改先clone；显式owned-copy工具与返回到host的独立存储契约另列。
修改须同步`test_independent_copies`、query/region根及恢复；batch字段用unbind统一线性转置。
新张量窗口接口见 [weighted-tensor-windows](weighted-tensor-windows.md)：公共
`weighted.execution.eager_runner` 的 `run_positions/run_ports` 在能力许可时直接采用
`settle_columns`；`window_execution` 负责公共身份/能力分派，`window_records` 只负责
显式后备的记录装配。KV 与循环各有 typed 私有状态，消息/区域列连接两者。
C++ `weighted_sequence_columns.h` 公开 KV/recurrent typed状态与独立整链；
`weighted_column_checkpoint.h/.cpp`公开 `WindowExecutor`，在完整位置cut上验证/import/export
真实raw continuation、history与逐物理端口ledger。`tidegraph-column-checkpoint-check`连接
独立Streaming、三步SGD与exec新进程精确恢复；`tools/weighted_benchmark/eager_columns`
把它接入standalone默认入口，整列task loss及反向后CPU解码与完整记录诊断分开。
修改同时维护CLI/规划执行身份、`weighted-run-check`和`test_experiment_columns`的旧锁保持检查。
SDK消费者的`check_query_sdk.py`从`effective_execution`区分列与records；列路径的实际工作量、
独立scalar Streaming任务对照及精确恢复，与complete模式的独立事件根分别检查。
`test_query_sdk_capability`和`test_standalone_columns`防止新默认入口仍被旧统计计数误拒绝。
`weighted_column_modules` 负责私有状态分派，`weighted_column_recurrent_state/math/query`
分别负责raw编解码/scan、系数与原始时间卷积、Read/Full。`weighted_column_state/query` 与
`weighted_settle_columns` 分别拥有局部数学和调度。相对位置的 logits/共享 value-bank
transpose 在 `column_attention` / `weighted_column_attention`；位置元数据保持 int64。修改同步该契约、column/window 测试、
旧一般图回归和实际默认客户端，不把低层 Plan 成绩当默认入口成绩。
`message_columns`与`weighted_settle_columns`的聚合/区域交接复用已对齐行域；
`test_column_assembly`核对物理重边、同类独立根、全零断开及非对齐纤维，
完整默认路径仍由`test_settle_columns`与安装窗口验收。
跨节点KV准备由`column_modules.prepare_many` / `prepare_columns_many`分派，
`kv_columns_state.prepare_kv_columns_many` / `columns::prepare_many`按实际写入尺寸和预算成组。
`row_linear.linear_blocks` / `weighted_row_linear`提供整块矩阵与一阶转置，
相同参数身份用共享GEMM，独立矩阵用BMM；`test_linear_blocks`、`test_kv_column_groups`
及SDK `weighted_sequence_columns`核对独立根、共享贡献、版本、零写与raw状态。
`column_modules.read_many/full_many`与C++同名分派把默认调度的KV查询接入
`kv_column_groups` / `weighted_column_read_groups/weighted_column_full_groups`；
完整配置、实际query/bank形状和额外workspace决定分组，不按拓扑名称选择。
`column_query_ops` / `weighted_column_query_ops`负责node×B收缩与独立多输出一阶边界，
`dense_describe_groups` / `weighted_describe_groups`负责各Describe头的分阶段投影。
`column_position_ops` / `weighted_column_positions`拥有单列/跨节点共同使用的位置求值，
按dtype和实际时间/相位域选择共同整数原点的Q/K旋转或原relative展开；可见性仍用int64 age。
其显式一阶旋转边界保留原值版本，`test_column_position_ops`核对独立公式、域外反例和默认路径。
`column_policy`与planning/execution、两个实验客户端和standalone身份共同拒绝旧v5列锁静默升级。
修改同步`test_column_query_groups/dispatch`、SDK `check_query_groups`、默认调度和放置测试，
尤其区分普通独立列根与旧事件Content适配器的connected-zero。
`weighted/regional_packing`与C++ `weighted_regional_packing`拥有候选数量分组的int64关联布局，
`regional_groups` / `weighted_regional_groups`组合完整区域数学、原始版本守卫及块反向。
改变这些边界须同步`test_regional_packing`、`test_regional_groups`、一般Settle独立调度与恢复。
固定单位传输的数值检查由`weighted/transport_validation.py` / `weighted_unit_transport.h`
按device/dtype及临时字节预算合批；每次读取全部实际值，未使用跨窗缓存或省略版本检查。
修改同步`test_unit_transport_validation`、模型绑定/checkpoint拒绝与默认执行器检查。
卷积tap的有序一阶转置在`weighted/tap_gather.py` / `weighted_column_gather.cpp`，
对应`test_tap_gather`的独立线性映射与`test_native_recurrent_columns`的完整循环检查。
修改同时检查`column_capacity` / `workspace_bytes`与实际设备精确恢复。
`column_placement` / `weighted_column_placement`拥有Read/control/selection临时列视图，
按消费者选择实际参数/字段，复用已有显式copy/VJP和通信策略。私有raw状态留在payload owner。
`test_column_placement`检查独立标量放置、共享根、None/零、评分dtype和正支持cast拒绝。
`column_sharing.shared_uses`保留Aggregate计算共享并按稳定节点次序归并使用者伴随；
`test_column_sharing`检查字面公式/None零/版本，`check_column_adjoint_order.py`定位同图重复反向差异。
`column_transport`负责实际消息到目标owner的整列copy复用和区域描述汇集；
Python/C++列调度按节点/区域分别解析位置。readout及内置loss可归约不同输出owner，
`test_column_payload_owners`与安装窗口的peer选项检查独立调度/VJP/optimizer/恢复。
`block_attention` / `weighted_block_attention`及位置logits算子拥有共享bank的一阶转置和FP16内部FP32策略；
`test_half_columns`覆盖溢出点积、原始版本、五记忆独立根、master更新和raw恢复，显式评分与消息精度不混同。
`experiment/column_task.py`拥有内置任务的共享来源列、逐物理端口loss和无梯度CPU输出解码；
`experiment/eager.py`按观察/模块能力选择，`planning/execution.py`固定新列策略的execution-v6身份，
`planning/materialize.py`把参数构造与旧事件输入构造分离。修改同步
`test_experiment_columns`、`test_selected_boundary_execution`、历史身份和CLI新进程恢复。
`planning/execution`区分窗口能力与record-query后备；half/peer的auto窗口不得提前变成显式legacy。
内置half端口loss在乘法前提升FP32，独立公式与master SGD/恢复用例检查实际默认入口。
插件/完整事件观察不自动改成列根；单纯调用新的局部核不证明上层消费者已迁移。

隐藏KV查询块见[块接口契约](weighted-query-blocks.md)：`weighted/kv_block_*`与
`weighted_kv_block_*`接收actual prepared bank、B×L列与原始时间，块伴随直接归还原始写入；
`block_attention`避免五维VJP展开。trace/terminal/不适用布局保留原独立根路径。
修改须同步执行身份、actual block计数、`test_kv_block_query`及训练/恢复检查。

KV快照数值共享位于`weighted/snapshot_bank.py`/`weighted_snapshot_bank.cpp`；
[存储与伴随契约](weighted-snapshot-bank.md)区别共享存储、独立根及只读State视图，
修改须同步`test_snapshot_bank`、query与sequence VJP、暖状态/精确恢复和实际投影工作量。
共享bank上的Full矩阵边界由`weighted/bank_attention.py`/`weighted_bank_attention.cpp`
负责，`query_packed`按实际布局能力接入默认路径；前向共享数值，反向仍保留独立快照根。
修改须同步`test_bank_attention`、上述snapshot/sequence检查及[查询契约](weighted-dense-query.md)。
局部成本用`benchmark_bank_attention.py`，层/组合对照用`benchmark_layer_chain.py`和
`benchmark_dense_query.py`；[有状态基线资格](evidence/additive-interface-qualification-20261110.md)
与完整公共记录资格分列，不能将投影baseline移入默认factory后继续自称独立参考。
通用批量有限值守卫在`finite.py`/`finite_guard.h`，Read调用在`readout.py`/`read_fields.cpp`；
分配元数据回读在`allocation_rows.py`/`weighted_allocation_rows.cpp`，保持数值求解/VJP与host元数据分层。
输出推理的记录生命周期由`experiment/record_policy.py`共同连接执行和准入；
`planning/record_capacity.py`只缩减已审核host分量窗口推理的事件预留，完整记录估算仍可查询。
修改须同步`test_selected_boundary_capacity`、实际trace/恢复检查和`diagnose_experiment_resume.py`；
容量估算不替代实际峰值与硬件执行证据。

| 层 | 位置 | 职责 |
| --- | --- | --- |
| 显式正支持数值策略 | `weighted/support_policy.py`、`stable_mass.py`；[契约](positive-support-policy.md) | Shannon最小正值存储与未补值VJP、稳定Aggregate/KV；默认严格路径及调度独立性保持 |
| core-5配置与实验身份 | `python/tidegraph/planning/`；[候选入口](core5-study.md) | allocation-free配置/shape/共享清单，数学/工作负载/执行/测量身份；公共构造器，CLI仅适配 |
| 配置事实与覆盖证据 | `planning/facts.py`、`scripts/core5_coverage.py`、`record_core5_checks.py` | 全部局部配置分组、逐用例JUnit记录、正确性/性能独立资格与来源查询；不执行参考或补造历史结果 |
| 人类实验入口 | `python/tidegraph/experiment/`；[操作手册](human-workflow.md) | 安装身份、无分配plan、同模型放置预算、独立check、run/matrix/job、完整及按需返回、CSV/HTML查询；不依赖内部benchmark导入 |
| 实验资源准入 | `planning/placement.py`、`planning/capacity.py`、`experiment/admission.py`、`experiment/storage.py` | 不可拆共享组与逐卡预算；训练CPU快照、tmpfs联合主存及checkpoint代数，保持静态估计与实测分列；`test_training_host_capacity`连接实际欠估见证 |
| 离线结果浏览 | `experiment/report.py`、`html_report.py` | 审阅归档校验、独立状态分列、CSV与带本地筛选的HTML；页面筛选不改变原记录 |
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
| 参数与持久化 | `checkpoint*.py`（含`checkpoint_metrics.py`分阶段观测）, `token_checkpoint.py`; `cpp/src/parameters.cpp`, `optimizer.cpp`, `checkpoint*.cpp` | 别名归属、值保存/恢复、显式梯度边界；格式版本见 semantics |
| 下游协作 | [协作契约](downstream-contract.md)、`templates/downstream/`、`examples/consumer_core5/` | 固定依赖、请求/交付/验收互链；应用训练/checkpoint在实验侧，基座公共接口不依赖模板 |
| 历史迁移与源码身份 | [迁移规则](history-migration.md)、`scripts/history_map.py`、`verify_history_mapping.py` | 旧→新提交映射、独立逐提交验证、单分支检出；保留历史实测身份 |
| 诊断与产物生命周期 | [复现导航](reproduction.md)、[保留规则](artifact-retention.md) | 原问题→生成器/小输入→诊断代码→独立检查→证据；原始运行目录不是开发依赖 |
| 完整返回/推理观察 | `python/tidegraph/experiment/returns.py`、`tools/weighted_benchmark/return_packing.cpp`、`scripts/weighted_benchmark_work.py` | [显式返回packing](complete-return-packing.md)、独立推理/新owner续接、实际桶/MM/BMM；不提供调度轨迹 |
| 运行时有限值检查/活动观察 | `experiment/finite.py`、`experiment/activity.py` | 显式诊断packing保留非有限拒绝、原默认身份；真实逐节点活动样例、缓存与工作量位于计时外，见[独立验收](experiment-validation.md) |
| 输入事务有限检查 | `finite.py`、`validation.py`、`cpp/src/input_validation.cpp` | 同device/dtype按8MiB合批检查；超大单字段单独检查；拒绝后不发布ledger/state。`test_prefill_guards`核对末行非有限、异形与重试 |
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
| 查询与Describe默认策略 | `weighted/query_execution.py`、C++ `configure_query_execution`；[默认auto与显式路径](weighted-dense-query.md)、query_layout/query_packed及独立字段/别名/训练/恢复；[代表回归](prefill-regression.md) |
| 默认列入口、一般图后备与实际执行身份 | [tensor windows](weighted-tensor-windows.md)、`window_execution.column_reason`、`experiment/column_task.reason`、standalone `column_owner`；`test_window_execution/test_experiment_columns/test_standalone_columns`分别核验公共、实验及SDK默认路径 |
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
当前维护入口的交付标准是：新上下文能从配置找到实际分派和局部模块、从行为找到独立检查、
从报告找到固定源码/命令/结果，并知道修改时同步哪个契约与消费者。未修改的历史模块沿用
其证据边界；新resident、位置算法或其他重写层在自身提交/资格中补审，不制造无限历史重审任务。
