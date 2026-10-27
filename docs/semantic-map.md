# 数学语义 → 表示 → 实现 → 验证

本页是导航与实现域说明，不另定义数学语义。固定版本与逐字节校验见
[upstream.json](upstream.json)；`python scripts/check_upstream.py` 可离线核验。
[semantics](semantics.md) 拥有本地公式、精度、梯度和序列化版本。
以下正典与局部教材固定到 core-5 / `0770593`，不跟随上游工作区草稿。
采用来源不等于全部候选已完成；实施与证据边界见 [ROADMAP](ROADMAP.md)。

## 教材索引

| 简称 | 固定教材 | 主要定位 |
| --- | --- | --- |
| Legacy | [旧语义锚点](upstream/tide-core-3/semantics-anchor.md) | 仅供既有 profile 和历史证据追溯；旧提交单独校验 |
| TD | [TimedDAG](upstream/tide-core-5/timed-dag-region-selector-learning-note.md) | §2–5 对象与函数；§6 直接语义；§8–10 就绪与作用依赖；§11 切面；附录 H 指定反向 |
| PDG | [正时延图](upstream/tide-core-5/positive-delay-graph-finite-cut-learning-note.md) | §1 局部函数；§2 有限递归；§4–5 作用与关闭；§6–8 continuation/合法调度；§10 因果块 |
| Batch | [时间批](upstream/tide-core-5/timed-dag-chunk-prefill-learning-note.md) | §2–3 精确联合求值；§4 契约相对最大前沿；§5 动态切口；§8 组合 |
| Settle | [单次结算图](upstream/tide-core-5/settlegraph-learning-note.md) | §2–6 单次结算；§9 分段继续；§10 嵌入；§11 时间批；§12 函数保持 |

数学域可以是抽象集合；当前实现采用下面的具体 Tensor/记录表示。
“符合教材”不自动意味着当前所有表示、profile、Python/native 或设备后端都能承载。
先检查定义域，再检查精确批量能力和硬件能力；不支持时按公开接口明确拒绝。

## 带权支线的固定定义与实施边界

| 定义 | 固定入口 | 实施追溯 |
| --- | --- | --- |
| 共同局部架构 | [architecture](upstream/tide-core-5/local-semantics/weighted-message/architecture-learning-note.md) §1–7 | [具名 Read / int64 状态 / SrcClass](structured-records.md)；[带权 KV 绑定](weighted-message.md)给出 payload/content、初态、Full/Next、共享参数与逐函数测试 |
| 区域函数与配置 | [allocation](upstream/tide-core-5/local-semantics/weighted-message/region-allocation-learning-note.md) 附录 S、§2–6 | [十种发送/增益与独立反向](weighted-sending.md)、[完整评分/证据/key](weighted-scoring.md)和 [quad/bern/ent 精确解 VJP](weighted-allocation.md)已接入；各候选与后端资格按 ROADMAP 继续，不能复用旧 HST 身份 |
| 记忆 | [KV](upstream/tide-core-5/local-semantics/weighted-message/kv-memory-learning-note.md)、[循环](upstream/tide-core-5/local-semantics/weighted-message/recurrent-memory/affine-delta-learning-note.md)、[GDN](upstream/tide-core-5/local-semantics/weighted-message/recurrent-memory/gated-delta-learning-note.md)、[SSD](upstream/tide-core-5/local-semantics/weighted-message/recurrent-memory/ssd-learning-note.md)、[卷积](upstream/tide-core-5/local-semantics/weighted-message/recurrent-memory/causal-convolution-learning-note.md) | KV 按 [绑定契约](weighted-message.md) 接入；[循环/短卷积](weighted-recurrent.md) 对应全部状态、函数、快照和独立公式；旧 Attention/SSM/Delta 不冒充新定义 |
| 实例 | [FC](upstream/tide-core-5/instances/fully-connected-layered-weighted-message.md)、[HB](upstream/tide-core-5/instances/hb-lattice-weighted-message.md) | [首组合](weighted-message.md)接入 FC/HB、非对齐 DAG/反馈 PDG；HB 同秩不改写时间；各后端证据独立 |
| 验收 | [核对 §1.1–1.4](upstream/tide-core-5/reference/weighted-message-implementation-checks.md) | 定义→配置/参数/快照→符号→独立测试→固定源码证据；各后端分别记录 |

以下已有局部 profile 仍是明确的独立绑定，不自动变成带权支线。

## 局部函数的域、值域与依赖

带权局部 Upd 的 [精确联合/实际 packing 对应](weighted-packed.md)独立列出
KV ragged 投影、四种循环的仿射前缀和原始时间卷积；
[State 批量 VJP](weighted-state-vjp.md)独立声明嵌套参数、结构分组、输出槽与运行资格。
带权 [Aggregate 批量计算/VJP](weighted-aggregate-batch.md)单列原子/class/fiber 布局、独立根和零总量分支。
[Read/Full 批量数值求值](weighted-query-batch.md)列出完整快照、KV 可见性、卷积槽和字段解码；
[Read/Full 批量 VJP](weighted-query-vjp.md)逐项声明私有图、公开字段/行根、共享参数与资格边界。

`θ` 是逻辑时间；`B` 是已完整且非空的同时间带来源纤维；`q` 是旧状态，
`q̃` 是 Upd proposal，`q̄` 是选择后的 comparison，`h` 是本地内容。
`C` 是区域完整候选集合，`y` 是旧区域历史，`A ⊆ C` 是活动集合。
参数及布局是每个程序的显式环境，不由调度器偷偷改写。

| 数学接口 / 锚点 | 输入域 → 输出域；依赖边界 | 实现入口 | 独立验证入口 |
| --- | --- | --- | --- |
| `Agg_v` / [TD §4.1](upstream/tide-core-5/timed-dag-region-selector-learning-note.md#41-状态本地内容与描述量)，[PDG §1.4](upstream/tide-core-5/positive-delay-graph-finite-cut-learning-note.md#14-内部消息输出和局部函数) | `(θ,B)` → 内容 `h`；保留物理标签和逻辑来源，空纤维不调用；当前内容含 summary Tensor、原始 atoms 和可选逐来源贡献 | [AggregateKernel](../cpp/include/tide/aggregate.h)、[Python Aggregate](../python/tidegraph/aggregate.py)；[完整内容](content-programs.md) | [公式](../tests/test_aggregate_formulas.py)、[契约](../tests/test_aggregate_contract.py)、[原生自定义](../cpp/test/custom_aggregate.cpp) |
| `Upd_v` / [TD §4.1](upstream/tide-core-5/timed-dag-region-selector-learning-note.md#41-状态本地内容与描述量) | `(q,θ,h)` → `q̃`；无 Full/选择结果依赖；当前 State 有 value、时间/计数和命名 Tensor slots，布局由 profile 校验 | [StateKernel](../cpp/include/tide/kernel.h)、[StateProgram](../python/tidegraph/state_program.py)、[state](state-programs.md) | [Add 手算](../tests/test_add_formulas.py)、[原生自定义](../cpp/test/custom_kernel.cpp)、[状态时钟](../tests/test_state_clock_contract.py) |
| `Read_v` / [TD §4.2](upstream/tide-core-5/timed-dag-region-selector-learning-note.md#42-三种选择描述量)、§5.2 | 按 content/old/proposal 模式取得 `(θ,h[,q或q̃])` → 描述量 `d`；只暴露该模式的状态；当前输出是有限浮点标量或显式具名 Tensor 字段，字段形状及精度/设备显式 | [ReadInput/ReadKernel](../cpp/include/tide/read.h)、[Python Read](../python/tidegraph/readout.py)、[契约](read-programs.md) | [Read 契约](../tests/test_read_contract.py)、[原生 Read](../cpp/test/read_programs.cpp)、[批量 VJP](../tests/test_read_batched_vjp.py) |
| `SelStep_{j,C}` / [TD §5.3](upstream/tide-core-5/timed-dag-region-selector-learning-note.md#53-带历史的选择函数)，[PDG §1.5](upstream/tide-core-5/positive-delay-graph-finite-cut-learning-note.md#15-选择历史控制量与两种状态) | `(y,θ,(d_v)_{v∈C})` → `(A,(c_v)_{v∈C},y')`；`A⊆C, card(A)≤K`，controls 覆盖全部候选；空候选不调用且保留 history | [RegionInput/Selection](../cpp/include/tide/region.h)、[Python Region](../python/tidegraph/region.py)、[契约](region-programs.md) | [Region 契约](../tests/test_region_contract.py)、[原生 Region](../cpp/test/region_programs.cpp) |
| 状态采用 / [TD §5.4](upstream/tide-core-5/timed-dag-region-selector-learning-note.md#54-本次计算状态与下一持久状态)，[PDG §1.5](upstream/tide-core-5/positive-delay-graph-finite-cut-learning-note.md#15-选择历史控制量与两种状态) | 由 observe-all/active-only 从旧状态与 proposal 决定 `q̄`；Full 读 comparison，Next 不能覆盖它 | [block_step.cpp](../cpp/src/block_step.cpp)、[参考调度](../python/tidegraph/reference.py)、[本地公式](semantics.md) | [手算与结构](../tests/test_semantics.py)、[Next 调度](../tests/test_next_schedules.py) |
| `Next_v` / [TD §5.4](upstream/tide-core-5/timed-dag-region-selector-learning-note.md#54-本次计算状态与下一持久状态)，[PDG §1.4](upstream/tide-core-5/positive-delay-graph-finite-cut-learning-note.md#14-内部消息输出和局部函数) | `(q,q̄,θ,h,active,c)` → 持久下一状态；**不接收 Full 输出**；clear 可产生 connected zero，不能抹去本次 comparison | [NextInput/NextKernel](../cpp/include/tide/next.h)、[Python Next](../python/tidegraph/next.py)、[契约](next-programs.md) | [Next 契约](../tests/test_next_contract.py)、[原生 Next](../cpp/test/next_programs.cpp) |
| `Full_v` / [TD §4.3](upstream/tide-core-5/timed-dag-region-selector-learning-note.md#43-节点的完整输出函数)，[PDG §1.4](upstream/tide-core-5/positive-delay-graph-finite-cut-learning-note.md#14-内部消息输出和局部函数) | 对活动候选的 `(q̄,θ,h,c)` → 本地辅助结果与稀疏输出槽族；只发送 present 槽，零载荷仍 present；不写持久状态 | [FullInput/FullKernel](../cpp/include/tide/full.h)、[Python Full](../python/tidegraph/full.py)、[契约](full-programs.md) | [Full 契约](../tests/test_full_contract.py)、[原生 Full](../cpp/test/custom_full.cpp) |
| Emit/传输 / [TD §3](upstream/tide-core-5/timed-dag-region-selector-learning-note.md#3-外部输入记录内部消息与时间纤维)、§6；附录 H | 输出槽 → 带物理边 ID、到达时间和位置的消息/输出；HARD/HST/SOFTP 是本地声明的不同 forward/VJP profile | [ops.py](../python/tidegraph/ops.py)、[delivery.cpp](../cpp/src/delivery.cpp)、[传输](packed-transport.md) | [Emit 手算 VJP](../tests/test_semantics.py)、[transport](../tests/test_transfer.py)、[resident emission](resident-emission-vjp.md) |

当前 Region 描述量不是任意结构化 `D_v`，control 是每候选有限浮点 Tensor，
内建 Emit/Next 可能进一步要求标量。History 使用命名 int64、node maps 和 Tensor slots。
配置 [SourceDomain](source-domains.md) 别名时，同一完整纤维每个逻辑来源最多一条；
重复逻辑到达明确失败，不能隐式合并或丢弃。
各模块的 shape、dtype、device、有效值和参数共享由对应 validate 入口检查。
扩展这些表示须同步适配 checkpoint、比较器、批量化与目标后端，不能只放宽一处校验。

## 依赖次序与可观察量

完整纤维 → Agg → Upd/Read（取决于 Read 模式） → 区域完整候选/旧 history
→ SelStep → comparison/Next → 仅活动节点 Full → 消息发送。
节点旧状态与区域旧 history 各有独立的跨时间前驱；输入完整、状态就绪和 history
就绪必须分别证明。实现可以采用合法联合求值，不能越过这些依赖。

正时延保证有限 cut 上进展；逻辑时间、输入位置、观察次数和墙钟时间不同。
Python 入口要求精确 int64 范围整数，拒绝 bool/float 冒充坐标。
消息保留重复边身份、稳定次序、缺失/零区别；梯度还保留 None/connected-zero。
比较完整事件、输出、状态槽、history、pending、控制/路由及声明的一阶梯度，
不能仅比较最终输出或事件数量。

## 三层级、调度与编码

| 关系 | 数学锚点 / 实现域 | 源码与检查 |
| --- | --- | --- |
| PDG | [PDG §1](upstream/tide-core-5/positive-delay-graph-finite-cut-learning-note.md#1-正时延图与共同局部函数)–8：有限固定多重图、严格正时延，可有反馈；sealed 有限输入窗口 | [graph](../python/tidegraph/graph.py)、[reference](../python/tidegraph/reference.py)、[native Streaming](../cpp/src/stream.cpp)、[手算闭环](../tests/test_semantics.py) |
| TimedDAG ⊂ PDG | [TD §2](upstream/tide-core-5/timed-dag-region-selector-learning-note.md#2-固定结构时间节点图端口与区域划分)、[PDG §1](upstream/tide-core-5/positive-delay-graph-finite-cut-learning-note.md#1-正时延图与共同局部函数)：固定节点图无环；包含关系不改变时间、标签、状态或值；region 商图不必无环 | [frontier 契约](frontier-contract.md)、[跨类策略](cross-family-policies.md)、[native Frontier](../cpp/src/frontier.cpp) |
| SettleGraph → TimedDAG | [Settle §10](upstream/tide-core-5/settlegraph-learning-note.md#10-嵌入-timeddag)：一般端口绑定原样嵌入；同秩独立区域、任意合法端口日程；旧广播/汇总另列绑定 | [逐观察量映射](settle-embedding.md)、[Python encoding](../python/tidegraph/settle.py)、[C++ frontend](../cpp/src/settle.cpp)、[C++ projection](../cpp/src/settle_projection.cpp)、[独立 recurrence](../cpp/test/settle.cpp) |
| 独立特化 | 定义域内固定结构的独立依赖顺序；共享局部公式不等于共享调度 | [Python specializations](../python/tidegraph/specialized.py)、[C++](../cpp/src/specialized.cpp)、[适用域](settle-embedding.md)、[拓扑测试](../tests/test_specialized_topologies.py) |
| streaming / prefill | [Batch §2](upstream/tide-core-5/timed-dag-chunk-prefill-learning-note.md#2-作用块外部边界与精确联合求值)–5，[PDG §10](upstream/tide-core-5/positive-delay-graph-finite-cut-learning-note.md#10-有限-cut-上的因果作用块与最大前沿)：streaming 是合法单作用进展；prefill 是当前实际信息下的契约相对贪心批 | [online greedy](greedy-scheduler.md)、[Greedy](../cpp/src/greedy.cpp)、[就绪计划](../cpp/src/greedy_plan.cpp)、[测试](../tests/test_greedy.py) |
| continuation | [TD §11](upstream/tide-core-5/timed-dag-region-selector-learning-note.md#11-在逻辑时间切面停止与继续)，[PDG §6](upstream/tide-core-5/positive-delay-graph-finite-cut-learning-note.md#6-在完整逻辑时间切面停止)–7，[Batch §8](upstream/tide-core-5/timed-dag-chunk-prefill-learning-note.md#8-切面组合因果性与分段继续)：cut、节点状态、history、在途消息和输入 ledger；窗口必须正确组合 | [cursor](streaming-cursor.md)、[checkpoint](checkpoint-ownership.md)、[坐标测试](../tests/test_coordinate_checkpoints.py) |

Settle `ports` 在任意逻辑 cut 原样保留状态和 pending；`broadcast_sum` 的旧边界投影
仅在完整位置 cut 成立。一般端口、独立根 VJP 和每切点续接见
[test_settle_ports](../tests/test_settle_ports.py)；独立 C++ 公式见
[settle_ports.cpp](../cpp/test/settle_ports.cpp)。
特例扩展须给出参数别名、状态、消息身份、输出和梯度根的映射，不能只称“同一拓扑”。

## 批量计算、反向与设备实现

- `joint_batch`、`exact_sequence`、`joint_sequence`、`comparison_identity` 等是
  **声明并验证的能力**，不是每个合法局部函数天然具有的性质。缺少合法批量契约时
  使用报告出来的 streaming/causal 后备；不能预计算真实轨迹后离线分批。
- State/Read/Full/Aggregate 的批量 VJP 保留公开根连接与版本检查。
  [packed autograd](packed-autograd.md)、[Full](full-batched-autograd.md)、
  [Aggregate](aggregate-batched-autograd.md) 说明 replay 与批量反向边界。
  HST 是指定 VJP，不是 hard forward 的普通导数；高阶支持不能从一阶测试推断。
- eager 的 host 调度与张量设备位置分开；[placement](execution-placement.md)
  配置 Read/控制/选择/载荷。切换精度保留数学操作与离散规则，浮点近似平票
  可能改变轨迹，原严格失败仍单列。
- [常驻库](resident-library.md) 和 [训练 owner](resident-training.md) 使用
  `tools/device_online` 的设备队列、动态选择、journals、显式 VJP 和优化器。
  [控制接口](../tools/device_online/control_flow.h) 由 CANN/CUDA 独立实现。
  自定义 Python/C++ 程序不会自动被降到设备；前向、VJP、容量/续接分别补齐。
- CUDA 共享语义源码不等于已真机认证；[device-control](device-control.md)
  记录独立后端、构建要求及尚待真机门禁。

## 新定义或新特例的最小变更检查

1. 固定教材提交与具体定义/公式，说明定义域、值域和观察量是否变化。
2. 检查现有表示能否表达；如果只能表达受限实例，明确限制或扩展接口。
3. 沿本页依赖追踪 Python 参考、C++、特化、batch/VJP、常驻、比较/序列化。
4. 先加手算或独立 recurrence/调度对照，再修改实现；验证反例和拒绝边界。
5. 更新 profile/图身份/checkpoint 版本的实际需要分别判断，并声明旧配置兼容性。
6. 只运行受影响的必要门禁；性能以明确问题和有限预算开展，不重启旧队列。

工程辅助函数沿所在模块追溯即可；本页不为内存管理、日志、构建脚本虚构教材同名定义。
