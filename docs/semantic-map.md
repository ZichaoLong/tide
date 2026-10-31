# 数学语义 → 表示 → 实现 → 验证

本页源码定位使用清理后的提交号；历史实测身份仍在原报告中，见[迁移与映射](history-migration.md)。

[core-5候选/配置/拓扑/维护入口](core5-study.md)连接公共planning、真实材料化与有限目录。
[配置事实](../python/tidegraph/planning/facts.py)从全部节点/区域配置生成窗口、尺寸、来源分类与共享摘要；
[覆盖检查](../tests/test_core5_coverage.py)与[代表训练](../tests/test_core5_recipe_training.py)核验实际结构和模块，测试数量不替代定义域覆盖。[f456160资格](evidence/core5-coverage-correction-20261030.md)及[逐配置索引](evidence/core5-index.json)给出实际小图范围。
[自然维度有限测量](evidence/core5-natural-study-20261029.md)单列完整通过、严格失败和实际区域合批；
[native文件预算](native-graph-checkpoint.md)与[固定资格](evidence/core5-checkpoint-budget-20261029.md)
对应精确raw状态持久化，预算改变不改变数学或codec格式。查询历史审阅证据不执行参考补写。

[有限性能与停止证据](evidence/core5-boundary-performance-20261029.md)连接单行VJP、异形字段packing、常驻因子KV、实际训练/恢复与分段成本；数值严格失败不冒充吞吐资格。

[三项审查修复资格](evidence/core5-review-fixes-20261028.md)补验ClockedState带权VJP、quad正支持拒绝与resident参数bank分配前准入；数学/图/checkpoint版本保持。

[公共性能消费者](weighted-performance.md)只使用静态参数/输入和候选实际结果，
[参数清单与匹配用例](weighted-workloads.md)逐角色核对固定公式、真实共享、二进制材料和容量下界；
[有限运行证据](evidence/core5-weighted-performance-20261028.md)区分完整记录通过、严格拒绝、profile和吞吐。
[参数匹配/历史经验迁移证据](evidence/core5-performance-transfer-20261028.md)把逐角色清单、
同源材料、共享存储与复制优化连到独立资格和有限测量；大图FP32及分配拒绝单列，不扩大语义资格。

[Weighted公共训练](resident-weighted-training.md)定义完整字段根、跨窗token、参数发布、四分区预算与独立checkpoint绑定；运行资格单列。

其FP16存储/FP32算术、伴随与master边界由同页规定；独立精度参考和单节点KV手算根
分别核验完整记录与局部数学，不能由旧eager half轨迹或投影比较代替。

[带权样本副本](resident-weighted-peers.md)把独立私有状态和共享参数贡献分开，通过
完整参数梯度和与共同提交保持一个optimizer边界；单样本跨卡节点切分未由此接口实现。

[单样本节点放置](resident-weighted-placement.md)另行实现完整纤维请求、全区域选择与共同提交；
前向/精确恢复已有[固定源码资格](evidence/core5-weighted-node-placement-20261028.md)。
放置后的窗口VJP、公共训练和optimizer发布见[固定窗口反向/公共训练资格](evidence/core5-weighted-node-training-20261028.md)，包含跨放置恢复及数值拒绝后的续接。

[Weighted保留窗口](resident-weighted-retained.md)描述实际journal快照、版本检查与跨切面VJP连接；其公共生命周期由[训练owner](resident-weighted-training.md)组合并独立验收。

[Weighted实际窗口VJP](resident-weighted-window-vjp.md)组合设备journal逆序、消息与状态前驱及canonical参数累加；公共训练owner另验收。

[Weighted消息反向路由](resident-weighted-reverse-route.md)从实际journal恢复逐边祖先和独立边界根，完整训练owner另行验收。

[常驻完整数值阶段 VJP](resident-weighted-stage-vjp.md)绑定真实局部 tape、三种快照、完整区域和 canonical 参数角色；跨窗口训练 owner 另行验收。

[完整 weighted 常驻图](resident-weighted-flow.md)记录已实现的设备队列、五记忆、
参数角色、完整返回与原始状态续接绑定；实际资格状态从 STATUS/证据进入。

[Resident recurrent numerical functions](resident-weighted-recurrent.md) implement the four recurrent Upd/Mem/Full formulas with exact-time snapshots, fixed-time convolution and independently rooted saved adjoints. Full graph lifecycle is separately qualified.

带权常驻分配的完整区域表示、精确解VJP、CPU独立公式与设备检查入口见
[resident weighted allocation](resident-weighted-allocation.md)。组件资格不代表全图常驻完成。
[常驻发送原语](resident-weighted-sending.md)对应Attention/比例/分数填充/硬选择的实际buffer、
独立精确/分支VJP与验证入口；完整组合分别由[Region](resident-weighted-region.md)与[图owner](resident-weighted-flow.md)负责。
[常驻ScalarHead/Query](resident-weighted-head.md)定义厂商批量算子、动态owner共享参数、
独立根连接与实际权重版本的显式VJP。
[常驻完整区域组合](resident-weighted-region.md)接完整描述、head、十种发送、实际支持/history及字段/参数VJP。

[常驻完整Aggregate](resident-weighted-aggregate.md)对应物理原子/源参数bank、vendor归一化/变换和完整纤维VJP。
按物理入边/端口压缩局部槽的[固定资格/实际成本](evidence/core5-compact-fibers-20261029.md)验证全局原子身份、前后向及peer布局；厂商流资源拒绝另列。

[常驻来源低秩链](resident-weighted-source.md)对应有效A/B与真实基座/因子的一阶VJP及实际保存版本。

[常驻canonical参数归约](resident-weighted-parameters.md)汇总所有模块/角色实际别名根并连接设备optimizer/发布。

[常驻原始状态与缓存作用](resident-weighted-state.md)对应五记忆typed布局、精确解码、原始时间缓存和独立字段VJP。
[常驻固定时间短卷积](resident-weighted-convolution.md)对应原始时间tap打包、vendor归约、私有cache/共享参数VJP和新旧快照组合。
[常驻共享投影](resident-weighted-projection.md)对应可选N_d、vendor BMM、独立行根/共享参数VJP及保存版本。

[常驻KV接收更新](resident-weighted-kv-update.md)对应实际基座/低秩/摘要参数链、原始缓存和完整独立字段VJP。

[常驻带权归约](resident-weighted-pool.md)对应sum/Attention数值、独立向量/总量返回及保存版本一阶VJP。

[常驻KV原始时间窗口/sum记忆](resident-weighted-kv-sum.md)对应精确时间mask、四个Mem返回及私有raw字段VJP。

[常驻相对位置旋转](resident-weighted-rotary.md)对应精确age到FP32相位、vendor Sin/Cos及保存线性VJP。

[常驻KV内容检索](resident-weighted-kv-query.md)对应独立Q/K/V、Read相对旋转、两通道Attention和四个Mem根的完整VJP。

[常驻Full残差](resident-weighted-residual.md)对应实际内容/记忆输入、SwiGLU及四个参数角色的一阶VJP；
[常驻KV Full](resident-weighted-kv-full.md)组合实际选择、候选快照、多头Attention/位置与完整未缩放内容G。

[常驻完整Describe](resident-weighted-describe.md)对应实际Mem四字段、cat/match、两种evidence/key及三类增益的独立字段/参数VJP。

本页是导航与实现域说明，不另定义数学语义。固定版本与逐字节校验见
[upstream.json](upstream.json)；`python scripts/check_upstream.py` 可离线核验。
[semantics](semantics.md) 拥有本地公式、精度、梯度和序列化版本。
以下正典与局部教材固定到 core-5 / `0770593`，不跟随上游工作区草稿。
采用来源不等于全部候选已完成；实施与证据边界见 [ROADMAP](ROADMAP.md)。

[有限性能消费者](weighted-performance.md)规定共同输入、完整计时与事后独立记录对照；
测量不扩展数学资格，profile也不作为吞吐样本。

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
| 区域函数与配置 | [allocation](upstream/tide-core-5/local-semantics/weighted-message/region-allocation-learning-note.md) 附录 S、§2–6 | [十种发送/增益与独立反向](weighted-sending.md)、[完整评分/证据/key](weighted-scoring.md)和 [quad/bern/ent 精确解 VJP](weighted-allocation.md)已接入；各候选与后端资格由模块文档/证据定位，剩余边界见 ROADMAP，不能复用旧 HST 身份 |
| 记忆 | [KV](upstream/tide-core-5/local-semantics/weighted-message/kv-memory-learning-note.md)、[循环](upstream/tide-core-5/local-semantics/weighted-message/recurrent-memory/affine-delta-learning-note.md)、[GDN](upstream/tide-core-5/local-semantics/weighted-message/recurrent-memory/gated-delta-learning-note.md)、[SSD](upstream/tide-core-5/local-semantics/weighted-message/recurrent-memory/ssd-learning-note.md)、[卷积](upstream/tide-core-5/local-semantics/weighted-message/recurrent-memory/causal-convolution-learning-note.md) | KV 按 [绑定契约](weighted-message.md) 接入；[循环/短卷积](weighted-recurrent.md) 对应全部状态、函数、快照和独立公式；旧 Attention/SSM/Delta 不冒充新定义 |
| 实例 | [FC](upstream/tide-core-5/instances/fully-connected-layered-weighted-message.md)、[HB](upstream/tide-core-5/instances/hb-lattice-weighted-message.md) | [首组合](weighted-message.md)接入 FC/HB、非对齐 DAG/反馈 PDG；HB 同秩不改写时间；各后端证据独立 |
| 验收 | [核对 §1.1–1.4](upstream/tide-core-5/reference/weighted-message-implementation-checks.md) | 定义→配置/参数/快照→符号→独立测试→固定源码证据；各后端分别记录 |

以下已有局部 profile 仍是明确的独立绑定，不自动变成带权支线。

## 局部函数的域、值域与依赖

带权局部 Upd 的 [精确联合/实际 packing 对应](weighted-packed.md)独立列出
KV ragged 投影、四种循环的仿射前缀和原始时间卷积；
[State 批量 VJP](weighted-state-vjp.md)独立声明嵌套参数、结构分组、输出槽与运行资格；一般ClockedState组合的递归捕获与独立根由`test_weighted_clocked_vjp.py`核对。
带权 [Aggregate 批量计算/VJP](weighted-aggregate-batch.md)单列原子/class/fiber 布局、独立根和零总量分支。
[Read/Full 批量数值求值](weighted-query-batch.md)列出完整快照、KV 可见性、卷积槽和字段解码；
[Read/Full 批量 VJP](weighted-query-vjp.md)逐项声明私有图、公开字段/行根、共享参数与资格边界。
[区域完整批量/VJP](weighted-region-batch.md)对应分配分支、control 根、history 能力和在线调度。
[来源函数四种参数绑定](weighted-source-binding.md)对应 A/B/embedding 别名、私有低秩因子、类别宽度及训练恢复。
[Aggregate 计算复用](weighted-compute-sharing.md)对应相同参数/输入的充分判据、物理来源解码与所有使用者反向。
[KV 基座投影复用](weighted-kv-sharing.md)对应临时准备操作数、节点私有适配器/状态及全部使用者 VJP。
[不同输入矩阵批](weighted-kv-sharing.md#不同输入与基座的矩阵批)保持KV §1.2逐行参数/epsilon与独立根，
由 `kv_independent_batch` / `prepare_independent_kv` 和 `test_weighted_independent_preparation` 对应，
不是扩大计算复用判据；预算、完整图与实际运行资格分列。
[8d97020独立资格/有限对照](evidence/core5-independent-projection-20261029.md)记录CPU与两套NPU ABI、
D64 FP64完整梯度/optimizer/精确恢复及未见明确加速的单次测量；原FP32边界不改。
[带权执行位置](weighted-placement.md)对应结构化 Read 临时快照、control/selection 分置、精度与独立转移 VJP。
[带权设备检查](weighted-device-validation.md)对应实际 mixed 执行、精确 int64 排序、独立状态根、跨 owner 续接及候选 profile。
[独立LibTorch FP32 master与typed checkpoint](native-master-optimizer.md)对应显式精度策略、具名owner和格式迁移。
[带权 eager FP16 契约](weighted-precision.md)对应显式评分精度、归一化的一阶收缩与尚未解除的严格训练见证。

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
| streaming / prefill | [Batch §2](upstream/tide-core-5/timed-dag-chunk-prefill-learning-note.md#2-作用块外部边界与精确联合求值)–5，[PDG §10](upstream/tide-core-5/positive-delay-graph-finite-cut-learning-note.md#10-有限切面上的因果作用块与最大前沿)：streaming 是合法单作用进展；prefill 是当前实际信息下的契约相对贪心批 | [online greedy](greedy-scheduler.md)、[Greedy](../cpp/src/greedy.cpp)、[就绪计划](../cpp/src/greedy_plan.cpp)、[测试](../tests/test_greedy.py) |
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
  记录独立后端、构建要求和分层门禁；[RTX3090单卡资格](evidence/core5-cuda-resident-20261031.md)
  与[实际时间线/有限对照](evidence/core5-cuda-finite-20261031.md)限定各自已测配置，不能认证全部GPU范围。

## 新定义或新特例的最小变更检查

1. 固定教材提交与具体定义/公式，说明定义域、值域和观察量是否变化。
2. 检查现有表示能否表达；如果只能表达受限实例，明确限制或扩展接口。
3. 沿本页依赖追踪 Python 参考、C++、特化、batch/VJP、常驻、比较/序列化。
4. 先加手算或独立 recurrence/调度对照，再修改实现；验证反例和拒绝边界。
5. 更新 profile/图身份/checkpoint 版本的实际需要分别判断，并声明旧配置兼容性。
6. 只运行受影响的必要门禁；性能以明确问题和有限预算开展，不重启旧队列。

工程辅助函数沿所在模块追溯即可；本页不为内存管理、日志、构建脚本虚构教材同名定义。

完整独立 C++ 数值 checkpoint 的切面、int64、参数/optimizer 与独立训练追溯见
[native-graph-checkpoint](native-graph-checkpoint.md)。

私有同形 KV 投影进一步合批仍对应 KV §1.2 / FC §6.2，入口与独立根/VJP、实际矩阵批统计见
[KV 计算复用](weighted-kv-sharing.md#独立私有行的一阶边界)；共同评分偏移的实际响应修复见
[分配契约](weighted-allocation.md)。两项不改变图 v13/checkpoint v5。

分配§6/S.2的数值坐标进一步对应`quadratic_coordinates.coordinates`、C++ `detail::quad_pivot`
与设备`weighted_quad_coordinates.h`：按cap质量分位评分建立局部根区间。返回全部原始响应、
exact-map VJP和拒绝阈值不变；`test_weighted_quad_scale`保留两份实际D2048失败行并独立手算，
`test_weighted_large_shift`核对共同阈值消去，`diagnose_allocation_failure.py`使用独立Decimal。
失败快照的公开结构、行号/节点/int64时间和编码由[分配](weighted-allocation.md)与
[测量设施](weighted-workloads.md)说明；开发检查、固定资格和完整模型的严格失败由[137f530报告](evidence/core5-local-allocation-factorized-kv-20261028.md)分别列出。
早期[6612ed3资格/有限对照](evidence/core5-private-projection-20261028.md)覆盖共同偏移响应与私有矩阵批，
原D64训练和D2048残差失败仍保留在该历史报告；新的局部根区间资格对应137f530，不能回填到早期来源。

全部eager KV行的低秩式及独立dense公式/Jacobian由[KV契约](weighted-kv-sharing.md)定位；
[137f530固定资格](evidence/core5-local-allocation-factorized-kv-20261028.md)覆盖该改动与局部分配修复。
大图已越过原残差拒绝但完整状态/训练仍严格不一致，不把局部解准确或FP64诊断扩展为FP32通过。

FC §6.2的共享准备仍由相同输入/参数身份许可；`prepare_shared_state`按首个事件稳定求值，
opaque key只作相等判断，不能以内存地址决定反向图创建次序。
独立C++反例见`weighted_kv_sharing.cpp`（故意让key顺序与事件顺序相反）；
两dtype/Python与native的换owner下一窗口完整记录、梯度和optimizer见
`test_weighted_workload_training.py`，要求实际checkpoint解码而非参考补写。
CPU显式FP64测量、最终梯度/SGD槽验证及同材料精度/预算边界见[性能契约](weighted-performance.md)
与[工作负载设施](weighted-workloads.md)；它们不解除既有FP32严格失败。

[0f22ebc固定证据](evidence/core5-stable-preparation-precision-20261028.md)覆盖共享准备稳定顺序、
精确续接和真实optimizer观测；CPU整体572通过/2原FP32 GDN失败保持分列。
D64/D2048显式FP64通过不解除原FP32边界；新增GDN近零梯度见证由[精度契约](weighted-precision.md)追溯。

GDN §1 epsilon-L2、GDN/SSD §3 scaled-RMS的FP32一阶收缩由Python/C++
`weighted_normalization`实现；FP32前向不变、FP64路径不变，独立实数Jacobian及广播/零根由
`test_weighted_half_normalization.py`和C++ `weighted_normalization.cpp`验证。
原GDN完整AdamW/checkpoint测试保持输入和容差，局部通过不替代整图资格。
[9e6dc52固定资格](evidence/core5-fp32-normalization-20261028.md)记录原FP32见证通过、
FP16严格失败保留，以及CPU/纯Python NPU/standalone NPU/CUDA构建的不同验证范围。

Architecture §1 RMS和各线性映射的[算术边界诊断](weighted-arithmetic-diagnostics.md)
由`weighted_arithmetic_probe`显式隔离CPU FP32乘加/归约/normalized VJP；
`test_weighted_arithmetic_probe`以相消手算、literal Jacobian和实际checkpoint验证诊断机制。
这不是新的公共数学绑定或生产精度策略，默认Python/C++函数和图/checkpoint版本不变。

本次执行层增量：Architecture §2–4的单行一阶边界对应`singleton_vjp.boundary`/
`SingletonVjp`及Query/Aggregate独立根测试；异形字段copy见[packing](weighted-packed.md)。
Allocation §6/S.2的标量拒绝/原生实际调用观察见[分配契约](weighted-allocation.md)，
诊断返回不构成部分成功，也不改变图v13/checkpoint v5。

常驻KV §1.2低秩映射对应`append_weighted_kv_update`与其独立VJP，
实际B/V/U tapes、域/尺寸/旧新raw解码见[KV接收](resident-weighted-kv-update.md)。
`weighted_kv_update_check`使用独立逐行dense公式与独立字段根；公共训练/恢复另验。
常驻Allocation §6/S.2错误20–22的实际部分行与无效owner见[分配](weighted-allocation.md)，
由`weighted_flow_boundary_check`及`test_resident_weighted_flow`真实下溢反例检查。
实际梯度/SGD槽/更新次数解码、磁盘换owner精确续接和主机反向阶段计量见
[测量设施](weighted-workloads.md)与[常驻训练](resident-weighted-training.md)，不调用参考补写返回。

上述resident增量与实际消费者的[57308e7固定资格](evidence/core5-resident-boundary-20261029.md)
分别记录SDK、Python-owned、原失败和初始化边界；该历史来源没有CUDA真机资格。
后续[RTX3090单卡资格](evidence/core5-cuda-resident-20261031.md)及[时间线/有限对照](evidence/core5-cuda-finite-20261031.md)
使用各自固定来源和显式库存，不追溯提升旧证据。

公共配置/参数清单/自然规模与维护映射见 [core5-study](core5-study.md)。
`planning.model.resolve_model`复用固定Config适用域；`planning.recipes`绑定FC/HB实例§4.2—4.4，
`test_core5_planning`独立核对拓扑计数/边/时钟和实际五族/混合参数；不作为真机性能资格。

有限候选目录、完整执行身份/结果门禁及自然recipe多步核对由[core5-study](core5-study.md)进入。
resident半精度参考的显式存储/算术及串行换owner验收见[性能契约](weighted-performance.md)；
独立参考不参与候选调度或补写解码结果。

[GPU规模推理](evidence/core5-cuda-inference-20261031.md)连接不齐输入生成器、显式FP64、完整前向/精确恢复、实际packing观察与[完整返回传输策略](complete-return-packing.md)；原数学/图/checkpoint版本不变，规模推理不认证训练。
