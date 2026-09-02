# SettleGraph 实现与等价性验证计划

> 本文是实现计划，不是新的神经网络语义来源。
>
> [实验语义、命名与数学符号](experiment-semantics-and-naming.md) 是 SettleGraph 计算含义的权威文档。若实现计划与语义文档冲突，以语义文档为准；若语义文档不足以唯一决定实现行为，应先对齐语义，而不是在代码里自行选择一种解释。

## 1. 目标与完成标准

首轮实现同时建设三条执行路径：

| 执行路径 | 作用 | 是否必须覆盖任意合法 Plan |
| --- | --- | ---: |
| 逐 Token 解释器 | 最直接地复现语义文档第 2.4 节，作为正确性基准，也可用于 decode | 是 |
| 通用 packed prefill 执行器 | 批量处理一段 Token，避免按 Token、样本或 node 发起大量 Python 调用 | 是，具体性能能力按算子记录 |
| 特化 prefill 执行器 | 利用单层、HB-Lattice 等 Plan 的额外规则提高效率 | 否，只接受自己声明支持的 Plan |

这三条路径必须复用同一份规范化 Plan、参数、输入、初始状态和随机数规则。特化执行器不是另一套模型语义；它只是通用执行的优化实现。

这里的“通用”首先表示不限制合法拓扑。任意合法 Plan 都必须能在逐 Token 和通用 prefill 接口下执行；但任意新加入的自定义算子不会自动获得高性能，其 reference、packed 和设备优化能力必须分别验证。

“函数等价”至少包括：

- 每个有效 Token 的 SettleGraph 输出；
- 最终 receiver state 和 selector-history；
- reached、Observe、active、发送以及边结算结果；
- selector logits、soft probability 和辅助损失；
- 训练模式下的语言模型损失以及选定输入和参数的梯度；
- padding 旁路、chunk continuation、状态 reset 和 detach 行为。

在完成执行器等价性、CPU/NPU 一致性和基本训练测试以前，不开始新的科学实验。性能优化不能代替这些正确性门槛。

## 2. 共同的数据契约

### 2.1 一次 prefill 的输入与输出

对单个 site 上的一次 SettleGraph 调用，通用接口至少接收：

| 输入 | 含义 |
| --- | --- |
| \(H^{\mathrm{in}}\in\mathbb R^{B\times T\times d_{\mathrm{model}}}\) | \(B\) 条序列、每条至多 \(T\) 个 Token 的图输入 |
| `valid_mask`，shape 为 \([B,T]\) | 标识真实 Token；无效位置旁路且不更新状态 |
| `sequence_id`，长度为 \(B\) | 跨 chunk 稳定标识每条序列 |
| `token_position`，shape 为 \([B,T]\) | 同一序列内跨 chunk 不重置的全局 Token 位置 |
| 初始状态 | 按语义文档第 2.5 节的键读取的 receiver state 与 selector-history |
| 规范化 Plan 与参数 | 固定拓扑、局部运算及其参数 |

同一 `sequence_id` 的有效 `token_position` 必须严格递增。一次调用中不应出现两段无法确定先后顺序、却写入同一个状态键的数据。

执行结果至少包含：

- \(B_{\mathcal G}\in\mathbb R^{B\times T\times d_{\mathrm{model}}}\)；
- chunk 结束后的全部状态；
- 训练所需的辅助 loss 统计；
- 可选的执行 trace。完整 trace 只用于小规模测试，正常训练只保留聚合后的诊断量。

逐 Token 解释器一次处理同一批序列的一个位置 \([B,d_{\mathrm{model}}]\)。把它按 \(t=0,\ldots,T-1\) 调用后，应得到与一次通用 prefill 相同的逐 Token 结果和最终状态。

### 2.2 规范化 Plan

原始图描述必须先经过语义文档第 2.4 节的静态校验，再转成规范化 Plan。Plan 至少保存：

- 稳定的 node、edge、region ID；
- node 所属 region、固定父边和子边；
- 入口和终端 receivers；
- region 依赖图及其规范拓扑序；
- Aggregate、Update、两类 Read、Score、Top-K、NodeCompute 和 Emit 的配置；
- 状态、参数、hidden 和读出的 shape/dtype 契约；
- forced-active、\(K^{\max}\) 和动态 \(K^{\mathrm{req}}\) 的规则；
- HB-Lattice 可选的 Line、phase 和边来源标签。

进入同一 receiver 的父消息按稳定 edge ID 排列；一个 region 的 candidates 按稳定 node ID 排列。规范化序列化必须能产生稳定哈希。参数值、运行期 reached/active 结果和某个 batch 的状态不属于 Plan。

为高性能执行派生的 region 批次、CSR 索引、算子分组和缓存生命周期称为编译后调度信息。它可以重新生成，不能改变规范化 Plan 的含义，也不能取代 Plan 哈希。

### 2.3 参数与算子实现

每项局部运算分成两层：

1. **语义配置**：说明数学上计算什么，必须能对应到语义文档中的公式或明确的自定义公式。
2. **实现变体**：说明使用 eager Torch、packed Torch、编译图或某个设备自定义算子来完成同一计算。

每个语义配置至少有一个标准 Torch 参考实现。可选优化实现必须声明支持的 device、dtype、shape、forward/backward 和布局；不支持时只能显式选择已经验证等价的参考实现，或明确失败，不能暗中换算法、换 dtype、转 CPU 或丢失梯度。

各执行器应直接读取同一组参数 Tensor。为了 grouped GEMM 或批量状态更新，可以为具有相同算子签名的 nodes 堆叠参数；不同执行器不能维护彼此独立、可能逐渐失配的参数副本。参数共享若被某个实验启用，也必须通过明确的参数索引表达并验证梯度累加。

## 3. 三条执行路径

### 3.1 逐 Token 解释器

逐 Token 解释器忠实实现语义文档中的 `InterpretToken` 顺序：

1. 为当前 Token 建立入口消息；
2. 按 region 依赖的合法拓扑序推进；
3. 等固定父边全部结算后，按 edge ID 收集 `DATA`，忽略 `CLOSED`；
4. 完成 Aggregate、Read、Score、Top-K、Observe/commit、NodeCompute 和 Emit；
5. 结算固定出边并最终聚合 active 终端输出。

首版允许在 Python 中按 region、node 和 edge 循环，因为它的首要职责是清楚、可检查和适合逐步调试。Tensor 数学仍使用标准 Torch，使 CPU float64、autograd 和 NPU eager 路径能复用同一代码。

解释器应支持可控 trace，至少能记录每个 \((b,t,\mathrm{region},v)\) 的 reached、Observe、active、\(p\)、输入 hidden、输出 hidden 和状态摘要，以及每条边最终是 `DATA` 还是 `CLOSED`。trace 的排列必须只依赖稳定 ID，便于逐项比较。

### 3.2 通用 packed prefill 执行器

通用 prefill 仍按 Plan 的 region 依赖顺序推进，但一次处理一个 region 或一组兼容 regions 的整个 \([B,T]\) 工作，而不是先完整执行 Token 0、再执行 Token 1。

当某个目标 region 的全部数据依赖和控制依赖 regions 已经处理完以后，所有固定父边对整个 chunk 都已经逻辑结算。实现只需保存实际的 `DATA` 记录；在此前提下，没有 `DATA` 记录的固定父边可等价视为 `CLOSED`，不必物理存储大量 `CLOSED` Tensor。

一条实际消息可表示为一行记录：

```text
hidden       [M, d_model]
sequence_row [M]
token_pos    [M]
source_node  [M]
dest_node    [M]
edge_id      [M]
gate         [M]       # 需要时保存
scatter_id   [M]       # 恢复目标位置或诊断使用
```

\(V\) 和 \(E\) 分别是 Plan 中的 receiver 与固定边集合，\(M\) 是当前实际消息数。实现不应构造 \([B,T,|E|,d_{\mathrm{model}}]\) 或 \([B,T,|V|,d_{\mathrm{model}}]\) 这种与全部固定边或 nodes 成比例的巨大 hidden Tensor。

入口广播与终端聚合使用独立的边界记录，不需要为了实现方便向规范化 Plan 虚构新的语义边。

通用 prefill 的主要步骤是：

1. 按目标 node、样本、Token 和 edge ID 对实际消息分组；
2. 对每个 reached node event 执行 Aggregate，形成 \(h_{v,b,t}\)；
3. 按 region、样本和 Token 形成 candidate events；
4. 根据 profile 和 selector 时序执行状态更新与选择；
5. 把 active events 按 node 及算子签名重排，批量执行较大 Read 和 NodeCompute；
6. 把输出映射到固定子边，供后续 regions 使用；
7. 按稳定 node ID 聚合终端记录。

常用的设备内积木包括布尔筛选、`nonzero`、稳定排序、计数、前缀和、`gather`、`scatter`、`index_add`、分段归约和 grouped GEMM。具体算子在 CPU 与 NPU 上都必须做真实 forward/backward 探测。

#### 3.2.1 两种 packing 视图

同一批路由记录至少需要两种重排视图：

| 视图 | 分段键 | 段内顺序 | 用途 |
| --- | --- | --- | --- |
| 状态视图 | \((\mathrm{sid},v)\) | 全局 Token 位置递增 | receiver Update 与状态 Read |
| 计算视图 | node ID 与算子签名 | 任意确定顺序 | node-specific FFN、投影和 grouped GEMM |

状态视图使用扁平 `values`、长度为“段数 + 1”的 `offsets`、每段的 owner，以及每行的原始 Token 位置。不同段可以并行；一个段内部仍遵守因果顺序。原始 Token 位置只在语义公式需要时作为输入，不能因为 packing 自动改变 EMA 等 Update 对“跳过 Token”的定义。

对 region selector 而言，静态 region 宽度有界。令 \(N_{\mathrm{event}}\) 为当前批次的选择事件数、\(R_{\max}\) 为同一算子组的最大 region 宽度、\(d_r\) 为 selector 读出维度；实现可以使用带 mask 的 \([N_{\mathrm{event}},R_{\max},d_r]\) 小型稠密表示，也可以使用 offsets 表示变长 candidates。两者都必须保持稳定 node ID 顺序和完全相同的 Top-K 平票规则。

传统的 `pack_padded_sequence` 主要服务 RNN，不能单独表达这里按 \((\mathrm{sid},v)\) 划分的多段状态、region 选择和 node-specific 参数。Nested Tensor 或其他 ragged 容器可以作为实现变体，但不能取代本节的显式 owner、offset、时间顺序和 scatter 契约。

#### 3.2.2 不同 profile 的因果难度

“没有 Python Token 循环”不等于“数学上没有跨 Token 递归”。不同语义需要不同的 packed 实现：

| 条件 | 是否可先得到完整 Observe/active 列表 | 合适的实现 |
| --- | ---: | --- |
| N + content-only | 是 | 对整个 region/chunk 批量 Score、Top-K 和计算 |
| SD + content-only | 是 | 先批量选择，再按 \((\mathrm{sid},v)\) 打包 active Token 做 Update |
| BO + content-only | Observe 列表是 reached 列表 | 可先批量选择，再按 \((\mathrm{sid},v)\) 做分段状态更新 |
| BO + pre/post，无选择历史 | Observe 列表是 reached 列表 | 按 \((\mathrm{sid},v)\) 扫描得到对应时刻的 pre/proposal 读出，再批量选择 |
| SD + pre-update | 否 | 按 \((\mathrm{sid},\mathrm{region})\) 融合“读旧状态—选择—提交”递推 |
| 带选择历史的负载感知 selector | 通常否 | 按 \((\mathrm{sid},\mathrm{region})\) 做 selector-history 递推 |

EMA 等更新适合 segmented/prefix scan；Gated DeltaNet/KDA 可以采用 recurrent 或 chunkwise kernel；Attention 状态适合 varlen causal attention。任意自定义 Update 未必能变成并行 scan，但仍可把段内循环放在编译图或设备 kernel 内，而不是为每个 Token 发起一次 Python 调用。

通用 prefill 必须为每个算子变体记录能力状态：

- `reference`：语义正确但不作性能承诺；
- `packed`：不按 Token、样本或 node 进行 Python 热循环；
- `optimized`：经过目标设备 profiling 的编译或融合实现。

某个自定义算子只有 `reference` 实现时，执行器可以在显式 reference 模式下运行它，但不能把该组合报告为高性能 packed 路径。首轮核心实验所采用的全部算子必须先达到 `packed`，NPU 主路径还必须证明关键操作没有 CPU fallback。

#### 3.2.3 调度与内存

Plan 编译阶段应把相同状态算法、状态 shape、selector 形式、NodeCompute shape 和 dtype 的 nodes/regions 分组。运行时允许按少量算子组循环，不应按每个 Token、每个样本或每个 node 发起细粒度 Python 调用。

跨多层的提前消息保存到目标 region 可见的缓冲区。每条消息在最后一个消费者完成后即可释放；可由 Plan 的固定依赖预先计算生命周期。正确性首版可以保留更多中间量，但性能版不得用完整边状态的稠密复制换取简单实现。

训练反向需要保存或重算 packed permutation、segment offsets、active indices 和必要状态。activation checkpointing、chunk detach 或重计算必须作为显式配置，不能因后端不同而改变梯度语义。

### 3.3 特化 prefill 执行器

至少实现两个特化对照：

1. **单层特例**：直接把相同图输入分派给并列 receivers，完成一次 region 选择、node grouped compute 和终端聚合；它应接近平铺 MoE 的执行形态。
2. **HB-Lattice**：读取规范化 Plan 中的 Line 元数据，按 Line barrier 批量处理相互独立的 regions，并利用相邻 Line 和规则边布局减少调度与索引开销。

特化执行器必须消费已经展开并规范化的 Plan，不能只读取 Builder 配置后自行重建一张可能不同的图。它在运行前应检查 Plan 是否满足自己的结构条件：调用者显式指定该特化实现时应直接失败；只有显式的自动选择模式才可回退到通用执行器，并记录原因。

只有同时满足以下条件，特化实现才能保留：

- 与逐 Token 解释器和通用 prefill 的结果、状态、路由及梯度一致；
- 对声明支持的全部算子变体没有隐式语义降级；
- 在目标 workload 上经过同步、预热后的 benchmark 确有价值。

后续还可以增加 forced-active chain、规则树或其他常见 Plan 的特化，但不把它们变成新的执行语义。

## 4. 首轮需要覆盖的语义多样性

不必穷举所有组合，但不能只实现一个恰好能跑通的配置。首轮以一组有意多样化的基元覆盖不同数据流和梯度路径：

| 维度 | 至少覆盖的实现 | 主要验证点 |
| --- | --- | --- |
| Aggregate | mean、learned convex、按 edge 做线性变换后再聚合 | 单父、多父、父边身份和确定顺序 |
| receiver state | none、历史激活、EMA、Gated DeltaNet、窗口 Attention | 无状态、向量状态、矩阵状态、变长历史 |
| selector 时序 | content、pre、post | 当前内容、旧状态、proposal 参与选择 |
| profile | N、SD、BO | 无状态、只更新 active、更新全部 reached |
| Score | 固定/可构造分数、线性、两层 MLP、状态读出参与 | 平票、数值边界和真实可训练参数 |
| active budget | Top-1、Top-2、all、按 region/event 变化 | singleton、候选少于请求 K 和变长 active set |
| Emit | hard、Hard-ST、soft probability | 前向值与 selector 主任务梯度 |
| NodeCompute | 简单 affine 测试算子、SwiGLU MLP、状态读出 + 双 residual | 解析核验、真实昂贵计算和 residual |
| 参数关系 | node 独立参数、显式共享参数组 | 默认独立与共享梯度累加 |
| 状态首值 | 零、固定非零、可学习首状态 | reset、序列隔离和序列 continuation |

窗口 Attention、Gated DeltaNet 等具体公式仍以语义文档附录 A 和实验记录为准。若实现的是另一种算法家族变体，必须使用新的明确配置，不能只复用旧名称。

组合测试采用三层覆盖：

1. 每个基元的针对性单元测试；
2. 对主要语义轴做 pairwise 或小规模穷举组合；
3. 用固定 seed 随机采样更大的合法组合。

这样既能扩大覆盖面，又避免把所有维度做成无法运行的完整笛卡尔积。

## 5. Plan 与输入的测试语料

### 5.1 手工 Plan

至少长期保留以下可读、可人工推导的小图：

| Plan | 要覆盖的情况 |
| --- | --- |
| singleton forced-active | 最小入口/终端、\(p=1\) |
| 单层 \(R=1,2,8\) | Top-1、Top-2、Top-all 和多终端聚合 |
| chain | 多层顺序传播和状态更新 |
| diamond | fan-out、一个父分支关闭、fan-in 聚合 |
| unequal-path | 短路径缓存、skip edge 与不同长度路径汇合 |
| multi-entry/multi-terminal | 图输入广播和图输出聚合 |
| mixed regions | 同一拓扑层多个 regions、singleton 与竞争 region 并存 |
| forced backbone + optional branches | 始终有终端输出，同时产生丰富 active 子图 |
| 小型 HB-Lattice | 扩展、平台、收拢、mirror/local/shortcut 边 |

还应保留故意非法的图，验证 cycle、region 内边、重复边、错误 shape、非法 K、无入口—终端路径、region 依赖环和不稳定/重复 ID 会在执行前失败。

### 5.2 自动生成合法 Plan

自动生成器先生成 region DAG，再在每个 region 内生成互不连接的 receivers，最后只沿 region 拓扑序生成 receiver edges。生成过程应可配置：

- nodes、regions、路径深度和宽度；
- region 大小、fan-in/fan-out 上界；
- 单/多入口、单/多终端；
- 等长和不等长路径、skip/mirror 类边；
- forced-active backbone 与可关闭旁路；
- 每个 region 的 K、profile、selector、Aggregate 和 receiver 类型；
- 同构算子组和混合算子组。

成功样例中的每个 receiver 必须位于入口—终端固定路径上。大多数随机正确性样例应包含 forced-active 终端路径，另有一组样例专门验证“全部终端关闭”会报告配置失败。

生成器同时提供非法变异：对一个合法 Plan 注入一条反向边、region 内边、重复边、错误 ID、错误 shape 或非法操作组合，并验证静态检查器准确拒绝。

小规模测试可以枚举或系统采样 nodes 很少的图；较大图使用固定 seed 随机生成。任何失败都保存最小必要的规范化 Plan、seed、输入、参数、初始状态、运行配置和 executor 名称，使下一次运行无需依赖原随机过程即可复现。

### 5.3 输入与状态样例

Plan 多样性之外，还要交叉覆盖：

- \(B=1\) 与多样本 batch；
- \(T=1\)、短 prefill 和较长 prefill；
- 每条序列不同有效长度、内部 padding 和空尾部；
- 同一个 chunk 内，各 \((\mathrm{sid},v)\) 拥有完全不同的 Observe/active Token 列表；
- 一次完整 prefill、多个不等长 chunks 和逐 Token 输入；
- 零状态、随机状态和从上一 chunk 延续的状态；
- 正常 logits、刻意平票和接近 Top-K 边界的 logits。

## 6. 差分验证方法

### 6.1 比较顺序

所有比较使用同一份 CPU 创建并序列化的 fixture。建议按以下顺序建立证据：

1. 人工可计算的小例子对照逐 Token 解释器；
2. CPU float64：逐 Token解释器对通用 prefill；
3. CPU float32：逐 Token 与通用 prefill 互比，并对照 float64；
4. CPU 上的特化执行器对照两种通用路径；
5. NPU FP32 eager 对照 CPU FP32；
6. NPU packed/optimized 对照 NPU eager 与 CPU；
7. 明确需要后再验证 BF16 等低精度路径。

CPU FP32 是可移植的基础正确性路径；CPU float64 是小规模高精度 oracle。不得从 CPU float64 推断 NPU 也需要或支持 float64。NPU 首先验证 FP32，低精度属于后续独立能力。

### 6.2 要逐项比较的量

同一 backend/dtype 的执行器差分至少检查：

- 输出 shape、dtype、device 和有效 Token 数；
- 每个 Token 的 \(b_{\mathcal G}\)；
- 所有状态键及最终状态值；
- reached/Observe/active/发送 mask 和 Top-K ID，离散量要求完全相同；
- 每个 region 的 logits、probability、balance 统计和 loss；
- 聚合前后的 hidden；
- 输入 hidden、Aggregate、receiver、selector 参数的选定梯度；
- 一次 optimizer step 后的选定参数；
- checkpoint 保存、重新加载后的结果。

浮点 Tensor 按 dtype、算子和数值规模设置绝对/相对误差，不要求不同设备或不同归约 kernel bitwise 相同。平票测试使用可以精确构造的相同 logits，并要求按稳定 node ID 得到完全相同的 Top-K；一般数值一致性样例应让 Top-K 边界留出足够 margin，避免把正常浮点误差误判为路由实现错误。

必要时提供仅用于测试的 route replay：先保存参考 active set，再令另一实现复用它，以分别定位“selector 数值不同”和“给定同一路由后的 receiver 计算不同”。自然路由的端到端一致性仍然必须单独通过，不能只用 replay 代替。

### 6.3 状态、chunk 与随机数

在 eval/deterministic 模式下，以下三种执行应产生相同的逐 Token 输出和最终状态：

```text
完整 prefill
= 任意合法 chunk 切分后的连续 prefill
= 逐 Token 解释执行
```

训练时若 chunk 边界 detach，则梯度只能与采用相同 detach 位置的参考执行比较；forward 和最终状态仍应一致。

首轮等价性测试默认关闭 dropout。以后若加入 dropout、路由噪声或其他随机操作，随机数必须由稳定键决定，例如 seed、site、node、sequence ID、全局 Token 位置和 operation ID，不能依赖 executor 的实际调用先后。否则不同 packing 顺序会生成不同掩码，无法判断是执行器错误还是随机数流不同。

### 6.4 Base 模型接入

SettleGraph 独立测试通过后，再覆盖语义文档第 1.3 节的 POST、PARBLK、PARATTN 和 PARMLP：

- 对每种 placement 检查输入 hidden 与 residual 合入位置；
- identity 初始化时，接入模型与原 Base 模型的输出和选定梯度一致；
- 非 identity 初始化时，逐 Token与 prefill 仍相互等价；
- 多个 sites 的参数和状态互不错误共享。

### 6.5 训练能力

每个核心算子路径至少通过：

- forward 与辅助 loss；
- selected gradients 有限且符合参考；
- Hard-ST 的前向恒等关系和 selector 梯度；
- 一个 optimizer step；
- stateful chunk 的反向与 detach；
- checkpoint round trip；
- 一个很短的过拟合或下降测试。

CPU float64 可用于小图的 `gradcheck` 或有限差分；离散 Top-K 附近不适合直接做连续梯度检查，应使用远离边界的 logits 或固定 route。

## 7. CPU、NPU 与运行时边界

### 7.1 共同运行接口

公共运行接口应统一表达以下概念：

```text
--device auto|cpu|cuda|npu
--device-index LOGICAL_INDEX
--dtype auto|float64|float32|float16|bfloat16
--seed NONNEGATIVE_INTEGER
--output-dir PATH
--init-from PATH
--resume PATH
```

训练和 benchmark 入口必须要求调用者显式给出 `--device`，包括显式写出 `--device auto`；其余选项按入口需要提供。只在项目确实支持相应语义的入口提供 `--init-from` 和 `--resume`，两者互斥。`float64` 首先只承诺 CPU 小规模正确性测试；显式请求未支持的 backend、dtype、算子或 executor 必须失败，不能静默降级。

设备解析、vendor import、同步、autocast、RNG、内存统计和 profiler 放在一个窄运行时边界。模型与 Plan 算法代码从输入或参数派生 device/dtype，不散布 `.npu()`、`.cuda()` 或全局设备探测。

当前必需目标是 CPU 和本机 NPU。CLI 和代码边界保留 CUDA 入口，但在真实 CUDA 机器通过测试以前只能记为 `planned`，不能声称已验证。

### 7.2 CPU 路径

CPU 至少保留两种 dtype：

- FP64：小规模数学、状态和梯度的高精度 oracle；
- FP32：跨平台的标准 Torch 正确性基线和 NPU 对照输入。

CPU 路径必须能在没有 TorchNPU、CANN 或其他 vendor 模块的环境中 import、显示帮助并运行测试。x86_64 与 aarch64 是不同验证目标，不能因其中一端通过就自动宣称另一端通过。

### 7.3 本机 NPU 路径

实际开始实现和验证 NPU 时，同时遵循 `develop-portable-torch` 与 `use-local-ascend` 的最新约定。本机环境激活、物理卡选择和 launcher 规则留在 site 层，不写入模型代码。

NPU bring-up 依次验证：

1. 当前 Torch/TorchNPU/CANN/设备组合可用；
2. 真实 NPU allocation、算子、同步和回传 CPU；
3. 本项目实际使用的 sort/Top-K、mask/nonzero、count/cumsum、gather/scatter、`index_add`、归约、线性层、normalization 和 Attention；
4. 上述操作的边界 shape、空段、尾块、非连续输入和 backward；
5. CPU FP32 与 NPU FP32 的 executor 差分；
6. mixed precision、编译图和自定义 kernel。

关键操作需要 profiler 或等价证据确认实际在 NPU 上执行。数值相同并不能证明没有 CPU fallback。若某个标准 Torch 算子在 NPU 上不可用，应选择经过验证的等价 NPU 实现或开发窄自定义算子；不允许把 Tensor 悄悄搬到 CPU。

### 7.4 编译、自定义算子与 LibTorch

优化顺序为：

1. eager 标准 Torch 参考；
2. 张量化 packing 和少量粗粒度调用；
3. 显式可选的编译图；
4. 只为 profiling 证明的热点增加 CUDA/NPU 自定义算子；
5. 只有独立 C++ 部署或 host 调度仍被证明是瓶颈时，才考虑完整 LibTorch runtime。

完整 LibTorch 不是消除 Python Token 循环的前提。普通 LibTorch 调用和 Python PyTorch 最终使用相同的 ATen/设备 kernel；更常见的有效边界是保留 Python 训练栈，只把 segmented state update、region recurrence 或 packed dispatch 等热点封装成自定义算子。

`torch.compile`、Triton-Ascend、vendor fused kernel 和自定义 C++/NPU op 都是明确的实现变体。它们必须保留 eager oracle，并分别验证 forward、backward、dtype、shape 和无 fallback；CUDA Triton 与 Triton-Ascend 环境不能混用。

## 8. 性能验证

性能测试只在正确性通过后进行，并至少比较：

- 逐 Token 解释器与通用 packed prefill；
- 通用 prefill 与单层/HB 特化；
- eager、编译和自定义算子变体；
- 不同 batch、sequence length、Plan 深度、region 宽度、fan-in/out 和 active 密度；
- none、EMA、Gated DeltaNet 和 Attention 等不同状态成本。

benchmark 必须明确计时是否包含 packing、状态装载、图编译、SettleGraph 外部 Base block 和数据传输。预热在计时外完成，设备计时前后同步，并记录吞吐、延迟分布、峰值内存和各阶段耗时。

通用 prefill 的性能验收至少要求：

- 热路径没有按 Token、样本或 node 的 Python 调度；
- packing、状态更新、selector、NodeCompute 和 scatter 能在 profiler 中分别定位；
- 相对逐 Token 解释器在代表性长 prefill 上有明确收益；
- 中间 Tensor 的规模与实际消息数及有界 region/状态规模一致，而不是与全部可能路径组合数成比例。

特化实现若没有稳定收益，可以删除；正确性不依赖其存在。

## 9. 实现阶段

逐 Token 与 prefill 是同一实现里程碑的两个必需交付物。逐 Token 解释器可以稍早落地以提供 oracle，但不能在缺少通用 prefill 等价验证时进入正式实验。

### 阶段 A：运行时、Plan 与 CPU oracle

- 建立 CPU-safe 的 Python package、设备/dtype 解析和测试入口；
- 定义规范化 Plan 序列化、静态校验、稳定哈希和编译后索引；
- 实现首批 Aggregate、receiver、selector、profile 和 Emit 参考算子；
- 实现 CPU FP64/FP32 的逐 Token 解释器及完整 trace；
- 完成人工 Plan 与非法 Plan 测试。

### 阶段 B：通用 packed prefill

- 实现稀疏消息记录、candidate events、两种 packing 视图和恢复索引；
- 先完成 N、SD content 和 BO 的 packed 路径；
- 再完成 SD pre 与 selector-history 的 region 递推路径；
- 对全部核心算子完成 CPU FP64/FP32 executor 差分和梯度差分；
- 加入随机 Plan、随机输入和失败 fixture 保存。

### 阶段 C：特化执行与 Base 模型接入

- 实现单层和 HB-Lattice 特化执行器；
- 与逐 Token、通用 prefill 做三方差分；
- 接入 Base Qwen block 的四种 placement；
- 验证 identity 初始化、多个 sites 和训练 loss。

### 阶段 D：本机 NPU

- 按本机技能加载环境并建立项目能力探测；
- 先跑 NPU eager FP32，再跑 packed FP32；
- 完成 CPU/NPU forward、state、route、loss、gradient 和 optimizer 差分；
- profile packing 与递推热点，只对确认瓶颈增加编译或自定义算子；
- 再单独验证 BF16、Attention 优化和其他低精度路径。

### 阶段 E：训练与性能就绪

- 完成 checkpoint portable handoff 与同栈 resume 测试；
- 完成短训练和小样本过拟合；
- 完成通用/特化 executor benchmark；
- 固化首轮实验实际采用的已验证算子组合、Plan 和运行命令。

## 10. 项目工件与证据

实现阶段应在仓库中形成以下可审查工件：

- 语义算子参考实现与可选优化实现；
- 规范化 Plan schema、validator、hash 和 Builders；
- 三类 executor 及共同结果/trace 契约；
- 人工、自动生成和非法 Plan 测试；
- CPU golden fixtures；
- `.torch-portability/contract.json`，记录 CPU/NPU 必需目标、数值容差和验证命令；
- 每次 benchmark、短训练和迁移验证的机器可读 manifest；
- 失败时可独立复现的 Plan、输入、初始状态和配置。

支持状态只使用 `planned`、`implemented`、`verified` 和 `unsupported`。写出代码但尚未在本机 NPU 实测，只能记为 `implemented`；只有真实设备上的算子、差分和 fallback 检查通过后，才可记为 `verified`。

运行记录必须包含仓库 commit/dirty 状态、Plan 哈希、executor、算子实现变体、device、dtype、编译/自定义 kernel 选择、seed、输入与 checkpoint 身份以及明确的 fallback 情况。不能仅凭安装了某个包或一次 allocation 成功就宣称 SettleGraph 已支持该后端。

## 11. 开始科学实验前的最终检查

只有以下项目同时成立，才进入新的 finetune 或预训练实验：

- 当前实验使用的 Plan 已规范化、通过静态检查并保存哈希；
- 逐 Token、通用 prefill 和适用的特化执行器三方一致；
- 完整 prefill、分块 prefill 和逐 Token 执行的 forward/state 一致；
- 训练 loss、Hard-ST、balance loss 和关键梯度已经核验；
- CPU FP64/FP32 oracle、NPU FP32 eager 与 NPU packed 路径均通过对应门槛；
- NPU 关键算子没有未经说明的 CPU fallback；
- identity 初始化及所用 placement 已验证；
- checkpoint、manifest、失败复现和磁盘保留策略已经可用；
- benchmark 证明所选“高性能”路径名副其实。

这套检查的目的不是要求所有未来算法都一次完成，而是保证每一个进入实验的具体组合都同时拥有清楚的语义、独立的参考路径、可验证的高性能路径和可追溯的运行证据。
