# SettleGraph 实现与等价性验证计划

> 本文是实现计划，不改写 SettleGraph 的主模型语义。明确标为“可选外部控制扩展”的接口行为只描述和验证该扩展本身，不属于主语义或标准科学实验条件。
>
> [实验语义、命名与数学符号](experiment-semantics-and-naming.md) 是 SettleGraph 计算含义的权威文档。若实现计划与语义文档冲突，以语义文档为准；若语义文档不足以唯一决定实现行为，应先对齐语义，而不是在代码里自行选择一种解释。
>
> [等价性测试契约](equivalence-test-contract.md) 把本文的验证要求具体化为 fixture、trace、comparator 和证据门槛；它同样不增加模型语义。

## 1. 目标与完成标准

首轮实现同时建设三条执行路径：

| 执行路径 | 作用 | 是否必须覆盖任意 `core-v1` Plan |
| --- | --- | ---: |
| 逐 Token 解释器 | 最直接地复现语义文档第 2.4 节，作为正确性基准，也可用于 decode | 是 |
| 通用 packed prefill 执行器 | 批量处理一段 Token，避免按 Token、样本或 node 发起大量 Python 调用 | 是，具体性能能力按算子记录 |
| 特化 prefill 执行器 | 利用单层、HB-Lattice 等 Plan 的额外规则提高效率 | 否，只接受自己声明支持的 Plan |

这三条路径必须复用同一份规范化 Plan、参数、输入、初始状态和随机数规则。特化执行器不是另一套模型语义；它只是通用执行的优化实现。

这里的“`core-v1` Plan”指首轮实现和资格测试采用的固定图子集：每个 region 独立声明固定 active budget，每份可变状态和 SettleGraph 内部可训练参数都有唯一 owner；所有局部公式和当前 FP32/FP64 reference 子集的 dtype roles 已完整声明。当前 Plan schema 还接受可选的外部 `requested_k` 控制，但它不属于 `core-v1` Plan，必须作为单独的接口扩展标记和验证，也不能记作固定 \(K_{\mathcal R}\) 的标准实验。selector-history 的局部测试公式虽已定义，但通用 owner、字段和序列化 schema 尚未闭合，因此同样不属于当前 `core-v1` Plan。这里的“通用”首先表示不限制 `core-v1` Plan 的合法拓扑。逐 Token 和通用 prefill 接口都必须能执行任意 `core-v1` Plan；selector-history、由模型内容推导的 adaptive budget、混合/低精度 accumulation policy 或其他未来扩展，必须先增加语义和能力声明。任意新加入的自定义算子也不会自动获得高性能，其 reference、packed 和设备优化能力必须分别验证。

“函数等价”至少包括：

- 每个图执行 Token 的 SettleGraph 输出；
- 最终 receiver state 和 selector-history；
- reached、Observe、active、发送以及边结算结果；
- selector logits、soft probability 和辅助损失；
- 在端到端 Base 接入 case 中的语言模型损失，以及各适用 case 的选定输入和参数梯度；
- padding 旁路、chunk continuation、状态 reset 和 detach 行为。

在完成执行器等价性、CPU/NPU 一致性和基本训练测试以前，不开始新的科学实验。性能优化不能代替这些正确性门槛。

### 1.1 当前仓库能力边界

本节记录当前软件边界，避免把后文的完成标准误读成已有能力；它不改变语义，也不构成 `verified` 声明。

| 当前工件 | 已有范围 | 尚未达到的边界 |
| --- | --- | --- |
| 运行时与 Plan | CPU-safe 的包入口；device/dtype 解析；logical/typed Plan 规范化、分类静态校验与哈希；当前 reference formula config 的 exact schema、默认值物化、数值规范化及跨字段 shape/timing 校验；运行期拒绝不同 owner 键通过不同 Tensor views 共享可变 storage；手工拓扑和小型 HB fixture Builders；单 site 的实现无关 parameter-schema manifest 与独立 eager locator binding；`tide.failure.v1` envelope；CPU-only、weights-only-safe 的 `tide.settlegraph.fixture.v1` no-replace 保存、单次 bytes 装载、Tensor 值/stride/storage group 认证、真实负 mutation 与 preflight；development corpus 的项目内运行记录入口 | 跨 site 的独立参数 schema 组合、可学习首状态 schema、混合/低精度 accumulation role、跨语言 canonicalizer conformance、完整资格 bundle 集、benchmark/训练/硬件资格运行 manifest、能力矩阵证据和 Builder qualification |
| token-major eager reference | 当前 reference 算子子集的逐 Token 解释与 token-major prefill；N/SD/BO、content/pre/post、状态 carry、mask、运行期 K、事务、balance 统计和 trace；forced-active singleton reached 后直接取精确概率 1，并跳过 selector Read、Score、softmax 和 Top-K | 任意自定义公式、selector context/history，以及测试契约要求的完整多类别 golden/exact-trace 资格语料 |
| region-major eager reference | `prefill_region_major` 独立于 `interpret_token`，按规范 region 顺序执行同一 Plan，并已有 forward/state/trace/gradient 定向差分测试 | 仍按 region、Token 和 batch row 使用 Python 循环；不是第 3.2 节的通用 packed prefill，没有 packed 或性能能力声明 |
| 通用 packed executor | `tide.generic-packed.torch.v1` 直接复用同一 `SettleGraph` owner 的参数，静态接受当前 256 个固定 K `core-v1` candidates；按 region 处理整段 ([B,T])，只在 regions 间保留实际 `DATA` records，并覆盖 N/SD/BO、content/pre/post、EMA、Gated DeltaNet、窗口 Attention、当前 Aggregate/Read/Score/NodeCompute/Emit、forced-active singleton、prefill/decode 和 trace；代表性 lifecycle 覆盖 carry、reset、row reorder 与 empty tail；SD/pre 的 hard route 发现及必须保持因果顺序的状态扫描位于 TorchScript 循环中，随后按固定 route 可微重算；不调用任一 eager scheduler | 这是 tensorized packed development implementation，不是已经通过 `C04`/`X07` 的性能资格实现：StateStore 打包/发布、下一位置和静态参数分组仍有 host-side mapping，部分严格 FP32 顺序路径在 TorchScript 中串行；非 trace 的中间量生命周期、反向时的小 usage-mask host copy、真实 callback/内存 profiler、长序列 benchmark 和 NPU fallback closure 仍须单独收口；`k.input.v1`、selector context/history、未注册自定义公式和混合/低精度仍静态拒绝 |
| 拓扑特化 executors | `single-layer.v1` 对无状态 N/content 单层拓扑采用平铺 Tensor 路径；`hb-line.v1` 直接消费 fully expanded HB Plan，独立按 Line barrier 推进；二者共享 owner 参数而没有第二份参数 namespace，支持 full/chunk/decode 和静态拒绝，不调用通用 eager scheduler | HB 版本当前是独立拓扑调度 oracle，不是融合或高性能 HB kernel，也没有形成独立的局部公式 oracle；支持集合目前只是 256 candidates 中预先由谓词选出的单层 8 个与 HB 16 个，尚缺资格 bundles、目标设备 profiling 和正式性能门 |
| placement | POST、PARBLK、PARATTN、PARMLP 的通用 Tensor 方程及 identity 退化测试 | 真实 Qwen block、causal mask、position IDs、KV cache、logits、LM loss 和 Base 参数梯度接入 |
| comparator 与解析 oracle | 统一 nested comparator、trace invariant 检查、route-boundary 分类，一个不调用共享局部算子或执行器 helper、并按第 2.3.3 节省略 selector Read/Score/K/Top-K 的 singleton exact-trace golden；48-Plan/6-VJP/24-invalid 快速 corpus 比较 token-major 与 region-major；固定 identity 的 256-candidate/64-marked-VJP executor corpus 对通用 packed 做两 dtype output/state/balance/full-trace 和 full/two-chunk/decode forward，每段 chunk/每步 decode 的 trace 均直接对照 eager，规范合并后的 trace 再与 full 对照。64 个 VJP cases 在每种 dtype 下依次隔离查询 output、可微 balance loss、每个最终状态 owner/component、每个 region `soft_sum`、每个可微 trace region event 的 logits/probabilities、组合目标和重复 output，并比较 hidden/全部具名参数的数值与 `None` 连通性；另以固定多候选反例逐个检查 node-event logit root 的 connected-zero/`None` 结构。FP64 定向回归覆盖 EMA/Gated DeltaNet/窗口 Attention 的可微外部初态、Attention keys/values 和跨两个 chunks 的保留图 objectives。对静态适用的全部单层 8 个和 HB 16 个做 eager—packed—specialized 三方两 dtype full/two-chunk/decode forward 与另行 full-prefill VJP；有状态代表集另覆盖 reset、row reorder 和 empty tail；共享执行入口还定向验证只接受 FP32/FP64，并在执行前拒绝空 batch。受控 runner 从执行期回执而非静态 support 数量导出实际覆盖，缺少任一 case/dtype/mode 或 objective completion 时失败 | 256 个仍是生成式 development candidates，不是长期物化的资格 bundles；仍缺完整多类别独立 goldens、资格 objective/cotangent 与 exact logical-key records、32 FD、16 optimizer、完整 pairwise/event multiplicity、其余非法 mutants、失败收缩和 `C00`—`C12` 可追溯 artifacts |
| checkpoint v1 | SettleGraph 参数、logical/typed Plan 与参数 dtype 校验、CPU 规范 receiver Tensor/窗口 Attention 状态、进度/训练元数据和 CPU RNG 的 `init-from`/`resume` round trip；通用状态序列化器严格编码、解码 selector-history 容器，但 eager executor/checkpoint attach 拒绝非空 history；root 键集 exact，序列位置使用下一待执行位置并拒绝旧字段；Adam/AdamW 类型与超参数域、稳定模型参数组/顺序、已初始化 state manifest、Tensor shape/dtype/storage alias 均在 commit 前校验；保存端只接受 weights-only-safe 元数据并自检；CPU 序列状态先做 owner/alias 校验再转目标 device；基础 CPU checkpoint 跨 device 装载路径和 receiver-state continuation 用例已实现；注入式 commit failure 对 model `state_dict`、optimizer containers/defaults 与 CPU RNG 联合回滚 | selector-history continuation、scheduler、scaler、backend RNG、sampler/data cursor、未归约统计窗口和窗口中途恢复未实现；portable handoff 仍缺完整训练状态、optimizer 下一步、规定数量和可追溯证据，未达到完整资格；不支持任意 optimizer/schema，也不承诺回滚任意 Python 属性或 load hook 外部副作用；仍缺第 7.5 节与测试契约第 8 节的完整资格证据 |
| live backend 入口 | CPU/NPU/CUDA 的显式 backend semantic 测试入口；2026-09-03 在本机 aarch64 `Ascend910_9392`、Torch `2.10.0+cpu`、TorchNPU `2.10.0`、CANN `9.0.0` 上，对后来提交为 `c6e2cc5` 的 eager-reference 基线内容完成一次 FP32 定向 attempt：由 site launcher 分配设备并在进程内使用 logical index 0，显式 NPU runtime suite 22/22、live semantic 3/3、独立 CPU→NPU fixture parity 和 CPU checkpoint continuation 通过，parity 最大绝对/相对误差为 \(5.96\times10^{-8}\)/\(1.43\times10^{-6}\)，CPU parity artifact/checkpoint SHA-256 分别为 `944378eb1ad4e7ba20205eeb81f8243b4aebad85763f7db27139dde29964861f`/`5a4c155bb5ada1e1b47a30fc5628e622a225600f282fd42964d2df8fe6614172`；另一次较早的 EMA、Gated DeltaNet 与窗口 Attention region-major forward/backward profiler attempt 观察到 NPU kernels，未观察到 AI_CPU task 或显式 fallback 记录 | 该记录不是 clean exact-commit 证据，且当前扩展代码尚未复验；parity 只有一个 BO/post fixture，未达到契约的 64/32/8 数量与完整 operator/shape/layout 覆盖，也未 profile optimizer/checkpoint 或建立 fallback closure；较早 profiler 有默认 schedule 可能不完整的 warning，并无 packed、低精度、短训练，因此 NPU 仍为 `implemented`；CUDA 仍为 `planned` |

因此，Stage A 的 eager reference 主体、forced-active singleton 简化、独立调度参考、基础 comparator/invariant、一个解析 golden、单-site parameter manifest 和 bundle 基础设施已经存在；Stage B 的当前 `core-v1` 通用 packed 路径以及 Stage C 中单层/HB 的拓扑特化路径也已实现并进入扩大版 development regression。Stage C 的真实 Base 接入、selector-history，以及 Stage B/C 的正式 qualification 与性能收口仍未完成。本机已有 eager-reference 基线快照的 NPU 定向 parity/checkpoint attempt 和一次较早的 profiler attempt，但当前 packed/特化代码与完整 NPU qualification、CUDA parity 均未验证。现有单元、定向差分和这些硬件 attempt 都不能代替第 10 节要求的完整可追溯资格 artifact。

selector-history 仍存在通用 schema 级未闭合项。测试契约中的 `TEST-HISTORY-ACTIVE-EMA-V1` 已经唯一规定一个 node-level history 的数值递推、写回时序、首值/decay 语义及其加入 Read 的位置，因此该局部公式本身可以生成 golden；尚未唯一规定的是通用 Plan 如何选择 region-level 或 node-level owner、规范 owner 键和字段、Read 维度，以及 trace/checkpoint 中的统一序列化表示。在这些选择形成版本化 schema 前，它不能进入当前实现的通用标准测试子集；本计划不替这些待定项预选软件字段。

单个 SettleGraph site 当前从 Plan 派生实现无关 parameter schema，并另存 owner model 的 locator binding；fixture Tensor 因而可以使用逻辑参数键，而不把 module 路径当作跨 executor 身份。packed 与特化绑定直接持有并使用同一个 `SettleGraph` 参数 owner，不注册参数副本或第二个 `state_dict` namespace，所以当前不需要另一份参数 locator。跨 sites 的稳定 site ID 与独立 parameter schema 组合仍未闭合；当前 schema 只解决了独立参数的单-site bundle 身份，不代表端到端资格已经完成。

## 2. 共同的数据契约

### 2.1 一次 prefill 的输入与输出

对单个 site 上的一次 SettleGraph 调用，通用接口至少接收：

| 输入 | 含义 |
| --- | --- |
| \(H^{\mathrm{in}}\in\mathbb R^{B\times T\times d_{\mathrm{model}}}\) | \(B\) 条序列、每条至多 \(T\) 个 Token 的图输入 |
| `execution_mask`，shape 为 \([B,T]\) | 图执行/context mask；false 位置旁路且不更新状态 |
| 可选 `lm_target_mask`，shape 为 \([B,T]\) | 只标识端到端 Base 接入中的 LM loss 目标，不改变图执行；必须是 `execution_mask` 的子集 |
| `routing_stats_mask`，shape 为 \([B,T]\)，可省略 | 只选择路由统计事件；默认等于 `execution_mask`，显式值必须是其子集 |
| `sequence_id`，长度为 \(B\) | 跨 chunk 稳定标识每条序列 |
| `token_position`，shape 为 \([B,T]\) | 同一序列内跨 chunk 不重置的全局 Token 位置 |
| 可选 `requested_k` 控制 | 当前实现的外部控制扩展；对明确开放它的 regions，按事件提供非可微整数 |
| 初始状态 | 按语义文档第 2.5 节的键读取的 receiver state 与 selector-history |
| logical Plan、concrete execution binding 与参数 | 固定拓扑、局部运算、dtype role 绑定，以及按 Plan 稳定参数键提供的 Tensor 数值 |

同一调用中的 `sequence_id` 必须两两不同；状态所有权由稳定 ID 而不是 batch row 决定，因此跨调用重排 batch rows 不改变状态归属。对每个序列，`execution_mask=true` 的 `token_position` 必须从状态仓库记录的下一位置开始逐一递增；重复、倒序、跳号或重放均在执行前失败。新序列和显式 reset 后的第一位置为 0。false 位置的 `token_position` 不参与因果校验。

执行结果至少包含：

- \(B_{\mathcal G}\in\mathbb R^{B\times T\times d_{\mathrm{model}}}\)；
- chunk 结束后的全部状态；
- 路由辅助 loss 可合并的充分统计量；
- 在拥有 logits、目标 Token 和 shift 语义的端到端 Base 接入层，额外返回 LM 负对数似然总和与目标数；独立 SettleGraph 调用只校验或透传 `lm_target_mask`，不凭 hidden 自行构造 LM loss；
- 可选的执行 trace。完整 trace 只用于小规模测试，正常训练只保留聚合后的诊断量。

逐 Token 解释器一次处理同一批序列的一个位置 \([B,d_{\mathrm{model}}]\)。把它按 \(t=0,\ldots,T-1\) 调用后，应得到与一次通用 prefill 相同的逐 Token 结果和最终状态。

#### 2.1.1 状态生命周期与调用事务

一个序列在首个成功执行位置创建声明的首状态，并在后续 chunks 中按 `sequence_id` 延续。显式 reset 在调用的第一个 Token 前把该序列在当前 site 的全部 receiver state、selector-history 和下一位置恢复到首值；顶层模型对一个序列做 reset 时，必须对其全部 sites 使用同一边界。释放状态是独立的生命周期操作，只能针对明确列出的、没有进行中调用的序列。

一次公开调用使用私有暂存状态执行。调用内部较早位置的写入可以被同一调用中较晚位置读取，但 receiver state、selector-history、下一位置、reset 结果、输出和辅助统计只在全部输入校验与全部图执行 Token 成功后一起发布。任一图执行 Token 的 `requested_k` 越界、位置非法、局部操作失败或空终端执行不变量失败时，整个调用失败并丢弃暂存写入；调用前状态保持不变，也不得返回可用于继续训练的部分结果。语义文档第 2.4 节的合法 Plan 约束本身保证非空终端，空终端 case 只通过测试专用故障注入验证防御与回滚，不能当作合法路由样例。

chunk 边界的状态值默认延续，autograd 边界默认 detach；这两件事必须分别表达。若某项测试或训练选择跨 chunk 保留梯度，必须显式记录，并与采用相同边界的 oracle 比较。并发调用不能同时写入同一个 `sequence_id`；调度器必须串行化或在开始前拒绝冲突。

### 2.2 规范化 Plan 与数值绑定

原始图描述必须先经过语义文档第 2.4 节的静态校验，再转成规范化 Plan。对当前 reference 已注册的公式版本，规范化会校验必需/允许键并物化全部默认值，因此省略默认键与显式写出同值不会产生两个哈希。已注册 type/formula 的错配、缺失或裸未知 ID 在 Plan gate 拒绝；框架中立的 Plan 层只可保留携带自包含数学定义的显式自定义公式作为未来能力描述，当前 eager reference 再在执行器构造时把这种合法 Plan 拒绝为未实现能力，不得运行时降级。Plan 至少保存：

- 稳定的 node、edge、region ID；
- node 所属 region、固定父边和子边；
- 入口和终端 receivers；
- region 依赖图及其规范拓扑序；
- Aggregate、Update、两类 Read、Score、Top-K、NodeCompute 和 Emit 的配置；
- 状态、参数、hidden、读出和归约量的 shape 与 dtype role 契约；
- forced-active 和每个 region 的固定 \(K_{\mathcal R}\)；若使用实现扩展，还要保存运行期 `requested_k` 契约；
- HB-Lattice 可选的 Line、phase 和边来源标签。

进入同一 receiver 的父消息按稳定 edge ID 排列；一个 region 的 candidates 按稳定 node ID 排列。logical Plan 的规范化序列化必须产生稳定 logical Plan hash。另一个规范记录把 dtype roles 映射到具体 dtype，连同 logical Plan hash 产生 typed Plan hash。可训练/装载的参数 Tensor 数值、device、executor、运行期 reached/active 结果和某个 batch 的状态不属于这两个 Plan hash；稳定参数键/schema 与会改变公式的固定常量、尺寸和开关属于 logical Plan。CPU FP64 oracle、CPU FP32 和 NPU FP32 分别建立 concrete execution binding，不能通过修改同一个 concrete Plan 的未记录运行时 dtype 来切换。

当前 schema 中的 site、node、edge、region、参数角色等稳定 ID 都是非空的 Unicode scalar-value 字符串：必须采用 NFC 规范化，不含 NUL，首尾既不是 Unicode `White_Space` 字符，也不是 C0 information separators U+001C–U+001F。稳定顺序按 Unicode scalar value 序列做字典序比较，与 locale、自然数排序和声明顺序无关；合法 UTF-8 对 scalar value 保序，因此实现也可以对这些字符串的 UTF-8 bytes 使用无符号字典序。

当前 logical Plan schema `1` 的规范 bytes 由仓库 reference canonicalizer `tide-plan-json-v1` 产生：所有 object keys 按上述稳定字符串顺序排列，语义集合/序列先按各自稳定 ID 或规范拓扑序排列；JSON 直接写 Unicode、不写无意义空白、拒绝 NaN/Inf，再编码为 UTF-8。logical/typed Plan hash 都是相应 bytes 的小写 SHA-256。公式实数先按测试契约第 2.2 节完成整数/浮点同值与负零规范化；其他 JSON number 的 byte 表示以这个 reference canonicalizer 为准。fixture bundle 必须携带 canonicalizer ID 和原始规范 bytes；尚未通过 byte-for-byte golden 的其他语言 executor 应消费这些 bytes，而不能自行猜测另一种 JSON number renderer。当前仓库还没有独立于 Python reference 的跨语言 canonicalizer conformance suite，因此跨语言重新生成相同 Plan hash 仍是完整资格缺口，不影响本轮同一 Python reference 内的定向比较。

为高性能执行派生的 region 批次、CSR 索引、算子分组和缓存生命周期称为编译后调度信息。它可以重新生成，不能改变规范化 Plan 的含义，也不能取代 logical/typed Plan hash。

### 2.3 当前 `core-v1` 的实现取舍

本节把首轮实现有意采用的简单构造集中在一起。它们约束当前实现与资格测试，但不是 SettleGraph 数学语义的唯一可能实现。

#### 2.3.1 每个 region 独立配置 active budget

主语义中的固定 \(K_{\mathcal R}\) 在当前 Plan schema 中表示为

$$
K^{\max}_{\mathcal R}
=K^{\mathrm{req}}_{\mathcal R}
=K_{\mathcal R}.
$$

不同 regions 可以使用不同的固定值，标准实验只使用这种配置。当前 Plan schema 可以表示两种请求来源：

- `fixed`：Plan 中固定的 \(K^{\mathrm{req}}_{\mathcal R}\)，用于标准实验；
- `input`：对明确开放该接口的 region，由调用方通过 `requested_k` 提供非可微整数，属于可选的外部控制扩展。

若当前事件有 \(C_{\mathcal R,t}>0\) 个 candidates，则先校验

$$
1\le K^{\mathrm{req}}_{\mathcal R,t}\le K^{\max}_{\mathcal R},
$$

再取

$$
K^{\mathrm{actual}}_{\mathcal R,t}
=\min\!\left(K^{\mathrm{req}}_{\mathcal R,t},C_{\mathcal R,t}\right).
$$

候选为空时不执行选择，也不读取该事件的 `requested_k`。`requested_k` 可以随调用事件变化，因此只用于接口与调度实验，不能记作主语义中的固定 \(K_{\mathcal R}\) 实验。由 Token hidden、selector logits 或 receiver state 在模型内部推导请求值的 adaptive \(K\) 暂不支持。

#### 2.3.2 状态与参数各自独占

每份 receiver state、selector-history 和 SettleGraph 内部可训练参数都由一个稳定逻辑键唯一拥有。当前 validator 和 binding 必须拒绝跨 node、region 或 site 的共享、Tensor alias 以及 backing-storage alias；SettleGraph 权重不做绑定。批量执行可以重排或临时打包互不共享的参数，但不能把不同逻辑参数变成同一个可训练自由度。若以后重新引入共享，必须另行定义 owner、参数身份、更新顺序、梯度累加和 checkpoint 契约，不能把它当作当前 Plan 的自然扩展。

#### 2.3.3 forced-active singleton

forced-active singleton 被 reached 后直接取

$$
p_{v,t}=1,
\qquad
\mathcal A_{\mathcal R,t}=\{v\}.
$$

它不需要执行 selector 的 \(\operatorname{Read}^{\mathrm{sel}}\)、Score、softmax 或 Top-K；receiver 的 Update、\(\operatorname{Read}^{\mathrm{ffn}}\)、NodeCompute 和 Emit 仍按各自配置执行。当前 eager 路径和 singleton golden 已遵守这一规则；trace 中 selector readout、logit、请求/实际 K 和 Top-K IDs 显式 absent，只保留精确的 \(p=1\) 与 active 结果。

#### 2.3.4 结构化状态与 Attention 时间位置

语义中的单个状态 \(s_{v,t}\) 可以是由多个 Tensor 和元数据组成的结构化值，而不必压成一个 Tensor。窗口 Attention 的数值状态是主语义附录 A.5 定义的有序 key/value 序列；当前 runtime 和 checkpoint 还为每个有效项附带 Observe 时的 Token 位置，其实现记录记为

$$
\bar s^{\mathrm{impl}}_{v,t}
=\bigl((\tau_i,k_i,\nu_i)\bigr)_{i=1}^{n_s},
\qquad
0\le n_s\le W,
$$

其中 \((k_i,\nu_i)\) 是主语义附录 A.5 中按 Observe 顺序排列的第 \(i\) 个有效 key/value pair，\(\tau_i\) 是它在同一序列中的全局 Token 位置；丢弃各 \(\tau_i\) 后即得到主语义中的 \(s_{v,t}\)。当前 Attention 读出只消费有效的 keys 和 values，不使用 \(\tau_i\) 改变权重；位置元数据仍随 checkpoint 保存并参与实现状态比较，为以后定义时间衰减等另一种 receiver 公式保留信息。固定 ring buffer、有效长度加 head，或 packed 变长表示都只是物理实现；比较和保存前应恢复为同一有序有效序列，未使用槽位没有语义。

#### 2.3.5 identity 的首轮充分构造

当前实现不尝试构造“局部变化相互抵消、整体仍为 identity”的复杂初始化。对任意合法 hidden \(h\)、当前可见状态 \(s\)、soft probability \(p\) 和非空重复消息序列，首轮直接采用以下局部充分条件：

$$
\operatorname{Aggregate}_v(h,\ldots,h)=h,
\qquad
\operatorname{NodeCompute}_v(h,N_{R,v}(h),s)=h,
$$

$$
\operatorname{Emit}_v(h,h,p)=h,
\qquad
\operatorname{Aggregate}_{\mathrm{out}}(h,\ldots,h)=h.
$$

再加上每个图执行 Token 至少有一个 active 终端 receiver，即可得到图整体 identity。该验收默认只承诺 Base 模型前向输出和由它定义的 LM loss 不变；非零路由辅助项、新增参数梯度和完整训练目标不自动与 Base 等价。

### 2.4 参数与算子实现

每项局部运算分成两层：

1. **语义配置**：说明数学上计算什么，必须能对应到语义文档中的公式或明确的自定义公式。
2. **实现变体**：说明使用 eager Torch、packed Torch、编译图或某个设备自定义算子来完成同一计算。

每个语义配置至少有一个标准 Torch 参考实现。可选优化实现必须声明支持的 device、dtype、shape、forward/backward 和布局；不支持时只能显式选择已经验证等价的参考实现，或明确失败，不能暗中换算法、换 dtype、转 CPU 或丢失梯度。

各执行器应读取同一组逻辑参数。为了 grouped GEMM 或批量状态更新，可以重排或临时堆叠具有相同算子签名、但彼此独立的 node 参数；不同执行器不能维护会逐渐失配的可训练副本，也不能借打包引入权重绑定。

## 3. 三条执行路径

### 3.1 逐 Token 解释器

逐 Token 解释器忠实实现语义文档中的 `InterpretToken` 顺序：

1. 为当前 Token 建立入口消息；
2. 按 region 依赖的合法拓扑序推进；
3. 等固定父边全部结算后，按 edge ID 收集 `DATA`，忽略 `CLOSED`；
4. 完成 Aggregate、选择、Observe/commit、NodeCompute 和 Emit；普通竞争 region 执行 selector Read、Score 和 Top-K，forced-active singleton 按第 2.3.3 节直接激活；
5. 结算固定出边并最终聚合 active 终端输出。

首版允许在 Python 中按 region、node 和 edge 循环，因为它的首要职责是清楚、可检查和适合逐步调试。Tensor 数学仍使用标准 Torch，使 CPU float64、autograd 和 NPU eager 路径能复用同一代码。

解释器应支持可控 exact trace。小 fixture 中不能只保存状态摘要，而要按 [等价性测试契约](equivalence-test-contract.md) 记录父边结算与 payload、消息序列、\(s^-\)、proposal、candidates、Observe/active、\(s^{\mathrm{cmp}}\)、NodeCompute、Emit、状态与历史写回，以及终端聚合；selector readout、logits、probability、请求与实际 K 只在对应操作存在时记录。trace 的排列只依赖稳定 site/region/node/edge ID 和序列位置。

至少一组解析 golden 必须独立写出期望事件和值，不能调用解释器与 packed 路径共同使用的 Aggregate、Update、Score、NodeCompute、Emit 或 balance-loss helper 来生成期望结果。执行器差分发现调度差异，解析 golden 负责发现多条路径共同实现错同一公式。

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
| active budget | 各 region 独立的固定 Top-1、Top-2、all；另测外部 `requested_k` 扩展 | region 间上限不同、singleton、候选少于请求 K 和变长 active set |
| Emit | hard、Hard-ST、soft probability | 前向值与 selector 主任务梯度 |
| NodeCompute | 简单 affine 测试算子、SwiGLU MLP、状态读出 + 双 residual | 解析核验、真实昂贵计算和 residual |
| 参数关系 | 所有 SettleGraph 参数独立 | 参数 owner、Tensor/storage alias 拒绝和跨执行器同一逻辑参数 |
| 状态首值 | 零、固定非零、可学习首状态 | reset、序列隔离和序列 continuation |

窗口 Attention、Gated DeltaNet 等具体公式仍以语义文档附录 A 和实验记录为准；Attention 状态的当前物理约束见第 2.3.4 节。若实现的是另一种算法家族变体，必须使用新的明确配置，不能只复用旧名称。

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
| singleton forced-active | 最小入口/终端、reached 后直接取 \(p=1\)，且不执行 selector Read/Score/Top-K |
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

成功样例中的每个 receiver 必须位于入口—终端固定路径上。随机正确性样例应覆盖有无 forced-active backbone 的合法拓扑；两者在标准 K/Emit 规则下都必须产生非空终端。另用不计入合法集合的故障注入样例破坏发送不变量，验证空终端会触发执行失败和事务回滚。

生成器同时提供非法变异：对一个合法 Plan 注入一条反向边、region 内边、重复边、错误 ID、错误 shape 或非法操作组合，并验证静态检查器准确拒绝。

小规模测试可以枚举或系统采样 nodes 很少的图；较大图使用固定 seed 随机生成。任何失败都保存最小必要的规范化 Plan、seed、输入、参数、初始状态、运行配置和 executor 名称，使下一次运行无需依赖原随机过程即可复现。

当前 development corpus 固定保存生成 seed，并产生 48 个 canonical hash 各异的合法 Plan：45 个来自上述手工 motif 的局部公式变体，3 个由有界分层 DAG 生成器产生。它覆盖六种合法 profile/timing、四类 state、三类 Emit、三类 receiver Aggregate、两类 output Aggregate，以及 fixed-K core 与 input-K 接口扩展；每个 Plan 均在 CPU FP32/FP64 上比较 token-major 与独立 region-major 的 output、state、balance 和当前 trace，6 个 Plan 另比较 hidden 与所有具名参数的 gradient/`None` 记录并要求至少一项参数梯度非零。另有 24 个命名单变换非法 Plan 覆盖 cycle、重复 edge ID、region 内边、hidden shape、固定 K、profile/timing、稳定 ID 和终端集合；测试直接检查 validator 产生的 `failure_codes`。这些是开发语料；数量、完整规范 trace、持久化 bundle、pairwise/event 计数、逐 logical key 连通性、optimizer 和失败收缩均未达到测试契约第 7.2 节的资格门槛。

### 5.3 输入与状态样例

Plan 多样性之外，还要交叉覆盖：

- \(B=1\) 与多样本 batch；
- \(T=1\)、短 prefill 和较长 prefill；
- 每条序列不同的 `execution_mask`、`execution_mask=true` 但 `lm_target_mask=false` 的 prompt-only 位置、内部 padding 和空尾部；
- 同一个 chunk 内，各 \((\mathrm{sid},v)\) 拥有完全不同的 Observe/active Token 列表；
- 一次完整 prefill、多个不等长 chunks 和逐 Token 输入；
- 零状态、随机状态和从上一 chunk 延续的状态；
- 正常 logits、刻意平票和接近 Top-K 边界的 logits。

## 6. 差分验证方法

### 6.1 比较顺序

所有比较使用同一 fixture family；相同 concrete execution binding 使用 byte-identical 的 CPU 序列化 bundle，不同 dtype binding 按测试契约从同一逻辑源值确定性 materialize。CPU 与每种 accelerator case 在相互独立的新进程中运行；应用进程解析并选定 backend 后才创建 Torch Tensor 或 autograd 状态，进程间只交换 CPU fixture 和结果 artifact，不能先运行 CPU autograd 再晚加载 TorchNPU。建议按以下顺序建立证据：

1. 人工可计算的小例子对照逐 Token 解释器；
2. CPU float64：逐 Token 解释器对通用 prefill；
3. CPU float32：逐 Token 与通用 prefill 互比，并对照 float64；
4. CPU 上的特化执行器对照两种通用路径；
5. NPU FP32 eager 对照 CPU FP32；
6. NPU packed/optimized 对照 NPU eager 与 CPU；
7. 明确需要后再验证 BF16 等低精度路径。

CPU FP32 是可移植的基础正确性路径；CPU float64 是相同 logical Plan 公式的小规模高精度 oracle，两者具有不同 typed Plan hash。不得从 CPU float64 推断 NPU 也需要或支持 float64。NPU 首先验证 FP32，低精度属于后续独立能力。

### 6.2 要逐项比较的量

同一 backend/dtype 的执行器差分至少检查：

- 输出 shape、dtype、device、图执行 Token 数、LM target 数和路由统计事件数；
- 每个 Token 的 \(b_{\mathcal G}\)；
- 所有状态键及最终状态值；
- reached/Observe/active/发送 mask 和 Top-K ID，离散量要求完全相同；
- 每个 region 的 logits、probability、balance 统计和 loss；
- 聚合前后的 hidden；
- 输入 hidden、Aggregate、receiver、selector 参数的选定梯度；
- 一次 optimizer step 后的选定参数；
- checkpoint 保存、重新加载后的结果。

浮点 Tensor 按 [等价性测试契约](equivalence-test-contract.md) 逐元素比较；小 fixture 的默认门槛为 CPU FP64 `atol=1e-10, rtol=1e-8`、同 backend FP32 `atol=1e-6, rtol=1e-5`、CPU 与 NPU FP32 `atol=1e-4, rtol=1e-4`，具体量可以更严但不能无记录放宽。任一待比较值出现 NaN 或 Inf 都失败。shape、mask、ID、请求值、候选顺序、Top-K、状态键、schema 和其他离散量要求 exact。

路由样例明确分为 exact tie、margin-safe 和 near-boundary。exact tie 验证稳定 node ID 平票；margin-safe 用大于 comparator guard band 的第 K 与 K+1 分数差建立跨实现自然路由证据；near-boundary 专门暴露浮点敏感性。声称端到端等价的 case 即使属于 near-boundary，离散 route 也必须 exact；route replay 通过只能形成“给定同一路由时等价”的诊断证据，不能把自然路由不一致改判为通过。

必要时提供仅用于测试的 route replay：先保存参考 active set，再令另一实现复用它，以分别定位“selector 数值不同”和“给定同一路由后的 receiver 计算不同”。自然路由的端到端一致性仍然必须单独通过，不能只用 replay 代替。

### 6.3 状态、chunk 与随机数

在 eval/deterministic 模式下，以下三种执行应产生相同的逐 Token 输出和最终状态：

```text
完整 prefill
= 任意合法 chunk 切分后的连续 prefill
= 逐 Token 解释执行
```

loss 的 chunk 等价性使用可加的充分统计量，不能平均各 chunk 已经归约后的 loss。端到端 Base 接入层的 LM loss 返回负对数似然总和与目标数 \((L_{\mathrm{nll,sum}},N_T)\)，完整统计窗口结束后计算 \(L_{\mathrm{nll,sum}}/N_T\)。独立 SettleGraph 执行器只负责下面的路由统计。对 `BAL-AVAIL-SOFT` 的每个 site-region 和固定 receiver \(v\)，每个 chunk 返回

$$
N=\sum_e 1,
\qquad
P_v=\sum_e\mathbf 1[v\in\mathcal C_e]p_{e,v},
\qquad
A_v=\sum_e\frac{\mathbf 1[v\in\mathcal C_e]}{|\mathcal C_e|},
$$

$$
F_v=\sum_e
\frac{\mathbf 1[v\in\mathcal A_e]}{|\mathcal A_e|},
\qquad
Q=\sum_e\mathbf 1[|\mathcal C_e|\ge2],
$$

其中 \(e\) 遍历 routing-stat mask 选中的非空候选事件。跨 chunks 和 devices 先分别对这些量求和，再用 \(P_v/N\)、\(A_v/N\)、\(F_v/N\) 代入语义文档第 6.2 节；\(Q>0\) 决定 region 是否进入竞争集合。\(P_v\) 必须保留到 selector probability 的梯度，\(N,A_v,F_v,Q\) stop-gradient。统计窗口、跨 rank 归并时点和 reduction 必须与未切 chunk 的参考完全相同。

因为 balance loss 在全窗口平均之后平方，逐 chunk loss 的平均或 Token 加权平均一般都不等价。要求 loss/gradient chunk 等价的训练必须延后到完整统计窗口再构造该 loss，或者采用经过同一 VJP 验证的两遍/充分统计梯度算法；不能在不知道后续统计时对各 chunk mean 立即 backward 后声称等价。

训练时若 chunk 边界 detach，则梯度只能与采用相同 detach 位置的参考执行比较；forward 和最终状态仍应一致。

首轮等价性测试默认关闭 dropout。以后若加入 dropout、路由噪声或其他随机操作，随机数必须由稳定键决定，例如 seed、site、node、sequence ID、全局 Token 位置和 operation ID，不能依赖 executor 的实际调用先后。否则不同 packing 顺序会生成不同掩码，无法判断是执行器错误还是随机数流不同。

### 6.4 Base 模型接入

SettleGraph 独立测试通过后，再覆盖语义文档第 1.3 节的 POST、PARBLK、PARATTN 和 PARMLP：

- 对每种 placement 检查输入 hidden 与 residual 合入位置；
- 按第 2.3.5 节初始化时，接入模型与原 Base 模型的前向输出和 LM loss 一致；
- 非 identity 初始化时，逐 Token 与 prefill 仍相互等价；
- 多个 sites 的参数和状态保持各自独立。

identity 的默认验收边界以第 2.3.5 节为准。共同输入或参数的梯度、全部新增参数梯度和含辅助项的总训练目标若要声明等价，必须作为更强契约逐项定义和验证。

### 6.5 训练能力

每个核心算子路径至少通过：

- forward 与辅助 loss；
- selected gradients 有限且符合参考；
- Hard-ST 的前向恒等关系和 selector 梯度；
- 一个 optimizer step；
- stateful chunk 的反向与 detach；
- checkpoint round trip；
- 一个很短的过拟合或下降测试。

梯度差分不以“对所有输出隐式求和”作为未记录约定。每个 fixture 保存固定 cotangent，并定义标量目标，其中分别覆盖图输出、最终状态读出、LM loss 和路由辅助项；执行器对相同标量目标计算 VJP。必查键至少覆盖输入 hidden、初始可微状态，以及实际出现的 Aggregate、Update、selector Read/Score、NodeCompute 和 Emit 参数。`None` 梯度只允许用于语义上不连通的键；已连通但数值为零必须返回并比较零 Tensor。

除数值 VJP 外，还要做梯度路径正负断言：post-update proposal 到 selector 默认可导，Hard-ST 的 selector 路径存在；候选/active 离散集合、availability 基准和历史写回 stop-gradient；inactive NodeCompute 不产生梯度；chunk detach 两侧的状态梯度按声明截断。CPU float64 可用于小图的 `gradcheck` 或有限差分；离散 Top-K 附近不适合直接做连续梯度检查，应使用 margin-safe logits 或固定 route。完整目标、键集合和判据见等价性测试契约。

### 6.6 验证范围边界

SettleGraph executor 的核心等价性范围是 fully expanded 的 `core-v1` Plan。HB-Lattice 只要已经展开并满足语义文档第 4.2 节的约束，就进入该范围；某个 HB Builder 是否生成预期端点则是独立的生成器测试，必须用固定名称/版本/config、规范化展开 Plan 和 golden hash 验证。在实际实验选定并版本化一个确切 Builder 以前，不能把语义文档第 4.5 节的候选规则称为已实现的默认 HB 拓扑。

Dense、Dense 扩展和 Flat MoE 是实验对照，不经过 SettleGraph executor，因此不属于三执行器差分的通过条件。若某轮科学实验使用它们，则必须另有覆盖其 placement、expert/gate 合并、capacity/drop/reroute、辅助 loss、梯度和 checkpoint 的 reference 与接入测试。相同地，真实 Base Qwen 接入还需验证 causal attention mask、position IDs、KV-cache prefill/decode、最终 logits、LM target mask 和 base 参数梯度；独立 SettleGraph fixture 通过不能替代这些端到端证据。

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

### 7.5 Checkpoint 与恢复边界

checkpoint 使用版本化 schema，并保存 logical Plan 规范记录与 logical Plan hash、保存时的 concrete execution binding 与 typed Plan hash、参数 schema、Builder 身份（若适用）、base/tokenizer/data identity，以及 checkpoint 内容 hash。Tensor 在可行时以 CPU 表示保存，Attention 状态按第 2.3.4 节的有序有效窗口规范化；路径只是可重定位提示，不作为身份。

`--init-from` 只装载声明的 base/SettleGraph 参数和可学习首状态，并从新的 receiver state、selector-history、序列位置、optimizer、数据进度与 RNG 轨迹开始。`--resume` 恢复完整训练状态，至少包括：

- 全部模型参数、receiver state、selector-history 和每个 `sequence_id` 的下一位置；
- optimizer、scheduler、AMP scaler；
- global step、Token 计数、epoch，以及 gradient-accumulation microstep；若允许在 accumulation 中途保存，还包括已经累积的参数梯度；
- sampler/data cursor、data identity、worker/采样器状态；
- CPU 与所选 backend 的适用 RNG 状态、确定性和精度设置；
- 尚未归约的 LM/balance 充分统计量及其窗口位置；若承诺在窗口中途 exact resume，还要保存可重建其 VJP 的 replay 输入或所用梯度算法状态，否则必须规定只在统计窗口边界保存。

`--init-from` 与 `--resume` 互斥，二者都在创建最终输出目录和修改外部状态前完成 schema、hash、shape、dtype role 与参数键校验。portable handoff 允许在另一个已验证 binding/backend 上载入 CPU checkpoint 并开始新的数值轨迹；exact resume 只对匹配的 backend、软件栈、数据顺序和确定性配置承诺。跨 CPU/NPU 默认只验证 portable handoff，不承诺 bitwise 或同轨迹 resume。保存—退出—新进程加载—继续的 round trip 是验收的一部分。

checkpoint 只在没有进行中状态事务的原子安全点发布。若 autograd 图跨 checkpoint 边界延续，普通 Tensor checkpoint 不能恢复该图；此时 exact resume 必须保存可重放前缀并重建同一图，或者把合法保存点限制在声明的 detach/backward 边界。

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
- 定义规范化 Plan 序列化、静态校验、logical/typed Plan hash 和编译后索引；
- 实现首批 Aggregate、receiver、selector、profile 和 Emit 参考算子；
- 按第 2.3.3 节简化 forced-active singleton，并更新 trace 与解析 golden；
- 实现 CPU FP64/FP32 的逐 Token 解释器及完整 trace；
- 完成人工 Plan 与非法 Plan 测试。

### 阶段 B：通用 packed prefill

- 实现稀疏消息记录、candidate events、两种 packing 视图和恢复索引；
- 先完成 N、SD content 和 BO 的 packed 路径；
- 再完成 SD pre 与 selector-history 的 region 递推路径；
- 对全部核心算子完成 CPU FP64/FP32 executor 差分和梯度差分；
- 加入随机 Plan、随机输入和失败 fixture 保存。

当前进度：固定 K `core-v1` 的 N/SD/BO、content/pre/post、三种已注册状态及当前局部公式已有通用 packed 实现；SD/pre 使用编译的因果递推，selector-history 仍未实现。256-candidate/64-VJP 是开发回归层，不能替代本阶段要求的独立 bundle、失败 fixture、optimizer/scenario cells 和性能证据。

### 阶段 C：特化执行与 Base 模型接入

- 实现单层和 HB-Lattice 特化执行器；
- 与逐 Token、通用 prefill 做三方差分；
- 接入 Base Qwen block 的四种 placement；
- 验证 identity 初始化、多个 sites 和训练 loss。

当前进度：`single-layer.v1` 和 `hb-line.v1` 已实现静态支持谓词、独立拓扑调度和开发层三方差分；HB 路径仍是 correctness scheduler，不是优化 kernel。Base Qwen、多个 sites、端到端 LM/KV-cache 与训练 loss 尚未实现。

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
- 规范化 Plan schema、validator、logical/typed Plan hash 和 Builders；
- 三类 executor 及共同结果/trace 契约；
- 人工、自动生成和非法 Plan 测试；
- CPU golden fixtures；
- `.torch-portability/contract.json`，记录 CPU/NPU 必需目标、数值容差和验证命令；
- 每次 benchmark、短训练和迁移验证的机器可读 manifest；
- 失败时可独立复现的 Plan、输入、初始状态和配置。

支持状态只使用 `planned`、`implemented`、`verified` 和 `unsupported`。写出代码但尚未在本机 NPU 实测，只能记为 `implemented`；只有真实设备上的算子、差分和 fallback 检查通过后，才可记为 `verified`。

运行记录必须包含仓库 commit/dirty 状态、logical/typed Plan hash、executor、算子实现变体、resolved backend 与 `resolution_reason`、host architecture、device 精确 SKU 与逻辑索引、Torch/TorchNPU/CANN/driver 等可观测版本、dtype 与精度/确定性设置、编译/自定义 kernel 选择、seed、输入与 checkpoint 身份、测试命令、artifact hash，以及明确的 fallback 情况。NPU 能力证据还要链接 profiler/dispatch artifact 和版本相关的 warning 分类。不能仅凭安装了某个包或一次 allocation 成功就宣称 SettleGraph 已支持该后端。

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
- 等价性测试契约要求的 fixture、trace、梯度、随机覆盖和 capability cells 均有可追溯通过证据；
- 实际采用的 HB Builder、Base 接入以及 Dense/MoE 对照若在实验范围内，已分别通过第 6.6 节的专项测试；
- benchmark 证明所选“高性能”路径名副其实。

这套检查的目的不是要求所有未来算法都一次完成，而是保证每一个进入实验的具体组合都同时拥有清楚的语义、独立的参考路径、可验证的高性能路径和可追溯的运行证据。
