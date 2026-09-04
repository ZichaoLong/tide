# TIDE：固定拓扑上的有界局部容量扩展

> **TIDE: A Topology-Invariant Degree-bounded Expansion Architecture for Autoregressive Token Inference**
>
> **TIDE：面向自回归 Token 推理的拓扑固定、度有界容量扩展架构**

本仓库研究 TIDE 的神经网络架构与 reference semantics。更上层的研究背景和长期设想见 [ObsidianVault / TIDE](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/README.md)。

## 1. 研究目标

TIDE 从三个结构要求出发：

1. **固定空间拓扑**：模型确定后，底层节点和边不随 Token 临时改变；每个 Token 的 active 子图可以变化。
2. **单节点成本有界**：每个节点的参数、状态、连接、候选处理、通信和计算都有与总容量无关的上界。
3. **可达容量增长**：模型扩容后，输入仍能沿固定局部连接到达并利用更多潜在容量。

固定拓扑本身不表示度有界；但每条连接都会占用节点资源，因此单节点成本有界会进一步约束局部连接度。在入口数和局部度都有界时，一步平铺不能持续覆盖增长的总容量，模型需要通过多跳、逐级、层次化或空间化传播接触更多节点。

如果每个 Token 还只能执行少量昂贵模块，就需要在传播途中反复进行固定局部选择，并约束传播深度、消息数和 active 计算量。

## 2. 当前语义对象：SettleGraph

本仓库把一个具有单输入、单输出和固定空间拓扑的 Single-Settlement Graph 简称为 **SettleGraph**。它可以插入 decoder-only Base LLM 的不同位置，并保持 base checkpoint 的 always-on 路径。

对每个 Token：

1. SettleGraph 接收一个 \(d_{\mathrm{model}}\) 维 hidden。
2. hidden 沿固定 DAG 的边传播；每条边最终恰好结算为完整数据或关闭。
3. receiver node 聚合实际到达的父消息，维护可选私有状态，并提供轻量 selector 读出。
4. 每个固定局部 region 的 selector 只在已经 reached 的 nodes 中选择少量 active nodes。
5. propagation profile 决定哪些 reached nodes Observe/commit；active nodes 执行完整计算并发送。
6. 终端 active 消息被聚合成一个 \(d_{\mathrm{model}}\) 维输出，再按 placement 合回 Base LLM。

三种核心 propagation profiles 是：

| Profile | 状态提交 | 完整计算与发送 |
| --- | --- | --- |
| **N** | 无状态 | active nodes |
| **SD** | active nodes | active nodes |
| **BO** | 全部 reached nodes | active nodes |

selector 可以读取当前内容、更新前状态或更新后 proposal，分别形成 content-only、pre-update 和 post-update 三种时序。BO 对三者都天然兼容；SD 不使用 post-update 选择。

本文还定义两种拓扑实例：

- **单层并列 receivers**：用于隔离验证 selector、状态、N/SD/BO、Emit 和聚合；
- **HB-Lattice**：在同一公共语义上增加规则化 Lines、波前 barrier 和扩展—平台—收拢拓扑。

完整数学定义、执行顺序、loss 和命名见 [实验语义、命名与数学符号](docs/experiment-semantics-and-naming.md)。该文档是“模型实际怎样计算”的权威来源；软件边界与建设顺序见 [SettleGraph 实现与等价性验证计划](docs/settlegraph-implementation-plan.md)，资格测试的 fixture、trace、comparator 和证据门槛见 [SettleGraph 等价性测试契约](docs/equivalence-test-contract.md)。[SettleGraph core-v1 资格计划](docs/core-v1-qualification-plan.md)把当前闭合子集落实为待执行的语料、计数和 capability cells；它是工作单，不是当前通过证据。已完成的 executor development run、其证据边界和后续交接事项见 [执行器等价性开发验证状态](docs/executor-equivalence-development-status.md)。

从真实 Base checkpoint 开始的 BO 实验、特化加速与后续多卡扩展顺序见 [checkpoint 到 SettleGraph + BO 实验路线](docs/checkpoint-to-bo-experiment-roadmap.md)。

## 3. 主要实验问题

后续实验需要逐步回答：

- 新增 SettleGraph 能否从函数等价的 checkpoint 起点稳定离开中性初始化；
- 私有状态是否被后续 selector 或 NodeCompute 真正读取，而不只是发生数值变化；
- 在匹配 Plan、active route 和计算预算时，BO 相比 SD 是否具有可复现价值；
- content、pre-update 和 post-update selector 分别带来什么影响；
- 多父聚合、局部交叉边和多层传播能否缓解下游历史覆盖不足；
- 增加可达容量和传播深度时，训练稳定性、信用分配、节点饥饿和路径集中是否可控；
- 被跳过的昂贵计算能否覆盖 selector、状态、消息、packing 和通信成本。

主要外部基线包括原生 Dense checkpoint、匹配参数或计算量的 Dense 扩展，以及实现细节明确的 Flat MoE。SettleGraph 内部则使用 N/SD/BO、状态 knockout、selector 时序、边开关和 route replay 建立可归因对照。

## 4. 当前仓库状态

当前仓库已有 Stage A 的 eager reference、固定 K `core-v1` 的通用 packed executor，以及单层和 HB-Lattice 的拓扑特化 executor，但尚无科学实验结果，也没有完成等价性测试契约所定义的完整资格验证。

当前代码和定向测试覆盖：

- CPU-safe 的运行时解析边界，logical/typed Plan 的规范化、静态校验和稳定哈希，当前 reference formula config 的严格键集、默认值物化、数值规范化与跨字段 shape/timing 校验，以及外部可变状态在不同 owner 键间共享 Tensor storage（包括不同 view）的运行期拒绝；当前 Plan gate 也明确拒绝非空 `parameter_group`，保证 SettleGraph 参数各自独占；单个 SettleGraph site 已能从 Plan 派生版本化、实现无关的 parameter-schema manifest，并把 eager `state_dict` locator 单列为实现 binding，逐项校验公式角色、owner、shape、dtype 和一一对应关系；
- 若干完全展开的手工拓扑与 HB fixture Builder；
- 一个固定 seed 的 development corpus：48 个合法 Plan 覆盖 singleton、单层 \(R=2,8\)、chain、diamond、unequal path、multi-entry/multi-terminal、mixed regions、forced backbone、小型 HB 和有界分层 DAG；六种合法 profile/timing、none/EMA/Gated DeltaNet/窗口 Attention、三种 Emit、三种 receiver Aggregate、两种 output Aggregate，以及 fixed-K core 与 input-K 接口扩展，均进入 FP32/FP64 token-major—region-major 差分，其中 6 个 Plan 比较 hidden 以及每个具名参数的 gradient/`None` 记录，并要求至少一个参数梯度非零；另有 24 个命名的单变换非法 Plan，直接验证 validator 自产的 schema/topology/formula code，不把期望 code 回灌给捕获器；
- 一个固定身份的 256-slot、固定 K `core-v1` executor candidate corpus，其中 64 个预标记 VJP；通用 packed 的静态支持谓词接受 256/256，单层与 HB 特化谓词分别预选 8/16 个。开发 suite 对全部 256 个 Plan 在 FP64/FP32 下比较 eager—packed 的 output、state、balance、full trace、route invariant，并直接比较整段 prefill、`T=3` 的两种非空 two-chunk splits 和逐 Token decode 的 forward；每段 chunk 和每步 decode 的 trace 也直接对照 eager，并在规范合并后与整段 trace 对照。每个预标记 VJP case 在两种 dtype 下从同一次保留计算图依次隔离查询 output、可微 balance loss、每个最终状态 owner/component、每个 region 的 `soft_sum`、每个可微 trace region event 的 logits/probabilities、组合目标和再次查询的 output，并逐项比较 hidden、全部具名参数及其 `None` 连通性。另有 FP64 定向回归覆盖 EMA、Gated DeltaNet、窗口 Attention 的空初态与可微外部初态（Attention 分开检查 keys/values），递归比较公开 `ExecutionResult` 中全部 Tensor 的 shape 与 `requires_grad`，并覆盖保留计算图跨 chunks 的 public-result 及 trace state objectives；固定反例还逐个检查 node-event、edge、parent-message 和 terminal-message 的事件局部 autograd provenance，防止 packed 分组执行产生 lane 间的 `requires_grad` 或 connected-zero/`None` 污染。对全部 24 个静态适用 case，suite 还执行 eager—packed—specialized 三方 FP64/FP32 full/`T=3` 的两种非空 two-chunk splits/decode forward 与另行 full-prefill VJP，FP64 下另比较全部 live 公开 Tensor 的 autograd 元数据；一个同时含 EMA、GDN 和 Attention 的 HB case 另将 17 个调用方初始状态 leaves 纳入三方 VJP。有状态代表集还覆盖 reset、row reorder、empty tail 与失败回滚；这些 candidates 尚未物化为资格计划要求的冻结 bundles；
- 当前 reference 算子子集上的逐 Token/token-major eager 执行和独立 region-major eager prefill，包括 N/SD/BO、content/pre/post、状态 carry、mask、运行期 K、事务回滚、balance 充分统计和规范 trace；
- `tide.generic-packed.torch.v1`：直接使用同一 `SettleGraph` owner 的参数，对当前任意固定 K `core-v1` DAG 按 region 处理整段 \([B,T]\)，跨 region 的待传消息缓冲只保存实际 `DATA` records；覆盖当前三种 state、六种 profile/timing、prefill/decode、forced singleton、trace 和梯度路径，不调用 eager scheduler。SD/pre 的离散 route discovery 和因果 state replay 放在 TorchScript 循环中；
- `single-layer.v1` 与 `hb-line.v1`：前者是无状态 N/content 单层的平铺 Tensor 特例，后者按 fully expanded HB Plan 的 Lines 与 barriers 独立调度。二者都静态拒绝不适用 Plan、复用同一参数 owner 且不回退到通用 eager；HB 当前是独立 Line-barrier 调度 oracle，不是独立局部公式 oracle，也不是高性能 kernel；
- eager、通用 packed 与单层特化的共享执行入口当前都只接受 FP32/FP64，并在任何公式执行前拒绝空 batch；这只是已实现调用域的一致性，不表示低精度或空 batch 获得了语义资格；
- `tide.settlegraph.fixture.v1` CPU bundle 的 no-replace 原子发布与单次 bytes 读取的 weights-only-safe 装载，包括 logical/typed canonical bytes 与 hash、Plan 派生 parameter schema、按逻辑参数键保存的 Tensor、逐 Tensor 的值、stride、storage offset/group 与完整 backing-storage hash、类型分域的内容 hash、独立文件 hash、序列输入、mask、K、chunk/detach 控制、初始状态、期望与 VJP 路径记录；梯度键集必须精确覆盖 hidden 和 bundle 中的全部逻辑参数，loader 按固定阶段检查完整性、schema、Plan/binding、Tensor、mask、状态 owner、位置和 K 容器。三种命名 mutation 已能保存并复现真正的 Plan topology、mask 和不同 owner 共享 backing storage 负 bundle；仅声明 failure 而没有可达缺陷的假负例会被拒绝；另有 `tide.failure.v1` 的稳定 phase/code envelope、唯一 JSON 解析、比较和不解析异常文本的捕获 helper；
- `scripts/run_development_corpus.py` 会拒绝覆盖已有目录，在导入 Torch 和项目代码前取 source snapshot，锁定本 checkout 的 module/test 来源和 CPU default runtime，运行上述有界语料，并持久化 `run.json`、原始 `metrics.jsonl`、`stdout.log`、`summary.json` 和完整 corpus manifest/hash；非法 Plan 的变异后结构身份也进入 corpus hash，skip、缺测试、expected failure 或运行中 source 变化都产生失败终态。记录明确标记 `qualification=false`，不会把 development run 提升为 capability 资格；
- `scripts/run_executor_equivalence.py` 以同样的 no-overwrite 和运行前后 source fingerprint 约束执行 executor candidate suite，固定 candidate/support digest 与关键测试发现集，只允许 clean exact-commit 成功终态；除语料、原始日志和终态外，它还保存逐 case/dtype/mode 与逐 VJP objective 的执行回执，并用期望键全集校验实际 forward、VJP 和 lifecycle 覆盖后才写成功。该 runner 同样明确记录 `qualification=false` 及未完成的资格门；
- POST、PARBLK、PARATTN、PARMLP 四种 placement 的独立方程测试；
- 基础 CPU checkpoint v1 的 `init-from`/`resume` round trip，包括 Plan/binding/参数 dtype、Adam/AdamW 类型及超参数域、稳定模型参数组与顺序、已初始化 optimizer state manifest、optimizer Tensor shape/dtype/storage alias，以及 receiver Tensor/窗口 Attention 状态的校验和继续执行；通用状态序列化器能严格编码、解码 selector-history 容器，但当前 eager executor 和 checkpoint attach 明确拒绝非空 selector-history，不能据此声称 history continuation 已实现；保存端只接受 weights-only-safe 元数据并自检，CPU 序列状态先做 owner/storage-alias 校验再转到目标 device；注入式 commit failure 会联合回滚 model `state_dict`、optimizer containers/defaults 和 CPU RNG；序列状态使用“下一待执行位置”字段并拒绝旧字段；
- 2026-09-03 在本机 aarch64 `Ascend910_9392`、Torch `2.10.0+cpu`、TorchNPU `2.10.0`、CANN `9.0.0` 上，对后来提交为 `c6e2cc5` 的 eager-reference 基线内容完成一次 FP32 定向 attempt：由 site launcher 分配设备并在进程内使用 logical index 0，显式 NPU runtime suite 22/22、live semantic 3/3、独立 CPU→NPU fixture parity 和 CPU checkpoint continuation 均通过，parity 最大绝对/相对误差分别为 \(5.96\times10^{-8}\) 和 \(1.43\times10^{-6}\)；CPU parity artifact/checkpoint SHA-256 分别为 `944378eb1ad4e7ba20205eeb81f8243b4aebad85763f7db27139dde29964861f` 和 `5a4c155bb5ada1e1b47a30fc5628e622a225600f282fd42964d2df8fe6614172`。该运行记录不是 clean exact-commit 资格证据，且不覆盖此后新增的 fixture、parameter manifest、failure envelope 和扩大语料。另一次较早的 EMA、Gated DeltaNet 与窗口 Attention region-major forward/backward profiler attempt 观察到 NPU kernels，未观察到 AI_CPU task 或显式 fallback 记录。

region-major reference 仍按 region、Token 和 batch row 使用 Python 循环，只是 eager 调度 oracle。新的 packed 路径已经 tensorized，但当前实现仍有位置校验、StateStore 打包/发布、下一位置和静态参数分组等 host-side mapping，严格 FP32 顺序的部分递推位于编译循环。一次 aarch64 CPU 探索性 benchmark 已采集长 prefill 与单 Token stateful decode 延迟：它不是冻结的 `X07` workload，缺少峰值内存与 profiler 资格工件，并且 decode 仍是长度 1 的 packed prefill wrapper；因此不能把它记为通过性能资格。尚未完成的主要范围包括：

- packed/特化的正式 qualification、性能与目标设备优化 kernel，以及混合/低精度的逐公式 accumulation role 与资格门槛；
- 真实 Qwen 的 causal mask、position IDs、KV cache、logits、LM loss 和 Base 参数梯度接入；
- selector-history 递推；
- 可学习首状态的 Plan owner/key/shape 与 reset、梯度累加 schema；当前 fixture v1 因而明确只接受空的 `learnable_initial_state`；
- 完整多类别的独立 golden、把 development candidates 物化为长期保存 bundle 的资格语料、跨语言 canonicalizer conformance、按资格规则选择并冻结 256 legal/64 VJP/16 optimizer/96 Plan-or-runtime-input negative 加另计 8 artifact negative、满足 pairwise/event 计数、失败收缩、系统化故障注入、短训练与完整 checkpoint qualification；当前 256 candidates 达到了合法 Plan 数量并提供更广开发覆盖，但尚未满足这些 bundle、选择和证据约束；
- checkpoint v1 尚未覆盖 scheduler、scaler、backend RNG、sampler/data cursor、未归约统计窗口或窗口中途恢复；基础 CPU checkpoint 的跨 device 装载路径已经实现并有定向 continuation 用例，但完整 portable handoff 资格（完整训练状态、optimizer 下一步、规定数量和可追溯证据）尚未完成；它只支持已声明的 Adam/AdamW schema，也不承诺回滚任意 Python 属性或 load hook 的外部副作用；
- 基线 NPU attempt 仍只有一个 BO/post 定向 parity fixture，运行记录没有 clean exact-commit 身份，且当前扩展代码尚未复验；它没有达到契约的 64 个 forward、32 个 VJP、8 个 optimizer/checkpoint case。bundle 已能保留并认证非连续 stride，但 NPU attempt 尚未覆盖完整算子、shape/layout、optimizer/checkpoint profiling、packed、低精度或短训练。较早 profiler attempt 还报告默认 schedule 可能不完整，其结果不能替当前代码提供 fallback closure，也不能外推为全路径无 CPU fallback。因此 NPU capability 仍为 `implemented`，完整 qualification 未完成；CUDA 的相应能力仍属规划。

因此，现有测试通过只说明对应 reference 子集的开发检查结果，不等于任一完整 capability cell 已达到 `verified`。语义文档仍是模型计算含义的权威来源；实现计划中的未完成阶段也不增加新的模型语义。

## 5. 术语

- **TIDE Architecture / TIDE Network**：模型结构与 reference semantics；
- **TIDE Model**：按某个 TIDE 架构训练得到的具体模型；
- **TIDE Engine**：训练或推理 TIDE Model 的 runtime；
- **SettleGraph**：本文当前定义的固定 DAG 单次结算语义；
- **Plan**：经过静态校验、完全展开的 SettleGraph 图描述；
- **receiver node**：图中的有状态或无状态计算节点；
- **region**：共享一次局部选择的固定 receiver 集合。

## 6. 当前不能主张的结论

目前不能声称：

- TIDE 已经优于 Dense Transformer、Mamba 或 Flat MoE；
- BO 已经具有 learning、scaling 或系统收益；
- 状态发生变化就等于状态已被模型有效使用；
- 多父交叉汇聚必然改善下游历史覆盖或模型质量；
- HB-Lattice 是唯一或最优的容量扩展拓扑；
- 固定逻辑邻接会自动转化为更低的物理通信成本。
