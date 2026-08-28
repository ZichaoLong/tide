# `experiment-semantics-and-naming.md` 审视意见台帐

> 状态：逐条对齐中。
>
> 审视基线：提交 `8d15efb` 中的
> [`experiment-semantics-and-naming.md`](experiment-semantics-and-naming.md)。
>
> 本台帐只记录问题、决定和处理状态，不替代规范正文。未经对齐，不据此预先修改正文。
> “涉及章节”使用规范正文的当前章节号。

## 1. 使用方式

每条意见使用稳定编号。后续逐条讨论时，更新对应行的“状态”，并把决定追加到第 3 节；正文修改完成并复核后，再把该项标为“关闭”。

状态含义：

| 状态 | 含义 |
| --- | --- |
| **待对齐** | 尚未决定是否修改以及采用什么做法 |
| **部分对齐** | 已决定一部分，仍有相关问题待讨论 |
| **已对齐** | 已形成明确结论，尚未修改正文 |
| **已修改** | 正文已经修改，尚待最终复核 |
| **关闭** | 修改和复核均已完成，或明确决定不修改 |

优先级含义：

| 优先级 | 含义 |
| --- | --- |
| **P0** | 会使规范无法唯一确定计算语义，应优先解决 |
| **P1** | 不阻塞核心公式，但会影响自包含性或实验可比性 |
| **P2** | 表述、术语、可验证性或局部数学细节问题 |

## 2. 审视意见

| 编号 | 优先级 | 状态 | 涉及章节 | 审视意见 | 建议方向或完成标准 |
| --- | --- | --- | --- | --- | --- |
| **ESN-001** | P0 | 已修改 | 1.1、2—4、6.2、7.2 | base block、site、Token、Line、node、region 和 routing event 已分开编号。 | 保持 \(b,\ell,j,t,d,w/v,r\) 的职责分离；正文可省略不影响理解的下标。 |
| **ESN-002** | P0 | 部分对齐 | 文首、2—4、7.1、7.2 | H>1 已收敛为手动规定波前的受限 HB-Lattice，不再强行解释成唯一树形 H2；具体实验由已展开 `HBLatticePlan` 与执行配置共同确定。 | 逐项核验 Plan grammar、波前顺序、镜像直通、region 与多父聚合；R/H/K 只作摘要。 |
| **ESN-003** | P0 | 已修改 | 2.3、2.4、3.3、3.4、3.7.2、6.4、7.1 | `SEL-CONTENT` 被定义为只读当前消息，但历史负载等持久信息也可能影响 selector。 | 不设置独立集中式 SelectorState；节点把可选的轻量选择历史保存在 receiver state 中，并通过 `Read^sel` 发给 selector。 |
| **ESN-004** | P0 | 已修改 | 3.7.5 | Attention state 的示例默认 receiver 保存窗口内每个 Token，但 SD 只能保存该 receiver 实际 Observe 的 Token；当前公式实际只自然对应 BO。key 维度 \(K\) 与 key 矩阵 \(K_{\ell,t}\) 也发生重名，堆叠的 key/value 矩阵未定义。 | 用“实际 Observe 的时间集合”定义历史；明确 Append/Evict 的 `Update`；定义堆叠矩阵并用 \(d_k\) 等符号表示维度；说明完整历史是否只在固定最大上下文下视为有界。 |
| **ESN-005** | P0 | 已修改 | 6、6.1、6.2 | H1 使用固定候选的 soft balance；HB-Lattice 首个设置使用 availability-conditioned soft balance。 | 核验 \(\bar p,\bar p^{\mathrm{avail}},\bar f\) 与 region reduction；其他统计目标必须另行命名。 |
| **ESN-006** | P1 | 关闭 | 2.3、2.4、3.3、3.4 | `Score_i(m,s_i)`只允许 logit \(i\) 读取本 receiver 的状态，无法表达联合打分。 | receivers 局部执行轻量 `Read^sel`；向量值 `Score` 输出全部 logits，可以逐候选独立打分，也可以联合处理这些读出。active receivers 另行执行较大的 `Read^ffn`。 |
| **ESN-007** | P1 | 已修改 | 2.4、3.4、3.5、4、7.1、8 | 单层与 HB-Lattice 曾在不同接口使用 selector 概率。 | 所有 receiver 统一由 `EmitPolicy` 承担概率与主任务梯度；`MessageAggregate` 不复用该概率；核验 EMIT-HST 的前向、反向、\(\zeta^{\mathrm{ST}}\) 与 identity 初始化行为。 |
| **ESN-008** | P1 | 关闭 | 2.4、3.3、3.4、7.1 | 允许值表没有阻止无定义组合，例如严格 SD + `SEL-POST`，以及没有 SelectorState 扩展时的 N + `SEL-PRE/POST`。broadcast-proposal 若进入实验，也不能继续冒充 SD 或 BO。 | 增加简短兼容表：N 仅 content、SD 支持 content/pre、BO 支持三者；SelectorState 和 broadcast-proposal 另行扩展并命名。 |
| **ESN-009** | P1 | 已修改 | 5.2、6.3、7.4 | M8 在损失章节直接出现，也没有自包含地写出复制原 dense MLP、Top-1、无 capacity、无 token drop、无 reroute 和不乘 soft 概率的完整设置。 | 在 5.2 首次把 M8 定义为 MOE-R8 的简写并说明初始化/dispatch 语义；把不设 capacity、不丢 Token 等设置写清楚。 |
| **ESN-010** | P1 | 关闭 | 3.6、3.7、8 | 公式从 \(S_{t-1}\) 开始，但没有定义序列首状态；也没有在核心语义中明确状态逐序列隔离、无效 Token 是否 Update、chunk 是继承还是清零、跨 chunk 是否截断梯度。`prefill = decode` 也未作通俗解释。 | 已定义空首状态、有效 Token 规则、逐序列隔离、跨 chunk carry 与默认 detach，并明确 `prefill = decode` 的判定。 |
| **ESN-011** | P1 | 已修改 | 文首、7.1、7.3 | 文首称名称应说明“从什么权重开始训练”，但 TRAIN 只表达 PT/CPT/FT/SFT 类别，精确 checkpoint 和 revision 留在 manifest；表述承诺超过了名称实际承载的信息。 | 把文首改为“权重初始化类别与训练阶段”，或扩充名称；继续让精确 checkpoint 谱系由 manifest 保存。 |
| **ESN-012** | P1 | 已修改 | 6—8 | `0.01`、`0.001` 和 Soft-P 等既像固定规范，又实际上可以由配置改变；当前实现、历史实验默认值和未来允许值的边界不够清楚。 | 对每项明确标注“当前历史实验值”“新实验默认值”或“规范固定值”；任何可配置且影响比较的值都必须进入 manifest，关键实验轴进入短名称。 |
| **ESN-013** | P2 | 关闭 | 文首、2、3、3.7、6.3 | 若文档面向可独立阅读的领导或新读者，TIDE、N、SD、BO、M8、SSM、SSD、ST-MoE、`noaux_tc` 等首次出现时仍缺少展开或一句解释。EMA 的 \(\lambda_i\) 是标量还是向量、GDN 的 q/k/value 维度也被省略。 | 首次出现时补最短定义；补 \(\lambda_i\) 的取值范围/形状和 GDN 核心张量维度，不扩写成综述。 |
| **ESN-014** | P2 | 已修改 | 3.7.4、3.7.6、6.3 备注 | 外部模型事实基本正确，但负载均衡表缺少官方出处；“公认有效”偏强，“KDA 是 GDN 的近期改进”也容易被理解为严格继承关系。 | 为模型/报告名加入官方链接；把 z-loss 改成“常用的可选稳定项”；把 KDA 表述为 delta-rule 家族中采用更细粒度门控的后续路线。 |
| **ESN-015** | P1 | 已修改 | 7.1、8 | manifest 已覆盖 Plan、builder、边类别、region、MessageAggregate、EmitPolicy、BalancePolicy、参数共享和诊断范围。 | 核验 K、EMIT、AGG、BAL 字段是否足够且没有职责重叠。 |
| **ESN-016** | P0 | 已修改 | 2.3、3.1、3.4、3.5、3.7、4.4、7.2、8 | receiver state 只条件化 FFN 输入，无法表达状态/Attention residual 后再接 Pre-Norm FFN 的默认节点模板。 | `Read^ffn` 统一返回 hidden residual；默认模板依次执行状态/上下文 residual 与 FFN residual；N 令该读出为零；两个子层合计仍算一个 H 层级。 |
| **ESN-017** | P0 | 关闭 | 2.3、2.4、3.1—3.4、3.7、7.1、8 | group 公共入口 norm 让所有 receivers 共享同一个可学习输入适配器，也混合了 selector 公共输入与 receiver 本地消息两种角色。 | selector 使用独立 `N_sel`；每个 receiver node 使用自己的 `N_R,i`，只向 selector 发送轻量 `Read^sel`；RMS 统计可复用，但可学习 scale 不共享。 |
| **ESN-018** | P1 | 已修改 | 2.1—2.3、3.1、3.4、4.2、4.4、8 | receiver node 的稳定外部契约不应等同于当前 Pre-Norm 双 residual 实现。 | 拓扑只依赖轻量读出、状态提交和完整 hidden 输出；内部状态模块、昂贵计算、归一化与 residual 由可替换的 `ReceiverNodeTemplate` 定义。 |
| **ESN-019** | P1 | 已修改 | 文首、1—4 | 单层特例、HB-Lattice、selector、receiver node 和传播 profile 在读者建立全局图景前交叉出现，主干与可选样例也未分开。 | 阅读入口只保留概括；第 2 节先按层次定义共用组件与策略，第 3、4 节再依次展开单层特例与 HB-Lattice。 |
| **ESN-020** | P1 | 已修改 | 文首、2—4、6—8 | `receiver group` 与固定单层结构重合，H1/H2 又被同时当作结构名和深度字段；K 还被重复编码进 AGG。 | 删除 `receiver group`；H 只在命名节作为派生深度摘要；K 独立表示 active 数，AGG 只表示 AggregatePort 的消息聚合。 |
| **ESN-021** | P0 | 已修改 | 文首、2.1、2.2、2.4、2.5、3.1、3.5、4、7、8 | 单层末端汇合与 HB 多父聚合被写成两套接口，selector 也容易被误解为数据图上的发散点。 | 两种拓扑共用唯一输入、输出端口；receiver 输入与 GraphBranch 输出统一使用 `AggregatePort + MessageAggregate`；selector 只控制固定 region 中的 reached nodes；概率语义统一进入 `EmitPolicy`。 |

## 3. 对齐记录

| 日期 | 意见编号 | 对齐结论 | 正文修改提交 |
| --- | --- | --- | --- |
| 2026-08-25 | ESN-001 | 已分开 batch、base block 与 site 下标；递归节点下标随 ESN-002 对齐。 | 本次提交 |
| 2026-08-25 | ESN-004、007、009、011、012、014 | 接受建议并修改正文。 | 本次提交 |
| 2026-08-25 | ESN-003、006 | 不设置独立 SelectorState；receivers 局部执行轻量 `Read^sel`，向量值 `Score` 产生全部 logits；active receivers 执行较大的 `Read^ffn`，Top-1 时只有一个。 | 本次提交 |
| 2026-08-25 | ESN-002、005 | H2 形态、局部 router 的 Token 集与 loss 聚合一起对齐。 | — |
| 2026-08-25 | ESN-008 | 不增加兼容表；broadcast-proposal 只保留为帮助发散的说明。 | 不修改 |
| 2026-08-25 | ESN-010 | 独立序列从空状态开始；无效 Token 不更新；同一序列跨 chunk 继承状态并默认 detach；整段、分块和逐 Token 执行保持等价。 | 本次提交 |
| 2026-08-25 | ESN-013 | 不再展开缩写与额外内部维度，具体实现细节由实验记录保存。 | 不修改 |
| 2026-08-25 | ESN-005、015 | H1 在每个 site 的全部有效 Token 上独立计算 balance loss，再等权平均；按 micro-batch 统计。H2 聚合随递归 topology 一起对齐。 | 本次提交（部分） |
| 2026-08-25 | ESN-015 | 参数不跨 site/层级共享；合并系数由 ESN-007 处理；状态跨 chunk 的梯度规则已补充；递归 topology 与 H2 loss 聚合仍待对齐。 | 本次提交（部分） |
| 2026-08-26 | ESN-002、015 | 四种 placement 统一接入单入口、单出口 GraphBranch；placement 只决定输入和一个 residual 的返回位置，内部递归、Top-K、聚合、平台期、交叉汇聚与收拢仍需继续对齐。 | 本次提交（部分） |
| 2026-08-26 | ESN-002、015 | GraphBranch 内部边统一传递完整 hidden 并始终使用 MIX；只有 GraphBranch 与 backbone 的边界使用 RESIDUAL_ADD。具体 H2 topology 仍待对齐。 | 本次提交（部分） |
| 2026-08-26 | ESN-002、007、015 | receiver group 只产生完整候选；Soft-P、Hard-ST、Top-K 和 RESIDUAL_ADD 统一由一个 \(\beta\) 公式表达，router 概率不再被重复使用；GATE 名称字段改为 AGG。 | 本次提交 |
| 2026-08-26 | ESN-016 | 默认 receiver node 模板先把 `Read^ffn` 的 hidden residual 加回输入，再执行 Pre-Norm FFN；N 的该 residual 为零；Attention/EMA/GDN output projection 收入 `Read^ffn`；H 只计算 receiver node 深度。 | 本次提交 |
| 2026-08-26 | ESN-017 | selector 公共消息与 receiver 本地消息分离；receiver-local `N_R,i`、状态模块、`N_F,i` 和 FFN 组成独立 receiver node，selector 只接收轻量本地读出。 | `06dafb1` |
| 2026-08-28 | ESN-001、002、015 | H>1 的执行边界改为手动规定 Line 的受限 HB-Lattice；执行器消费已展开 Plan，TopologyBuilder 只负责生成 Plan，不扩展为一般 DAG runtime。 | 本次提交（待逐项核验） |
| 2026-08-28 | ESN-002、005、007 | 增加 reached set、region selector、一次性多父 `ParentAggregate` 和波前 barrier；HB balance loss 与 selector 概率进入主任务梯度的位置继续保留为待对齐项。 | 本次提交（部分） |
| 2026-08-28 | ESN-001、005、007、015 | HB selector 概率在 sender 的 delta Hard-ST `EmitPolicy` 进入主任务梯度；多父 `ParentAggregate` 独立使用均值；region balance 采用 availability-conditioned soft 目标，并单列诊断量与命名字段。 | 本次提交（待逐项核验） |
| 2026-08-28 | ESN-018 | receiver node 的稳定输入、轻量 selector 读出、状态提交和完整 hidden 输出与内部模板分离；当前默认模板仍为 Pre-Norm 双 residual。 | 本次修改 |
| 2026-08-28 | ESN-019 | 增加自包含阅读入口，重写 H1 主线并重排 HB-Lattice 的概念顺序；状态实现样例标为可选参考。 | 本次修改 |
| 2026-08-28 | ESN-020 | 删除 `receiver group`、`MIX` 和重复的选择事件类型；正文用单层特例教学，H/T 只在命名节出现；Observe 明确为状态 commit，K 与 AGG 分离。 | 本次修改 |
| 2026-08-28 | ESN-002、007、015、021 | 单层输出与 HB 多父输入统一为 `AggregatePort + MessageAggregate`；GraphBranch 输入、输出端口不算 receiver；selector 是 region 控制模块而非发散点；所有 receiver 统一使用 `EmitPolicy`。 | 本次修改 |
| 2026-08-28 | ESN-019、021 | 阅读入口缩为一段概括；共用组件独立成章，单层特例与 HB-Lattice 改为同一接口支持的两种拓扑形态。 | 本次修改 |
| 2026-08-28 | ESN-019、021 | 明确 `GraphInputPort` / `GraphOutputPort` 也是两种拓扑共用的边界组件，并按数据图、执行策略、控制与训练期接口整理全部共用角色。 | 本次修改 |

## 4. 已核验、修改时应保持的部分

以下内容在基线审视中没有发现实质错误。后续修改相关章节时，应避免无意改变这些语义。

| 编号 | 已核验内容 |
| --- | --- |
| **OK-001** | Base Qwen3 block 正确表达了 Pre-Norm 与 causal prefix 依赖。 |
| **OK-002** | 第 1.3、1.4 节中，POST、PARBLK、PARATTN、PARMLP 四种 placement 共享同一个 GraphBranch 契约；其公式与 RESIDUAL_ADD 一致。 |
| **OK-003** | 第 3.3、3.4 节中，content-only/pre/post 的选择、active set、状态提交、`Read^sel` 和默认模板的 `Read^ffn` 顺序自洽。 |
| **OK-004** | 第 3.5 节的 EMIT-HST 前向完整保留 active receiver 输出，梯度通过 selector 概率返回；离散 Top-1 / Top-K 本身不反传；`MessageAggregate` 不重复使用该概率。 |
| **OK-005** | 第 6.1、6.3 节的单层 receiver balance loss、M8 Switch-style balance loss、stop-gradient 和 router z-loss 与当前代码一致。 |
| **OK-006** | 第 6.4 节对训练期 balance loss 与推理期负载感知 selector 的区分正确。 |
| **OK-007** | Qwen3-Next/Qwen3.5 使用 Gated DeltaNet、Kimi K3 使用 Quantile Balancing 且推理时冻结最终 bias、GLM-5.2 配置使用 `noaux_tc`，这些事实未发现硬错误。 |
| **OK-008** | 文档标题编号连续，数学与代码围栏成对，基线通过 `git diff --check`。 |
| **OK-009** | 第 3.4、3.5 节的默认 receiver node 模板、N 退化、`Read^ffn` 输出维度、Emit 与输出 `MessageAggregate` 采用同一套 block-like 语义。 |
| **OK-010** | 第 3.2—3.4、3.7 节的 `N_sel`、receiver-local `N_R,i`、三种 `Read^sel` 时序、状态样例和计算量说明使用同一套独立入口语义。 |

## 5. 建议对齐顺序

建议按照编号顺序推进，但可分为四组：

1. **索引与递归基础**：ESN-001、ESN-002；
2. **状态与 selector**：ESN-003、ESN-006、ESN-008、ESN-010；
3. **具体机制与损失**：ESN-004、ESN-005、ESN-007、ESN-009、ESN-012；
4. **命名、必填清单和文字收尾**：ESN-011、ESN-013、ESN-014、ESN-015。

前一组的结论可能改变后一组最合适的符号和命名，因此不建议先从文字润色开始。

## 6. HB-Lattice 逐项核验台帐

| 编号 | 当前写入内容 | 状态 |
| --- | --- | --- |
| **HB-001** | 第一层是受限 `HBLatticePlan + HBLatticeExecutionConfig + WavefrontExecutor`，第二层是一个或多个 `TopologyBuilder`；不实现一般 DAG。 | 待核验 |
| **HB-002** | Plan 只允许相邻 Line 普通边和逐节点声明的镜像直通；平台各 Line 共享坐标集合，每对 Line 邻接可分别指定。 | 待核验 |
| **HB-003** | 多父消息在目标 Line 一次聚合；region 只在 reached nodes 中选择；BO 更新全部 reached nodes，SD 只更新 active nodes。 | 待核验 |
| **HB-004** | receiver 输入与 GraphBranch 输出都使用 `AggregatePort + MessageAggregate`；首个基线使用归一化平均。 | 已对齐 |
| **HB-005** | 首个设置使用 EMIT-HST、AGG-MEAN 和 BAL-AVAIL-SOFT；同一次 selector 概率不在消息聚合中重复使用。 | 待核验 |
| **HB-006** | 非平凡 HB-Lattice 使用 `TOPO_ID` 指向已展开 Plan；R/H/K 只作可读摘要。 | 待核验 |
