# `experiment-semantics-and-naming.md` 审视意见台帐

> 状态：逐条对齐中。
>
> 审视基线：提交 `8d15efb` 中的
> [`experiment-semantics-and-naming.md`](experiment-semantics-and-naming.md)。
>
> 本台帐只记录问题、决定和处理状态，不替代规范正文。未经对齐，不据此预先修改正文。

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
| **ESN-001** | P0 | 部分对齐 | 1.1、1.3、4、5.2 | \(\ell\) 先被定义为 base block 编号，后面又承担 routed insertion site 的索引；micro-batch 公式也只写 \(t\)，没有区分不同序列。进入 H2 或异构 site 后，状态、路由和 loss 的归属无法唯一表达。 | 分开定义序列/Token、base block、插入 site、递归节点或层级的索引；允许正文省略不必要下标，但首次必须说明。 |
| **ESN-002** | P0 | 部分对齐 | 文首、1.3、2、5.1、5.2 | GraphBranch 已有统一外部接口，但 H2 仍可能表示两组串联、树形递归、共享下级 group 等多种不同内部计算图。 | 补充唯一的标准 H>1 递归语义；在此之前，H>1 的内部 topology 必须在实验设置中完整声明。 |
| **ESN-003** | P0 | 已修改 | 1.3.1、1.3.2、1.4.2、4.3、5.1 | `SEL-CONTENT` 被定义为只读当前消息，但 N 又允许持久 SelectorState，4.3 的负载感知 selector 也会读取旧负载。现有三种 selector 标签和 \(\mathcal R\) 接口无法表示这种条件。 | 把 ReceiverState 与 SelectorState 作为两个明确坐标；为 SelectorState 定义符号、读写时序和命名方式，并决定 `SEL-CONTENT` 是否严格排除任何持久状态。 |
| **ESN-004** | P0 | 已修改 | 1.4.5 | Attention state 的示例默认 receiver 保存窗口内每个 Token，但 SD 只能保存该 receiver 实际 Observe 的 Token；当前公式实际只自然对应 BO。key 维度 \(K\) 与 key 矩阵 \(K_{\ell,t}\) 也发生重名，堆叠的 key/value 矩阵未定义。 | 用“实际 Observe 的时间集合”定义历史；明确 Append/Evict 的 `Update`；定义堆叠矩阵并用 \(d_k\) 等符号表示维度；说明完整历史是否只在固定最大上下文下视为有界。 |
| **ESN-005** | P0 | 部分对齐 | 4、4.1、4.2 | 当前统一使用一个 \(\mathcal V\)。对现有 flat H1 sites 没有问题，但层次递归中不同局部 router 可能实际处理不同 Token 子集。另一方面，同一个 router 下所有 receivers 应共享同一输入集合，而不是使用各 receiver 的激活 Token 集。 | H1 使用每个 router 处理的全部有效 Token，并在 sites 间等权平均；H2 的局部 Token 集、异构宽度和多个 router loss 聚合随递归语义一起对齐。 |
| **ESN-006** | P1 | 关闭 | 1.3.1、1.3.2 | `Score_i(m,s_i)`只允许 logit \(i\) 读取本 receiver 的状态，无法表达联合打分。 | receivers 局部执行轻量 `Read^sel`；向量值 `Score` 输出全部 logits，可以逐候选独立打分，也可以联合处理这些读出。active receivers 另行执行较大的 `Read^ffn`。 |
| **ESN-007** | P1 | 已修改 | 1.3.2、2.1、5.1、6 | Soft-P、Hard-ST 与 Top-K 聚合的前向及反向语义不同，同名 run 不能采用不同的合并系数。 | router 概率只在 `ActiveBranchAggregate` 中使用一次；科学条件用 AGG 字段记录 MIX policy，完整设置记录精确 \(\beta\) 公式。 |
| **ESN-008** | P1 | 关闭 | 1.3、5.1 | 允许值表没有阻止无定义组合，例如严格 SD + `SEL-POST`，以及没有 SelectorState 扩展时的 N + `SEL-PRE/POST`。broadcast-proposal 若进入实验，也不能继续冒充 SD 或 BO。 | 增加简短兼容表：N 仅 content、SD 支持 content/pre、BO 支持三者；SelectorState 和 broadcast-proposal 另行扩展并命名。 |
| **ESN-009** | P1 | 已修改 | 3.2、4.2、5.4 | M8 在损失章节直接出现，也没有自包含地写出复制原 dense MLP、Top-1、无 capacity、无 token drop、无 reroute 和不乘 soft 概率的完整设置。 | 在 3.2 首次把 M8 定义为 MOE-R8 的简写并说明初始化/dispatch 语义；把不设 capacity、不丢 Token 等设置写清楚。 |
| **ESN-010** | P1 | 关闭 | 1.3、1.4、6 | 公式从 \(S_{t-1}\) 开始，但没有定义序列首状态；也没有在核心语义中明确状态逐序列隔离、无效 Token 是否 Update、chunk 是继承还是清零、跨 chunk 是否截断梯度。`prefill = decode` 也未作通俗解释。 | 已定义空首状态、有效 Token 规则、逐序列隔离、跨 chunk carry 与默认 detach，并明确 `prefill = decode` 的判定。 |
| **ESN-011** | P1 | 已修改 | 文首、5.1、5.3 | 文首称名称应说明“从什么权重开始训练”，但 TRAIN 只表达 PT/CPT/FT/SFT 类别，精确 checkpoint 和 revision 留在 manifest；表述承诺超过了名称实际承载的信息。 | 把文首改为“权重初始化类别与训练阶段”，或扩充名称；继续让精确 checkpoint 谱系由 manifest 保存。 |
| **ESN-012** | P1 | 已修改 | 4、5、6 | `0.01`、`0.001` 和 Soft-P 等既像固定规范，又实际上可以由配置改变；当前实现、历史实验默认值和未来允许值的边界不够清楚。 | 对每项明确标注“当前历史实验值”“新实验默认值”或“规范固定值”；任何可配置且影响比较的值都必须进入 manifest，关键实验轴进入短名称。 |
| **ESN-013** | P2 | 关闭 | 文首、1.3、1.4、4.2 | 若文档面向可独立阅读的领导或新读者，TIDE、N、SD、BO、M8、SSM、SSD、ST-MoE、`noaux_tc` 等首次出现时仍缺少展开或一句解释。EMA 的 \(\lambda_i\) 是标量还是向量、GDN 的 q/k/value 维度也被省略。 | 首次出现时补最短定义；补 \(\lambda_i\) 的取值范围/形状和 GDN 核心张量维度，不扩写成综述。 |
| **ESN-014** | P2 | 已修改 | 1.4.4、1.4.6、4.2 备注 | 外部模型事实基本正确，但负载均衡表缺少官方出处；“公认有效”偏强，“KDA 是 GDN 的近期改进”也容易被理解为严格继承关系。 | 为模型/报告名加入官方链接；把 z-loss 改成“常用的可选稳定项”；把 KDA 表述为 delta-rule 家族中采用更细粒度门控的后续路线。 |
| **ESN-015** | P1 | 部分对齐 | 5.1、6 | 名称之外的必填语义仍缺少几项容易改变实验含义的内容：参数是否跨 site/层级共享、递归 topology/branch grammar、SelectorState 生命周期、状态跨 chunk 的梯度处理、loss 聚合范围和 branch aggregate policy。 | 补入第 6 节；若 ESN-002/003/005/007 已分别解决，这一项只负责检查清单完整性，避免重复解释。 |
| **ESN-016** | P0 | 已修改 | 1.1、1.3.2、1.4、2.1、5.2、6 | receiver state 只条件化 FFN 输入，无法表达状态/Attention residual 后再接 Pre-Norm FFN 的标准节点。 | `Read^ffn` 统一返回 hidden residual；receiver branch 依次执行状态/上下文 residual 与 FFN residual；N 令该读出为零；两个子层合计仍算一个 H 层级。 |

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
| 2026-08-26 | ESN-016 | 标准 receiver branch 先把 `Read^ffn` 的 hidden residual 加回输入，再执行 Pre-Norm FFN；N 的该 residual 为零；Attention/EMA/GDN output projection 收入 `Read^ffn`；H 只计算 receiver group 深度。 | 本次提交 |

## 4. 已核验、修改时应保持的部分

以下内容在基线审视中没有发现实质错误。后续修改相关章节时，应避免无意改变这些语义。

| 编号 | 已核验内容 |
| --- | --- |
| **OK-001** | Base Qwen3 block 正确表达了 Pre-Norm 与 causal prefix 依赖。 |
| **OK-002** | POST、PARBLK、PARATTN、PARMLP 四种 placement 共享同一个 GraphBranch 契约；其公式与 RESIDUAL_ADD 一致。 |
| **OK-003** | 当前 H1 下，content-only/pre/post 的选择、active set、状态提交、`Read^sel` 和 `Read^ffn` 顺序自洽。 |
| **OK-004** | `ActiveBranchAggregate` 只使用一次 \(\beta\)；Soft-P 与 Hard-ST 公式正确，Hard-ST 前向为 1、对被选概率的导数为 1，离散 Top-1 本身不反传。 |
| **OK-005** | Receiver balance loss、M8 Switch-style balance loss、stop-gradient 和 router z-loss 与当前代码一致。 |
| **OK-006** | 训练期 balance loss 与推理期负载感知 selector 的区分正确。 |
| **OK-007** | Qwen3-Next/Qwen3.5 使用 Gated DeltaNet、Kimi K3 使用 Quantile Balancing 且推理时冻结最终 bias、GLM-5.2 配置使用 `noaux_tc`，这些事实未发现硬错误。 |
| **OK-008** | 文档标题编号连续，数学与代码围栏成对，基线通过 `git diff --check`。 |
| **OK-009** | 标准 receiver branch、N 退化、`Read^ffn` 输出维度和 `ActiveBranchAggregate` 展开采用同一套 block-like 语义。 |

## 5. 建议对齐顺序

建议按照编号顺序推进，但可分为四组：

1. **索引与递归基础**：ESN-001、ESN-002；
2. **状态与 selector**：ESN-003、ESN-006、ESN-008、ESN-010；
3. **具体机制与损失**：ESN-004、ESN-005、ESN-007、ESN-009、ESN-012；
4. **命名、必填清单和文字收尾**：ESN-011、ESN-013、ESN-014、ESN-015。

前一组的结论可能改变后一组最合适的符号和命名，因此不建议先从文字润色开始。
