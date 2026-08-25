# `experiment-semantics-and-naming.md` 审视意见台帐

> 状态：待逐条对齐。
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
| **已对齐** | 已形成明确结论，尚未修改正文 |
| **已修改** | 正文已经修改，尚待最终复核 |
| **关闭** | 修改和复核均已完成，或明确决定不修改 |

优先级含义：

| 优先级 | 含义 |
| --- | --- |
| **P0** | 会使规范无法唯一确定计算语义，应优先解决 |
| **P1** | 不影响当前 H1 历史实验，但会影响自包含性或新实验可比性 |
| **P2** | 表述、术语、可验证性或局部数学细节问题 |

## 2. 待对齐意见

| 编号 | 优先级 | 状态 | 涉及章节 | 审视意见 | 建议方向或完成标准 |
| --- | --- | --- | --- | --- | --- |
| **ESN-001** | P0 | 待对齐 | 1.1、1.3、4、5.2 | \(\ell\) 先被定义为 base block 编号，后面又承担 routed insertion site 的索引；micro-batch 公式也只写 \(t\)，没有区分不同序列。进入 H2 或异构 site 后，状态、路由和 loss 的归属无法唯一表达。 | 分开定义序列/Token、base block、插入 site、递归节点或层级的索引；允许正文省略不必要下标，但首次必须说明。 |
| **ESN-002** | P0 | 待对齐 | 文首、1.3、5.1、5.2 | 文档只定义了单层 receiver group，却允许正式名称使用 H2。H2 可能表示两组串联、树形递归、共享下级 group 等多种不同计算图。 | 二选一：补充唯一的标准 H>1 递归语义；或明确当前规范只完整定义 H1，H>1 在递归语义落定前只是保留字段、不得用于正式条件名。 |
| **ESN-003** | P0 | 待对齐 | 1.3.1、1.3.2、1.4.2、4.3、5.1 | `SEL-CONTENT` 被定义为只读当前消息，但 N 又允许持久 SelectorState，4.3 的负载感知 selector 也会读取旧负载。现有三种 selector 标签和 \(\mathcal R\) 接口无法表示这种条件。 | 把 ReceiverState 与 SelectorState 作为两个明确坐标；为 SelectorState 定义符号、读写时序和命名方式，并决定 `SEL-CONTENT` 是否严格排除任何持久状态。 |
| **ESN-004** | P0 | 待对齐 | 1.4.5 | Attention state 的示例默认 receiver 保存窗口内每个 Token，但 SD 只能保存该 receiver 实际 Observe 的 Token；当前公式实际只自然对应 BO。key 维度 \(K\) 与 key 矩阵 \(K_{\ell,t}\) 也发生重名，堆叠的 key/value 矩阵未定义。 | 用“实际 Observe 的时间集合”定义历史；明确 Append/Evict 的 `Update`；定义堆叠矩阵并用 \(d_k\) 等符号表示维度；说明完整历史是否只在固定最大上下文下视为有界。 |
| **ESN-005** | P0 | 待对齐 | 4、4.1、4.2 | 当前统一使用一个 \(\mathcal V\)。对现有 flat H1 sites 没有问题，但层次递归中不同局部 router 可能实际处理不同 Token 子集。另一方面，同一个 router 下所有 receivers 应共享同一输入集合，而不是使用各 receiver 的激活 Token 集。 | 改为每个 router/node 的 \(\mathcal V_j\) 和 \(N_{V,j}\)，并明确“同一 router 下与 receiver \(i\) 无关”；同时定义异构 \(R_j\) 和多个 router loss 的聚合方式。 |
| **ESN-006** | P1 | 待对齐 | 1.3.1、1.3.2 | `Score_i(m,s_i)`只允许 logit \(i\) 读取本 receiver 的状态；“足够通用的中央 selector”则可能联合读取整个 \(S\)。两种函数族并不相同。 | 明确 selector 是 candidate-local 还是 sibling-joint；若两者都允许，统一写成可读取 \(S\) 的一般形式，并把只读 \(s_i\) 作为受限实现。 |
| **ESN-007** | P1 | 待对齐 | 1.3.2、5.1、6 | Soft-P 与 Hard-ST 的前向及反向语义不同，但科学条件名没有 gate 字段，同名 run 可能实际采用不同 gate。 | 增加 `GATE-SOFTP / GATE-HST`；或声明 Soft-P 是唯一无后缀默认值，任何非默认 gate 必须进入名称。第 6 节应记录精确 \(G\) 公式，而不只写“merge 权重”。 |
| **ESN-008** | P1 | 待对齐 | 1.3、5.1 | 允许值表没有阻止无定义组合，例如严格 SD + `SEL-POST`，以及没有 SelectorState 扩展时的 N + `SEL-PRE/POST`。broadcast-proposal 若进入实验，也不能继续冒充 SD 或 BO。 | 增加简短兼容表：N 仅 content、SD 支持 content/pre、BO 支持三者；SelectorState 和 broadcast-proposal 另行扩展并命名。 |
| **ESN-009** | P1 | 待对齐 | 3.2、4.2、5.4 | M8 在损失章节直接出现，但没有说明它是历史名称，也没有自包含地写出复制原 dense MLP、Top-1、无 capacity、无 token drop、无 reroute 和 \(g=1\) 的完整设置。 | 在 3.2 首次定义 M8 历史别名与初始化/dispatch 语义；把不设 capacity、不丢 Token 等当前事实写清楚，并说明正式新名称怎样表示它。 |
| **ESN-010** | P1 | 待对齐 | 1.3、1.4、6 | 公式从 \(S_{t-1}\) 开始，但没有定义序列首状态；也没有在核心语义中明确状态逐序列隔离、无效 Token 是否 Update、chunk 是继承还是清零、跨 chunk 是否截断梯度。`prefill = decode` 也未作通俗解释。 | 定义 \(S_{-1}\) 或统一初始状态接口；写明有效 Token mask；说明当前历史实验每条序列清零。把 lifecycle、carry/reset、detach/BPTT 作为正式设置必填项，并解释 `prefill = decode`。 |
| **ESN-011** | P1 | 待对齐 | 文首、5.1、5.3 | 文首称名称应说明“从什么权重开始训练”，但 TRAIN 只表达 PT/CPT/FT/SFT 类别，精确 checkpoint 和 revision 留在 manifest；表述承诺超过了名称实际承载的信息。 | 把文首改为“权重初始化类别与训练阶段”，或扩充名称；继续让精确 checkpoint 谱系由 manifest 保存。 |
| **ESN-012** | P1 | 待对齐 | 4、5、6 | `0.01`、`0.001` 和 Soft-P 等既像固定规范，又实际上可以由配置改变；当前实现、历史实验默认值和未来允许值的边界不够清楚。 | 对每项明确标注“当前历史实验值”“新实验默认值”或“规范固定值”；任何可配置且影响比较的值都必须进入 manifest，关键实验轴进入短名称。 |
| **ESN-013** | P2 | 待对齐 | 文首、1.3、1.4、4.2 | 若文档面向可独立阅读的领导或新读者，TIDE、N、SD、BO、M8、SSM、SSD、ST-MoE、`noaux_tc` 等首次出现时仍缺少展开或一句解释。EMA 的 \(\lambda_i\) 是标量还是向量、GDN 的 q/k/value 维度也被省略。 | 首次出现时补最短定义；补 \(\lambda_i\) 的取值范围/形状和 GDN 核心张量维度，不扩写成综述。 |
| **ESN-014** | P2 | 待对齐 | 1.4.4、1.4.6、4.2 备注 | 外部模型事实基本正确，但负载均衡表缺少官方出处；“公认有效”偏强，“KDA 是 GDN 的近期改进”也容易被理解为严格继承关系。 | 为模型/报告名加入官方链接；把 z-loss 改成“常用的可选稳定项”；把 KDA 表述为 delta-rule 家族中采用更细粒度门控的后续路线。 |
| **ESN-015** | P1 | 待对齐 | 5.1、6 | 名称之外的必填语义仍缺少几项容易改变实验含义的内容：参数是否跨 site/层级共享、递归 topology/branch grammar、SelectorState 生命周期、状态跨 chunk 的梯度处理、loss 聚合范围和 gate 类型。 | 补入第 6 节；若 ESN-002/003/005/007 已分别解决，这一项只负责检查清单完整性，避免重复解释。 |

## 3. 对齐记录

| 日期 | 意见编号 | 对齐结论 | 正文修改提交 |
| --- | --- | --- | --- |

## 4. 已核验、修改时应保持的部分

以下内容在基线审视中没有发现实质错误。后续修改相关章节时，应避免无意改变这些语义。

| 编号 | 已核验内容 |
| --- | --- |
| **OK-001** | Base Qwen3 block 正确表达了 Pre-Norm 与 causal prefix 依赖。 |
| **OK-002** | POST、PARBLK、PARATTN、PARMLP 四种 placement 的公式、文字和比较表相互一致。 |
| **OK-003** | 当前 H1 下，content-only/pre/post 的选择、状态提交和更新后 `Read` 顺序自洽。 |
| **OK-004** | Soft-P 与 Hard-ST 公式正确；Hard-ST 前向为 1，反向对被选概率的导数为 1，离散 Top-1 本身不反传。 |
| **OK-005** | Receiver balance loss、M8 Switch-style balance loss、stop-gradient 和 router z-loss 与当前代码一致。 |
| **OK-006** | 训练期 balance loss 与推理期负载感知 selector 的区分正确。 |
| **OK-007** | Qwen3-Next/Qwen3.5 使用 Gated DeltaNet、Kimi K3 使用 Quantile Balancing 且推理时冻结最终 bias、GLM-5.2 配置使用 `noaux_tc`，这些事实未发现硬错误。 |
| **OK-008** | 文档标题编号连续，数学与代码围栏成对，基线通过 `git diff --check`。 |

## 5. 建议对齐顺序

建议按照编号顺序推进，但可分为四组：

1. **索引与递归基础**：ESN-001、ESN-002；
2. **状态与 selector**：ESN-003、ESN-006、ESN-008、ESN-010；
3. **具体机制与损失**：ESN-004、ESN-005、ESN-007、ESN-009、ESN-012；
4. **命名、必填清单和文字收尾**：ESN-011、ESN-013、ESN-014、ESN-015。

前一组的结论可能改变后一组最合适的符号和命名，因此不建议先从文字润色开始。
