# TIDE 实验语义、命名与数学符号

> 状态：新实验的规范性文档。
>
> 本文只定义“模型实际怎样计算”和“实验名称怎样反映计算图”。实验晋级、结果报告组织和 checkpoint 保留策略另行讨论。

当前阶段，实验名称应当分别说明：

1. base 权重的初始化类别与训练阶段；
2. GraphBranch 接在 base block 的什么位置；
3. 使用 N、SD 还是 BO；
4. 每个局部 group 或 selector region 有多少 receivers；
5. 模型中有多少个顺序插入位置；
6. 每个插入位置内部的最大 receiver-node 深度；
7. 使用什么状态结构；
8. selector 使用 content-only、pre-update state 还是 post-update state；
9. 多个 active branches 使用什么聚合策略；
10. 非平凡 HB-Lattice 使用哪一份已展开 topology Plan。

## 1. 统一数学符号

### 1.1 下标和主要变量

全文使用：

| 符号 | 含义 |
| --- | --- |
| \(b\) | micro-batch 中的序列编号；正文通常省略这一维 |
| \(\ell\) | base Transformer block 编号 |
| \(j\) | routed insertion site 编号；site \(j\) 所在的 base block 记为 \(\ell(j)\) |
| \(t\) | 序列中的 Token 位置 |
| \(i\) | 当前 routed module 中的候选编号；候选可以是 receiver 或 MoE expert |
| \(d\) | GraphBranch 内部的波前 Line 编号 |
| \(u,v\) | HB-Lattice 中的父节点与当前节点；每个节点只属于一个 Line |
| \(r\) | 一个 Line 内的 selector region 编号 |
| \(Q\) | 平台期每个 Line 共享的空间坐标集合 |
| \(x_{\ell,t}\) | 实际送入第 \(\ell\) 个 base block 的 hidden；对 \(\ell>0\)，有 \(x_{\ell,t}=y_{\ell-1,t}\) |
| \(u_{\ell,t}\) | 当前 block 完成 Attention residual merge 后的表示 |
| \(v_{\ell,t}\) | 当前 block 完成原 dense MLP residual merge 后的表示，即完整 base block 输出 |
| \(y_{\ell,t}\) | 当前 block（包括可选的 GraphBranch merge）最终送往下一个 base block 的表示 |
| \(\mu_{j,t}\) | H1 中 selector 从共享 parent hidden 得到的公共归一化消息 |
| \(m_{j,t}^{(i)}\) | H1 receiver \(i\) 在本地归一化共享 parent hidden 后得到的入口消息 |
| \(\operatorname{Inbox}_{v,t}\) | HB-Lattice 节点 \(v\) 在当前 Token 实际收到的父消息集合 |
| \(q_{v,t}\) | 节点 \(v\) 是否收到至少一条父消息的 reached 标记 |
| \(h_{v,t}\) | 节点 \(v\) 对当前 inbox 执行 `ParentAggregate` 后的完整 hidden |
| \(g_{v,t}\) | active 节点 \(v\) 完整计算后向固定 children 发送的 hidden |
| \(m_{j,t}\) | 标准 MoE site \(j\) 交给 router 和 experts 的公共归一化消息 |
| \(a_{j,t}\) | site \(j\) 的 router 在 softmax 之前产生的 logits |
| \(p_{j,t}^{(i)}\) | site \(j\) 的 router 分给候选 \(i\) 的 soft 概率 |
| \(\mathcal A_{j,t}\) | site \(j\) 为当前 Token 激活的候选集合 |
| \(c_{j,t}\) | Top-1 时唯一的激活候选，即 \(\mathcal A_{j,t}=\{c_{j,t}\}\) |
| \(\operatorname{Score}\) | 产生全部 router logits 的打分操作；具体输入由 selector 语义决定 |
| \(s_{j,t}^{(i)}\) | H1 site \(j\) 中 receiver \(i\) 的私有状态；HB-Lattice 对应写成 \(s_{j,v,t}\) |
| \(S_{j,t}\) | site \(j\) 的 receiver group 的全部私有状态 |
| \(r_{j,t,\tau}^{\mathrm{sel},(i)}\) | receiver \(i\) 在 \(\tau\in\{\mathrm{content},\mathrm{pre},\mathrm{post}\}\) 时刻局部产生的轻量 selector 读出 |
| \(\rho_{j,t}^{(i)}\) | active receiver \(i\) 在状态提交后局部读出的 hidden residual |
| \(u_{j,t}^{(i)}\) | receiver branch \(i\) 完成状态/上下文 residual merge 后的表示 |
| \(z_{j,t}^{(i)}\) | receiver branch 或 MoE candidate \(i\) 的昂贵 FFN 实际输入 |
| \(E_{j,i}\) | site \(j\) 中候选 \(i\) 对应的昂贵 routed FFN |
| \(\widehat b_{j,t}^{(i)}\) | active branch \(i\) 在当前汇合点之前返回的完整 hidden |
| \(\beta_{j,t}^{(i)}\) | `ActiveBranchAggregate` 分给 active branch \(i\) 的合并系数 |
| \(\Delta_{\mathcal G,j,t}\) | GraphBranch 相对其输入产生、发往 placement merge 的 residual |

本文按“作用”统一基本符号，具体实现只在对应小节内部展开：各类归一化都写成 \(N\)，私有状态始终写成 \(s/S\)，状态操作写成 \(\operatorname{Update}\)、\(\operatorname{Read}^{\mathrm{sel}}\) 和 \(\operatorname{Read}^{\mathrm{ffn}}\)。共享 parent 的 H1 用 \(\mu\) 表示 selector 公共消息、用 \(m^{(i)}\) 表示 receiver 本地入口消息；HB-Lattice 改由每个节点自己的 \(h_{v,t}\) 产生本地消息。标准 MoE 继续用 \(m\) 表示 router 与 experts 的公共输入。各结构都用 \(a,p,\mathcal A,c\) 表示路由、用 \(z\) 表示昂贵 FFN 的输入、用 \(E\) 表示昂贵 routed FFN。TIDE receiver 用 \(\rho\) 表示状态/上下文读出的 hidden residual，用 \(u\) 表示该 residual merge 后的表示；显式 fork-join 用 \(\widehat b\) 表示完整候选输出、用 \(\beta\) 表示聚合权重。只有 GraphBranch 与 placement 的边界才对外暴露 \(\Delta_{\mathcal G}\)。相同基本符号表示它们在框架中承担相同作用，并不表示共享参数或采用相同算法；下标说明位置或候选，EMA、GDN 等实现标签只在展开内部做法时出现。下面会省略 batch 下标 \(b\) 和部分不影响理解的下标。

### 1.2 Base Qwen3 block

令 \(N_A\) 和 \(N_F\) 分别表示 Attention 前与 MLP 前、逐 Token 执行的归一化操作；当前 Qwen3 base block 中二者都实现为 RMSNorm。\(A_\ell\) 表示 causal self-attention，\(F_\ell\) 表示原 dense SwiGLU MLP。先把第 \(\ell\) 个 block 到位置 \(t\) 为止的输入前缀记为：

$$
X_{\ell,\le t}
:=(x_{\ell,0},x_{\ell,1},\ldots,x_{\ell,t}).
$$

\(N_A(X_{\ell,\le t})\) 表示分别归一化前缀中的每个 Token，方括号后的下标 \(t\) 表示取 self-attention 在当前位置的输出。一个原始 Pre-Norm Qwen3 block 对位置 \(t\) 计算：

$$
u_{\ell,t}
=x_{\ell,t}
+\left[
A_\ell\!\left(N_A(X_{\ell,\le t})\right)
\right]_t,
$$

$$
v_{\ell,t}
=u_{\ell,t}
+F_\ell\!\left(N_F(u_{\ell,t})\right).
$$

因此 \(x_{\ell,t}\) 是实际送入第 \(\ell\) 个 base block 的当前 Token hidden；对 \(\ell>0\)，有 \(x_{\ell,t}=y_{\ell-1,t}\)，而 \(y_{\ell-1,t}\) 已经包含上一层可能存在的 GraphBranch merge。

Dense 基线没有 receiver，直接令：

$$
y_{\ell,t}=v_{\ell,t}.
$$

### 1.3 一个共享 parent 的单层 receiver group

本节定义 H1 和显式 fork-join 中最简单的特例：同一 group 的 receivers 共享一个 parent hidden \(h_t\)，选择后在当前 group 出口汇合。第 2.1 节再把它推广到多父、不同节点可有不同入口消息的 HB-Lattice。

selector 使用自己的归一化 \(N_{\mathrm{sel}}\)，receiver \(i\) 使用自己独立的入口归一化 \(N_{R,i}\)：

$$
\mu_{j,t}=N_{\mathrm{sel}}(h_t),
\qquad
m_{j,t}^{(i)}=N_{R,i}(h_t),
\quad i=0,1,\ldots,R-1.
$$

各 receiver 只在本地使用 \(m_{j,t}^{(i)}\)，并只向 selector 发送第 1.3.1 节定义的轻量 \(\operatorname{Read}^{\mathrm{sel}}\)；selector 不读取所有 receiver 的完整入口消息。使用相同 \(\epsilon\) 的 RMSNorm 时，各归一化的无参数 RMS 统计相同，可以只计算一次 \(\bar h_t=h_t/\operatorname{RMS}(h_t)\)，再分别应用互不共享的可学习 scale：

$$
\mu_{j,t}=g_{\mathrm{sel}}\odot\bar h_t,
\qquad
m_{j,t}^{(i)}=g_{R,i}\odot\bar h_t.
$$

对于有状态的 SD 和 BO，在处理 Token \(t\) 之前，把该 receiver group 的完整私有状态统一记为：

$$
S_{j,t-1}
:=\left(
s_{j,t-1}^{(0)},
s_{j,t-1}^{(1)},
\ldots,
s_{j,t-1}^{(R-1)}
\right).
$$

这里的 \(s_{j,t}^{(i)}\) 只表示“receiver \(i\) 的完整私有状态”，不预先限定它是 EMA 向量、GDN 矩阵，还是在内部额外包含历史激活记录。具体内部结构不改变框架符号。

#### 1.3.1 三种 selector 语义

selector 在什么时刻读取状态有三种可能语义：

| Selector 语义 | 打分时可读取 | SD | BO |
| --- | --- | --- | --- |
| **Content-only** | 公共消息 \(\mu_{j,t}\) 与各 receiver 从当前本地消息发出的轻量读出 | 自然兼容 | 自然兼容 |
| **Pre-update state** | \(\mu_{j,t}\) 与 receivers 从旧状态发出的轻量读出 | 自然兼容；选完后只更新 active receivers | 自然兼容；选完后更新全部 receivers |
| **Post-update state** | \(\mu_{j,t}\) 与 receivers 从更新后状态发出的轻量读出 | 严格 SD 不兼容 | 天然兼容；全部更新后再选择 |

先定义当前本地消息更新后的临时状态，以及三种时刻的轻量 receiver 读出：

$$
\widetilde s_{j,t}^{(i)}
=\operatorname{Update}_i(s_{j,t-1}^{(i)},m_{j,t}^{(i)}),
$$

$$
r_{j,t,\mathrm{content}}^{\mathrm{sel},(i)}
=\operatorname{Read}_i^{\mathrm{sel}}(m_{j,t}^{(i)}),
$$

$$
r_{j,t,\mathrm{pre}}^{\mathrm{sel},(i)}
=\operatorname{Read}_i^{\mathrm{sel}}(s_{j,t-1}^{(i)},m_{j,t}^{(i)}),
\qquad
r_{j,t,\mathrm{post}}^{\mathrm{sel},(i)}
=\operatorname{Read}_i^{\mathrm{sel}}(\widetilde s_{j,t}^{(i)},m_{j,t}^{(i)}).
$$

三种打分统一写成：

$$
a_{j,t}
=
\begin{cases}
\operatorname{Score}\!\left(
\mu_{j,t},
\left(r_{j,t,\mathrm{content}}^{\mathrm{sel},(k)}\right)_{k=0}^{R-1}
\right),
& \text{content-only},\\[4pt]
\operatorname{Score}\!\left(
\mu_{j,t},
\left(r_{j,t,\mathrm{pre}}^{\mathrm{sel},(k)}\right)_{k=0}^{R-1}
\right),
& \text{pre-update state},\\[4pt]
\operatorname{Score}\!\left(
\mu_{j,t},
\left(r_{j,t,\mathrm{post}}^{\mathrm{sel},(k)}\right)_{k=0}^{R-1}
\right),
& \text{post-update state}.
\end{cases}
$$

$$
p_{j,t}=\operatorname{softmax}(a_{j,t}),
\qquad
\mathcal A_{j,t}=\operatorname{TopKIndex}(p_{j,t},K),
\qquad 1\le K\le R.
$$

Top-1 是 \(K=1\) 的特例，此时继续记：

$$
c_{j,t}=\arg\max_i p_{j,t}^{(i)},
\qquad
\mathcal A_{j,t}=\{c_{j,t}\}.
$$

每个直接 receiver 在局部执行 \(N_{R,i}\) 和 \(\operatorname{Read}^{\mathrm{sel}}\)，只向 selector 提供小向量、范数或历史激活统计等少量标量；\(\operatorname{Score}\) 一次输出全部 logits，可以逐候选独立打分，也可以联合处理这些轻量读出。\(\widetilde s\) 是同一个 \(s\) 在“已经 Observe 当前消息、尚未完成本次选择”阶段的值。Pre 与 Post 不是包含关系：如果 \(\operatorname{Update}\) 会覆盖、压缩或遗忘旧状态，post-update state 未必还能恢复 pre-update state；若 selector 同时读取二者，应另行明确声明。

历史激活也可以作为 \(s\) 的内部内容。当前 Token 的激活结果只能影响以后 Token，因此会形成时间维上的因果递归；这不妨碍在一个 chunk 内用 scan、Torch 算子或专用 kernel 执行，但实现必须保持 `prefill = decode`。

#### 1.3.2 传播 profile、状态提交与候选输出

令 \(\mathcal O_{j,t}\) 表示当前消息实际 Observe / Update 并提交状态的 receiver 集合，\(\mathcal A_{j,t}\) 表示执行较大读出和昂贵 FFN 的 active receivers。三种传播 profile 的规则是：

| Profile | \(\mathcal O_{j,t}\)：哪些 receivers 提交 Observe / Update | 哪些 receivers 继续执行较大读出与昂贵 FFN |
| --- | --- | --- |
| **N** | 无状态，不执行 Observe / Update | \(\mathcal A_{j,t}\)；没有状态读出 |
| **SD** | \(\mathcal A_{j,t}\) | \(\mathcal A_{j,t}\) |
| **BO** | 当前 parent 的全部固定直接 receivers | \(\mathcal A_{j,t}\) |

这张表中的“当前 parent 的全部固定直接 receivers”只描述共享 parent 的 H1 group；在 HB-Lattice 中，BO 的提交集合改为当前 Token 实际收到至少一条父消息的全部 reached nodes，见第 2.1 节。

Content-only 和 pre-update state 都先完成选择，再按上表提交状态：

$$
s_{j,t}^{(i)}
=
\begin{cases}
\operatorname{Update}_i\!\left(s_{j,t-1}^{(i)},m_{j,t}^{(i)}\right),
& i\in\mathcal O_{j,t},\\[4pt]
s_{j,t-1}^{(i)},
& i\notin\mathcal O_{j,t}.
\end{cases}
$$

Post-update state 与 BO 的顺序是“全部 receivers Update → selector → 全部提交”，所以直接有：

$$
s_{j,t}^{(i)}=\widetilde s_{j,t}^{(i)},
\qquad i=0,1,\ldots,R-1.
$$

严格的 SD 不自然兼容 post-update state，因为选择发生前还不知道哪些 receivers 可以 Update。N 没有 receiver 私有状态，因此当前规范下只使用 content-only。

状态提交完成后的完整状态统一记为：

$$
S_{j,t}
:=\left(
s_{j,t}^{(0)},
s_{j,t}^{(1)},
\ldots,
s_{j,t}^{(R-1)}
\right).
$$

选择与状态提交完成后，在有状态的 SD 和 BO 中，每个 active receiver \(i\in\mathcal A_{j,t}\) 在局部执行较大的 \(\operatorname{Read}^{\mathrm{ffn}}\)：

$$
\rho_{j,t}^{(i)}
=\operatorname{Read}_{i}^{\mathrm{ffn}}\!\left(
s_{j,t}^{(i)},m_{j,t}^{(i)}
\right),
\qquad
\rho_{j,t}^{(i)}\in\mathbb R^{d_{\mathrm{model}}}.
$$

无论 selector 使用 content-only、pre-update state 还是 post-update state，\(\operatorname{Read}^{\mathrm{ffn}}\) 都读取已经提交当前 Token 后的状态。它直接返回 hidden 维 residual；Attention output projection、EMA/GDN 状态投影等实现细节都包含在该操作内部。无状态的 N 不执行状态读出，对每个 active receiver 令：

$$
\rho_{j,t}^{(i)}=0.
$$

每个 active receiver branch 先把状态/上下文读出加回未归一化的 residual stream，再执行一个 Pre-Norm FFN。令 \(N_{F,i}\) 表示 receiver branch \(i\) 的 FFN 前归一化：

$$
u_{j,t}^{(i)}
=h_t+\rho_{j,t}^{(i)},
$$

$$
z_{j,t}^{(i)}
=N_{F,i}\!\left(u_{j,t}^{(i)}\right),
$$

$$
\widehat b_{j,t}^{(i)}
=u_{j,t}^{(i)}
+E_{j,i}\!\left(z_{j,t}^{(i)}\right),
\qquad i\in\mathcal A_{j,t}.
$$

\(N_{\mathrm{sel}}\)、各 \(N_{R,i}\) 与各 \(N_{F,i}\) 的可学习参数互不共享，也不与 base block 共享。因此一个标准 receiver branch 包含“本地入口归一化与状态/上下文 residual → FFN 前归一化与 FFN residual”两个串行子层；这两个子层合计仍只算一个 receiver group 层级。只有当其完整输出继续进入下一层 receiver group 时，H 才增加。在本节的 H1 特例中，各 branches 共享 parent 输入、selector 决策和汇合操作；HB-Lattice 中同一 region 的节点可以拥有不同 parent 和不同入口消息。\(\widehat b_{j,t}^{(i)}\) 始终是当前汇合点之前的完整 hidden。单层 group 的全部直接 receivers 都执行本地入口归一化和轻量 \(\operatorname{Read}^{\mathrm{sel}}\)；每个 active branch 再执行一次较大读出和一次昂贵 FFN，N 省略较大状态读出，更深子树另计。

后文把上述 router、Observe、Update、状态读取和候选分支计算合并记为 receiver group 操作，但不在 \(\mathcal R_j\) 内使用 router 概率缩放或合并候选。对有状态的 SD 和 BO：

$$
\left(
(\widehat b_{j,t}^{(i)})_{i\in\mathcal A_{j,t}},
p_{j,t},
\mathcal A_{j,t},
S_{j,t}
\right)
=\mathcal R_j(h_t,S_{j,t-1}).
$$

对无状态的 N，省略状态输入和输出：

$$
\left(
(\widehat b_{j,t}^{(i)})_{i\in\mathcal A_{j,t}},
p_{j,t},
\mathcal A_{j,t}
\right)
=\mathcal R_j(h_t).
$$

本节 H1 中，router 概率怎样影响最终输出只在第 2.2 节的 `ActiveBranchAggregate` 中定义；HB-Lattice 的概率使用位置仍按第 2.1 节另行记录。

#### 1.3.3 状态边界

每条独立序列都从空状态开始：EMA、GDN 和 SSM 状态置零，Attention 历史为空，历史激活计数清零。padding 等无效 Token 不执行 Observe / Update，也不进入路由辅助 loss。

同一逻辑序列跨 chunk 时继承状态值，不同序列之间清零；默认在每个 chunk 边界 detach，状态继续前传，但梯度只在当前 chunk 内传播。对同一有效前缀，整段 prefill、任意分块和逐 Token decode 应在数值误差范围内得到相同的逐 Token 输出与最终状态。

### 1.4 Receiver 状态样例

第 1.3 节中的 \(s\)、\(\operatorname{Update}\)、\(\operatorname{Read}^{\mathrm{sel}}\) 和 \(\operatorname{Read}^{\mathrm{ffn}}\) 是稳定接口。本节列出有代表性的内部实现，目的是建立设计空间，不表示它们已经通过 TIDE 实验，也不预设哪一种必然最好。状态实现与 selector 时序是两个独立坐标：content-only 的 \(\operatorname{Read}^{\mathrm{sel}}\) 只读取当前本地消息，pre/post state 则额外读取对应时刻的状态；\(\operatorname{Read}^{\mathrm{ffn}}\) 不受这一选择影响。

#### 1.4.1 一览

| 样例 | 主要保留什么 | 典型消费者 | 主要特点 |
| --- | --- | --- | --- |
| **历史激活** | 激活次数、最近激活位置、概率或局部预算 | selector | 最轻量；记录控制历史，不直接保存内容语义 |
| **EMA** | 一个固定长度的低通内容摘要 | selector / FFN | 简单、稳定，但不同历史会持续混合 |
| **Gated DeltaNet（GDN）** | 固定大小的 key-value 关联矩阵 | selector / FFN | 可以按 query 关联读取，并按预测误差写入 |
| **Kimi Delta Attention（KDA）** | 带细粒度门控的 delta-rule 矩阵状态 | selector / FFN | delta-rule 家族的近期增强，门控更细但实现更复杂 |
| **SSM / Mamba-2** | 固定大小的状态空间递归状态 | selector / FFN | 与 delta-rule 不同的成熟有界状态路线 |
| **Attention** | 完整历史、局部窗口或压缩后的 key/value | selector / FFN | 设计空间大；信息保留与状态/计算成本由具体实现决定 |

两类读出都在 receiver 局部完成：\(\operatorname{Read}^{\mathrm{sel}}\) 通常只输出低维投影、范数或历史统计，\(\operatorname{Read}^{\mathrm{ffn}}\) 则在内部完成必要的 output projection，并统一输出 hidden 维 residual。“典型消费者”只是常见用法，不是硬限制。

#### 1.4.2 历史激活

历史激活可以记录每个候选被选中的次数、距上次激活的 Token 数、soft probability 的移动平均或剩余局部预算、历史 selector 打分。本次选择只能在 selector 决策完成后写回，因此只影响以后 Token。若它只服务于 selector，则对应的 \(\rho_i=0\)。

#### 1.4.3 EMA

EMA\(D\) 把收到的内容压缩成一个长度为 \(D\) 的固定向量：

$$
s_{j,t}^{(i)}\in\mathbb R^D,
\qquad
o_{j,t}^{(i)}
=\tanh\!\left(W_i^{\mathrm{obs}}m_{j,t}^{(i)}+b_i^{\mathrm{obs}}\right).
$$

它对统一接口的实现为：

$$
\operatorname{Update}_i^{\mathrm{EMA}}(s_{j,t-1}^{(i)},m_{j,t}^{(i)})
=\lambda_i\odot s_{j,t-1}^{(i)}
+(1-\lambda_i)\odot o_{j,t}^{(i)},
$$

$$
\operatorname{Read}_i^{\mathrm{ffn,EMA}}(s_{j,t}^{(i)},m_{j,t}^{(i)})
=W_i^{\mathrm{out}}s_{j,t}^{(i)}.
$$

EMA 是最简单的内容记忆基线：新观察按 \(1-\lambda_i\) 写入，旧状态按 \(\lambda_i\) 保留。EMA128 就是 \(D=128\)。

#### 1.4.4 Gated DeltaNet 与 KDA

Gated DeltaNet（GDN）把同一个框架状态 \(s\) 实现为固定大小的关联矩阵：

$$
s_{j,t}^{(i)}\in\mathbb R^{K\times V}.
$$

这里先抽取 gated delta-rule 的核心状态语义，不默认复制完整开放模型 block 中的短卷积、输出门或其他外围结构；若实验加入这些部件，必须单独声明。

调用 \(\operatorname{Update}\) 或 \(\operatorname{Read}^{\mathrm{ffn}}\) 时，receiver \(i\) 从本地消息 \(m_{j,t}^{(i)}\) 生成归一化的 query/key、value 和写入门：

$$
q_{j,t}^{(i)}=N_q(W_i^q m_{j,t}^{(i)}),
\qquad
k_{j,t}^{(i)}=N_k(W_i^k m_{j,t}^{(i)}),
\qquad
\nu_{j,t}^{(i)}=W_i^\nu m_{j,t}^{(i)},
$$

$$
\eta_{j,t}^{(i)}
=\sigma\!\left((w_i^\eta)^\top m_{j,t}^{(i)}+b_i^\eta\right),
\qquad
\gamma_{j,t}^{(i)}
=\exp\!\left[
-\exp(\alpha_i)\,
\operatorname{softplus}\!\left((w_i^\gamma)^\top m_{j,t}^{(i)}+b_i^\gamma\right)
\right].
$$

其中 \(N_q,N_k\) 表示 query/key 的向量归一化，\(\gamma\) 控制旧状态保留量，\(\eta\) 控制本次误差写入量，\(\alpha_i\) 是可学习的衰减参数。

GDN 先衰减旧状态，再只写入当前 value 与已有预测之间的误差：

$$
s_{j,t,\mathrm{decay}}^{(i)}
=\gamma_{j,t}^{(i)}s_{j,t-1}^{(i)},
\qquad
e_{j,t}^{(i)}
=\nu_{j,t}^{(i)}
-(k_{j,t}^{(i)})^\top s_{j,t,\mathrm{decay}}^{(i)},
$$

$$
\operatorname{Update}_i^{\mathrm{GDN}}(s_{j,t-1}^{(i)},m_{j,t}^{(i)})
=s_{j,t,\mathrm{decay}}^{(i)}
+\eta_{j,t}^{(i)}k_{j,t}^{(i)}(e_{j,t}^{(i)})^\top,
$$

$$
\operatorname{Read}_i^{\mathrm{ffn,GDN}}(s_{j,t}^{(i)},m_{j,t}^{(i)})
=W_i^{\mathrm{out}}
\left[(q_{j,t}^{(i)})^\top s_{j,t}^{(i)}\right].
$$

因此 GDN 比 EMA 多了“按 key 写入、按 query 读取”的结构。它已经被开放权重的 [Qwen3-Next](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) 和 [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) 系列直接采用，是很强的现代参考点，但这不证明它对 TIDE receiver 必然最优。

[Kimi Linear](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base) 使用的 Kimi Delta Attention（KDA）同样为 delta rule 引入细粒度门控，并公开了训练权重与 chunk/recurrent kernel。它可以作为 GDN 之后的增强候选；代价是状态更新、参数匹配和 kernel 移植都更复杂，因此不必在第一轮同时实现。

#### 1.4.5 Attention 状态

Attention receiver 可以把实际 Observe 到的 key/value 作为状态 \(s\)，再用当前 query 执行普通 Attention。下面以保留最近 \(W\) 次 Observe 为例：

$$
q_{j,t}^{(i)}=N_q(W_i^q m_{j,t}^{(i)}),
\qquad
k_{j,t}^{(i)}=N_k(W_i^k m_{j,t}^{(i)}),
\qquad
\nu_{j,t}^{(i)}=W_i^\nu m_{j,t}^{(i)}.
$$

$$
\mathcal H_{j,t}^{(i)}
:=\{\tau\le t\mid i\in\mathcal O_{j,\tau}\},
\qquad
\mathcal W_{j,t}^{(i)}
:=\operatorname{Last}_W(\mathcal H_{j,t}^{(i)}),
$$

$$
s_{j,t}^{(i)}
=\left((k_{j,\tau}^{(i)},\nu_{j,\tau}^{(i)})\right)_{
\tau\in\mathcal W_{j,t}^{(i)}}.
$$

令 \(\mathbf K_{j,t}^{(i)}\) 和 \(\mathbf V_{j,t}^{(i)}\) 分别表示状态中按时间堆叠的 key 和 value，key 维度记为 \(d_k\)，则：

$$
\operatorname{Read}_i^{\mathrm{ffn,Attn}}(s_{j,t}^{(i)},m_{j,t}^{(i)})
=W_i^{\mathrm{out}}
\left[
\operatorname{softmax}\!\left(
\frac{q_{j,t}^{(i)}\mathbf K_{j,t}^{(i)\top}}{\sqrt {d_k}}
\right)\mathbf V_{j,t}^{(i)}
\right].
$$

对应的 \(\operatorname{Read}^{\mathrm{sel}}\) 可以只在 receiver 局部输出该向量的范数和少量历史激活统计。

实际实现也可以保留完整历史，或使用分层/稀疏选择、压缩 key/value、固定记忆槽位。完整历史的状态和读取成本随上下文增长；其他方案成本更可控，但会引入不同的信息选择。实验应如实记录实际状态量、读取成本和被保留的历史范围。

#### 1.4.6 其他有界状态路线与当前定位

SSM / Mamba-2 是另一类重要的固定状态候选，开放权重的 [Falcon-H1](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Base) 已采用 Transformer 与 Mamba 的混合结构；RWKV-7、Lightning Attention 等也提供了可参考的递归或线性注意力状态。它们证明“有界 recurrent state”有多条成熟路线，但不必全部进入首轮 TIDE 实现。

当前更合适的定位是：历史激活用于最轻量的 selector 控制，EMA 作为简单内容基线，GDN 作为第一种先进关联记忆锚点，Attention 保留为可按预算选择的宽泛设计族；KDA 和 Mamba/SSD 则是增强或跨家族候选。这只是帮助建立全局观，不是固定实验顺序。维度和状态量必须在名称中明确，例如 **GDN-K32-V32** 有 \(32\times32=1024\) 个状态标量，不能与 EMA128 当作等状态量对照。

## 2. GraphBranch 的 HB-Lattice 拓扑、内部合并与四种插入位置

以下令 site \(j\) 插在 base block \(\ell=\ell(j)\)。每个 site 在原有 base computation 之外只接入一个单入口、单出口的主旁路，记为 \(\mathcal G_j\)。它接收完整 hidden \(h\)，内部始终传递完整 hidden，最终返回一个同维表示：

$$
b_{\mathcal G,j,t}(h)=\mathcal G_j(h),
\qquad
\Delta_{\mathcal G,j,t}(h)=b_{\mathcal G,j,t}(h)-h.
$$

递归、局部选择、平台传播、多父汇聚和最终收拢都封装在 \(\mathcal G_j\) 内；外层 placement 只决定 \(h\) 从哪里取得，以及唯一的 \(\Delta_{\mathcal G}\) 在哪里合并。

### 2.1 受限 HB-Lattice 与波前执行

H>1 的标准执行对象是手动规定波前的 **HB-Lattice**，不是一般 DAG。数学上它是一个分层 DAG，但执行器只处理下面这类受限结构：

- GraphBranch 只有一个入口和一个出口；
- 节点被静态分配到有序 Line \(L_0,L_1,\ldots,L_D\)，每个节点只属于一个 Line；
- Line 内没有消息依赖，普通边只连接相邻 Line；
- 递归扩展节点可以沿显式声明的镜像直通边连接到对应收拢节点；
- 平台期各 Line 使用相同的空间坐标集合，每对相邻平台 Line 的连接可以分别指定；
- 每个 Line 被划分为固定、不重叠的 selector regions，每个节点只有一个 activation owner；
- 同一 Token 在一个节点只聚合一次父消息、更新一次状态并至多执行一次昂贵计算。

因此近期实现不需要任意拓扑排序、同层依赖、异步 event queue、有环执行或一般 DAG 接口。

#### 2.1.1 两层拓扑接口

第一层由规范化的 `HBLatticePlan` 和逐 Line 执行它的 `WavefrontExecutor` 组成。一个 Plan 至少完整列出：

~~~text
HBLatticePlan
├── Lines：每层的 phase、节点、坐标和 selector regions
├── adjacent edges：扩展、平台和收拢的相邻 Line 边
├── mirror map / edges：扩展与收拢节点的对应关系及逐节点直通开关
├── entry / sink
├── node、selector 与 ParentAggregate 配置
└── edge class：tree / local / shortcut / mirror
~~~

扩展树和收拢树可以采用不均匀但有界的分支结构；平台期每对 Line 的邻接也可以不同。`WavefrontExecutor` 只消费展开后的 Plan，不负责猜测树形、空间邻接或镜像关系。

Plan 载入时必须检查 Line 顺序、region 唯一归属、边类型与端点、entry 到 sink 的静态可达性、镜像对应关系，以及声明的 fan-in/fan-out 和 region-size 上界。

第二层由一个或多个 `TopologyBuilder` 组成：

~~~text
TopologyBuilder(config) → HBLatticePlan
~~~

builder 可以生成规则树、逐坐标混合、重复空间 Graph 或其他 HB-Lattice 模板。每个正式实验同时保存最终 Plan 的规范化内容与哈希，以及 builder 名称、版本和配置；真正决定前向语义的是最终 Plan，而不是生成器名称。

#### 2.1.2 一个 Token 的波前语义

设节点 \(v\in L_d\) 的固定父节点集合为 \(P(v)\)。当前 Token 上，只有已经激活并执行的父节点才会发送完整 hidden。节点 \(v\) 的实际 inbox 为：

$$
\operatorname{Inbox}_{v,t}
=\left\{
(u,g_{u,t})
\mid u\in P(v),\ u\text{ 在 Token }t\text{ 已激活并 Emit}
\right\}.
$$

GraphBranch 输入 \(h_t\) 作为 entry node 的一条外部消息，因此入口不依赖图内父节点也能 reached；最终 sink 的完整输出就是 \(b_{\mathcal G,j,t}\)。

镜像直通消息可以在较早的 Line 产生，但只保存在目标 inbox 中；目标 Line 到来、所有可能父节点都已经结算后，才把“未到达”和“尚未到达”区分开。令：

$$
q_{v,t}=\mathbf 1[\operatorname{Inbox}_{v,t}\ne\varnothing]
$$

表示节点是否 reached。若 \(q_{v,t}=0\)，节点不参加当前选择，不更新状态，也不输出；若收到一条或多条消息，则先执行一次与消息到达顺序无关的 `ParentAggregate`：

$$
h_{v,t}
=\operatorname{ParentAggregate}_v(\operatorname{Inbox}_{v,t})
=\sum_{(u,g_{u,t})\in\operatorname{Inbox}_{v,t}}
\alpha_{u\to v,t}g_{u,t},
$$

$$
\alpha_{u\to v,t}\ge0,
\qquad
\sum_{(u,g_{u,t})\in\operatorname{Inbox}_{v,t}}\alpha_{u\to v,t}=1.
$$

首个简单设置对实际到达的父消息取均匀平均，并把到达数量或有界 parent-presence mask 作为额外轻量信息。归一化聚合不会因父消息数量变化而重复放大公共 hidden，也能在各节点初始化为 identity 时保持 identity。学习型局部聚合可以后续单独实验。

节点 \(v\) 用自己的入口归一化产生本地消息：

$$
m_{v,t}=N_{R,v}(h_{v,t}),
$$

再按 content-only、pre-update 或 post-update 语义产生轻量 \(\operatorname{Read}^{\mathrm{sel}}\)。同一 region 的节点可能拥有不同的 \(h_{v,t}\)，因此一般 HB-Lattice 不要求存在第 1.3 节那样由共同 parent hidden 产生的单一 \(\mu\)；region selector 只联合处理各 reached nodes 发来的轻量读出和 presence 信息。

对 region \(\mathcal R_{d,r}\subseteq L_d\)，当前候选集合为：

$$
\mathcal C_{d,r,t}
=\{v\in\mathcal R_{d,r}\mid q_{v,t}=1\}.
$$

region selector 可以联合处理本 region 的轻量读出，但只为 reached nodes 产生有效概率：

$$
a_{v,t}
=\left[
\operatorname{Score}_{d,r}
\left((r_{w,t,\tau}^{\mathrm{sel}},q_{w,t})_{w\in\mathcal R_{d,r}}\right)
\right]_v,
\qquad v\in\mathcal C_{d,r,t},
$$

$$
(p_{v,t})_{v\in\mathcal C_{d,r,t}}
=\operatorname{softmax}
\left((a_{v,t})_{v\in\mathcal C_{d,r,t}}\right).
$$

不同 regions 的 selector 参数默认不共享；若实验共享，必须在 Plan 中声明。

若 \(\mathcal C_{d,r,t}=\varnothing\)，该 region 选择空集；否则必须满足：

$$
1\le |\mathcal A_{d,r,t}|
\le \min(K_{d,r}^{\max},|\mathcal C_{d,r,t}|).
$$

全激活只是令所有 reached nodes 都进入 \(\mathcal A_{d,r,t}\) 的特例。入口和最终 sink 可以在 Plan 中声明为 forced-active。

传播 profile 在 HB-Lattice 中统一解释为：

- **N**：无状态，只有 active nodes 执行完整节点计算；
- **SD**：只有 active nodes 提交 Update 并执行完整节点计算；
- **BO**：全部 reached nodes 提交 Update，只有 active nodes 执行较大读出和昂贵计算。

因此 BO + post-update state 的顺序是“全部 reached nodes 聚合并 Update → 各 region 选择 → active nodes 执行”；SD 仍只自然兼容 content-only 和 pre-update state。一个节点即使收到多个父消息，也只在 `ParentAggregate` 后 Update 一次。

active 节点沿用第 1.3 节的完整 block-like 计算，只把共享输入 \(h_t\) 换成本节点入口 \(h_{v,t}\)：

$$
u_{v,t}=h_{v,t}+\rho_{v,t},
$$

$$
g_{v,t}
=u_{v,t}+E_v\!\left(N_{F,v}(u_{v,t})\right).
$$

节点随后把完整 \(g_{v,t}\) 发送到 Plan 声明的全部固定 children。只有当前 Line 的 inbox、选择、状态提交、完整节点计算和 Emit 全部结算后，才开始下一个 Line。实现可以做等价的批处理或流水线，但不得改变这个 reference semantics。

`ParentAggregate` 是目标节点入口处的多父合并；它不要求各父消息具有共同输入。`ActiveBranchAggregate` 则描述共享 parent 的显式 fork-join 或 GraphBranch 与 backbone 的边界，二者不能因为都叫“汇聚”而混用。

HB-Lattice 中 selector 概率怎样进入主任务梯度仍需逐项对齐：可以在 sender 端缩放节点相对入口的变化，也可以把概率作为目标节点 `ParentAggregate` 的权重。首个 Plan 必须明确选择其中一种，或明确只使用 hard active set 与辅助 loss；不能把 H1 的 `AGG` 名称自动解释成某种 HB-Lattice message weight。`AGG-NONE` 只表示没有显式 fork-join，不表示没有 `ParentAggregate`。

#### 2.1.3 两类标准 TopologyBuilder

第一类 builder 使用 \(B\) 叉扩展、逐坐标平台混合和镜像收拢。设扩展深度为 \(D_{\mathrm{up}}\)，平台 hop 数为 \(P\)，最大宽度为 \(W=B^{D_{\mathrm{up}}}\)，Line 宽度为：

$$
1,B,\ldots,B^{D_{\mathrm{up}}},
\underbrace{B^{D_{\mathrm{up}}},\ldots,B^{D_{\mathrm{up}}}}_{P\text{ 个额外 Line}},
B^{D_{\mathrm{up}}-1},\ldots,B,1.
$$

例如 \(B=2,D_{\mathrm{up}}=2,P=2\) 时，可以用二进制空间地址写成：

~~~text
L0:  ε
L1:  0, 1
L2:  00, 01, 10, 11
L3:  00', 01', 10', 11'       # 混合第一位
L4:  00'', 01'', 10'', 11''   # 混合第二位
L5:  0', 1'
L6:  ε'
~~~

扩展时追加一位；第一个平台 hop 令 \((a,b)\to(0,b),(1,b)\)，第二个 hop 令 \((a,b)\to(a,0),(a,1)\)；收拢时删除一位。\(L_0\to L_6\)、\(L_1\to L_5\) 的对应节点可以逐个开启镜像直通。样例可以把 \(L_1\) 和 \(L_5\) 各划成一个二节点 region，把每个宽度为 4 的 Line 划成 \(\{00,01\}\)、\(\{10,11\}\) 两个 regions；入口和 sink 强制激活。相同地址出现在不同 Line 时仍表示不同节点，默认不共享参数或状态。

对称 \(B\) 叉模板中，扩展节点最多连接 \(B\) 个 children 和一个镜像节点，平台节点的入度、出度为 \(B\)，收拢节点最多接收 \(B\) 个深层 parents 和一个镜像 parent；这些上界不随平台宽度或长度增长。

第二类 builder 使用统一空间有向 Graph。设平台坐标集合为 \(Q\)，空间图为 \(G_{\mathrm{space}}=(Q,E_{\mathrm{space}})\)，则每个相邻平台 hop 生成：

$$
E_d
=\left\{
((d,q),(d+1,q'))
\mid(q,q')\in E_{\mathrm{space}}
\right\}.
$$

同一空间图可以在所有平台 hop 重复，也可以由 builder 为不同 hop 产生不同 \(E_d\)。即使 \(G_{\mathrm{space}}\) 自身有环，逐 Line 展开后的 HB-Lattice 仍然无环。

一个空间图可以同时包含固定大小的局部邻域和每节点固定数量的长程 shortcut。长程边宜采用置换或其他入度、出度同时有界的规则，避免宽度增长时形成高入度枢纽。平台 shortcut 只跨一个逻辑 Line、但跨越较远空间坐标；镜像直通则跨越多个逻辑 Line。两类边必须分别标记、记录成本并支持独立消融。

### 2.2 共享 parent 的 fork-join 与 ActiveBranchAggregate

H1 可以是只含一个共享 parent receiver group 的最简单 GraphBranch：

$$
\left(
(\widehat b_{j,t}^{(i)})_{i\in\mathcal A_{j,t}},
p_{j,t},
\mathcal A_{j,t},
S_{j,t}
\right)
=\mathcal R_j(h_t,S_{j,t-1}),
$$

$$
\mathcal G_j(h_t)
=\operatorname{ActiveBranchAggregate}_{\mathrm{MIX}}
\left(
h_t,
\{(\widehat b_{j,t}^{(i)},\beta_{j,t}^{(i)})
\mid i\in\mathcal A_{j,t}\}
\right).
$$

对于 N，省略上式的状态输入和输出。

更深 GraphBranch 也可以包含显式的共享 parent fork-join。设一次分支的共同输入为 \(h\)，当前激活分支集合为 \(\mathcal A\)，每个分支或已经收拢的子结构返回未经当前汇合点加权的完整 hidden：

$$
\widehat b_i=\mathcal B_i(h),
\qquad i\in\mathcal A.
$$

这里的 \(\mathcal B_i\) 可以是一个 receiver 节点，也可以是已经完成内部波前传播与收拢的子结构。这是 HB-Lattice 可以表达的一种局部结构，不代表所有 active nodes 都必须立即汇合。

`ActiveBranchAggregate` 接收共同输入、若干完整分支输出及其系数，返回一个完整 hidden。统一定义为：

$$
\operatorname{ActiveBranchAggregate}
\left(h,\{(\widehat b_i,\beta_i)\}_{i\in\mathcal A}\right)
=h+
\sum_{i\in\mathcal A}
\beta_i(\widehat b_i-h).
$$

内部和 GraphBranch 边界都使用这个接口，但采用不同的合并 policy：

| 合并位置 | Policy | 语义 |
| --- | --- | --- |
| GraphBranch 内部的显式 fork-join | **MIX** | 用一组 \(\beta_i\) 从共享 parent 的 active branches 中选择或混合出一个后继表示 |
| GraphBranch 与 always-on base 的边界 | **RESIDUAL_ADD** | 保留 base 输出，再叠加 GraphBranch 相对共同输入产生的变化 |

内部 **MIX** 的主要候选是：

| MIX policy | Active set | 合并系数 |
| --- | --- | --- |
| **Top-1 Soft-P** | \(\mathcal A=\{c\}\) | \(\beta_c=p_c\) |
| **Top-1 Hard-ST** | \(\mathcal A=\{c\}\) | \(\beta_c=1+p_c-\operatorname{sg}(p_c)\) |
| **Top-K 均匀平均** | \(\lvert\mathcal A\rvert=K\) | \(\beta_i=1/K\) |
| **Top-K router 加权** | \(\lvert\mathcal A\rvert=K\) | \(\displaystyle\beta_i=p_i/\sum_{k\in\mathcal A}p_k\) |
| **学习型局部聚合** | \(\lvert\mathcal A\rvert=K\) | active branches 上归一化的学习权重 |

令 \(\operatorname{sg}(x)\) 表示 stop-gradient：前向仍等于 \(x\)，反向梯度为零。Top-1 Hard-ST 的 \(\beta_c\) 在前向中等于 1，且 \(\partial\beta_c/\partial p_c=1\)；离散 Top-1 选择本身仍不参与反向传播。

学习型局部聚合可以写成：

$$
(\beta_i)_{i\in\mathcal A}
=\operatorname{softmax}\!\left(
\operatorname{MergeScore}
\left(h,\{(\widehat b_i,p_i)\}_{i\in\mathcal A}\right)
\right).
$$

均匀平均适合作为简单对照，归一化 router 加权作为 Top-K 主设置，学习型局部聚合留作后续候选。Top-K router 加权在 \(K=1\) 时会归一化为 1，且对该概率的导数为 0：其前向值与 Top-1 Hard-ST 相同，但反向不同，也不同于 Top-1 Soft-P。

均匀平均的系数不依赖 \(p_i\)，主任务 loss 不会通过合并权重训练 router；归一化 router 加权则可以训练 active branches 之间的相对权重，但离散的 Top-K 成员选择仍不参与反向传播。

router 概率只通过本表中的 \(\beta_i\) 影响当前汇合，不在 receiver branch 内再次缩放。\(\beta_i\) 作用于整个分支变化 \(\widehat b_i-h\)，而不是只作用于分支最后一个 FFN residual。对标准 receiver branch：

$$
\widehat b_i-h
=\rho_i+E_i\!\left(N_{F,i}(h+\rho_i)\right),
$$

所以 Top-1 Soft-P 返回：

$$
h+p_c(\widehat b_c-h)
=h+p_c\left[
\rho_c+E_c\!\left(N_{F,c}(h+\rho_c)\right)
\right].
$$

N 中 \(\rho_c=0\)；Top-1 Hard-ST 的前向则完整保留被选 receiver branch。

若直接求和完整分支输出，则：

$$
\sum_{i\in\mathcal A}\widehat b_i
=K h+\sum_{i\in\mathcal A}(\widehat b_i-h),
$$

公共输入会被加入 \(K\) 次。`ActiveBranchAggregate` 只在汇合点内部计算 \(\widehat b_i-h\)；GraphBranch 内部边始终传递完整的 \(\widehat b_i\)。

在 GraphBranch 与 backbone 的边界，令 \(b_0\) 表示 placement 的 always-on 输出，\(b_{\mathcal G}=\mathcal G(h)\) 表示 GraphBranch 的完整输出，则 **RESIDUAL_ADD** 为：

$$
\operatorname{ActiveBranchAggregate}_{\mathrm{RESIDUAL\_ADD}}
(h;b_0,b_{\mathcal G})
=\operatorname{ActiveBranchAggregate}
\left(h,\{(b_0,1),(b_{\mathcal G},1)\}\right)
=b_0+(b_{\mathcal G}-h).
$$

这正是后面四种 placement 的 merge；POST 中 \(b_0=h=v\)，结果直接退化为 \(\mathcal G(v)\)。显式 fork-join 使用 **MIX**，HB-Lattice 多父入口使用第 2.1 节的 `ParentAggregate`；**RESIDUAL_ADD** 只用于 GraphBranch 与 backbone 的边界。

### 2.3 POST：完整 block 后串联 GraphBranch

先完整执行原 block：

$$
u_{\ell,t}
=x_{\ell,t}
+\left[A_\ell\!\left(N_A(X_{\ell,\le t})\right)\right]_t,
$$

$$
v_{\ell,t}
=u_{\ell,t}+F_\ell\!\left(N_F(u_{\ell,t})\right).
$$

GraphBranch 再读取完整 block 输出：

$$
b_{\mathcal G,j,t}
=\mathcal G_j(v_{\ell,t}),
$$

$$
y_{\ell,t}
=v_{\ell,t}+\Delta_{\mathcal G,j,t}(v_{\ell,t})
=b_{\mathcal G,j,t}.
$$

~~~text
x → Attention → u → 原 dense MLP → v → TIDE GraphBranch → y
~~~

POST 与 base block 串行执行。GraphBranch 能看到当前 block 的 Attention 和原 MLP 结果；其内部可以包含一层或多层 receiver 节点。

### 2.4 PARBLK：与完整 block 并列

对位置 \(t\)，GraphBranch 与完整 base block 都从当前层输入开始，只在 block 末尾合并；base Attention 仍按第 1.2 节读取因果前缀 \(X_{\ell,\le t}\)：

$$
u_{\ell,t}
=x_{\ell,t}
+\left[A_\ell\!\left(N_A(X_{\ell,\le t})\right)\right]_t,
$$

$$
v_{\ell,t}
=u_{\ell,t}+F_\ell\!\left(N_F(u_{\ell,t})\right),
$$

$$
y_{\ell,t}
=v_{\ell,t}
+\Delta_{\mathcal G,j,t}(x_{\ell,t}).
$$

~~~text
          ┌→ 完整 base block → v ─────┐
x ────────┤                            + → y
          └→ GraphBranch → Δ_G(x) ───┘
~~~

GraphBranch 看不到当前 block 的 Attention 或 MLP 结果，也不改变它们的输入。PARBLK 可以让 base block 与 GraphBranch 并行执行。

### 2.5 PARATTN：与 Attention 并列

GraphBranch 读取 \(x_{\ell,t}\)，在 Attention residual 位置返回唯一 residual；原 dense MLP 再读取合并后的表示：

$$
u'_{\ell,t}
=u_{\ell,t}
+\Delta_{\mathcal G,j,t}(x_{\ell,t}),
$$

$$
y_{\ell,t}
=u'_{\ell,t}+F_\ell\!\left(N_F(u'_{\ell,t})\right).
$$

~~~text
          ┌→ self-attention ─┐
x ────────┤                   + → u' → 原 dense MLP → y
          └→ GraphBranch → Δ_G(x) ─┘
~~~

PARATTN 只说明 GraphBranch residual 在 Attention merge 位置接入，不限制 GraphBranch 内部只能包含 Attention。

### 2.6 PARMLP：与 MLP 并列

当前 Attention 先完成 residual merge；原 dense MLP 和 GraphBranch 都读取 \(u\)，最后在 MLP residual 位置合并：

$$
u_{\ell,t}
=x_{\ell,t}
+\left[A_\ell\!\left(N_A(X_{\ell,\le t})\right)\right]_t,
$$

$$
y_{\ell,t}
=v_{\ell,t}
+\Delta_{\mathcal G,j,t}(u_{\ell,t}).
$$

~~~text
x → self-attention → u
                      ├→ 原 dense MLP ─┐
                      └→ GraphBranch ──── + → y
~~~

GraphBranch 能看到当前 Attention 的结果，但看不到当前原 MLP 的结果，也不改变原 MLP 的输入。本文统一使用名称 **PARMLP**；**PARFFN** 指同一个 placement。

原 dense MLP 可以理解为 always-on shared branch，GraphBranch 是与它并列的稀疏有状态主旁路。两条分支的未归一化输入必须是同一个 \(u\)，才能称为 PARMLP。

### 2.7 四种位置的直接比较

| Placement | GraphBranch 读取 | 看见当前 Attention 结果 | 看见当前原 MLP 结果 | 改变当前原 MLP 输入 | 逻辑关系 |
| --- | --- | ---: | ---: | ---: | --- |
| **POST** | \(v\) | 是 | 是 | 否 | 完整 block 后串联 |
| **PARBLK** | \(x\) | 否 | 否 | 否 | 与完整 block 并列 |
| **PARATTN** | \(x\) | 否 | 否 | 是 | 与 Attention 并列 |
| **PARMLP** | \(u\) | 是 | 否 | 否 | 与 MLP 并列 |

四种 placement 共享同一个 GraphBranch 契约，只改变输入和 residual 返回位置。GraphBranch 的每个内部 branch 都可初始化为 identity；对标准 receiver branch，这要求 \(\rho_i=0\) 且 FFN residual 为零，于是 \(\widehat b_i=h\)。此时任意内部 MIX 都返回 \(h\)，因此 \(\mathcal G_j(h)=h\) 且 \(\Delta_{\mathcal G,j}=0\)，base 模型函数保持不变。离开初始点后，四种 placement 的前向耦合、梯度路径和有效深度不同，不能视为同一架构。

语义上保留四种 placement；实现可以先从 POST 开始，其他 placement 只需增加较薄的外层连接。

## 3. Dense 与标准 MoE 基线

### 3.1 DENSE

DENSE 使用原 block：

$$
y=v
=u+F_\ell(N_F(u)).
$$

### 3.2 MOE

M8 是每个 site 有 8 个 experts 的简写；正式名称分别用 **MOE-R8** 和 **I** 表示 expert 数与 site 数。它把每个 expert 初始化为原 dense MLP 的副本，每个有效 Token 只执行一个 expert，不设 capacity、不丢 Token、也不 reroute。

对插在 block \(\ell=\ell(j)\) 的 MoE site \(j\)，计算为：

$$
m_{j,t}=N_F(u_{\ell,t}).
$$

$$
a_{j,t}=W_{\mathrm{moe}}m_{j,t},
\qquad
p_{j,t}=\operatorname{softmax}(a_{j,t}),
$$

$$
c_{j,t}=\arg\max_i p_{j,t}^{(i)}.
$$

$$
z_{j,t}^{(c_{j,t})}=m_{j,t}.
$$

$$
y_{\ell,t}
=u_{\ell,t}
+E_{j,c_{j,t}}\!\left(z_{j,t}^{(c_{j,t})}\right).
$$

这里的 \(m,a,p,c,z,E\) 与 receiver group 使用同一组角色符号。M8 直接合并被选 expert 的输出，不乘 soft 概率；它没有 receiver 私有状态。

它与 PARMLP 处于相同的 block 接口，但语义不同：

- MOE 用一个 routed expert 替换原 dense MLP；
- PARMLP 保留原 dense MLP，再增加一个并列 GraphBranch residual。

当 PARMLP 的 GraphBranch 只经过一层 receiver group 时，也可以直观地看作一种 shared-expert MoE：原 dense MLP 是 always-on shared expert，receivers 是 routed experts。

## 4. 实际训练时的损失函数

令 \(\mathcal T\) 表示一个 micro-batch 中所有有效目标 Token 的 \((b,t)\) 集合，\(N_T=|\mathcal T|\)。自回归语言模型损失为：

$$
\mathcal L_{\mathrm{LM}}
=-\frac{1}{N_T}
\sum_{(b,t)\in\mathcal T}
\log P_\theta(w_{b,t}\mid w_{b,<t}).
$$

路由辅助项使用的 Token 集合略有不同。当前 H1 中，每个 site 的 router 都处理同一个集合 \(\mathcal V\)：attention mask 标记为有效、实际经过 router 的全部 \((b,t)\) 位置，\(N_V=|\mathcal V|\)。\(\mathcal V\) 与 receiver 或 expert \(i\) 无关，不是候选 \(i\) 实际被选中的 Token 集。balance loss 不要求单个 Token 均匀选择所有候选，而是避免整个 micro-batch 长期集中到少数 receivers 或 experts。令 \(\mathcal I\) 表示所有 routed sites，\(I=|\mathcal I|\)。

每个 site 独立计算 balance loss，再在 sites 间等权平均。统计范围是当前 micro-batch；梯度累积只累积各 micro-batch 的梯度，不预先把多个 micro-batches 合并成 global-batch balance loss。

### 4.1 H1 中 N、SD、BO 的 receiver balance loss

对 site \(j\) 的 \(R\) 个 receivers，平均 softmax 概率为：

$$
\bar p_{j,i}
=\frac{1}{N_V}
\sum_{(b,t)\in\mathcal V}p_{j,b,t}^{(i)}.
$$

当前 H1 的 N、SD、BO 共同使用：

$$
\mathcal L_{\mathrm{bal}}^{\mathrm{receiver}}
=\frac{1}{I}
\sum_{j\in\mathcal I}
\sum_{i=0}^{R-1}
\left(\bar p_{j,i}-\frac1R\right)^2.
$$

它约束的是平均 soft 概率，不直接约束 \(\operatorname{TopKIndex}\) 后各 receiver 真正执行了多少次；Top-1 时，后者就是 \(\arg\max\) 的选择次数。因此它鼓励均衡，但不能严格保证 hard active counts 均衡。

N、SD、BO 的实际反向传播目标都是：

$$
\boxed{
\mathcal L_{\mathrm{train}}^{\mathrm{N/SD/BO}}
=\mathcal L_{\mathrm{LM}}
+\omega_{\mathrm{receiver}}\mathcal L_{\mathrm{bal}}^{\mathrm{receiver}}
}.
$$

\(\omega_{\mathrm{receiver}}\ge0\) 由实验设置记录；本文定义的 H1 receiver 目标不含 router z-loss。

### 4.2 HB-Lattice 的 region balance loss

HB-Lattice 中每个 region 只处理实际 reached 的节点。对 site \(j\)、Line \(d\)、region \(r\)，令：

$$
\mathcal V_{j,d,r}
=\left\{
(b,t)\mid\mathcal C_{j,d,r,b,t}\ne\varnothing
\right\}
$$

表示该 region 在当前 micro-batch 中真正发生选择的 Token 事件。selector 只在 \(\mathcal C_{j,d,r,b,t}\) 内做 masked softmax；未 reached 节点不进入当前候选集合，也不能被 balance loss 当作本次本应选择的候选。

这里的 \(\mathcal C_{j,d,r,b,t}\) 就是第 2.1 节的 \(\mathcal C_{d,r,t}\) 补回 site \(j\) 和序列 \(b\) 下标。

同一 region 的不同节点可能拥有不同的 reached 机会，因此 H1 中固定以 \(1/R\) 为目标的公式不能直接照搬。HB-Lattice 最终采用按当前 availability 条件化的目标，还是按一段时间内各节点的可达机会归一化，仍需结合具体 topology 逐项对齐。在此之前，任何 HB-Lattice 实验都必须记录 region 的 reached mask、实际选择次数和所用辅助 loss 公式，不能只写“沿用 H1 balance loss”。

### 4.3 标准 MoE（M8）的 balance loss 与 router z-loss

M8 使用不同的 Switch-style balance loss。令 \(p_{j,b,t}^{(i)}\) 为 MoE router 的 softmax 概率，\(c_{j,b,t}\) 为硬 Top-1 expert，定义：

$$
\bar p_{j,i}
=\frac{1}{N_V}
\sum_{(b,t)\in\mathcal V}p_{j,b,t}^{(i)},
\qquad
f_{j,i}
=\frac{1}{N_V}
\sum_{(b,t)\in\mathcal V}
\mathbf 1[c_{j,b,t}=i].
$$

其中 \(f_{j,i}\) 是 expert \(i\) 真正收到的 Token 比例。M8 使用：

$$
\mathcal L_{\mathrm{bal}}^{\mathrm{MoE}}
=\frac{1}{I}
\sum_{j\in\mathcal I}
R\sum_{i=0}^{R-1}
\operatorname{sg}(f_{j,i})\,\bar p_{j,i}.
$$

其中 \(\operatorname{sg}\) 表示 stop-gradient（停止梯度）：前向值不变，反向梯度为零。\(f_{j,i}\) 来自不可导的硬路由，梯度只通过 \(\bar p_{j,i}\) 返回 router。完全均衡时，\(\mathcal L_{\mathrm{bal}}^{\mathrm{MoE}}=1\)，而 receiver balance loss 完全均衡时等于 0，所以两种 `balance_loss` 的原始数值不能直接比较。

沿用第 3.2 节，MoE router 收到的消息、expert 输入和 router logits 为：

$$
m_{j,t}=N_F(u_{\ell(j),t}),
\qquad
z_{j,t}^{(c_{j,t})}=m_{j,t},
\qquad
a_{j,t}=W_{\mathrm{moe}}m_{j,t}.
$$

M8 还使用 router z-loss，限制 logits 的整体尺度：

$$
\mathcal L_z
=\frac{1}{I N_V}
\sum_{j\in\mathcal I}
\sum_{(b,t)\in\mathcal V}
\left[
\log\sum_{i=0}^{R-1}\exp(a_{j,b,t}^{(i)})
\right]^2.
$$

因此 M8 的实际反向传播目标是：

$$
\boxed{
\mathcal L_{\mathrm{train}}^{\mathrm{M8}}
=\mathcal L_{\mathrm{LM}}
+\omega_{\mathrm{MoE}}\mathcal L_{\mathrm{bal}}^{\mathrm{MoE}}
+\omega_z\mathcal L_z
}.
$$

\(\omega_{\mathrm{MoE}},\omega_z\ge0\) 由实验设置记录。

> **备注：**M8 采用的是成熟、可靠且便于对照的经典 MoE 基线，但不是所有先进 MoE 统一采用的唯一方案。

| 机制或路线 | 当前定位 | 代表性采用情况 |
| --- | --- | --- |
| **Switch-style balance loss** | 常见的标准基线，但不是唯一推荐路线 | [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)、[OLMoE](https://huggingface.co/allenai/OLMoE-1B-7B-0924) 使用；[Qwen3](https://huggingface.co/Qwen/Qwen3-235B-A22B) 使用 global-batch 变体 |
| **Router z-loss** | 常用的可选稳定项，但采用并不统一 | [ST-MoE](https://arxiv.org/abs/2202.08906) 推荐，OLMoE 使用 |
| **其他负载均衡路线** | 用动态 bias、分位数校准或系统级 dispatch 替代或补充经典辅助损失 | [DeepSeek-V3/R1](https://arxiv.org/abs/2412.19437)：动态 expert bias；[Kimi K3](https://github.com/MoonshotAI/Kimi-K3)：Quantile Balancing；[GLM-5.2](https://huggingface.co/zai-org/GLM-5.2)/5.3：`noaux_tc`，5.3 沿用 5.2 base；[MiniMax-Text-01](https://huggingface.co/MiniMaxAI/MiniMax-Text-01)：GShard-style auxiliary loss + global token dispatch |

这里的动态 expert bias 和 Quantile Balancing 都是训练期均衡；Kimi K3 的最终 bias 在推理时冻结，不等于第 4.4 节的推理期负载感知 selector。

DENSE 没有 router，实际目标只有 \(\mathcal L_{\mathrm{LM}}\)。训练日志中的 `loss` 是包含上述辅助项的总损失，`lm_loss` 只表示 Token 预测损失；跨架构比较模型质量时应使用验证集 `lm_loss` 或 perplexity，而不是直接比较总 `loss` 或两种定义不同的 `balance_loss`。

### 4.4 训练期均衡与推理期负载感知

| 机制 | 训练时 | 推理时 | 作用 |
| --- | --- | --- | --- |
| **训练期 balance loss** | 加入训练目标 | 不再计算 | 让模型学出较均衡的路由倾向，但不保证推理时始终均衡 |
| **负载感知 selector** | 作为前向规则参与训练 | 继续使用 | 根据当前序列的路由历史动态调整后续选择 |

下面给出一个最简单的样例，实际实现可根据训练和推理情况调整。

每个 receiver 可以把近期激活负载作为一个历史标量发给 selector。令 \(\operatorname{load}_{j,b,t}^{(i)}\) 表示这一标量：

$$
a_{j,b,t}^{(i)}
=\left[\operatorname{Score}(\cdots)\right]_i
-\kappa_{\mathrm{load}}\,\operatorname{load}_{j,b,t-1}^{(i)},
$$

完成选择后更新：

$$
\operatorname{load}_{j,b,t}^{(i)}
=\lambda_{\mathrm{load}}\,\operatorname{load}_{j,b,t-1}^{(i)}
+(1-\lambda_{\mathrm{load}})\mathbf 1[i\in\mathcal A_{j,b,t}].
$$

其中 \(\kappa_{\mathrm{load}}\ge0\)，\(0\le\lambda_{\mathrm{load}}<1\)。

同一前向规则可以同时用于训练和推理。训练期 balance loss 只留下学到的均衡倾向；动态负载路由则形成更强的闭环反馈，但也会引入跨 Token 递归，并可能造成路由振荡或增加训练难度。

## 5. 规范命名

### 5.1 科学条件名

共享 parent 的 H1 候选继续采用：

~~~text
<TRAIN>-<PLACEMENT>-<PROFILE>-R<WIDTH>-I<SITES>-H<DEPTH>-<STATE>-<SELECTOR>-<AGG>
~~~

例如：

~~~text
CPT-PARMLP-BO-R8-I4-H1-EMA128-SEL-POST-AGG-T1-HST
PT-POST-SD-R8-I4-H1-EMA128-SEL-PRE-AGG-T1-SOFTP
~~~

非平凡 HB-Lattice 在 **H** 后追加 `T<TOPO_ID>`，例如：

~~~text
PT-POST-BO-R2-I4-H7-THBL2D2P2CMIR-GDN-K32-V32-SEL-POST-AGG-NONE
~~~

`TOPO_ID` 是已展开 `HBLatticePlan` 的可读索引，不代替 manifest 中的完整 Plan 与规范化哈希。

字段定义如下：

| 字段 | 允许值或形式 | 含义 |
| --- | --- | --- |
| TRAIN | PT / CPT / FT / SFT | 初始化与训练阶段 |
| PLACEMENT | POST / PARBLK / PARATTN / PARMLP | GraphBranch 的输入与 residual 返回位置 |
| PROFILE | N / SD / BO | 状态接收与稀疏计算语义 |
| R | R4、R8、R16、RVAR 等 | H1 group 或 HB-Lattice region 的候选数；不统一时用 RVAR |
| I | I1、I4、I8 等 | 一个 Token 顺序经过的插入位置数 |
| H | H1、H2 等 | 每个插入位置内部从入口到出口的最大 receiver-node 深度 |
| T | T\<TOPO_ID\> | 非平凡 HB-Lattice 的已展开 topology 索引；H1 可以省略 |
| STATE | NONE、EMA128、GDN-K32-V32、ATTN-FULL、ATTN-W128、ATTN-COMP 等 | 状态结构和必要尺寸 |
| SELECTOR | SEL-CONTENT / SEL-PRE / SEL-POST | 第 1.3.1 节定义的 selector 输入时序 |
| AGG | AGG-NONE / AGG-T1-SOFTP / AGG-T1-HST / AGG-K2-MEAN / AGG-K2-ROUTER / AGG-K2-LEARNED / AGG-VAR | 第 2.2 节定义的显式 fork-join MIX policy；没有显式 fork-join 时用 NONE |

**SEL-CONTENT**、**SEL-PRE** 和 **SEL-POST** 分别表示 \(\operatorname{Read}^{\mathrm{sel}}\) 只读取 receiver 当前本地消息、额外读取旧状态或额外读取更新后状态。H1 的 \(\operatorname{Score}\) 还读取由共享 parent 产生的公共 \(\mu\)；HB-Lattice region 不要求存在单一公共 \(\mu\)，只联合处理 reached nodes 的轻量读出和 presence 信息。名称不限定打分采用线性层、MLP 或其他实现；精确读出、打分公式以及状态中是否包含历史激活记录仍由 manifest 和实验设置保存。

如果历史激活记录会影响 selector 或输出，它就是模型前向语义的一部分，不能隐藏在同一个纯 EMA/GDN 条件名下。具体实现确定后，应在 **STATE** 中增加明确的复合状态标签；记录维度、衰减、写回规则等细节再放入 manifest。

**AGG** 只表示显式 fork-join 的 MIX；HB-Lattice 节点入口的 `ParentAggregate` 由 topology Plan 单独记录，GraphBranch 与 backbone 的 RESIDUAL_ADD 已由 placement 固定。如果不同 regions、sites 或 Line 使用不同 active 数量或 policy，则写 **AGG-VAR**，并在 manifest 中列出完整设置。

TRAIN 的含义必须严格区分：

- **PT**：随机初始化后做自回归预训练；
- **CPT**：加载预训练 checkpoint，继续做语言模型目标训练；
- **FT**：加载预训练 checkpoint，使用不同于基础自回归预训练的下游任务目标；
- **SFT**：FT 中特指有监督的指令或输入输出微调。

TRAIN 描述 base 权重与训练目标；新分支的初始化方式由实验设置单独记录。

口语中的“finetune”不能直接写入正式名称：如果实际仍是 FineWeb 或领域语料上的自回归语言模型训练，应记为 CPT；只有训练目标确实改变时才记为 FT 或 SFT。

### 5.2 R、I、H 与 K 不得混用

- **R8** 只表示每个局部 group 或 region 有 8 个候选，不表示模型共有 8 个 receivers。
- **I8** 表示每个 Token 顺序经过 8 个插入位置，不表示 Transformer 只有 8 个 blocks。
- **H2** 表示一个插入位置内部的最大 receiver-node 深度为 2，不表示模型中有两个插入位置，也不能唯一确定拓扑。
- **AGG-K2** 表示对应局部 group 激活两个候选，不表示该 group 只有两个候选。

receiver branch 内部串行的状态/上下文 residual 与 FFN residual 合计仍算一层；只有完整分支输出继续进入下一个 receiver group 时，H 才增加。

例如 **R4-I8-H1** 表示 8 个顺序插入位置，每处只有一层局部 receiver group，每个 group 有 4 个候选。它不是 8 层递归。

如果不同插入位置、Line 或 region 采用不同宽度，短名字中使用 **RVAR**，并在 manifest 和报告中列出完整宽度；不得用一个看似统一的 R 值掩盖异构拓扑。H>1、平台期、多父边或镜像直通不能只靠 R/H 推断，必须同时给出 **TOPO_ID** 和完整 Plan。

### 5.3 具体 run 实例名

科学条件之外，真实 run 还需要模型、seed 和尝试编号：

~~~text
<MODEL>-<scientific-condition>-s<SEED>-r<ATTEMPT>
~~~

例如：

~~~text
q3-06b-cpt-parmlp-bo-r8-i4-h1-ema128-sel-post-agg-t1-hst-s42-r1
~~~

模型 checkpoint、数据 revision、精确 block 编号、Token 预算、学习率、dtype、设备和代码 commit 仍由 manifest 保存，不强行塞进短名字。名称是可读索引，不代替完整实验设置。

### 5.4 基线名称

Dense 与标准 MoE 不使用 TIDE placement/profile 字段：

~~~text
PT-DENSE
CPT-DENSE
PT-MOE-R8-I4
CPT-MOE-R8-I4
~~~

MOE 的精确插入 block、Top-K、capacity、token-drop、shared expert 和路由辅助项必须在完整设置中声明。

## 6. 名称之外仍必须明确的语义

即使规范名称相同，每个正式设置仍要明确记录：

- 精确插入 block 编号；
- 不同 sites、Lines 和节点之间是否共享参数；默认不共享；
- 完整 `HBLatticePlan`、规范化哈希，以及 TopologyBuilder 的名称、版本和配置；
- 每条 Line 的 phase、节点与 region 划分，每条边的端点和 tree/local/shortcut/mirror 类别，以及逐节点镜像直通开关；
- 最大 fan-in/fan-out、region 大小、逐 region 的 \(K^{\max}\) 与 forced-active 节点；
- `ParentAggregate`、parent-presence、selector 概率进入 Emit 或父消息聚合的精确公式；
- 显式 fork-join 的 `ActiveBranchAggregate` policy 与权重；
- \(N_{\mathrm{sel}}\)、\(N_{R,i}\) 与 \(N_{F,i}\) 的精确实现和初始化；
- \(\operatorname{Read}^{\mathrm{sel}}\)、\(\operatorname{Read}^{\mathrm{ffn}}\) 与 \(\operatorname{Score}\) 的精确公式、输出维度以及是否包含历史激活记录；
- Observe、selector、状态提交和历史激活写回的精确顺序；
- GraphBranch 与 backbone 的 RESIDUAL_ADD 公式以及任何额外缩放；
- 各辅助 loss 的系数；
- 状态初始化、有效 Token mask、跨 chunk 的 carry/reset 与梯度 detach 规则；
- 辅助 loss 的 Token 范围、site/Line/region 聚合范围、reached mask 处理以及是否跨 micro-batch 或设备统计；
- 每个 Token 实际执行多少次本地入口归一化、轻量 selector 读出、Observe / Update、较大状态读出和昂贵 FFN；
- 初始化怎样保持或改变 base 函数；
- MOE 是否有 expert capacity、token drop 或 reroute。

这些项目不会全部进入短名字，但它们决定两个 run 是否真的是 matched comparison、到底哪些地方是 matched。
