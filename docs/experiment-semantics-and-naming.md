# TIDE 实验语义、命名与数学符号

> 状态：新实验的规范性文档。
>
> 本文只定义“模型实际怎样计算”和“实验名称怎样反映计算图”。实验晋级、结果报告组织和 checkpoint 保留策略另行讨论。
>
> HB-Lattice 中标为“首个设置”或“推荐”的部分是待逐项核验的候选默认值，核验台帐见 [`experiment-semantics-review-ledger.md`](experiment-semantics-review-ledger.md)。

本文按计算发生的顺序组织：第 1 节先定义 base block 以及 GraphBranch 从哪里接入、向哪里返回；第 2 节用仅分岔一层的 H1 介绍 selector、receiver state、昂贵计算和分支聚合；第 3 节再把这些组件推广到多父、逐 Line 执行的 HB-Lattice；第 4、5 节给出基线与训练目标；第 6、7 节把完整语义映射到名称和实验记录。

## 1. Base block 与 GraphBranch 顶层边界

### 1.1 Base 与顶层接口符号

本节只引入理解 base block 和 GraphBranch 接入位置所需的符号；H1 和 HB-Lattice 的内部符号分别在第 2、3 节首次使用时定义。

| 符号 | 含义 |
| --- | --- |
| \(b\) | micro-batch 中的序列编号；正文通常省略这一维 |
| \(\ell\) | base Transformer block 编号 |
| \(j\) | GraphBranch 插入位置（site）编号；site \(j\) 所在的 base block 记为 \(\ell(j)\) |
| \(t\) | 序列中的 Token 位置 |
| \(x_{\ell,t}\) | 实际送入第 \(\ell\) 个 base block 的 hidden |
| \(u_{\ell,t}\) | 当前 block 完成 Attention residual merge 后的 hidden |
| \(v_{\ell,t}\) | 当前 block 完成原 dense MLP residual merge 后的 hidden |
| \(y_{\ell,t}\) | 当前 block 连同可选 GraphBranch 最终送往下一个 block 的 hidden |
| \(h^{\mathrm{in}}_{j,t}\) | site \(j\) 的 GraphBranch 实际入口 hidden |
| \(b_{\mathcal G,j,t}\) | GraphBranch 返回的完整 hidden |
| \(\Delta_{\mathcal G,j,t}\) | GraphBranch 相对入口产生的 residual |

后文始终按“作用”复用符号：各类归一化写成 \(N\)，私有状态写成 \(s/S\)，状态操作写成 \(\operatorname{Update}\)、\(\operatorname{Read}^{\mathrm{sel}}\) 和 \(\operatorname{Read}^{\mathrm{ffn}}\)。相同基本符号表示组件承担相同作用，不表示共享参数或采用相同算法。

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

### 1.3 GraphBranch 的单入口、单出口契约

每个 site 在原有 base computation 之外只接入一个 GraphBranch，记为 \(\mathcal G_j\)。**GraphBranch** 是整个单入口、单出口模块的专名；后文小写的 branch 只表示它内部一次 fork-join 的候选计算路径。对当前 Token，GraphBranch 从 placement 指定的位置接收一个完整 hidden：

$$
h^{\mathrm{in}}_{j,t}
=
\begin{cases}
v_{\ell,t}, & \text{POST},\\
x_{\ell,t}, & \text{PARBLK},\\
x_{\ell,t}, & \text{PARATTN},\\
u_{\ell,t}, & \text{PARMLP},
\end{cases}
\qquad \ell=\ell(j).
$$

GraphBranch 内部可以是一层 H1，也可以是多层 HB-Lattice；它对外始终只返回一个同维 hidden：

$$
b_{\mathcal G,j,t}
=\mathcal G_j\!\left(h^{\mathrm{in}}_{j,t}\right),
\qquad
\Delta_{\mathcal G,j,t}
=b_{\mathcal G,j,t}-h^{\mathrm{in}}_{j,t}.
$$

这里的 \(\mathcal G_j\) 省略了逐序列持久状态；第 2 节会展开状态怎样读取和更新。无论内部多复杂，placement 只看见入口 \(h^{\mathrm{in}}\)、完整输出 \(b_{\mathcal G}\) 和唯一 residual \(\Delta_{\mathcal G}\)。

若 placement 的 always-on 输出记为 \(b^0_{j,t}\)，GraphBranch 与 base 的边界统一使用 **RESIDUAL_ADD**：

$$
\operatorname{BoundaryMerge}
\left(h^{\mathrm{in}}_{j,t};b^0_{j,t},b_{\mathcal G,j,t}\right)
=b^0_{j,t}+\left(b_{\mathcal G,j,t}-h^{\mathrm{in}}_{j,t}\right)
=b^0_{j,t}+\Delta_{\mathcal G,j,t}.
$$

它保留 always-on 路径，只叠加 GraphBranch 相对共同入口产生的变化。第 2.5 节会说明它与内部 `ActiveBranchAggregate` 的关系。

### 1.4 四种 placement

四种 placement 只改变 \(h^{\mathrm{in}}\)、always-on 输出和 residual 合入位置，不改变 GraphBranch 的内部接口。

#### 1.4.1 POST：完整 block 后串联

先按第 1.2 节得到完整 base block 输出 \(v_{\ell,t}\)，再令：

$$
h^{\mathrm{in}}_{j,t}=v_{\ell,t},
\qquad
y_{\ell,t}
=v_{\ell,t}+\Delta_{\mathcal G,j,t}
=b_{\mathcal G,j,t}.
$$

~~~text
x → Attention → u → 原 dense MLP → v → GraphBranch → y
~~~

GraphBranch 能看到当前 block 的 Attention 和原 MLP 结果。POST 是串联结构，也是最直接的首个实现位置。

#### 1.4.2 PARBLK：与完整 block 并列

GraphBranch 和完整 base block 都从 \(x_{\ell,t}\) 开始，最后在 block 出口合并：

$$
h^{\mathrm{in}}_{j,t}=x_{\ell,t},
\qquad
y_{\ell,t}
=v_{\ell,t}+\Delta_{\mathcal G,j,t}.
$$

~~~text
          ┌→ 完整 base block → v ─────┐
x ────────┤                            + → y
          └→ GraphBranch → Δ_G(x) ────┘
~~~

GraphBranch 看不到当前 block 的 Attention 或 MLP 结果，也不改变它们的输入；两条路径可以并行执行。

#### 1.4.3 PARATTN：与 Attention 并列

GraphBranch 与 Attention 都读取 \(x_{\ell,t}\)。先在 Attention residual 位置合并，再让原 dense MLP 读取合并后的表示：

$$
h^{\mathrm{in}}_{j,t}=x_{\ell,t},
\qquad
u'_{\ell,t}=u_{\ell,t}+\Delta_{\mathcal G,j,t},
$$

$$
y_{\ell,t}
=u'_{\ell,t}+F_\ell\!\left(N_F(u'_{\ell,t})\right).
$$

~~~text
          ┌→ self-attention ─┐
x ────────┤                   + → u' → 原 dense MLP → y
          └→ GraphBranch ────┘
~~~

PARATTN 只说明 GraphBranch residual 的接入位置，不限制 GraphBranch 内部只能使用 Attention。

#### 1.4.4 PARMLP：与 MLP 并列

Attention residual 先得到 \(u_{\ell,t}\)；原 dense MLP 与 GraphBranch 都读取这个共同输入，最后在 MLP residual 位置合并：

$$
h^{\mathrm{in}}_{j,t}=u_{\ell,t},
\qquad
y_{\ell,t}
=v_{\ell,t}+\Delta_{\mathcal G,j,t}.
$$

~~~text
x → self-attention → u
                      ├→ 原 dense MLP ─┐
                      └→ GraphBranch ── + → y
~~~

GraphBranch 能看到当前 Attention 的结果，但看不到当前原 MLP 的结果，也不改变原 MLP 的输入。本文统一使用 **PARMLP**；**PARFFN** 指同一个 placement。原 dense MLP 是 always-on 路径，GraphBranch 是与它并列的稀疏有状态主旁路。

#### 1.4.5 直接比较与初始化

| Placement | \(h^{\mathrm{in}}\) | always-on 输出 | 看见当前 Attention | 看见当前原 MLP | 改变原 MLP 输入 | GraphBranch merge 后 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| **POST** | \(v\) | \(v\) | 是 | 是 | 否 | 直接得到 \(y\) |
| **PARBLK** | \(x\) | \(v\) | 否 | 否 | 否 | 直接得到 \(y\) |
| **PARATTN** | \(x\) | \(u\) | 否 | 否 | 是 | 得到 \(u'\)，再执行原 MLP |
| **PARMLP** | \(u\) | \(v\) | 是 | 否 | 否 | 直接得到 \(y\) |

若 GraphBranch 初始化为 identity，则 \(b_{\mathcal G}=h^{\mathrm{in}}\)、\(\Delta_{\mathcal G}=0\)，四种 placement 都保持原 base 函数不变。离开初始点后，它们的前向耦合、梯度路径和有效深度不同，不能视为同一架构。语义上保留四种 placement；实现可以先从 POST 开始。

## 2. H1：最简单的 GraphBranch

### 2.1 概念与总体数据流

H1 GraphBranch 内部只有一个 **receiver group**：site \(j\) 的 \(R\) 个 receiver nodes 全部接收第 1.3 节定义的同一个入口 \(h^{\mathrm{in}}_{j,t}\)，selector 选择其中少数节点执行完整计算，节点输出随后立即汇合为 \(b_{\mathcal G,j,t}\)。这就是后文“共享 parent”的确切含义。

本文区分以下概念：

| 概念 | 含义 |
| --- | --- |
| **GraphBranch** | 接在一个 site 上的完整单入口、单出口模块；内部可以是 H1，也可以是多层 HB-Lattice |
| **receiver node** | 持久的计算与拓扑单元，拥有自己的参数和可选私有状态；无歧义时简称 receiver |
| **receiver group** | H1 中“一个 selector + \(R\) 个共享入口的 receiver nodes + 一个汇合点”的局部结构 |
| **branch** | 从一次 fork 到对应 join 的候选计算路径，不是额外的参数或状态拥有者 |
| **path / active subgraph** | HB-Lattice 中一个 Token 实际经过的节点和边；存在 fan-in/fan-out 时，整体通常是子图而不是一条 branch |

在 H1 中，candidate \(i\)、receiver node \(i\) 和“经过该节点的 branch \(i\)”一一对应，因此口语中容易混用；正文用 **receiver node** 表示被选择、更新或执行的持久单元，只在讨论 fork-join 及其输出时使用 **branch**。进入 HB-Lattice 后，一个节点可以有多个 parents 和 children，二者不再能互换。

一个标准有状态 receiver node 包含 receiver-local 入口归一化、私有记忆模块及其 `Update/Read` 接口，以及被激活后执行的 Pre-Norm 昂贵 FFN；selector 和汇合操作属于 group 或 region，不在 receiver node 内部。无状态 N 保留相同节点接口，但省略私有记忆，令状态读出 residual 为零。第 2.4 节给出完整公式。

~~~text
h_in
  ├→ N_sel → 公共消息 μ ──────────────────┐
  └→ 各 N_R,i → 本地消息 m_i → 轻量 Read_sel ┤
                                            └→ 按 content / pre / post 时序完成 selector
                                               → 按 N / SD / BO 提交状态
                                               → active receiver node compute
                                               → ActiveBranchAggregate
                                               → b_G
~~~

H1 使用以下局部下标和集合：

| 符号 | 含义 |
| --- | --- |
| \(i\) | 当前 group 中一一对应的 candidate / receiver node / branch 编号 |
| \(R\) | group 中的 receiver node 总数 |
| \(K_{\mathrm{act}}\) | 当前激活的 receiver node 数量 |
| \(\mathcal A_{j,t}\) | active receiver node 集合 |
| \(\mathcal O_{j,t}\) | 当前 Token 实际提交 Observe / Update 的 receiver node 集合 |
| \(s_{j,t}^{(i)}\)、\(S_{j,t}\) | receiver node \(i\) 的私有状态，以及本 group 的全部私有状态 |
| \(\widehat b_{j,t}^{(i)}\) | receiver node \(i\) 的完整输出；在 H1 中也就是 branch \(i\) 的输出 |
| \(\beta_{j,t}^{(i)}\) | 汇合时分给 active branch \(i\) 的系数 |

其中 \(H=1\) 不是“GraphBranch 里总共只有一个节点”，而是任一入口到出口的候选路径最多经过一个 receiver node。一个 receiver node 内部可以依次包含状态/上下文 residual 和 FFN residual；这两个子层不会把 H 增加为 2。

### 2.2 Receiver node 的入口消息与私有状态

selector 使用自己的归一化 \(N_{\mathrm{sel}}\)，receiver \(i\) 使用自己独立的入口归一化 \(N_{R,i}\)：

$$
\mu_{j,t}=N_{\mathrm{sel}}\!\left(h^{\mathrm{in}}_{j,t}\right),
\qquad
m_{j,t}^{(i)}=N_{R,i}\!\left(h^{\mathrm{in}}_{j,t}\right),
\quad i=0,1,\ldots,R-1.
$$

\(\mu_{j,t}\) 是 selector 的公共内容消息，\(m_{j,t}^{(i)}\) 是 receiver \(i\) 的本地入口消息。各 receiver 只在本地使用 \(m_{j,t}^{(i)}\)，并只向 selector 发送第 2.3 节定义的轻量 \(\operatorname{Read}^{\mathrm{sel}}\)；selector 不读取所有 receivers 的完整入口消息。

使用相同 \(\epsilon\) 的 RMSNorm 时，各归一化的无参数 RMS 统计相同，可以只计算一次：

$$
\bar h^{\mathrm{in}}_{j,t}
=\frac{h^{\mathrm{in}}_{j,t}}
{\operatorname{RMS}(h^{\mathrm{in}}_{j,t})},
$$

再分别应用互不共享的可学习 scale：

$$
\mu_{j,t}=g_{\mathrm{sel}}\odot\bar h^{\mathrm{in}}_{j,t},
\qquad
m_{j,t}^{(i)}=g_{R,i}\odot\bar h^{\mathrm{in}}_{j,t}.
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

### 2.3 三种 selector 时序

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
\mathcal A_{j,t}=\operatorname{TopKIndex}(p_{j,t},K_{\mathrm{act}}),
\qquad 1\le K_{\mathrm{act}}\le R.
$$

Top-1 是 \(K_{\mathrm{act}}=1\) 的特例，此时继续记：

$$
c_{j,t}=\arg\max_i p_{j,t}^{(i)},
\qquad
\mathcal A_{j,t}=\{c_{j,t}\}.
$$

每个直接 receiver 在局部执行 \(N_{R,i}\) 和 \(\operatorname{Read}^{\mathrm{sel}}\)，只向 selector 提供小向量、范数或历史激活统计等少量标量；\(\operatorname{Score}\) 一次输出全部 logits，可以逐候选独立打分，也可以联合处理这些轻量读出。\(\widetilde s\) 是同一个 \(s\) 在“已经 Observe 当前消息、尚未完成本次选择”阶段的值。Pre 与 Post 不是包含关系：如果 \(\operatorname{Update}\) 会覆盖、压缩或遗忘旧状态，post-update state 未必还能恢复 pre-update state；若 selector 同时读取二者，应另行明确声明。

历史激活也可以作为 \(s\) 的内部内容。当前 Token 的激活结果只能影响以后 Token，因此会形成时间维上的因果递归；这不妨碍在一个 chunk 内用 scan、Torch 算子或专用 kernel 执行，但实现必须保持 `prefill = decode`。

### 2.4 传播 profile、状态提交与 receiver node compute

令 \(\mathcal O_{j,t}\) 表示当前消息实际 Observe / Update 并提交状态的 receiver node 集合，\(\mathcal A_{j,t}\) 表示执行较大读出和昂贵 FFN 的 active receiver nodes。三种传播 profile 的规则是：

| Profile | \(\mathcal O_{j,t}\)：哪些 receiver nodes 提交 Observe / Update | 哪些 receiver nodes 继续执行较大读出与昂贵 FFN |
| --- | --- | --- |
| **N（stateless）** | 无状态，不执行 Observe / Update | \(\mathcal A_{j,t}\)；没有状态读出 |
| **SD（selected-dispatch）** | \(\mathcal A_{j,t}\) | \(\mathcal A_{j,t}\) |
| **BO（broadcast-observe）** | 全部 \(R\) 个 receivers | \(\mathcal A_{j,t}\) |

H1 的全部 receivers 都直接收到同一个入口；在 HB-Lattice 中，BO 的提交集合改为当前 Token 实际收到至少一条父消息的全部 reached nodes，见第 3 节。

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

选择与状态提交完成后，在有状态的 SD 和 BO 中，每个 active receiver node \(i\in\mathcal A_{j,t}\) 在局部执行较大的 \(\operatorname{Read}^{\mathrm{ffn}}\)：

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

每个 active receiver node 先把状态/上下文读出加回未归一化的 residual stream，再执行一个 Pre-Norm FFN。令 \(N_{F,i}\) 表示 receiver node \(i\) 的 FFN 前归一化：

$$
u_{j,t}^{(i)}
=h^{\mathrm{in}}_{j,t}+\rho_{j,t}^{(i)},
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

因此，一个标准有状态 receiver node 的 residual 顺序是：

~~~text
入口 h
  → 记忆/上下文读出 residual：u = h + ρ
  → 昂贵 FFN residual：output = u + E(N_F(u))
~~~

这直接对标 Pre-Norm Transformer block：记忆模块的读出承担第一个 residual 子层，昂贵 FFN 承担第二个 residual 子层；\(N_{R,i}\) 和 \(N_{F,i}\) 分别是两个子层的入口归一化。记忆模块内部可以采用 EMA、GDN、Attention 等实现。N 没有第一个 residual 子层，节点输出退化为 \(h+E_i(N_{F,i}(h))\)，但仍可计算供 selector 使用的本地消息和轻量读出。selector 和 `ActiveBranchAggregate` 都位于 receiver node 之外。

\(N_{\mathrm{sel}}\)、各 \(N_{R,i}\) 与各 \(N_{F,i}\) 的可学习参数互不共享，也不与 base block 共享。这两个 residual 子层合计仍只算一个 receiver node，只有该节点的完整输出继续进入下一层节点时，H 才增加。

在 H1 中，各 receiver nodes 共享入口和 group selector，但各自拥有独立的参数与可选状态。\(\widehat b_{j,t}^{(i)}\) 是 receiver node \(i\) 的完整输出；由于 H1 的 branch \(i\) 只经过这一个节点，它同时也是该 branch 在汇合前的输出。H1 的全部 receiver nodes 都执行本地入口归一化和轻量 \(\operatorname{Read}^{\mathrm{sel}}\)；每个 active node 再执行一次较大读出和一次昂贵 FFN。

至此，receiver group 已产生 soft probabilities \(p_{j,t}\)、active node set \(\mathcal A_{j,t}\)、提交后的状态 \(S_{j,t}\) 和各 active nodes 的完整输出 \(\widehat b_{j,t}^{(i)}\)。本节不使用 router 概率缩放或合并这些输出；H1 的 branch 汇合由第 2.5 节定义，HB-Lattice 的对应位置则是第 3.5 节的 `EmitPolicy`。若概率还写入历史状态，必须另外声明写回与梯度规则。

### 2.5 H1 branch 汇合与 ActiveBranchAggregate

H1 的 active receiver node \(i\) 产生完整输出 \(\widehat b_{j,t}^{(i)}\)；因为 branch \(i\) 只经过这一个节点，\(\widehat b_{j,t}^{(i)}\) 也就是该 branch 的完整输出。所有 branches 共享入口 \(h^{\mathrm{in}}_{j,t}\)，`ActiveBranchAggregate` 只在汇合点计算各 branch 相对共同入口产生的变化：

$$
\operatorname{ActiveBranchAggregate}
\left(h,\{(\widehat b_i,\beta_i)\}_{i\in\mathcal A}\right)
=h+
\sum_{i\in\mathcal A}
\beta_i(\widehat b_i-h).
$$

因此 H1 GraphBranch 的完整输出是：

$$
b_{\mathcal G,j,t}
=\mathcal G_j\!\left(h^{\mathrm{in}}_{j,t}\right)
=\operatorname{ActiveBranchAggregate}_{\mathrm{MIX}}
\left(
h^{\mathrm{in}}_{j,t},
\left\{
(\widehat b_{j,t}^{(i)},\beta_{j,t}^{(i)})
\mid i\in\mathcal A_{j,t}
\right\}
\right).
$$

内部 **MIX** 的主要候选是：

| MIX policy | Active set | 合并系数 |
| --- | --- | --- |
| **Top-1 Soft-P** | \(\mathcal A=\{c\}\) | \(\beta_c=p_c\) |
| **Top-1 Hard-ST** | \(\mathcal A=\{c\}\) | \(\beta_c=1+p_c-\operatorname{sg}(p_c)\) |
| **Top-K 均匀平均** | \(\lvert\mathcal A\rvert=K_{\mathrm{act}}\) | \(\beta_i=1/K_{\mathrm{act}}\) |
| **Top-K router 加权** | \(\lvert\mathcal A\rvert=K_{\mathrm{act}}\) | \(\displaystyle\beta_i=p_i/\sum_{k\in\mathcal A}p_k\) |
| **学习型局部聚合** | \(\lvert\mathcal A\rvert=K_{\mathrm{act}}\) | active branches 上归一化的学习权重 |

\(\operatorname{sg}\) 表示 stop-gradient：前向值不变，反向梯度为零。Top-1 Hard-ST 的 \(\beta_c\) 在前向等于 1，且 \(\partial\beta_c/\partial p_c=1\)；离散 Top-1 选择本身仍不参与反向传播。

学习型局部聚合可以写成：

$$
(\beta_i)_{i\in\mathcal A}
=\operatorname{softmax}\!\left(
\operatorname{MergeScore}
\left(h,\{(\widehat b_i,p_i)\}_{i\in\mathcal A}\right)
\right).
$$

均匀平均适合作为简单对照，归一化 router 加权作为 Top-K 主设置，学习型局部聚合留作后续候选。均匀平均不通过合并系数训练 selector；router 加权可以训练 active branches 之间的相对权重，但离散的 Top-K 成员选择仍不求导。Top-K router 加权在 \(K_{\mathrm{act}}=1\) 时会归一化为 1，且对该概率的导数为 0：其前向值与 Top-1 Hard-ST 相同，反向却不同。

router 概率只通过 \(\beta_i\) 直接缩放本次汇合，不在 receiver node 内再次缩放。对一个标准 receiver node 的完整输出：

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

N 中 \(\rho_c=0\)；Top-1 Hard-ST 的前向完整保留被选 receiver node 的计算结果。若直接求和完整 branch 输出，公共输入 \(h\) 会被重复加入 \(K_{\mathrm{act}}\) 次，因此多分支必须聚合变化 \(\widehat b_i-h\)，而不是直接相加 \(\widehat b_i\)。

更深 GraphBranch 中，一个显式 fork-join 的分支 \(\mathcal B_i(h)\) 也可以是已经完成内部传播和收拢的子结构；只要各分支具有同一个入口并返回完整 hidden，就能复用上述 **MIX**。一般 HB-Lattice 的边不会在每层都立即汇合，其多父入口和 sender 输出分别由第 3.3、3.5 节定义。

第 1.3 节的 GraphBranch 外部 **RESIDUAL_ADD** 与这里使用同一个代数接口，但职责不同。令 \(b^0\) 为 placement 的 always-on 输出，则：

$$
\operatorname{ActiveBranchAggregate}_{\mathrm{RESIDUAL\_ADD}}
(h;b^0,b_{\mathcal G})
=b^0+(b_{\mathcal G}-h)
=\operatorname{BoundaryMerge}(h;b^0,b_{\mathcal G}).
$$

GraphBranch 内部的显式 fork-join 使用 **MIX**；GraphBranch 与 base 的唯一边界使用 **RESIDUAL_ADD**。对标准 receiver node，identity 初始化要求 \(\rho_i=0\) 且 FFN residual 为零；此时任意内部 MIX 都返回共同入口，进而得到 \(\Delta_{\mathcal G}=0\)。

### 2.6 状态生命周期

以下规则同时适用于 H1 和 HB-Lattice 的 receiver nodes。每条独立序列都从空状态开始：EMA、GDN 和 SSM 状态置零，Attention 历史为空，历史激活计数清零。padding 等无效 Token 不执行 Observe / Update，也不进入路由辅助 loss。

同一逻辑序列跨 chunk 时继承状态值，不同序列之间清零；默认在每个 chunk 边界 detach，状态继续前传，但梯度只在当前 chunk 内传播。对同一有效前缀，整段 prefill、任意分块和逐 Token decode 应在数值误差范围内得到相同的逐 Token 输出与最终状态。

### 2.7 Receiver node 的状态模块样例

第 2.2 至 2.4 节中的 \(s\)、\(\operatorname{Update}\)、\(\operatorname{Read}^{\mathrm{sel}}\) 和 \(\operatorname{Read}^{\mathrm{ffn}}\) 是 H1 与 HB-Lattice 共用的稳定接口。本节只展开 receiver node 内部的状态模块；第 2.4 节定义的昂贵 FFN \(E\) 保持不变。下面的样例用于建立设计空间，不表示它们已经通过 TIDE 实验，也不预设哪一种必然最好。状态实现与 selector 时序是两个独立坐标：content-only 的 \(\operatorname{Read}^{\mathrm{sel}}\) 只读取当前本地消息，pre/post state 则额外读取对应时刻的状态；\(\operatorname{Read}^{\mathrm{ffn}}\) 不受这一选择影响。

#### 2.7.1 一览

| 样例 | 主要保留什么 | 典型消费者 | 主要特点 |
| --- | --- | --- | --- |
| **历史激活** | 激活次数、最近激活位置、概率或局部预算 | selector | 最轻量；记录控制历史，不直接保存内容语义 |
| **EMA** | 一个固定长度的低通内容摘要 | selector / FFN | 简单、稳定，但不同历史会持续混合 |
| **Gated DeltaNet（GDN）** | 固定大小的 key-value 关联矩阵 | selector / FFN | 可以按 query 关联读取，并按预测误差写入 |
| **Kimi Delta Attention（KDA）** | 带细粒度门控的 delta-rule 矩阵状态 | selector / FFN | delta-rule 家族的近期增强，门控更细但实现更复杂 |
| **SSM / Mamba-2** | 固定大小的状态空间递归状态 | selector / FFN | 与 delta-rule 不同的成熟有界状态路线 |
| **Attention** | 完整历史、局部窗口或压缩后的 key/value | selector / FFN | 设计空间大；信息保留与状态/计算成本由具体实现决定 |

两类读出都在 receiver 局部完成：\(\operatorname{Read}^{\mathrm{sel}}\) 通常只输出低维投影、范数或历史统计，\(\operatorname{Read}^{\mathrm{ffn}}\) 则在内部完成必要的 output projection，并统一输出 hidden 维 residual。“典型消费者”只是常见用法，不是硬限制。

#### 2.7.2 历史激活

历史激活可以记录每个候选被选中的次数、距上次激活的 Token 数、soft probability 的移动平均或剩余局部预算、历史 selector 打分。本次选择只能在 selector 决策完成后写回，因此只影响以后 Token。若它只服务于 selector，则对应的 \(\rho_i=0\)。

#### 2.7.3 EMA

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

#### 2.7.4 Gated DeltaNet 与 KDA

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

#### 2.7.5 Attention 状态

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

#### 2.7.6 其他有界状态路线与当前定位

SSM / Mamba-2 是另一类重要的固定状态候选，开放权重的 [Falcon-H1](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Base) 已采用 Transformer 与 Mamba 的混合结构；RWKV-7、Lightning Attention 等也提供了可参考的递归或线性注意力状态。它们证明“有界 recurrent state”有多条成熟路线，但不必全部进入首轮 TIDE 实现。

当前更合适的定位是：历史激活用于最轻量的 selector 控制，EMA 作为简单内容基线，GDN 作为第一种先进关联记忆锚点，Attention 保留为可按预算选择的宽泛设计族；KDA 和 Mamba/SSD 则是增强或跨家族候选。这只是帮助建立全局观，不是固定实验顺序。维度和状态量必须在名称中明确，例如 **GDN-K32-V32** 有 \(32\times32=1024\) 个状态标量，不能与 EMA128 当作等状态量对照。

## 3. HB-Lattice：从 H1 推广到多父波前

H1 先让读者看清一个共享入口怎样经过“选择—状态提交—昂贵计算—汇合”变成 GraphBranch 输出。HB-Lattice 保留同一种 receiver node，把多个节点沿预先规定的多层空间拓扑连接起来。

当前规范中，`HBLatticePlan` 的每个 node 都是第 2 节定义的 receiver node；Plan 不另外创建一个叫作“branch”的计算单元。对某个 Token，实际 active 的 nodes 和 Emit edges 共同形成 active subgraph，未必能划成若干互不相交的 branches。只有 Plan 中显式存在共享入口、共享汇合点的 fork-join 子结构时，才继续使用第 2.5 节的 branch 和 `ActiveBranchAggregate` 语义。

H1 与 HB-Lattice 最重要的对应关系是：

| 位置 | H1 | HB-Lattice |
| --- | --- | --- |
| receiver node 入口 | 所有候选节点共享 \(h^{\mathrm{in}}_{j,t}\) | 每个 reached node 先由自己的 inbox 得到 \(h_{v,t}\) |
| 当前候选 | 固定的 \(R\) 个 receiver nodes | 当前 region 中实际 reached 的 receiver nodes |
| selector | 一个共享-parent group 内选择 | 每个 Line 的各 region 分别选择 |
| 状态与节点计算 | 第 2.2 至 2.4 节的接口 | 复用相同接口，只把入口换成 \(h_{v,t}\) |
| receiver node 输出 | 同时作为一层 branch 输出，立即进入 `ActiveBranchAggregate` | 先由 `EmitPolicy` 发向固定 children |
| 多路合并 | 共享入口的 branch outputs 在 group 出口汇合 | 目标 node 先用 `ParentAggregate` 聚合实际父消息 |
| GraphBranch 出口 | 汇合结果就是 \(b_{\mathcal G,j,t}\) | 最终 sink 的 Emit 是 \(b_{\mathcal G,j,t}\) |

HB-Lattice 从本节开始使用以下局部符号：

| 符号 | 含义 |
| --- | --- |
| \(d\) | GraphBranch 内部的波前 Line 编号 |
| \(r\) | 一个 Line 内的 selector region 编号 |
| \(v\) | 当前 receiver node；每个 node 只属于一个 Line |
| \(w\) | \(v\) 的一个固定 parent node |
| \(Q\) | 平台期每个 Line 共享的空间坐标集合 |
| \(\operatorname{Inbox}_{v,t}\) | 节点 \(v\) 在当前 Token 实际收到的父消息集合 |
| \(q_{v,t}\) | 节点 \(v\) 是否收到至少一条父消息的 reached 标记 |
| \(\mathcal C_{d,r,t}\) | region \((d,r)\) 当前实际 reached 的候选集合 |
| \(h_{v,t}\) | 节点 \(v\) 聚合 inbox 后的完整入口 hidden |
| \(s_{v,t}\) | 节点 \(v\) 的 receiver 私有状态；site 下标在本节省略 |
| \(\mathcal A_{d,r,t}\) | region \((d,r)\) 当前选出的 active 节点集合 |
| \(g_{v,t}\) | active receiver node 完成计算、尚未 Emit 的完整 hidden |
| \(\widehat g_{v,t}\) | active receiver node 经 `EmitPolicy` 实际发送的完整 hidden |

### 3.1 受限 HB-Lattice 与波前执行

\(H>1\) 的标准执行对象是手动规定波前的 **HB-Lattice**，不是一般 DAG。**H2** 只表示入口到出口的最大 receiver node 深度为 2，并不能单独确定节点数、边、region 或多父关系；这些内容由 topology Plan 明确给出。数学上 HB-Lattice 是一个分层 DAG，但执行器只处理下面这类受限结构：

- GraphBranch 只有一个入口和一个出口；
- 节点被静态分配到有序 Line \(L_0,L_1,\ldots,L_D\)，每个节点只属于一个 Line；
- Line 内没有消息依赖，普通边只连接相邻 Line；
- 递归扩展节点可以沿显式声明的镜像直通边连接到对应收拢节点；
- 平台期各 Line 使用相同的空间坐标集合，每对相邻平台 Line 的连接可以分别指定；
- 每个 Line 被划分为固定、不重叠的 selector regions，每个节点只有一个 activation owner；
- 同一 Token 在一个节点只聚合一次父消息、更新一次状态并至多执行一次昂贵计算。

因此近期实现不需要任意拓扑排序、同层依赖、异步 event queue、有环执行或一般 DAG 接口。

### 3.2 两层拓扑接口

第一层是执行层，由规范化的 `HBLatticePlan`、`HBLatticeExecutionConfig` 和逐 Line 运行二者的 `WavefrontExecutor` 组成。Plan 只回答“谁与谁连接、谁由哪个 region 选择”，至少完整列出：

~~~text
HBLatticePlan
├── Lines：每层的 phase、节点、坐标和 selector regions
├── adjacent edges：扩展、平台和收拢的相邻 Line 边
├── mirror map / edges：扩展与收拢节点的对应关系及逐节点直通开关
├── entry / sink
├── forced-active 节点
└── edge class：tree / local / shortcut / mirror
~~~

`HBLatticeExecutionConfig` 则回答 reached 节点怎样计算，包含 propagation profile、receiver/state、selector 时序、逐 region 的 \(K^{\max}\)、`RegionSelector`、`EmitPolicy`、`ParentAggregate` 和仅训练使用的 `BalancePolicy`。这些接口可以统一配置，也可以按节点或 region 映射到不同配置。

扩展树和收拢树可以采用不均匀但有界的分支结构；平台期每对 Line 的邻接也可以不同。`WavefrontExecutor` 只消费已展开的 Plan 和执行配置，不负责猜测树形、空间邻接或镜像关系。

Plan 载入时必须检查 Line 顺序、region 唯一归属、边类型与端点、entry 到 sink 的静态可达性、镜像对应关系，以及声明的 fan-in/fan-out 和 region-size 上界。

第二层由一个或多个 `TopologyBuilder` 组成：

~~~text
TopologyBuilder(config) → HBLatticePlan
~~~

builder 可以生成规则树、逐坐标混合、重复空间 Graph 或其他 HB-Lattice 模板。每个正式实验同时保存最终 Plan 的规范化内容与哈希、完整执行配置，以及 builder 名称、版本和配置。`TOPO_ID` 只索引拓扑 Plan；完整计算语义由 Plan 与执行配置共同决定，不能只看生成器名称。

### 3.3 Inbox、reached 与多父聚合

设节点 \(v\in L_d\) 的固定父节点集合为 \(P(v)\)。当前 Token 上，只有已经激活、完成计算并 Emit 的父节点才会发送完整 hidden。节点 \(v\) 的实际 inbox 为：

$$
\operatorname{Inbox}_{v,t}
=\left\{
(w,\widehat g_{w,t})
\mid w\in P(v),\ w\text{ 在 Token }t\text{ 已 Emit}
\right\}.
$$

GraphBranch 入口 \(h^{\mathrm{in}}_{j,t}\) 作为 entry node 的一条外部消息，因此入口不依赖图内父节点也能 reached。镜像直通消息可以在较早的 Line 产生，但只保存在目标 inbox 中；目标 Line 到来、所有可能父节点都已经结算后，才把“未到达”和“尚未到达”区分开。令：

$$
q_{v,t}=\mathbf 1[\operatorname{Inbox}_{v,t}\ne\varnothing]
$$

表示节点是否 reached。若 \(q_{v,t}=0\)，节点不参加当前选择，不更新状态，也不输出；若收到一条或多条消息，则先执行一次与消息到达顺序无关的 `ParentAggregate`：

$$
h_{v,t}
=\operatorname{ParentAggregate}_v(\operatorname{Inbox}_{v,t})
=\sum_{(w,\widehat g_{w,t})\in\operatorname{Inbox}_{v,t}}
\alpha_{w\to v,t}\widehat g_{w,t},
$$

$$
\alpha_{w\to v,t}\ge0,
\qquad
\sum_{(w,\widehat g_{w,t})\in\operatorname{Inbox}_{v,t}}
\alpha_{w\to v,t}=1.
$$

首个设置使用 **PAGG-MEAN**：对实际到达的父消息均匀平均，并把到达数量或有界 parent-presence mask 作为额外轻量信息。归一化聚合不会因父消息数量变化而重复放大公共 hidden，也能在各节点初始化为 identity 时保持 identity。以后若使用学习型局部聚合，应由目标节点自己的 \(\operatorname{ParentMergeScore}\) 产生 \(\alpha\)，单独记为 **PAGG-LEARNED**。

`ParentAggregate` 默认不使用各父节点的 selector 概率：不同父节点的概率可能来自不同 regions，数值不能直接比较；只有一个实际父节点时，归一化权重又恒为 1，无法给上游 selector 提供梯度。selector 的主任务梯度统一放在 sender 的 `EmitPolicy`，见第 3.5 节。

### 3.4 Region selector 与节点计算

节点 \(v\) 用自己的入口归一化产生本地消息：

$$
m_{v,t}=N_{R,v}(h_{v,t}),
$$

再按第 2.3 节的 content-only、pre-update 或 post-update 时序产生轻量 \(\operatorname{Read}^{\mathrm{sel}}\)。同一 region 的节点可能拥有不同的 \(h_{v,t}\)，因此一般 HB-Lattice 不要求存在 H1 中由共同入口产生的单一 \(\mu\)；region selector 只联合处理各 reached nodes 发来的轻量读出和 presence 信息。

对 region \(\mathcal R_{d,r}\subseteq L_d\)，当前候选集合为：

$$
\mathcal C_{d,r,t}
=\{v\in\mathcal R_{d,r}\mid q_{v,t}=1\}.
$$

region selector 可以联合处理本 region 的轻量读出，但只为 reached nodes 产生有效概率。对未 reached 节点，约定其读出以零占位并由 \(q_{v',t}=0\) mask 掉，不在 receiver 端实际计算：

$$
a_{v,t}
=\left[
\operatorname{Score}_{d,r}
\left((r_{v',t,\tau}^{\mathrm{sel}},q_{v',t})_{v'\in\mathcal R_{d,r}}\right)
\right]_v,
\qquad v\in\mathcal C_{d,r,t},
$$

$$
(p_{v,t})_{v\in\mathcal C_{d,r,t}}
=\operatorname{softmax}
\left((a_{v,t})_{v\in\mathcal C_{d,r,t}}\right).
$$

`RegionSelector` 返回 `SelectionDecision`，其中包含当前 \(\mathcal C_{d,r,t}\)、soft probabilities 和 hard active set \(\mathcal A_{d,r,t}\)。不同 regions 的 selector 参数默认不共享；若实验共享，必须在执行配置中声明。

若 \(\mathcal C_{d,r,t}=\varnothing\)，该 region 选择空集；否则必须满足：

$$
1\le |\mathcal A_{d,r,t}|
\le \min(K_{d,r}^{\max},|\mathcal C_{d,r,t}|).
$$

全激活只是令所有 reached nodes 都进入 \(\mathcal A_{d,r,t}\) 的特例。入口和最终 sink 可以在 Plan 中声明为 forced-active，含义是“只要 reached 就必定 active”，不能让未 reached 节点凭空激活。

传播 profile 在 HB-Lattice 中统一解释为：

- **N**：无状态，只有 active nodes 执行完整节点计算；
- **SD**：只有 active nodes 提交 Update 并执行完整节点计算；
- **BO**：全部 reached nodes 提交 Update，只有 active nodes 执行较大读出和昂贵计算。

因此 BO + post-update state 的顺序是“全部 reached nodes 聚合并 Update → 各 region 选择 → active nodes 执行”；SD 仍只自然兼容 content-only 和 pre-update state。一个节点即使收到多个父消息，也只在 `ParentAggregate` 后 Update 一次。

active 节点沿用第 2.4 节的完整 block-like 计算，只把 H1 的共享入口 \(h^{\mathrm{in}}_{j,t}\) 换成本节点入口 \(h_{v,t}\)。有状态节点在状态提交后读取：

$$
\rho_{v,t}
=\operatorname{Read}_{v}^{\mathrm{ffn}}(s_{v,t},m_{v,t}),
$$

无状态的 N 令 \(\rho_{v,t}=0\)。随后计算：

$$
u_{v,t}=h_{v,t}+\rho_{v,t},
$$

$$
g_{v,t}
=u_{v,t}+E_v\!\left(N_{F,v}(u_{v,t})\right).
$$

这里的 \(g_{v,t}\) 是尚未应用 `EmitPolicy` 的完整节点输出。

### 3.5 EmitPolicy、主任务梯度与 Line barrier

`EmitPolicy` 只作用于 active 节点，并决定它实际发给全部固定 children 的消息 \(\widehat g_{v,t}\)。首个推荐设置是 **EMIT-HST**：

$$
\xi_{v,t}^{\mathrm{emit}}
=1+\zeta_{d,r,t}^{\mathrm{ST}}
\left(p_{v,t}-\operatorname{sg}(p_{v,t})\right),
\qquad v\in\mathcal A_{d,r,t},
$$

$$
\widehat g_{v,t}
=h_{v,t}
+\xi_{v,t}^{\mathrm{emit}}
\left(g_{v,t}-h_{v,t}\right).
$$

\(\operatorname{sg}\) 表示 stop-gradient：前向值不变，反向梯度为零。因此前向恒有 \(\xi^{\mathrm{emit}}=1\) 和 \(\widehat g=g\)，但主任务梯度仍可通过 \(p_v\) 返回 selector：

$$
\left.
\frac{\partial\mathcal L_{\mathrm{LM}}}{\partial p_{v,t}}
\right|_{\mathrm{Emit}}
=\zeta_{d,r,t}^{\mathrm{ST}}
\left\langle
\frac{\partial\mathcal L_{\mathrm{LM}}}{\partial\widehat g_{v,t}},
g_{v,t}-h_{v,t}
\right\rangle.
$$

这是本次 Emit 带来的直接梯度；若概率还进入历史状态，可能另有跨 Token 梯度路径。这里把 hard active set 视为常量，离散的 Top-1/Top-K 成员选择本身仍不求导。Top-1 的首个设置取 \(\zeta_{d,r,t}^{\mathrm{ST}}=1\)；Top-K 可先取 \(\zeta_{d,r,t}^{\mathrm{ST}}=1/|\mathcal A_{d,r,t}|\) 控制梯度尺度。该值只改变反向，不改变前向，必须写入实验设置。

可对照的 `EmitPolicy` 为：

| Policy | 实际 Emit | selector 从主任务得到的梯度 | 用途 |
| --- | --- | --- | --- |
| **EMIT-HARD** | \(\widehat g_v=g_v\) | 不通过 Emit 返回 | 诊断 selector 只靠辅助 loss 时能否训练 |
| **EMIT-HST** | 前向 \(\widehat g_v=g_v\) | 通过上面的 delta Hard-ST 返回 | 首个推荐设置 |
| **EMIT-SOFTP** | \(\widehat g_v=h_v+p_v(g_v-h_v)\) | 通过 soft \(p_v\) 返回 | 会改变前向强度，作为消融 |
| **EMIT-CUSTOM** | 由实验明确 | 由实验明确 | 后续 utility surrogate 等候选 |

未激活节点不执行昂贵计算，也不 Emit。首个 `ParentAggregate` 只聚合 \(\widehat g\)，不再使用 selector \(p\)；以后即使引入 **PAGG-LEARNED**，也应使用独立的 \(\operatorname{ParentMergeScore}\)。

对同一次选择，selector 概率只能在一个位置直接缩放当前分支或消息：普通 HB-Lattice 节点使用 `EmitPolicy`；共享 parent 的 H1 或显式 fork-join 使用第 2.5 节的 `ActiveBranchAggregate`。不得把同一个 \(p\) 同时用于二者或再次用于 `ParentAggregate`；若把 \(p\) 写入历史状态，必须单独记录是否 stop-gradient。

每个 region 还把 \((\mathcal C,p,\mathcal A)\) 连同 site、Line、region、序列和 Token 标识记录为一个 `RoutingEvent`，交给训练期 `BalancePolicy`；它不改变推理前向。第 5.2 节定义首个 **BAL-AVAIL-SOFT**。

至此，一个 Line 的数据流可以概括为：

~~~text
Inbox → ParentAggregate → receiver-local light stage
      → 按 content / pre / post 时序完成 RegionSelector
      → profile 所规定的状态提交 → active receiver node compute
      → EmitPolicy → child inbox

RoutingEvent(C, p, A) → BalancePolicy（仅训练）
~~~

BO + post-update state 的 Update 位于 `RegionSelector` 之前，其他 profile 和 selector 时序按第 2.3、2.4 节执行。只有当前 Line 的 inbox、选择、状态提交、完整节点计算和 Emit 全部结算后，才开始下一个 Line。实现可以做等价的批处理或流水线，但不得改变这个 reference semantics。最终 sink 经 `EmitPolicy` 得到的完整 hidden 就是 \(b_{\mathcal G,j,t}\)。

`ParentAggregate` 是目标节点入口处的多父合并；`EmitPolicy` 是 sender 端的主任务梯度接口；`BalancePolicy` 只产生训练期辅助 loss；`ActiveBranchAggregate` 则处理共享 parent 的显式 fork-join，或以 RESIDUAL_ADD 形式表达 GraphBranch 外部边界。四者职责互不替代。`AGG-NONE` 只表示没有显式 fork-join，不表示没有 `ParentAggregate` 或 `EmitPolicy`。

当节点初始化为严格 identity 时，\(g_v-h_v=0\)，EMIT-HST 的主任务 selector 梯度也为零。首轮训练可以让 balance loss 先提供路由牵引，并使用全激活或较大的 \(K^{\max}\) 做短 warmup；节点离开 identity 后，主任务梯度才会逐渐进入 selector。

这四个接口分别配置，首个 HB-Lattice 实现建议采用：

| 接口 | 首个设置 | 可替换方向 |
| --- | --- | --- |
| `RegionSelector` | reached candidates 上的 masked Top-1/Top-K | 全激活或其他局部选择 |
| `EmitPolicy` | EMIT-HST | EMIT-HARD、EMIT-SOFTP 或后续 surrogate |
| `ParentAggregate` | PAGG-MEAN | PAGG-LEARNED |
| `BalancePolicy` | BAL-AVAIL-SOFT | BAL-NONE 或另行定义的统计目标 |

它们是可独立替换的接口，不是整个框架唯一允许的实现。

### 3.6 两类标准 TopologyBuilder

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

## 4. Dense 与标准 MoE 基线

### 4.1 DENSE

DENSE 使用原 block：

$$
y=v
=u+F_\ell(N_F(u)).
$$

### 4.2 MOE

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

## 5. 实际训练时的损失函数

令 \(\mathcal T\) 表示一个 micro-batch 中所有有效目标 Token 的 \((b,t)\) 集合，\(N_T=|\mathcal T|\)。自回归语言模型损失为：

$$
\mathcal L_{\mathrm{LM}}
=-\frac{1}{N_T}
\sum_{(b,t)\in\mathcal T}
\log P_\theta(w_{b,t}\mid w_{b,<t}).
$$

路由辅助项使用的 Token 集合略有不同。当前 H1 中，每个 site 的 router 都处理同一个集合 \(\mathcal V\)：attention mask 标记为有效、实际经过 router 的全部 \((b,t)\) 位置，\(N_V=|\mathcal V|\)。\(\mathcal V\) 与 receiver 或 expert \(i\) 无关，不是候选 \(i\) 实际被选中的 Token 集。balance loss 不要求单个 Token 均匀选择所有候选，而是避免整个 micro-batch 长期集中到少数 receivers 或 experts。令 \(\mathcal I\) 表示所有 routed sites，\(I=|\mathcal I|\)。

每个 site 独立计算 balance loss，再在 sites 间等权平均。统计范围是当前 micro-batch；梯度累积只累积各 micro-batch 的梯度，不预先把多个 micro-batches 合并成 global-batch balance loss。

### 5.1 H1 中 N、SD、BO 的 receiver balance loss

对 site \(j\) 的 \(R\) 个 receivers，平均 softmax 概率为：

$$
\bar p_{j,i}
=\frac{1}{N_V}
\sum_{(b,t)\in\mathcal V}p_{j,b,t}^{(i)}.
$$

当前 H1 的 N、SD、BO 共同使用；它也是 **BAL-AVAIL-SOFT** 在全部候选始终 reached 时的特例：

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

### 5.2 HB-Lattice 的 region balance loss

HB-Lattice 中每个 region 只处理实际 reached 的节点。对 site \(j\)、Line \(d\)、region \(r\)，固定节点集合记为 \(\mathcal R_{j,d,r}\)，并令：

$$
\mathcal V_{j,d,r}
=\left\{
(b,t)\mid\mathcal C_{j,d,r,b,t}\ne\varnothing
\right\},
\qquad
N_{j,d,r}=|\mathcal V_{j,d,r}|
$$

表示该 region 在当前 micro-batch 中真正发生选择的 Token 事件。selector 只在 \(\mathcal C_{j,d,r,b,t}\) 内做 masked softmax；未 reached 节点不进入当前候选集合，也不能被 balance loss 当作本次本应选择的候选。

这里的 \(\mathcal C_{j,d,r,b,t}\) 就是第 3.4 节的 \(\mathcal C_{d,r,t}\) 补回 site \(j\) 和序列 \(b\) 下标。

首个 `BalancePolicy` 使用 **BAL-AVAIL-SOFT**。对 \(N_{j,d,r}>0\) 的 region，约定未 reached 时 \(p_{j,d,r,b,t}^{(v)}=0\)，并定义节点 \(v\) 实际得到的平均 soft mass：

$$
\bar p_{j,d,r,v}
=\frac{1}{N_{j,d,r}}
\sum_{(b,t)\in\mathcal V_{j,d,r}}
\mathbf 1[v\in\mathcal C_{j,d,r,b,t}]
p_{j,d,r,b,t}^{(v)}.
$$

同一 availability 下，若每次都在当前 reached candidates 中均匀选择，节点 \(v\) 应得到的基准 mass 为：

$$
\bar p_{j,d,r,v}^{\mathrm{avail}}
=\frac{1}{N_{j,d,r}}
\sum_{(b,t)\in\mathcal V_{j,d,r}}
\frac{
\mathbf 1[v\in\mathcal C_{j,d,r,b,t}]
}{
|\mathcal C_{j,d,r,b,t}|
}.
$$

region 的 loss 为：

$$
\mathcal L_{\mathrm{bal},j,d,r}^{\mathrm{avail}}
=
\sum_{v\in\mathcal R_{j,d,r}}
\left(
\bar p_{j,d,r,v}
-\bar p_{j,d,r,v}^{\mathrm{avail}}
\right)^2.
$$

令 \(\mathcal Z\) 表示当前 micro-batch 中至少出现过一次 \(|\mathcal C_{j,d,r,b,t}|\ge2\) 的 region 实例，则：

$$
\mathcal L_{\mathrm{bal}}^{\mathrm{HB}}
=\frac{1}{|\mathcal Z|}
\sum_{(j,d,r)\in\mathcal Z}
\mathcal L_{\mathrm{bal},j,d,r}^{\mathrm{avail}},
$$

$$
\boxed{
\mathcal L_{\mathrm{train}}^{\mathrm{HB}}
=\mathcal L_{\mathrm{LM}}
+\omega_{\mathrm{HB}}\mathcal L_{\mathrm{bal}}^{\mathrm{HB}}
}.
$$

\(\omega_{\mathrm{HB}}\ge0\) 由实验设置记录；若 \(\mathcal Z=\varnothing\)，约定 \(\mathcal L_{\mathrm{bal}}^{\mathrm{HB}}=0\)。这个 reduction 先在每个 region 内计算，再对本 micro-batch 中真正出现过竞争选择的 regions 等权平均；始终只有一个候选的 entry、sink 或 singleton region 不稀释 loss。

在这个 policy 中，reached mask、\(\mathcal C\)、\(\bar p^{\mathrm{avail}}\) 和 hard active set 都视为 stop-gradient；balance 梯度只通过当前 region 的 \(p\) 返回 selector。

这个目标只比较“在同样已经 reached 的候选范围内，selector 是否长期偏向某些节点”：

- 若所有候选始终 reached，则 \(\bar p_v^{\mathrm{avail}}=1/R\)，退化为第 5.1 节的 H1 目标；
- 若某次只有一个候选 reached，该事件的实际 \(p\) 与均匀基准都为 1，不会产生无法完成的均衡要求；
- 从未 reached 的节点在 \(\bar p\) 和 \(\bar p^{\mathrm{avail}}\) 中都为 0。

例如一个 region 只有 A、B 两个节点：第一个 Token 只 reached A，第二个 Token 同时 reached A、B，则两个事件的均匀基准分别是 \((1,0)\) 和 \((1/2,1/2)\)，micro-batch 基准为 \(\bar p^{\mathrm{avail}}=(3/4,1/4)\)，而不是强行要求 \((1/2,1/2)\)。

它在 micro-batch 的平均值上约束分布，不强迫每个 Token 的 selector 概率都均匀，因此仍允许按内容形成专业化。它也不负责修复上游路由或 topology 造成的 reach starvation；那是独立问题。

同时记录 hard active share：

$$
\bar f_{j,d,r,v}
=\frac{1}{N_{j,d,r}}
\sum_{(b,t)\in\mathcal V_{j,d,r}}
\frac{
\mathbf 1[v\in\mathcal A_{j,d,r,b,t}]
}{
|\mathcal A_{j,d,r,b,t}|
}.
$$

\(\bar p\) 是可导的 soft mass，\(\bar p^{\mathrm{avail}}\) 是 availability 基准，\(\bar f\) 是实际 active slots 的份额；三者应一起报告。另按所有有效 site-Token 事件分别记录每个节点的 reached、Observe、active 和 Emit rate，用来区分：

| 现象 | 首先检查 |
| --- | --- |
| 节点很少 reached | topology 与上游路径选择 |
| reached 后总是落选 | region selector 的 \(\bar p,\bar f\) |
| Observe 少 | reached 情况与 N/SD/BO profile |
| Emit 少或计算量失衡 | active set、forced-active 与实际执行 |

**BAL-NONE** 可作为无辅助均衡的消融；其他 opportunity-normalized 或跨 micro-batch 方案统一写 **BAL-CUSTOM**，并在实验设置中给出完整公式、统计范围和 reduction。

### 5.3 标准 MoE（M8）的 balance loss 与 router z-loss

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

沿用第 4.2 节，MoE router 收到的消息、expert 输入和 router logits 为：

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

这里的动态 expert bias 和 Quantile Balancing 都是训练期均衡；Kimi K3 的最终 bias 在推理时冻结，不等于第 5.4 节的推理期负载感知 selector。

DENSE 没有 router，实际目标只有 \(\mathcal L_{\mathrm{LM}}\)。训练日志中的 `loss` 是包含上述辅助项的总损失，`lm_loss` 只表示 Token 预测损失；跨架构比较模型质量时应使用验证集 `lm_loss` 或 perplexity，而不是直接比较总 `loss` 或两种定义不同的 `balance_loss`。

### 5.4 训练期均衡与推理期负载感知

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

## 6. 规范命名

### 6.1 科学条件名

共享 parent 的 H1 候选继续采用：

~~~text
<TRAIN>-<PLACEMENT>-<PROFILE>-R<WIDTH>-I<SITES>-H<DEPTH>-<STATE>-<SELECTOR>-<AGG>-<BAL>
~~~

例如：

~~~text
CPT-PARMLP-BO-R8-I4-H1-EMA128-SEL-POST-AGG-T1-HST-BAL-AVAIL-SOFT
PT-POST-SD-R8-I4-H1-EMA128-SEL-PRE-AGG-T1-SOFTP-BAL-AVAIL-SOFT
~~~

非平凡 HB-Lattice 使用：

~~~text
<TRAIN>-<PLACEMENT>-<PROFILE>-R<WIDTH>-I<SITES>-H<DEPTH>-T<TOPO_ID>-<STATE>-<SELECTOR>-K<ACTIVE>-<EMIT>-<PAGG>-<AGG>-<BAL>
~~~

例如：

~~~text
PT-POST-BO-R2-I4-H7-THBL2D2P2CMIR-GDN-K32-V32-SEL-POST-K1-EMIT-HST-PAGG-MEAN-AGG-NONE-BAL-AVAIL-SOFT
~~~

`TOPO_ID` 是已展开 `HBLatticePlan` 的可读索引，不代替 manifest 中的完整 Plan 与规范化哈希。

字段定义如下：

| 字段 | 允许值或形式 | 含义 |
| --- | --- | --- |
| TRAIN | PT / CPT / FT / SFT | 初始化与训练阶段 |
| PLACEMENT | POST / PARBLK / PARATTN / PARMLP | GraphBranch 的输入与 residual 返回位置 |
| PROFILE | N / SD / BO | 状态接收与稀疏计算语义 |
| R | R4、R8、R16、RVAR 等 | H1 group 或 HB-Lattice 非平凡 selector region 的候选数；不统一时用 RVAR |
| I | I1、I4、I8 等 | 一个 Token 顺序经过的插入位置数 |
| H | H1、H2 等 | 每个插入位置内部从入口到出口的最大 receiver node 深度 |
| T | T\<TOPO_ID\> | 非平凡 HB-Lattice 的已展开 topology 索引；H1 可以省略 |
| STATE | NONE、EMA128、GDN-K32-V32、ATTN-FULL、ATTN-W128、ATTN-COMP 等 | 状态结构和必要尺寸 |
| SELECTOR | SEL-CONTENT / SEL-PRE / SEL-POST | 第 2.3 节定义的 selector 输入时序 |
| K | K1 / K2 / KALL / KVAR | 非平凡 HB-Lattice region 的 \(K^{\max}\) 摘要；H1 由 AGG 表示 |
| EMIT | EMIT-HARD / EMIT-HST / EMIT-SOFTP / EMIT-CUSTOM / EMIT-VAR | 第 3.5 节定义的 HB-Lattice sender Emit 语义；H1 省略 |
| PAGG | PAGG-MEAN / PAGG-LEARNED / PAGG-CUSTOM / PAGG-VAR | 第 3.3 节定义的 HB-Lattice 多父聚合；H1 省略 |
| AGG | AGG-NONE / AGG-T1-SOFTP / AGG-T1-HST / AGG-K2-MEAN / AGG-K2-ROUTER / AGG-K2-LEARNED / AGG-VAR | 第 2.5 节定义的显式 fork-join MIX policy；没有显式 fork-join 时用 NONE |
| BAL | BAL-AVAIL-SOFT / BAL-NONE / BAL-CUSTOM / BAL-VAR | 第 5.1、5.2 节定义的训练期路由均衡；不改变推理前向 |

**SEL-CONTENT**、**SEL-PRE** 和 **SEL-POST** 分别表示 \(\operatorname{Read}^{\mathrm{sel}}\) 只读取 receiver 当前本地消息、额外读取旧状态或额外读取更新后状态。H1 的 \(\operatorname{Score}\) 还读取由共享 parent 产生的公共 \(\mu\)；HB-Lattice region 不要求存在单一公共 \(\mu\)，只联合处理 reached nodes 的轻量读出和 presence 信息。名称不限定打分采用线性层、MLP 或其他实现；精确读出、打分公式以及状态中是否包含历史激活记录仍由 manifest 和实验设置保存。

如果历史激活记录会影响 selector 或输出，它就是模型前向语义的一部分，不能隐藏在同一个纯 EMA/GDN 条件名下。具体实现确定后，应在 **STATE** 中增加明确的复合状态标签；记录维度、衰减、写回规则等细节再放入 manifest。

**EMIT**、**PAGG** 和 **AGG** 分别表示 sender Emit、多父入口聚合和共享 parent 的显式 fork-join MIX，不能混用。GraphBranch 与 backbone 的 RESIDUAL_ADD 已由 placement 固定。若同一实验内部的对应 policy 不统一，则使用各自的 **VAR**，并在 manifest 中列出完整设置；逐 region 的 active 数量由执行配置中的 \(K^{\max}\) 记录，不由 **AGG** 代替。

**BAL-AVAIL-SOFT** 在 H1 中退化为第 5.1 节的固定候选均衡，在 HB-Lattice 中使用第 5.2 节的 availability 基准。**BAL-CUSTOM** 和 **BAL-VAR** 必须附完整公式与聚合范围。

TRAIN 的含义必须严格区分：

- **PT**：随机初始化后做自回归预训练；
- **CPT**：加载预训练 checkpoint，继续做语言模型目标训练；
- **FT**：加载预训练 checkpoint，使用不同于基础自回归预训练的下游任务目标；
- **SFT**：FT 中特指有监督的指令或输入输出微调。

TRAIN 描述 base 权重与训练目标；新增 GraphBranch 及其 receiver nodes 的初始化方式由实验设置单独记录。

口语中的“finetune”不能直接写入正式名称：如果实际仍是 FineWeb 或领域语料上的自回归语言模型训练，应记为 CPT；只有训练目标确实改变时才记为 FT 或 SFT。

### 6.2 R、I、H 与 K 不得混用

- **R8** 只表示每个局部 group 或非平凡 selector region 有 8 个候选，不表示模型共有 8 个 receivers。
- **I8** 表示每个 Token 顺序经过 8 个插入位置，不表示 Transformer 只有 8 个 blocks。
- **H2** 表示一个插入位置内部的最大 receiver node 深度为 2，不表示模型中有两个插入位置，也不能唯一确定拓扑。
- HB-Lattice 的 **K2** 表示每个非平凡 region 最多激活两个 reached nodes，**KALL** 表示全部 reached nodes 都 active；不同 regions 不统一时使用 **KVAR**。
- H1 的 **AGG-K2** 表示对应共享-parent group 激活两个候选，不表示该 group 只有两个候选。

receiver node 内部串行的状态/上下文 residual 与 FFN residual 合计仍算一层；只有该 node 的完整输出继续进入下一层 receiver node 时，H 才增加。

例如 **R4-I8-H1** 表示 8 个顺序插入位置，每处只有一层局部 receiver group，每个 group 有 4 个候选。它不是 8 层递归。

如果不同插入位置、Line 或非平凡 selector region 采用不同宽度，短名字中使用 **RVAR**，并在 manifest 和报告中列出完整宽度；forced-active 的 singleton entry/sink 不参与 R 的摘要。H>1、平台期、多父边或镜像直通不能只靠 R/H 推断，必须同时给出 **TOPO_ID** 和完整 Plan。

### 6.3 具体 run 实例名

科学条件之外，真实 run 还需要模型、seed 和尝试编号：

~~~text
<MODEL>-<scientific-condition>-s<SEED>-r<ATTEMPT>
~~~

例如：

~~~text
q3-06b-cpt-parmlp-bo-r8-i4-h1-ema128-sel-post-agg-t1-hst-bal-avail-soft-s42-r1
~~~

模型 checkpoint、数据 revision、精确 block 编号、Token 预算、学习率、dtype、设备和代码 commit 仍由 manifest 保存，不强行塞进短名字。名称是可读索引，不代替完整实验设置。

### 6.4 基线名称

Dense 与标准 MoE 不使用 TIDE placement/profile 字段：

~~~text
PT-DENSE
CPT-DENSE
PT-MOE-R8-I4
CPT-MOE-R8-I4
~~~

MOE 的精确插入 block、Top-K、capacity、token-drop、shared expert 和路由辅助项必须在完整设置中声明。

## 7. 名称之外仍必须明确的语义

即使规范名称相同，每个正式设置仍要明确记录：

- 精确插入 block 编号；
- 不同 sites、Lines 和节点之间是否共享参数；默认不共享；
- 完整 `HBLatticePlan`、规范化哈希、`HBLatticeExecutionConfig`，以及 TopologyBuilder 的名称、版本和配置；
- 每条 Line 的 phase、节点与 region 划分，每条边的端点和 tree/local/shortcut/mirror 类别，以及逐节点镜像直通开关；
- 最大 fan-in/fan-out、region 大小与 forced-active 节点；
- 逐 region 的 \(K^{\max}\) 及其 SelectionDecision 规则；
- `ParentAggregate`、parent-presence 与任何 `ParentMergeScore` 的精确公式；
- `EmitPolicy` 的精确公式，以及 EMIT-HST 的 \(\zeta^{\mathrm{ST}}\)；
- 显式 fork-join 的 `ActiveBranchAggregate` policy 与权重；
- \(N_{\mathrm{sel}}\)、\(N_{R,i}\) 与 \(N_{F,i}\) 的精确实现和初始化；
- \(\operatorname{Read}^{\mathrm{sel}}\)、\(\operatorname{Read}^{\mathrm{ffn}}\) 与 \(\operatorname{Score}\) 的精确公式、输出维度以及是否包含历史激活记录；
- Observe、selector、状态提交和历史激活写回的精确顺序，以及写入 \(p\) 时是否 stop-gradient；
- GraphBranch 与 backbone 的 RESIDUAL_ADD 公式以及任何额外缩放；
- `BalancePolicy`、各辅助 loss 的公式、系数与 reduction；
- 状态初始化、有效 Token mask、跨 chunk 的 carry/reset 与梯度 detach 规则；
- 辅助 loss 的 Token 范围、site/Line/region 聚合范围、reached mask 处理以及是否跨 micro-batch 或设备统计；
- reached、Observe、active、Emit、soft mass 与 hard share 等诊断量的分母和聚合范围；
- 每个 Token 实际执行多少次本地入口归一化、轻量 selector 读出、Observe / Update、较大状态读出和昂贵 FFN；
- 初始化怎样保持或改变 base 函数；
- MOE 是否有 expert capacity、token drop 或 reroute。

这些项目不会全部进入短名字，但它们决定两个 run 是否真的是 matched comparison、到底哪些地方是 matched。
