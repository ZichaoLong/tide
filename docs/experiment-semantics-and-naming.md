# TIDE 实验语义、命名与数学符号

> 本文从更上层研究计划继承 **[TIDE](https://github.com/ZichaoLong/ObsidianVault.git)** 这个名字，其余内容均可独立阅读。
>
> 本文描述的是新实验的目标语义，只定义“模型实际怎样计算”和“实验名称怎样反映计算图”。
>
> 实验晋级、结果报告组织和 checkpoint 保留策略另行讨论。
>
## 阅读入口：先看完整图景

我们希望设计一套 TIDE Graph 的语义服务于三个要求：**固定空间拓扑**（底层节点和边固定，但每个 Token 的 active 子图可以变化）、**单节点成本有界**（每个节点的参数、状态、连接和工作量有上界）以及**可达容量增长**（扩容后仍能沿这些固定局部连接到达更多节点）。

为此，本文档定义一种 Single-Settlement Graph，下文简称 **SettleGraph**。它是一个具备单输入、单输出和固定空间拓扑的计算图：
- 对每个 Token，SettleGraph 接受一个输入 hidden，沿内部固定边传播 hidden，在固定的局部候选 receiver nodes 中选择少量节点做昂贵计算，再把结果送往下游并最终输出一个与输入 hidden 同维度的张量。
- SettleGraph 的每个节点称为 **receiver node**。它聚合实际到达的入口或父消息，是否更新状态由 propagation profile 决定，是否执行完整计算并向下游发送由局部 region selector 决定。节点内部设计的典型例子是 Transformer block，具体定义见 2.2 节。
SettleGraph 的更详细定义见下文。

因此，SettleGraph 可以多种方式接入各类由 blocks 串连组成的 Base LLM 模型。

Base 模型接入 SettleGraph 后，Base 模型部分可原样复用已训练 checkpoint，SettleGraph 部分可进行适当的初始化。我们还额外要求，SettleGraph 在某些初始化下，接入后的总模型与 Base 模型函数等价，这时可避免研究起步时，待验证点过多、过于不成熟，永远无法获得有效正面验证结果。

## 1. Base block 与 SettleGraph 顶层边界

### 1.1 Base Qwen3 block
第 \(\ell\) 个 block 接收到的第 \(0,...,t\) 个 token 的输入 hidden 是 \(t+1\) 个 \(d_{\mathrm{model}}\) 维向量，记作
$$
X_{\ell,\le t}
:=(x_{\ell,0},x_{\ell,1},\ldots,x_{\ell,t}).
$$

则对位置 \(t\)，一个原始 Pre-Norm Qwen3 block 计算：

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

其中， \(N_A\) 和 \(N_F\) 分别表示两个位置的归一化操作；当前 Qwen3 base block 中二者都实现为 RMSNorm。\(A_\ell\) 表示 causal self-attention，\(F_\ell\) 表示原 dense SwiGLU MLP。

### 1.2 SettleGraph 的单入口、单出口契约
在 Base 模型中第 \(\ell(j)\) 个 block 插入的第 \(j\) 个 SettleGraph，记作 \(\mathcal G_j\)，下文也把这个插入位置称为 site \(j\)。其中，每个 block 位置最多插入一个 SettleGraph。

对每个 Token，\(\mathcal G_j\) 可作为一个有逐序列持久状态的函数接受一个输入 hidden \(h^{\mathrm{in}}_{j,t}\)，并始终对外返回一个同维 hidden \(b_{\mathcal G,j,t}\)：
$$
b_{\mathcal G,j,t}
=\mathcal G_j\!\left(h^{\mathrm{in}}_{j,t}\right),
$$
另外记 \(\mathcal G_j\) 输出值的 residual 为：
$$
\Delta_{\mathcal G,j,t}
=b_{\mathcal G,j,t}-h^{\mathrm{in}}_{j,t}.
$$

记 block \(\ell(j)\) 的最终输出，也就是下一个 block \(\ell+1\) 的输入为 \(y_{\ell,t}\triangleq x_{\ell+1,t}\)，则 \(y_{\ell,t}\) 同时取决于 SettleGraph 的 placement 配置及其输出 hidden \(b_{\mathcal G,j,t}\)。对于未插入 SettleGraph 的情形有
$$
y_{\ell,t}=v_{\ell,t}.
$$

下面给出不同的 placement 配置如何起作用的说明。

### 1.3 SettleGraph 的四种 placement

SettleGraph 有四种 **placement** 配置，它们分别对应
- 不同的接入位置：因此 \(\mathcal G_j\) 将获得不同的输入 hidden \(h^{\mathrm{in}}\)
- 不同的合入位置：因此将与 \(u_{\ell,t}\) 或 \(v_{\ell,t}\) 合并 residual，并最终影响 \(y_{\ell,t}\)

#### 1.3.1 POST：完整 block 后串联

$$
h^{\mathrm{in}}_{j,t}=v_{\ell,t},
\qquad
y_{\ell,t}
=v_{\ell,t}+\Delta_{\mathcal G,j,t}
=b_{\mathcal G,j,t}.
$$

~~~text
x → Attention → u → 原 dense MLP → v → SettleGraph → y
~~~

SettleGraph 能看到当前 block 的 Attention 和原 MLP 结果。POST 是串联结构。

#### 1.3.2 PARBLK：与完整 block 并列

SettleGraph 和完整 base block 都从 \(x_{\ell,t}\) 开始，最后在 block 出口合并：

$$
h^{\mathrm{in}}_{j,t}=x_{\ell,t},
\qquad
y_{\ell,t}
=v_{\ell,t}+\Delta_{\mathcal G,j,t}.
$$

~~~text
          ┌→ 完整 base block → v ─────┐
x ────────┤                            + → y
          └→ SettleGraph → Δ_G(x) ────┘
~~~

SettleGraph 看不到当前 block 的 Attention 或 MLP 结果，也不改变它们的输入；两条路径可以并行执行。

#### 1.3.3 PARATTN：与 Attention 并列

SettleGraph 与 Attention 都读取 \(x_{\ell,t}\)。先在 Attention residual 位置合并，再让原 dense MLP 读取合并后的表示：

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
          └→ SettleGraph ────┘
~~~

PARATTN 只说明 SettleGraph residual 的接入位置，不限制 SettleGraph 内部只能使用 Attention。

#### 1.3.4 PARMLP：与 MLP 并列

Attention residual 先得到 \(u_{\ell,t}\)；原 dense MLP 与 SettleGraph 都读取这个共同输入，最后在 MLP residual 位置合并：

$$
h^{\mathrm{in}}_{j,t}=u_{\ell,t},
\qquad
y_{\ell,t}
=v_{\ell,t}+\Delta_{\mathcal G,j,t}.
$$

~~~text
x → self-attention → u
                      ├→ 原 dense MLP ─┐
                      └→ SettleGraph ── + → y
~~~

SettleGraph 能看到当前 Attention 的结果，但看不到当前原 MLP 的结果，也不改变原 MLP 的输入。本文统一使用 **PARMLP**；**PARFFN** 指同一个 placement。原 dense MLP 是 always-on 路径，SettleGraph 是与它并列的稀疏、可有状态主旁路。

#### 1.3.5 直接比较与初始化

| Placement | \(h^{\mathrm{in}}\) | always-on 输出 | 看见当前 Attention | 看见当前原 MLP | 改变原 MLP 输入 | SettleGraph merge 后 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| **POST** | \(v\) | \(v\) | 是 | 是 | 否 | 直接得到 \(y\) |
| **PARBLK** | \(x\) | \(v\) | 否 | 否 | 否 | 直接得到 \(y\) |
| **PARATTN** | \(x\) | \(u\) | 否 | 否 | 是 | 得到 \(u'\)，再执行原 MLP |
| **PARMLP** | \(u\) | \(v\) | 是 | 否 | 否 | 直接得到 \(y\) |

若 SettleGraph 可经过某种初始化变成 identity 函数，则 \(b_{\mathcal G}=h^{\mathrm{in}}\)、\(\Delta_{\mathcal G}=0\)，此时接入 \(\mathcal G_j\) 后的模型与原始 Base 模型函数等价。对于后文中给出的多种 SettleGraph 内部构型，这种初始化不难实现。

## 2. 一个 Token 如何穿过 SettleGraph

### 2.1 固定图、边结算与输入聚合

#### 固定图

SettleGraph 的内部拓扑是一张有限固定 DAG：

$$
G=(V,E,\mathfrak R).
$$

其中：

- \(V\) 是 receiver nodes 的集合；下文 \(v\in V\) 表示 receiver ID，与第 1 节的 base hidden \(v_{\ell,t}\) 不是同一个量；
- \(E\subseteq V\times V\) 是固定有向边集合；
- \(\mathfrak R\) 是对 \(V\) 的固定 region 划分，每个 receiver 恰好属于一个 region，每个 region 配置一个控制 receivers 激活与否的 selector。由 DAG \((V,E)\) 自然诱导的关于 \(\mathfrak R\) 的图，也被要求是 DAG。

令 \(\operatorname{In}(v)\) 和 \(\operatorname{Out}(v)\) 分别表示 receiver \(v\) 的固定入边和出边。没有 receiver 父节点的 nodes 是**入口 receivers**：

$$
V_{\mathrm{in}}
=\{v\in V\mid \operatorname{In}(v)=\varnothing\};
$$

没有 receiver 子节点的 nodes 是**终端 receivers**：

$$
V_{\mathrm{out}}
=\{v\in V\mid \operatorname{Out}(v)=\varnothing\}.
$$

每个入口 receiver 都获得同一个图输入 \(h^{\mathrm{in}}_{j,t}\)，每个 receiver 都必须位于某条从入口 receiver 到终端 receiver 的固定路径上。单个 receiver 可以同时是入口和终端，这正是第 3 节单层实例在只有一个候选时的最小形式。

#### 每条边恰好结算一次

对同一个 Token \(t\)，每条固定边 \(e\in E\) 的最终结果记为

$$
z_{e,t}
\in
\{\operatorname{CLOSED}\}
\cup
\{\operatorname{DATA}(y)\mid y\in\mathbb R^{d_{\mathrm{model}}}\}.
$$

- \(\operatorname{DATA}(y)\) 表示该边实际携带 hidden \(y\)；
- \(\operatorname{CLOSED}\) 表示该边已经确认本 Token 不会再有数据。

\(\operatorname{CLOSED}\) 只是完成标记，不是 hidden，也不参加聚合。一条边一旦结算就不能改写或再次结算。先到的结果可以缓存，但 receiver 不能在全部固定父边结算前提前重复执行。

语义上，每条 \(\operatorname{DATA}\) 边都携带声明 dtype 下的完整 \(d_{\mathrm{model}}\) 维 hidden。物理实现可以打包、重排或分片传输，但必须无损恢复同一 Tensor；有损消息压缩不属于本文定义的 SettleGraph。

#### Receiver 如何得到一个输入 hidden

入口 receiver 的消息序列只包含图输入；其他 receiver 等全部固定父边结算后，只收集其中的 \(\operatorname{DATA}\)。消息按固定 edge ID 排列，并允许不同父边携带数值相同的 hidden：

$$
\mathcal M_{v,t}
=
\begin{cases}
\bigl(h^{\mathrm{in}}_{j,t}\bigr),
&v\in V_{\mathrm{in}},\\[4pt]
\bigl(y_e:\ e\in\operatorname{In}(v),\
z_{e,t}=\operatorname{DATA}(y_e)\bigr)_{\text{按 edge ID}},
&v\notin V_{\mathrm{in}}.
\end{cases}
$$

由此定义 receiver 是否 **reached**：

$$
q_{v,t}=\mathbf 1[\mathcal M_{v,t}\ne\varnothing].
$$

当 \(q_{v,t}=1\) 时，receiver 的唯一入口 hidden 为

$$
h_{v,t}
=\operatorname{Aggregate}_v(\mathcal M_{v,t}).
$$

当 \(q_{v,t}=0\) 时，不产生 \(h_{v,t}\)，该 receiver 不参加当前选择、状态更新或完整计算。这里的 \(\operatorname{Aggregate}_v\) 是 receiver 的输入操作，不是另一个拓扑节点，也不增加 receiver 深度。

默认的 `AGG-MEAN` 为

$$
\operatorname{Aggregate}_v(\mathcal M_{v,t})
=\frac{1}{|\mathcal M_{v,t}|}
\sum_{y\in\mathcal M_{v,t}}y.
$$

因此只有一条实际消息时，输入 hidden 就是该消息本身。可选聚合只需满足输出仍为一个 \(d_{\mathrm{model}}\) 维 hidden：

| 设置 | 定义 |
| --- | --- |
| `AGG-MEAN` | 对实际到达的消息取均值；当前默认 |
| `AGG-LEARNED` | \(\sum_k\alpha_{v,k,t}y_k\)，其中 \(\alpha_k\ge0\)、\(\sum_k\alpha_k=1\) |
| `AGG-CUSTOM` | 自定义聚合，例如可针对每条父边进行线性变换后再平均；须记录完整公式、输入顺序及额外参数成本 |

若聚合依赖父边身份，可以同时读取对应 edge ID；其参数和计算量计入 receiver 的输入成本。selector 概率不在聚合中再次相乘。

#### 一个最小示意

下面是一张最小运行时示意图。固定图中 \(a,b\) 属于同一个 region；本例 selector 选择 \(a\)，因此二者都收到图输入，但只有 \(a\) 继续发送。receiver \(c\) 等来自 \(a,b\) 的两条固定父边都结算后，忽略 \(b\) 的 \(\operatorname{CLOSED}\)，聚合实际收到的 \(\operatorname{DATA}(\widehat g_a)\)。本例把 \(c\) 放在 forced-active singleton region 中。

```mermaid
flowchart LR
    IN(["图输入 h_in"])

    subgraph R["region R"]
        direction TB
        A["receiver a<br/>reached + active"]
        B["receiver b<br/>reached + inactive"]
    end

    SEL{"selector R<br/>在 reached candidates 中选择"}
    C["receiver c<br/>singleton forced-active"]
    OUT(["聚合终端消息<br/>得到 b_G"])

    IN -->|"边界输入 h_in"| A
    IN -->|"边界输入 h_in"| B
    A -->|"DATA(g_hat_a)"| C
    B -->|"CLOSED"| C
    C -->|"终端输出 g_hat_c"| OUT

    A -.->|"轻量 Read^sel"| SEL
    B -.->|"轻量 Read^sel"| SEL
    SEL -.->|"active"| A
    SEL -.->|"inactive"| B
```

一次局部结算可以概括成四步：

1. **收齐**：等全部固定父边返回 \(\operatorname{DATA}\) 或 \(\operatorname{CLOSED}\)。
2. **选择**：聚合实际数据，确定 reached candidates，再由 region selector 选出 active nodes。
3. **计算**：按状态传播规则提交状态；active nodes 执行完整 receiver 计算。
4. **发送**：active node 的固定出边全部携带同一个输出，其他出边全部关闭。

selector 不是发散点。固定边决定消息可能去往哪里，selector 只决定哪些已经 reached 的 receivers 本次继续计算和发送。一个 receiver 向多个 children 发送只是固定边的 fan-out；多个 parents 到达同一 receiver 时，则由该 receiver 的输入聚合操作完成 fan-in。

### 2.2 Receiver：状态与昂贵计算

receiver 是图中唯一的拓扑计算节点。每个 receiver 持有自己的参数、可选私有状态和昂贵计算；参数默认不跨 node 或 site 共享。拓扑只依赖本节规定的输入、状态和输出契约，不依赖状态模块内部采用 EMA、Gated DeltaNet、Attention 还是其他算法。

对 reached receiver \(v\)，先对入口 hidden 做本地归一化：

$$
m_{v,t}=N_{R,v}(h_{v,t}).
$$

令 \(s^-_{v,t}\) 表示当前 Token 到来前的 receiver 状态。状态模块可以根据当前输入产生 proposal：

$$
\widetilde s_{v,t}
=\operatorname{Update}_v(s^-_{v,t},m_{v,t}).
$$

proposal 本身不等于状态已经更新；只有第 2.3 节定义的 Observe/commit 才会把它提交为当前计算可见的状态。无状态 receiver 的 \(s^-\)、\(\operatorname{Update}\) 和 commit 都是空操作。

receiver 向 selector 提供一个固定、有界的轻量读出。三种 selector 时序对应：

$$
r^{\mathrm{sel}}_{v,t,\mathrm{content}}
=\operatorname{Read}^{\mathrm{sel}}_v(m_{v,t}),
$$

$$
r^{\mathrm{sel}}_{v,t,\mathrm{pre}}
=\operatorname{Read}^{\mathrm{sel}}_v(s^-_{v,t},m_{v,t}),
$$

$$
r^{\mathrm{sel}}_{v,t,\mathrm{post}}
=\operatorname{Read}^{\mathrm{sel}}_v(\widetilde s_{v,t},m_{v,t}).
$$

\(\operatorname{Read}^{\mathrm{sel}}\) 可以是输出范数、历史激活统计或低维投影，不要求把完整 receiver 状态交给 selector。post-update 必须在选择前生成 proposal；content-only 和 pre-update 只需在选择后为实际 Observe 的 receivers 生成 proposal。

active receiver 使用已经提交、可供当前计算读取的状态 \(s^{\mathrm{cmp}}_{v,t}\)，执行一次完整计算：

$$
g_{v,t}
=\operatorname{NodeCompute}_v
\left(h_{v,t},m_{v,t},s^{\mathrm{cmp}}_{v,t}\right).
$$

当前默认的 \(\operatorname{NodeCompute}\) 采用 Pre-Norm 双 residual：

$$
r^{\mathrm{ffn}}_{v,t}
=\operatorname{Read}^{\mathrm{ffn}}_v
\left(s^{\mathrm{cmp}}_{v,t},m_{v,t}\right),
\qquad
u^{\mathrm{node}}_{v,t}
=h_{v,t}+r^{\mathrm{ffn}}_{v,t},
$$

$$
g_{v,t}
=u^{\mathrm{node}}_{v,t}
+E_v\!\left(N_{F,v}(u^{\mathrm{node}}_{v,t})\right).
$$

其中，\(\operatorname{Read}^{\mathrm{ffn}}\) 是 receiver 内部较大的状态/上下文读出，可以包含 normalization、Attention/SSM 和 output projection；\(E_v\) 是昂贵 FFN 或实验声明的等价计算。第一条 residual 的基底是未归一化的 \(h_{v,t}\)。无状态 receiver 默认令 \(r^{\mathrm{ffn}}_{v,t}=0\)。

\(g_{v,t}\) 是一个完整 hidden，而不是相对 \(h_{v,t}\) 的增量。receiver 不自行决定是否 active，也不在 \(\operatorname{NodeCompute}\) 内部乘 selector 概率；只有 active receiver 执行较大的 \(\operatorname{Read}^{\mathrm{ffn}}\) 和昂贵计算。

#### 内部设计速览

不同 receiver 复用上述接口，只替换状态更新与两类读出；状态形状和完整公式见附录 A。

| 样例 | \(\operatorname{Update}\) 或历史写回 | 轻量 \(\operatorname{Read}^{\mathrm{sel}}\) | active node 的 \(\operatorname{Read}^{\mathrm{ffn}}\) |
| --- | --- | --- | --- |
| 无状态 MLP | 无 | 当前 \(m\) 的低维投影或范数 | \(0\) |
| 历史激活 | 选择后写回激活次数、最近激活位置或 \(p\) 的移动平均 | 当前 \(m\) 与少量历史标量 | 只服务 selector 时为 \(0\) |
| EMA | \(\widetilde s=\lambda\odot s^-+(1-\lambda)\odot o(m)\) | 状态或 proposal 的低维摘要 | \(W^{\mathrm{out}}s^{\mathrm{cmp}}\) |
| GDN / KDA | 有界关联状态的 delta-rule 更新 | 状态或 query 读出的轻量摘要 | 投影后的关联读出 |
| SSM / Mamba | 有界递归状态更新 | 状态或递归输出的轻量摘要 | 投影后的 state-space 输出 |
| Attention | 写入完整、窗口或压缩 key/value 历史 | 历史统计或 Attention 读出摘要 | 投影后的 Attention 输出 |

content-only 始终只读当前 \(m\)；历史激活通常在 selector 决策后写回。无状态 MLP 令 \(\operatorname{Read}^{\mathrm{ffn}}=0\)，因此

$$
\operatorname{NodeCompute}_v(h_{v,t},m_{v,t},\varnothing)
=h_{v,t}+E_v\!\left(N_{F,v}(h_{v,t})\right).
$$

其他样例把各自的 \(\operatorname{Read}^{\mathrm{ffn}}\) 代入本节已经定义的双 residual；较大的读出仍只由 active nodes 执行。

### 2.3 Region：选择、Observe 与发送

每个 region \(\mathcal R\in\mathfrak R\) 对应一个 selector。region 中所有 receivers 的输入状态都已经确定后，它对当前 Token 只选择和结算一次；这要求每个成员要么已经 reached，要么已经确认所有父边关闭。候选集只包含前者：

$$
\mathcal C_{\mathcal R,t}
=\{v\in\mathcal R\mid q_{v,t}=1\}.
$$

#### Selector

selector 只接收 candidates 各自在本地产生的轻量 \(r^{\mathrm{sel}}\)。候选 nodes 及其读出始终按稳定 node ID 排列。令 \(c^{\mathrm{ctx}}_{\mathcal R,t}\) 表示可选的固定、有界局部公共摘要；没有时取空。一次局部打分得到：

$$
(a_{v,t})_{v\in\mathcal C_{\mathcal R,t}}
=\operatorname{Score}_{\mathcal R}
\left(
c^{\mathrm{ctx}}_{\mathcal R,t},
(r^{\mathrm{sel}}_{v,t,\tau})_{v\in\mathcal C_{\mathcal R,t}}
\right),
$$

$$
(p_{v,t})_{v\in\mathcal C_{\mathcal R,t}}
=\operatorname{softmax}
\left((a_{v,t})_{v\in\mathcal C_{\mathcal R,t}}\right).
$$

这里 \(\tau\in\{\mathrm{content},\mathrm{pre},\mathrm{post}\}\) 是 selector 时序，\(p_{v,t}\) 是 soft 选择概率，不是消息聚合权重。Top-K 得到 active set：

$$
\mathcal A_{\mathcal R,t}
=\operatorname{TopKIndex}
\left(
p,
\min(K^{\mathrm{req}}_{\mathcal R,t},|\mathcal C_{\mathcal R,t}|)
\right),
$$

其中候选非空时

$$
1\le K^{\mathrm{req}}_{\mathcal R,t}
\le K^{\max}_{\mathcal R};
$$

平票按固定 node ID 打破。候选为空时 \(\mathcal A_{\mathcal R,t}=\varnothing\)，该 region 不执行 Score。forced-active receiver 使用独立的 singleton region；它只要 reached 就令 \(p=1\) 并直接 active。

三种 selector 时序为：

| 时序 | selector 读取的信息 |
| --- | --- |
| **Content-only** | 当前本地输入的轻量读出 |
| **Pre-update state** | 当前输入与旧状态 \(s^-\) 的轻量读出 |
| **Post-update state** | 当前输入产生的 proposal \(\widetilde s\) 的轻量读出 |

Pre 与 post 不是包含关系：如果 \(\operatorname{Update}\) 会覆盖、压缩或遗忘旧状态，post readout 不一定能恢复 pre readout 的信息。\(c^{\mathrm{ctx}}\) 的信息时刻必须与 \(\tau\) 一致；content-only 只能使用当前内容，不能读取持久状态或历史激活。任何跨 region 公共摘要或控制输入都必须来自固定、有界的上游，并在图描述中声明依赖；selector 不读取未声明的全图信息。

#### Observe 与状态提交

propagation profile 决定哪些 reached receivers 提交当前内容；selector 决定哪些 receivers 执行完整计算：

| Profile | Observe/commit 范围 | 完整计算与发送范围 | 标准 selector 时序 |
| --- | --- | --- | --- |
| **N（stateless）** | 无状态 | active | content-only |
| **SD（selected-dispatch）** | active | active | content-only、pre-update |
| **BO（broadcast-observe）** | 全部 reached | active | content-only、pre-update、post-update |

令 Observe set 为

$$
\mathcal O_{\mathcal R,t}
=
\begin{cases}
\varnothing,&\text{N},\\
\mathcal A_{\mathcal R,t},&\text{SD},\\
\mathcal C_{\mathcal R,t},&\text{BO}.
\end{cases}
$$

对有状态 receiver，commit 后供当前 \(\operatorname{NodeCompute}\) 使用的状态为

$$
s^{\mathrm{cmp}}_{v,t}
=
\begin{cases}
\widetilde s_{v,t},&v\in\mathcal O_{\mathcal R,t},\\
s^-_{v,t},&v\notin\mathcal O_{\mathcal R,t}.
\end{cases}
$$

content-only 和 pre-update 先选择，再只为 \(\mathcal O\) 中的 receivers 生成 proposal 并 commit；标准 post-update + BO 则先为全部 reached receivers 生成 proposal，再选择并全部 commit。pre-update 只限制 selector 读取旧状态，active receiver 的默认完整计算仍读取本 Token 已提交的 \(s^{\mathrm{cmp}}\)。post-update 中 proposal 到 selector 的计算图默认保留，任何 detach 或 stop-gradient 都必须记录。其他组合必须标为自定义 profile，并明确 proposal、read、selection 和 commit 的顺序。

#### 发送

active receiver 得到 \(g_{v,t}\) 后，发送规则产生实际消息：

$$
\widehat g_{v,t}
=\operatorname{Emit}_v(h_{v,t},g_{v,t},p_{v,t}).
$$

同一个 \(\widehat g_{v,t}\) 被复制到该 receiver 的全部固定出边。当前推荐的 `EMIT-HST`（Hard Straight-Through）为

$$
\rho_{v,t}
=1+\zeta^{\mathrm{ST}}_v
\bigl(p_{v,t}-\operatorname{sg}(p_{v,t})\bigr),
$$

$$
\widehat g_{v,t}
=h_{v,t}+\rho_{v,t}(g_{v,t}-h_{v,t}).
$$

\(\operatorname{sg}\) 是 stop-gradient，\(\zeta^{\mathrm{ST}}_v\) 是固定梯度缩放常数，默认取 1。前向恒有 \(\widehat g=g\)，反向则允许主任务梯度经 \(p\) 返回 selector；离散 Top-K 本身不求导。`EMIT-HARD` 直接令 \(\widehat g=g\) 且不提供这条梯度；`EMIT-SOFTP` 令 \(\widehat g=h+p(g-h)\)，会同时改变前向强度，使用时必须单独标记。

active receiver 的每条固定出边结算为 \(\operatorname{DATA}(\widehat g_{v,t})\)；inactive 或未 reached receiver 的每条固定出边结算为 \(\operatorname{CLOSED}\)。训练期 balance loss 只提供辅助梯度，不改变这套推理数据流。

### 2.4 合法图与通用结算算法

#### 哪些固定图可以执行

一张合法 SettleGraph 至少满足：

1. receiver 图有限、无环、没有重复平行边，每个 receiver 位于某条入口—终端固定路径上；
2. 每个 receiver 恰好属于一个 region，同一 region 内不存在 receiver-to-receiver 边；
3. 将每个 region 收缩成一个点，并加入全部已声明的跨 region 控制依赖后，所得 region 依赖图仍然无环；
4. hidden、状态、读出和参数的 shape/dtype 已确定，所有聚合、receiver、selector、profile 和发送规则均已完整定义；
5. 固定 fan-in、fan-out、region 大小、入口/终端 receiver 数，以及单节点参数、状态和计算成本满足实验声明的上界。

第三条保证 region 不会为了执行 selector 而相互等待。例如，只要存在 receiver 边 \(u\to v\)，就必须满足

$$
\lambda(\mathcal R(u))
<\lambda(\mathcal R(v)),
$$

其中 \(\mathcal R(v)\) 是 receiver \(v\) 所属 region，\(\lambda\) 是 region 依赖图的某个拓扑序。共享只读参数不会产生控制依赖；任何跨 region 共享可变状态或 selector 输入都必须声明读写顺序，并一起通过无环检查。

原始图描述经过这些静态校验和规范化后得到的静态记录称为 **Plan**，记为 \(\Pi\)。Plan 包含 receivers、固定边、regions、稳定 ID、各项运算、Tensor/状态契约和依赖顺序；它不包含某个 Token 的 reached、active 或边结算结果，也不是人工编写的动态执行步骤。

#### 一个解释器执行所有合法 Plan

令 \(\Theta\) 表示 Plan 绑定的参数与 Tensor 操作，\(S^-_t\) 表示当前 Token 前的全部 receiver state 和 selector-history。单 Token 执行统一写成

$$
(b_{\mathcal G,j,t},S_t)
=\operatorname{Interpret}
(\Pi,\Theta,S^-_t,h^{\mathrm{in}}_{j,t}).
$$

同一个解释器可以执行任意合法 Plan，不需要枚举入口—终端路径：

~~~text
InterpretToken(Plan, states_before, h_in):
  把每条固定边初始化为“未结算”
  入口 receivers 的消息序列初始化为 [h_in]

  只要仍有未结算 region：
    选取一个数据依赖、控制依赖和跨 Token 状态依赖都已满足的 region R

    对每个 v ∈ R：
      若 v 不是入口 receiver，断言全部固定父边已经结算
      按 edge ID 收集父边中的全部 DATA；CLOSED 不进入消息序列
      消息非空则 reached，并执行 Aggregate 与入口归一化

    按第 2.3 节为 R 完成一次：
      Read^sel / Score / Top-K
      Observe commit / NodeCompute / Emit

    对每个 v ∈ R 的每条固定出边：
      v active  ⇒ 结算为 DATA(g_hat[v])
      其他情况 ⇒ 结算为 CLOSED

  等所有终端 receivers 完成本 Token 的角色结算
  收集 active 终端 receivers 的 g_hat
  若集合为空，报告配置失败
  否则用 Aggregate_out 聚合为 b_G，并返回最终状态
~~~

图输出的数学定义为

$$
\mathcal M_{\mathrm{out},t}
=\bigl(\widehat g_{v,t}:
v\in V_{\mathrm{out}},\ v\text{ active}\bigr)_{\text{按 node ID}},
$$

$$
b_{\mathcal G,j,t}
=\operatorname{Aggregate}_{\mathrm{out}}
(\mathcal M_{\mathrm{out},t}).
$$

\(\operatorname{Aggregate}_{\mathrm{out}}\) 遵守第 2.1 节相同的聚合契约。有效 Token 上 \(\mathcal M_{\mathrm{out},t}\) 必须非空；否则动态可达性保证失效，该 run 记为配置失败，不能静默回退或伪造 hidden。

每个 region 对每个 Token 只结算一次，每个 receiver 最多执行一次完整计算，每条固定边恰好返回一次 \(\operatorname{DATA}\) 或 \(\operatorname{CLOSED}\)。“单次结算”不表示每个 receiver 都会 active。独立、同时 ready 的 regions 可以串行或并行结算；只要满足声明的依赖并使用确定的聚合顺序，结果相同。

不等长路径不需要另一套算法：短路径结果先缓存，receiver 和图输出继续等待其余固定父结果；未激活路径用 \(\operatorname{CLOSED}\) 完成等待。模型语义不依赖 wall-clock timeout。拓扑若另外记录逻辑层级或边延迟，它只用于调度与 profile；除非某项运算明确把它作为输入，否则不改变上述 Tensor 结果。

第 3 节的单层实例是所有 receivers 同时属于 \(V_{\mathrm{in}}\cap V_{\mathrm{out}}\) 的最小规则图。第 4 节的 HB-Lattice 则用规则化生成器产生 \(V\)、\(E\) 和 \(\mathfrak R\)，并附加适合批处理的 Line/phase 信息；它仍遵守同一个结算算法。

### 2.5 跨 Token 状态

令 \(\mathrm{sid}\) 表示一条稳定序列的标识。receiver 状态按 \((\mathrm{site},\mathrm{receiver},\mathrm{sid})\) 隔离，selector-history 若存在，则按 \((\mathrm{site},\mathrm{region},\mathrm{sid})\) 或声明的 node-level 键隔离。默认不跨 site、receiver 或稳定序列共享可变状态。

对同一个状态键，有效 Token 必须按全局 \(t=0,1,\ldots\) 的因果顺序执行。Token \(t\) 开始时读取 \(s^-_{v,t}\)，按第 2.3 节完成 proposal、selection 和 commit；若还要写入本次 active、\(p\) 等历史统计，则在完整计算后写回，形成最终状态 \(s_{v,t}\)，并从下一有效 Token 起可见。写入 active 或 \(p\) 的历史默认 stop-gradient：

$$
s^-_{v,t+1}=s_{v,t}.
$$

未 reached receiver 以“不更新状态”的空操作完成自己的因果位置。物理调度即使提前得到 Token \(t+1\) 的父边结果，也必须等同一状态键的 Token \(t\) 完成后，才能读取或提交 \(t+1\) 的状态；提前结果只保存在对应 Token 的缓存中。

每条独立序列从声明的首状态开始：EMA、Gated DeltaNet/KDA 和 SSM 通常置零，Attention 历史为空，历史激活统计清零；使用可学习或其他首状态时必须记录。padding 等无效 Token 不进入 SettleGraph，图直接返回入口 hidden，因此 \(\Delta_{\mathcal G}=0\)，也不更新状态或产生路由 loss。

chunk 是一次前向接收的连续 Token 片段；prefill 是一次处理一段已有 Token，decode 是逐 Token 生成。状态值跨 chunk 保留，chunk 边界默认 detach：只截断 chunk 之间的梯度，chunk 内仍保留因果梯度。在 deterministic/eval（或固定随机掩码）且聚合顺序相同的条件下，同一有效前缀的整段 prefill、分块 prefill 和逐 Token decode 应得到相同的逐 Token 输出与最终状态。

写作 \(s_{t-1}\) 时，只表示同一稳定序列上一个有效 Token 结算后的状态；\(t\) 是跨 chunk 不重置的全局 Token 索引。具体状态模块见附录 A。

## 3. 最小实例：单层并列 receivers

本节把第 2 节的公共语义放入一个最小 Plan，展示固定拓扑、selector、receiver 和图输出怎样共同工作。它不引入新的组件或执行规则。

### 3.1 展开的 Plan

令 \(R\ge1\) 为并列 receiver 数量，并定义

$$
V=V_{\mathrm{in}}=V_{\mathrm{out}}=\{0,1,\ldots,R-1\},
\qquad
E=\varnothing,
\qquad
\mathfrak R=\{V\}.
$$

所有 receivers 同时是入口和终端，属于同一个 region；它们之间没有 receiver-to-receiver 边。图边界把同一个 \(h^{\mathrm{in}}_{j,t}\) 交给全部 receivers，selector 再从这 \(R\) 个 reached candidates 中选择少量 active nodes：

~~~text
图输入 h_in
  ├→ receiver 0 ─┐
  ├→ receiver 1 ─┤
  ├→ ...         ├→ Aggregate_out → b_G
  └→ receiver R-1┘
       一个 selector 在 R 个 candidates 中选择
~~~

“单层”表示任一入口—终端路径只经过一个 receiver node，不表示图中只有一个 node，也不限制 receiver 内部只能有一个计算子层。公共 selector 摘要若存在，直接使用第 2.3 节的 \(c^{\mathrm{ctx}}\)；它是可选配置，不属于该拓扑的必要组成。

### 3.2 一个 Token 的完整结算

固定一个 site \(j\)，并省略唯一 region 的下标。默认聚合对单条消息原样返回，因此对每个 \(v\in V\)，

$$
h_{v,t}=h^{\mathrm{in}}_{j,t},
\qquad
q_{v,t}=1,
\qquad
\mathcal C_t=V.
$$

selector 按第 2.3 节得到 \((p_{v,t})_{v\in V}\)；将该 region 的请求激活数简写为 \(K^{\mathrm{req}}_t\)，则

$$
\mathcal A_t
=\operatorname{TopKIndex}
\left((p_{v,t})_{v\in V},\min(K^{\mathrm{req}}_t,R)\right).
$$

三种 propagation profile 在本例中的 Observe 集为

$$
\mathcal O_t=
\begin{cases}
\varnothing,&\mathrm{N},\\
\mathcal A_t,&\mathrm{SD},\\
V,&\mathrm{BO}.
\end{cases}
$$

每个 \(v\in\mathcal A_t\) 按第 2.2、2.3 节完成状态提交、\(\operatorname{NodeCompute}\) 和发送。图输出为

$$
b_{\mathcal G,j,t}
=\operatorname{Aggregate}_{\mathrm{out}}
\left(
(\widehat g_{v,t}:v\in\mathcal A_t)_{\text{按 node ID}}
\right).
$$

Top-1 时只有一条终端消息；Top-K 使用 `AGG-MEAN` 时，聚合的是 active receivers 的完整输出，不再额外乘 selector 概率。因为所有 receivers 始终 reached，本例的 balance loss 使用第 6.2 节给出的固定候选退化形式。

### 3.3 实验作用与边界

单层实例适合分别验证 receiver 状态、content/pre/post selector 时序、N/SD/BO、Top-K、发送规则和输出聚合。它也便于与平铺 MoE 对照。

它不是固定局部度有界的容量扩展方案：增大 \(R\) 会同时扩大 selector region、入口宽度和终端宽度。验证沿固定局部连接到达更多容量，需要第 2.4 节的非平凡 Plan；第 4 节的 HB-Lattice 是其中一种规则化形态。

## 4. HB-Lattice：多层固定波前

HB-Lattice 是第 2 节合法 Plan 的一个规则化子集。它不改变 receiver、selector、聚合或发送语义，只增加有序 Lines、Line barrier、逻辑边延迟以及生成这种 Plan 的规则。

第 4.1—4.4 节定义 HB-Lattice 的规范语义；第 4.5 节只给出候选拓扑 Builder，不规定默认连接。

### 4.1 先看完整波前

一个小型 HB-Lattice 可以采用“扩展—平台—收拢”的宽度序列：

~~~text
图输入
   ↓
L0:  root                         扩展起点
L1:  0, 1                         扩展
L2:  00, 01, 10, 11               平台首层
L3:  00, 01, 10, 11               平台混合
L4:  0, 1                         收拢
L5:  root                          收拢终点
   ↓
Aggregate_out → b_G
~~~

相邻扩展、平台和收拢 Lines 之间由固定局部边连接；还可以配置从较浅扩展节点直达更深收拢节点的 mirror 边。每个 Line 划分为一个或多个 regions，同一 Token 结算完整个 Line 后才进入下一 Line。扩展、平台和收拢只是宽度形状及拓扑生成标签，不改变 node 的计算公式。

例如，tree 边可以包含 \(L_0:\mathrm{root}\to L_1:0,1\) 和 \(L_1:0\to L_2:00,01\)，平台边可以包含 \(L_2:00\to L_3:00,01\)，mirror 边可以包含 \(L_1:0\to L_4:0\)。这些只用于展示边的角色；实际实验必须在展开 Plan 中列出全部确切端点。

### 4.2 HB Plan 的形式约束

HB-Lattice 把 receiver 集合静态划分到 \(D+1\) 个有序 Lines：

$$
V=\bigsqcup_{d=0}^{D}L_d,
\qquad
\operatorname{level}(v)=d\iff v\in L_d.
$$

除第 2.4 节的一般合法性条件外，HB Plan 还必须满足：

1. 每个 region 完全位于唯一一个 Line；同一 Line 内没有 receiver 边，也没有 region 间控制依赖。
2. 所有边都严格指向更深的 Line：
   $$
   (u,v)\in E
   \Longrightarrow
   \operatorname{level}(u)<\operatorname{level}(v).
   $$
3. 图边界固定为 \(V_{\mathrm{in}}=L_0\)、\(V_{\mathrm{out}}=L_D\)。
4. Line \(L_d\) 只有在同一 Token 的 \(L_0,\ldots,L_{d-1}\) 全部结算后才开始；同一 Line 内相互独立的 regions 可以并行结算。

selector 的公共上下文若存在，只能来自固定常量或更浅 Line 已结算的有界读出，并作为控制依赖写入 Plan。跨多个 Lines 的消息在产生后缓存，到目标 Line 开始时再参与该 receiver 唯一一次输入聚合。

边 \((u,v)\) 的逻辑延迟定义为

$$
\delta(u,v)
=\operatorname{level}(v)-\operatorname{level}(u)>0.
$$

因此，对任意从 \(v_0\in L_0\) 到 \(v_n\in L_D\) 的完整路径 \(P=(v_0,\ldots,v_n)\)，

$$
\sum_{k=1}^{n}\delta(v_{k-1},v_k)
=\operatorname{level}(v_n)-\operatorname{level}(v_0)
=D.
$$

这保证不同长度的固定路径在同一个逻辑波前终点结算。\(t\) 始终表示序列中的 Token 位置，\(d\) 才是图内的逻辑波前位置；两者不是同一个时间轴。

site \(j\) 的 Line \(d\) 中第 \(r\) 个 region 记作 \(\mathcal R_{j,d,r}\)，其稳定 region ID 可以编码 \((d,r)\)。补回 batch/序列下标 \(b\) 后，其当前候选集仍是第 2.3 节的 reached nodes：

$$
\mathcal C_{j,d,r,b,t}
=\{v\in\mathcal R_{j,d,r}\mid q_{v,b,t}=1\}.
$$

HB Plan 的静态检查只需在第 2.4 节基础上增加 Line 唯一归属、region 不跨 Line、边严格向深层、边界位于首尾 Line，以及所有边类合计后的 fan-in/fan-out 上界。有效 Token 仍须得到非空终端消息；若 Builder 依靠 forced-active backbone 保证这一点，完整路径必须出现在展开 Plan 中。

### 4.3 一个 Token 如何逐 Line 结算

第 2.4 节的通用解释器可以直接执行 HB Plan；HB 的 Line barrier 给出一个更规则的合法调度：

~~~text
对一个 Token：
  for d = 0 ... D:
    等待发往 L_d 的全部固定父边结算
    对 L_d 的各 regions，按第 2 节完成：
      输入聚合 → selector → Observe/commit → NodeCompute → Emit
    等 L_d 的全部 regions 与出边完成结算

  按第 2.4 节聚合 L_D 的终端消息并返回最终状态
~~~

fan-out 仍是把 active node 的同一输出复制到固定 children，fan-in 仍由目标 receiver 的 \(\operatorname{Aggregate}\) 完成；HB-Lattice 不增加发散节点或汇合节点。未发送的固定边仍结算为 \(\operatorname{CLOSED}\)，跨 Line 的提前消息只做缓存，不会提前触发目标 receiver。

Line barrier 按 Token 生效，不要求整个 batch 同步停住。token-major、Line-major 或其他等价批处理都可以使用；有状态 receiver 和 selector-history 仍必须遵守第 2.5 节的逐序列跨 Token 因果顺序。不同调度只要保持这些依赖和确定性聚合顺序，就必须得到与第 2.4 节解释器相同的输出和最终状态。

### 4.4 Builder 与展开 Plan

拓扑 Builder 根据紧凑配置 \(\Gamma\) 产生完全展开的 HB Plan：

$$
\Pi_{\mathrm{HB}}
=\operatorname{Build}_{\mathrm{name},\mathrm{version}}(\Gamma).
$$

展开 Plan 必须列出 Lines 及其 phase、nodes、regions、固定边及其来源标签、稳定 ID、forced-active 设置和所有第 2.4 节要求的运算契约。执行语义只由展开 Plan 决定；Builder 名称和配置不能替代它。正式实验同时保存规范化 Plan 及其哈希，以及 Builder 的名称、版本和配置。

Builder 可以给边附加以下来源标签，用于生成、诊断和消融：

| 标签 | 生成含义 |
| --- | --- |
| tree | 扩展、收拢及其与平台首尾之间的结构边 |
| local | 相邻平台 Lines 之间的局部空间边 |
| shortcut | 相邻平台 Lines 之间的长程空间边 |
| mirror | 从较浅 Line 直达更深 Line 的直通边 |

这些标签不对应不同的执行操作；进入解释器后，它们都是第 2 节定义的普通固定边。所有标签的边都计入同一组入度、出度和消息成本上界。

### 4.5 两类候选拓扑 Builder

下面两类 Builder 只展示如何从紧凑规则生成 HB Plan，不规定默认连接。无论采用哪一类，实际实验都必须保存完整展开 Plan，并满足第 4.2 节。

#### 4.5.1 B 叉扩展与逐坐标平台混合

设分支因子 \(B\ge2\)、扩展深度 \(D_{\mathrm{up}}\ge1\)、额外平台 Line 数 \(P_{\mathrm{plat}}\ge0\)。最大宽度为 \(W_{\max}=B^{D_{\mathrm{up}}}\)，Line 宽度为

$$
1,B,\ldots,B^{D_{\mathrm{up}}},
\underbrace{B^{D_{\mathrm{up}}},\ldots,B^{D_{\mathrm{up}}}}_{P_{\mathrm{plat}}\text{ 个额外 Lines}},
B^{D_{\mathrm{up}}-1},\ldots,1,
$$

因此最后一个 Line 的下标为 \(D=2D_{\mathrm{up}}+P_{\mathrm{plat}}\)。峰值 Line 的 node 可以使用长度为 \(D_{\mathrm{up}}\) 的 base-\(B\) 空间坐标；扩展、平台 hop、收拢、region 划分和可选 mirror 映射都由 Builder 配置生成，并在展开 Plan 中列出确切端点。

例如 \(B=2,D_{\mathrm{up}}=2,P_{\mathrm{plat}}=2\) 时：

~~~text
L0:  root
L1:  0, 1
L2:  00, 01, 10, 11
L3:  00, 01, 10, 11
L4:  00, 01, 10, 11
L5:  0, 1
L6:  root
~~~

相同坐标出现在不同 Line 时仍表示不同 receiver，默认不共享参数或状态。tree、平台和 mirror 边的合计 fan-in/fan-out 必须保持为与宽度和平台长度无关的固定上界。

#### 4.5.2 统一空间图平台

第二类 Builder 沿用已声明的扩展与收拢映射，用一个有向空间图生成相邻平台 Lines 的连接。设平台坐标集合为 \(Q\)，\(G_{\mathrm{space}}=(Q,E_{\mathrm{space}})\)，则一个平台 hop 可以生成

$$
E_d
=\left\{
((d,\xi),(d+1,\xi'))
\mid(\xi,\xi')\in E_{\mathrm{space}}
\right\}.
$$

同一个 \(G_{\mathrm{space}}\) 可以在各 hop 重复，也可以逐 hop 指定不同的 \(E_d\)。空间图自身即使有环，逐 Line 展开后仍是 DAG。local 与 shortcut 可以同时存在，但每个坐标的合计入度、出度必须具有与 \(|Q|\) 无关的固定上界；长程连接不能随着宽度增长形成高入度枢纽。

## 5. 对照基线与可归因比较

本节定义后续实验需要区分的计算条件，不规定实验顺序，也不记录任何历史结果。所有 SettleGraph 条件都沿用第 1—4 节的同一套语义；对照实验只改变明确声明的坐标。

### 5.1 原生 Dense 与 Dense 扩展

原生 Dense 基线不插入 SettleGraph，直接使用第 1.1 节的 base block：

$$
y_{\ell,t}
=v_{\ell,t}
=u_{\ell,t}+F_\ell\!\left(N_F(u_{\ell,t})\right).
$$

它既是质量基线，也是 checkpoint 装载、prefill/decode、训练和保存恢复的正确性基线。

如果使用 Dense 扩展作为参数量对照，必须明确新增计算放在哪里、怎样初始化以及是否始终执行。Dense 扩展不经过 selector，其主要作用是区分“结构机制带来的收益”和“单纯增加参数或计算带来的收益”。

### 5.2 Flat MoE 基线

Flat MoE 在一个 block 内一次面对完整 expert 池，选择后立即回到共同 hidden。它不等同于 SettleGraph，但可以对照“潜在容量扩大、每 Token 只执行少量昂贵模块”这一基本能力。

设 site \(j\) 位于 block \(\ell(j)\)，有 \(R_j^{\mathrm{moe}}\) 个 experts，并令 \(1\le K_j^{\mathrm{moe}}\le R_j^{\mathrm{moe}}\)。每个 expert 把 \(d_{\mathrm{model}}\) 维输入映射回同维输出。router 权重

$$
W_j^{\mathrm{moe}}
\in
\mathbb R^{R_j^{\mathrm{moe}}\times d_{\mathrm{model}}}.
$$

router 输入、logits 和概率为

$$
m^{\mathrm{moe}}_{j,t}
=N_F(u_{\ell(j),t}),
$$

$$
(a^{\mathrm{moe}}_{j,t,i})_{i=0}^{R_j^{\mathrm{moe}}-1}
=W^{\mathrm{moe}}_j m^{\mathrm{moe}}_{j,t},
\qquad
p^{\mathrm{moe}}_{j,t}
=\operatorname{softmax}(a^{\mathrm{moe}}_{j,t}).
$$

候选按稳定 expert ID 排列，Top-K 平票也按该 ID 打破。令

$$
\mathcal A^{\mathrm{moe}}_{j,t}
=\operatorname{TopKIndex}
\left(
p^{\mathrm{moe}}_{j,t},
K_j^{\mathrm{moe}}
\right).
$$

选中 expert 的输出统一写成

$$
y_{\ell(j),t}
=u_{\ell(j),t}
+\sum_{i\in\mathcal A^{\mathrm{moe}}_{j,t}}
\alpha_{j,t,i}
E^{\mathrm{moe}}_{j,i}
\left(m^{\mathrm{moe}}_{j,t}\right).
$$

其中合并权重 \(\alpha\) 是 MoE 条件的一部分。常见选择包括：

| 合并方式 | \(\alpha_{j,t,i}\) | 说明 |
| --- | --- | --- |
| hard mean | \(1/\lvert\mathcal A^{\mathrm{moe}}_{j,t}\rvert\) | 前向不乘 soft 概率；Top-1 时为 1 |
| soft gate | \(p^{\mathrm{moe}}_{j,t,i}\) | 主任务梯度可经选中 gate 返回 router |
| normalized gate | \(p^{\mathrm{moe}}_{j,t,i}/\sum_{k\in\mathcal A^{\mathrm{moe}}_{j,t}}p^{\mathrm{moe}}_{j,t,k}\) | Top-K 内重新归一化；Top-1 时退化为 1 |

上式定义的是用 routed experts 替换原 dense MLP 的基线。若保留原 dense MLP 或其他 shared expert，必须显式写出额外路径和合并公式。expert 怎样初始化、是否设置 capacity、是否丢弃或 reroute 过载 Token，也必须作为实验条件明确记录，不能由“MoE”一词推断。MoE expert 没有第 2 节定义的 receiver 私有状态，也不使用 SettleGraph 的多跳边结算。

### 5.3 SettleGraph 内部的配对比较

SettleGraph 的核心比较应尽量一次只改变一个语义坐标：

| 要回答的问题 | 主要配对条件 |
| --- | --- |
| 私有状态本身是否有用 | 相同 Plan 和路由下，STATE=NONE 对照具体状态模块 |
| 未 active receiver 的 Observe 是否有用 | 相同 Plan、receiver 和 active set 下，SD 对照 BO |
| 状态是否参与选择 | SEL-CONTENT、SEL-PRE、SEL-POST 配对比较 |
| post-update 选择是否有额外价值 | BO + SEL-POST 对照 BO + SEL-PRE/SEL-CONTENT |
| 主任务梯度是否需要经过 selector 概率 | EMIT-HARD 对照 EMIT-HST |
| 多父消息是否有用 | 保持 nodes 与预算不变，开关对应固定边 |
| 多层局部传播是否有用 | 单层实例对照具有匹配成本的非平凡 Plan |
| HB 规则拓扑是否有用 | 通用 Plan 语义不变，只替换展开拓扑 |

比较 SD 与 BO 时，优先保持 Plan、参数量、状态容量、active budget、NodeCompute、Emit、Aggregate、训练 Token 和优化器一致。若两者的 selector 输入允许相同，可以 replay 同一 active route，把差异限制为未 active receiver 是否 Observe；SEL-POST 依赖 BO 的当前 proposal，不能伪装成完全相同的 SD 输入。

状态是否真的被使用，不能只看状态数值发生变化。至少需要比较正常运行与 state freeze、clear、shuffle、no-read 或 reset 后的输出、loss 或行为差异。交叉边和其他拓扑机制也应使用同样的因果消融原则。

## 6. 训练目标与路由均衡

### 6.1 自回归语言模型损失

令 \(\theta\) 表示全部可训练参数，\(\mathcal T\) 表示一个统计批次内所有有效目标 Token 的 \((b,t)\) 集合，\(N_T=|\mathcal T|>0\)。目标 Token 为 \(w_{b,t}\)，模型条件概率为 \(P_\theta(w_{b,t}\mid w_{b,<t})\)，则

$$
\mathcal L_{\mathrm{LM}}
=-\frac{1}{N_T}
\sum_{(b,t)\in\mathcal T}
\log P_\theta(w_{b,t}\mid w_{b,<t}).
$$

next-token shift 由数据管线完成。padding 和其他无效位置不进入 \(\mathcal T\)，也不产生第 2.5 节已排除的状态更新或路由统计。

### 6.2 SettleGraph 的 availability-aware region balance loss

同一公式同时适用于第 3 节单层实例、一般合法 Plan 和 HB-Lattice，本文将它记为 `BAL-AVAIL-SOFT`。它只约束 selector 在实际可选的 candidates 中是否长期偏向少数 nodes，不试图用下游 loss 修复上游 topology 造成的 reach starvation。下面补回 batch/序列下标 \(b\)。

给 site \(j\) 的每个 region 一个稳定 ID \(\rho\)，其固定 node 集记为 \(\mathcal R_{j,\rho}\)。在当前统计范围内，令

$$
\mathcal V_{j,\rho}
=
\left\{
(b,t)\mid
\mathcal C_{j,\rho,b,t}\ne\varnothing
\right\},
\qquad
N_{j,\rho}=|\mathcal V_{j,\rho}|.
$$

没有选择事件的 region 不参加本次 loss。以下只对 \(N_{j,\rho}>0\) 的 region 定义统计量；对 \(v\notin\mathcal C_{j,\rho,b,t}\)，仅为统计方便扩展定义 \(p_{j,\rho,b,t,v}=0\)。receiver \(v\) 的平均 soft mass 为

$$
\bar p_{j,\rho,v}
=
\frac{1}{N_{j,\rho}}
\sum_{(b,t)\in\mathcal V_{j,\rho}}
\mathbf 1[v\in\mathcal C_{j,\rho,b,t}]
p_{j,\rho,b,t,v}.
$$

如果每次选择都在当时 reached 的 candidates 中均匀分配，receiver \(v\) 应得到的 availability 基准为

$$
\bar p^{\mathrm{avail}}_{j,\rho,v}
=
\frac{1}{N_{j,\rho}}
\sum_{(b,t)\in\mathcal V_{j,\rho}}
\frac{
\mathbf 1[v\in\mathcal C_{j,\rho,b,t}]
}{
\lvert\mathcal C_{j,\rho,b,t}\rvert
}.
$$

令 \(\mathcal Z\) 表示当前统计范围内至少发生过一次 \(|\mathcal C_{j,\rho,b,t}|\ge2\) 的普通竞争 regions；forced-active singleton region 不加入 \(\mathcal Z\)。定义

$$
\mathcal L_{\mathrm{bal}}^{\mathrm{SG}}
=
\begin{cases}
\displaystyle
\frac{1}{|\mathcal Z|}
\sum_{(j,\rho)\in\mathcal Z}
\frac{1}{|\mathcal R_{j,\rho}|}
\sum_{v\in\mathcal R_{j,\rho}}
\left(
\bar p_{j,\rho,v}
-\bar p^{\mathrm{avail}}_{j,\rho,v}
\right)^2,
&|\mathcal Z|>0,\\[12pt]
0,&|\mathcal Z|=0.
\end{cases}
$$

候选集合、availability 基准和离散 active set 都视为 stop-gradient；该辅助项只通过 \(p\) 把梯度传回 selector。N、SD 和 BO 使用同一个公式，因为 balance loss 约束的是选择分布，而不是 Observe 集。

第 3 节单层实例中，所有 \(R\) 个 receivers 对每个有效 Token 都 reached，因此

$$
\bar p^{\mathrm{avail}}_{j,\rho,v}=\frac1R,
$$

上式自然退化为固定候选均衡。若某次只有一个 candidate reached，实际概率和 availability 基准都为 1，不产生无法完成的均衡要求。

实际 active slots 的份额另记为

$$
\bar f_{j,\rho,v}
=
\frac{1}{N_{j,\rho}}
\sum_{(b,t)\in\mathcal V_{j,\rho}}
\frac{
\mathbf 1[v\in\mathcal A_{j,\rho,b,t}]
}{
\lvert\mathcal A_{j,\rho,b,t}\rvert
}.
$$

\(\bar p\)、\(\bar p^{\mathrm{avail}}\) 和 \(\bar f\) 应一起报告；它们分别表示 soft 倾向、当时可达性给出的参考分布和真实 active 份额。reached、Observe、active 和发送率还应分别报告，不能用其中一个代替其他三个。

启用该均衡项时，SettleGraph 的训练目标为

$$
\boxed{
\mathcal L_{\mathrm{train}}^{\mathrm{SG}}
=
\mathcal L_{\mathrm{LM}}
+\omega_{\mathrm{SG}}
\mathcal L_{\mathrm{bal}}^{\mathrm{SG}}
},
\qquad
\omega_{\mathrm{SG}}\ge0.
$$

上标 \(\mathrm{SG}\) 表示 SettleGraph。把 \(\omega_{\mathrm{SG}}\) 设为 0 即得到 `BAL-NONE` 对照。若改变统计窗口、跨设备聚合方式、region reduction 或目标分布，就属于 `BAL-CUSTOM`，必须给出完整公式。

### 6.3 Flat MoE 的路由辅助项

下面给出一种经典 Switch-style 基线，不把它规定为所有 MoE 的唯一做法。为简化公式，假设 routed sites 集合为 \(\mathcal J_{\mathrm{moe}}\)，每个 site 都有 \(R^{\mathrm{moe}}\) 个 experts，并在相同有效 Token 集 \(\mathcal V\) 上统计，\(N_V=|\mathcal V|>0\)。

令

$$
\bar p^{\mathrm{moe}}_{j,i}
=
\frac1{N_V}
\sum_{(b,t)\in\mathcal V}
p^{\mathrm{moe}}_{j,b,t,i},
$$

$$
f^{\mathrm{moe}}_{j,i}
=
\frac1{N_V}
\sum_{(b,t)\in\mathcal V}
\frac{
\mathbf 1[i\in\mathcal A^{\mathrm{moe}}_{j,b,t}]
}{
\lvert\mathcal A^{\mathrm{moe}}_{j,b,t}\rvert
}.
$$

Switch-style balance loss 为

$$
\mathcal L_{\mathrm{bal}}^{\mathrm{moe}}
=
\frac1{|\mathcal J_{\mathrm{moe}}|}
\sum_{j\in\mathcal J_{\mathrm{moe}}}
R^{\mathrm{moe}}
\sum_{i=0}^{R^{\mathrm{moe}}-1}
\operatorname{sg}\!\left(f^{\mathrm{moe}}_{j,i}\right)
\bar p^{\mathrm{moe}}_{j,i}.
$$

其中 \(\operatorname{sg}\) 表示 stop-gradient：前向值不变，反向梯度为零。router z-loss 为

$$
\mathcal L_z
=
\frac1{|\mathcal J_{\mathrm{moe}}|N_V}
\sum_{j\in\mathcal J_{\mathrm{moe}}}
\sum_{(b,t)\in\mathcal V}
\left[
\log
\sum_{i=0}^{R^{\mathrm{moe}}-1}
\exp(a^{\mathrm{moe}}_{j,b,t,i})
\right]^2.
$$

采用这两项时，

$$
\boxed{
\mathcal L_{\mathrm{train}}^{\mathrm{moe}}
=
\mathcal L_{\mathrm{LM}}
+\omega_{\mathrm{moe}}\mathcal L_{\mathrm{bal}}^{\mathrm{moe}}
+\omega_z\mathcal L_z
}.
$$

hard Top-1/no-gate 前向没有从主任务经离散 expert ID 返回 router 的梯度，router 只能依赖辅助项或其他显式梯度路径。soft gate 可以让主任务梯度经选中 gate 返回，但离散 Top-K 成员关系仍不求导。不同 MoE 条件必须分别记录其合并方式和辅助项，不能直接比较定义不同的原始 balance-loss 数值。

### 6.4 训练期均衡与推理期负载感知

| 机制 | 训练时 | 推理时 | 含义 |
| --- | --- | --- | --- |
| balance loss | 加入训练目标 | 不计算 | 学出较均衡的平均倾向 |
| 负载感知 selector | 参与模型前向 | 继续参与前向 | 根据已声明的历史动态调整本次选择 |

负载感知 selector 可以直接训练。下面只给一个最简单样例，实际实现可根据训推情况调整。令 \(\ell^-_{v,t}\) 是 receiver \(v\) 在当前 Token 前的激活 EMA，先用它修正普通分数：

$$
a_{v,t}
=a^{\mathrm{base}}_{v,t}
-\kappa_{\mathrm{load}}\ell^-_{v,t},
\qquad
\kappa_{\mathrm{load}}\ge0.
$$

选择完成后更新：

$$
\ell_{v,t}
=
\begin{cases}
\lambda_{\mathrm{load}}\ell^-_{v,t}
+(1-\lambda_{\mathrm{load}})
\mathbf 1[v\in\mathcal A_{\mathcal R,t}],
&v\in\mathcal C_{\mathcal R,t},\\
\ell^-_{v,t},
&v\notin\mathcal C_{\mathcal R,t},
\end{cases}
$$

其中 \(0\le\lambda_{\mathrm{load}}<1\)。这是按稳定序列维护的模型内部选择历史，不是硬件队列的实时负载。它比纯训练期辅助损失更强，也会形成“历史选择影响以后选择”的跨 Token 递归，可能带来振荡或额外训练难度。若使用这种规则，应把它记为自定义 selector-history，并明确初始化、跨 chunk 状态和写回时序。

## 7. 实验条件命名

短名称只用于让人快速区分主要科学条件；完整 Plan、公式和训练设置始终以第 8 节的实验记录为准。推荐格式为：

~~~text
<TRAIN>-<PLACEMENT>-<PROFILE>-<TOPOLOGY>-<STATE>-<SELECTOR>-K<ACTIVE>-<EMIT>-<AGG>-<BAL>
~~~

字段含义如下：

| 字段 | 常用值 | 含义 |
| --- | --- | --- |
| TRAIN | `PT`、`CPT`、`FT`、`SFT` | 随机初始化预训练、checkpoint continued pretraining、下游微调、监督微调 |
| PLACEMENT | `POST`、`PARBLK`、`PARATTN`、`PARMLP` | 第 1.3 节的 SettleGraph 接入位置 |
| PROFILE | `N`、`SD`、`BO`、`CUSTOM` | 第 2.3 节的传播与 Observe 语义 |
| TOPOLOGY | `SL-R8`、`SG-<plan-id>`、`HB-<plan-id>` | 单层实例、一般 SettleGraph Plan 或 HB Plan |
| STATE | `NONE`、`EMA128`、`GDN-K32-V32`、`ATTN-W128`、`CUSTOM` | receiver 状态算法与主要尺寸 |
| SELECTOR | `SEL-CONTENT`、`SEL-PRE`、`SEL-POST`、`SEL-CUSTOM` | selector 读取的状态时刻 |
| K | `K1`、`K2`、`KALL`、`KVAR` | region 请求的 active 数摘要 |
| EMIT | `EMIT-HARD`、`EMIT-HST`、`EMIT-SOFTP`、`EMIT-CUSTOM` | 第 2.3 节的发送公式 |
| AGG | `AGG-MEAN`、`AGG-LEARNED`、`AGG-CUSTOM`、`AGG-VAR` | receiver 输入和图输出聚合 |
| BAL | `BAL-AVAIL-SOFT`、`BAL-NONE`、`BAL-CUSTOM` | 第 6.2 节的训练期均衡 |

例如：

~~~text
CPT-PARMLP-BO-SL-R8-EMA128-SEL-POST-K1-EMIT-HST-AGG-MEAN-BAL-AVAIL-SOFT
PT-POST-BO-HB-hb2d2p2-GDN-K32-V32-SEL-POST-K1-EMIT-HST-AGG-MEAN-BAL-AVAIL-SOFT
~~~

`SL-R8` 表示第 3 节的单层 Plan 有 8 个固定 receivers，不表示整个模型只有 8 个 nodes。`SG-<plan-id>` 和 `HB-<plan-id>` 只是展开 Plan 的可读索引，不能代替 Plan 本身及其哈希。最大静态路径深度、Line 数、site 数、总 nodes、region 宽度和 fan-in/fan-out 都是 Plan 或模型接入方式的派生摘要，不强行塞入短名称。

当同一 run 的 sites、nodes 或 regions 使用不同设置时，相应字段使用 `VAR` 或 `CUSTOM`，并在完整记录中列出映射。TRAIN 只描述 base 权重来源与训练目标；新增 SettleGraph 参数怎样初始化另行记录。

Dense 和 Flat MoE 基线可使用更短的条件名，例如：

~~~text
CPT-DENSE
CPT-MOE-TOP1-HARD-E8
CPT-MOE-TOP2-GATE-E8
~~~

真实 run 可以在条件名前加入模型缩写，并追加 seed 与尝试编号：

~~~text
<MODEL>-<CONDITION>-s<SEED>-r<ATTEMPT>
~~~

名称是人类索引，不承担配置解析或实验复现职责。

## 8. 一个完整实验条件必须说明什么

本节只规定实验记录必须包含的信息，不规定将来采用哪种文件格式或软件结构。任何结果都必须能从一份自包含记录中判断“模型实际算了什么”以及“它与对照只差在哪里”。

### 8.1 Base 模型与顶层边界

至少记录：

- base checkpoint 或随机初始化配置、tokenizer 和模型 revision；
- 插入 SettleGraph 的确切 blocks/sites，以及每个 site 的 placement；
- 可训练、冻结和共享的参数集合；
- SettleGraph 参数和状态的初始化规则；
- 初始化是否要求保持 base 函数；若要求，记录四种 placement 中实际使用者的 equality 验证条件；
- dtype、有效 Token mask，以及训练、prefill、decode 和 chunk 的输入约定。

本文只规定 identity 初始化应满足 \(b_{\mathcal G}=h^{\mathrm{in}}\) 和 \(\Delta_{\mathcal G}=0\)，不规定具体采用零输出投影、residual scale 还是其他构造。

### 8.2 展开 Plan 与所有局部运算

至少记录：

- 完整 \(V,E,\mathfrak R\)、稳定 node/edge/region ID、入口和终端 receivers；
- 每个 receiver 的固定 parents/children、最大 fan-in/fan-out、region 大小和 forced-active 设置；
- 规范 region 依赖顺序，以及逐 Token 至少产生一个 active 终端消息的保证；
- 每个 receiver 输入和图输出的 Aggregate 公式；
- 每个 region 的 Score、\(c^{\mathrm{ctx}}\)、候选排列、Top-K 规则和 \(K^{\max}\)；
- 每个 receiver 的 Update、两类 Read、NodeCompute 和 Emit 公式；
- 参数是否跨 nodes、regions、Lines 或 sites 共享；
- 规范化展开 Plan、其哈希，以及生成它的 Builder 名称、版本和配置。

HB-Lattice 还要记录每个 Line 的 nodes、regions、phase 和 barrier，以及每条边的 tree、local、shortcut 或 mirror 来源标签。

所有语义边都携带完整 \(d_{\mathrm{model}}\) hidden。物理实现可以改变布局、分片或传输顺序，但必须按声明 dtype 无损恢复相同 Tensor；不能把有损消息压缩隐藏在相同条件名下。

### 8.3 状态、选择与跨 Token 时序

至少记录：

- propagation profile 及每个 region 的 Observe 集定义；
- receiver 状态和 selector-history 的 shape、dtype、首状态与归属键；
- content/pre/post 或自定义 selector 时序；
- proposal、Score、Top-K、commit、NodeCompute 和历史写回的确切顺序；
- proposal 到 selector 是否保留梯度，历史激活或 \(p\) 写回是否 stop-gradient；
- 跨 chunk 的 carry、reset 和 detach 规则；
- EMIT-HST 的 \(\zeta^{\mathrm{ST}}\)，以及其他自定义梯度路径；
- 与调度顺序无关的随机数键或确定性规则。

### 8.4 训练条件与统计范围

至少记录：

- 数据集、revision、样本顺序、有效 Token 数和训练阶段；
- optimizer、学习率、参数组、batch、gradient accumulation、scheduler 和 gradient clipping；
- 语言模型损失及所有辅助项的公式、系数和 reduction；
- balance loss 的统计窗口、reached mask、site/region 聚合范围及跨设备同步方式；
- Flat MoE 的 expert 数、Top-K、gate、capacity、token drop、reroute、shared expert 和辅助项；
- checkpoint、验证与停止策略。

### 8.5 可观测量与配对关系

至少报告：

- train/validation LM loss、perplexity 和任务指标；
- 每个 node 的 reached、Observe、active、发送和有效梯度次数；
- 每个 region 的 soft mass、availability 基准、hard share、熵和 active-set 变化；
- 状态变化量、读出量、write-to-read 延迟及状态干预结果；
- 每 Token 的聚合、轻量读出、proposal、commit、较大状态读出、昂贵计算和发送次数；
- 参数量、active parameters、FLOPs、状态容量、显存、吞吐和通信成本；
- 与主对照共享的条件，以及唯一被改变的坐标。

这些内容可以逐步转成机器可读 manifest，但 manifest 的软件设计不属于本文的神经网络语义。

## 附录 A：Receiver 状态模块样例

本附录只展示怎样用第 2.2 节的统一接口表达若干状态模块，不规定首轮实验必须选择哪一种。

对任意 reached receiver \(v\)，本地入口 hidden 和归一化输入始终是

$$
h_{v,t}
=\operatorname{Aggregate}_v(\mathcal M_{v,t}),
\qquad
m_{v,t}=N_{R,v}(h_{v,t}).
$$

当前 Token 前的状态为 \(s^-_{v,t}\)，proposal 为

$$
\widetilde s_{v,t}
=\operatorname{Update}_v(s^-_{v,t},m_{v,t}).
$$

第 2.3 节决定 proposal 是否 commit，并把当前完整计算可见的状态记为 \(s^{\mathrm{cmp}}_{v,t}\)。下列公式中的 \(s\) 表示传给某次 Read 的实际状态，可以是 \(s^-\)、\(\widetilde s\) 或 \(s^{\mathrm{cmp}}\)，具体由 selector 时序和 propagation profile 决定。

### A.1 设计空间一览

| 样例 | 状态主要保存什么 | 典型 Read |
| --- | --- | --- |
| 历史激活 | 次数、最近激活位置、概率或局部预算 | 供 selector 使用的少量标量 |
| EMA | 固定长度的低通内容摘要 | 低维 selector 摘要或 \(d_{\mathrm{model}}\) residual |
| Gated DeltaNet / KDA | 固定大小的 key-value 关联矩阵 | 按 query 读取关联内容 |
| SSM / Mamba | 固定大小的递归状态 | state-space 输出或轻量摘要 |
| Attention | 完整、窗口化或压缩后的 key/value 历史 | Attention 输出或其低维统计 |

\(\operatorname{Read}^{\mathrm{sel}}\) 必须保持固定、有界，通常只输出低维投影、范数或历史统计。\(\operatorname{Read}^{\mathrm{ffn}}\) 可以包含较大的状态读取和 output projection，但只由 active receiver 执行，并最终输出 \(d_{\mathrm{model}}\) 维 residual。

### A.2 历史激活

历史激活可以记录 receiver 的累计或近期 active 次数、距上次 active 的 Token 数、soft probability 的移动平均或局部预算。它在本次选择完成后写回，因此只能影响以后 Token。

如果历史只服务于 selector，可以把它作为独立 selector-history，按 \((\mathrm{site},\mathrm{region},\mathrm{sid})\) 或声明的 node-level 键保存；它不属于 receiver 的 Observe 集。如果把它并入 receiver state，则其更新必须服从 SD/BO 的 Observe 规则。两种做法必须在实验条件中区分。

只服务 selector 时可令

$$
\operatorname{Read}^{\mathrm{ffn}}_v(s,m)=0.
$$

### A.3 EMA

EMA 状态是固定长度向量：

$$
s_{v,t}\in\mathbb R^{d_s}.
$$

先从本地归一化输入产生观察量：

$$
o_{v,t}
=\tanh\!\left(
W_v^{\mathrm{obs}}m_{v,t}
+b_v^{\mathrm{obs}}
\right),
\qquad
o_{v,t}\in\mathbb R^{d_s}.
$$

更新和较大读出可以定义为

$$
\operatorname{Update}^{\mathrm{EMA}}_v(s^-_{v,t},m_{v,t})
=
\lambda_v\odot s^-_{v,t}
+(1-\lambda_v)\odot o_{v,t},
$$

$$
\operatorname{Read}^{\mathrm{ffn,EMA}}_v(s,m)
=
W_v^{\mathrm{out}}s,
\qquad
W_v^{\mathrm{out}}\in
\mathbb R^{d_{\mathrm{model}}\times d_s},
$$

其中 \(0\le\lambda_v<1\) 可以是标量或逐维向量。selector 读出可以使用 \(s\) 的低维投影、范数或与当前 \(m\) 的简单相似度，但必须给出具体公式。

### A.4 Gated DeltaNet

Gated DeltaNet 用固定大小的关联矩阵保存状态：

$$
s_{v,t}\in\mathbb R^{d_k\times d_v}.
$$

其中 \(k_{v,t},q^{\mathrm{qry}}_{v,t}\in\mathbb R^{d_k}\)，\(\nu_{v,t},e_{v,t}\in\mathbb R^{d_v}\)，\(N_k,N_q\) 表示 key/query 的向量归一化，且

$$
W_v^{\mathrm{out}}
\in\mathbb R^{d_{\mathrm{model}}\times d_v}.
$$

需要 proposal 时，从 \(m_{v,t}\) 产生 key、value 和写入门：

$$
k_{v,t}=N_k(W_v^k m_{v,t}),
\qquad
\nu_{v,t}=W_v^\nu m_{v,t},
$$

$$
\eta_{v,t}
=\sigma\!\left((w_v^\eta)^\top m_{v,t}+b_v^\eta\right),
$$

$$
\gamma_{v,t}
=\exp\!\left[
-\exp(\beta_v)
\operatorname{softplus}
\left((w_v^\gamma)^\top m_{v,t}+b_v^\gamma\right)
\right].
$$

先衰减旧状态，再写入当前 value 与已有预测之间的误差：

$$
s^{\mathrm{decay}}_{v,t}
=\gamma_{v,t}s^-_{v,t},
$$

$$
e_{v,t}
=\nu_{v,t}
-\left(s^{\mathrm{decay}}_{v,t}\right)^\top k_{v,t},
$$

$$
\operatorname{Update}^{\mathrm{GDN}}_v(s^-_{v,t},m_{v,t})
=
s^{\mathrm{decay}}_{v,t}
+\eta_{v,t}k_{v,t}e_{v,t}^\top.
$$

active receiver 产生 query 并读取：

$$
q^{\mathrm{qry}}_{v,t}
=N_q(W_v^q m_{v,t}),
$$

$$
\operatorname{Read}^{\mathrm{ffn,GDN}}_v(s,m_{v,t})
=
W_v^{\mathrm{out}}
\left(s^\top q^{\mathrm{qry}}_{v,t}\right).
$$

这里 \(q^{\mathrm{qry}}\) 是 query，与第 2.1 节表示 reached 的 \(q_{v,t}\) 无关。若 selector 也需要关联读出，可以为 reached receiver 提前计算低维 query/readout；这项计算必须计入 selector 成本。KDA 等 delta-rule 变体可以复用相同接口，但要给出自己的完整门控和更新公式。

### A.5 Attention 状态

Attention 状态保存此前 Observe 的 key/value。下面给出保留最近 \(W\) 次 Observe 的有界窗口样例：

$$
k_{v,t},q^{\mathrm{qry}}_{v,t}\in\mathbb R^{d_k},
\qquad
\nu_{v,t}\in\mathbb R^{d_v},
\qquad
W_v^{\mathrm{out}}
\in\mathbb R^{d_{\mathrm{model}}\times d_v}.
$$

$$
k_{v,t}=N_k(W_v^k m_{v,t}),
\qquad
\nu_{v,t}=W_v^\nu m_{v,t},
$$

$$
\operatorname{Update}^{\mathrm{Attn}}_v(s^-_{v,t},m_{v,t})
=
\operatorname{AppendEvict}_W
\left(
s^-_{v,t},
(k_{v,t},\nu_{v,t})
\right).
$$

若状态 \(s\) 中按时间排列的 key/value 矩阵分别为

$$
\mathbf K(s)\in\mathbb R^{n_s\times d_k},
\qquad
\mathbf V(s)\in\mathbb R^{n_s\times d_v},
\qquad
0\le n_s\le W,
$$

则 active receiver 的读出可以写为

$$
q^{\mathrm{qry}}_{v,t}=N_q(W_v^q m_{v,t}),
$$

$$
\operatorname{Read}^{\mathrm{ffn,Attn}}_v(s,m_{v,t})
=
W_v^{\mathrm{out}}
\left[
\mathbf V(s)^\top
\operatorname{softmax}
\left(
\frac{\mathbf K(s)q^{\mathrm{qry}}_{v,t}}{\sqrt{d_k}}
\right)
\right].
$$

空历史时，状态相关读出定义为零。上式默认当前 Observe 已经 commit 后再读，因此 BO/SD 中 active receiver 可以读取本 Token 写入的 key/value；如果实验要求只读旧历史，应显式改用 \(s^-_{v,t}\)。

固定窗口只是一个有界样例。也可以使用固定记忆槽位、分层或稀疏 Attention；完整历史则使状态和读取成本随上下文增长，不满足单节点成本有界的核心要求。

### A.6 其他有界状态

SSM、Mamba、KDA、RWKV 和其他线性 Attention 状态都可以接入同一组 Update、Read 与 commit 接口。要成为一个可比较条件，必须明确：

- 状态 shape 和首状态；
- 每次 proposal 的公式与成本；
- selector 读取什么、在 pre 还是 post 时刻读取；
- active receiver 的较大读出公式；
- 状态是否真正保持固定上界；
- chunk continuation 与 detach 规则。

这些名字只标识算法家族，不能替代具体神经网络定义。
