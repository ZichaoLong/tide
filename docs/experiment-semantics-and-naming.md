# SettleGraph 实验实例、局部公式与命名

> 本文继承 20-tide 的 **tide-core-3**，固定上游提交为 `e529de212c605f7f417c3a1ce97e780a2ee59824`。通用对象、接口及定理以该提交中的[语义锚点][up-anchor]、[SettleGraph 教材][up-settle]和 [TimedDAG 教材][up-timed]为准。本文负责 fractal-latcarf 的具体实验实例：模型接入、局部公式、标准 profile、拓扑生成、训练目标与可复现条件。
>
> [整理前的语义文档](archive/experiment-semantics-and-naming-pre-tide-core-2.md)仅用于追溯旧条件，不是当前规范。本文的继承声明不代表现有实现已经通过上游语义验证；实现范围见[实现与等价性验证计划](settlegraph-implementation-plan.md)，具体比较与证据见[等价性测试契约](equivalence-test-contract.md)和 [core-v1 资格计划](core-v1-qualification-plan.md)。`core-v1` 是本地资格范围名称，与上游语义版本不是同一版本轴。父边感知 receiver 聚合和终端节点感知输出聚合现在是 tide-core-3 的标准 SettleGraph 实例；具体公式、成本与验证范围仍由本地契约限定。

上游版本、同步方式及本地扩展登记见[上游语义关系](upstream-semantics.md)。

[up-anchor]: https://github.com/ZichaoLong/ObsidianVault/blob/e529de212c605f7f417c3a1ce97e780a2ee59824/20-tide-decentralized-neural-network/semantics-anchor.md
[up-settle]: https://github.com/ZichaoLong/ObsidianVault/blob/e529de212c605f7f417c3a1ce97e780a2ee59824/20-tide-decentralized-neural-network/settlegraph-learning-note.md
[up-timed]: https://github.com/ZichaoLong/ObsidianVault/blob/e529de212c605f7f417c3a1ce97e780a2ee59824/20-tide-decentralized-neural-network/timed-dag-region-selector-learning-note.md
[up-chunk]: https://github.com/ZichaoLong/ObsidianVault/blob/e529de212c605f7f417c3a1ce97e780a2ee59824/20-tide-decentralized-neural-network/timed-dag-chunk-prefill-learning-note.md

## 阅读入口：从上游对象到本地实验

本平台研究怎样把 SettleGraph 接入已有 Base LLM，并沿固定局部连接扩大可达容量。固定拓扑、单节点成本有界和可达容量增长是实验目标；具体拓扑与模块仍须分别核算这些条件。

上游 SettleGraph 已定义单输入、单输出的固定 DAG、区域共同选择与逐输入位置的单次结算。本文沿用这些对象：每个 receiver 接收并聚合实际数据，区域从 reached candidates 中选择 active nodes，只有 active nodes 做完整计算并沿固定出边发送。本地 receiver 可以采用无状态 MLP、EMA、Gated DeltaNet 或窗口 Attention；具体公式从第 2.2 节开始。

第 1 节定义 Qwen3 的接入位置；第 2 节给出本平台的局部运算及其上游映射；第 3--4 节实例化单层和 HB-Lattice；第 5--8 节规定对照、loss、命名与实验记录；附录 A 保存状态模块样例。一般有限性、唯一性、分段继续与嵌入证明由上游教材承担，本文只说明采用这些结论所需的本地条件。

上游核心聚合接收规范排序的身份—值序列；具体函数可以读取父边或终端节点身份，也可以忽略标签而退化为只读值的简单实例。`CUSTOM` 只是实验字段，不能自动扩大上游接口。所有自定义设置仍须给出完整公式、读取范围、成本分类和比较目标。

Base 权重可来自已训练 checkpoint。若要求从保持 Base 前向函数的起点开始，必须声明使图输出等于图输入的初始化及状态范围，并验证第 1.3.5 节的条件；具体训练效果由实验检验。

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
上式省略了绝对输入位置、旧状态和下一状态；这些自变量与返回值在第 2.4--2.5 节展开，不表示每个 Token 相互独立。另记 \(\mathcal G_j\) 输出值的 residual 为：
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

SettleGraph 能看到当前 Attention 的结果，但看不到当前原 MLP 的结果，也不改变原 MLP 的输入。原 dense MLP 是 always-on 路径，SettleGraph 是与它并列的稀疏、可有状态主旁路。

#### 1.3.5 直接比较与初始化

| Placement | \(h^{\mathrm{in}}\) | always-on 输出 | 看见当前 Attention | 看见当前原 MLP | 改变原 MLP 输入 | SettleGraph merge 后 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| **POST** | \(v\) | \(v\) | 是 | 是 | 否 | 直接得到 \(y\) |
| **PARBLK** | \(x\) | \(v\) | 否 | 否 | 否 | 直接得到 \(y\) |
| **PARATTN** | \(x\) | \(u\) | 否 | 否 | 是 | 得到 \(u'\)，再执行原 MLP |
| **PARMLP** | \(u\) | \(v\) | 是 | 否 | 否 | 直接得到 \(y\) |

若某种初始化使 SettleGraph 在声明的运行范围内始终满足 \(b_{\mathcal G}=h^{\mathrm{in}}\)，等价地 \(\Delta_{\mathcal G}=0\)，则接入 \(\mathcal G_j\) 后的模型与原始 Base 模型前向函数相同。首轮实现采用的充分构造及其验证边界见[实现与等价性验证计划](settlegraph-implementation-plan.md)。

## 2. 一个 Token 如何穿过 SettleGraph

### 2.1 继承的固定图与本地聚合

#### 固定图

本平台采用上游 SettleGraph 的固定 DAG 与严格区域依赖限制。为沿用本地实验记号，把拓扑写为：

$$
G=(V,E,\mathfrak R).
$$

其中：

- \(V\) 是 receiver nodes 的非空有限集合；下文 \(v\in V\) 表示 receiver ID，与第 1 节的 base hidden \(v_{\ell,t}\) 不是同一个量；
- \(E\subseteq V\times V\) 是固定有向边集合；
- \(\mathfrak R\) 是对 \(V\) 的固定 region 划分；每个 receiver 恰属一个 region，同一区域内没有 receiver 边。每个 region 配置一个选择函数，将各 region 收缩后得到的消息依赖图也必须无环。

本地静态实验记录称为 Plan，第 2.4 节列出其内容。它为 site、node、edge 和 region 的稳定 ID 规定与声明顺序和执行器内部索引无关的固定全序。父消息按 edge ID 排列；candidates、Top-K 平票和终端消息按 node ID 排列。具体 ID 编码属于[实现计划](settlegraph-implementation-plan.md)中的记录约定。

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

每个入口 receiver 都获得同一个图输入 \(h^{\mathrm{in}}_{j,t}\)。有限 DAG 中每个节点都位于某条入口—终端路径上；单个 receiver 可以同时是入口和终端，第 3 节的单层实例即采用这种边界。

#### 向上游适配的统一逻辑时间

令 \(\mathcal R(v)\) 表示 receiver \(v\) 所属 region。把跨 region 的 receiver 边与只约束等待的显式 control dependency 都视为 region 依赖；若依赖从 \(\mathcal R_a\) 指向 \(\mathcal R_b\)，则固定正整数秩须满足 \(r_{\mathcal R_a}<r_{\mathcal R_b}\)。一般 Plan 的默认选择是：无依赖 region 的秩为 1，其余 region 的秩为全部直接数据／控制依赖最大秩加 1；第 4 节的 HB 实例另按 Line 固定秩。定义

$$
r_v=r_{\mathcal R(v)},\qquad
r_{\mathrm{out}}=1+\max_{v\in V}r_v,\qquad
T_{\mathrm{step}}=r_{\mathrm{out}}+1.
$$

对每个有效输入位置 \(t\)，适配时间为

$$
\theta_{\mathrm{in},t}=T_{\mathrm{step}}t,\qquad
\theta_{v,t}=T_{\mathrm{step}}t+r_v,\qquad
\theta_{\mathrm{out},t}=T_{\mathrm{step}}t+r_{\mathrm{out}}.
$$

同一区域的 \(\theta_{v,t}\) 相同，记为 \(\theta_{\mathcal R,t}\)。原边时延取 \(r_v-r_u>0\)；入口和输出适配节点按[上游 SettleGraph 教材][up-settle]第 10 节加入。日程由固定结构确定并随实验记录保存，不随执行器的遍历顺序变化。\(T_{\mathrm{step}}\) 是输入位置间的逻辑步幅，与 HB 的最大 Line 下标 \(D\) 不同。

本文已写出的 Aggregate、Update、Read、Score、NodeCompute 和 Emit 公式不直接读取 \(\theta\)，向上游传入它时这些函数忽略该参数。第 2.5 节的状态延续保留这个参数位置，以便明确声明时间相关实例。若增加按时间衰减，必须使用这份固定日程的时间差并保存时间戳；不能用 Observe 次数、压紧列表位置或墙钟等待代替。现有 EMA、GDN 的每次更新系数是局部递推的一部分，不自动表示空档期间的时间衰减。

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

\(\operatorname{DATA}(y)\) 对应上游边结果 \(y\)，\(\operatorname{CLOSED}\) 对应 \(\bot\)。未结算表示记录中尚缺该坐标，不能与已确定的 \(\operatorname{CLOSED}\) 混同。关闭结果不参加聚合；本平台沿用每条边每位置只结算一次的规则。

语义上，每条 \(\operatorname{DATA}\) 边都携带声明 dtype 下的完整 \(d_{\mathrm{model}}\) 维 hidden。物理实现可以打包、重排或分片传输，但必须无损恢复同一 Tensor；有损消息压缩不属于本文采用的实验实例。上游实数公式与具体 dtype 的算术实现还需通过明确的数值比较契约连接；无损传输本身不证明浮点求值与实数求值相等。

#### Receiver 如何得到一个输入 hidden

为每个入口 receiver \(v\) 固定一个不属于 \(E\) 的形式标签 \(\mathrm{in}_v\)，并定义它可接收的标签集合

$$
I_v=
\begin{cases}
\{\mathrm{in}_v\},&v\in V_{\mathrm{in}},\\
\operatorname{In}(v),&v\notin V_{\mathrm{in}}.
\end{cases}
$$

本地 trace 中的 `boundary:<node_id>` 是 \(\mathrm{in}_v\) 的稳定序列化编码，不是新增 receiver 边。入口 receiver 的消息序列只包含带该标签的图输入；其他 receiver 等全部固定父边结算后，只收集其中的 \(\operatorname{DATA}\)，保留 edge ID，并按固定 edge ID 排列。不同父边即使携带数值相同的 hidden，仍是不同的带身份项：

$$
\mathcal M_{v,t}
=
\begin{cases}
\bigl((\mathrm{in}_v,h^{\mathrm{in}}_{j,t})\bigr),
&v\in V_{\mathrm{in}},\\[4pt]
\bigl((e,y_e):\ e\in\operatorname{In}(v),\
z_{e,t}=\operatorname{DATA}(y_e)\bigr)_{\text{按 edge ID}},
&v\notin V_{\mathrm{in}}.
\end{cases}
$$

这里 \(\mathcal M_{v,t}\) 是按标签规范排序的身份—值序列，空序列记为 \(()\)。由此定义 receiver 是否 **reached**：

$$
q_{v,t}=\mathbf 1[\mathcal M_{v,t}\ne()].
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
\sum_{(i,y)\in\mathcal M_{v,t}}y.
$$

因此只有一条实际消息时，输入 hidden 就是该消息本身。可选聚合只需满足输出仍为一个 \(d_{\mathrm{model}}\) 维 hidden：

| 设置 | 定义 |
| --- | --- |
| `AGG-MEAN` | 忽略标签，对实际到达的消息值取均值；当前默认 |
| `AGG-LEARNED` | \(\sum_k\alpha_{v,i_k,t}y_k\)，其中权重可按实际标签 \(i_k\) 取参数，\(\alpha_k\ge0\)、\(\sum_k\alpha_k=1\) |
| `AGG-CUSTOM` | 自定义身份—值序列函数；须记录完整公式、规范输入顺序、读取的标签及额外参数成本 |

三种设置都使用 tide-core-3 的 SettleGraph 聚合接口。只读值的旧公式通过忽略每项第一坐标保守提升；identity-aware 公式则可以让同一值序列因来源标签不同而得到不同结果。selector 概率不在聚合中再次相乘。

父边相关线性变换等设置显式读取 \(((e_k,y_k))_k\)；终端相关聚合同理读取 \(((v_k,\widehat g_k))_k\)。它们现在是标准 SettleGraph 实例，并由上游第 10 节的嵌入保持标签。\(\operatorname{CLOSED}\) 不作为数值项传入，但聚合的固定标签域 \(I_v\) 与本次实际标签集合共同确定哪些父边缺席；实现不能用补零代替缺席。实验仍须登记完整函数、参数成本和规范排序。若身份感知 Aggregate 含逐边大矩阵等主要计算，还须把它计入昂贵作用范围，并为相应批处理结论另给精确 witness。

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

1. **收齐**：等本 region 所有成员的固定父边返回 \(\operatorname{DATA}\) 或 \(\operatorname{CLOSED}\)。
2. **选择**：聚合实际数据，确定 reached candidates，再由 region selector 选出 active nodes。
3. **计算**：确定本次状态快照和下一持久状态；active nodes 用本次快照执行完整 receiver 计算。
4. **发送**：active node 的固定出边全部携带同一个输出，其他出边全部关闭。

selector 不是发散点。固定边决定消息可能去往哪里，selector 只决定哪些已经 reached 的 receivers 本次继续计算和发送。一个 receiver 向多个 children 发送只是固定边的 fan-out；多个 parents 到达同一 receiver 时，则由该 receiver 的输入聚合操作完成 fan-in。

### 2.2 Receiver：状态与昂贵计算

receiver 是图中唯一的拓扑计算节点。每个 receiver 持有可选私有状态和昂贵计算。SettleGraph 中每份可变状态只属于一个 receiver 或 selector；各局部操作的参数默认独立，需要共享时由 Plan 显式声明哪些操作引用同一组逻辑参数。拓扑只依赖本节规定的输入、状态和输出契约，不依赖状态模块内部采用 EMA、Gated DeltaNet、Attention 还是其他算法。

对 reached receiver \(v\)，先对入口 hidden 做本地归一化：

$$
m_{v,t}=N_{R,v}(h_{v,t}).
$$

令非空集合 \(\mathsf S_v\) 表示 receiver 的状态空间，\(s^-_{v,t}\in\mathsf S_v\) 是当前输入位置以前的持久状态。具体 shape 与初态由局部模块声明；图通过 Update、两类 Read 和第 2.5 节的 Next 使用它。数学上，为每个 reached receiver 定义 proposal：

$$
\widetilde s_{v,t}
=\operatorname{Update}_v(s^-_{v,t},m_{v,t}).
$$

proposal 是一个数学函数值。第 2.3 节的 Observe 决定是否把它用作本次计算快照；第 2.5 节的 Next 再决定最终保存什么。无状态 receiver 对应上游单点状态空间，以下用 \(\varnothing\) 记其唯一值；Update 和状态延续都保持这个值。

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

\(\operatorname{Read}^{\mathrm{sel}}\) 可以是输出范数、历史激活统计或低维投影，不要求把完整 receiver 状态交给 selector。post-update 在选择前需要 proposal 的值。

content-only 和 pre-update 中，未 Observe 的 proposal 不参与本实例后续的选择、快照或状态延续，实现可以惰性地省略其求值。上游数学记录仍为全部候选定义这个值。采用省略实现时，比较前须固定删除这些未使用中间量的投影，并比较保留的完整输入、选择、控制、快照、状态、消息与输出；不能把未求值项填成零，也不能声称仅删除线程等实现坐标就得到了上游全部规范坐标。需要比较完整 proposal 记录时，应另行提供这些坐标及相应验证。

active receiver 使用第 2.3 节确定的本次快照 \(s^{\mathrm{cmp}}_{v,t}\)，执行一次 NodeCompute：

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

\(g_{v,t}\) 是一个完整 hidden，而不是相对 \(h_{v,t}\) 的增量。NodeCompute 不自行选择激活，也不乘 selector 概率；第 2.3 节的 Emit 再产生实际发送值。二者组合才对应上游 Full，不能把本地 \(g\) 直接当作上游的最终完整输出。只有 active receiver 执行较大的 \(\operatorname{Read}^{\mathrm{ffn}}\) 和昂贵计算。

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

每个 region \(\mathcal R\) 独立声明一个固定 active 上限 \(K_{\mathcal R}\)：

$$
1\le K_{\mathcal R}\le |\mathcal R|.
$$

selector 只接收 candidates 各自在本地产生的轻量 \(r^{\mathrm{sel}}\)。候选 nodes 及其读出始终按稳定 node ID 排列。令 \(\xi^-_{\mathcal R,t}\) 表示该 region 在选择前可读的显式 selector-history；它只作为 Score 的可选历史输入，缺省为 \(\varnothing\)。node-level 历史按稳定 node ID 排列。一次局部打分得到：

$$
(a_{v,t})_{v\in\mathcal C_{\mathcal R,t}}
=\operatorname{Score}_{\mathcal R}
\left(
(r^{\mathrm{sel}}_{v,t,\tau})_{v\in\mathcal C_{\mathcal R,t}},
\xi^-_{\mathcal R,t}
\right),
$$

$$
(p_{v,t})_{v\in\mathcal C_{\mathcal R,t}}
=\operatorname{softmax}
\left((a_{v,t})_{v\in\mathcal C_{\mathcal R,t}}\right).
$$

这里 \(\tau\in\{\mathrm{content},\mathrm{pre},\mathrm{post}\}\) 是 selector 时序，\(p_{v,t}\) 是 soft 选择概率，不是消息聚合权重。Top-K 按 logits \(a\) 排序得到 active set：

$$
\mathcal A_{\mathcal R,t}
=\operatorname{TopKIndex}
\left(
(a_{v,t})_{v\in\mathcal C_{\mathcal R,t}},
\min(K_{\mathcal R},|\mathcal C_{\mathcal R,t}|)
\right),
$$

logits 平票按固定 node ID 打破。候选少于 \(K_{\mathcal R}\) 时选中全部 candidates；候选为空时，不应用上面的 softmax，激活集合与控制族为空，selector-history 保持不变，也不执行 Read 或 Score。选择后的历史默认保持 \(\xi_{\mathcal R,t}=\xi^-_{\mathcal R,t}\)；其他确定更新须声明，例如第 6.4 节的规则。

forced-active receiver 使用 \(K_{\mathcal R}=1\) 的独立 singleton region。它 reached 时直接令 \(p_{v,t}=1\) 并 active；未 reached 时没有 candidate。未配置选择历史的实例可把描述量与历史空间取为单点，并省略常量 Read／Score。若该 singleton 另有激活 EMA 等选择历史，则须保留相应历史空间和 SelStep 更新；强制激活只固定 active set 与 \(p\)，不清除持久历史。这类历史更新只能读取已经声明的旧历史、候选、激活、控制量和逻辑时间；若还需要本地描述量，就不能省略对应 Read。

三种 selector 时序为：

| 时序 | selector 读取的信息 |
| --- | --- |
| **Content-only** | 当前本地输入的轻量读出 |
| **Pre-update state** | 当前输入与旧状态 \(s^-\) 的轻量读出 |
| **Post-update state** | 当前输入产生的 proposal \(\widetilde s\) 的轻量读出 |

Pre 与 post 不是包含关系：如果 \(\operatorname{Update}\) 会覆盖、压缩或遗忘旧状态，post readout 不一定能恢复 pre readout 的信息。content-only 表示 receiver 读出只使用当前内容；显式 selector-history 是与这三种时序正交的状态坐标，可按第 6.4 节的方式影响分数。selector 不读取候选本地读出和显式 selector-history 之外的未声明信息。

#### Observe 与计算快照

本平台的标准 propagation profiles 规定哪些 reached receivers 采用候选新状态作为本次快照；selector 决定哪些 receivers 执行完整计算。下表是本地实验选择空间，不是上游全部允许组合：

| Profile | 本次快照采用范围 | 完整计算与发送范围 | 本地标准 selector 时序 |
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

对有状态 receiver，Observe 确定供本次 \(\operatorname{NodeCompute}\) 使用的快照：

$$
s^{\mathrm{cmp}}_{v,t}
=
\begin{cases}
\widetilde s_{v,t},&v\in\mathcal O_{\mathcal R,t},\\
s^-_{v,t},&v\notin\mathcal O_{\mathcal R,t}.
\end{cases}
$$

content-only 和 pre-update 可先选择，再求出采用集合中需要的 proposal；其余 proposal 的省略遵守第 2.2 节的惰性投影约定。标准 post-update + BO 在选择前取得全部 reached proposal，选择后全部采用。pre-update 只限制选择读取旧状态，active receiver 的 NodeCompute 仍读取本次 \(s^{\mathrm{cmp}}\)。post-update 中 proposal 到 selector 的反向路径默认保留，任何 detach 或 stop-gradient 都必须记录。

SD/post 在上游数学上合法，但不属于本平台当前标准 profile。若研究它，应单独登记本地条件：先求全部候选 proposal 并作 post 选择，只有激活节点采用 proposal。不能把标准表中的缺项解释成一般不可能性。其他自定义组合同样必须明确 Read、采用和 Next；第 2.5 节的最终持久状态不能仅用 Observe 范围代替。

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

\(\operatorname{sg}\) 表示前向原样返回、反向贡献指定为零的操作；\(\zeta^{\mathrm{ST}}_v\) 是固定实数，默认取 1。HST 的前向恒有 \(\widehat g=g\)，但其反向是另行指定的替代规则，不是对恒等式 \(p-p=0\) 作普通求导。

暂把 \(h,g\in\mathbb R^{d_{\mathrm{model}}}\) 与 \(p\in[0,1]\) 当作独立输入，令 \(u\) 为从发送值传回的向量。三种发送方式及其局部反向贡献为：

| 设置 | 前向 \(\widehat g\) | 到 \(h\) 的贡献 | 到 \(g\) 的贡献 | 到 \(p\) 的贡献 |
| --- | --- | --- | --- | --- |
| `EMIT-HARD` | \(g\) | \(0\) | \(u\) | \(0\) |
| `EMIT-HST` | \(g\) | \(0\) | \(u\) | \(\zeta^{\mathrm{ST}}_v\langle u,g-h\rangle\) |
| `EMIT-SOFTP` | \(h+p(g-h)\) | \((1-p)u\) | \(pu\) | \(\langle u,g-h\rangle\) |

内积为 \(\langle u,w\rangle=\sum_k u_kw_k\)。HARD 与 SOFTP 使用通常的局部导数；HST 使用上面声明的替代规则。还须沿产生 \(g\) 与 \(p\) 的各条依赖传播并汇总贡献；表中的零不删除经 NodeCompute 返回 \(h\) 的间接路径。离散 Top-K 成员关系不在这里求导。未激活节点不应用 Emit，但其分数仍可能通过候选归一化影响激活节点的 \(p\)。前向记录相同不能推出训练梯度相同，比较反向时还须固定历史写回与 chunk detach。

active receiver 的每条固定出边结算为 \(\operatorname{DATA}(\widehat g_{v,t})\)；inactive 或未 reached receiver 的每条固定出边结算为 \(\operatorname{CLOSED}\)。训练期 balance loss 只提供辅助梯度，不改变这套推理数据流。

### 2.4 本地 Plan 与参考求值顺序

#### 本平台采用哪些固定图

继承的图条件与本地实验限制共同要求：

1. receiver 图有限、无环、没有重复平行边，每个 receiver 位于某条入口—终端固定路径上；
2. 每个 receiver 恰好属于一个 region，同一 region 内不存在 receiver-to-receiver 边；
3. 将每个 region 收缩后，消息依赖无环；若另加只约束等待的跨 region 调度依赖，合并后的次序也须无环；
4. hidden、状态、读出和参数的 shape/dtype，以及参数身份和共享关系已确定，所有聚合、receiver、selector、profile 和发送规则均已完整定义；
5. 固定 fan-in、fan-out、region 大小、入口/终端 receiver 数，以及单节点参数、状态和计算成本满足实验声明的上界。

前两项和消息区域图无环继承自上游 SettleGraph；第 2.1 节的秩显式编码了这种次序。额外等待可以缩小允许的求值次序，但不提供新的数据输入。若某个选择还读取别的区域结果，应明确经哪些消息或描述量传入；不能只画一条“控制依赖”就扩大 SelStep 的读取范围。超出这些接口的规则须单独登记。

原始图描述经过这些静态校验和规范化后得到的静态记录称为 **Plan**，记为 \(\Pi\)。Plan 包含 receivers、固定边、regions、稳定 ID、各项运算、参数身份与共享关系、Tensor/状态契约和依赖顺序；它不包含某个 Token 的 reached、active 或边结算结果，也不是人工编写的动态执行步骤。

#### 本地运算的参考求值顺序

令 \(\Theta\) 表示 Plan 绑定的参数与 Tensor 操作，\(S^-_t\) 表示当前输入位置以前的全部 receiver state 和 selector-history，\(S_t\) 表示结算后的最终状态。为说明本地操作怎样组成上游转移，写成

$$
(b_{\mathcal G,j,t},S_t)
=\operatorname{Interpret}
(\Pi,\Theta,t,S^-_t,h^{\mathrm{in}}_{j,t}).
$$

下面是本地公式的参考求值摘要，不声称某个现有执行器已支持全部可选模块。一般依赖次序及其正确性见上游教材；每项具体实现另按测试契约比较。

~~~text
InterpretToken(Plan, t, states_before, h_in):
  把每条固定边初始化为“未结算”
  入口 receivers 的消息序列初始化为 [(in_v, h_in)]

  只要仍有未结算 region：
    选取一个数据依赖、控制依赖和跨 Token 状态依赖都已满足的 region R

    对每个 v ∈ R：
      若 v 不是入口 receiver，断言全部固定父边已经结算
      按 edge ID 收集父边中的全部 (edge ID, DATA value)；CLOSED 不进入消息序列
      消息非空则 reached，并执行 Aggregate 与入口归一化

    按第 2.3 节为 R 完成一次选择：
      普通 region 执行 Read^sel / Score / Top-K
      forced-active singleton 直接 active
      用 Observe 确定计算快照，用 Next 和选择历史规则确定最终状态
      active receivers 用本次快照执行 NodeCompute / Emit

    对每个 v ∈ R 的每条固定出边：
      v active  ⇒ 结算为 DATA(g_hat[v])
      其他情况 ⇒ 结算为 CLOSED

  等所有终端 receivers 完成本 Token 的角色结算
  收集 active 终端 receivers 的 (node ID, g_hat)
  若集合为空，报告执行失败
  否则用 Aggregate_out 聚合为 b_G，并返回 Next 与选择历史规则确定的状态
~~~

图输出的数学定义为

$$
\mathcal M_{\mathrm{out},t}
=\bigl((v,\widehat g_{v,t}):
v\in V_{\mathrm{out}},\ v\text{ active}\bigr)_{\text{按 node ID}},
$$

$$
b_{\mathcal G,j,t}
=\operatorname{Aggregate}_{\mathrm{out}}
(\mathcal M_{\mathrm{out},t}).
$$

\(\operatorname{Aggregate}_{\mathrm{out}}\) 接收有序终端身份—值序列；均值选择忽略 node ID，node-aware 选择可以读取它。两者都是第 2.1 节定义的标准 SettleGraph 实例。上游 SettleGraph 教材的输出存在引理给出非空终端集合的保证。本平台对每个实际图输入要求该集合非空；若实现得到空集合，必须失败，不能静默回退或伪造 hidden。

本平台采用上游的单次结算规则：每个 region 每个输入位置只结算一次，每个 receiver 至多做一次完整计算，每条固定边给出一次有值或无值结果。独立区域可以按不同合法次序求值；上游唯一性定理提供数学比较目标，浮点、记录投影与反向规则则由本地测试契约单独固定。

不等长路径按第 2.1 节的秩差时延对齐。机器上先完成的结果可缓存，目标仍等待本次全部父边结算；这不把 wall-clock timeout 当作无值证明。逻辑日程已经固定，当前局部公式忽略其时间参数；改变遍历顺序不会因此改变时间坐标。

第 3 节的单层实例是所有 receivers 同时属于 \(V_{\mathrm{in}}\cap V_{\mathrm{out}}\) 的最小规则图。第 4 节的 HB-Lattice 则用规则化生成器产生 \(V\)、\(E\) 和 \(\mathfrak R\)，并附加适合批处理的 Line/phase 信息；它仍遵守同一个结算算法。

### 2.5 跨 Token 状态

令 \(\mathrm{sid}\) 表示一条稳定序列的标识，\(t=0,1,\ldots\) 表示该序列中跨 chunk 不重置的 Token 位置。receiver 状态按 \((\mathrm{site},\mathrm{receiver},\mathrm{sid})\) 隔离；selector-history 若存在，则按其声明的 region 或 node owner 隔离。不同 owner 不共享同一份可变状态。

对同一个状态键，输入位置按 \(t\) 的因果顺序连接。Observe 只定义第 2.3 节的本次快照 \(s^{\mathrm{cmp}}_{v,t}\)；最终持久状态由本地 Next 实例定义。令 \(P=\mathbb R^{d_{\mathrm{model}}}\)，其类型为

$$
\operatorname{Next}_v:
\mathsf S_v\times\mathsf S_v\times\mathbb N\times P\times\{0,1\}\times[0,1]
\to\mathsf S_v.
$$

对 reached receiver，

$$
s_{v,t}=\operatorname{Next}_v\left(
 s^-_{v,t},s^{\mathrm{cmp}}_{v,t},\theta_{v,t},h_{v,t},
 \mathbf1[v\in\mathcal A_{\mathcal R(v),t}],p_{v,t}\right),
\qquad s^-_{v,t+1}=s_{v,t}.
$$

默认 Next 返回 \(s^{\mathrm{cmp}}\)。如果需要选择后写入 active、\(p\) 统计，或激活后清理内容，必须声明对应 Next 的完整规则及所改坐标。未 reached receiver 不调用 Next，规定 \(s_{v,t}=s^-_{v,t}\)。这也说明 Observe 范围不能单独决定所有持久状态；配默认 Next 时才恢复“采用什么就保存什么”的状态规则。

只供 selector 使用的历史可以留在 \(\xi_{\mathcal R,t}\) 中，由同一次 SelStep 的历史输出确定，并满足 \(\xi^-_{\mathcal R,t+1}=\xi_{\mathcal R,t}\)。node-level 历史键若存放区域历史的各节点分量，必须声明它们与区域历史向量的对应；不能同时维护两份可独立改写的副本。

上述 Next 和选择历史更新不能读取或重算 NodeCompute／Emit 的结果，也不能读取其他未声明信息。实现即使安排在 Full 完成后保存 active 或 \(p\)，它们仍必须由当前选择与已给状态独立确定，并从下一输入位置起可见。本次 NodeCompute 始终读 \(s^{\mathrm{cmp}}\)，不能改读已清理的最终状态。需要 Full 结果反馈的设计超出 `tide-core-3`，不能作为本文标准 profile 的普通配置项。

写入 active 或 \(p\) 的历史默认 stop-gradient；这属于本地训练规则，不改变其前向数值。状态的其他梯度路径、清理与保留坐标必须随实验声明。

每条独立序列从声明的首状态开始：EMA、Gated DeltaNet/KDA 和 SSM 通常置零，Attention 历史为空，历史激活统计清零；使用可学习或其他首状态时必须记录。

对 batch 容器中序列 \(b\) 的槽位 \(u\)，三种 mask 都取 0 或 1。图执行/context mask \(e_{b,u}\) 决定该槽位是否承载一个 SettleGraph 输入。\(e_{b,u}=1\) 时还须给出该序列跨 chunk 不重置的绝对位置 \(t_{b,u}\)，并以它计算第 2.1 节的逻辑时间；每个真实 context Token 都属于这种情形。\(e_{b,u}=0\) 只表示 padding 或不存在的容器槽位：适配层返回入口 hidden，不产生图事件，也不推进绝对位置。容器下标 \(u\) 不能代替 \(t_{b,u}\)。

后面的损失与统计公式只索引实际执行的位置，并把 \(t_{b,u}\) 简写为 \(t\)。在这些位置，LM target mask 满足 \(\ell_{b,t}\le e_{b,t}=1\)，只决定是否进入第 6.1 节的语言模型损失，因此 SFT prompt 可以取 \(\ell=0\)。routing-stat mask \(r_{b,t}\le e_{b,t}=1\) 只决定已经发生的选择事件是否进入第 6.2 节的统计窗口，默认 \(r=e\)。padding 槽位不进入这些位置集合。

chunk 是一次前向接收的连续有效输入片段；prefill 是处理一段已有输入，decode 是逐位置继续。绝对位置、全部最终状态与时间戳跨 chunk 保留。上游 SettleGraph 教材第 9 节的继续定理提供前向目标；本地在 deterministic/eval 或使用相同已声明随机数据的条件下，比较整段、分块与逐位置的输出和状态。chunk 边界默认 detach，只截断段间梯度；即使前向相同，也不能据此要求跨块反向等价。

写作 \(s_{t-1}\) 时，只表示同一稳定序列上一个 Token 结算后的状态。具体状态模块见附录 A。

### 2.6 本地对象与上游接口的映射

以下在一个 site 内比较；本地 site 下标 \(j\) 不等于上游区域下标。将该 site 的稳定 region ID 映为上游区域索引，稳定 receiver ID 映为节点，边 ID 映为边身份。本地 \(q_{v,t}\) 是 reached 指示值；上游 \(q_v^t\) 表示持久状态，二者不可按同名字母替换。

| 本地对象 | 上游对象或实例化方式 |
| --- | --- |
| \(\operatorname{DATA}(y)\)、\(\operatorname{CLOSED}\) | SettleGraph 边值 \(y\)、\(\bot\)；进入 TimedDAG 编码后分别为实际消息、该固定槽位无消息 |
| reached、active | 候选集合 \(\mathcal C\)、激活集合 \(\mathcal A\) |
| Aggregate | Agg；输入是规范排序的身份—值序列，具体实例可以读取或忽略身份标签 |
| Update 与本地归一化 | \(\operatorname{Upd}_v(s,\theta,h)=\operatorname{Update}_v(s,N_{R,v}(h))\) |
| 三种 \(\operatorname{Read}^{\mathrm{sel}}\) | 上游 \(\operatorname{Read}^{0,-,+}\)，把 \(m=N_R(h)\) 组合进本地函数 |
| Score、softmax、Top-K 与选择历史更新 | 一次 SelStep 的整体函数值；返回激活集合、每候选控制量与下一历史 |
| \(p_{v,t}\) | 本实例的局部控制 \(c_{v,t}\in[0,1]\)；不是再次加入聚合的权重 |
| Observe／采用 proposal | 本次快照 \(q^{\mathrm{cmp}}=s^{\mathrm{cmp}}\) |
| \(s^-_{v,t}\)、\(s_{v,t}\) | 原节点的 \(q_v^t\)、\(q_v^{t+1}\)；后者由 Next 确定 |
| \(\xi^-_{\mathcal R,t}\)、\(\xi_{\mathcal R,t}\) | 区域选择历史的旧值与下一值 |
| NodeCompute 与 Emit 的组合 | 上游 Full；实际发送值 \(\widehat g\) 才是其结果 |

例如，在普通 receiver 实例中，映射后的 Full 为

$$
\operatorname{Full}_v(s^{\mathrm{cmp}},\theta,h,p)
=\operatorname{Emit}_v\!\left(
 h,\operatorname{NodeCompute}_v(h,N_{R,v}(h),s^{\mathrm{cmp}}),p\right).
$$

本地 Score 得到的 logits 可作为局部辅助记录保留；完整选择及其历史更新共同组成 SelStep。使用概率以外的控制内容时，必须另行声明控制空间和各字段的来源，不能让 Full 直接读取整个区域的隐式数据。

本文实例采用上游 SettleGraph 的节点、区域、带身份聚合与时间编码。向 TimedDAG 的消息、输出和切面对应使用[上游教材][up-settle]第 10 节；这里不重证一般嵌入。额外控制读入和惰性中间量记录分别按第 2.4、2.2 节声明范围。前向、数值容差、可观测记录和指定反向是不同的比较项目；任何一项通过都不能代替其余项目。

## 3. 最小实例：单层并列 receivers

本节为继承的 SettleGraph 选择一个最小本地 Plan，使用第 2 节的局部公式。它用于隔离实验变量，不另行定义图级语义。

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

“单层”表示任一入口—终端路径只经过一个 receiver node，不表示图中只有一个 node，也不限制 receiver 内部只能有一个计算子层。

### 3.2 一个 Token 的完整结算

固定一个 site \(j\)，并省略唯一 region 的下标。默认聚合对单条消息原样返回，因此对每个 \(v\in V\)，

$$
h_{v,t}=h^{\mathrm{in}}_{j,t},
\qquad
q_{v,t}=1,
\qquad
\mathcal C_t=V.
$$

selector 按第 2.3 节得到 \((a_{v,t},p_{v,t})_{v\in V}\)。将该 region 的固定 active 上限简写为 \(K\)，则

$$
\mathcal A_t
=\operatorname{TopKIndex}
\left((a_{v,t})_{v\in V},\min(K,R)\right).
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

按 Observe 范围确定快照，按 Next 确定全部 reached 节点的最终状态；每个 \(v\in\mathcal A_t\) 再用本次快照完成 NodeCompute 和 Emit。图输出为

$$
b_{\mathcal G,j,t}
=\operatorname{Aggregate}_{\mathrm{out}}
\left(
((v,\widehat g_{v,t}):v\in\mathcal A_t)_{\text{按 node ID}}
\right).
$$

Top-1 时只有一条终端消息；Top-K 使用 `AGG-MEAN` 时，聚合的是 active receivers 的完整输出，不再额外乘 selector 概率。因为所有 receivers 始终 reached，本例的 balance loss 使用第 6.2 节给出的固定候选退化形式。

### 3.3 实验作用与边界

单层实例适合分别验证 receiver 状态、content/pre/post selector 时序、N/SD/BO、Top-K、发送规则和输出聚合。它也便于与平铺 MoE 对照。

它不是固定局部度有界的容量扩展方案：增大 \(R\) 会同时扩大 selector region、入口宽度和终端宽度。验证沿固定局部连接到达更多容量，需要第 2.4 节的非平凡 Plan；第 4 节的 HB-Lattice 是其中一种规则化形态。

## 4. HB-Lattice：多层固定波前

HB-Lattice 是第 2 节合法 Plan 的一个规则化子集。它沿用 receiver、selector、聚合和发送公式，并具体固定有序 Lines、Line barrier、秩差时延以及生成这种 Plan 的规则。

第 4.1--4.4 节规定本地 HB 实例的结构与求值条件；第 4.5 节给出候选拓扑 Builder，不规定默认连接。

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
5. 任何额外 region 等待都从较浅 Line 指向较深 Line；把消息依赖、额外等待与上述 Line barrier 合并后，依赖图仍须无环。

跨多个 Lines 的消息在产生后缓存，到目标 Line 开始时再参与该 receiver 唯一一次输入聚合。

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

HB 按第 2.1 节固定

$$
r_v=\operatorname{level}(v)+1,\qquad
r_{\mathrm{out}}=D+2,\qquad
T_{\mathrm{step}}=D+3.
$$

于是第 \(t\) 个输入在 Line \(d\) 的统一逻辑时间是 \(T_{\mathrm{step}}t+d+1\)，原 HB 边的秩差时延仍等于上式的 level 差。路径时延求和使不同长度路径在本位置的同一终点对齐。\(t\) 是输入位置，\(d\) 是 Line 下标；二者按这份固定规则共同确定 \(\theta\)，不会变成两条彼此独立的衰减时钟。

site \(j\) 的 Line \(d\) 中第 \(r\) 个 region 记作 \(\mathcal R_{j,d,r}\)，其稳定 region ID 可以编码 \((d,r)\)。补回 batch/序列下标 \(b\) 后，其当前候选集仍是第 2.3 节的 reached nodes：

$$
\mathcal C_{j,d,r,b,t}
=\{v\in\mathcal R_{j,d,r}\mid q_{v,b,t}=1\}.
$$

HB Plan 的静态检查在第 2.4 节基础上增加 Line 唯一归属、region 不跨 Line、边和额外等待严格向深层、边界位于首尾 Line、合并 Line barrier 后仍无环，以及所有边类合计后的 fan-in/fan-out 上界。第 2.3 节的选择与发送规则以及入口—终端路径约束仍保证每个 \(e=1\) 的图执行 Token 得到非空终端消息，不要求 HB Builder 额外加入 forced-active backbone；若 Builder 配置了这类 backbone，其完整路径仍必须出现在展开 Plan 中。

### 4.3 一个 Token 如何逐 Line 结算

第 2.4 节的参考求值次序适用于 HB Plan；HB 的 Line barrier 给出一个更规则的合法调度：

~~~text
对一个 Token：
  for d = 0 ... D:
    等待发往 L_d 的全部固定父边结算
    对 L_d 的各 regions，按第 2 节完成：
      输入聚合 → selector → 本次快照与最终状态 → NodeCompute → Emit
    等 L_d 的全部 regions 与出边完成结算

  按第 2.4 节聚合 L_D 的终端消息并返回最终状态
~~~

fan-out 仍是把 active node 的同一输出复制到固定 children，fan-in 仍由目标 receiver 的 \(\operatorname{Aggregate}\) 完成；HB-Lattice 不增加发散节点或汇合节点。未发送的固定边仍结算为 \(\operatorname{CLOSED}\)，跨 Line 的提前消息只做缓存，不会提前触发目标 receiver。

Line barrier 按输入位置生效，不要求整个 batch 同步停住。token-major、Line-major 等求值方式仍须保持第 2.5 节的节点状态与选择历史依赖，并按同一比较投影和数值约定验证。能否按节点成批求值 Full，还依赖[分块教材][up-chunk]的函数类与精确联合求值条件；Line 标记本身不证明硬件加速。

### 4.4 Builder 与展开 Plan

拓扑 Builder 根据紧凑配置 \(\Gamma\) 产生完全展开的 HB Plan：

$$
\Pi_{\mathrm{HB}}
=\operatorname{Build}_{\mathrm{name},\mathrm{version}}(\Gamma).
$$

展开 Plan 必须列出 Lines 及其 phase、nodes、regions、固定边及其来源标签、稳定 ID、forced-active 设置和所有第 2.4 节要求的运算契约。本地实例由展开 Plan、绑定公式与参数及固定时间日程共同确定；Builder 名称和配置不能替代这些数据。正式实验同时保存规范化 Plan 及其哈希、日程和上游版本，以及 Builder 的名称、版本和配置。

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

相同坐标出现在不同 Line 时仍表示不同 receiver。tree、平台和 mirror 边的合计 fan-in/fan-out 必须保持为与宽度和平台长度无关的固定上界。

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
a^{\mathrm{moe}}_{j,t},
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

比较 SD 与 BO 时，优先保持 Plan、参数量、状态容量、active budget、NodeCompute、Emit、Aggregate、训练 Token 和优化器一致。若两者的 selector 输入允许相同，可以 replay 同一 active route，把差异限制为未 active receiver 是否 Observe；标准 BO/SEL-POST 读取当前 proposal，而标准 SD/SEL-PRE 读取旧状态，不能伪装成相同的选择输入。研究上游允许的 SD/post 时，应另立本地条件，而不是沿用标准 SD 的名称省略这个差别。

状态是否真的被使用，不能只看状态数值发生变化。至少需要比较正常运行与 state freeze、clear、shuffle、no-read 或 reset 后的输出、loss 或行为差异。交叉边和其他拓扑机制也应使用同样的因果消融原则。

## 6. 训练目标与路由均衡

### 6.1 自回归语言模型损失

令 \(\vartheta\) 表示全部可训练参数，以区别第 2.1 节的逻辑时间 \(\theta\)；\(\mathcal T=\{(b,t)\mid \ell_{b,t}=1\}\) 表示一个统计批次内由 LM target mask 选中的目标 Token 集合，\(N_T=|\mathcal T|>0\)。目标 Token 为 \(w_{b,t}\)，模型条件概率为 \(P_\vartheta(w_{b,t}\mid w_{b,<t})\)，则

$$
\mathcal L_{\mathrm{LM}}
=-\frac{1}{N_T}
\sum_{(b,t)\in\mathcal T}
\log P_\vartheta(w_{b,t}\mid w_{b,<t}).
$$

next-token shift 由数据管线完成。\(\ell=0\) 的上下文 Token 不进入 \(\mathcal T\)，但只要 \(e=1\)，仍按第 2.5 节参与模型上下文和状态更新；是否进入路由统计由 \(r\) 单独决定。

### 6.2 SettleGraph 的 availability-aware region balance loss

同一公式同时适用于第 3 节单层实例、一般合法 Plan 和 HB-Lattice，本文将它记为 `BAL-AVAIL-SOFT`。它只约束 selector 在实际可选的 candidates 中是否长期偏向少数 nodes，不试图用下游 loss 修复上游 topology 造成的 reach starvation。下面补回 batch/序列下标 \(b\)。

给 site \(j\) 的每个 region 一个稳定 ID \(\rho\)，其固定 node 集记为 \(\mathcal R_{j,\rho}\)。在当前统计范围内，令

$$
\mathcal V_{j,\rho}
=
\left\{
(b,t)\mid r_{b,t}=1,\quad
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

第 3 节单层实例中，所有 \(R\) 个 receivers 对每个进入当前统计窗口的选择事件都 reached，因此

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

下面给出一种经典 Switch-style 基线，不把它规定为所有 MoE 的唯一做法。为简化公式，假设 routed sites 的非空集合为 \(\mathcal J_{\mathrm{moe}}\)，每个 site 都有 \(R^{\mathrm{moe}}\) 个 experts，并在同一个明确声明的 MoE 路由统计 Token 集 \(\mathcal V\) 上统计，\(N_V=|\mathcal V|>0\)。

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

其中 \(0\le\lambda_{\mathrm{load}}<1\)。这组节点标量作为区域选择历史的分量，由 SelStep 在本次选择后确定；非候选分量保持。它是按选择事件更新的 EMA，不是硬件队列负载，也不是按逻辑时间间隔衰减的状态。历史影响以后选择，因此属于模型前向；这种跨 Token 递归可能带来振荡或额外训练难度，须通过实验检验。使用时应记录初态、owner、跨 chunk 保留和默认 stop-gradient。若另加时间衰减，须按第 2.1 节的日程和时间戳重新写出公式。

## 7. 实验条件命名

短名称只用于区分主要实验条件；第 8 节的完整记录还须注明上游版本、本地限制或扩展。`tide-core-3`、本地资格范围 `core-v1`、Plan schema 及 formula ID 分别记录，不互相替代。推荐短名格式仍为：

~~~text
<TRAIN>-<PLACEMENT>-<PROFILE>-<TOPOLOGY>-<STATE>-<SELECTOR>-<K>-<EMIT>-<AGG>-<BAL>
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
| K | `K1`、`K2`、`KALL`、`KVAR` | region 的固定 active 上限摘要 |
| EMIT | `EMIT-HARD`、`EMIT-HST`、`EMIT-SOFTP`、`EMIT-CUSTOM` | 第 2.3 节的发送公式 |
| AGG | `AGG-MEAN`、`AGG-LEARNED`、`AGG-CUSTOM`、`AGG-VAR` | receiver 输入和图输出聚合 |
| BAL | `BAL-AVAIL-SOFT`、`BAL-NONE`、`BAL-CUSTOM` | 第 6.2 节的训练期均衡 |

例如：

~~~text
CPT-PARMLP-BO-SL-R8-EMA128-SEL-POST-K1-EMIT-HST-AGG-MEAN-BAL-AVAIL-SOFT
PT-POST-BO-HB-hb2d2p2-GDN-K32-V32-SEL-POST-K1-EMIT-HST-AGG-MEAN-BAL-AVAIL-SOFT
~~~

`SL-R8` 表示第 3 节的单层 Plan 有 8 个固定 receivers，不表示整个模型只有 8 个 nodes。`SG-<plan-id>` 和 `HB-<plan-id>` 只是展开 Plan 的可读索引，不能代替 Plan 本身及其哈希。`KVAR` 表示不同 regions 使用不同的固定 \(K_{\mathcal R}\)，不表示 \(K\) 随 Token 自适应变化。最大静态路径深度、Line 数、site 数、总 nodes、region 宽度和 fan-in/fan-out 都是 Plan 或模型接入方式的派生摘要，不强行塞入短名称。

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

本节只规定实验记录必须包含的信息，不规定将来采用哪种文件格式或软件结构。任何结果都必须能从一份自包含记录中判断“模型实际算了什么”以及“它与对照只差在哪里”。记录首先固定上游仓库、提交 `e529de212c605f7f417c3a1ce97e780a2ee59824`、语义版本 `tide-core-3`、采用的教材与本地语义文档版本／hash；随后分别列出具体实例、局部扩展及采用的比较投影。

### 8.1 Base 模型与顶层边界

至少记录：

- base checkpoint 或随机初始化配置、tokenizer 和模型 revision；
- 插入 SettleGraph 的确切 blocks/sites，以及每个 site 的 placement；
- 可训练和冻结的参数集合；
- SettleGraph 参数和状态的初始化规则；
- 初始化是否要求保持 base 函数；若要求，验证 \(b_{\mathcal G}=h^{\mathrm{in}}\) 和 \(\Delta_{\mathcal G}=0\)；
- dtype、三类 mask，以及训练、prefill、decode 和 chunk 的输入约定。

### 8.2 展开 Plan 与所有局部运算

至少记录：

- 完整 \(V,E,\mathfrak R\)、稳定 node/edge/region ID、入口和终端 receivers；
- 每个 receiver 的固定 parents/children、最大 fan-in/fan-out、region 大小和 forced-active 设置；
- region 依赖及额外等待条件，固定的区域秩、输出秩与时间步幅，以及上游输出存在条件的适用范围；
- 每个 receiver 输入和图输出的 Aggregate 公式、读取哪些入口／edge／terminal 标签、规范排序及成本分类；只读值的设置也须明确声明忽略身份；
- 每个 region 的固定 \(K_{\mathcal R}\)；对普通竞争 region，记录 Score、候选排列和 Top-K 规则；
- 每个 receiver 的 Update、两类 Read、Next、NodeCompute 和 Emit 公式；数学 proposal 的定义域与惰性实现省略哪些中间量；
- 各项局部操作的参数身份和显式共享关系；
- 规范化展开 Plan、Plan hash，以及生成它的 Builder 名称、版本和配置。

HB-Lattice 还要记录每个 Line 的 nodes、regions、phase 和 barrier，以及每条边的 tree、local、shortcut 或 mirror 来源标签。

所有语义边都携带完整 \(d_{\mathrm{model}}\) hidden。物理实现可以改变布局、分片或传输顺序，但必须按声明 dtype 无损恢复相同 Tensor；不能把有损消息压缩隐藏在相同条件名下。

### 8.3 状态、选择与跨 Token 时序

至少记录：

- propagation profile 及每个 region 的 Observe 集、计算快照与最终状态延续；SD/post 等非标准组合单独标明；
- receiver 状态和 selector-history 的 shape、dtype、首状态与归属键；
- content/pre/post 或自定义 selector 时序；
- proposal、选择、快照、Next、NodeCompute 和历史保存的确切顺序；历史写回只依赖哪些已有数据，是否在逻辑上独立于 Full；
- proposal 到 selector 是否保留梯度，历史激活或 \(p\) 写回是否 stop-gradient；
- 跨 chunk 的绝对位置、carry、reset、时间戳解码和 detach 规则；时间衰减使用的确切逻辑间隔；
- EMIT-HST 的 \(\zeta^{\mathrm{ST}}\)，以及其他自定义梯度路径；
- 与调度顺序无关的随机数键或确定性规则。

### 8.4 训练条件与统计范围

至少记录：

- 数据集、revision、样本顺序、图执行 Token 数、LM target 数、路由统计事件数和训练阶段；
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
- 每 Token 的聚合、轻量读出、proposal、快照采用、Next、选择历史写回、较大状态读出、昂贵计算和发送次数；
- 参数量、active parameters、FLOPs、状态容量、显存、吞吐和通信成本；
- 与主对照共享的条件，以及唯一被改变的坐标。

这些内容可以逐步转成机器可读 manifest，但 manifest 的软件设计不属于本文的神经网络语义。

## 附录 A：Receiver 状态模块样例

本附录只展示怎样用第 2.2 节的统一接口表达若干状态模块，不规定首轮实验必须选择哪一种。下列样例未另行指定时，Next 返回本次计算快照。

对任意 reached receiver \(v\)，本地入口 hidden 和归一化输入始终是

$$
h_{v,t}
=\operatorname{Aggregate}_v(\mathcal M_{v,t}),
\qquad
m_{v,t}=N_{R,v}(h_{v,t}).
$$

当前输入位置以前的状态为 \(s^-_{v,t}\)，数学上为每个 reached receiver 定义 proposal：

$$
\widetilde s_{v,t}
=\operatorname{Update}_v(s^-_{v,t},m_{v,t}).
$$

第 2.3 节决定是否采用 proposal 作为 \(s^{\mathrm{cmp}}_{v,t}\)，最终持久状态再由 Next 决定。下列公式中的 \(s\) 是传给某次 Read 的实际值，可以是 \(s^-\)、\(\widetilde s\) 或 \(s^{\mathrm{cmp}}\)。未使用 proposal 的惰性省略遵守第 2.2 节的记录投影规则。以下 Update 与 Read 忽略逻辑时间；若增加时间衰减，应按第 2.1 节显式增加日程与时间戳计算。

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

只服务于 selector 的历史可作为 SelStep 的历史坐标，按 \((\mathrm{site},\mathrm{region},\mathrm{sid})\) 保存，或按已声明的 node-level 分量键编码；它不属于 receiver 的 Observe 集。若并入 receiver state，则由 Next 明确更新哪些分量。若本地 SD 条件还要求未激活候选的最终状态不变，就应同时规定其 Next 返回原状态；这不是上游 Next 的一般要求。两种 owner 和更新规则须区分，均不能读取 Full 结果。

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

其中 \(0\le\lambda_v<1\) 可以是标量或逐维向量，向量不等式逐坐标解释。它是每次 Update 的滤波系数；默认 Next 下，只有 Observe 才使这次更新进入持久状态。没有输入或没有采用时，不从该系数推出按时间自动衰减。selector 读出可以使用 \(s\) 的低维投影、范数或与当前 \(m\) 的简单相似度，具体公式须记录。

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

这次 Update 先以输入相关门值缩放旧状态，再写入当前 value 与已有预测之间的误差；这里的缩放不代表对未发生 Observe 的时间间隔补算衰减：

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

Attention 状态保存此前 Observe 的 key/value。下面给出保留最近 \(W\) 次 Observe 的有界窗口样例，其中 \(W\ge1\) 为固定整数：

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

这里的 \(s\) 是按 Observe 顺序排列、长度 \(n_s\le W\) 的 key/value 序列；\(\operatorname{AppendEvict}_W\) 追加当前 pair 后只保留最近 \(W\) 项。

若状态 \(s\) 中按 Observe 顺序排列的 key/value 矩阵分别为

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

空历史时，状态相关读出定义为零。上式默认读取本次 Observe 后的 \(s^{\mathrm{cmp}}\)，因此 BO/SD 中 active receiver 可以读取本位置写入的 key/value。若要让完整计算只读旧历史，应在状态结构中保留所需旧历史并使 Full 能从声明的快照恢复它；不能给 NodeCompute 增加一个上游 Full 未声明的隐式旧状态输入。

固定窗口只是一个有界样例。也可以使用固定记忆槽位、分层或稀疏 Attention；完整历史则使状态和读取成本随上下文增长，不满足单节点成本有界的核心要求。

### A.6 其他有界状态

SSM、Mamba、KDA、RWKV 等算法可以作为局部实例候选。只有其状态递推能写入 Update／Next、完整计算只读快照且不反馈写回时，才符合本文继承的接口。可比较条件必须明确：

- 状态 shape 和首状态；
- 每次 proposal 的公式与成本；
- selector 读取什么、在 pre 还是 post 时刻读取；
- active receiver 的较大读出公式；
- 状态是否真正保持固定上界；
- Next 保存或清理哪些状态，以及 chunk continuation、时间戳与 detach 规则。

这些名字只标识算法家族，不能替代具体神经网络定义。
