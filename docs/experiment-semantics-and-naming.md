# TIDE 实验语义、命名与数学符号

> 状态：新实验的规范性文档。
>
> 本文只定义“模型实际怎样计算”和“实验名称怎样反映计算图”。实验晋级、结果报告组织和 checkpoint 保留策略另行讨论。

当前阶段，实验名称应当分别说明：

1. 从什么权重开始训练；
2. receiver 插在 base block 的什么位置；
3. 使用 N、SD 还是 BO；
4. 每个局部 group 有多少 receivers；
5. 模型中有多少个顺序插入位置；
6. 每个插入位置内部递归多少层；
7. 使用什么状态结构；
8. selector 使用 content-only、pre-update state 还是 post-update state。

## 1. 统一数学符号

### 1.1 下标和主要变量

全文使用：

| 符号 | 含义 |
| --- | --- |
| \(\ell\) | base Transformer block 编号 |
| \(t\) | 序列中的 Token 位置 |
| \(i\) | 当前 routed module 中的候选编号；候选可以是 receiver 或 MoE expert |
| \(x_{\ell,t}\) | 实际送入第 \(\ell\) 个 base block 的 hidden；对 \(\ell>0\)，有 \(x_{\ell,t}=y_{\ell-1,t}\) |
| \(u_{\ell,t}\) | 当前 block 完成 Attention residual merge 后的表示 |
| \(v_{\ell,t}\) | 当前 block 完成原 dense MLP residual merge 后的表示，即完整 base block 输出 |
| \(y_{\ell,t}\) | 当前 block（包括可选的 receiver merge）最终送往下一个 base block 的表示 |
| \(m_{\ell,t}\) | 当前 routed module 接收并交给 router 的归一化消息 |
| \(a_{\ell,t}\) | router 在 softmax 之前产生的 logits |
| \(p_{\ell,t}^{(i)}\) | router 分给候选 \(i\) 的 soft 概率 |
| \(c_{\ell,t}\) | 当前 Token 选择的候选编号；候选可以是 receiver 或 MoE expert |
| \(\operatorname{Score}_i\) | 为候选 \(i\) 产生 router logit 的打分操作；具体输入由 selector 语义决定 |
| \(s_{\ell,t}^{(i)}\) | receiver \(i\) 的私有状态；具体形状由状态实现决定 |
| \(S_{\ell,t}\) | 一个 receiver group 的全部私有状态 |
| \(r_{\ell,t}^{(i)}\) | 从 receiver \(i\) 的私有状态中读出的结果 |
| \(z_{\ell,t}\) | 被选中昂贵 FFN 的实际输入 |
| \(g_{\ell,t}\) | 被选中 FFN 输出的 residual gate |
| \(E_i\) | 候选 \(i\) 对应的昂贵 routed FFN；在标准 MoE 中就是 expert \(i\) |
| \(\delta_{\ell,t}\) | 当前 routed module 最终发往 merge 位置的 residual |

本文按“作用”统一基本符号，具体实现只在对应小节内部展开：各类归一化都写成 \(N\)，私有状态始终写成 \(s/S\)，状态操作始终写成 \(\operatorname{Update}\) 和 \(\operatorname{Read}\)，receiver 与标准 MoE 都用 \(m\) 表示 router 收到的消息、用 \(a,p,c\) 表示路由、用 \(z\) 表示昂贵 FFN 的输入、用 \(E\) 表示昂贵 routed FFN、用 \(g,\delta\) 表示它的 gated residual。相同基本符号表示它们在框架中承担相同作用，并不表示共享参数或采用相同算法；下标说明位置或候选，EMA、GDN 等实现标签只在展开内部做法时出现。下面为便于阅读会省略部分 \(\ell,t\) 下标，但含义不变。

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

因此 \(x_{\ell,t}\) 是实际送入第 \(\ell\) 个 base block 的当前 Token hidden；对 \(\ell>0\)，有 \(x_{\ell,t}=y_{\ell-1,t}\)，而 \(y_{\ell-1,t}\) 已经包含上一层可能存在的 receiver merge。

Dense 基线没有 receiver，直接令：

$$
y_{\ell,t}=v_{\ell,t}.
$$

### 1.3 一个单层 receiver group

令 \(N_R\) 表示 receiver 输入处的归一化操作；当前实现为 RMSNorm。receiver 从所在 placement 取得消息：

$$
m_{\ell,t}=N_R(\text{placement input}).
$$

对于有状态的 SD 和 BO，在处理 Token \(t\) 之前，把该 receiver group 的完整私有状态统一记为：

$$
S_{\ell,t-1}
:=\left(
s_{\ell,t-1}^{(0)},
s_{\ell,t-1}^{(1)},
\ldots,
s_{\ell,t-1}^{(R-1)}
\right).
$$

这里的 \(s_{\ell,t}^{(i)}\) 只表示“receiver \(i\) 的完整私有状态”，不预先限定它是 EMA 向量、GDN 矩阵，还是在内部额外包含历史激活记录。具体内部结构不改变框架符号。

#### 1.3.1 三种 selector 语义

selector 在什么时刻读取状态有三种可能语义：

| Selector 语义 | 打分时可读取 | SD | BO |
| --- | --- | --- | --- |
| **Content-only** | 当前消息 \(m_{\ell,t}\) | 自然兼容 | 自然兼容 |
| **Pre-update state** | \(m_{\ell,t}\) 与旧状态 \(s_{\ell,t-1}^{(i)}\) | 自然兼容；选完后只更新被选 receiver | 自然兼容；选完后更新全部 receivers |
| **Post-update state** | \(m_{\ell,t}\) 与当前消息更新后的 \(\widetilde s_{\ell,t}^{(i)}\) | 严格 SD 不兼容 | 天然兼容；全部更新后再选择 |

三种打分统一写成：

$$
a_{\ell,t}^{(i)}
=
\begin{cases}
\operatorname{Score}_i(m_{\ell,t}),
& \text{content-only},\\[4pt]
\operatorname{Score}_i(m_{\ell,t},s_{\ell,t-1}^{(i)}),
& \text{pre-update state},\\[4pt]
\operatorname{Score}_i(m_{\ell,t},\widetilde s_{\ell,t}^{(i)}),
& \text{post-update state},
\end{cases}
$$

其中：

$$
\widetilde s_{\ell,t}^{(i)}
=\operatorname{Update}_i(s_{\ell,t-1}^{(i)},m_{\ell,t}),
$$

$$
p_{\ell,t}=\operatorname{softmax}(a_{\ell,t}),
\qquad
c_{\ell,t}=\arg\max_i p_{\ell,t}^{(i)}.
$$

\(\widetilde s\) 只是同一个 \(s\) 在“已经 Observe 当前消息、尚未完成本次选择”阶段的值。Pre 与 Post 不是包含关系：如果 \(\operatorname{Update}\) 会覆盖、压缩或遗忘旧状态，post-update state 未必还能恢复 pre-update state；若 selector 同时读取二者，应另行明确声明。

历史激活也可以作为 \(s\) 的内部内容。当前 Token 的激活结果只能影响以后 Token，因此会形成时间维上的因果递归；这不妨碍在一个 chunk 内用 scan、Torch 算子或专用 kernel 执行，但实现必须保持 `prefill = decode`。

当前实现采用 \(a_{\ell,t}=W_r m_{\ell,t}\)，属于 content-only。2026-08-21 至 2026-08-24 已运行的 N、SD、BO 实验都使用这一语义；它是首轮设置，不是整个框架的限制。

#### 1.3.2 传播 profile 与状态提交顺序

令 \(\mathcal O_{\ell,t}\) 表示当前消息实际 Observe / Update 并提交状态的 receiver 集合。三种传播 profile 的持久状态和昂贵计算规则是：

| Profile | \(\mathcal O_{\ell,t}\)：哪些 receivers 提交 Observe / Update | 哪些 receivers 执行昂贵 FFN |
| --- | --- | --- |
| **N** | 无状态，不执行 Observe / Update | 只有 \(c_{\ell,t}\) |
| **SD** | \(\{c_{\ell,t}\}\) | 只有 \(c_{\ell,t}\) |
| **BO** | 当前 parent 的全部固定直接 receivers | 只有 \(c_{\ell,t}\) |

Content-only 和 pre-update state 都先完成选择，再按上表提交状态：

$$
s_{\ell,t}^{(i)}
=
\begin{cases}
\operatorname{Update}_i\!\left(s_{\ell,t-1}^{(i)},m_{\ell,t}\right),
& i\in\mathcal O_{\ell,t},\\[4pt]
s_{\ell,t-1}^{(i)},
& i\notin\mathcal O_{\ell,t}.
\end{cases}
$$

Post-update state 与 BO 的顺序是“全部 receivers Update → selector → 全部提交”，所以直接有：

$$
s_{\ell,t}^{(i)}=\widetilde s_{\ell,t}^{(i)},
\qquad i=0,1,\ldots,R-1.
$$

严格的 SD 不自然兼容 post-update state，因为选择发生前还不知道谁可以 Update。SD 与 BO 之间存在一个 **broadcast-proposal** 夹层：全部候选只计算轻量 proposal 或临时状态，selector 选完后仅提交被选 receiver 的持久状态。它不是纯 SD，也不是完整 BO；也可以理解为把 BO receiver 内部进一步拆成 Proposal 与持久 Update。本文保留这一可能性，但暂不把它加入三种主 profile。

N 没有 receiver 私有状态，因此当前规范下只使用 content-only。N 可以增加独立的持久 **SelectorState**，保存全局或局部选择历史；它只通过 selector 的打分、选择或 gate 影响输出，不作为被选 receiver 的私有状态读出送入昂贵 FFN。足够通用的中央 selector 可以成为很宽泛的“模拟器”；但如果它按 receiver 分槽保存和更新完整状态，并把被选状态的 readout 交给对应 receiver 或 FFN，那么无论张量物理上存在哪里，语义上都属于“集中存储的 SD/BO”，而不是 N。

状态提交完成后的完整状态统一记为：

$$
S_{\ell,t}
:=\left(
s_{\ell,t}^{(0)},
s_{\ell,t}^{(1)},
\ldots,
s_{\ell,t}^{(R-1)}
\right).
$$

被选 receiver 统一通过 \(\operatorname{Read}_i\) 读取更新后的私有状态：

$$
r_{\ell,t}^{(c_{\ell,t})}
=\operatorname{Read}_{c_{\ell,t}}\!\left(
s_{\ell,t}^{(c_{\ell,t})},m_{\ell,t}
\right),
$$

$$
z_{\ell,t}
=m_{\ell,t}
+W_{c_{\ell,t}}^{\mathrm{state}}
r_{\ell,t}^{(c_{\ell,t})}.
$$

\(\operatorname{Read}\) 的输出维度可以随状态实现变化，\(W_i^{\mathrm{state}}\) 统一把它投影回 hidden 维度。在无状态的 N 中直接令：

$$
z_{\ell,t}=m_{\ell,t}.
$$

状态处理完成后，所有 profile 都只执行被选中的一个昂贵 FFN \(E_{c_{\ell,t}}\)。框架层统一写成：

$$
g_{\ell,t}
=G\!\left(p_{\ell,t}^{(c_{\ell,t})}\right),
\qquad
\delta_{\ell,t}
=g_{\ell,t}E_{c_{\ell,t}}(z_{\ell,t}).
$$

不同 gate 只替换 \(G\) 的内部定义。当前 **Soft-P gate** 使用：

$$
G_{\mathrm{SoftP}}(p)=p.
$$

也就是先用 \(c_{\ell,t}=\arg\max_i p_{\ell,t}^{(i)}\) 做硬 Top-1 dispatch，再用被选概率缩放其输出。2026-08-21 至 2026-08-24 已运行的 N、S/SD、B/BO 实验全部使用 Soft-P。

**Hard-ST gate** 是待验证的候选。令 \(\operatorname{sg}(x)\) 表示 stop-gradient：前向仍等于 \(x\)，反向梯度为零。Hard-ST 定义：

$$
G_{\mathrm{HST}}(p)
=1+p-\operatorname{sg}(p).
$$

因此 Hard-ST 的 \(g_{\ell,t}\) 在前向中等于 1，不会按被选概率缩小 residual；在反向中有 \(\partial G_{\mathrm{HST}}(p)/\partial p=1\)。当 receiver 输出非零后，主任务 loss 仍能通过被选概率训练 router。Top-1 的离散选择 \(c_{\ell,t}\) 本身仍不参与反向传播。Hard-ST 只替换 \(G\)，不改变 placement、\(\operatorname{Update}\)、\(\operatorname{Read}\)、昂贵 FFN 数量或第 4.1 节的 receiver balance loss。

后文把上述 router、Observe、Update、状态读取和昂贵 FFN 合并记为一个 receiver group 操作。对有状态的 SD 和 BO：

$$
(\delta_{\ell,t},S_{\ell,t})
=\mathcal R(m_{\ell,t},S_{\ell,t-1}).
$$

对无状态的 N，省略状态输入和输出：

$$
\delta_{\ell,t}=\mathcal R(m_{\ell,t}).
$$

### 1.4 Receiver 状态与 SelectorState 样例

第 1.3 节中的 \(s\)、\(\operatorname{Update}\) 和 \(\operatorname{Read}\) 是稳定接口。本节列出有代表性的内部实现，目的是建立设计空间，不表示它们已经通过 TIDE 实验，也不预设哪一种必然最好。状态实现与 selector 时序是两个独立坐标：同一种状态可以被 content-only 忽略，也可以被 pre-update 或 post-update selector 读取。

#### 1.4.1 一览

| 样例 | 主要保留什么 | 典型消费者 | 主要特点 |
| --- | --- | --- | --- |
| **历史激活** | 激活次数、最近激活位置、概率或局部预算 | selector | 最轻量；记录控制历史，不直接保存内容语义 |
| **EMA** | 一个固定长度的低通内容摘要 | selector / FFN | 简单、稳定，但不同历史会持续混合 |
| **Gated DeltaNet（GDN）** | 固定大小的 key-value 关联矩阵 | selector / FFN | 可以按 query 关联读取，并按预测误差写入 |
| **Kimi Delta Attention（KDA）** | 带细粒度门控的 delta-rule 矩阵状态 | selector / FFN | GDN 的近期改进，门控更细但实现更复杂 |
| **SSM / Mamba-2** | 固定大小的状态空间递归状态 | selector / FFN | 与 delta-rule 不同的成熟有界状态路线 |
| **Attention** | 完整历史、局部窗口或压缩后的 key/value | selector / FFN | 设计空间大；信息保留与状态/计算成本由具体实现决定 |

“典型消费者”只是常见用法，不是硬限制。只通过 selector 影响路由的状态，与还会被 receiver / FFN 直接读出的语义状态，必须在实验中分开说明。

#### 1.4.2 历史激活与 SelectorState

历史激活可以记录每个候选被选中的次数、距上次激活的 Token 数、soft probability 的移动平均或剩余局部预算。本次选择只能在 selector 决策完成后写回，因此只影响以后 Token。

如果这些记录只通过 selector 的打分、选择或 gate 影响输出，它们属于 **SelectorState**，可以与 N 组合；如果中央模块还按 receiver 保存完整内容状态，并把被选状态读出交给对应 receiver / FFN，则应归为集中存储的 SD/BO。

#### 1.4.3 EMA

EMA\(D\) 把收到的内容压缩成一个长度为 \(D\) 的固定向量：

$$
s_{\ell,t}^{(i)}\in\mathbb R^D,
\qquad
o_{\ell,t}^{(i)}
=\tanh\!\left(W_i^{\mathrm{obs}}m_{\ell,t}+b_i^{\mathrm{obs}}\right).
$$

它对统一接口的实现为：

$$
\operatorname{Update}_i^{\mathrm{EMA}}(s_{\ell,t-1}^{(i)},m_{\ell,t})
=\lambda_i\odot s_{\ell,t-1}^{(i)}
+(1-\lambda_i)\odot o_{\ell,t}^{(i)},
$$

$$
\operatorname{Read}_i^{\mathrm{EMA}}(s_{\ell,t}^{(i)},m_{\ell,t})
=s_{\ell,t}^{(i)}.
$$

EMA 是最简单的内容记忆基线：新观察按 \(1-\lambda_i\) 写入，旧状态按 \(\lambda_i\) 保留。EMA128 就是 \(D=128\)。

#### 1.4.4 Gated DeltaNet 与 KDA

Gated DeltaNet（GDN）把同一个框架状态 \(s\) 实现为固定大小的关联矩阵：

$$
s_{\ell,t}^{(i)}\in\mathbb R^{K\times V}.
$$

这里先抽取 gated delta-rule 的核心状态语义，不默认复制完整开放模型 block 中的短卷积、输出门或其他外围结构；若实验加入这些部件，必须单独声明。

调用 \(\operatorname{Update}\) 或 \(\operatorname{Read}\) 时，按需从 \(m_{\ell,t}\) 生成归一化的 query/key、value 和写入门：

$$
q_{\ell,t}^{(i)}=N_q(W_i^q m_{\ell,t}),
\qquad
k_{\ell,t}^{(i)}=N_k(W_i^k m_{\ell,t}),
\qquad
\nu_{\ell,t}^{(i)}=W_i^\nu m_{\ell,t},
$$

$$
\beta_{\ell,t}^{(i)}
=\sigma\!\left((w_i^\beta)^\top m_{\ell,t}+b_i^\beta\right),
\qquad
\gamma_{\ell,t}^{(i)}
=\exp\!\left[
-\exp(\alpha_i)\,
\operatorname{softplus}\!\left((w_i^\gamma)^\top m_{\ell,t}+b_i^\gamma\right)
\right].
$$

其中 \(N_q,N_k\) 表示 query/key 的向量归一化，\(\gamma\) 控制旧状态保留量，\(\beta\) 控制本次误差写入量，\(\alpha_i\) 是可学习的衰减参数。

GDN 先衰减旧状态，再只写入当前 value 与已有预测之间的误差：

$$
s_{\ell,t,\mathrm{decay}}^{(i)}
=\gamma_{\ell,t}^{(i)}s_{\ell,t-1}^{(i)},
\qquad
e_{\ell,t}^{(i)}
=\nu_{\ell,t}^{(i)}
-(k_{\ell,t}^{(i)})^\top s_{\ell,t,\mathrm{decay}}^{(i)},
$$

$$
\operatorname{Update}_i^{\mathrm{GDN}}(s_{\ell,t-1}^{(i)},m_{\ell,t})
=s_{\ell,t,\mathrm{decay}}^{(i)}
+\beta_{\ell,t}^{(i)}k_{\ell,t}^{(i)}(e_{\ell,t}^{(i)})^\top,
$$

$$
\operatorname{Read}_i^{\mathrm{GDN}}(s_{\ell,t}^{(i)},m_{\ell,t})
=(q_{\ell,t}^{(i)})^\top s_{\ell,t}^{(i)}.
$$

因此 GDN 比 EMA 多了“按 key 写入、按 query 读取”的结构。它已经被开放权重的 [Qwen3-Next](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) 和 [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) 系列直接采用，是很强的现代参考点，但这不证明它对 TIDE receiver 必然最优。

[Kimi Linear](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base) 使用的 Kimi Delta Attention（KDA）进一步为 delta rule 引入细粒度门控，并公开了训练权重与 chunk/recurrent kernel。它可以作为 GDN 之后的增强候选；代价是状态更新、参数匹配和 kernel 移植都更复杂，因此不必在第一轮同时实现。

#### 1.4.5 Attention 状态

Attention receiver 可以把历史 key/value 作为同一个状态 \(s\)，再用当前 query 执行普通 Attention。下面的固定窗口只是最容易写清楚的一种例子：

$$
s_{\ell,t}^{(i)}
=\left((k_\tau^{(i)},\nu_\tau^{(i)})\right)_{
\tau=\max(0,t-W+1)}^t,
$$

$$
\operatorname{Read}_i^{\mathrm{Attn}}(s_{\ell,t}^{(i)},m_{\ell,t})
=\operatorname{softmax}\!\left(
\frac{q_{\ell,t}^{(i)}K_{\ell,t}^{(i)\top}}{\sqrt K}
\right)V_{\ell,t}^{(i)}.
$$

实际实现不必采用固定窗口：可以保留完整历史，也可以使用滑动窗口、分层/稀疏选择、压缩 key/value 或固定记忆槽位。完整历史保留的信息最多，但状态和读取成本随序列增长；窗口或压缩方案成本更可控，但会引入不同的信息选择。本文不预设哪一种最好，实验时根据算力、状态预算和研究直觉选择，并如实报告实际状态量、读取成本和被保留的历史范围。

#### 1.4.6 其他有界状态路线与当前定位

SSM / Mamba-2 是另一类重要的固定状态候选，开放权重的 [Falcon-H1](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Base) 已采用 Transformer 与 Mamba 的混合结构；RWKV-7、Lightning Attention 等也提供了可参考的递归或线性注意力状态。它们证明“有界 recurrent state”有多条成熟路线，但不必全部进入首轮 TIDE 实现。

当前更合适的定位是：历史激活用于最轻量的 selector 控制，EMA 作为简单内容基线，GDN 作为第一种先进关联记忆锚点，Attention 保留为可按预算选择的宽泛设计族；KDA 和 Mamba/SSD 则是增强或跨家族候选。这只是帮助建立全局观，不是固定实验顺序。维度和状态量必须在名称中明确，例如 **GDN-K32-V32** 有 \(32\times32=1024\) 个状态标量，不能与 EMA128 当作等状态量对照。

## 2. Receiver 的四种插入位置

以下四种结构共享同一个 base block 和 receiver group 定义，唯一变化是 receiver 读取什么，以及 residual 在哪里合并。各节公式展示 SD/BO 的有状态接口；对于 N，只需把 \((\delta_{\ell,t},S_{\ell,t})=\mathcal R(m_{\ell,t},S_{\ell,t-1})\) 换成 \(\delta_{\ell,t}=\mathcal R(m_{\ell,t})\)。

### 2.1 POST：完整 block 后串联

这是 2026-08-21 至 2026-08-24 已运行 N/S/B 实验的真实语义。

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

receiver 再读取完整 block 输出：

$$
m_{\ell,t}=N_R(v_{\ell,t}),
\qquad
(\delta_{\ell,t},S_{\ell,t})
=\mathcal R(m_{\ell,t},S_{\ell,t-1}),
$$

$$
y_{\ell,t}=v_{\ell,t}+\delta_{\ell,t}.
$$

~~~text
x → Attention → u → 原 dense MLP → v → receiver → y
~~~

POST 在计算图上增加了一层串行的、有状态的 routed FFN 子层。它没有新的 self-attention，因此不是完整 Transformer block；但它确实增加了逻辑深度。receiver 能看到当前 block 的 Attention 和原 MLP 结果。

### 2.2 PARBLK：与完整 block 并列

对位置 \(t\)，receiver 的当前消息直接来自 \(x_{\ell,t}\)，而原 base block 的 Attention 按 1.2 节读取 \(X_{\ell,\le t}\)。两条分支以同一个当前层输入序列为源，只在 block 末尾合并：

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
m_{\ell,t}=N_R(x_{\ell,t}),
\qquad
(\delta_{\ell,t},S_{\ell,t})
=\mathcal R(m_{\ell,t},S_{\ell,t-1}),
$$

$$
y_{\ell,t}=v_{\ell,t}+\delta_{\ell,t}.
$$

也可以写成：

$$
y_{\ell,t}
=x_{\ell,t}
+\underbrace{(v_{\ell,t}-x_{\ell,t})}_{\text{完整 base block 分支}}
+\underbrace{\delta_{\ell,t}}_{\text{receiver 分支}}.
$$

~~~text
          ┌→ 完整 base block ─┐
x ────────┤                    + → y
          └→ receiver 分支 ───┘
~~~

receiver 看不到当前 block 的 Attention 或 MLP 结果，也不改变它们的输入。如果“完整 Transformer block”被视为一个空间节点，PARBLK 最接近“base block 与 receiver 子图是兄弟分支”的语义。

### 2.3 PARATTN：与 Attention 并列

对位置 \(t\)，receiver 的当前消息直接来自 \(x_{\ell,t}\)，Attention 则读取因果前缀 \(X_{\ell,\le t}\)。两条分支先在 Attention residual 位置合并；原 dense MLP 再读取合并后的表示：

$$
m_{\ell,t}=N_R(x_{\ell,t}),
\qquad
(\delta_{\ell,t},S_{\ell,t})
=\mathcal R(m_{\ell,t},S_{\ell,t-1}),
$$

$$
u'_{\ell,t}
=x_{\ell,t}
+\left[A_\ell\!\left(N_A(X_{\ell,\le t})\right)\right]_t
+\delta_{\ell,t},
$$

$$
y_{\ell,t}
=u'_{\ell,t}+F_\ell\!\left(N_F(u'_{\ell,t})\right).
$$

~~~text
          ┌→ self-attention ─┐
x ────────┤                   + → u' → 原 dense MLP → y
          └→ receiver ───────┘
~~~

### 2.4 PARMLP：与 MLP 并列

当前 Attention 先完成 residual merge；原 dense MLP 和 receiver 都读取 \(u\)，最后在 MLP residual 位置合并：

$$
u_{\ell,t}
=x_{\ell,t}
+\left[A_\ell\!\left(N_A(X_{\ell,\le t})\right)\right]_t,
$$

$$
m_{\ell,t}=N_R(u_{\ell,t}),
\qquad
(\delta_{\ell,t},S_{\ell,t})
=\mathcal R(m_{\ell,t},S_{\ell,t-1}),
$$

$$
y_{\ell,t}
=u_{\ell,t}
+F_\ell\!\left(N_F(u_{\ell,t})\right)
+\delta_{\ell,t}.
$$

~~~text
x → self-attention → u
                      ├→ 原 dense MLP ─┐
                      └→ receiver ───── + → y
~~~

receiver 能看到当前 Attention 的结果，但看不到当前原 MLP 的结果，也不改变原 MLP 的输入。原 dense MLP 可以理解为 always-on shared branch，receiver 是与它并列的稀疏有状态分支。

\(N_F\) 与 \(N_R\) 是否共享参数属于独立实现坐标。无论是否共享，两条分支的未归一化输入都必须是同一个 \(u\)，才能称为 PARMLP。

### 2.5 四种位置的直接比较

| Placement | receiver 读取 | 看见当前 Attention 结果 | 看见当前原 MLP 结果 | 改变当前原 MLP 输入 | 逻辑关系 |
| --- | --- | ---: | ---: | ---: | --- |
| **POST** | \(v\) | 是 | 是 | 否 | 完整 block 后串联 |
| **PARBLK** | \(x\) | 否 | 否 | 否 | 与完整 block 并列 |
| **PARATTN** | \(x\) | 否 | 否 | 是 | 与 Attention 并列 |
| **PARMLP** | \(u\) | 是 | 否 | 否 | 与 MLP 并列 |

只要 receiver residual 的输出投影初始化为零，四种位置都可以在初始化时保持 base 模型函数不变。但它们离开零点后的前向耦合、梯度路径和有效深度不同，不能共享训练结果或视为同一架构续跑。

## 3. Dense 与标准 MoE 基线

### 3.1 DENSE

DENSE 使用原 block：

$$
y=v
=u+F_\ell(N_F(u)).
$$

### 3.2 MOE

当前标准 Top-1 MoE 锚点在 MLP 位置用 routed expert 替换原 dense MLP：

$$
m_{\ell,t}=N_F(u_{\ell,t}),
\qquad
z_{\ell,t}=m_{\ell,t},
$$

$$
a_{\ell,t}=W_{\mathrm{moe}}m_{\ell,t},
\qquad
p_{\ell,t}=\operatorname{softmax}(a_{\ell,t}),
$$

$$
c_{\ell,t}=\arg\max_i p_{\ell,t}^{(i)},
\qquad
g_{\ell,t}=1,
$$

$$
\delta_{\ell,t}
=g_{\ell,t}E_{c_{\ell,t}}(z_{\ell,t}),
\qquad
y_{\ell,t}
=u_{\ell,t}
+\delta_{\ell,t}.
$$

这里的 \(m,a,p,c,z,g,E,\delta\) 与 receiver group 使用同一组角色符号。当前 M8 的区别在具体做法：归一化消息 \(m=N_F(u)\) 同时也是 expert 输入 \(z\)，MoE router 产生 \(a,p,c\)，并令 \(g=1\)，即被选 expert 的输出不再乘 soft 概率。它没有 receiver 私有状态，因此不使用 \(s\)、\(\operatorname{Update}\) 或 \(\operatorname{Read}\)。

它与 PARMLP 处于相同的 block 接口，但语义不同：

- MOE 用一个 routed expert 替换原 dense MLP；
- PARMLP 保留原 dense MLP，再增加一个并列 receiver residual。

当 PARMLP 内部只经过一层 receiver group 时，也可以直观地看作一种 shared-expert MoE：原 dense MLP 是 always-on shared expert，receivers 是 routed experts。

## 4. 实际训练时的损失函数

令 \(\mathcal T\) 表示一个 micro-batch 中所有有效的目标 Token，\(N_T=|\mathcal T|\)。自回归语言模型损失为：

$$
\mathcal L_{\mathrm{LM}}
=-\frac{1}{N_T}
\sum_{t\in\mathcal T}
\log P_\theta(w_t\mid w_{<t}).
$$

路由辅助项使用的 Token 集合略有不同。令 \(\mathcal V\) 表示 attention mask 标记为有效、实际经过 router 的全部输入位置，\(N_V=|\mathcal V|\)。balance loss 是额外的路由均衡目标：它不要求单个 Token 均匀选择所有候选，而是避免整个 micro-batch 长期集中到少数 receivers 或 experts。令 \(\mathcal I\) 表示所有 routed 插入位置，\(I=|\mathcal I|\)；每个位置先独立计算 balance loss，再对 \(I\) 个位置取平均。

### 4.1 N、SD、BO 的 receiver balance loss

对插入位置 \(\ell\) 的 \(R\) 个 receivers，平均 softmax 概率为：

$$
\bar p_{\ell,i}
=\frac{1}{N_V}
\sum_{t\in\mathcal V}p_{\ell,t}^{(i)}.
$$

当前 N、SD、BO 共同使用：

$$
\mathcal L_{\mathrm{bal}}^{\mathrm{receiver}}
=\frac{1}{I}
\sum_{\ell\in\mathcal I}
\sum_{i=0}^{R-1}
\left(\bar p_{\ell,i}-\frac1R\right)^2.
$$

它约束的是平均 soft 概率，不直接约束 \(\arg\max\) 后各 receiver 真正执行了多少次。因此它鼓励均衡，但不能严格保证 hard active counts 均衡。

N、SD、BO 的实际反向传播目标都是：

$$
\boxed{
\mathcal L_{\mathrm{train}}^{\mathrm{N/SD/BO}}
=\mathcal L_{\mathrm{LM}}
+0.01\,\mathcal L_{\mathrm{bal}}^{\mathrm{receiver}}
}.
$$

当前 receiver 没有 router z-loss。manifest 中即使存在 `router_z_coefficient` 字段，N、SD、BO 的实际 router z-loss 仍为零。

### 4.2 M8 的 MoE balance loss 与 router z-loss

M8 使用不同的 Switch-style balance loss。沿用第 1.1 节的统一符号，令 \(p_{\ell,t}^{(i)}\) 为 MoE router 的 softmax 概率，\(c_{\ell,t}\) 为硬 Top-1 expert，定义：

$$
\bar p_{\ell,i}
=\frac{1}{N_V}
\sum_{t\in\mathcal V}p_{\ell,t}^{(i)},
\qquad
f_{\ell,i}
=\frac{1}{N_V}
\sum_{t\in\mathcal V}
\mathbf 1[c_{\ell,t}=i].
$$

其中 \(f_{\ell,i}\) 是 expert \(i\) 真正收到的 Token 比例。M8 使用：

$$
\mathcal L_{\mathrm{bal}}^{\mathrm{MoE}}
=\frac{1}{I}
\sum_{\ell\in\mathcal I}
R\sum_{i=0}^{R-1}
\operatorname{sg}(f_{\ell,i})\,\bar p_{\ell,i}.
$$

其中 \(\operatorname{sg}\) 表示 stop-gradient（停止梯度）：前向值不变，反向梯度为零。\(f_{\ell,i}\) 来自不可导的硬路由，梯度只通过 \(\bar p_{\ell,i}\) 返回 router。完全均衡时，\(\mathcal L_{\mathrm{bal}}^{\mathrm{MoE}}=1\)，而 receiver balance loss 完全均衡时等于 0，所以两种 `balance_loss` 的原始数值不能直接比较。

沿用第 3.2 节，MoE router 收到的消息、expert 输入和 router logits 为：

$$
m_{\ell,t}=N_F(u_{\ell,t}),
\qquad
z_{\ell,t}=m_{\ell,t},
\qquad
a_{\ell,t}=W_{\mathrm{moe}}m_{\ell,t}.
$$

M8 还使用 router z-loss，限制 logits 的整体尺度：

$$
\mathcal L_z
=\frac{1}{I N_V}
\sum_{\ell\in\mathcal I}
\sum_{t\in\mathcal V}
\left[
\log\sum_{i=0}^{R-1}\exp(a_{\ell,t}^{(i)})
\right]^2.
$$

因此 M8 的实际反向传播目标是：

$$
\boxed{
\mathcal L_{\mathrm{train}}^{\mathrm{M8}}
=\mathcal L_{\mathrm{LM}}
+0.01\,\mathcal L_{\mathrm{bal}}^{\mathrm{MoE}}
+0.001\,\mathcal L_z
}.
$$

> **备注：**M8 采用的是成熟、可靠且便于对照的经典 MoE 基线，但不是所有先进 MoE 统一采用的唯一方案。

| 机制或路线 | 当前定位 | 代表性采用情况 |
| --- | --- | --- |
| **Switch-style balance loss** | 常见的标准基线，但不是唯一推荐路线 | Mixtral、OLMoE 使用；Qwen3 使用 global-batch 变体 |
| **Router z-loss** | 公认有效的可选稳定器，但采用并不统一 | ST-MoE 推荐，OLMoE 使用 |
| **其他负载均衡路线** | 用动态 bias、分位数校准或系统级 dispatch 替代或补充经典辅助损失 | DeepSeek-V3/R1：动态 expert bias；Kimi K3：Quantile Balancing；GLM-5.2/5.3：`noaux_tc`，5.3 沿用 5.2 base；MiniMax-Text-01：GShard-style auxiliary loss + global token dispatch |

这里的动态 expert bias 和 Quantile Balancing 都是训练期均衡；Kimi K3 的最终 bias 在推理时冻结，不等于第 4.3 节的推理期负载感知 selector。

DENSE 没有 router，实际目标只有 \(\mathcal L_{\mathrm{LM}}\)。训练日志中的 `loss` 是包含上述辅助项的总损失，`lm_loss` 只表示 Token 预测损失；跨架构比较模型质量时应使用验证集 `lm_loss` 或 perplexity，而不是直接比较总 `loss` 或两种定义不同的 `balance_loss`。

### 4.3 训练期均衡与推理期负载感知

| 机制 | 训练时 | 推理时 | 作用 |
| --- | --- | --- | --- |
| **训练期 balance loss** | 加入训练目标 | 不再计算 | 让模型学出较均衡的路由倾向，但不保证推理时始终均衡 |
| **负载感知 selector** | 作为前向规则参与训练 | 继续使用 | 根据当前序列的路由历史动态调整后续选择 |

负载感知 selector 可以使用一个简单的 SelectorState。例如，令 \(\operatorname{load}_{\ell,t}^{(i)}\) 表示 receiver \(i\) 的近期激活负载：

$$
a_{\ell,t}^{(i)}
=\operatorname{Score}_i(\cdots)
-\lambda\,\operatorname{load}_{\ell,t-1}^{(i)},
$$

完成选择后更新：

$$
\operatorname{load}_{\ell,t}^{(i)}
=\rho\,\operatorname{load}_{\ell,t-1}^{(i)}
+(1-\rho)\mathbf 1[c_{\ell,t}=i].
$$

同一前向规则可以同时用于训练和推理。训练期 balance loss 只留下学到的均衡倾向；动态负载路由则形成更强的闭环反馈，但也会引入跨 Token 递归，并可能造成路由振荡或增加训练难度。

## 5. 规范命名

### 5.1 科学条件名

TIDE 候选的科学条件名采用：

~~~text
<TRAIN>-<PLACEMENT>-<PROFILE>-R<WIDTH>-I<SITES>-H<DEPTH>-<STATE>-<SELECTOR>
~~~

例如：

~~~text
CPT-PARMLP-BO-R8-I4-H1-EMA128-SEL-POST
PT-POST-SD-R8-I4-H1-EMA128-SEL-PRE
PT-PARMLP-BO-R4-I4-H2-GDN-K32-V32-SEL-CONTENT
~~~

字段定义如下：

| 字段 | 允许值或形式 | 含义 |
| --- | --- | --- |
| TRAIN | PT / CPT / FT / SFT | 初始化与训练阶段 |
| PLACEMENT | POST / PARBLK / PARATTN / PARMLP | receiver 插入位置 |
| PROFILE | N / SD / BO | 状态接收与稀疏计算语义 |
| R | R4、R8、R16 等 | 每个局部 receiver group 的候选数 |
| I | I1、I4、I8 等 | 一个 Token 顺序经过的插入位置数 |
| H | H1、H2 等 | 每个插入位置内部的局部递归层数 |
| STATE | NONE、EMA128、GDN-K32-V32、ATTN-FULL、ATTN-W128、ATTN-COMP 等 | 状态结构和必要尺寸 |
| SELECTOR | SEL-CONTENT / SEL-PRE / SEL-POST | 第 1.3.1 节定义的 selector 输入时序 |

**SEL-CONTENT**、**SEL-PRE** 和 **SEL-POST** 分别表示 content-only、pre-update state 和 post-update state。它们只说明 selector 在哪个阶段读取状态，不说明 \(\operatorname{Score}\) 内部采用线性层、MLP、GDN readout 或其他实现；精确打分公式以及状态中是否包含历史激活记录仍由 manifest 和实验设置保存。

如果历史激活记录会影响 selector 或输出，它就是模型前向语义的一部分，不能隐藏在同一个纯 EMA/GDN 条件名下。具体实现确定后，应在 **STATE** 中增加明确的复合状态标签；记录维度、衰减、写回规则等细节再放入 manifest。

2026-08-21 至 2026-08-24 已完成或已经启动的历史 N、SD、BO run 都应解释为 **SEL-CONTENT**。为避免破坏已有实验目录和 checkpoint 谱系，不要求回头重命名这些 run；新报告在首次引用时补全 selector 语义即可。

TRAIN 的含义必须严格区分：

- **PT**：随机初始化后做自回归预训练；
- **CPT**：加载预训练 checkpoint，继续做语言模型目标训练；
- **FT**：加载预训练 checkpoint，使用不同于基础自回归预训练的下游任务目标；
- **SFT**：FT 中特指有监督的指令或输入输出微调。

口语中的“finetune”不能直接写入正式名称：如果实际仍是 FineWeb 或领域语料上的自回归语言模型训练，应记为 CPT；只有训练目标确实改变时才记为 FT 或 SFT。

### 5.2 R、I、H 不得混用

- **R8** 只表示每个局部 group 有 8 个候选，不表示模型共有 8 个 receivers。
- **I8** 表示每个 Token 顺序经过 8 个插入位置，不表示 Transformer 只有 8 个 blocks。
- **H2** 表示一个插入位置内部递归两层，不表示模型中有两个插入位置。

例如 **R4-I8-H1** 表示 8 个顺序插入位置，每处只有一层局部 receiver group，每个 group 有 4 个候选。它不是 8 层递归。

如果不同插入位置或递归层采用不同宽度，短名字中使用 **RVAR**，并在 manifest 和报告中列出完整宽度向量；不得用一个看似统一的 R 值掩盖异构拓扑。

### 5.3 具体 run 实例名

科学条件之外，真实 run 还需要模型、seed 和尝试编号：

~~~text
<MODEL>-<scientific-condition>-s<SEED>-r<ATTEMPT>
~~~

例如：

~~~text
q3-06b-cpt-parmlp-bo-r8-i4-h1-ema128-sel-post-s42-r1
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
- receiver norm 是否与 base norm 共享；
- selector 的精确 \(\operatorname{Score}\) 公式、状态读出方式以及是否包含历史激活记录；
- Observe、proposal、selector、状态提交和历史激活写回的精确顺序；
- receiver residual 的 merge 权重；
- 状态生命周期以及序列、chunk 和 batch 边界；
- 每个 Token 实际执行多少个昂贵 FFN；
- 初始化怎样保持或改变 base 函数；
- MOE 是否有 expert capacity、token drop 或 reroute。

这些项目不会全部进入短名字，但它们决定两个 run 是否真的是 matched comparison、到底哪些地方是 matched。
