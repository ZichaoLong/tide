---
type: mathematical-learning-note
status: active-learning
as-of: 2026-09-17
cssclasses:
  - textbook-math
tags:
  - tide
  - timed-dag
  - chunk
  - prefill
  - state
  - batching
  - mathematics
  - learning-note
---

# TimedDAG 的分块预填充：因果状态块与最大前沿

> [!summary] 阅读前提与目标
> 本文以前置教材 [[timed-dag-region-selector-learning-note|《带区域选择的 TimedDAG》]] 为唯一数学前提。前置教材已经定义完整记录 $\mathcal T_x$、四类作用 $P,S,U,F$、有效封闭下界、合法过滤与切面状态。
>
> 本文研究同一份 $\mathcal T_x$ 怎样按较大的时间块精确求值。核心对象依次是：作用块、联合求值契约、最大前沿递归和严格分层正例。状态递归与完整输出分别计量，也允许由同一个联合函数同时求值。
>
> 正文采用集合、函数、偏序与递归的语言。计算机系统中的 `prefill`、`kernel`、`launch`、`scan`、`heavy/control` 和 `runtime` 集中放在附录 S 对照。

一次输入分块包含许多已经给定的输入位置。它在节点图中展开后，可能产生许多逻辑时间的状态采用作用和完整输出作用。本文关心三个量：

1. 每个节点的状态轨迹被分成多少个因果状态块；
2. 每个节点的完整输出坐标被分成多少个时间批；
3. 求值过程需要多少轮“取得新作用值以后才能决定下一批作用”的自适应阶段。

状态依赖可以在规范记录中形成一条长链，同时由一个联合函数整段求值。Attention 的前缀状态和 SSM 的仿射递推都是这种现象的典型实例。因此，本文始终区分：

$$
\begin{gathered}
\text{规范作用偏序},\\
\text{联合求值块},\\
\text{联合函数内部的工作与并行深度}.
\end{gathered}
$$

前两项在本文内形式化；第三项需要另给具体函数族与计算模型。

## 1. 固定规格、输入分块与参考区间

### 1.1 从完整语义导入的对象

固定一个带区域选择的 TimedDAG 规格。节点、边、region 与归属映射分别记为：

$$
V,\qquad A,\qquad J,\qquad \rho:V\to J.
$$

规格还包含边时延与输入时间规则。对输入 $x$，前置教材唯一确定完整记录 $\mathcal T_x$，其中包含：

$$
B_{v,\theta},\quad
\mathcal C_{j,\theta},\quad
\mathcal A_{j,\theta},\quad
q_v^\theta,\quad
y_j^\theta,
$$

以及四类作用：

$$
P_{v,\theta},\qquad
S_{j,\theta},\qquad
U_{v,\theta},\qquad
F_{v,\theta}.
\tag{1}
$$

$P$ 确定本地内容、候选状态和选择描述量；$S$ 确定激活集合、局部控制与下一选择历史；$U$ 确定本次计算快照与下一持久状态；$F$ 确定内部消息和外部输出。

对节点 $v$ 与逻辑时间区间 $I=[b,c)$，定义该节点的状态采用坐标和完整输出坐标：

$$
\Lambda_v^U(I;x)
=
\{\theta\in I\mid U_{v,\theta}\in\mathscr V_x^U\},
\tag{2}
$$

$$
\Lambda_v^F(I;x)
=
\{\theta\in I\mid F_{v,\theta}\in\mathscr V_x^F\}.
\tag{3}
$$

式 (2) 对应具有非空时间纤维的全部节点事件；式 (3) 只保留其中被选择激活的坐标。因此两种时间集合可以具有不同大小与不同分块形状。

### 1.2 正则输入流与 chunk 区间

为集中说明时间分块，取一个正整数时间步长 $\tau$，令输入位置 $t$ 的逻辑时间为 $\tau t$：

$$
t\longmapsto \tau t,
\qquad
\tau\in\mathbb N_{>0}.
\tag{4}
$$

固定起始位置 $q$ 与长度 $T>0$，相应主时间区间为：

$$
I_{q,T}
=
[\tau q,\tau(q+T)).
\tag{5}
$$

它的左、右切面分别记为 $b=\tau q$ 与 $c=\tau(q+T)$。前置教材的切面状态是：

$$
Q_b
=
\left(
b,
(q_v^b)_{v\in V},
(y_j^b)_{j\in J},
W_b
\right),
\tag{6}
$$

其中 $W_b$ 保存从左侧发送、在 $b$ 以后到达的内部消息。给定 $Q_b$、区间内的完整外部输入以及足以覆盖 $c$ 的输入封闭下界，分段继续定理唯一确定区间记录与 $Q_c$。

多输入端口只需把式 (4)--(5) 换成各端口的有限记录集合与共同目标切面；后文的作用块和联合契约保持相同。

### 1.3 三个层次的命题

本文分别研究：

1. **切面精确性**：从 $Q_b$ 推进到 $Q_c$ 时得到参考区间记录；
2. **时间批暴露**：状态采用与完整输出只被分成少量联合求值块；
3. **块内复杂度**：每个联合函数内部具有怎样的工作量、并行深度与存储量。

第一项要求每个联合求值逐坐标返回同一份参考区间记录。第二项由第 3--6 节定义。第三项随 Attention、SSM、MoE expert 等具体函数族改变。

## 2. 作用块、外部边界与精确联合求值

### 2.1 有限作用块

固定输入 $x$。取事件 DAG $\mathscr G_x^{\mathrm{ev}}=(\mathscr V_x^{\mathrm{ev}},\mathscr A_x^{\mathrm{ev}})$。一个**作用块**是有限集合：

$$
K\subseteq\mathscr V_x^{\mathrm{ev}}.
\tag{7}
$$

对每个作用 $\xi\in\mathscr V_x^{\mathrm{ev}}$，先定义它在事件 DAG 中的直接前驱集合：

$$
\operatorname{Pred}(\xi)
=\{\eta\in\mathscr V_x^{\mathrm{ev}}
\mid(\eta,\xi)\in\mathscr A_x^{\mathrm{ev}}\}.
$$

给定已经完成的作用集合 $D\subseteq\mathscr V_x^{\mathrm{ev}}$，块 $K$ 的进入依赖条件为：

$$
\forall\xi\in K,
\quad
\operatorname{Pred}(\xi)\setminus K\subseteq D,
\tag{8}
$$

这允许块内保留依赖边，同时要求每条进入块的边已经在左边界取得函数值。

若 $K$ 含有 $P_{v,\theta}$，还要求相应时间纤维由有效封闭下界证明完整。若 $K$ 含有 $S_{j,\theta}$，同一 $(j,\theta)$ 的全部候选 $P$ 都属于 $K\cup D$，并且旧选择历史由左边界或块内更早的 $S$ 给出。

### 2.2 块边界

为了让联合函数在一个真正的集合上有定义，暂时允许输入变化而保持 TimedDAG 规格固定。一个**合法区间实例**是元组：

$$
\omega=(b,c,Q_b,E_{[b,c)},\boldsymbol\sigma,
\mathcal U_{[b,c)},Q_c),
$$

其中 $b<c$，$Q_b$ 是某份完整参考记录在切面 $b$ 的状态，$E_{[b,c)}$ 是区间外部记录，$\boldsymbol\sigma$ 是覆盖目标区间的有效封闭下界，而 $(\mathcal U_{[b,c)},Q_c)$ 是前置教材分段继续递归唯一确定的区间记录与右切面。令 $\mathsf{IntervalInst}$ 为全部这种元组所成的集合，并记该实例的作用集合为 $\mathscr V^{\mathrm{ev}}(\omega)$。对带标签作用块 $K$，定义：

$$
\mathsf{Inst}(K)
=\{\omega\in\mathsf{IntervalInst}
\mid K\subseteq\mathscr V^{\mathrm{ev}}(\omega)\}.
$$

每类带名字的语义坐标 $\gamma$ 都有前置教材指定的值空间，记为 $\mathsf{Type}(\gamma)$。定义块的输入坐标签名 $\partial^-K$，它恰好包含：

1. $\bigcup_{\xi\in K}\operatorname{Pred}(\xi)\setminus K$ 中全部作用的值坐标；
2. 在 $K$ 内首次读取、却不由 $K$ 内作用产生的节点状态与选择历史坐标；
3. 进入 $K$ 内时间纤维、却不由 $K$ 内 $F$ 产生的外部记录与跨界消息坐标；
4. 证明 $K$ 中各个 $P$ 所用时间纤维完整的封闭下界坐标。

定义输出坐标签名 $\partial^+K$，使它包含 $K$ 中全部规范作用值，以及这些作用产生并离开块的持久状态、选择历史、内部消息和外部输出。相应的类型化记录空间是：

$$
\mathsf{BndIn}(K)
=\prod_{\gamma\in\partial^-K}\mathsf{Type}(\gamma),
\qquad
\mathsf{BndOut}(K)
=\prod_{\gamma\in\partial^+K}\mathsf{Type}(\gamma).
$$

对 $\omega\in\mathsf{Inst}(K)$，分别把完整参考实例限制到这两组坐标，得到边界抽取函数：

$$
\beta^-_K:\mathsf{Inst}(K)\to\mathsf{BndIn}(K),
\qquad
\beta^+_K:\mathsf{Inst}(K)\to\mathsf{BndOut}(K).
$$

定义：

$$
\begin{aligned}
\mathsf{AdmIn}(K)
&=\{\beta^-_K(\omega)\mid\omega\in\mathsf{Inst}(K)\},\\
\mathsf{Out}(K)&=\mathsf{BndOut}(K).
\end{aligned}
$$

$\partial^-K$ 已经包含每个块外函数自变量以及所需封闭证据。因此，若两个合法实例具有相同的 $\beta^-_K$，对块内事件偏序归纳可知它们也具有相同的 $\beta^+_K$。于是下面的赋值与实例见证的选择无关，并定义一个全函数：


$$
\begin{aligned}
\operatorname{Ref}_K&:
\mathsf{AdmIn}(K)\longrightarrow\mathsf{Out}(K),\\
\operatorname{Ref}_K(\beta^-_K(\omega))
&=\beta^+_K(\omega)
\qquad(\omega\in\mathsf{Inst}(K)).
\end{aligned}
\tag{9}
$$

### 2.3 精确联合求值与精化

一个联合函数

$$
\mathcal K:
\mathsf{AdmIn}(K)
\longrightarrow
\mathsf{Out}(K)
$$

称为 $K$ 的**精确见证**，当且仅当：

$$
\forall z\in\mathsf{AdmIn}(K),\qquad
\mathcal K(z)=\operatorname{Ref}_K(z).
\tag{10}
$$

令 $\mathsf{EvalRec}(K)$ 为具体联合求值记录的集合，并固定只遗忘附加坐标的投影：

$$
\Pi_K^{\mathrm{blk}}:
\mathsf{EvalRec}(K)\longrightarrow\mathsf{Out}(K).
$$

具体记录 $r$ 在边界 $z\in\mathsf{AdmIn}(K)$ 上**精化到**规范块，当 $\Pi_K^{\mathrm{blk}}(r)=\operatorname{Ref}_K(z)$。投影只删除辅助坐标，并保留 $\partial^+K$ 的全部坐标及其原值。式 (10) 因而保证展开后的每个 $P,S,U,F$ 坐标与 $\mathcal T_x$ 一致。

为统一后文记号，定义契约种类与**归属标识**集合：

$$
\begin{aligned}
\mathsf{ContractKind}&=\{F,U,UF,R\},\\
\mathsf{Owner}
&=\bigl(\{\mathrm{node}\}\times V\bigr)
\sqcup
\bigl(\{\mathrm{region}\}\times J\bigr).
\end{aligned}
$$

一个**类型化联合契约**是四元组：

$$
B=(\operatorname{kind}(B),\operatorname{own}(B),K_B,\mathcal K_B),
$$

其中 $\operatorname{kind}(B)\in\mathsf{ContractKind}$，$\operatorname{own}(B)\in\mathsf{Owner}$，$K_B$ 是有限作用块，并且 $\mathcal K_B:\mathsf{AdmIn}(K_B)\to\mathsf{Out}(K_B)$ 是式 (10) 的精确见证。种类 $F,U,UF$ 的归属标识分别是相应节点 $(\mathrm{node},v)$；种类 $R$ 的归属标识是相应 region $(\mathrm{region},j)$。给定 $z\in\mathsf{AdmIn}(K_B)$ 后，二元组 $(B,z)$ 称为一次**契约调用**，它覆盖的规范作用集合是 $K_B$。

多个两两不交的作用块若满足进入边依赖，就可以按任一拓扑顺序替换参考递归。对块顺序作归纳即可得到：

> [!proposition] 命题 1：精确块替换
> 设有限作用块 $K_1,\ldots,K_m$ 覆盖目标区间的全部作用，每个块调用时满足式 (8) 和封闭条件，并具有式 (10) 的精确见证。依次应用这些联合函数，所得区间记录与右切面等于参考记录及 $Q_c$。

## 3. 三类节点契约与一种区域契约

### 3.1 语义作用与成本标记

式 (1) 的四类作用由语义固定。另取有限的成本种类集合 $\mathsf{CostKind}$、子集 $\mathsf{MajorKind}\subseteq\mathsf{CostKind}$，并固定一个与输入值无关的标记规则。它在每次运行上诱导函数：

$$
\operatorname{cost}_x:
\mathscr V_x^{\mathrm{ev}}\longrightarrow\mathsf{CostKind}.
$$

定义主要作用集与普通作用集：

$$
\mathscr V_x^{\mathrm{major}}
=\operatorname{cost}_x^{-1}(\mathsf{MajorKind}),
\qquad
\mathscr V_x^{\mathrm{ordinary}}
=\mathscr V_x^{\mathrm{ev}}
\setminus\mathscr V_x^{\mathrm{major}}.
$$

本文重点研究主要作用；常见标记包括：

- 只把 $F$ 标记为主要作用；
- 把状态递归涉及的 $P,U$ 与 $F$ 都标记为主要作用；
- 把一个融合状态—输出契约所覆盖的 $P,U,F$ 都标记为主要作用。

成本标记与 $P,S,U,F$ 标签相互独立。同一个 $U$ 在轻量计数状态中可以属于普通作用，在大规模记忆更新中可以属于主要作用。一个联合契约可以共同覆盖若干主要作用；所有批次数结论都相对于已经固定的成本标记和联合契约陈述。

### 3.2 逐坐标完整输出契约

固定节点 $v$ 与有限非空时间集合 $\Theta\subseteq\Lambda_v^F(I;x)$。当全部

$$
(q^{\mathrm{cmp}}_{v,\theta},\theta,
h_{v,\theta},c_{v,\theta})
\qquad(\theta\in\Theta)
$$

已经确定时，令：

$$
K^F_{v,\Theta}
=\{F_{v,\theta}\mid\theta\in\Theta\}.
$$

若 $\Theta=\{\theta_1<\cdots<\theta_k\}$，记号 $(a_\theta)_{\theta\in\Theta}^{\uparrow}$ 表示有序元组 $(a_{\theta_1},\ldots,a_{\theta_k})$。定义逐坐标完整输出联合函数及其类型：

$$
\operatorname{BatchFull}_{v,\Theta}:
\mathsf{AdmIn}(K^F_{v,\Theta})
\longrightarrow\mathsf{Out}(K^F_{v,\Theta}),
$$

$$
\operatorname{BatchFull}_{v,\Theta}
\left(
(q^{\mathrm{cmp}}_{v,\theta},\theta,
h_{v,\theta},c_{v,\theta})_{\theta\in\Theta}^{\uparrow}
\right).
\tag{11}
$$

式 (11) 只是把 $\partial^-K^F_{v,\Theta}$ 中的带名字坐标按递增时间展示；这个元组按定义属于 $\mathsf{AdmIn}(K^F_{v,\Theta})$。

其精确契约是：

$$
\operatorname{BatchFull}_{v,\Theta}(\cdots)
=
\left(
\operatorname{Full}_v
(q^{\mathrm{cmp}}_{v,\theta},\theta,
h_{v,\theta},c_{v,\theta})
\right)_{\theta\in\Theta}^{\uparrow}.
\tag{12}
$$

式 (12) 右侧采用规范输出记录的简写：每个 $\operatorname{Full}_v$ 值放到相应 $F_{v,\theta}$ 标签上，并按前置教材式 (27)--(28) 展开成由它确定的消息与外部输出坐标。式 (12) 的各输入坐标在调用前均已知，因而它描述相互独立的函数求值组成的时间批。

### 3.3 因果状态块契约

状态轨迹具有另一种输入形状。设 $\Theta=\{\theta_1<\cdots<\theta_k\}\subseteq\Lambda_v^U(I;x)$。取一个节点局部作用块 $K^U_{v,\Theta}$，它包含这些 $U$，并可包含尚未由块边界给出的相应 $P$。已经完成的 $P,S$ 通过块边界提供准备标签与选择结果；只有候选域属于单一节点的局部选择作用才可以收入这个节点块。凡是一个 $S$ 共同读取多个候选节点的描述量，它与所有参与节点的相应作用一同归入第 3.5 节的区域块。

把已经关闭的时间纤维、从块外进入的选择结果及其他边界量统记为 $z_i$。例如，当每步控制已经在块边界确定时，可以取：

$$
z_i=(B_{v,\theta_i},
\mathbf1[v\in\mathcal A_{\rho(v),\theta_i}],
c_{v,\theta_i}).
$$

连同 $q_{\mathrm{in}}$，这些带名字坐标构成 $\mathsf{AdmIn}(K^U_{v,\Theta})$ 的一个元素；下面的调用记号采用这一规范排列。

从左边界状态 $q_{\mathrm{in}}$ 出发，按时间递增应用块内保留的 $P,S,U$ 参考规则；把最后一个块内作用以后沿空档保持到块右边界的状态记为 $q_{\mathrm{out}}$。由此得到：

$$
\bigl(
(q^{\mathrm{cmp}}_{v,\theta_i},q_v^{\theta_i+1})_{i=1}^{k},
q_{\mathrm{out}}
\bigr).
\tag{13}
$$

定义因果状态联合函数及其完整类型：

$$
\operatorname{BatchState}_{v,\Theta}:
\mathsf{AdmIn}(K^U_{v,\Theta})
\longrightarrow\mathsf{Out}(K^U_{v,\Theta}),
$$

$$
\operatorname{BatchState}_{v,\Theta}
\left(q_{\mathrm{in}},(\theta_i,z_i)_{i=1}^{k}\right)
\tag{14}
$$

是 $K^U_{v,\Theta}$ 的式 (10) 精确见证；其值包含该块的全部规范标签，状态坐标逐项等于式 (13) 的标量递归。调用式 (14) 时给出左边界状态、关闭纤维与块外驱动量；块内控制和中间状态由参考顺序产生。空档保持持久状态，逻辑时间坐标仍保留在每个驱动量中。

这种契约称为**因果状态块契约**。它允许一次联合调用精化为一条包含许多个 $P,U$ 及其局部选择作用的状态链。共同 selector 读取多个节点描述量时，相应 $S$ 与所有参与节点一起进入区域契约。

### 3.4 状态—完整输出联合契约

有些节点自然地在生成状态轨迹时同时给出每个激活坐标的完整输出。令：

$$
K^{UF}_{v,\Theta}
=K^U_{v,\Theta}
\cup
\{F_{v,\theta}\mid
\theta\in\Theta\cap\Lambda_v^F(I;x)\}.
$$

定义：

$$
\operatorname{BatchNode}_{v,\Theta}:
\mathsf{AdmIn}(K^{UF}_{v,\Theta})
\longrightarrow\mathsf{Out}(K^{UF}_{v,\Theta}),
$$

$$
\operatorname{BatchNode}_{v,\Theta}
\left(q_{\mathrm{in}},(\theta_i,z_i)_{i=1}^{k}\right),
\tag{15}
$$

其值包含式 (13) 的完整状态轨迹、右边界状态，以及每个 $\theta_i\in\Lambda_v^F(I;x)$ 的

$$
(f^A_{v,\theta_i},f^O_{v,\theta_i}).
$$

它是 $K^{UF}_{v,\Theta}$ 的式 (10) 精确见证。一次式 (15) 调用经精化同时覆盖 $K^U_{v,\Theta}$ 与相应的 $F$。实现可以采用融合表示，数学输出仍包含足以确定完整规范轨迹的坐标。

### 3.5 区域控制—状态契约

一个 region 的选择在同一时间联合读取所有候选描述量，选择历史又跨时间递归。固定 $j\in J$ 与区间 $I=[b,c)$，定义区域状态作用块：

$$
\begin{aligned}
K^R_{j,I}
={}&\{P_{v,\theta}
\mid v\in\mathcal R_j,\ \theta\in I,
(v,\theta)\in\mathcal E_x^{\mathrm{node}}\}
\\
&\cup
\{U_{v,\theta}
\mid v\in\mathcal R_j,\ \theta\in I,
(v,\theta)\in\mathcal E_x^{\mathrm{node}}\}
\\
&\cup
\{S_{j,\theta}\mid
\theta\in I,\ (j,\theta)\in\mathcal E_x^{\mathrm{sel}}\}.
\end{aligned}
$$

若该区域全部相关时间纤维已经关闭，定义：

$$
\begin{aligned}
\mathsf{RegionIn}_{j,I}
&=\mathsf{AdmIn}(K^R_{j,I}),\\
\mathsf{RegionStateOut}_{j,I}
&=\mathsf{Out}(K^R_{j,I}).
\end{aligned}
$$

区域参考转导是式 (9) 的特例：

$$
\begin{aligned}
\operatorname{RefRegionState}_{j,I}:
\quad &\mathsf{RegionIn}_{j,I}
\longrightarrow\mathsf{RegionStateOut}_{j,I},\\
\operatorname{RefRegionState}_{j,I}
&=\operatorname{Ref}_{K^R_{j,I}}.
\end{aligned}
\tag{16}
$$

输入包括 $(q_v^b)_{v\in\mathcal R_j}$、$y_j^b$ 和区域内全部 $B_{v,\theta}$；输出包括区域内全部 $P,S,U$ 标签、完整节点状态轨迹、完整选择历史轨迹和右边界状态。函数值按逻辑时间递增应用前置教材的规则。

一个区域联合函数

$$
\operatorname{BatchRegionState}_{j,I}:
\mathsf{RegionIn}_{j,I}
\longrightarrow\mathsf{RegionStateOut}_{j,I}
$$

满足区域控制—状态契约，当它在整个合法定义域上等于式 (16)。它可以容纳选择描述量读取节点状态、选择结果再改变下一持久状态的递归。

若还同时返回区域内全部 $F$ 标签，令

$$
K^{RF}_{j,I}
=K^R_{j,I}\cup
\{F_{v,\theta}\mid
v\in\mathcal R_j,\ \theta\in I,
F_{v,\theta}\in\mathscr V_x^F\},
$$

并取一个从 $\mathsf{AdmIn}(K^{RF}_{j,I})$ 到 $\mathsf{Out}(K^{RF}_{j,I})$ 的式 (10) 精确见证；这就是区域状态—输出联合契约。

式 (14) 与式 (16) 首先给出外层因果块的数学边界。联合函数的内部工作量、并行深度和存储量在第 5 节及第 9 节登记；这些量与外层块数分别分析。

### 3.6 契约族

固定一个类型化联合契约词汇 $\mathfrak B$，并要求对每个有限作用集合、契约种类和归属标识只登记有限多个见证。按第 2.3 节已经定义的种类，令：

$$
\mathfrak B^\kappa
=\{B\in\mathfrak B\mid
\operatorname{kind}(B)=\kappa\}
\qquad
(\kappa\in\mathsf{ContractKind}).
$$

于是：

$$
\mathfrak B
=
\mathfrak B^F
\cup\mathfrak B^U
\cup\mathfrak B^{UF}
\cup\mathfrak B^R.
\tag{17}
$$

四部分分别由式 (11)、(14)、(15)、(16) 型见证组成；$\mathfrak B^R$ 也可以包含第 3.5 节的区域状态—输出见证。可以只选择其中一部分。例如，只给出 $\mathfrak B^F$ 时，后文算法退化为先求状态与选择，再批量求完整输出；给出 $\mathfrak B^U$ 或 $\mathfrak B^{UF}$ 时，状态递归本身也获得显式时间块。

为使最大前沿成为确定函数，在 $\mathsf{ContractKind}$、$\mathsf{Owner}$ 和规范作用坐标上各固定一个全序，并给同种类、同归属标识、同覆盖集合的多个登记见证固定优先序。有限集合总能取这样的全序。这些次序都是规格的一部分，不依赖本次作用值。

一个更强的契约族可以把更多规范边收进同一联合函数。后文的最大前沿、阶段数与批次数均写成相对于 $\mathfrak B$ 的量。

## 4. 相对于契约的最大前沿递归

### 4.1 部分求值状态

固定区间 $I=[b,c)$。一个部分求值状态写成：

$$
\Omega
=
(Q_b,E_I,\boldsymbol\sigma,
D,\mathsf{Val},H),
\tag{18}
$$

其中：

$$
E_I=\{e\in E_x\mid b\le\operatorname{time}(e)<c\},
$$

$\boldsymbol\sigma$ 是前置教材定义的输入边与消息边封闭下界族，且对当前 $(E_I,H)$ 有效；$D\subseteq\mathscr V_x^{\mathrm{ev}}$ 是已经由普通作用或联合函数完成的规范作用集合；$\mathsf{Val}$ 是 $D$ 上的类型正确值函数，并满足 $\mathsf{Val}=\mathcal T_x\!\upharpoonright_D$；$H\subseteq M^*$ 是已经进入当前截面的内部消息。要求 $(D,\mathsf{Val},H)$ 能嵌入前置教材的一条合法 $P/S/U/F$ 暴露轨迹。

一个普通作用 $\xi\in\mathscr V_x^{\mathrm{ordinary}}$ 在 $\Omega$ 中**可取**，当它的直接前驱属于 $D$，其时间纤维已经关闭，并且当前成本标记允许逐项求值其局部函数。

对契约 $B\in\mathfrak B$，若 $\Omega$ 已经确定 $z_B\in\mathsf{AdmIn}(K_B)$ 的全部坐标，就得到调用 $(B,z_B)$。这个调用在 $\Omega$ 中**可取**，当：

1. $K_B\cap D=\varnothing$；
2. 每条从块外进入块内的事件边起点属于 $D$；
3. 契约定义域要求的时间纤维、左边界状态、选择结果或驱动量已经确定；
4. 相关封闭下界覆盖块内全部 $P$；
5. 调用只读取 $\Omega$ 中已有的值。

第三项对不同契约呈现不同形状。`BatchFull` 需要每个坐标的计算快照；因果状态块只需要左边界状态和整段驱动量；区域状态块需要区域左边界与整段关闭纤维。

### 4.2 普通闭包

定义 $\operatorname{CtrlCl}(\Omega)$ 为反复加入所有可取普通作用直到集合稳定所得的状态。事件集合有限且每一步严格扩大 $D$，所以闭包在有限步后存在。当若干普通作用同时可取时，可以按任一顺序加入；命题 1 保证最终标签均取自同一参考记录。

普通闭包可以包含选择、轻量状态计数和 seal 证据的传播。被成本标记为主要作用的状态转移留给 $\mathfrak B^U$ 或 $\mathfrak B^{UF}$，因而不会在闭包中逐时间消耗。

### 4.3 最大可取块族

令 $\mathcal C_{\mathfrak B}(\Omega)$ 为 $\operatorname{CtrlCl}(\Omega)$ 上全部当前可取的调用 $(B,z_B)$。有限目标作用集、每个覆盖集合上的有限登记数以及第 3.6 节的固定次序保证它是有限有序集合。对同一契约种类和同一 $\operatorname{own}(B)$，$K_B$ 严格包含较多当前坐标的调用排在前面；其余次序依次由契约优先序、归属标识全序和坐标全序决定。

按这个全序扫描 $\mathcal C_{\mathfrak B}(\Omega)$。一个调用与此前已经选出的块两两不交时就收入当前族，否则跳过；扫描持续到候选集合末尾。所得族记为：

$$
\mathsf{Front}_{\mathfrak B}(\Omega).
\tag{19}
$$

它是当前全部可取调用中的极大不交族：任何未选调用都与某个已选调用相交。同一 $\operatorname{own}(B)$ 的大块优先给出尽可能长的当前坐标集合，继续扫描又会把所有与已选块相容的剩余调用加入本阶段。固定全序只消除表示歧义；它不读取未来作用值，也不改变任一块的参考函数。

### 4.4 最大前沿算法

定义递归 $\operatorname{EagerBlock}_{\mathfrak B}$：

1. 从当前 $\Omega_r$ 求普通闭包 $\overline\Omega_r=\operatorname{CtrlCl}(\Omega_r)$；
2. 取 $\mathsf{Front}_{\mathfrak B}(\overline\Omega_r)$；
3. 同时求值其中全部联合函数；
4. 把返回的规范标签加入 $D$，把实际内部消息加入 $H$，更新由这些值推出的封闭下界，得到 $\Omega_{r+1}$；
5. 当目标区间的全部作用已经完成并形成 $Q_c$ 时停止。

一次循环称为一个**自适应阶段**。同一阶段的联合函数只读取阶段开始以前的记录，其返回值在下一次普通闭包中使用。

若当前前沿为空而目标作用仍未完成，契约族需要提供一个可取的单作用见证，或者把该作用列为普通作用。称满足这项进展条件的 $\mathfrak B$ **完备**。

> [!theorem] 定理 2：最大前沿递归的终止与精确性
> 对任意有限输入、有限目标区间和完备精确契约族 $\mathfrak B$，$\operatorname{EagerBlock}_{\mathfrak B}$ 在有限阶段后停止；其返回区间记录与右切面逐坐标等于参考语义。

**证明。** 每个普通闭包和每个前沿阶段都只加入尚未完成的规范作用。完备性保证目标尚未完成时至少加入一个作用；有限事件 DAG 因而给出终止。每个普通作用取参考函数值，每个联合调用满足式 (10)，并且所有进入边前驱已完成。对实际加入顺序应用命题 1，得到完整区间记录与 $Q_c$。$\square$

### 4.5 契约相对的自适应秩

固定一次运行实际选出的联合调用，把每个调用收缩为一个宏作用；普通闭包中的作用收缩为零成本边。若宏作用 $B'$ 的调用输入读取 $B$ 的回答，写成：

$$
B\prec_{\mathfrak B} B'.
$$

所得有限偏序称为本次运行的**宏作用因果图**。定义：

$$
r(B)
=
1+\max_{B'\prec_{\mathfrak B}B}r(B'),
\qquad
\max\varnothing=0.
\tag{20}
$$

在“一个阶段中确定全部调用，回答于阶段末共同进入记录”的模型下，任何因果策略都必须把 $B$ 放在不小于 $r(B)$ 的阶段。最大前沿递归在每个回答层立即提交全部当前可取宏作用，因此：

> [!theorem] 定理 3：固定宏作用图上的阶段最优性
> 对固定精确契约族、固定优先规则和由此确定的宏作用分解，最大前沿递归使用 $\max_B r(B)$ 个自适应阶段；任何对同一组宏作用求值、遵守相同阶段可见性条件的因果策略至少需要这么多阶段。

**证明。** 每条 $B'\prec_{\mathfrak B}B$ 都要求 $B$ 严格晚于 $B'$ 的回答阶段，对式 (20) 归纳得到下界。对最大前沿实际产生的固定不交分解，若一个宏作用已经可取，它与较早阶段选出的宏作用不相交，式 (19) 的完整扫描会在当前阶段收入它；因此它只会等待尚未公开的宏前驱。对秩归纳可知第 $k$ 层在第 $k$ 个阶段全部提交，给出匹配的上界。$\square$

这个结论比较相同的联合函数词汇。增加一个精确的长状态块或状态—输出联合契约，会收缩原宏作用图并可能降低自适应秩。

### 4.6 三类节点批次数与区域块数

对一次区间求值，令 $\mathcal L$ 为各阶段实际选中契约调用的有限集合；以下的 $B$ 均遍历 $\mathcal L$，并用 $B$ 同时表示调用及其类型化契约部分。定义：

$$
m_v^U
=
\#\{B\in\mathcal L\mid\operatorname{kind}(B)=U,
\operatorname{own}(B)=(\mathrm{node},v)\},
$$

$$
m_v^F
=
\#\{B\in\mathcal L\mid\operatorname{kind}(B)=F,
\operatorname{own}(B)=(\mathrm{node},v)\},
$$

$$
m_v^{UF}
=
\#\{B\in\mathcal L\mid\operatorname{kind}(B)=UF,
\operatorname{own}(B)=(\mathrm{node},v)\},
$$

$$
m_j^R
=
\#\{B\in\mathcal L\mid\operatorname{kind}(B)=R,
\operatorname{own}(B)=(\mathrm{region},j)\}.
\tag{21}
$$

再令 $R$ 为自适应阶段数。一次联合块同时覆盖 $U$ 与 $F$ 时，它只计入 $m_v^{UF}$；若分析者还需要两个语义投影的覆盖数，可以从该块所含 $U,F$ 坐标分别统计。

这里的**共同规格族** $\mathfrak S$ 是一组共享下列数据的 TimedDAG 规格：

1. 有限集合 $V,A,J$、端口集合、映射 $\rho,\operatorname{src},\operatorname{dst}$、时延与输入时间规则；
2. 节点状态、选择历史、本地内容、描述量、控制量和输出的值空间，以及全部局部函数的定义域和值域；
3. 第 3.1 节的成本标记规则、联合契约的共同形状规则与全部固定全序。

一个**共同契约模式**是按规格索引的契约族 $(B_{\mathcal G})_{\mathcal G\in\mathfrak S}$：所有 $B_{\mathcal G}$ 具有相同契约种类、归属标识和作用坐标选择规则，而其见证函数分别在规格 $\mathcal G$ 的合法边界域上满足式 (10)。第 3.6 节的词汇现在理解为一组固定的共同契约模式。族中规格可以改变共同类型内的局部函数值，见证函数随 $\mathcal G$ 取相应成员；模式、调度规则与下列常数不随 $\mathcal G$ 改变。因此，这些常数由共同的 $V,J$ 索引，量词也确实比较同一组节点与 region。

称共同规格族 $\mathfrak S$ 具有**一致节点时间批暴露**，当存在上述固定完备精确契约词汇和常数：

$$
C_R,\qquad
(C_v^U,C_v^F,C_v^{UF})_{v\in V},
\qquad
(C_j^R)_{j\in J},
\tag{22}
$$

使任意规格 $\mathcal G\in\mathfrak S$、任意输入位置 $q$、任意长度 $T$ 和任意合法左边界都满足：

$$
\begin{gathered}
R\le C_R,\\
m_v^U\le C_v^U,\qquad
m_v^F\le C_v^F,\qquad
m_v^{UF}\le C_v^{UF},\\
m_j^R\le C_j^R.
\end{gathered}
\tag{23}
$$

这些常数独立于 $T$。式 (23) 计量外层分块；联合函数内部的标量步数和并行深度另行登记。

### 4.7 最大前沿与总批次数

最大前沿递归让固定宏作用图中的每个当前可取块尽早进入，因此达到定理 3 的最少自适应阶段。一个位于非关键路径、已经可取的坐标若被延迟，有时可以与以后出现的同节点坐标合并，从而减少总批次数，同时保持或增加完成阶段。

一次运行的总联合块数可以记为：

$$
B_{\mathrm{tot}}
=
\sum_{v\in V}(m_v^U+m_v^F+m_v^{UF})
+\sum_{j\in J}m_j^R.
\tag{24}
$$

因此，最少自适应阶段与最小化式 (24) 是两项目标。本文采用前者构造在线递归；第 9 节直接记录实际得到的式 (21)、(24) 和阶段数，供具体实现比较延迟合批的收益。

## 5. 状态递归、selector 与动态切口

### 5.1 状态顺序与因果状态块

对同一节点的实际事件时间 $\theta_1<\cdots<\theta_k$，规范记录含有链：

$$
U_{v,\theta_1}\to P_{v,\theta_2}\to
U_{v,\theta_2}\to\cdots\to P_{v,\theta_k}\to U_{v,\theta_k}.
\tag{25}
$$

式 (14) 可以把这条链及其中尚未由边界给出的节点局部 $P/S$ 收进一个因果状态块；跨节点共同选择则使用式 (16) 的区域块。规范顺序由联合函数内部保持，块外只提供进入边界并接收完整轨迹与右边界。

对于 Attention，可令状态保存键值前缀。整段输入的键值追加形成一个因果状态块，完整输出使用各逻辑时间对应的前缀；一个状态—输出联合契约可以用因果遮罩共同给出所有坐标。

对于仿射 SSM：

$$
q^{\theta+1}=A_\theta q^\theta+b_\theta,
\tag{26}
$$

一步转移由 $(A_\theta,b_\theta)$ 表示，复合满足：

$$
(A_2,b_2)\circ(A_1,b_1)
=
(A_2A_1,A_2b_1+b_2).
\tag{27}
$$

结合律允许前缀扫描给出全部状态。式 (14) 负责精确性；式 (27) 进一步给出一种低并行深度见证。

### 5.2 选择控制的可用时刻

状态块的驱动量包含激活指示和局部控制 $c_{v,\theta}$。它们由 $S_{\rho(v),\theta}$ 产生。若整段选择结果可以先由普通闭包确定，节点状态块可以直接跨越整个时间集合。

若时间 $\theta'$ 的选择描述量读取前一次状态，则会出现：

$$
U_{v,\theta}
\longrightarrow
P_{v,\theta'}
\longrightarrow
S_{j,\theta'}
\longrightarrow
U_{v,\theta'}.
\tag{28}
$$

节点级式 (14) 到 $S_{j,\theta'}$ 为止拥有一个外部边界。区域契约式 (16) 可以在具有相应精确见证时吸收这条边；否则，$S_{j,\theta'}$ 把两个状态块分开。

因此，`Next` 读取 $c$ 足以表达激活后清空、激活计数和阈值状态等语义。相应状态批还需要整段 $c$ 已经确定，或由更大的区域联合契约在内部产生。

### 5.3 完整输出反馈形成的切口

若一个完整输出产生的消息参与未来输入、选择或状态驱动，就会形成：

$$
F
\longrightarrow
P/S/U
\longrightarrow
F\ \text{或}\ U.
\tag{29}
$$

左侧 $F$ 的值在返回以前，右侧联合调用的实际输入尚未确定。最大前沿递归在这里结束当前阶段，并在回答进入部分状态以后继续。

更一般地，真正的外层切口具有形状：

$$
\text{联合求值块}
\longrightarrow
\text{块外必须读取的函数作用}
\longrightarrow
\text{后续联合求值块}.
\tag{30}
$$

状态链若完全属于式 (14) 或 (16) 的定义域，就成为块内依赖；只有离开契约边界并被后续决策读取的中间值才增加自适应秩。

### 5.4 TotalEmit profile

定义 **TotalEmit** 条件：

$$
v\in\mathcal A_{\rho(v),\theta}
\Longrightarrow
\forall a\in\operatorname{Out}(v),
\quad f^A_{v,\theta}(a)\in P.
\tag{31}
$$

在这个函数类中，节点一旦激活，每条出边都有一条实际消息。消息的存在性由 active set 确定，载荷仍由 $F$ 给出。最大前沿递归及四类契约保持式 (11)--(19) 的形状；式 (31) 改变具体运行中出现的消息边和后继作用密度。

payload 可以继续影响未来聚合、selector 和状态，所以 TotalEmit profile 仍可能反复实例化式 (29)。它适合作为常见函数 profile 单独登记。

## 6. 严格分层 region 的整块定理

### 6.1 区域商图与严格分层

定义区域商图边关系：

$$
Q_\rho
=
\{(j,k)\in J^2\mid j\ne k,
\exists a\in A:
\rho(\operatorname{src}(a))=j,
\rho(\operatorname{dst}(a))=k\}.
\tag{32}
$$

称 region 划分**严格分层**，当存在函数 $\ell:J\to\mathbb N$ 满足每条空间边 $a$ 都有：

$$
\ell(\rho(\operatorname{src}(a)))
<
\ell(\rho(\operatorname{dst}(a))).
\tag{33}
$$

式 (33) 同时推出：region 内没有空间边，且 $(J,Q_\rho)$ 是 DAG。具有相同 $\ell$ 值的 region 之间也没有空间边。

### 6.2 区域时间块

固定 $I=[b,c)$。对 region $j$，假设已经给定：

1. $(q_v^b)_{v\in\mathcal R_j}$ 与 $y_j^b$；
2. 区域内全部关闭时间纤维 $B_{v,\theta}$；
3. 从左切面和更早层 region 进入这些纤维的全部消息。

式 (16) 唯一确定区域内的 $P,S,U$ 轨迹。随后，式 (12) 可以按节点求全部 $F$；式 (15) 则可以在具有联合见证时同时求状态与输出。

### 6.3 分层推进算法

取 region 顺序 $j_1,\ldots,j_m$，使：

$$
\ell(j_1)\le\cdots\le\ell(j_m).
$$

从 $Q_b$ 开始，对 $r=1,\ldots,m$：

1. 汇集 $\mathcal R_{j_r}$ 在 $I$ 内的外部记录、跨切面消息与更早层 region 消息；
2. 用入边和输入端口的封闭下界证明这些纤维完整；
3. 应用一个精确区域状态契约，或应用它允许的节点状态契约族；
4. 对尚未由联合契约覆盖的 $F$ 按节点应用式 (12)；
5. 把右边界状态、选择历史和发往更后层的消息加入当前部分状态。

式 (33) 保证当前 region 的区间输入只来自左切面、外部输入或已经处理的更早层 region。

> [!theorem] 定理 4：严格分层整块定理
> 在严格分层条件下，若每个 region 的 $P/S/U$ 区间转导具有精确区域状态见证，且每个节点的剩余 $F$ 具有式 (12) 的精确见证，则上述算法从 $Q_b$ 精确推进到 $Q_c$。每个 region 使用一个状态块，每个节点至多使用一个额外完整输出时间批。

**证明。** 对 region 顺序归纳。处理 $j_r$ 时，每条入空间边来自更小的 $\ell$，相应区间消息已经由归纳前缀产生；跨切面消息属于 $Q_b$，外部输入由输入封闭下界覆盖。因此区域时间纤维完整。区域状态见证给出参考 $P/S/U$ 轨迹，节点 $F$ 见证给出参考输出与消息。处理完所有 region 后，目标区间全部作用与参考记录一致；命题 1 给出 $Q_c$。$\square$

若整个 region 具有区域状态—输出联合见证，可以把定理中的区域状态块与逐节点 $F$ 批融合为一个块。在选择结果已经从边界给定、各节点递归彼此分离的特例中，也可以分别使用式 (15) 的节点联合见证。若区域选择历史只需顺序递归，定理仍给出一个外层状态块；其块内并行深度需要另行分析。

### 6.4 按层最大前沿

具有相同 $\ell$ 值的 region 可以位于同一个最大前沿阶段。令 $H_\rho$ 为区域商图最长路径上的顶点数；这里允许只含一个顶点、零条边的路径，所以非空 $J$ 总有 $H_\rho\ge1$。于是：

$$
R\le 2H_\rho
\tag{34}
$$

这是定理 4 中“一个区域状态块，随后一层逐节点完整输出批”的阶段界。若每个 region 具有区域状态—输出联合见证，则：

$$
R\le H_\rho.
$$

若区域状态作用全部属于同一阶段的普通闭包，只有逐节点完整输出属于主要作用，也得到后一界。顺序逐 region、并把状态与输出分开的实现给出较宽松的 $R\le2|J|$。这些界以及主时间区间内的状态块数和完整输出批数都与 $T$ 无关。输入改变每个块中实际出现的坐标集合，而严格分层次序保持不变。

## 7. 典型结构与边界

### 7.1 串行 Transformer block

把每个 Transformer 层或每个具有共同选择的模块放入一个 region，并令空间边只从较早层指向较后层，就得到严格分层结构。对一个已知输入 chunk：

- 每层 Attention 可以用式 (15) 表示旧 KV 状态、整段新输入、全部因果输出和新 KV 状态；
- 逐位置前馈网络可以用式 (12) 表示；
- 层间消息在前一层完成以后一次提供给后一层。

当层内没有跨位置自适应 selector 时，每层主时间块通常对应一个状态—输出联合块或一个状态块加一个完整输出批。

### 7.2 SSM 层

对满足式 (26) 的 SSM，$A_\theta,b_\theta$ 若在状态块开始以前已知，就可以使用式 (27) 的扫描见证。输出若为：

$$
o_\theta=C_\theta q^\theta+d_\theta,
$$

可由式 (15) 与状态轨迹共同返回。输入依赖的 $A_\theta$ 仍可批量形成，只要其计算不等待同块内尚未产生的状态或完整输出值。

### 7.3 MoE region

设一个 region 含多个 expert，selector-history 保存负载摘要。区域状态契约先产生整段 active sets 和控制量。对 expert $e$ 定义：

$$
\Theta_e
=
\{\theta\in I\mid e\in\mathcal A_{\rho(e),\theta}\}.
\tag{35}
$$

随后对每个 expert 应用一个 $\operatorname{BatchFull}_{e,\Theta_e}$。不同 expert 的时间集合可以具有不同大小；式 (35) 正是运行时实际形成的 ragged 时间批。

若后一个时间的路由读取前一个 expert 的完整输出，式 (29) 会把区域时间块切开。若 history 只读取已知描述量和过去 active set，区域状态契约可以先完成整段选择。

### 7.4 不同到达时刻的重新汇合

当同一节点存在不同到达时刻的路径时，多个输入位置可以在同一时间纤维汇合。算法仍按 $B_{v,\theta}$、$P/S/U/F$ 和 $Q_b$ 工作。一个节点时间批由逻辑时间坐标集合定义，因此无需为每个事件指定唯一输入位置。

实际跨界内容由 $W_b$ 给出。状态块输入保留真实逻辑时间，所以从时间 $2$ 到时间 $10$ 的空档始终具有长度 $8$。

### 7.5 区域商图中的回返

固定节点图是 DAG，区域商图仍可因交错划分而含环。例如路径 $v_0\to v_1\to v_2\to v_3$ 按奇偶节点分为两个 region 时，商图含双向边。

这种划分会在 region 之间形成时间拉链。最大前沿算法仍然精确终止，但主区间可能分成随 $T$ 增长的多个状态块或完整输出批。把商图强连通分量收缩成较大的空间块可以恢复凝聚 DAG；分量内部仍需一个精确区域契约和相应批次数分析。

### 7.6 全局 selector

若所有节点属于同一个 region，选择历史可以跨越全部拓扑层。一个区域状态契约在数学上可以覆盖整段 $P/S/U$ 轨迹；它是否具有紧凑表示、低工作量或低并行深度，取决于 selector 与状态转移的结构。

若消息载荷反复改变全局 selector 的下一次选择，就会出现多轮式 (29)。此时单一大 region 的名称没有减少自适应秩；精确契约必须真实接收每轮所需的前序值。

### 7.7 静态拓扑与运行时形状

固定空间图、region 划分和契约族给出所有可能作用块的静态包络。输入、状态、selector 与函数输出共同决定一次运行的：

$$
\Lambda_v^U(I;x),\qquad
\Lambda_v^F(I;x),\qquad
\mathsf{Front}_{\mathfrak B}(\Omega_r).
$$

严格分层结构固定外层层次；输入主要改变块内坐标。具有回返的结构还会使块边界本身随输入改变。最大前沿递归在每个阶段从当前部分状态直接构造这三个对象。

## 8. 切面组合、因果性与分段继续

### 8.1 chunk 组合

先定义本节使用的区间转导。令 $\mathsf{Cut}(a)$ 为时间 $a$ 上全部类型正确切面状态 $Q_a$ 的集合，$\mathsf{SegIn}[a,b)$ 为区间外部记录与覆盖该区间的有效封闭下界所成的集合，$\mathsf{SegRec}[a,b)$ 为该区间全部类型正确规范记录的集合。合法输入域

$$
\mathsf{AdmRun}[a,b)
\subseteq
\mathsf{Cut}(a)\times\mathsf{SegIn}[a,b)
$$

只保留能够由前置教材的某个完整记录产生的左切面和区间输入。分段继续定理定义全函数：

$$
\operatorname{Run}_{[a,b)}:
\mathsf{AdmRun}[a,b)
\longrightarrow
\mathsf{SegRec}[a,b)\times\mathsf{Cut}(b),
$$

它返回唯一参考区间记录及其右切面。对相邻区间，符号 $\operatorname{Run}_{[b,c)}\circ\operatorname{Run}_{[a,b)}$ 表示把第一次返回的 $Q_b$ 连同第二段输入交给第二个转导，并把两段规范记录作不交并合。

取 $a<b<c$。每个 $\mathsf{AdmRun}[a,c)$ 元素按时间限制得到两段区间输入；以下等式是在这个共同定义域上的转导等式。前置教材的继续定理给出：

$$
\operatorname{Run}_{[b,c)}
\circ
\operatorname{Run}_{[a,b)}
=
\operatorname{Run}_{[a,c)}.
\tag{36}
$$

每个 $\operatorname{Run}$ 可以由不同的精确契约族分解，只要中间切面完整保留节点状态、选择历史和跨界消息。式 (36) 支持把长输入拆成多个 chunk，也支持在相邻 chunk 之间改变联合求值形状。

### 8.2 输入前缀不变性

若输入封闭下界覆盖目标切面 $c$，未来再增加时间不小于 $c$ 的外部记录，不会改变 $[b,c)$ 的参考记录。最大前沿递归的每个 $P$ 又要求相应纤维关闭，所以它只使用当前已经由 seal 保证稳定的候选集合。

这项性质允许在线地推进切面：输入源先公开一个有限 chunk 和相应封闭下界，算法完成 $Q_c$，随后以 $Q_c$ 作为下一段左边界。

### 8.3 自回归输出的结构检查

用于预测输入位置 $t+1$ 的某个输出作用，其事件 DAG 祖先应只包含位置不大于 $t$ 的外部输入记录。记作用 $\xi$ 的外部祖先集合为 $\operatorname{Anc}_{\mathrm{ext}}(\xi)$；结构条件写成：

$$
\operatorname{Anc}_{\mathrm{ext}}(\xi)
\subseteq
\{(i,k)\mid k\le t\}.
\tag{37}
$$

分块、状态扫描和联合 Full 契约都通过精化保留原事件 DAG 的语义坐标，因此不会改变式 (37) 的真值。

### 8.4 分块边界上的状态

完成 $[b,c)$ 后，下一段需要：

$$
Q_c
=
\left(
c,(q_v^c)_{v\in V},(y_j^c)_{j\in J},W_c
\right).
\tag{38}
$$

联合状态函数可以在内部采用压缩表示；其右边界必须映射到式 (38) 的规范状态。若还要重建完整历史记录，则同时保存各块的细粒度 $P/S/U/F$ 标签或一种可逆编码。

## 9. 性能分析应记录的量

### 9.1 静态记录

对每个规格与契约族，记录：

- 空间 DAG、region 划分与区域商图；
- 每个节点可用的 $\mathfrak B^F,\mathfrak B^U,\mathfrak B^{UF}$；
- 每个 region 可用的 $\mathfrak B^R$；
- 成本标记与联合函数内部的理论工作量、并行深度和存储界。

空间 DAG、region 划分、成本标记和契约族共同决定最大前沿算法的合法动作空间。

### 9.2 每次运行的动态记录

对每个输入 chunk，记录：

$$
R,\qquad
(m_v^U,m_v^F,m_v^{UF})_{v\in V},
\qquad
(m_j^R)_{j\in J},
\tag{39}
$$

以及每个块的坐标集合大小、边界状态大小、消息数、候选数和 active 数。进一步可记录：

- 每轮普通闭包中的 $P,S,U,F$ 数量；
- 每个动态切口对应的事件 DAG 边；
- TotalEmit 下实际实例化的全部出边消息；
- 联合函数的实际耗时、设备利用率与内存访问量。

式 (39) 直接测量真实拓扑和真实输入上的节点级时间批效果。

### 9.3 四个相互独立的结论

一项完整分析分别陈述：

1. **语义精确性**：精化结果等于 $\mathcal T_x$；
2. **外层分块性**：式 (23) 是否具有与 $T$ 无关的界；
3. **块内算法性**：联合函数的工作量和并行深度；
4. **设备效果**：在具体实现上的时间、吞吐、存储和通信测量。

Attention 的因果遮罩、SSM 的结合前缀复合和各 expert 的不等长时间集合分别为第三项提供不同见证。第四项由实验获得。

## 10. 主要结论

本文得到以下结构：

1. TimedDAG 的规范结果仍由逻辑时间递归 $x\mapsto\mathcal T_x$ 定义；
2. $P,S,U,F$ 事件 DAG 给出细粒度因果关系；
3. `BatchFull` 处理已经知道全部逐坐标输入的完整输出时间批；
4. `BatchState` 从左边界状态和整段驱动量产生因果状态轨迹；
5. `BatchNode` 同时产生状态轨迹与完整输出；
6. `BatchRegionState` 容纳跨节点选择与状态的共同递归；
7. 最大前沿递归相对于固定契约族在线构造当前最大的合法求值形状；
8. 在固定宏作用图上，它达到最少自适应阶段；
9. 节点状态批数、完整输出批数、联合批数和区域状态块数分别计量；
10. 严格分层 region 加精确状态与输出契约给出与 chunk 长度无关的外层批次数。

核心关系可以概括为：

$$
\boxed{
\begin{array}{c}
\text{封闭时间纤维}
+
\text{精确因果联合契约}
\\[1mm]
+
\text{契约相对的最大前沿}
\\[1mm]
\Longrightarrow
\text{精确的状态批与完整输出批};
\\[1mm]
\text{严格分层}
\Longrightarrow
\text{与 chunk 长度无关的外层块数}.
\end{array}
}
\tag{40}
$$

## 11. 建议练习

1. 对一个长度为 $T$ 的 KV 追加状态，写出式 (14) 的定义域、全部概念中间状态和右边界状态。
2. 对仿射 SSM 验证式 (27) 的结合律，并说明全部 $(A_\theta,b_\theta)$ 需要在扫描以前满足什么可用条件。
3. 构造一个 $c_{v,\theta}$ 依赖前一状态的节点例子，分别写出节点状态块与区域状态块的边界。
4. 对两层严格分层图写出最大前沿的每个阶段，分别使用 $\mathfrak B^F$ 和 $\mathfrak B^{UF}$。
5. 构造一条 $F\to P\to S\to U\to F$ 的反馈路径，说明哪一个值使后一个联合调用在前一阶段尚不可取。
6. 在 TotalEmit profile 下，给定 active sets，列出所有必然存在的消息坐标；再说明 payload 未知会保留哪些动态切口。
7. 为一个 MoE region 计算式 (35) 的 ragged 时间集合，并分别统计 $m_e^U,m_e^F,m_e^{UF}$。

---

## 附录 S：计算机系统词汇与数学对象的对应

> [!info]- S.1　prefill、chunk 与 logical-time tile
> `prefill` 对应：一段外部输入记录已经给定，并具有覆盖目标右切面的封闭下界；目标是从 $Q_b$ 精确推进到 $Q_c$。
>
> `input chunk` 对应式 (5) 所含输入位置。`logical-time tile` 对应区间 $I_{q,T}$ 内的一组 $P/S/U/F$ 作用。路径重新汇合时，一个 tile 可以混合多个输入位置。

> [!info]- S.2　batch、causal batch 与 ragged batch
> `batch` 在式 (11) 中对应一组逐坐标输入均已确定的 $F$ 作用。
>
> `causal batch / sequence chunk` 对应式 (14)：输入是左边界状态和有序驱动序列，中间状态由联合函数产生。
>
> `ragged batch` 对应每个节点各自的有限时间集合，例如式 (35) 的 $\Theta_e$；不同节点的集合大小可以不同。

> [!info|keep]- S.3　state update、KV cache 与 SSM scan
> `state update` 对应规范记录中由 $P$ 的候选状态计算、$S$ 的控制和 $U$ 的状态采用共同确定的持久状态转移。它在节点级可由式 (14) 表示，在具有共同 selector 时可由式 (16) 表示。
>
> `KV-cache update` 可以精化为一条状态轨迹；`causal attention` 可以由式 (15) 同时返回各前缀输出和右边界 KV 状态。
>
> `SSM scan` 是式 (26)--(27) 的具体联合见证。其低并行深度来自紧凑转移摘要的结合复合。

> [!info]- S.4　kernel、fusion 与 launch
> `kernel` 对应式 (10) 的某个具体联合函数。`fusion` 对应一个式 (15) 或更大的精确作用块同时覆盖多类规范作用。
>
> `launch count` 是具体实现发起联合函数的次数。式 (21) 记录各类数学块数，式 (24) 记录其总数；一次实现调用可以承载多个两两独立的数学块，也可以让一个联合块由多个实现调用完成。二者通过具体执行记录的精化投影关联。

> [!info]- S.5　heavy/control 与 adaptive round
> `cost profile` 对应第 3.1 节的成本标记和式 (17) 的契约族。
>
> `control closure` 对应 $\operatorname{CtrlCl}$。`heavy action` 对应被成本标记为主要作用、并通过 $\mathfrak B$ 求值的作用块。
>
> `heavy/control round` 对应第 4.4 节的一次自适应阶段。数学切口由式 (30) 给出：前一联合块的回答必须先被块外函数读取，随后才能确定后一联合块。

> [!info]- S.6　maximal frontier 与 scheduler
> `maximal frontier` 对应式 (19) 的极大相容可取块族。`scheduler` 对应固定契约优先序、owner 全序以及每阶段应用式 (19) 的递归函数。
>
> `online / no oracle` 表示每次选择只读取式 (18) 的当前部分状态；未来 $F$ 值、未来 selector 结果和未公开输入均不属于其自变量。

> [!info]- S.7　visibility、publication 与 completion
> 前置教材的 filtration 截面分别保存 $\mathsf P_n,\mathsf S_n,\mathsf U_n,\mathsf F_n,H_n$。具体执行的阶段末公开某些联合函数回答时，经阶段精化映射把相应规范作用加入这些集合。
>
> `message publication` 对应消息进入 $H_n$。源 $F$ 已经进入 $\mathsf F_n$ 是其前提。派生的 $\mathsf{Completed}_n$ 同时要求节点的 $U$ 已经进入，并要求激活节点的 $F$ 已经进入。

> [!info|keep]- S.8　continuation、checkpoint 与 resume
> `continuation` 对应式 (38) 的 $Q_c$。`checkpoint` 是 $Q_c$ 的可保存编码；无损编码满足读取以后恢复同一数学对象。
>
> `resume` 对应以 $Q_c$ 为左边界继续下一段参考递归。式 (36) 给出相邻 chunk 的组合等式。

> [!info]- S.9　work、span、latency 与 throughput
> `work` 是选定计算模型以后执行的基本操作总数，`parallel span` 是这些操作依赖图的最长链。式 (23) 只界定外层块数和自适应阶段数；它不界定联合函数内部的 parallel span。
>
> `latency`、`throughput`、设备利用率、内存带宽与通信量属于第 9.2 节的实测记录。它们与状态批数、完整输出批数共同构成性能报告。

> [!info]- S.10　refinement
> `refinement` 是具体执行记录到规范 $P/S/U/F$ 记录的固定投影。一次 `BatchState`、`BatchFull` 或 `BatchNode` 调用可以映射为多个规范作用；投影保留这些作用的原值与逻辑时间，只删除调用编号、线程、设备、缓存和墙钟坐标。
