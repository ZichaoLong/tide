---
type: mathematical-learning-note
status: active-learning
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
> 本文研究同一份 $\mathcal T_x$ 怎样按较大的时间块精确求值。核心对象依次是：作用块、联合求值契约、最大前沿递归、严格分层正例和静态分量窗口调度。状态递归与完整输出分别计量，也允许由同一个联合函数同时求值。
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

固定一个带区域选择的 TimedDAG 规格。节点、边、区域与归属映射分别记为：

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

### 1.2 正则输入流与输入分块区间

为集中说明时间分块，取一个正整数时间步长 $\Delta_{\mathrm{in}}$，令输入位置 $t$ 的逻辑时间为 $\Delta_{\mathrm{in}} t$：

$$
t\longmapsto \Delta_{\mathrm{in}} t,
\qquad
\Delta_{\mathrm{in}}\in\mathbb N_{>0}.
\tag{4}
$$

固定起始位置 $q$ 与长度 $T>0$，相应主时间区间为：

$$
I_{q,T}
=
[\Delta_{\mathrm{in}} q,\Delta_{\mathrm{in}}(q+T)).
\tag{5}
$$

它的左、右切面分别记为 $b=\Delta_{\mathrm{in}} q$ 与 $c=\Delta_{\mathrm{in}}(q+T)$。前置教材的切面状态是：

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

其中 $W_b$ 保存从左侧发送、在 $b$ 以后到达的内部消息。给定 $Q_b$ 和区间内的完整外部输入，区间转导 $\Phi_{b,c}$ 唯一确定区间记录与 $Q_c$。覆盖 $c$ 的输入封闭下界用于证明输入完整，作为求值的合法性证据。

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

固定目标区间 $I=[b,c)$，记 $\mathscr V_x^{\mathrm{ev}}(I)$ 为区间内的作用。给定已经完成的作用集合 $D\subseteq\mathscr V_x^{\mathrm{ev}}(I)$，块 $K\subseteq\mathscr V_x^{\mathrm{ev}}(I)$ 的进入依赖条件为：

$$
\forall\xi\in K,
\quad
(\operatorname{Pred}(\xi)\cap\mathscr V_x^{\mathrm{ev}}(I))\setminus K\subseteq D,
\tag{8}
$$

这允许块内保留依赖边。区间内的块外前驱须已完成；跨越左切面的状态、历史与消息依赖由 $Q_b$ 的相应坐标给出。

若 $K$ 含有 $P_{v,\theta}$，还要求相应时间纤维由有效封闭下界证明完整。若 $K$ 含有 $S_{j,\theta}$，同一 $(j,\theta)$ 的全部候选 $P$ 都属于 $K\cup D$，并且旧选择历史由左边界或块内更早的 $S$ 给出。

### 2.2 块边界

允许输入变化，保持 TimedDAG 规格固定。一个**合法区间实例**是

$$
\omega=(b,c,Q_b,E_{[b,c)},\mathcal U_{[b,c)},Q_c),
$$

其中 $b<c$，$Q_b$ 来自某份完整参考记录，后两个坐标由分段继续递归唯一确定。所有这类实例组成集合 $\mathsf{IntervalInst}$。

先固定可能作用的坐标全集

$$
\mathsf{ActCoord}_\infty
=(\{\mathrm{prep},\mathrm{adopt},\mathrm{full}\}\times V\times\mathbb N)
\sqcup(\{\mathrm{select}\}\times J\times\mathbb N).
$$

沿用 $P_{v,\theta}=(\mathrm{prep},v,\theta)$ 等四种坐标写法；这些坐标是否实际发生由实例确定。记 $\mathscr V^{\mathrm{ev}}(\omega)$ 为实例区间内的实际作用集合。一个有限**作用范围** $K\subseteq\mathsf{ActCoord}_\infty$ 指定允许联合求值的坐标。窗口、归属和下述边界签名一同固定。记 $\mathsf{Inst}(K)$ 为窗口包含 $K$ 的全部时间坐标、并满足该范围边界条件的合法实例集合。在实例 $\omega$ 上，实际作用块为

$$
K(\omega)=K\cap\mathscr V^{\mathrm{ev}}(\omega).
$$

固定作用块是所有实例均满足 $K(\omega)=K$ 的特例。一般范围允许候选与激活集合随输入变化。

对带名字坐标集合 $\Gamma$ 及其值空间 $\mathsf{Type}(\gamma)$，定义有限记录空间

$$
\mathsf{Rec}(\Gamma)
=\bigsqcup_{\substack{L\subseteq\Gamma\\L\text{ 有限}}}
\prod_{\gamma\in L}\mathsf{Type}(\gamma).
$$

记录同时携带定义域 $L$ 与各坐标的值。可变候选描述量取带节点标签的有限族；消息与输出分别取有限记录集合。这样，来源及实际支持集都是明确的数学数据。

为 $K$ 固定输入、输出签名 $\partial^-K,\partial^+K$。输入签名包含确定范围内作用是否存在、以及求出其函数值所需的外部自变量。块内产生的自变量由内部递归提供。常用的外部坐标及其来源为：

| 使用处 | 所需坐标 | 边界取值 |
|---|---|---|
| 首次准备作用 | 旧节点状态、完整纤维 | 左切面状态或此前的下一持久状态，以及完整原子集合 |
| 选择作用 | 旧选择历史、完整候选域及描述量族 | 左切面历史或此前选择的下一历史，以及各候选描述量 |
| 状态采用作用 | 旧状态、候选状态、本地内容、激活指示、控制量 | 旧状态由状态边界保留；其余量由准备与选择结果取得 |
| 完整输出作用 | $(q^{\mathrm{cmp}},\theta,h,c)$ | 本次快照、本地内容和本节点控制量 |

例如，$U_{v,\eta}\to P_{v,\theta}$ 使用 $U$ 的下一持久状态坐标，$U_{v,\theta}\to F_{v,\theta}$ 使用其计算快照坐标。跨越左切面的状态与历史依赖分别由 $Q_b$ 中的 $q_v^b,y_j^b$ 供给。整个前驱调用标签可作为附加记录保存。准备结果只有 $(h,\widetilde q,d)$；旧状态 $q$ 须显式保留在边界，或从 $\operatorname{Lab}_P=(B,q,h,\widetilde q,d)$ 读取。Next 的一般性使这一输入不可省略。

输入签名及其取值规则须由作用范围和已知边界确定。若选择在块内完成，外部输入给出它所需的描述量或生成描述量的驱动量；实际激活集合由块内递归产生。输出签名包含实际块的全部规范作用值、由块内作用确定的状态与历史轨迹及右边界值，以及产生的消息与外部输出集合。纯完整输出块只返回其作用值及相应消息和输出。定义

$$
\mathsf{BndIn}(K)=\mathsf{Rec}(\partial^-K),
\qquad
\mathsf{BlockOut}(K)=\mathsf{Rec}(\partial^+K).
$$

按上述坐标取值规则抽取参考实例，得到

$$
\beta^-_K:\mathsf{Inst}(K)\to\mathsf{BndIn}(K),
\qquad
\beta^+_K:\mathsf{Inst}(K)\to\mathsf{BlockOut}(K),
$$

并令 $\mathsf{AdmIn}(K)=\beta^-_K(\mathsf{Inst}(K))$。输出记录中的作用支持集恰为 $K(\omega)$；未发生的作用由该支持集的补集表示。

完整性证明另行给定。一次调用须有有效封闭下界，证明边界纤维已经完整，并证明全部块外依赖已经给出。相同边界可以具有不同的关闭证明，联合函数的语义值只依赖边界自变量。

对相同的输入边界，按时间与 $P/S/U/F$ 次序归纳：各作用的存在条件相同，实际作用的自变量相同，因而输出相同。这给出边界充分性

$$
\beta^-_K(\omega)=\beta^-_K(\omega')
\Longrightarrow
\beta^+_K(\omega)=\beta^+_K(\omega').
$$

据此定义全函数

$$
\begin{aligned}
\operatorname{RefBlock}_K&:
\mathsf{AdmIn}(K)\longrightarrow\mathsf{BlockOut}(K),\\
\operatorname{RefBlock}_K(\beta^-_K(\omega))
&=\beta^+_K(\omega).
\end{aligned}
\tag{9}
$$

每种具体边界都须满足上述外部自变量条件；第 3 节分别给出完整输出、节点状态与区域状态的边界。

### 2.3 精确联合求值与精化

一个联合函数

$$
\mathcal K:
\mathsf{AdmIn}(K)
\longrightarrow
\mathsf{BlockOut}(K)
$$

称为 $K$ 的**精确见证**，当且仅当：

$$
\forall z\in\mathsf{AdmIn}(K),\qquad
\mathcal K(z)=\operatorname{RefBlock}_K(z).
\tag{10}
$$

令 $\mathsf{EvalRec}(K)$ 为具体联合求值记录的集合，并固定只遗忘附加坐标的投影：

$$
\Pi_K^{\mathrm{blk}}:
\mathsf{EvalRec}(K)\longrightarrow\mathsf{BlockOut}(K).
$$

这里 $\mathsf{EvalRec}(K)$ 已使用数学值；紧凑实现表示先经固定 Decode 得到这些值，见前置教材附录 S.9b。具体记录 $r$ 在边界 $z\in\mathsf{AdmIn}(K)$ 上**精化到**规范块，当 $\Pi_K^{\mathrm{blk}}(r)=\operatorname{RefBlock}_K(z)$。投影保留实际支持集上的全部规范坐标及其原值，只删除辅助坐标。式 (10) 因而保证展开后的每个 $P,S,U,F$ 坐标与 $\mathcal T_x$ 一致。

为统一后文记号，定义契约种类与**归属标识**集合：

$$
\begin{aligned}
\mathsf{ContractKind}&=\{F,U,UF,R\},\\
\mathsf{ContractOwner}
&=\bigl(\{\mathrm{node}\}\times V\bigr)
\sqcup
\bigl(\{\mathrm{region}\}\times J\bigr).
\end{aligned}
$$

一个**类型化联合契约**是四元组：

$$
B=(\operatorname{kind}(B),\operatorname{cown}(B),K_B,\mathcal K_B),
$$

其中 $\operatorname{kind}(B)\in\mathsf{ContractKind}$，$\operatorname{cown}(B)\in\mathsf{ContractOwner}$，$K_B$ 是有限作用范围，并且 $\mathcal K_B:\mathsf{AdmIn}(K_B)\to\mathsf{BlockOut}(K_B)$ 是式 (10) 的精确见证。种类 $F,U,UF$ 的归属标识分别是相应节点 $(\mathrm{node},v)$；种类 $R$ 的归属标识是相应区域 $(\mathrm{region},j)$。给定 $z\in\mathsf{AdmIn}(K_B)$ 后，二元组 $(B,z)$ 称为一次**契约调用**，记它返回记录中的作用支持集为 $\operatorname{Acts}(B,z)\subseteq K_B$。其余范围坐标获得缺席结论；实际支持集由联合求值返回。固定作用块契约满足 $\operatorname{Acts}(B,z)=K_B$。

多个作用范围两两不交、实际作用满足进入边依赖时，可以按拓扑顺序替换参考递归。对块顺序作归纳即可得到：

> [!proposition] 命题 1：精确块替换
> 设有限作用范围 $K_1,\ldots,K_m$ 两两不交，其实际作用块覆盖目标区间的全部作用。每次调用满足式 (8) 和关闭条件，并具有式 (10) 的精确见证，则所得区间记录与右切面等于参考记录及 $Q_c$。普通作用可视为单作用精确求值。

## 3. 节点与区域的联合契约

以下联合函数统一按作用范围 $K$ 给出类型

$$
\mathsf{AdmIn}(K)\longrightarrow\mathsf{BlockOut}(K),
\qquad \mathcal K_K=\operatorname{RefBlock}_K.
$$

$\operatorname{BatchState}_K,\operatorname{BatchNode}_K,\operatorname{BatchRegionState}_K,\operatorname{BatchRegion}_K$ 分别表示节点状态、节点状态与输出、区域状态、区域状态与输出见证。下面用 $(v,\Theta)$ 或 $(j,I)$ 写出常用范围的别名；窗口、边界签名和覆盖的准备／选择作用一经固定，两种写法表示同一个函数。

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
\longrightarrow\mathsf{BlockOut}(K^F_{v,\Theta}),
$$

$$
\operatorname{BatchFull}_{v,\Theta}
\left(
(q^{\mathrm{cmp}}_{v,\theta},\theta,
h_{v,\theta},c_{v,\theta})_{\theta\in\Theta}^{\uparrow}
\right).
\tag{11}
$$

为该契约取输入签名恰为各时刻的四个自变量；式 (11) 按递增时间排列这些坐标。若调用处保存了完整的准备、选择与状态采用记录，先取相应坐标投影即可得到这个元组。激活与关闭证据用于证明调用合法。

其精确契约是：

$$
\operatorname{BatchFull}_{v,\Theta}(\cdots)
=
\operatorname{ExpandFull}_{v,\Theta}
\left(
\left(
\operatorname{Full}_v
(q^{\mathrm{cmp}}_{v,\theta},\theta,
h_{v,\theta},c_{v,\theta})
\right)_{\theta\in\Theta}^{\uparrow}
\right).
\tag{12}
$$

$\operatorname{ExpandFull}_{v,\Theta}$ 是固定的记录展开函数：每个 $\operatorname{Full}_v$ 值放到相应 $F_{v,\theta}$ 标签上，并按前置教材式 (27)--(28) 展开成由它确定的消息与外部输出坐标。式 (12) 的各输入坐标在调用前均已知，因而它描述相互独立的函数求值组成的时间批。

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
\longrightarrow\mathsf{BlockOut}(K^U_{v,\Theta}),
$$

$$
\operatorname{BatchState}_{v,\Theta}
\left(q_{\mathrm{in}},(\theta_i,z_i)_{i=1}^{k}\right)
\tag{14}
$$

是 $K^U_{v,\Theta}$ 的式 (10) 精确见证，亦记为 $\operatorname{BatchState}_{K^U_{v,\Theta}}$；其值包含该块的全部规范标签，状态坐标逐项等于式 (13) 的标量递归。调用式 (14) 时给出左边界状态、关闭纤维与块外驱动量；块内控制和中间状态由参考顺序产生。空档保持持久状态，逻辑时间坐标仍保留在每个驱动量中。

这种契约称为**因果状态块契约**。它允许一次联合调用精化为一条包含许多个 $P,U$ 及其局部选择作用的状态链。共同选择函数读取多个节点描述量时，相应 $S$ 与所有参与节点一起进入区域契约。

### 3.4 状态—完整输出联合契约

有些节点自然地在生成状态轨迹时同时给出每个激活坐标的完整输出。令：

$$
K^{UF}_{v,\Theta}
=K^U_{v,\Theta}
\cup
\{F_{v,\theta}\mid
\theta\in\Theta\}.
$$

定义：

$$
\operatorname{BatchNode}_{v,\Theta}:
\mathsf{AdmIn}(K^{UF}_{v,\Theta})
\longrightarrow\mathsf{BlockOut}(K^{UF}_{v,\Theta}),
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

式 (15) 在作用范围 $K^{UF}_{v,\Theta}$ 上满足式 (10)，亦记为 $\operatorname{BatchNode}_{K^{UF}_{v,\Theta}}$。选择已在边界给定时，实际 Full 支持集已知；选择位于块内时，由联合函数同时返回该支持集及其值。每个返回坐标都属于相应参考记录。

### 3.5 区域控制—状态契约

一个区域的选择在同一时间联合读取所有候选描述量，选择历史又跨时间递归。固定 $j\in J$ 与区间 $I=[b,c)$，定义区域状态作用范围：

$$
\begin{aligned}
K^R_{j,I}
={}&\{P_{v,\theta}
\mid v\in\mathcal R_j,\ \theta\in I\}
\\
&\cup
\{U_{v,\theta}
\mid v\in\mathcal R_j,\ \theta\in I\}
\\
&\cup
\{S_{j,\theta}\mid
\theta\in I\}.
\end{aligned}
$$

若该区域全部相关时间纤维已经关闭，定义：

$$
\begin{aligned}
\mathsf{RegionIn}_{j,I}
&=\mathsf{AdmIn}(K^R_{j,I}),\\
\mathsf{RegionStateOut}_{j,I}
&=\mathsf{BlockOut}(K^R_{j,I}).
\end{aligned}
$$

区域参考转导是式 (9) 的特例：

$$
\begin{aligned}
\operatorname{RefRegionState}_{j,I}:
\quad &\mathsf{RegionIn}_{j,I}
\longrightarrow\mathsf{RegionStateOut}_{j,I},\\
\operatorname{RefRegionState}_{j,I}
&=\operatorname{RefBlock}_{K^R_{j,I}}.
\end{aligned}
\tag{16}
$$

该契约的输入签名取 $(q_v^b)_{v\in\mathcal R_j}$、$y_j^b$ 和 $(B_{v,\theta})_{v\in\mathcal R_j,\theta\in I}$，包含空纤维；输出包括区域内全部 $P,S,U$ 标签、完整节点状态轨迹、完整选择历史轨迹和右边界状态。函数值按逻辑时间递增应用前置教材的规则。

一个区域联合函数

$$
\operatorname{BatchRegionState}_{j,I}:
\mathsf{RegionIn}_{j,I}
\longrightarrow\mathsf{RegionStateOut}_{j,I}
$$

满足区域控制—状态契约，当它在整个合法定义域上等于式 (16)，亦记为 $\operatorname{BatchRegionState}_{K^R_{j,I}}$。此名称只覆盖 $P/S/U$。它可以容纳选择描述量读取节点状态、选择结果再改变下一持久状态的递归。

若还同时返回区域内全部 $F$ 标签，令

$$
K^{RF}_{j,I}
=K^R_{j,I}\cup
\{F_{v,\theta}\mid
v\in\mathcal R_j,\ \theta\in I\},
$$

定义区域状态—输出联合见证

$$
\operatorname{BatchRegion}_{j,I}
=\operatorname{BatchRegion}_{K^{RF}_{j,I}}:
\mathsf{AdmIn}(K^{RF}_{j,I})\to\mathsf{BlockOut}(K^{RF}_{j,I}),
\qquad
\operatorname{BatchRegion}_{j,I}=\operatorname{RefBlock}_{K^{RF}_{j,I}}.
$$

它与 BatchRegionState 都归为 $R$ 类契约，实际覆盖由各自作用范围明确区分。输入由区域左状态、历史与完整驱动纤维给出，实际 $P/S/U/F$ 支持集随联合记录返回。

式 (14) 与式 (16) 首先给出外层因果块的数学边界。联合函数的内部工作量、并行深度和存储量在第 5 节及第 9 节登记；这些量与外层块数分别分析。

### 3.6 契约族

固定一个类型化联合契约词汇 $\mathfrak B$，并要求对每个有限作用范围、契约种类和归属标识只取有限多个见证。按第 2.3 节已经定义的种类，令：

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

为使最大前沿成为确定函数，在 $\mathsf{ContractKind}$、$\mathsf{ContractOwner}$ 和规范作用坐标上各固定一个全序，并给同种类、同归属标识、同覆盖集合的多个登记见证固定优先序。有限集合总能取这样的全序。这些次序都是规格的一部分，不依赖本次作用值。

一个更强的契约族可以把更多规范边收进同一联合函数。后文的最大前沿、阶段数与批次数均写成相对于 $\mathfrak B$ 的量。

## 4. 相对于契约的最大前沿递归

### 4.1 部分求值状态

固定区间 $I=[b,c)$。一个部分求值状态写成：

$$
\Omega
=
(Q_b,E_I,\boldsymbol\sigma,
D,D^0,\mathsf{Val},H,\mathcal L),
\tag{18}
$$

其中：

$$
E_I=\{e\in E_x\mid b\le\operatorname{time}(e)<c\},
$$

$\boldsymbol\sigma$ 是前置教材定义的输入端口与消息边封闭下界族；$D\subseteq\mathscr V_x^{\mathrm{ev}}(I)$ 是目标区间内已经完成的规范作用集合；$D^0$ 是目标区间内已经证明缺席的作用坐标集合；$\mathsf{Val}$ 是 $D$ 上的类型正确值函数，并满足 $\mathsf{Val}=\mathcal T_x\!\upharpoonright_D$；$H\subseteq M^*$ 是已经进入当前截面的内部消息；$\mathcal L$ 记录此前的联合调用，其作用范围两两不交。要求 $(D,D^0,\mathsf{Val},H)$ 能嵌入前置教材的一条合法 $P/S/U/F$ 暴露轨迹。

封闭下界的有效性相对于补齐完整左前缀后的记录判定：输入集合为 $E_{x,<b}\cup E_I$，消息集合为 $\{m\in M^*:\operatorname{send}(m)<b\}\cup H$。左前缀由切面完成条件见证；其中能影响未来的消息已保存在 $Q_b$ 的 $W_b$ 中。初始 $H=W_b$，以后加入本窗口产生的消息。

一个普通作用 $\xi\in\mathscr V_x^{\mathrm{ordinary}}$ 在 $\Omega$ 中**可取**，当它在目标区间内的直接前驱属于 $D$，跨左切面依赖由 $Q_b$ 给出，其时间纤维已经关闭，并且当前成本标记允许逐项求值其局部函数。

对契约 $B\in\mathfrak B$，若 $\Omega$ 已经确定 $z_B\in\mathsf{AdmIn}(K_B)$ 的全部坐标，就得到调用 $(B,z_B)$。这个调用在 $\Omega$ 中**可取**，当：

1. $K_B\cap D=\varnothing$，与 $\mathcal L$ 中各调用的范围不交，且 $K_B\setminus(D\cup D^0)\ne\varnothing$；
2. 实际块的全部外部自变量已经由 $D$ 或左切面给出；
3. 契约定义域要求的时间纤维、左边界状态、选择结果或驱动量已经确定；
4. 已有关闭证据覆盖从块外进入的全部纤维；块内消息的存在性与关闭由契约内部递归确定；
5. 调用只读取 $\Omega$ 中已有的值。

普通闭包已证明缺席的坐标可以包含在作用范围内；调用至少确定一个新坐标。若整个区域窗口均已证明为空，其状态与历史保持，可直接形成右边界。

作用范围须由当前已知数据选定，范围内的激活支持可以由联合函数产生。范围判定、参数提取和关闭证明均只使用当前部分记录。第三项对不同契约呈现不同形状。`BatchFull` 需要每个坐标的计算快照；因果状态块只需要左边界状态和整段驱动量；区域状态块需要区域左边界与整段关闭纤维。

### 4.2 普通闭包

定义 $\operatorname{CtrlCl}(\Omega)$ 为反复加入所有可取普通作用及其关闭、缺席结论，直到目标区间的相关坐标和命题稳定所得的状态。只记录每个相关命题的确定结果；目标坐标有限，所以闭包在有限步后存在。当若干普通作用同时可取时，可以按任一顺序加入；命题 1 保证最终标签均取自同一参考记录。

普通闭包可以包含选择、轻量状态计数和封闭下界证据的传播。被成本标记为主要作用的状态转移留给 $\mathfrak B^U$ 或 $\mathfrak B^{UF}$，因而不会在闭包中逐时间消耗。

### 4.3 最大可取块族

令 $\mathcal C_{\mathfrak B}(\Omega)$ 为 $\operatorname{CtrlCl}(\Omega)$ 上全部当前可取的调用 $(B,z_B)$。有限目标坐标集、每个作用范围上的有限见证数以及第 3.6 节的固定次序保证它是有限有序集合。对同一契约种类和同一 $\operatorname{cown}(B)$，$K_B$ 严格包含较多当前坐标的调用排在前面；其余次序依次由契约优先序、归属标识全序和坐标全序决定。

按这个全序扫描 $\mathcal C_{\mathfrak B}(\Omega)$。一个调用与此前已经选出的块两两不交时就收入当前族，否则跳过；扫描持续到候选集合末尾。所得族记为：

$$
\mathsf{Front}_{\mathfrak B}(\Omega).
\tag{19}
$$

它是当前全部可取调用中的极大不交族：任何未选调用都与某个已选调用相交。同一 $\operatorname{cown}(B)$ 的大块优先给出尽可能长的当前坐标集合，继续扫描又会把所有与已选块相容的剩余调用加入本阶段。固定全序只消除表示歧义；它不读取未来作用值，也不改变任一块的参考函数。

### 4.4 最大前沿算法

定义递归 $\operatorname{EagerBlock}_{\mathfrak B}$：

1. 从当前 $\Omega_r$ 求普通闭包 $\overline\Omega_r=\operatorname{CtrlCl}(\Omega_r)$；
2. 取 $\mathsf{Front}_{\mathfrak B}(\overline\Omega_r)$；
3. 同时求值其中全部联合函数；
4. 把实际作用标签加入 $D$，把范围内的缺席坐标加入 $D^0$，把实际内部消息加入 $H$，并把本阶段调用加入 $\mathcal L$；更新封闭下界，得到 $\Omega_{r+1}$；
5. 当目标区间的全部作用已经完成并形成 $Q_c$ 时停止。

一次含有非空联合调用族的循环称为一个**自适应阶段**。同一阶段的联合函数读取本阶段普通闭包结束时的记录，其返回值在阶段末共同进入记录。最后一次普通闭包若已完成目标，直接停止。

若当前前沿为空而目标作用仍未完成，契约族需要提供一个可取的单作用见证，或者把该作用列为普通作用。称满足这项进展条件的 $\mathfrak B$ **完备**。

> [!theorem] 定理 2：最大前沿递归的终止与精确性
> 对任意有限输入、有限目标区间和完备精确契约族 $\mathfrak B$，$\operatorname{EagerBlock}_{\mathfrak B}$ 在有限阶段后停止；其返回区间记录与右切面逐坐标等于参考语义。

**证明。** 每个前沿阶段都确定此前尚未确定的范围坐标，分别加入 $D$ 或 $D^0$。完备性保证目标尚未完成时可以继续；有限目标坐标集因而给出终止。每个普通作用取参考函数值，每个联合调用满足式 (10)，并且所有进入边前驱已完成。对实际加入顺序应用命题 1，得到完整区间记录与 $Q_c$。$\square$

### 4.5 契约相对的自适应秩

固定一次运行实际选出的联合调用及所采用的边界证明，把每个调用收缩为一个宏作用。宏依赖同时记录调用范围的确定、数值读取、作用支持的确定和输入关闭证明；普通闭包对这些信息的传递收缩为零成本边。若 $B$ 的回答经上述任一依赖用于确定 $B'$ 的调用，写成：

$$
B\prec_{\mathfrak B} B'.
$$

例如，某条潜在入边对应的 Full 坐标返回 $\bot$ 后，接收纤维才能关闭；这个回答仍给出一条关闭证明依赖。所得有限偏序称为本次运行的**宏作用因果图**。定义：

$$
r(B)
=
1+\max_{B'\prec_{\mathfrak B}B}r(B'),
\qquad
\max\varnothing=0.
\tag{20}
$$

在“一个阶段中确定全部调用，回答于阶段末共同进入记录”的模型下，任何遵守同一宏依赖图的因果策略都必须把 $B$ 放在不小于 $r(B)$ 的阶段。最大前沿递归在每个回答层立即提交全部当前可取宏作用，因此：

> [!theorem] 定理 3：固定宏作用图上的阶段最优性
> 对固定精确契约族、固定优先规则，以及本次运行确定的宏作用分解与证明依赖，最大前沿递归使用 $\max_B r(B)$ 个自适应阶段；任何对同一宏依赖图求值、遵守相同阶段可见性条件的因果策略至少需要这么多阶段。

**证明。** 每条 $B'\prec_{\mathfrak B}B$ 都要求 $B$ 严格晚于 $B'$ 的回答阶段，对式 (20) 归纳得到下界。对最大前沿实际产生的固定不交分解，若一个宏作用已经可取，它与较早阶段选出的宏作用不相交，式 (19) 的完整扫描会在当前阶段收入它；因此它只会等待尚未公开的宏前驱。对秩归纳可知第 $k$ 层在第 $k$ 个阶段全部提交，给出匹配的上界。$\square$

这个结论比较相同的联合函数词汇。增加一个精确的长状态块或状态—输出联合契约，会收缩原宏作用图并可能降低自适应秩。

### 4.6 三类节点批次数与区域块数

固定因果求值策略 $\mathscr S$，即每次选择只依赖已有部分记录的确定规则。对其一次区间求值，令 $\mathcal L$ 为各阶段实际选中契约调用的有限集合；以下的 $B$ 均遍历 $\mathcal L$，并用 $B$ 同时表示调用及其类型化契约部分。定义：

$$
m_v^U
=
\#\{B\in\mathcal L\mid\operatorname{kind}(B)=U,
\operatorname{cown}(B)=(\mathrm{node},v)\},
$$

$$
m_v^F
=
\#\{B\in\mathcal L\mid\operatorname{kind}(B)=F,
\operatorname{cown}(B)=(\mathrm{node},v)\},
$$

$$
m_v^{UF}
=
\#\{B\in\mathcal L\mid\operatorname{kind}(B)=UF,
\operatorname{cown}(B)=(\mathrm{node},v)\},
$$

$$
m_j^R
=
\#\{B\in\mathcal L\mid\operatorname{kind}(B)=R,
\operatorname{cown}(B)=(\mathrm{region},j)\}.
\tag{21}
$$

这些量完整记为 $m_v^U(I;x,\mathscr S,\mathfrak B)$ 等，固定上下文时省略参数。再令 $R=R(I;x,\mathscr S,\mathfrak B)$ 为自适应阶段数。一次联合块同时覆盖 $U$ 与 $F$ 时，它只计入 $m_v^{UF}$；若分析者还需要两个语义投影的覆盖数，可以从该块所含 $U,F$ 坐标分别统计。

这里的**共同规格族** $\mathfrak S$ 是一组共享下列数据的 TimedDAG 规格：

1. 有限集合 $V,A,J$、端口集合、映射 $\rho,\operatorname{src},\operatorname{dst}$、时延与输入时间规则；
2. 节点状态、选择历史、本地内容、描述量、控制量和输出的值空间，以及全部局部函数的定义域和值域；
3. 第 3.1 节的成本标记规则、联合契约的共同形状规则与全部固定全序。

一个**共同契约模式**是按规格索引的契约族 $(B_{\mathcal G})_{\mathcal G\in\mathfrak S}$：所有 $B_{\mathcal G}$ 具有相同契约种类、归属标识和作用坐标选择规则，而其见证函数分别在规格 $\mathcal G$ 的合法边界域上满足式 (10)。第 3.6 节的词汇现在理解为一组固定的共同契约模式。族中规格可以改变共同类型内的局部函数值，见证函数随 $\mathcal G$ 取相应成员；模式、调度规则与下列常数不随 $\mathcal G$ 改变。因此，这些常数由共同的 $V,J$ 索引，量词也确实比较同一组节点与区域。

称共同规格族 $\mathfrak S$ 具有**一致节点时间批暴露**，当存在上述固定完备精确契约词汇、一个共同因果策略 $\mathscr S$ 和常数：

$$
C_R,\qquad
(C_v^U,C_v^F,C_v^{UF})_{v\in V},
\qquad
(C_j^R)_{j\in J},
\tag{22}
$$

使策略 $\mathscr S$ 对任意规格 $\mathcal G\in\mathfrak S$、任意输入位置 $q$、任意长度 $T$ 和任意合法左边界都精确求值，并满足：

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

最大前沿与第 6 节的整窗口求值分别给出具体策略；采用哪一策略须在应用结论时固定。这些常数独立于 $T$。式 (23) 计量外层分块；联合函数内部的标量步数和并行深度另行登记。

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

## 5. 状态递归、选择函数与动态切口

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

### 5.4 TotalEmit 条件

定义 **TotalEmit** 条件：

$$
v\in\mathcal A_{\rho(v),\theta}
\Longrightarrow
\forall a\in\operatorname{Out}(v),
\quad f^A_{v,\theta}(a)\in P.
\tag{31}
$$

在这个函数类中，节点一旦激活，每条出边都有一条实际消息。消息的存在性由激活集合确定，载荷仍由 $F$ 给出。最大前沿递归及四类契约保持式 (11)--(19) 的形状；式 (31) 改变具体运行中出现的消息边和后继作用密度。

载荷可以继续影响未来聚合、选择函数和状态，所以 TotalEmit 条件仍可能反复实例化式 (29)。它适合作为常见函数条件单独登记。

## 6. 严格分层区域的整块定理

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

称区域划分**严格分层**，当存在函数 $\ell:J\to\mathbb N$ 满足每条空间边 $a$ 都有：

$$
\ell(\rho(\operatorname{src}(a)))
<
\ell(\rho(\operatorname{dst}(a))).
\tag{33}
$$

式 (33) 同时推出：区域内没有空间边，且 $(J,Q_\rho)$ 是 DAG。具有相同 $\ell$ 值的区域之间也没有空间边。

### 6.2 区域时间块

固定 $I=[b,c)$。对区域 $j$，假设已经给定：

1. $(q_v^b)_{v\in\mathcal R_j}$ 与 $y_j^b$；
2. 区域内全部关闭时间纤维 $B_{v,\theta}$；
3. 从左切面和更早层区域进入这些纤维的全部消息。

式 (16) 唯一确定区域内的 $P,S,U$ 轨迹。随后，式 (12) 可以按节点求全部 $F$；式 (15) 则可以在具有联合见证时同时求状态与输出。

### 6.3 分层推进算法

取区域顺序 $j_1,\ldots,j_m$，使：

$$
\ell(j_1)\le\cdots\le\ell(j_m).
$$

从 $Q_b$ 开始，对 $r=1,\ldots,m$：

1. 汇集 $\mathcal R_{j_r}$ 在 $I$ 内的外部记录、跨切面消息与更早层区域消息；
2. 用入边和输入端口的封闭下界证明这些纤维完整；
3. 应用一个精确区域状态契约，或应用它允许的节点状态契约族；
4. 对尚未由联合契约覆盖的 $F$ 按节点应用式 (12)；
5. 把右边界状态、选择历史和发往更后层的消息加入当前部分状态。

式 (33) 保证当前区域的区间输入只来自左切面、外部输入或已经处理的更早层区域。

> [!theorem] 定理 4：严格分层整块定理
> 在严格分层条件下，若每个区域的 $P/S/U$ 区间转导具有精确区域状态见证，且每个节点的剩余 $F$ 具有式 (12) 的精确见证，则上述算法从 $Q_b$ 精确推进到 $Q_c$。每个区域至多使用一个状态块，每个节点至多使用一个额外完整输出时间批。

**证明。** 对区域顺序归纳。处理 $j_r$ 时，每条入空间边来自更小的 $\ell$，相应区间消息已经由归纳前缀产生；跨切面消息属于 $Q_b$，外部输入由输入封闭下界覆盖。因此区域时间纤维完整。区域状态见证给出参考 $P/S/U$ 轨迹，节点 $F$ 见证给出参考输出与消息。处理完所有区域后，目标区间全部作用与参考记录一致；命题 1 给出 $Q_c$。$\square$

若整个区域具有区域状态—输出联合见证，可以把定理中的区域状态块与逐节点 $F$ 批融合为一个块。在选择结果已经从边界给定、各节点递归彼此分离的特例中，也可以分别使用式 (15) 的节点联合见证。若区域选择历史只需顺序递归，定理仍给出一个外层状态块；其块内并行深度需要另行分析。

### 6.4 按层最大前沿

具有相同 $\ell$ 值的区域可以位于同一个最大前沿阶段。令 $H_\rho$ 为区域商图最长路径上的顶点数；这里允许只含一个顶点、零条边的路径，所以非空 $J$ 总有 $H_\rho\ge1$。于是：

$$
R\le 2H_\rho
\tag{34}
$$

这是定理 4 中“一个区域状态块，随后一层逐节点完整输出批”的阶段界。若每个区域具有区域状态—输出联合见证，则：

$$
R\le H_\rho.
$$

若区域状态作用全部属于同一阶段的普通闭包，只有逐节点完整输出属于主要作用，也得到后一界。顺序逐区域、并把状态与输出分开的实现给出较宽松的 $R\le2|J|$。这些界以及主时间区间内的状态块数和完整输出批数都与 $T$ 无关。输入改变每个块中实际出现的坐标集合，而严格分层次序保持不变。

这里按层取的是已关闭的**整个目标区间**。第 4 节立即消费当前可执行前缀的规则，不会仅因输入规格满足式 (33)，就自动采用这个分块。下面给出一个能自动识别此结构、也能处理一般区域划分的算法。

### 6.5 保留完整区域的静态分量

式 (32) 排除了自边，不能单独检出区域内的消息依赖。为调度另定义

$$
\widehat Q_\rho
=\{(\rho(\operatorname{src}(a)),\rho(\operatorname{dst}(a)))\mid a\in A\}.
$$

这一次保留 $(j,j)$。在有向图 $(J,\widehat Q_\rho)$ 中，互相可达的区域构成一个**强连通分量**，允许长度为零的可达路径。记分量集合为 $\mathcal D$；对 $D\in\mathcal D$，令 $V_D=\bigcup_{j\in D}\mathcal R_j$。不同分量之间保留原方向得到缩点图，它是 DAG：若缩点后还有环，环上分量就应当互相可达，违反分量的定义。

一个分量包含完整区域，因而同时包含各候选的准备、共同选择函数、选择历史和状态采用依赖。分量之间只剩原消息边。固定共享参数不增加一次前向中的可变依赖。这个构造等价于先将每个节点与其选择函数归属标识双向连接，再求 SCC；后者在 [PDG 教材 §9.4](positive-delay-graph-finite-cut-learning-note.md#94-依赖完备静态图) 中有独立定义和证明。

称 $D$ 是**可整区间处理的单区域分量**，当 $D=\{j\}$ 且 $(j,j)\notin\widehat Q_\rho$。最后一个条件不可省略：只有一个区域的分量也可能含区域内消息，后一次输入可能等待本区域较早的 Full。

静态分量只决定可能的依赖方向。实际消息仍保留每条原边的身份、端点和时延；两条同向边不能因为缩点而合并成一个输入值。

### 6.6 静态分量窗口算法

固定输入已经封闭的有限窗口 $I=[b,c)$、合法左切面 $Q_b$，以及前文的精确契约族；以逐作用契约作为后备。按缩点图的任一拓扑序处理分量 $D$：

1. 取 $V_D$ 的左边界节点状态与 $D$ 的选择历史。汇集目标在 $V_D$、到达时间在 $I$ 的外部记录、$W_b$ 中的消息，以及所有前驱分量在本窗口产生的跨入消息。
2. 此时来自分量外部的窗口输入已完整。若 $D$ 是 §6.5 的单区域分量，则全部窗口纤维也已完整。提供定理 4 的区域状态与逐节点 Full 契约时，调用一个区域状态块，取得整个窗口的选择和快照，再按节点各调用至多一个 Full 时间批；缺少这些联合契约时也可按第 3 步后备求值。
3. 其他分量内部仍可能有消息依赖；缺少联合契约的单区域分量也使用本步。在其已关闭的外部边界上，运行第 4 节的安全最大前沿递归；没有更强联合见证时使用逐作用后备。内部纤维仍须取得内部消息的关闭证据，不能先等待本分量的整个窗口完成。
4. 把到达时间在 $I$ 的跨出消息交给后继分量。所有到达时间不小于 $c$ 的新消息都进入右切面队列；它们即使发往已完成的内部节点，也不能丢弃。保留 $W_b$ 中尚未到达的原消息，并按原消息身份去重归属。
5. 收集各分量的区间记录、右边界节点状态及选择历史，组成完整记录与 $Q_c$。

这里等待的是**目标区间的上游全部完成**，不要求预知上游会选谁。前驱完成后，其实际输出和无消息结果共同给出封闭证据。外部输入封闭下界不会单独证明内部反馈纤维关闭；第 3 步负责推进这部分证据。若只需更低延迟，可以先缩小目标窗口，再按第 8 节组合。

> [!theorem] 定理 4a：静态分量窗口求值与严格分层退化
> 在上述边界与精确契约条件下，§6.6 算法有限结束，返回参考区间记录和 $Q_c$。若区域严格分层且提供定理 4 的精确契约，它自动采用 §6.3 的整区间形状：每区域至多一个状态块，每节点至多一个额外 Full 时间批。

**证明。** 沿分量拓扑序归纳。来自其他分量的新消息只能来自前驱；正时延保证窗口右侧尚未执行的源事件不会向窗口内补发消息。结合左切面和输入封闭下界，第 1 步获得分量的全部窗口外部输入。单区域且无内部消息时，定理 4 的区域论证直接适用。其余分量的节点状态和选择历史都在本分量内，受限事件图仍是有限 DAG；第 4 节递归在这个完整边界上精确且有限终止。归纳并合记录，并保留所有跨右切面的消息，得到同一个 $Q_c$。

若满足式 (33)，区域图无环且没有自边，每个分量恰是可整区间处理的单区域分量，算法从不进入内部递归分支。所得调用正是 §6.3 的区域状态块与逐节点 Full 批。$\square$

这个结论也可在正时延有环消息图上成立：分量内部改用有限切面递归即可，见 [PDG 教材 §9.5](positive-delay-graph-finite-cut-learning-note.md#95-默认静态分量的窗口调度)。分量数与窗口宽度无关，**不推出反馈分量内部的批次数或跨度与窗口宽度无关**。严格分层时，一个有 Full 坐标的节点已经把本窗口全部坐标放入一个逻辑批；具体设备上的调用次数与耗时见附录 S 的对应。

### 6.7 对齐路径仍可能被即时前缀分开

取四个按秩 $1,2,3,4$ 排列的区域，节点分别为 $\{A,B\},\{C\},\{D\},\{X\}$；第一区域容量为 $1$。消息边及正时延为

$$
A\xrightarrow{3}X,\qquad
B\xrightarrow{1}C\xrightarrow{1}D\xrightarrow{1}X.
$$

令前两个输入位置使用 Settle 编码 $\theta(t,j)=6t+j$，两入口 $A,B$ 都接收输入，位置 $0$ 选择 $A$，位置 $1$ 选择 $B$；其余区域在收到消息时选择唯一候选。两条路径的总时延同为 $3$，所以这本身就是路径对齐的 SettleGraph 例子。考察窗口 $[0,12)$，暂把一个区域的状态处理与随后 Full 批记为一个表格步骤。

| 步骤 | 立即消费安全前缀 | §6.6 的窗口调度 |
| --- | --- | --- |
| 1 | $A@1,B@7$ | $A@1,B@7$ |
| 2 | $C@8$ 与 $X@4$ | $C@8$ |
| 3 | $D@9$ | $D@9$ |
| 4 | $X@10$ | $X@4,X@10$ 同批 |

第二步时，待处理的 $C@8$ 最早只能导致 $X@10$，因此 $X@4$ 已安全可做。即时前缀策略做它并没有语义错误，却把 $X$ 拆成两批。窗口策略让这个已经安全的事件等待上游整段完成，收集 $X$ 的全部坐标后再做。只在同一轮改变就绪块的排序，不能阻止立即执行其他互不冲突的短块。

另一种保守策略可以静态展开潜在事件时间，包括本次可能为空的 $C@2,D@3$，并让这些潜在坐标参与关闭判断。它在本例也能形成一个 $X$ 批；空坐标只提供不存在性证据，不因此执行 Full。因此不能把所有通用 TimedDAG 算法都等同于表格左列。三种调度策略的对照见附录 S.11。

### 6.8 静态分析需要哪些信息

区域图及其 SCC 可以由节点、边与区域归属静态构造，无需枚举路径，也不依赖本次输入或选择值。任意正时延都足以支持 §6.6 的正确性；严格分层整块结论同样不要求不同路径的总时延相等。

若另外要检验 Settle 的时间对齐形状，可以求区域势 $r_j$ 并逐边检查

$$
\delta(a)=r_{\rho(\operatorname{dst}(a))}-r_{\rho(\operatorname{src}(a))},
$$

再核对输入注入日程、区域内共同时间与教材的边界编码。沿路径求和会望远镜消去中间势，因此不必枚举全部路径。配合严格区域秩、共同位置范围及端口注入日程，这些条件给出 SettleGraph 的单次结算结构。它直接沿用 TimedDAG 的局部函数、容量上界和逐边／逐端口输出；共同输入与终端聚合可另作边界实例。

静态分量、即时前缀与潜在时间展开具有不同的内存和延迟代价。窗口调度为了合批可能多保留待处理消息，并推迟已经可以给出的输出。它提供可证明的整块形状，不声明对所有输入、窗口大小和设备都最快。

## 7. 典型结构与边界

### 7.1 串行 Transformer block

把每个 Transformer 层或每个具有共同选择的模块放入一个区域，并令空间边只从较早层指向较后层，就得到严格分层结构。对一个已知输入分块：

- 每层 Attention 可以用式 (15) 表示旧 KV 状态、整段新输入、全部因果输出和新 KV 状态；
- 逐位置前馈网络可以用式 (12) 表示；
- 层间消息在前一层完成以后一次提供给后一层。

当层内没有跨位置自适应选择函数时，每层主时间块通常对应一个状态—输出联合块或一个状态块加一个完整输出批。

### 7.2 SSM 层

对满足式 (26) 的 SSM，$A_\theta,b_\theta$ 若在状态块开始以前已知，就可以使用式 (27) 的扫描见证。输出若为：

$$
o_\theta=C_\theta q^\theta+d_\theta,
$$

可由式 (15) 与状态轨迹共同返回。输入依赖的 $A_\theta$ 仍可批量形成，只要其计算不等待同块内尚未产生的状态或完整输出值。

### 7.3 MoE 区域

设一个区域含多个 expert，选择历史保存负载摘要。区域状态契约先产生整段激活集合族和控制量。对 expert $e$ 定义：

$$
\Theta_e
=
\{\theta\in I\mid e\in\mathcal A_{\rho(e),\theta}\}.
\tag{35}
$$

随后对每个 expert 应用一个 $\operatorname{BatchFull}_{e,\Theta_e}$。不同 expert 的时间集合可以具有不同大小；式 (35) 正是运行时实际形成的不等长时间批。

若后一个时间的路由读取前一个 expert 的完整输出，式 (29) 会把区域时间块切开。若 history 只读取已知描述量和过去激活集合，区域状态契约可以先完成整段选择。

### 7.4 不同到达时刻的重新汇合

当同一节点存在不同到达时刻的路径时，多个输入位置可以在同一时间纤维汇合。算法仍按 $B_{v,\theta}$、$P/S/U/F$ 和 $Q_b$ 工作。一个节点时间批由逻辑时间坐标集合定义，因此无需为每个事件指定唯一输入位置。

实际跨界内容由 $W_b$ 给出。状态块输入保留真实逻辑时间，所以从时间 $2$ 到时间 $10$ 的空档始终具有长度 $8$。

### 7.5 区域商图中的回返

固定节点图是 DAG，区域商图仍可因交错划分而含环。例如路径 $v_0\to v_1\to v_2\to v_3$ 按奇偶节点分为两个区域时，商图含双向边。

这种划分会在区域之间形成时间拉链。最大前沿算法仍然精确终止，但主区间可能分成随 $T$ 增长的多个状态块或完整输出批。§6.5--6.6 的分量构造给出精确外层调度；分量内部仍需因果递归或更强的联合契约，内部批次数另行分析。

### 7.6 全局选择函数

若所有节点属于同一个区域，选择历史可以跨越全部拓扑层。一个区域状态契约在数学上可以覆盖整段 $P/S/U$ 轨迹；它是否具有紧凑表示、低工作量或低并行深度，取决于选择函数与状态转移的结构。

若消息载荷反复改变全局选择函数的下一次选择，就会出现多轮式 (29)。此时单一大区域的名称没有减少自适应秩；精确契约必须真实接收每轮所需的前序值。

### 7.7 静态拓扑与运行时形状

固定空间图、区域划分和契约族给出所有可能作用块的静态包络。输入、状态、选择函数与函数输出共同决定一次运行的：

$$
\Lambda_v^U(I;x),\qquad
\Lambda_v^F(I;x),\qquad
\mathsf{Front}_{\mathfrak B}(\Omega_r).
$$

严格分层结构固定外层层次；输入主要改变块内坐标。具有回返的结构还会使块边界本身随输入改变。最大前沿递归在每个阶段从当前部分状态直接构造这三个对象。

## 8. 切面组合、因果性与分段继续

### 8.1 输入分块组合

沿用前置教材 §11.5 的共同区间接口。固定规格与全部类型正确输入的集合 $\mathfrak X$，令

$$
\operatorname{Reach}_a=\{Q_a(x):x\in\mathfrak X\},\qquad
\operatorname{Adm}_{a,b}
=\{(Q_a(x),E_{x,[a,b)}):x\in\mathfrak X\}.
$$

定义

$$
\Phi_{a,b}:\operatorname{Adm}_{a,b}\to\mathsf{Rec}_{a,b}\times\operatorname{Reach}_b,
\qquad
\Phi_{a,b}(Q_a,E_{[a,b)})=(\mathcal U_{[a,b)},Q_b).
$$

区间记录包含 $a\le\theta<b$ 的纤维、候选、激活、本地量和全部实际作用结果，新消息按发送时间归入本段，新输出按输出时间归入本段；节点状态与选择历史保存 $a\le\theta\le b$ 的坐标。$W_a$ 只提供已有边界输入。封闭下界证明输入完整，函数值由 $(Q_a,E_{[a,b)})$ 决定。

两段共享同一个 $Q_b$ 时，$\mathcal U_{[a,b)}\odot_b\mathcal U_{[b,c)}$ 按时间并合作用和输出、按发送时间并合新消息，并将时间 $b$ 的共同状态与历史识别一次。$W_b$ 参与右段纤维，不重复登记为右段新消息。空区间返回仅有边界状态坐标的恒等记录和原 $Q_a$。

对 $a\le b\le c$，分段继续定理给出

$$
\begin{aligned}
\Phi_{a,b}(Q_a,E_{[a,b)})&=(\mathcal U_{[a,b)},Q_b),\\
\Phi_{b,c}(Q_b,E_{[b,c)})&=(\mathcal U_{[b,c)},Q_c)\\
\Longrightarrow\quad
\Phi_{a,c}(Q_a,E_{[a,c)})
&=(\mathcal U_{[a,b)}\odot_b\mathcal U_{[b,c)},Q_c).
\end{aligned}
\tag{36}
$$

每段可以使用不同的精确联合契约，只要结果满足同一接口。正时延图教材 §7 采用完全相同的定义，允许有限输入 DAG 以外的规格。

### 8.2 输入前缀不变性

若输入封闭下界覆盖目标切面 $c$，未来再增加时间不小于 $c$ 的外部记录，不会改变 $[b,c)$ 的参考记录。最大前沿递归的每个 $P$ 又要求相应纤维关闭，所以它只使用当前已经由封闭下界保证稳定的候选集合。

这项性质允许在线地推进切面：输入源先公开一个有限输入分块和相应封闭下界，算法完成 $Q_c$，随后以 $Q_c$ 作为下一段左边界。

### 8.3 自回归输出的结构检查

自回归任务需声明每个输出可见的输入前缀，并保证其余输入位置的值不能改变这个输出。固定规格、各端口输入长度 $L_i$ 与时间规则 $\iota_i$，定义输入位置集合及其上的限制：

$$
\mathsf{InputPos}=\{(i,k)\mid i\in\mathsf I,\ k\in[L_i]\},
\qquad
x|_D=(x_i(k))_{(i,k)\in D}
\quad(D\subseteq\mathsf{InputPos}).
$$

外部记录 $(\mathrm{ext},i,k,p)$ 的来源标签为 $(i,k)$。固定输出坐标 $(\theta,o)$，记 $z_{\theta,o}(x)\in P_\bot$ 为该坐标的值：有记录 $(\mathrm{out},\theta,o,p)$ 时取 $p$，缺席时取 $\bot$。这样同时比较输出值与输出是否存在。

为这个输出指定各端口可见的位置数 $n_i\in\{0,\ldots,L_i\}$，记 $n=(n_i)_{i\in\mathsf I}$，允许的输入前缀为

$$
\mathsf{Prefix}(n)=\{(i,k)\in\mathsf{InputPos}\mid k<n_i\}.
$$

自回归要求对任意 $x,x'\in\mathfrak X$ 都有

$$
x|_{\mathsf{Prefix}(n)}=x'|_{\mathsf{Prefix}(n)}
\quad\Longrightarrow\quad
z_{\theta,o}(x)=z_{\theta,o}(x').
$$

称一个固定集合 $D_{\theta,o}\subseteq\mathsf{InputPos}$ 为该输出的**保守外部依赖集合**，若对任意 $x,x'\in\mathfrak X$ 都有

$$
x|_{D_{\theta,o}}=x'|_{D_{\theta,o}}
\quad\Longrightarrow\quad
z_{\theta,o}(x)=z_{\theta,o}(x').
$$

因此，自回归要求具有如下充分条件：

$$
D_{\theta,o}\subseteq\mathsf{Prefix}(n).
\tag{37}
$$

**证明。** 两份输入在允许前缀上相等，由式 (37) 可得它们在 $D_{\theta,o}$ 上相等，再由保守外部依赖集合的定义得到输出相等。$\square$

单输入流或任务已对齐各端口位置时，预测位置 $t+1$ 可取 $n_i=\min(L_i,t+1)$；此时允许前缀恰为输入位置集合中满足 $k\le t$ 的部分。一般多端口任务分别声明各 $n_i$。

一种构造 $D_{\theta,o}$ 的方法，是把参考递归在覆盖 $\theta$ 的有限切面内展开，保留全部可能的消息槽、候选标记与激活标记。消息槽用值或 $\bot$ 表示发送与缺席。外部输入的来源为自身标签，固定初态的来源为空集；函数结果取自变量来源的并，条件分支取条件与全部分支来源的并。按递归顺序归纳，所得输出来源集合满足上述定义。

这个检查须覆盖消息存在性的依赖。例如某个输入决定是否额外发送一条消息，即使本次未发送，下游结果仍可能受该输入影响。一次运行的 $P/S/U/F$ 事件 DAG 以实际作用为顶点，消息边只来自实际发送；单独追溯它可能漏掉这种依赖。

精确分块、状态扫描和联合 Full 求值对每份输入都保留同一个 $z_{\theta,o}(x)$，因而保留上述前缀不变性；同一个 $D_{\theta,o}$ 也继续满足其定义。

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

联合状态函数可以在内部采用压缩表示；其右边界须由固定 Decode 恢复式 (38) 的确切规范状态。若还要重建完整历史记录，则同时保存各块的细粒度 $P/S/U/F$ 标签或一种可逆编码。

## 9. 性能分析应记录的量

### 9.1 静态记录

对每个规格与契约族，记录：

- 空间 DAG、区域划分与区域商图；
- 每个节点可用的 $\mathfrak B^F,\mathfrak B^U,\mathfrak B^{UF}$；
- 每个区域可用的 $\mathfrak B^R$；
- 成本标记与联合函数内部的理论工作量、并行深度和存储界。

空间 DAG、区域划分、成本标记和契约族共同决定最大前沿算法的合法动作空间。

### 9.2 每次运行的动态记录

对每个输入分块，记录：

$$
R,\qquad
(m_v^U,m_v^F,m_v^{UF})_{v\in V},
\qquad
(m_j^R)_{j\in J},
\tag{39}
$$

以及每个块的坐标集合大小、边界状态大小、消息数、候选数和激活数。进一步可记录：

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
6. `BatchRegionState` 容纳跨节点选择与状态的共同递归；`BatchRegion` 另包含区域完整输出；
7. 最大前沿递归相对于固定契约族在线构造当前极大的不交可取块族；
8. 在固定宏作用图上，它达到最少自适应阶段；
9. 节点状态批数、完整输出批数、联合批数和区域状态块数分别计量；
10. 严格分层区域加精确状态与输出契约给出与输入分块长度无关的外层批次数；
11. 静态分量窗口算法适用于一般区域划分，并在严格分层时自动采用整区间形状；即时消费安全前缀是另一种调度策略。

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
\text{与输入分块长度无关的外层块数}.
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
6. 在 TotalEmit 条件下，给定激活集合族，列出所有必然存在的消息坐标；再说明载荷未知会保留哪些动态切口。
7. 为一个 MoE 区域计算式 (35) 的不等长时间集合，并分别统计 $m_e^U,m_e^F,m_e^{UF}$。

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
> `refinement` 先将具体表示通过固定 Decode 解码为数学值记录，再用遗忘投影取得规范 $P/S/U/F$ 记录。解码只解释表示，不重执行参考递归或修补错误结果。一次 `BatchState`、`BatchFull` 或 `BatchNode` 调用可以映射为多个规范作用；投影保留这些作用的原值与逻辑时间，只删除调用编号、线程、设备、缓存和墙钟坐标。

> [!info]- S.11　三种调度策略的语义对照
> 下表比较 §6.7 中的封闭证据与等待规则。
>
> | 策略 | 封闭与等待规则 | §6.7 的例子 |
> | --- | --- | --- |
> | 即时安全前缀 | 由实际待处理纤维与正时延下界证明前缀关闭，立即消费 | $X$ 分成两批 |
> | 潜在时间展开 | 先沿节点拓扑传播潜在事件时间，用潜在空坐标提供不存在性证据 | $X$ 在一批内；空坐标不产生 Full |
> | 区域整窗口 | 在严格分层条件下，等前驱区域完成整个窗口后再处理当前区域 | $X$ 在一批内 |
>
> 三种安排都必须精化到同一参考记录，但可以采用不同的封闭证据和分块。§6.6 的静态分量窗口算法在严格分层时自动采用第三种形状；一般分量内部仍须因果推进。这些结论来自结构与精确契约，具体设备效果另行测量。
