---
type: mathematical-learning-note
status: active-learning
as-of: 2026-09-17
cssclasses:
  - textbook-math
tags:
  - tide
  - settlegraph
  - mathematics
  - learning-note
---

# 单次结算图 SettleGraph：从一次输入到有状态序列

> [!summary] 本文的阅读前提
> 本文面向第一次接触 SettleGraph 的数学读者，只假设读者熟悉集合、函数、实向量、有限求和与数学归纳法。有向图、区域、逻辑时间、候选、激活、计算快照与状态延续都会在本文中定义。理解前九节不需要先读其他教材；系统用语集中在附录 A。

本文的 SettleGraph 是一种受限的有状态图：固定节点图无环，region 配有使每条节点边严格向前的次序；每个输入位置上，每条边只确定一次结果，每个节点至多做一次完整计算。本文从一次输入的结算出发，逐步区分输入、记忆、共同选择和完整输出，再把一次次结算连接成有状态序列。符号与 [[timed-dag-region-selector-learning-note|TimedDAG 教材]] 保持一致，第 10 节给出两种对象之间的明确编码。

全文依次回答三个数学问题：一次输入能否得到唯一输出，保存什么可以继续下一段输入，以及这种计算如何成为 TimedDAG 的受限实例。第 11 节再解释节点级时间批需要哪些额外条件。

> [!tip] 分三次阅读
> 第一次读第 1--7 节，手算两个连续输入；第二次读第 8--9 节，理解逻辑时间衰减与分段继续；第三次结合 TimedDAG 教材读第 10--12 节，研究嵌入、节点级时间批与函数保持接入。附录 B 另用微分记号讨论指定的反向规则。

## 1. 先看一个问题

设输入是一个实数 $x$，有两个节点 $a,b$ 都接收它。我们希望根据它们各自产生的简单描述，选择一个做完整计算。第三个节点 $c$ 接收被选节点的结果并输出。

```text
         a ──┐
输入 x ──┤    ├── c ── 输出
         b ──┘
```

图中的两条内部边 $a\to c,b\to c$ 始终存在。某次只选择 $b$，表示第一条边本次无值，第二条边本次有值；不表示把第一条边从固定图中删去。

节点 $c$ 必须知道两条父边的最终结果，才能确定自己的输入。仅仅“暂时没有看见 $a$ 的值”不够；必须已经确定它本次不会产生值。第 3 节用一个额外符号表示这种结果。

这个例子还留下三个问题：节点没被选中时是否记忆输入，选择依据是否读取记忆，以及被选中后是否清理记忆。下面把它们分别定义。

## 2. 固定集合、图与逻辑时间

### 2.1 数、函数与有限序列

定义 $\mathbb N=\{0,1,2,\ldots\}$，$\mathbb N_{>0}=\mathbb N\setminus\{0\}$。对 $L\in\mathbb N_{>0}$ 以及 $b,c\in\mathbb N$、$b\le c$，写

$$
[L]=\{0,\ldots,L-1\},\qquad
[b,c)=\{\theta\in\mathbb N:b\le\theta<c\}.
$$

固定整数 $d\ge1$，令 $P=\mathbb R^d$ 为输入与输出的值空间。标量例子取 $d=1$。

一个全函数 $f:X\to Y$ 为每个 $x\in X$ 恰好指定一个 $f(x)\in Y$；它不保证这个值容易计算。若 $C$ 是有限集合、每个 $v\in C$ 有一个集合 $D_v$，则

$$
\prod_{v\in C}D_v
=\{f:f\text{ 是定义在 }C\text{ 上的函数，且 }f(v)\in D_v\}.
$$

记 $(d_v)_{v\in C}$ 为这种按 $v$ 标记的坐标族；它不是依赖纸面排列顺序的列表。$C$ 为空时只有唯一空函数，记为 $()$。对命题 $Q$，指示值 $\mathbf1[Q]$ 在 $Q$ 成立时为 $1$，否则为 $0$。

对任意集合 $X$，$X^n$ 是长度为 $n$ 的序列所成的集合。先定义下式使用的不交并。

对带标号集合族 $(X_i)_{i\in I}$，记：

$$
\bigsqcup_{i\in I}X_i
=\{(i,x)\mid i\in I,\ x\in X_i\}
$$

为它们的**不交并**。若元素自身已经唯一携带标号 $i$，便省略外层二元组的书写。长度为 $n$ 的序列自身确定 $n$，所以所有有限非空序列的集合可以定义为：

$$
\operatorname{Seq}^{+}(X)=\bigsqcup_{n\ge1}X^n.
$$

这里的不交并保留序列长度；两个数值相同的项仍可占据不同位置。空序列也写作 $()$，但不属于 $\operatorname{Seq}^{+}(X)$。

若 $I$ 是带全序的有限标签集合，定义由互异、递增标签标记的非空序列：

$$
\begin{aligned}
&\operatorname{LSeq}^{+}(I,P)\\
&\quad=
\bigsqcup_{n\ge1}
\left\{
\begin{aligned}
&((i_1,p_1),\ldots,(i_n,p_n))\in(I\times P)^n:\\
&i_1<\cdots<i_n
\end{aligned}
\right\}.
\end{aligned}
$$

标签区分来源，全序给这些二元组一个规范排列；后文的聚合接收整个二元组序列。

### 2.2 固定节点图

给定有限非空节点集 $V$、有限边集 $A$，以及函数

$$
\operatorname{src},\operatorname{dst}:A\to V.
$$

把

$$
G=(V,A,\operatorname{src},\operatorname{dst})
$$

称为固定节点图。

一列首尾相接的有向边称为有向游走；若它回到出发节点且至少含一条边，其中就含有有向环。本文要求不存在这样的游走，即图是有向无环图，简称 DAG；还要求不同边不具有同一对起终点。没有重复节点的游走称为路径，并允许只含一个节点的零长度路径。对 $v\in V$，定义

$$
\begin{aligned}
\operatorname{In}(v)
&=\{a\in A:\operatorname{dst}(a)=v\},\\
\operatorname{Out}(v)
&=\{a\in A:\operatorname{src}(a)=v\}.
\end{aligned}
$$

入口与终端节点分别是

$$
V_{\mathrm{in}}=\{v:\operatorname{In}(v)=\varnothing\},\qquad
V_{\mathrm{out}}=\{v:\operatorname{Out}(v)=\varnothing\}.
$$

有限 DAG 的入口、终端均非空。事实上，从任一节点反复向父节点走，必在入口停止，否则有限集合中会出现重复节点并形成环；沿子节点方向同理。因此每个节点都位于某条入口到终端的路径上。单个无边节点同时是入口和终端。

给 $V,A$ 分别固定全序，也就是为任意两个不同元素规定相容的先后顺序，用于确定聚合序列及选择平票结果。

### 2.3 区域划分与严格次序

给定有限非空集合 $J$ 和满射 $\rho:V\to J$。“满射”表示 $J$ 中每个元素至少对应一个节点。定义逆像 $\rho^{-1}(\{j\})=\{v\in V:\rho(v)=j\}$，并令

$$
\mathcal R_j=\rho^{-1}(\{j\}).
$$

这些集合把节点划分成互不相交的区域。区域负责在成员节点之间共同选择；消息沿节点之间的边传递。节点各自保存记忆，区域另有一份用于选择的历史，第 4 节将分别定义它们。

固定函数 $\ell:J\to\mathbb N_{>0}$，要求

$$
a\in A\Longrightarrow
\ell(\rho(\operatorname{src}(a)))<
\ell(\rho(\operatorname{dst}(a))).
\tag{1}
$$

把式 (1) 命名为关系：

$$
\begin{aligned}
&\operatorname{LegalRegionRank}_{G,\rho}(\ell)\\
&\quad\Longleftrightarrow
\forall a\in A:\quad
\ell(\rho(\operatorname{src}(a)))
<\ell(\rho(\operatorname{dst}(a))).
\end{aligned}
$$

因此同一区域内没有边，区域之间也没有有向环。节点图无环本身还不足以推出式 (1)：例如 $u\to v\to w$，把 $u,w$ 放在一个区域、$v$ 放在另一个区域，就不满足式 (1)。本教材显式采用这一较强限制。

区域拓扑次序是把所有区域排成一列，并让每条跨区域边的起点区域排在终点以前。按 $\ell$ 从小到大排列、相同层级任意排序，就得到这样的次序。这个构造也直接证明所需次序存在。

### 2.4 输入位置与统一逻辑时间

固定有限输入 $x:[L]\to P$，其中 $L\ge1$。$t\in[L]$ 表示输入位置。令

$$
r_v=\ell(\rho(v)),\qquad
r_{\mathrm{out}}=1+\max_{v\in V}r_v,
$$

并固定整数 $D>r_{\mathrm{out}}$。为节点与边界规定统一逻辑时间：

$$
\theta_{v,t}=Dt+r_v,\qquad
\theta_{\mathrm{in},t}=Dt,\qquad
\theta_{\mathrm{out},t}=Dt+r_{\mathrm{out}}.
\tag{2}
$$

称 $(\ell,D)$ 是图与区域划分的一组**合法时间编码**，记作：

$$
\begin{aligned}
\operatorname{LegalTimeCode}_{G,\rho}(\ell,D)
\Longleftrightarrow {}&
\operatorname{LegalRegionRank}_{G,\rho}(\ell)
\\
&{}\land D>1+\max_{v\in V}\ell(\rho(v)).
\end{aligned}
$$

同一区域的节点具有相同的 $\theta_{v,t}$，记为 $\theta_{j,t}$。式 (2) 为每个输入位置留出一个区间 $[Dt,D(t+1))$：输入在左端进入，各区域位于中间，输出坐标也在右端以前。它给可能发生的计算指定时间；没有输入的节点在这个坐标上仍不计算。

$D$ 与 $\ell$ 是这个具体数学规格的一部分。它们不是墙钟耗时，也不要求实际程序按这些整数逐步等待。例如，把第 1 节的 $a,b$ 放在层级 $1$ 的区域，把 $c$ 单独放在层级 $2$ 的区域，再取 $D=4$；第一个输入对应时间 $0,1,2,3$，第二个对应 $4,5,6,7$。

若局部函数忽略时间，不同满足 $\operatorname{LegalTimeCode}_{G,\rho}(\ell,D)$ 的编码可以给出相同数值；若状态按时间衰减，改变这些坐标就可能改变数值，不能再把这种变化视为纯调度重排。本文所有时间衰减均按式 (2) 的统一逻辑时间，不按“访问了几次节点”计时。

## 3. 每条边的一次结果

取 $\bot\notin P$，令 $P_\bot=P\cup\{\bot\}$。对输入位置 $t$，每条边 $a$ 的最终结果是

$$
z_{a,t}\in P_\bot.
$$

$z_{a,t}=\bot$ 表示已经确定这条边在本位置无值；它不是零向量。求值途中“尚未确定”是部分记录中缺少这个坐标，不能与已经确定的 $\bot$ 混同。

为每个入口节点 $v$ 固定一个形式标签 $\mathrm{in}_v\notin A$；这些标签两两不同。定义节点 $v$ 的输入标签集合

$$
I_v=
\begin{cases}
\{\mathrm{in}_v\},&v\in V_{\mathrm{in}},\\
\operatorname{In}(v),&v\notin V_{\mathrm{in}}.
\end{cases}
$$

入口标签集合只有一个元素；非入口标签集合继承固定边序。入口节点只接收图输入，其他节点收集有值的父边：

$$
\mathcal M_{v,t}=
\begin{cases}
((\mathrm{in}_v,x(t))),&v\in V_{\mathrm{in}},\\
((a,z_{a,t}):a\in\operatorname{In}(v),\ z_{a,t}\ne\bot)_{\text{按边序}},&\text{其余情形}.
\end{cases}
\tag{3}
$$

因此 $\mathcal M_{v,t}$ 要么是空序列，要么属于 $\operatorname{LSeq}^{+}(I_v,P)$；只有后一种情形才会把它交给聚合函数。

定义候选集合

$$
\mathcal C_{j,t}=\{v\in\mathcal R_j:\mathcal M_{v,t}\ne()\}.
\tag{4}
$$

只对候选节点定义本地内容。给每个节点固定全函数

$$
\operatorname{Agg}_v:
\mathbb N\times\operatorname{LSeq}^{+}(I_v,P)\to P,
$$

并令 $h_{v,t}=\operatorname{Agg}_v(\theta_{v,t},\mathcal M_{v,t})$。常见实例是均值：

$$
\operatorname{Agg}_v
(\theta,((i_1,m_1),\ldots,(i_n,m_n)))
=\frac1n\sum_{k=1}^n m_k.
\tag{5}
$$

这里 $i_k\in I_v$ 是入口或父边标签，$m_k\in P$ 是数值。式 (5) 忽略标签；一般聚合则可同时使用标签与数值。任意裸值聚合 $F_v:\mathbb N\times\operatorname{Seq}^{+}(P)\to P$ 都可提升为

$$
\operatorname{Agg}_v
(\theta,((i_1,m_1),\ldots,(i_n,m_n)))
=F_v(\theta,(m_1,\ldots,m_n)).
$$

所以原有的裸值聚合仍是新接口的实例。

一个简单的身份感知实例是给每个 $i\in I_v$ 固定标量 $\eta_{v,i}$，只在本次实际出现的标签上归一化：

$$
\begin{aligned}
\alpha_k
&=\frac{\exp(\eta_{v,i_k})}
{\sum_{r=1}^n\exp(\eta_{v,i_r})},\\
\operatorname{Agg}_v
(\theta,((i_1,m_1),\ldots,(i_n,m_n)))
&=\sum_{k=1}^n\alpha_km_k.
\end{aligned}
$$

这里 $\exp$ 是自然指数函数；权重随标签而变。

例如，把第 1 节的两条父边记为 $e_a:a\to c$ 与 $e_b:b\to c$。若结果分别为 $\bot,5$，则 $\mathcal M_{c,t}=((e_b,5))$，式 (5) 的均值为 $5$；$e_a$ 缺席，而不是以零值出现。

## 4. 节点状态、描述量与区域选择

### 4.1 旧状态与候选新状态

对每个节点固定非空集合 $S_v,D_v,\mathsf C_v$，分别作为状态、选择描述量、局部控制量的取值空间。初态为 $q_v^{\mathrm{init}}\in S_v$。

记 $q_v^t\in S_v$ 为处理输入位置 $t$ 以前的持久状态；令 $q_v^0=q_v^{\mathrm{init}}$。这里上标 $t$ 是输入位置，尚不是 TimedDAG 中逐逻辑时刻的状态上标。

给定全函数

$$
\operatorname{Upd}_v:S_v\times\mathbb N\times P\to S_v.
$$

对候选节点，定义

$$
\widetilde q_{v,t}=\operatorname{Upd}_v(q_v^t,\theta_{v,t},h_{v,t}).
\tag{6}
$$

它只是候选值，不表示已经改变持久状态。无状态节点可取单点集 $S_v=\{*\}$，并让所有状态函数返回 $*$。

### 4.2 三种描述量

固定全函数

$$
\begin{aligned}
\operatorname{Read}_v^0&:\mathbb N\times P\to D_v,\\
\operatorname{Read}_v^-&:S_v\times\mathbb N\times P\to D_v,\\
\operatorname{Read}_v^+&:S_v\times\mathbb N\times P\to D_v.
\end{aligned}
$$

分别得到

$$
\begin{aligned}
d_{v,t}^0
&=\operatorname{Read}_v^0(\theta_{v,t},h_{v,t}),\\
d_{v,t}^-
&=\operatorname{Read}_v^-(q_v^t,\theta_{v,t},h_{v,t}),\\
d_{v,t}^+
&=\operatorname{Read}_v^+(\widetilde q_{v,t},\theta_{v,t},h_{v,t}).
\end{aligned}
\tag{7}
$$

每个区域固定模式 $\tau_j\in\{0,-,+\}$，使用 $d_{v,t}^{\tau_j}$。模式 $0$ 只读取当前内容，模式 $-$ 读取旧记忆，模式 $+$ 读取包含本次更新的候选记忆。这里的读取只产生选择依据；候选新状态是否用于本次计算，将由第 5 节决定。

### 4.3 选择结果与局部控制量

给每个区域固定非空历史空间 $Y_j$、初态 $y_j^0\in Y_j$，以及整数 $1\le K_j\le|\mathcal R_j|$。记 $y_j^t$ 为处理位置 $t$ 以前的选择历史。对 $C\subseteq\mathcal R_j$，定义

$$
\mathsf{Choose}_{j,C}
=\{A'\subseteq C:|A'|=\min(K_j,|C|)\}.
$$

固定全函数族

$$
\operatorname{SelStep}_{j,C}:
Y_j\times\mathbb N\times\prod_{v\in C}D_v
\to
\mathsf{Choose}_{j,C}\times\prod_{v\in C}\mathsf C_v\times Y_j.
\tag{8}
$$

空积是只含空元组的集合；要求空候选时返回 $(\varnothing,(),y)$。实际选择定义为

$$
(\mathcal A_{j,t},(c_{v,t})_{v\in\mathcal C_{j,t}},y_j^{t+1})
=\operatorname{SelStep}_{j,\mathcal C_{j,t}}
\left(y_j^t,\theta_{j,t},(d_{v,t}^{\tau_j})_{v\in\mathcal C_{j,t}}\right).
\tag{9}
$$

$\mathcal A_{j,t}$ 是激活集合，$K_j$ 是该区域的选择容量。本文要求尽量填满容量：候选数不超过 $K_j$ 时全选，否则选 $K_j$ 个。因此非空候选必选出至少一个节点，这会用于证明输出存在。

$c_{v,t}$ 可以是实数权重，也可以是有限元组。它只交给节点 $v$。新的选择历史可以记录本次候选、激活集合和权重，但不能读取本次尚未计算的完整输出。

最简单的例子是按实数描述量从大到小选择，平票按节点全序处理。先让控制和历史空间都取单点集，就得到不带额外权重和历史的选择；第 5.4 节再给出实数控制量怎样影响完整输出的例子。

## 5. 本次计算与下一次记忆

### 5.1 状态采用

每个区域固定 $\kappa_j\in\{0,1\}$。定义采用集合

$$
O_j(C,A')=
\begin{cases}
A',&\kappa_j=0,\\
C,&\kappa_j=1.
\end{cases}
$$

固定区域 $j$，对其中的候选节点 $v$，定义本次计算快照：

$$
q_{v,t}^{\mathrm{cmp}}=
\begin{cases}
\widetilde q_{v,t},&v\in O_j(\mathcal C_{j,t},\mathcal A_{j,t}),\\
q_v^t,&\text{其余情形}.
\end{cases}
\tag{10}
$$

“快照”在这里就是本次完整计算将读取的那个状态值。$\kappa_j=0$ 时只有激活者采用候选新状态，$\kappa_j=1$ 时全部候选都采用。采用后的值和下一次留下的记忆还需分开：后者由下面的延续函数决定。

### 5.2 状态延续

固定全函数

$$
\operatorname{Next}_v:
S_v\times S_v\times\mathbb N\times P\times\{0,1\}\times\mathsf C_v
\to S_v.
\tag{11}
$$

令 $a_{v,t}=\mathbf1[v\in\mathcal A_{j,t}]$。候选节点的下一状态是

$$
q_v^{t+1}
=\operatorname{Next}_v(q_v^t,q_{v,t}^{\mathrm{cmp}},
\theta_{v,t},h_{v,t},a_{v,t},c_{v,t}).
\tag{12}
$$

非候选节点规定 $q_v^{t+1}=q_v^t$。默认延续返回第二个自变量 $q^{\mathrm{cmp}}$。如果要求激活后清理某份记忆，可以给定重置函数 $\operatorname{Reset}_v:S_v\times\mathbb N\to S_v$，并在激活时返回 $\operatorname{Reset}_v(q^{\mathrm{cmp}},\theta)$，否则返回 $q^{\mathrm{cmp}}$。重置哪些坐标必须写明；清理内容记忆并不自动清理选择历史。

这些值都不读取完整计算结果，$\operatorname{Next}$ 的定义和求值也不得调用或重算 Full。因此，即使下一状态被清零，本次完整计算仍使用清理前的快照。在默认延续下，快照与下一持久状态相同；其他延续规则必须单独说明保留哪些坐标。

### 5.3 完整计算与发送

固定全函数

$$
\operatorname{Full}_v:S_v\times\mathbb N\times P\times\mathsf C_v\to P.
\tag{13}
$$

只有激活节点应用它，得到

$$
g_{v,t}=\operatorname{Full}_v(q_{v,t}^{\mathrm{cmp}},\theta_{v,t},h_{v,t},c_{v,t}).
\tag{14}
$$

该节点所有出边都取同一值 $g_{v,t}$；未激活或非候选节点的所有出边都取 $\bot$。终端节点没有出边，但其 $g_{v,t}$ 仍供下一节的输出聚合读取。

至此，节点记忆与计算的分工完整了：Upd 给出候选记忆，Read 给出选择依据，Next 决定后续记忆，Full 产生本次输出。选择先确定激活集合与局部控制，再分别确定下一状态与完整输出。

### 5.4 一个带权输出的实例

取 $D_v=\mathbb R$、$\mathsf C_v=[0,1]=\{p\in\mathbb R:0\le p\le1\}$，历史空间取单点集并保持不变。选择时记 $d_{v,t}=d_{v,t}^{\tau_j}$，按这些实数描述量选出前 $\min(K_j,|\mathcal C_{j,t}|)$ 个，平票按节点全序处理；同时向每个候选节点传入

$$
c_{v,t}=p_{v,t}
=\frac{\exp(d_{v,t})}{\sum_{w\in\mathcal C_{j,t}}\exp(d_{w,t})}.
$$

这里 $\exp$ 是自然指数函数。非空候选时分母严格为正，权重之和为 $1$。选择仍按确定的排序规则进行，这些权重只作为后续函数的数值输入。

给定全函数 $G_v:S_v\times\mathbb N\times P\to P$，记 $g=G_v(q^{\mathrm{cmp}},\theta,h)$。可以把 Full 定义为 $g$，也可以定义为

$$
\operatorname{Full}_v(q^{\mathrm{cmp}},\theta,h,p)=h+p(g-h).
$$

后一个式子按 $p$ 在当前内容 $h$ 和完整结果 $g$ 之间取值：$p=0$ 时返回 $h$，$p=1$ 时返回 $g$。只有激活节点才应用 Full；未激活者即使有控制量，也不发送这个式子的值。两种输出形式与系统中的发送名称见附录 B。

## 6. 一个输入的完整结算

### 6.1 按区域次序定义

给定位置 $t$ 的输入 $x(t)$、全部 $q_v^t,y_j^t$。按任何符合式 (1) 的区域拓扑次序，依次做以下数学构造：

1. 已确定父边结果后，由式 (3)--(5) 得到输入序列、候选集合与本地内容。
2. 对候选节点用式 (6)--(7) 定义候选新状态与描述量。
3. 用式 (9) 定义激活集合、局部控制量与下一选择历史。
4. 用式 (10)--(12) 定义本次快照与下一持久状态。
5. 对激活节点用式 (14) 计算；把本区域全部出边确定为其值或 $\bot$。

空候选区域不应用节点函数，激活集合与控制族为空，选择历史保持。非候选节点在记录中没有本次 $h,\widetilde q,q^{\mathrm{cmp}},c,g$ 坐标；不能以伪造的零向量代替。

给定输出聚合全函数

$$
\operatorname{Agg}_{\mathrm{out}}:
\mathbb N\times
\operatorname{LSeq}^{+}(V_{\mathrm{out}},P)
\to P.
$$

等全部终端节点的角色确定后，取激活终端的身份和值，按节点序排列：

$$
\mathcal M_{\mathrm{out},t}
=((v,g_{v,t}):v\in V_{\mathrm{out}}\cap\bigcup_j\mathcal A_{j,t})_{\text{按节点序}},
$$

并定义输出

$$
b_t=\operatorname{Agg}_{\mathrm{out}}(Dt+r_{\mathrm{out}},\mathcal M_{\mathrm{out},t}).
\tag{15}
$$

下文证明这个序列非空，所以式 (15) 不会越出定义域。与第 3 节相同，忽略第一坐标便得到只读取终端值的聚合。

第 2 节的固定图、区域、顺序和逻辑时间规则，连同第 3--6 节的空间、初态与全部函数，构成一个 **SettleGraph 规格**。输入长度 $L$ 与值序列 $x$ 是给定规格以后另选的数据；改变输入不改变这些函数或固定边。

### 6.2 有限性与唯一性

先只执行第 6.1 节的区域构造，暂不做最后的输出聚合。有限区域可以按拓扑次序依次处理，所以全部区域角色与边结果都能确定。还需检查输出聚合的非空定义域。

> [!lemma] 引理 1：输出存在
> 每个输入位置至少有一个激活终端节点。

**证明。** 入口非空且都获得输入，所以至少一个区域有非空候选，从中必选出节点。在全部激活节点中，取区域层级最大的一个 $v$。若 $v$ 不是终端，它会沿某条出边向更高区域发送值；那个区域因而有非空候选，并必选出一个节点。这与 $v$ 的层级最大矛盾。因此 $v$ 是激活终端。$\square$

如果允许非空候选选出空集，引理就可能失效。例如第 1 节的首区域一个节点也不选，终端便没有输入。式 (8) 的非空选择条件正是这里需要的前提。

> [!theorem] 定理 2：有限性与唯一性
> 对每个输入位置及给定旧状态，第 6.1 节在有限次函数作用后确定所有边结果、候选与激活集合、控制量、状态及输出；任何符合依赖的区域次序得到相同结果。

**证明。** 某区域的全部父边来自更前区域；按拓扑次序归纳，每一步的函数自变量都已唯一确定。每区域只处理一次，区域和节点有限，每个被调用函数均为全函数，故函数作用数有限。引理 1 又保证最终聚合的序列非空，从而输出也有定义。

再固定任一拓扑序得到的记录。对另一合法次序中的区域归纳，它的输入父边来自已经完成的前驱，而这些前驱按归纳假设与固定记录相同。其旧节点状态和旧选择历史也相同，所有确定函数因而给出同一结果。于是全部区域结果及最终输出都与固定记录相同。$\square$

“有限次函数作用”是抽象数学结论；若每个局部函数另有终止的实现，才相应得到有限实现工作。

### 6.3 完整记录包含哪些坐标

给定整个输入 $x:[L]\to P$，定义完整记录 $\mathcal T_x^{\mathrm S}$ 为下列坐标族组成的有序元组：

1. 每个 $t\in[L]$、$v\in V$ 的输入序列 $\mathcal M_{v,t}$，以及每条 $a\in A$ 的边结果 $z_{a,t}$；
2. 每个 $t\in[L]$、$j\in J$ 的候选集合与激活集合；
3. 每个 $t\in[L]$、$v\in\mathcal C_{\rho(v),t}$ 的 $h_{v,t},\widetilde q_{v,t},d_{v,t}^{\tau_{\rho(v)}},c_{v,t},q_{v,t}^{\mathrm{cmp}}$；
4. 每个 $t\in[L]$、$v\in\mathcal A_{\rho(v),t}$ 的完整输出值 $g_{v,t}$；
5. 每个 $0\le t\le L$ 的全部持久节点状态 $q_v^t$ 与选择历史 $y_j^t$；
6. 每个 $t\in[L]$ 的终端序列 $\mathcal M_{\mathrm{out},t}$ 与最终输出 $b_t$。

第三、四项的定义域不同：未激活候选有本次快照和下一状态，却没有完整输出值。固定这些定义域以后，定理 2 按输入位置递推，唯一确定整份有限记录。后面说“记录对应”，比较的是这些全部坐标，而不只是最后一项中的数值。

## 7. 两个输入的手算

取标量情形 $P=\mathbb R$，以及第 1 节的节点 $V=\{a,b,c\}$。给两条边命名为 $e_a:a\to c$、$e_b:b\to c$，边序为 $e_a$ 在 $e_b$ 前，节点序为 $a,b,c$。区域 $j_0$ 为 $\{a,b\}$，区域 $j_1$ 为 $\{c\}$，两者都取 $K=1$。令 $\ell(j_0)=1,\ell(j_1)=2,D=4$。所有节点和输出聚合均采用与式 (5) 同形的规则，也就是忽略标签并对值取均值。

全部局部控制空间和两区域的历史空间都是单点集 $\{*\}$，历史初态为 $*$，每次选择返回各候选的单点控制量并保持历史。区域 $j_0$ 取 $\tau_{j_0}=+$、$\kappa_{j_0}=1$。节点 $a,b$ 的状态与描述量空间都是 $\mathbb R$，状态初值分别为 $0,1$，函数为

$$
\begin{aligned}
\operatorname{Upd}(q,\theta,h)&=q+h,\\
\operatorname{Read}^0(\theta,h)&=h,\\
\operatorname{Read}^-(q,\theta,h)&=q,\\
\operatorname{Read}^+(q,\theta,h)&=q.
\end{aligned}
$$

所以本次使用的 $+$ 描述量就是候选新状态。对任意非空候选集合，选择描述量较大的一个，平票按节点序；空候选按第 4.3 节的恒等规则。两个候选均采用更新，完整计算为

$$
\operatorname{Full}(q^{\mathrm{cmp}},\theta,h,*)=h+q^{\mathrm{cmp}}.
$$

节点 $a,b$ 的 Next 在激活时返回 $0$，否则返回本次快照。这样所有模式的读取函数都有定义，实际求值只采用声明的 $+$ 模式。

节点 $c$ 的状态和描述量空间也取单点集 $\{*\}$，状态初值为 $*$，Upd、Next 和三种 Read 恒返回 $*$；完整计算为 $\operatorname{Full}_c(*,\theta,h,*)=h$。其区域取 $\tau_{j_1}=0,\kappa_{j_1}=0$，有候选时总选中 $c$。至此，例子的全部集合、初态与局部函数均已指定。

输入序列是 $x(0)=2,x(1)=1$。

| 输入位置 | 旧状态 $(q_a^t,q_b^t)$ | 两个计算快照 | 激活节点 | 下一状态 | 输出 |
|---|---|---|---|---|---|
| $t=0$ | $(0,1)$ | $(2,3)$ | $b$ | $(2,0)$ | $2+3=5$ |
| $t=1$ | $(2,0)$ | $(3,1)$ | $a$ | $(0,1)$ | $1+3=4$ |

在第一个位置，$z_{e_a,0}=\bot$，$z_{e_b,0}=5$。所以 $c$ 的聚合输入是单项序列 $((e_b,5))$，不是补零后的 $((e_a,0),(e_b,5))$，其均值为 $5$。第二个位置两条边的角色相反。

这个例子同时区分了四件事：两个节点都有输入；两个节点都接纳输入；只有一个做完整计算；被选节点本次读取旧记忆与新输入共同形成的快照，却不把它留到下一位置。

若把延续函数改成始终保留快照，第一步之后状态变成 $(2,3)$，第二步会再次选择 $b$。因此清理是模型语义，而不仅是临时存储回收。

**练习。** 每次从本节原设定出发，分别计算以下三个变式。

1. 将区域 $j_0$ 的描述量模式改为 $\tau_{j_0}=-$，重新计算两行表格。
2. 将区域 $j_0$ 的采用模式改为 $\kappa_{j_0}=0$，重新计算两行表格。
3. 将节点 $a,b$ 的控制空间改为 $[0,1]$，按第 5.4 节产生 $p_v$，并把它们的 Full 改为 $h+p_vq^{\mathrm{cmp}}$；Next 保持原来的清理规则并忽略控制量。节点 $c$ 的全部定义保持不变。计算输出如何变化；权重只在 $a,b$ 的完整输出公式中使用。

## 8. 按统一逻辑时间衰减

固定正整数 $k$。一种状态表示是 $q=(z,\tau)\in\mathbb R^k\times\mathbb N$：保存值 $z$ 和该值对应的逻辑时间 $\tau$。本节取已声明初始值 $z_0\in\mathbb R^k$，初态为 $(z_0,0)$。固定 $0<\alpha\le1$，在 $\theta\ge\tau$ 时定义有效值

$$
\operatorname{Eff}(q,\theta)=\alpha^{\theta-\tau}z.
\tag{16}
$$

为使函数在整个声明定义域上为全函数，以下统一把指数写为 $\max(0,\theta-\tau)$。固定全函数 $H:P\to\mathbb R^k$，取

$$
\operatorname{Upd}((z,\tau),\theta,h)
=(\operatorname{Eff}((z,\tau),\theta)+H(h),\theta).
$$

本节的 Next 只采用第 5.2 节的两种形式：保留本次快照，或在激活时重置为 $(0,\theta)$。初始时间戳为零，更新和重置只写当前逻辑时间，未采用时保留原时间戳；由于同节点后续输入的逻辑时间递增，归纳得到每次实际读取都有 $\theta\ge\tau$。若另允许写入未来时间戳，就不能援用这个结论。

本例的旧状态描述量通过 $\operatorname{Eff}(q,\theta)$ 读取衰减后的内容；候选新状态已经标记为当前时间，读取它时不会重复衰减。

空档期间保存的 $(z,\tau)$ 保持不变；下一次读取才计算经过了多少逻辑时间。这与每步衰减给出相同有效值，因为

$$
\alpha^{a+b}z=\alpha^b(\alpha^az).
$$

它没有在空档产生自主激活或输出。若比较逐时间状态，必须比较经过式 (16) 解码的有效量，而不是声称保存坐标本身逐时变化。

同一区域的选择历史也可以保存值与上次更新时间，在下一次非空选择时补算衰减，再根据激活集合增加恢复惩罚或扣减积累值。这仍不依赖 Full。

式 (2) 中同节点相邻输入位置相隔 $D$，所以这种例子的相邻位置衰减因子是 $\alpha^D$。若一个节点从逻辑时间 $2$ 到 $10$ 才再次收到输入，即使它们在压紧列表中相邻，也必须使用间隔 $8$。

## 9. 整段输入与分段继续

定义全部持久状态的乘积空间及位置 $t$ 以前的状态值：

$$
\mathsf S=\left(\prod_{v\in V}S_v\right)\times
\left(\prod_{j\in J}Y_j\right),\qquad
S_t=((q_v^t)_{v\in V},(y_j^t)_{j\in J})\in\mathsf S.
$$

定理 2 定义了一个带绝对输入位置的转移：

$$
T:\mathbb N\times\mathsf S\times P\to P\times\mathsf S,\qquad
T(t,S_t,x(t))=(b_t,S_{t+1}).
$$

对 $0\le p\le q\le L$，用 $T$ 依次计算位置 $p,\ldots,q-1$，得到区间转移 $T_{p:q}$，返回有序输出序列和右边界状态。空区间返回空输出与原状态。

> [!theorem] 定理 3：分段继续
> 若 $p\le m\le q$，先执行 $T_{p:m}$，再以其右状态执行 $T_{m:q}$，拼接输出，结果恰等于 $T_{p:q}$。

**证明。** 第一段与整段在每一步使用同一绝对位置、同一输入和同一旧状态，归纳得到相同输出和 $S_m$。第二段于是具有同样的初态；对余下位置继续归纳即可。$\square$

因此在输入位置边界停止时，保存 $t,S_t$ 足以继续；状态中记录的逻辑时间戳也必须保留。不能在每个新块把 $t$ 重置为零。若在一个输入的内部停止，还需要保存尚未完成的局部记录，不能只用这个较小边界。

这证明前向分段语义，不证明梯度相同。若人为截断段间反向传播，训练梯度可以改变，而这里的前向等式仍成立。

## 10. 嵌入 TimedDAG

本节使用 TimedDAG 教材已经定义的带标记原子、时间纤维和记录坐标。证明的思路是：给入口和输出各加一个边界节点，用区域层级之差作为边时延，再逐项比较两边的记录。前九节已经独立定义了 SettleGraph，本节只建立它与更一般对象的关系。

### 10.1 增加边界适配节点

本节用帽子标记 TimedDAG 的对象，以区别它们与 SettleGraph 的坐标。取两个不属于 $V$ 的节点 $s,o$，以及不属于 $J$ 的两个区域标记 $j_s,j_o$。增加边

$$
a_v^{\mathrm{in}}:s\to v\quad(v\in V_{\mathrm{in}}),
\qquad
a_v^{\mathrm{out}}:v\to o\quad(v\in V_{\mathrm{out}}),
$$

并规定这些新边身份彼此不同，也不属于原边集 $A$。扩展后的节点集为 $\widehat V=V\cup\{s,o\}$；保留原区域划分，并把 $s,o$ 各放入独立单节点区域。

令 $r_s=0,r_o=r_{\mathrm{out}}$。对新旧全部边定义

$$
\widehat\delta(a)
=r_{\widehat{\operatorname{dst}}(a)}
-r_{\widehat{\operatorname{src}}(a)}>0.
\tag{17}
$$

因此扩展图也无环。唯一外部输入端口 $i_\star$ 进入 $s$，位置 $t$ 在时间 $Dt$ 注入 $x(t)$；唯一外部输出端口 $o_\star$ 由节点 $o$ 持有。所有节点的本地内容空间均取 $P$。

原节点保留自己的状态、描述量、控制空间、初态、Upd、Read 与 Next；原区域保留历史、初态和选择函数，并令

$$
\widehat\tau_j=\tau_j,\qquad
\widehat\kappa_j=\kappa_j,\qquad
\widehat K_j=K_j.
$$

因此描述量模式、状态采用模式与选择容量也逐项相同。式 (8) 的激活数恰等于容量与候选数的较小值，满足 TimedDAG 的“不超过容量”条件。

对两个适配节点 $u\in\{s,o\}$，把状态、描述量、局部控制和所属区域的历史空间全部取为单点集 $\{*\}$，初态均为 $*$。Upd、Next 和三种 Read 都恒返回 $*$；区域模式取 $\tau=0,\kappa=0,K=1$。对任意 $C\subseteq\{u\}$，选择函数完整定义为

$$
\widehat{\operatorname{SelStep}}_{j_u,C}
(*,\theta,(*)_{v\in C})
=(C,(*)_{v\in C},*).
$$

这也包括空候选的恒等情形。输入聚合 $\widehat{\operatorname{Agg}}_s(\theta,B)$ 定义为有限原子集合 $B$ 中全部数值的和，空和为零向量；实际非空纤维只有唯一外部输入，所以它返回 $x(t)$。其 Full 在每条出边上返回本地内容 $h$，外部输出端口族为空。

输出节点的聚合把入边 $a_v^{\mathrm{out}}$ 还原为终端节点 $v$，按节点序将 $(v,m_v)$ 交给 $\operatorname{Agg}_{\mathrm{out}}$。这个定义在非空、每条入边至多一个原子且所有原子的到达时间均为 $\theta$ 时使用；其他自变量统一返回零向量，使聚合在 TimedDAG 要求的整个定义域上为全函数。输出节点的 Full 在唯一端口 $o_\star$ 返回 $h$，内部出边族为空。

两个新节点只是编码中的边界适配对象，不能算作原 SettleGraph 新增的可训练层。入口、终端宽度仍属于需要核算的边界成本。

### 10.2 消息与聚合如何对齐

对原边 $a:u\to v$，有

$$
(Dt+r_u)+\widehat\delta(a)=Dt+r_v.
\tag{18}
$$

新输入边同样把 $Dt$ 的值送至 $Dt+r_v$，新输出边把终端值送至 $Dt+r_o$。沿路径相加时中间秩抵消，所以任意从 $s$ 到同一节点的路径都落到同一个坐标。不同输入位置在该节点具有不同坐标；同一区域的候选具有相同坐标。

原节点的 TimedDAG 聚合先把入口消息映成 $(\mathrm{in}_v,m)$，或把内部消息映成 $(a,m)$，按原顺序排列后应用 $\operatorname{Agg}_v$。在空集合、同一边多次出现、原子到达时间不全等于 $\theta$ 等其他自变量上，统一返回零向量。实际计算不会进入这些额外情形；它们只用于补全函数定义域。

对任意原节点 $v$，先用原 Full 计算 $g=\operatorname{Full}_v(q^{\mathrm{cmp}},\theta,h,c)$，再令 TimedDAG 的 Full 在扩展图的每条出边坐标上返回 $g$，外部端口族为空。这也包括原终端到 $o$ 的新增边。未激活节点不应用 Full，不产生消息。至此扩展规格的全部局部函数已经确定。

### 10.3 完整记录对应定理

记 TimedDAG 的持久状态与历史为 $\widehat q_v^\theta,\widehat y_j^\theta$。输入位置 $t$ 对应逻辑时间块 $[Dt,D(t+1))$。有限计算最后一个可能非空时刻为 $D(L-1)+r_o$；若它的下一时刻仍小于 $DL$，把状态与历史保持不变、事件与消息保持为空，延拓到 $DL$。这个规范空闲延拓不增加任何实际计算，只使最后一个位置边界也有统一记号。

对原节点与原区域，所要证明的状态映射是

$$
\begin{aligned}
\widehat q_v^{Dt}
=\widehat q_v^{Dt+r_v}&=q_v^t,\\
\widehat q_v^{Dt+r_v+1}&=q_v^{t+1},\\
\widehat y_j^{Dt}
=\widehat y_j^{\theta_{j,t}}&=y_j^t,\\
\widehat y_j^{\theta_{j,t}+1}&=y_j^{t+1}.
\end{aligned}
$$

以下三类局部坐标逐项对应；每行只在原记录定义了该坐标时比较：

| SettleGraph 坐标 | TimedDAG 坐标 |
|---|---|
| 候选集合、激活集合 $\mathcal C_{j,t},\mathcal A_{j,t}$ | $\widehat{\mathcal C}_{j,\theta_{j,t}},\widehat{\mathcal A}_{j,\theta_{j,t}}$ |
| 候选节点的 $h,\widetilde q,d^{\tau_j},c,q^{\mathrm{cmp}}$ | 同一原节点在逻辑时间 $Dt+r_v$ 的相应五个坐标 |
| 激活节点的 $g_{v,t}$ | 本节点 Full 在每条扩展出边上的同一个值 |

此外，若 $z_{a,t}\in P$，则原边槽位对应消息

$$
(\mathrm{msg},Dt+r_{\operatorname{src}(a)},a,z_{a,t}).
$$

若 $z_{a,t}=\bot$，则这个边和发送时间的槽位没有消息。输出 $b_t$ 对应外部记录

$$
(\mathrm{out},Dt+r_o,o_\star,b_t).
$$

由完整消息记录与固定槽位，可以反向恢复每个 $z_{a,t}$，再按式 (3) 恢复全部输入序列。适配节点和它们的边是辅助坐标；删除这些坐标并作上述还原，就得到第 6.3 节的完整记录 $\mathcal T_x^{\mathrm S}$。

> [!theorem] 定理 4：向 TimedDAG 的记录嵌入
> 上述编码在全部原节点、原区域和输入位置满足上述对应关系；包括本地量、选择、控制、计算快照、持久状态、历史、全部边结果和输出。

**证明。** 对输入位置归纳，并在同一位置内按区域次序归纳。位置 $0$ 的原节点和区域初态按构造相同。输入适配节点把同一值送至各入口。式 (18) 使全部原父消息与目标位置对齐；第 10.2 节的映射给出相同的聚合输入。状态、读取、选择、快照、延续与完整输出函数相同，因而本区域全部相应坐标相同；非候选节点和空候选区域则分别保持状态与历史。输出适配节点应用同一输出聚合，所得输出也相同。

所有消息的发送、到达和输出逻辑坐标均不大于 $Dt+r_o<D(t+1)$。因此规范切面 $D(t+1)$ 处没有本位置遗留的消息，原节点与区域的右状态正好成为下一位置的左状态。这个逻辑时间事实没有规定实际机器上的完成时刻。归纳完成，同时得到原边的有值与无值槽位对应。$\square$

这证明的是按明确映射的完整记录等价，不要求两个模型原始记录具有相同坐标集合。较一般的 TimedDAG 可具有多输入输出、不对齐的路径时间、区域商图有环或每边不同输出，均不必属于本文 SettleGraph。

### 10.4 无值结果、封闭下界与完整切面

SettleGraph 的 $z_{a,t}=\bot$ 只说明本位置的这个边槽位没有消息。TimedDAG 的边封闭下界则表示：该边全部到达时间小于此下界的实际消息已经公开。这是一个关于到达时间前缀的命题。

在此前位置的消息均已公开、本位置的有值结果也已公开或无值结果已确定后，编码中的该边可以把封闭下界推进到严格大于 $Dt+r_{\operatorname{dst}(a)}$。取全部父边下界的最小值，即可证明目标纤维关闭。一次无值结果与整条边永久关闭仍是两个不同命题。

在完整输入位置边界 $Dt$，编码没有前一位置遗留的在途消息；删除适配节点的单点状态以后，TimedDAG 的切面状态恰对应第 9 节的 $t,S_t$。内部逻辑时间切面仍可能有在途消息，必须保留一般教材规定的全部边界坐标。

## 11. 节点级时间批的含义

本节先固定四个术语。**作用成本标记**是把第 10 节编码中的每个规范函数作用归入一个有限成本种类，并指定其中哪些种类是本次分析的主要作用。一个**精确联合契约**是一个带明确输入、输出类型的全函数；它在整个合法边界域上返回与逐作用参考递归相同的规范坐标。若这个联合函数从左边界节点状态和一段已知驱动量返回整段中间状态及右边界状态，其覆盖集合称为一个**状态块**。状态块数量是外层计数；联合函数内部依赖图的最长路径成本称为**块内跨度**。这些定义与 [[timed-dag-chunk-prefill-learning-note|分块教材]] 的一般定义一致，本节只使用它们的严格分层特例。

固定一个输入位置区间，并假定某区域整段父输入已经由更前区域确定。因为 Next 和选择历史更新都不读取本区域 Full，便可以先沿时间扫描本区域的全部准备、选择与状态处理，得到每次完整计算的四个自变量

$$
(q_{v,t}^{\mathrm{cmp}},\theta_{v,t},h_{v,t},c_{v,t}).
$$

随后按节点收集所有激活位置，逐节点成批求值 Full。再把结果交给更后区域。式 (1) 保证不会返回已经扫描的区域。

要得到正式的节点级时间批结论，还需声明作用成本，并为状态递归、完整输出或二者的联合求值给出精确契约。满足这些条件时，[[timed-dag-chunk-prefill-learning-note|分块教材]] 的严格分层证明可用于第 10 节的编码。状态块的外层数量、块内并行深度与具体设备效果分别计量。

本节依赖于 Next 和选择历史更新不读取 Full。若下一状态依赖完整输出，后续调用所需的快照就可能无法在本批 Full 以前确定；若该状态还影响共同选择，则选择本身也可能需要等待。这些扩展需要重新定义事件图与批求值次序，不属于本文规格。

## 12. 接入已有模型时的函数保持

若已有模型某处在位置 $t$ 输出 $h_t\in P$，新图输出 $b_t\in P$，则图残差为 $b_t-h_t$。一种接入方式是在指定位置加上这个残差。函数保持不是图脱离接入范围的一元性质；先固定一个非空的**接入实例类** $\varnothing\ne\mathfrak X$。每个 $\chi\in\mathfrak X$ 明确给出：

1. 已有模型输入、所比较的初始化及插入位置；
2. 某个 $L_\chi\in\mathbb N_{>0}$ 以及一段有限参考值 $(h_{\chi,t})_{t\in[L_\chi]}$；
3. SettleGraph 的初始节点状态与 selector-history；
4. 以 $h_{\chi,t}$ 为图输入得到的第 6 节完整记录。

下文把该记录中的坐标加上标 $\chi$。这四项使“可达”只表示由某个 $\chi\in\mathfrak X$ 的完整记录实际出现。若 $\mathbf m=((i_1,m_1),\ldots,(i_n,m_n))$，记 $\operatorname{lab}(\mathbf m)=(i_1,\ldots,i_n)$。定义节点聚合的可达支持域：

$$
\begin{aligned}
\operatorname{AdmAgg}_{\mathfrak X}(v)
=
\{(&\theta_{v,t},
\operatorname{lab}(\mathcal M^\chi_{v,t}),
h_{\chi,t})
\mid
\chi\in\mathfrak X,\ t\in[L_\chi],
\mathcal M^\chi_{v,t}\ne()\}.
\end{aligned}
\tag{19}
$$

类似地定义实际 Full 调用域：

$$
\begin{aligned}
\operatorname{AdmFull}_{\mathfrak X}(v)
=
\{(&q^{\mathrm{cmp},\chi}_{v,t},\theta_{v,t},
h^\chi_{v,t},c^\chi_{v,t})
\mid
\chi\in\mathfrak X,\ t\in[L_\chi],
v\in\mathcal A^\chi_{\rho(v),t}\},
\end{aligned}
\tag{20}
$$

以及输出聚合的可达支持域：

$$
\begin{aligned}
\operatorname{AdmOut}_{\mathfrak X}
=
\{(&Dt+r_{\mathrm{out}},
\operatorname{lab}(\mathcal M^\chi_{\mathrm{out},t}),
h_{\chi,t})
\mid
\chi\in\mathfrak X,\ t\in[L_\chi]\}.
\end{aligned}
\tag{21}
$$

这里式 (19) 与式 (21) 只从实际记录读取标签支持，再把第三坐标 $h_{\chi,t}$ 用作待检查的共同 payload；它们没有预先假定实际消息值已经等于 $h_{\chi,t}$。

定义 $\operatorname{FunctionPreserving}_{\mathfrak X}$ 成立，当且仅当下列三个全称条件同时成立：

$$
\begin{aligned}
&\forall v\in V,
\forall(\theta,(i_1,\ldots,i_n),h)
\in\operatorname{AdmAgg}_{\mathfrak X}(v):\\
&\qquad
\operatorname{Agg}_v
(\theta,((i_1,h),\ldots,(i_n,h)))=h,\\[2mm]
&\forall v\in V,
\forall(q,\theta,h,c)
\in\operatorname{AdmFull}_{\mathfrak X}(v):\\
&\qquad
\operatorname{Full}_v(q,\theta,h,c)=h,\\[2mm]
&\forall(\theta,(v_1,\ldots,v_n),h)
\in\operatorname{AdmOut}_{\mathfrak X}:\\
&\qquad
\operatorname{Agg}_{\mathrm{out}}
(\theta,((v_1,h),\ldots,(v_n,h)))=h.
\end{aligned}
\tag{22}
$$

若 $\operatorname{FunctionPreserving}_{\mathfrak X}$ 成立，则对每个 $\chi\in\mathfrak X$ 与 $t\in[L_\chi]$，按区域顺序归纳，所有有值消息均等于 $h_{\chi,t}$，最终：

$$
b_{\chi,t}=h_{\chi,t}.
$$

具体地，入口序列的载荷本来就是 $h_{\chi,t}$。若某个区域以前的全部有值边结果都等于它，则当前节点的实际标签支持属于式 (19)，式 (22) 的第一行使聚合结果仍为 $h_{\chi,t}$；实际激活调用属于式 (20)，第二行又使每条新有值边结果仍为 $h_{\chi,t}$。区域归纳完成后，实际终端支持属于式 (21)，第三行给出上述输出等式。状态、描述量、选择与控制量可以影响哪些标签实际出现，但不会越出这些可达域。

因而图残差在声明的实例类上为零。忽略标签的均值、按标签产生但归一化后权重和为 $1$ 的加权平均，都是满足相应条件的实例；任意身份感知聚合则未必满足。若希望使用比可达域更强、与 $\mathfrak X$ 无关的充分条件，可以把式 (22) 的定义域扩大到全部类型正确的时间、非空标签列、状态、控制量与 $h\in P$，但必须明确写出这个更强量词。

新增私有状态可以在后台演化，只要它在 $\mathfrak X$ 的可达域内不破坏式 (22)。若函数保持声明还比较已有模型状态，则必须另外给出初始化嵌入 $\eta$ 与状态投影 $\pi$。把两边在位置边界的状态轨迹分别记为 $s^{\mathrm{base}}_{\chi,t}$ 与 $s^{\mathrm{grown}}_{\chi,t}$；还必须明确要求：

$$
\begin{aligned}
s^{\mathrm{grown}}_{\chi,0}
&=\eta(s^{\mathrm{base}}_{\chi,0}),\\
\pi(s^{\mathrm{grown}}_{\chi,t})
&=s^{\mathrm{base}}_{\chi,t}
&&\left(\chi\in\mathfrak X,\ 0\le t\le L_\chi\right).
\end{aligned}
$$

而不是要求所有新增坐标永远等于初始值。进一步推导见 [[memos/mathematics/function-preserving-growth|函数保持生长备忘]]。

这里证明的是相对于明示实例类与特定函数选择的输出保持。具体节点算法、插入位置、训练目标和实验结果由实验平台维护；式 (22) 并不保证训练以后仍保持原模型。

## 附录 A：系统用语对照

| 系统用语 | 本文数学对象 |
|---|---|
| receiver | 固定节点 $v\in V$ |
| reached | $\mathcal M_{v,t}\ne()$，即属于候选集合 |
| active | 属于 $\mathcal A_{j,t}$ |
| DATA / CLOSED | 已确定的 $z_{a,t}\in P$ / $z_{a,t}=\bot$ |
| seal / continuation | 第 10.4 节的封闭下界／完整切面上的继续状态 |
| region selector | 函数族 SelStep，不是普通消息节点 |
| proposal | 候选新状态 $\widetilde q_{v,t}$ |
| Observe | 式 (10) 的输入接纳／计算快照规则 |
| commit / carry | 显式规定的下一持久状态；不能替代快照与延续的区分 |
| N / SD / BO | 单点状态；仅激活者采用；全部候选采用。后两者配默认 Next 才恢复旧完整状态规则 |
| content / pre / post | 描述量模式 $0,-,+$ |
| Plan | 固定集合、函数、初态、时间编码与选择规则的一份具体规格 |
| prefill / decode | 一段输入的求值／逐输入位置继续；第 9 节给出前向相容性 |
| checkpoint | 在模型生长语境中指已有模型的权重；运行中断点所需的继续状态另见第 9、10.4 节 |

## 附录 B：发送权重与指定反向规则

本附录另外使用微分与链式法则。以下先固定激活集合，把 $h,g,p$ 视为一个局部函数的独立输入；它们若由共同参数产生，反向贡献还需沿各自依赖汇总。

设 $h,g\in\mathbb R^d,p\in[0,1]$。HARD 的前向为 $g$；SOFTP 的前向为 $h+p(g-h)$。两者是不同函数。

HST 的前向也为 $g$，但附加一种指定反向规则。若用符号 $\operatorname{sg}(p)$ 表示前向等于 $p$、反向贡献指定为零的操作，可写

$$
\eta=1+\zeta\bigl(p-\operatorname{sg}(p)\bigr),\qquad
\widehat g=h+\eta(g-h),
$$

其中 $\zeta$ 是固定实数。前向有 $\eta=1$。设某个标量损失传到输出的反向信号为向量 $\bar g\in\mathbb R^d$，则指定到 $p$ 的这一条局部反向贡献为

$$
\bar p=\zeta\langle\bar g,g-h\rangle.
$$

这里 $\langle u,v\rangle=\sum_{k=1}^d u_kv_k$。对独立输入 $g,h$ 的另外两条局部贡献分别为 $\bar g$ 和零向量。

这不是把恒等于 $1$ 的普通函数求导得到非零结果，而是在普通前向函数之外另选了一条训练规则。离散激活集合的导数也不由本文自动定义。是否保持这种指定反向规则，需要另外比较训练计算图；前向记录相同不足以证明梯度相同。
