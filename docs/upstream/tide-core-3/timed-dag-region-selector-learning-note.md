---
type: mathematical-learning-note
status: active-learning
as-of: 2026-09-17
cssclasses:
  - textbook-math
tags:
  - tide
  - timed-dag
  - region
  - selector
  - selector-history
  - logical-time
  - mathematics
  - learning-note
---

# 带区域选择的 TimedDAG：从零开始的数学定义

> [!summary] 本文的阅读前提
> 本文只假设读者熟悉集合、函数、自然数、有限求和、最小值和数学归纳法。有向图、逻辑时间、输入端口、消息、时间纤维、区域、候选集合、选择函数、选择历史、封闭下界以及事件图都会在本文中重新定义。
>
> 本文使用的 TIDE 对象都在文内定义，其他文档与讨论均不是阅读前提。区域、选择函数和完整 TimedDAG 规格分别在第 2.4、5.3、6.5 节获得精确定义。
>
> 正文先给数学对象，再给例子和定理；计算机系统中常用说法的集中解释放在附录 S。删除附录 S 后，正文仍然构成完整的数学规格。

> [!tip] 分四次阅读
> 第一次读第 1--3 节，目标是能从式 (8) 独立算出一个时间纤维。第二次读第 4--7 节，先手算第 7.1--7.5 节的完整例子，再用第 7.6--7.9 节分别检查状态、选择历史与局部控制。第三次研究第 8--9 节的封闭下界与阶段化暴露。第四次再读第 10 节以后的事件图、切面与联合求值；它们不是理解主定义的前置。

本文要定义的是一个有限确定性模型。它具有以下能力：

1. 有有限多个外部输入端口，也有有限多个外部输出端口；
2. 每个外部输入位置带有一个自然数时间；
3. 节点沿固定有向边发送值，每条边增加一个正整数时间；
4. 到达同一节点、同一时间的全部外部输入和内部消息被共同处理；
5. 节点集合被划分为若干区域；同一区域中，在同一时间收到至少一个输入的全部节点共同参加一次选择；
6. 在阶段化暴露模型中，只有能够证明这组节点不会再增加时，才可以暴露相应的选择；
7. 每个区域的选择函数可以保存显式跨时间历史，并给每个候选节点一个局部控制量；
8. 区域不是节点，不接收普通消息，也不发送普通消息；
9. 区域之间不要求构成 DAG；
10. 节点区分本次计算状态与下一持久状态，后者可以按选择结果清空或写回历史，但不读取完整输出。

`Tide` 在本文中只是这条研究线的名称，不是额外的数学对象；本文实际定义的对象是第 1--6 节给出的带区域选择的 TimedDAG 规格。

正文先区分完整结果与它的部分暴露。

- 第 1--7 节先假定一次输入全部给定，定义唯一的完整结果。这是模型本身的含义。
- 第 8--9 节再在固定的 $\mathcal T_x$ 上定义阶段化暴露轨迹，并把“可见、完成、关闭”对应到其中的函数与集合关系。它排列规范记录的坐标，不另行定义计算结果。

这个顺序很重要：必须先知道“正确的完整结果是什么”，才能定义“一个部分暴露何时已经包含足够的规范坐标”。第 10--12 节随后研究这份结果的依赖图、分段继续与联合求值条件；第 13--16 节整理定义边界和学习线索。

## 1. 基础记号

### 1.1 自然数、有限集合与半开区间

定义：

$$
\mathbb N=\{0,1,2,\ldots\},
\qquad
\mathbb N_{>0}=\{1,2,3,\ldots\}.
$$

若 $L\in\mathbb N_{>0}$，定义：

$$
[L]=\{0,1,\ldots,L-1\}.
$$

若 $r,s\in\mathbb N$ 且 $r\le s$，定义：

$$
[r,s)=\{\theta\in\mathbb N\mid r\le\theta<s\}.
$$

对任意集合 $X$，用：

$$
\mathcal P_{\mathrm{fin}}(X)
$$

表示 $X$ 的所有有限子集所成的集合，用 $|X|$ 表示有限集合 $X$ 的元素个数。

对任意集合 $E,Y$，定义：

$$
Y^E=\{f\mid f:E\to Y\}.
$$

也就是说，$Y^E$ 是所有从 $E$ 到 $Y$ 的函数所成的集合。

本文说一个映射是**全函数**，只表示定义域中的每个元素都恰有一个函数值；它不表示这个函数一定容易计算。

本文还使用以下逻辑记号：

$$
\forall x\in X,\ P(x)
$$

表示集合 $X$ 的每个元素都满足命题 $P$；记号 $\exists x\in X:P(x)$ 表示至少有一个元素满足它。若 $f:X\to Y$ 且 $D\subseteq Y$，定义 $D$ 在 $f$ 下的**逆像**：

$$
f^{-1}(D)=\{x\in X\mid f(x)\in D\}.
$$

逆像是一种集合构造，不要求 $f$ 可逆。例如，后文的 $\rho^{-1}(\{j\})$ 就是所有被 $\rho$ 映到 $j$ 的节点。

对一个命题 $P$，记 $\mathbf1[P]$ 为它的指示值：$P$ 成立时取 $1$，不成立时取 $0$。例如 $\mathbf1[v\in A]$ 表示节点 $v$ 是否属于集合 $A$。

若有限个集合 $X_1,\ldots,X_k$ 两两不交，记它们的并为：

$$
X_1\sqcup\cdots\sqcup X_k.
$$

符号 $\sqcup$ 强调每个元素的来源分支可以由元素本身唯一恢复。对未必两两不交的带标号集合族 $(X_i)_{i\in I}$，其带标号不交并写成：

$$
\bigsqcup_{i\in I}X_i
=\{(i,x)\mid i\in I,\ x\in X_i\}.
$$

### 1.2 带名字的函数坐标

设 $C$ 是有限集合，并且每个 $v\in C$ 都有一个集合 $D_v$。定义：

$$
\prod_{v\in C}D_v
=
\{d\mid d\text{ 是定义在 }C\text{ 上的函数，且 }d(v)\in D_v\}.
$$

因此，$(d_v)_{v\in C}$ 表示“由 $v$ 标记的函数坐标族”，不是一个依赖排列顺序的列表。若 $C=\{a,b\}$，那么 $(d_v)_{v\in C}$ 就是同时规定 $d(a)$ 与 $d(b)$；交换纸面上书写 $a,b$ 的顺序不会改变这个函数。

当 $C=\varnothing$ 时，上式的积集合只含唯一的空函数；本文把这个空函数记为 $()$。

### 1.3 一个表示“不产生值”的符号

固定一个非空集合 $P$，其元素称为**承载值**。本文不限制 $P$ 是实数、向量还是别的集合。

另取一个不属于 $P$ 的符号 $\bot$，定义：

$$
P_\bot=P\cup\{\bot\}.
$$

以后，一个以 $P_\bot$ 为值域的函数在某个坐标上取 $\bot$，只表示“这次在该坐标上不产生承载值”。

## 2. 固定结构：时间、节点图、端口与区域划分

本节定义一次具体输入到来以前就已经固定的数据。

### 2.1 逻辑时间

定义逻辑时间集合：

$$
\mathbb T=\mathbb N.
$$

本文中的 $\theta\in\mathbb T$ 只是一项数学坐标。它可以相加、比较大小，并作为函数的自变量。定义本身没有说 $\theta=5$ 等于五秒，也没有说它是某个程序执行的第五步。

两个对象具有相同逻辑时间，严格地说只表示它们的时间坐标是同一个自然数。

### 2.2 有限有向图与 DAG

给定五元组：

$$
G=(V,A,\operatorname{src},\operatorname{dst},\delta).
\tag{1}
$$

其中：

- $V$ 是有限非空集合，其元素称为节点；
- $A$ 是有限集合，其元素称为有向边；
- $\operatorname{src}:A\to V$ 给出边的起点；
- $\operatorname{dst}:A\to V$ 给出边的终点；
- $\delta:A\to\mathbb N_{>0}$ 给出边的正整数时延。

若起点节点在逻辑时间 $\eta$ 产生一个准备沿边 $a$ 传递的值，则规定该值在逻辑时间 $\eta+\delta(a)$ 到达终点节点。因此，$\delta(a)$ 是这两个逻辑时间坐标之差；本模型不另设节点完成时间。附录 S.1 说明总时延与计算耗时、传递耗时的可选对应。

两条边即使具有相同起点和终点，只要它们是 $A$ 中不同的元素，仍然是两条不同的边。

对 $v\in V$ 定义：

$$
\operatorname{In}(v)
=\{a\in A\mid\operatorname{dst}(a)=v\},
$$

$$
\operatorname{Out}(v)
=\{a\in A\mid\operatorname{src}(a)=v\}.
$$

若 $(a_1,\ldots,a_k)$ 是非空有限边序列，并且：

$$
\operatorname{dst}(a_\ell)
=
\operatorname{src}(a_{\ell+1})
\qquad(1\le\ell<k),
$$

则称它为一条有向路径。若还满足：

$$
\operatorname{dst}(a_k)=\operatorname{src}(a_1),
$$

则称它为有向环。

本文规定式 (1) 中不存在有向环。满足这个条件的有限有向图称为有向无环图，英文缩写为 **DAG**。

有限 DAG 的任一有向路径都不会重复经过节点，否则两次经过之间便给出有向环；所以路径长度至多为 $|V|-1$。

本文还会用二元关系表示不需要独立边身份与边时延的辅助有向图。若 $W$ 是有限集合、$R\subseteq W\times W$，则 $(w_0,\ldots,w_k)$ 称为关系图 $(W,R)$ 的有向路径，当且仅当 $k\ge 1$ 且 $(w_{r-1},w_r)\in R$ 对每个 $1\le r\le k$ 成立；若再有 $w_0=w_k$，它就是一条非空有向闭路。不含这种闭路的关系图也称为 DAG。第 10 节的事件图与区域商图采用这个定义。

因此，本文标题中的 `DAG` 首先指式 (1) 的固定节点图是 DAG。第 10 节还会从一次具体计算构造另一张事件 DAG；两张图不是同一个数学对象。

### 2.3 多个输入端口与多个输出端口

给定两个有限非空集合：

$$
\mathsf I=\text{输入端口集合},
\qquad
\mathsf O=\text{输出端口集合},
$$

以及两个函数：

$$
\gamma:\mathsf I\to V,
\qquad
\varepsilon:\mathsf O\to V.
\tag{2}
$$

$\gamma(i)$ 是端口 $i$ 的值进入的节点；$\varepsilon(o)$ 是唯一可以向端口 $o$ 产生值的节点。端口只标记外部输入与输出的位置，不向图中增加节点。

对节点 $v$ 定义：

$$
\operatorname{InPort}(v)
=\{i\in\mathsf I\mid\gamma(i)=v\},
$$

$$
\operatorname{OutPort}(v)
=\{o\in\mathsf O\mid\varepsilon(o)=v\}.
$$

允许多个输入端口进入同一节点，也允许同一节点连接多个输出端口。

对每个 $i\in\mathsf I$，固定一个正整数 $L_i$。端口 $i$ 的输入位置集合是 $[L_i]$。一次完整外部输入是函数族：

$$
x=(x_i)_{i\in\mathsf I},
\qquad
x_i:[L_i]\to P.
\tag{3}
$$

再为每个端口固定严格递增函数：

$$
\iota_i:[L_i]\to\mathbb N,
\qquad
k<k'\Longrightarrow\iota_i(k)<\iota_i(k').
\tag{4}
$$

输入位置 $(i,k)$ 带有值 $x_i(k)$，并具有逻辑时间 $\iota_i(k)$。严格递增只发生在同一个端口内部；不同端口的位置可以具有相同逻辑时间。

### 2.4 区域是节点集合的划分

给定有限非空集合 $J$ 和满射：

$$
\rho:V\to J.
\tag{5}
$$

这里“满射”明确表示：

$$
\forall j\in J,\quad
\exists v\in V:\rho(v)=j.
$$

对每个 $j\in J$ 定义：

$$
\mathcal R_j=\rho^{-1}(\{j\})
=\{v\in V\mid\rho(v)=j\}.
\tag{6}
$$

由于 $\rho$ 是满射，每个 $\mathcal R_j$ 都非空；并且：

$$
V=\bigcup_{j\in J}\mathcal R_j,
\qquad
j\ne j'\Longrightarrow
\mathcal R_j\cap\mathcal R_{j'}=\varnothing.
$$

称每个集合 $\mathcal R_j$ 为一个**区域**。式 (5)--(6) 表明每个节点恰好属于一个区域。

区域是共同选择的节点范围，不是另加的消息收发节点。第 4 节为各节点定义状态，第 5.3 节再为各区域的选择定义历史；这是两类分别索引的状态坐标。$J$ 上目前没有定义任何边。第 10.6 节会为了比较额外构造一张区域商图，但那张商图不参与本模型的消息传递。

## 3. 外部输入记录、内部消息与时间纤维

本节先定义哪些记录可以到达一个节点，再用目标节点与逻辑时间从这些记录中取出一个有限子集。

### 3.1 外部输入记录

先取三个两两不同的形式标签 $\mathrm{ext},\mathrm{msg},\mathrm{out}$。它们只用于区分下述三类有序组。

定义所有可能的外部输入记录所成的集合：

$$
\mathsf{Ext}
=
\{(\mathrm{ext},i,k,y)\mid
i\in\mathsf I,\ k\in[L_i],\ y\in P\}.
$$

这里 $\mathrm{ext}$ 表示外部输入记录；下面的 $\mathrm{msg}$ 与 $\mathrm{out}$ 分别表示内部消息与输出记录。对 $e=(\mathrm{ext},i,k,y)\in\mathsf{Ext}$ 定义：

$$
\operatorname{inport}(e)=i,
\qquad
\operatorname{position}(e)=k,
$$

$$
\operatorname{target}(e)=\gamma(i),
\qquad
\operatorname{time}(e)=\iota_i(k),
\qquad
\operatorname{value}(e)=y.
$$

坐标 $i$ 与 $k$ 被保留。因此，即使两个端口给出相等的值，它们仍然是两个不同的记录。

### 3.2 内部消息与外部输出记录

定义所有可能的内部消息所成的集合：

$$
\mathsf{Msg}
=
\{(\mathrm{msg},\eta,a,y)\mid
\eta\in\mathbb N,\ a\in A,\ y\in P\}.
$$

对 $m=(\mathrm{msg},\eta,a,y)\in\mathsf{Msg}$ 定义：

$$
\operatorname{send}(m)=\eta,
\qquad
\operatorname{edge}(m)=a,
$$

$$
\operatorname{target}(m)=\operatorname{dst}(a),
\qquad
\operatorname{time}(m)=\eta+\delta(a),
\qquad
\operatorname{value}(m)=y.
\tag{7}
$$

$\eta$ 是源节点处理相应输入并产生该消息时所用的逻辑时间坐标，称为消息的**发送时间**；$\eta+\delta(a)$ 称为**到达时间**。两者都是自然数坐标，并不记录机器执行时刻。

定义所有可能的外部输出记录：

$$
\mathsf{OutRec}
=
\{(\mathrm{out},\theta,o,y)\mid
\theta\in\mathbb N,\ o\in\mathsf O,\ y\in P\}.
$$

对 $z=(\mathrm{out},\theta,o,y)\in\mathsf{OutRec}$ 定义：

$$
\operatorname{outtime}(z)=\theta,
\qquad
\operatorname{outport}(z)=o,
\qquad
\operatorname{outvalue}(z)=y.
$$

外部输出记录不会再次进入节点图，所以不为它定义 $\operatorname{target}$。

### 3.3 到达原子的集合

由于外部输入记录以 $\mathrm{ext}$ 开头，内部消息以 $\mathrm{msg}$ 开头，两个集合不相交。定义：

$$
\mathsf{Atom}=\mathsf{Ext}\cup\mathsf{Msg},
$$

以及节点 $v$ 所能接收的原子集合：

$$
\mathsf{Atom}_v
=
\{z\in\mathsf{Atom}\mid\operatorname{target}(z)=v\}.
$$

函数 $\operatorname{target}$、$\operatorname{time}$ 和 $\operatorname{value}$ 在 $\mathsf{Ext}$ 与 $\mathsf{Msg}$ 上已经分别定义，因此也在它们的不交并 $\mathsf{Atom}$ 上定义。

### 3.4 由节点与时间取出的纤维

任取有限集合 $E\subseteq\mathsf{Ext}$ 和 $M\subseteq\mathsf{Msg}$。对 $v\in V$ 与 $\theta\in\mathbb N$，定义：

$$
B_{v,\theta}(E,M)
=
\{z\in E\cup M
\mid
\operatorname{target}(z)=v,
\ \operatorname{time}(z)=\theta\}.
\tag{8}
$$

令

$$
\pi_{E,M}:E\cup M\to V\times\mathbb N,
\qquad
\pi_{E,M}(z)=(\operatorname{target}(z),\operatorname{time}(z)).
$$

则 $B_{v,\theta}(E,M)=\pi_{E,M}^{-1}(\{(v,\theta)\})$。因此称式 (8) 为节点 $v$ 在时间 $\theta$ 的**时间纤维**。这里取逆像的定义域是已经给定的 $E\cup M$。

时间纤维不是一个额外容器公理；它就是式 (8) 所定义的有限集合。由集合的定义可立即得到：

- 同一时间纤维中的元素没有先后次序；
- 一条边的身份 $a$ 和一个输入端口的身份 $i$ 没有被删除；
- 一个时间纤维可以包含零个、一个或多个原子；
- 两个原子数值相等，并不推出它们是同一个原子。

例如，若：

$$
e_0=(\mathrm{ext},i_0,0,y),
\qquad
e_1=(\mathrm{ext},i_1,0,y),
$$

其中 $i_0\ne i_1$，且 $\gamma(i_0)=\gamma(i_1)=v$、$\iota_{i_0}(0)=\iota_{i_1}(0)=5$，那么：

$$
B_{v,5}(\{e_0,e_1\},\varnothing)=\{e_0,e_1\}.
$$

这个集合有两个元素，即使两者承载的值都是 $y$。

本节尚未定义一次计算实际会产生哪些内部消息。因此，式 (8) 目前是任意 $E,M$ 上的集合构造。第 6 节通过递归定义实际消息集合；到那以后，才会定义一次输入对应的**完整时间纤维**。

## 4. 节点的局部函数

本节定义：在一个节点的旧状态与一个非空时间纤维已经给定时，该节点能算出哪些数学量。它尚不决定哪些节点被选择。

### 4.1 状态、本地内容与描述量

对每个 $v\in V$，给定：

- 非空状态集合 $S_v$；
- 初始状态 $q_v^{\mathrm{init}}\in S_v$；
- 非空本地内容集合 $X_v$；
- 非空选择描述量集合 $D_v$；
- 非空局部控制集合 $\mathsf C_v$。

局部控制量是区域选择交给一个候选节点的本次数据，例如一个介于 $0$ 与 $1$ 之间的权重。它不自动成为持久状态。$\mathsf C_v$ 是值的集合，与第 5 节由节点组成的候选集合 $C$ 不同。

给定两个全函数：

$$
\operatorname{Agg}_v:
\mathbb N\times\mathcal P_{\mathrm{fin}}(\mathsf{Atom}_v)
\to X_v,
$$

$$
\operatorname{Upd}_v:
S_v\times\mathbb N\times X_v
\to S_v.
\tag{9}
$$

任取旧状态 $q\in S_v$、时间 $\theta\in\mathbb N$ 和非空有限集合 $B\subseteq\mathsf{Atom}_v$，并假定 $B$ 中每个原子的时间都是 $\theta$。定义：

$$
h=\operatorname{Agg}_v(\theta,B),
\qquad
\widetilde q=\operatorname{Upd}_v(q,\theta,h).
\tag{10}
$$

$h$ 称为本地内容，$\widetilde q$ 称为候选新状态。这里“候选”表示 $\widetilde q$ 只是一个已经定义的函数值；它是否供本次完整计算读取，以及本次结束后保留什么状态，要到第 5 节分别决定。

$\operatorname{Agg}_v$ 的输入是原子集合本身。因此它可以读取外部记录中的端口 $i$，也可以读取内部消息中的边 $a$。若一个具体 $\operatorname{Agg}_v$ 只对所有 $\operatorname{value}(z)$ 求和，那是这个函数的特殊选择，不是式 (9) 强迫它忘记来源。

### 4.2 三种选择描述量

对每个节点给定三个全函数：

$$
\operatorname{Read}^{0}_v:
\mathbb N\times X_v\to D_v,
$$

$$
\operatorname{Read}^{-}_v:
S_v\times\mathbb N\times X_v\to D_v,
$$

$$
\operatorname{Read}^{+}_v:
S_v\times\mathbb N\times X_v\to D_v.
$$

对式 (10) 中的 $q,h,\widetilde q$，定义：

$$
d^0=\operatorname{Read}^{0}_v(\theta,h),
$$

$$
d^-=\operatorname{Read}^{-}_v(q,\theta,h),
$$

$$
d^+=\operatorname{Read}^{+}_v(\widetilde q,\theta,h).
\tag{11}
$$

三个上标的数学含义如下：

- $0$：不读取状态；
- $-$：读取处理当前非空时间纤维以前的状态 $q$；
- $+$：读取由本次输入算出的候选新状态 $\widetilde q$。

上标 $+$ 不表示 $\widetilde q$ 已经被采用。它只说明描述量函数以 $\widetilde q$ 为自变量。

### 4.3 节点的完整输出函数

对每个节点给定全函数：

$$
\operatorname{Full}_v:
S_v\times\mathbb N\times X_v\times\mathsf C_v
\to
(P_\bot)^{\operatorname{Out}(v)}
\times
(P_\bot)^{\operatorname{OutPort}(v)}.
\tag{12}
$$

任取 $q^{\mathrm{cmp}}\in S_v$、$\theta\in\mathbb N$、$h\in X_v$ 与 $c\in\mathsf C_v$。其中 $q^{\mathrm{cmp}}$ 表示供本次计算读取的状态，$c$ 是本节点的局部控制量；第 5 节再规定它们如何由选择结果确定。令：

$$
(f^A,f^O)=\operatorname{Full}_v(q^{\mathrm{cmp}},\theta,h,c),
$$

则：

- 对每条 $a\in\operatorname{Out}(v)$，$f^A(a)\in P_\bot$ 决定是否沿边 $a$ 产生一个值；
- 对每个 $o\in\operatorname{OutPort}(v)$，$f^O(o)\in P_\bot$ 决定是否向外部输出端口 $o$ 产生一个值。

不同出边和不同输出端口是函数的不同坐标。因此模型直接支持一个节点向多个内部去向和多个外部去向给出不同的值。

$\operatorname{Full}_v$ 的值只规定各出边与输出端口上的值坐标；第 6 节才用这些坐标构造消息与输出记录。其值域不含下一节点状态，也没有改写节点状态或选择历史的其他作用。

## 5. 区域候选集合、选择函数、选择历史与状态采用

本节在第 2.4 节的节点划分上定义共同选择。本次选择只读取同一逻辑时间的候选描述量，但选择历史会连接同一区域在不同时间的选择。

### 5.1 候选节点集合

先固定 $\theta\in\mathbb N$，并任取：

$$
B=(B_v)_{v\in V}
\in
\prod_{v\in V}\mathcal P_{\mathrm{fin}}(\mathsf{Atom}_v),
$$

其中每个 $B_v$ 的全部元素都具有时间 $\theta$。对区域 $j\in J$ 定义：

$$
C_j(B)
=
\{v\in\mathcal R_j\mid B_v\ne\varnothing\}.
\tag{13}
$$

这个集合的元素是节点，而不是原子。即使 $B_v$ 含有十个原子，节点 $v$ 也只在 $C_j(B)$ 中出现一次。

称 $C_j(B)$ 为区域 $j$ 的**候选集合**，其中的节点称为**候选节点**。第 6 节把 $B_v$ 取为一次完整计算在 $(v,\theta)$ 的纤维，并把所得候选集合记为 $\mathcal C_{j,\theta}$。

### 5.2 选择描述量模式

对每个 $j\in J$ 固定：

$$
\tau_j\in\{0,-,+\}.
$$

若 $v\in\mathcal R_j$，则该区域为节点 $v$ 使用式 (11) 中的 $d^{\tau_j}$。区域中的所有候选节点使用同一种模式，但各节点的读取函数可以不同。

### 5.3 带历史的选择函数

对每个 $j\in J$，给定非空集合 $Y_j$ 和元素：

$$
y_j^{\mathrm{init}}\in Y_j.
$$

$Y_j$ 称为区域 $j$ 的**选择历史状态集合**，$y_j^{\mathrm{init}}$ 是初始选择历史。

对每个 $j\in J$ 固定整数：

$$
1\le K_j\le|\mathcal R_j|.
$$

对每个子集 $C\subseteq\mathcal R_j$，定义允许的选择结果集合：

$$
\mathsf{Allowed}_{j,C}
=
\{A'\subseteq C\mid |A'|\le K_j\}.
$$

再给定全函数族：

$$
\operatorname{SelStep}_{j,C}:
Y_j\times\mathbb N\times\prod_{v\in C}D_v
\to\mathsf{Allowed}_{j,C}\times\prod_{v\in C}\mathsf C_v\times Y_j.
\tag{14}
$$

对空候选集合，规定：

$$
\operatorname{SelStep}_{j,\varnothing}(y,\theta,())
=(\varnothing,(),y).
$$

若某个时间的完整候选集合是 $C$，选择以前的历史是 $y\in Y_j$，候选描述量族是 $(d_v)_{v\in C}$，定义：

$$
(A,(c_v)_{v\in C},y')
=
\operatorname{SelStep}_{j,C}
\left(y,\theta,(d_v)_{v\in C}\right).
\tag{15}
$$

这里 $c_v\in\mathsf C_v$ 只交给节点 $v$，$y'$ 是本次选择以后的历史。三个输出属于同一个确定函数值，因此局部控制和 $y'$ 都可以由旧历史、$C$ 与全部当前描述量共同决定；$y'$ 也可记录 $A$ 或控制量的摘要。它们不能读取尚未应用的 $\operatorname{Full}_v$ 输出。非空 $C$ 中未被选中的节点也有 $c_v$，供下文的状态延续函数使用；$C=\varnothing$ 时只有唯一的空控制族。

把每个 $\mathsf C_v$ 取为单点集 $\{*\}$，选择函数就不向节点提供有效控制信息。第 7.1--7.7 节先用这个受限情形作手算。

式 (14) 称为区域的**选择函数**，式 (15) 是它的一次**选择作用**。集合 $A$ 称为**激活集合**，其中的节点称为**激活节点**。由值域定义自动得到：

$$
A\subseteq C,
\qquad
|A|\le K_j.
$$

式 (14) 没有规定必须按最大分数选择。最大值、Top-$K$、固定查表或任何别的确定性规则都可以成为一个具体的 $\operatorname{SelStep}_{j,C}$。若规则可能遇到相等描述量，必须把确定的平局规则写进函数；不得让观察顺序代替函数定义。

若 $Y_j$ 是单点集，就精确退化为不保存历史的选择函数。若取 $Y_j=\mathbb N^{\mathcal R_j}$，其元素可以记录区域内各节点过去被选择的次数；取实数坐标族还可以记录确定性移动平均。即使某些描述量或控制量取值于 $[0,1]$，它们在这里仍是确定的数值。随机选择需要另行给定概率模型，不属于当前规格。

### 5.4 本次计算状态与下一持久状态

对每个区域固定一个二值参数：

$$
\kappa_j\in\{0,1\}.
$$

给定候选集合 $C$ 与激活集合 $A\subseteq C$，定义状态采用集合：

$$
O_j(C,A)
=
\begin{cases}
A,&\kappa_j=0,\\
C,&\kappa_j=1.
\end{cases}
\tag{16}
$$

对候选节点 $v\in C$，旧状态为 $q_v$，本地内容为 $h_v$，候选新状态为 $\widetilde q_v$。先定义**本次计算状态**：

$$
q_v^{\mathrm{cmp}}
=
\begin{cases}
\widetilde q_v,&v\in O_j(C,A),\\
q_v,&v\notin O_j(C,A).
\end{cases}
\tag{17}
$$

称 $q_v^{\mathrm{cmp}}$ 为本次计算的**状态快照**；本次完整输出读取这个值。

再为每个节点给定全函数：

$$
\operatorname{Next}_v:
S_v\times S_v\times\mathbb N\times X_v\times\{0,1\}\times\mathsf C_v
\longrightarrow S_v.
$$

其中两个 $S_v$ 坐标依次是旧状态与本次计算状态。令 $a_v=\mathbf1[v\in A]$，并定义**下一持久状态**：

$$
q_v'
=\operatorname{Next}_v(q_v,q_v^{\mathrm{cmp}},\theta,h_v,a_v,c_v).
\tag{17a}
$$

只有实际候选节点应用这个函数；没有输入的节点保持原状态。$\operatorname{Next}_v$ 的输入不含完整输出，它的定义和求值也不得调用或重算 $\operatorname{Full}_v$。它可以依据当前控制量更新激活统计，或让本次计算使用已有内容后把下一状态清零；这都不要求等完整输出的值。

默认取

$$
\operatorname{Next}_v(q,q^{\mathrm{cmp}},\theta,h,a,c)=q^{\mathrm{cmp}}.
$$

此时式 (17) 也直接给出下一持久状态。连同单点控制集合，就得到不额外传递控制、也不另行改写下一状态的受限情形。若 $S_v$ 也是单点集，则没有有效节点状态。

因此需要分别回答三个问题：

1. 哪些节点把当前内容更新用于本次计算，由 $O_j(C,A)$ 决定；
2. 哪些节点计算完整输出，由 $A$ 决定；
3. 哪些状态留给未来，由声明的 $\operatorname{Next}_v$ 决定。

状态采用规则 (16)--(17) 与持久状态规则 (17a) 因而需要分别指定。若要让未采用候选新状态的节点仍改写某个历史坐标，应把这条规则明确写在 $\operatorname{Next}_v$ 中。特别地，在 $\kappa_j=0$ 时，未被选中的候选满足 $q^{\mathrm{cmp}}=q$；若要求其全部持久状态保持，还须有 $\operatorname{Next}_v(q,q,\theta,h,0,c)=q$。默认规则满足这个等式。只服务选择函数的历史则可放在 $Y_j$ 中，避免两处持有同一份可变状态。每份持久节点状态仍只由该节点持有，每份选择历史仍只由对应区域的选择函数持有；局部控制传值不建立共享可变状态。

### 5.5 同一时间内的数学依赖顺序

对一个非空候选集合，定义顺序为：

$$
\begin{aligned}
(q,B)&\longrightarrow(h,\widetilde q,d),\\
(y,d)&\longrightarrow(A,(c_v)_{v\in C},y'),\\
(A,q,\widetilde q,h,(c_v)_{v\in C})
&\longrightarrow(q^{\mathrm{cmp}},q'),\\
(q_v^{\mathrm{cmp}},\theta,h_v,c_v)&\longrightarrow\operatorname{Full}_v
\quad(v\in A).
\end{aligned}
\tag{18}
$$

时间 $\theta$ 在每一行都是已给自变量。式 (18) 是函数依赖，不是处理器指令或现实耗时。下一持久状态可以先于完整输出求出，因为它不读取完整输出；但本次完整输出始终读取 $q^{\mathrm{cmp}}$，不能因 $q'$ 已经清零便改读 $q'$。后续节点事件只读取已确定的 $q'$。本次完整输出也不参与本次选择历史更新。

## 6. 完整输入下的直接语义

前五节只给出了固定数据与局部函数。本节把它们组合成一次输入 $x$ 上的唯一完整计算。

### 6.1 一个有限逻辑时间上界

允许长度为零的路径停留在某个 $\gamma(i)$，并规定它的总时延为 $0$。对非空路径 $\zeta=(a_1,\ldots,a_k)$ 定义：

$$
\Delta(\zeta)=\sum_{\ell=1}^{k}\delta(a_\ell).
$$

因为有向路径长度至多为 $|V|-1$，而边集有限，所以从输入目标节点 $\gamma(i)$ 出发的路径总数有限。定义：

$$
\Delta_{\max}
=
\max\{\Delta(\zeta)\mid
\zeta\text{ 从某个 }\gamma(i)\text{ 出发，包括长度零路径}\}.
$$

再定义：

$$
\Theta_{\max}
=
\max_{i\in\mathsf I,\ k\in[L_i]}\iota_i(k)
+\Delta_{\max}.
\tag{19}
$$

第 6.5 节将证明，任何满足 $B_{v,\theta}\ne\varnothing$ 的坐标都满足 $\theta\le\Theta_{\max}$。

### 6.2 实际外部输入集合

由式 (3) 的具体输入 $x$ 定义：

$$
E_x
=
\{(\mathrm{ext},i,k,x_i(k))\mid
i\in\mathsf I,\ k\in[L_i]\}
\subseteq\mathsf{Ext}.
\tag{20}
$$

$E_x$ 包含每个输入位置恰好一条记录。

### 6.3 按逻辑时间递归

对每个节点，定义递归开始时的状态：

$$
q_v^{0}=q_v^{\mathrm{init}}.
$$

对每个区域，同时定义递归开始时的选择历史：

$$
y_j^0=y_j^{\mathrm{init}}.
$$

以后，$y_j^\theta$ 表示处理时间 $\theta$ 的区域选择以前的历史；上标是逻辑时间切面，不是已经发生的选择次数。

令 $M_{<0}=\varnothing$。依次对：

$$
\theta=0,1,\ldots,\Theta_{\max}
$$

作下列递归定义：给定更小时间已经产生的消息以及当前状态，由局部函数依次确定时间 $\theta$ 的全部量。正时延保证当前产生的消息只能影响更大时间；第 6.5 节据此证明递归的唯一性。

#### 第一步：定义完整时间纤维与候选集合

假设更小逻辑时间产生的消息集合 $M_{<\theta}$ 已经定义。令：

$$
B_{v,\theta}
=
B_{v,\theta}(E_x,M_{<\theta}).
\tag{21}
$$

这里左边的 $B_{v,\theta}$ 是本次输入的一个确定集合；右边是式 (8) 的集合构造。从现在起，左边称为本次计算在 $(v,\theta)$ 的**完整时间纤维**。

对每个 $j\in J$ 定义：

$$
\mathcal C_{j,\theta}
=
\{v\in\mathcal R_j\mid B_{v,\theta}\ne\varnothing\}.
\tag{22}
$$

#### 第二步：为每个候选节点定义本地量

若 $v\in\mathcal C_{j,\theta}$，令它的旧状态为 $q_v^{\theta}$，并定义：

$$
h_{v,\theta}
=\operatorname{Agg}_v(\theta,B_{v,\theta}),
$$

$$
\widetilde q_{v,\theta}
=\operatorname{Upd}_v(q_v^{\theta},\theta,h_{v,\theta}).
$$

再按 $\tau_j$ 取式 (11) 中相应的描述量：

$$
d_{v,\theta}
=
\begin{cases}
\operatorname{Read}^{0}_v(\theta,h_{v,\theta}),&\tau_j=0,\\
\operatorname{Read}^{-}_v(q_v^{\theta},\theta,h_{v,\theta}),&\tau_j=-,\\
\operatorname{Read}^{+}_v(\widetilde q_{v,\theta},\theta,h_{v,\theta}),&\tau_j=+.
\end{cases}
\tag{23}
$$

#### 第三步：每个区域选择一次

定义：

$$
(\mathcal A_{j,\theta},(c_{v,\theta})_{v\in\mathcal C_{j,\theta}},y_j^{\theta+1})
=
\operatorname{SelStep}_{j,\mathcal C_{j,\theta}}
\left(
y_j^\theta,
\theta,
(d_{v,\theta})_{v\in\mathcal C_{j,\theta}}
\right).
\tag{24}
$$

当 $\mathcal C_{j,\theta}=\varnothing$ 时，第 5.3 节的规定给出：

$$
\mathcal A_{j,\theta}=\varnothing,
\qquad
y_j^{\theta+1}=y_j^\theta.
$$

#### 第四步：定义本次计算状态与下一持久状态

对 $v\in\mathcal C_{j,\theta}$，先定义：

$$
q^{\mathrm{cmp}}_{v,\theta}
=
\begin{cases}
\widetilde q_{v,\theta},
&v\in O_j(\mathcal C_{j,\theta},\mathcal A_{j,\theta}),\\
q_v^\theta,&\text{其余候选节点}.
\end{cases}
\tag{25a}
$$

再对所有 $v\in\mathcal R_j$ 定义：

$$
q_v^{\theta+1}
=
\begin{cases}
\operatorname{Next}_v(q_v^\theta,q^{\mathrm{cmp}}_{v,\theta},
\theta,h_{v,\theta},\mathbf1[v\in\mathcal A_{j,\theta}],c_{v,\theta}),
&v\in\mathcal C_{j,\theta},\\
q_v^\theta,&v\notin\mathcal C_{j,\theta}.
\end{cases}
\tag{25}
$$

不属于候选集合的节点没有本次控制量、计算快照或状态延续作用；其持久状态保持不变。

#### 第五步：激活节点产生内部消息与外部输出

对每个 $v\in\mathcal A_{j,\theta}$ 定义：

$$
(f^A_{v,\theta},f^O_{v,\theta})
=
\operatorname{Full}_v
(q^{\mathrm{cmp}}_{v,\theta},\theta,h_{v,\theta},c_{v,\theta}).
\tag{26}
$$

定义时间 $\theta$ 产生的消息集合：

$$
\begin{aligned}
M_\theta
=\{&(\mathrm{msg},\theta,a,y)
\mid
j\in J,\ v\in\mathcal A_{j,\theta},\\
&a\in\operatorname{Out}(v),\\
&f^A_{v,\theta}(a)=y\in P\}.
\end{aligned}
\tag{27}
$$

定义时间 $\theta$ 产生的外部输出记录集合：

$$
\begin{aligned}
Z_\theta
=\{&(\mathrm{out},\theta,o,y)
\mid
j\in J,\ v\in\mathcal A_{j,\theta},\\
&o\in\operatorname{OutPort}(v),\\
&f^O_{v,\theta}(o)=y\in P\}.
\end{aligned}
\tag{28}
$$

最后令：

$$
M_{<\theta+1}=M_{<\theta}\cup M_\theta.
$$

这就同时完成节点状态与选择历史从时间 $\theta$ 到 $\theta+1$ 的递归。

注意，由式 (7) 和 $\delta(a)>0$，任意 $m\in M_\theta$ 都满足：

$$
\operatorname{time}(m)=\theta+\delta(\operatorname{edge}(m))>\theta.
\tag{29}
$$

所以时间 $\theta$ 才产生的消息不会反过来改变式 (21) 已经定义的当前时间纤维。

### 6.4 节点事件、区域选择事件与四类作用

本节中的“事件”不是概率论中作为样本空间子集的事件。这里先定义相应集合，再把属于这些集合的元素称为相应事件；“发生一次”只是对这种元素的文字称呼，不是额外的数学关系。

固定输入 $x$ 以及式 (21)--(29) 的递归结果，定义**节点事件集合**：

$$
\mathcal E_x^{\mathrm{node}}
=
\{(v,\theta)\in V\times[0,\Theta_{\max}+1)
\mid B_{v,\theta}\ne\varnothing\}.
$$

一个元素：

$$
e=(v,\theta)\in\mathcal E_x^{\mathrm{node}}
$$

称为输入 $x$ 的一个**节点事件**。它的节点坐标是 $v$，逻辑时间坐标是 $\theta$。本文直接用二元组 $(v,\theta)$ 表示这个事件，不再另外创造一个具有相同坐标的对象。

节点事件 $(v,\theta)$ 所处理的输入是完整时间纤维 $B_{v,\theta}$。二者不是同一个对象：前者是 $V\times\mathbb N$ 中的一个二元组，后者是 $\mathcal P_{\mathrm{fin}}(\mathsf{Atom}_v)$ 中的一个有限集合。因此：

- 一个时间纤维含有多个原子时，仍然只对应一个节点事件；
- $B_{v,\theta}=\varnothing$ 时，$(v,\theta)$ 不是节点事件；
- $(v,\theta)$ 与 $(v',\theta)$ 在 $v\ne v'$ 时是两个不同的节点事件；
- 即使 $v\notin\mathcal A_{\rho(v),\theta}$，只要 $B_{v,\theta}\ne\varnothing$，$(v,\theta)$ 仍然是节点事件。

节点事件的逻辑时间坐标没有定义机器执行时刻或持续长度。$q_v^\theta$ 是处理这个节点事件以前的状态，$q^{\mathrm{cmp}}_{v,\theta}$ 是本次计算快照，$q_v^{\theta+1}$ 是按式 (25) 留给后续节点事件的持久状态。

再定义**区域选择事件集合**：

$$
\mathcal E_x^{\mathrm{sel}}
=
\{(j,\theta)\in J\times[0,\Theta_{\max}+1)
\mid\mathcal C_{j,\theta}\ne\varnothing\}.
$$

其中的元素 $(j,\theta)$ 称为输入 $x$ 的一个**区域选择事件**。它表示把区域 $j$ 在时间 $\theta$ 的旧选择历史与整个候选描述量族一次交给式 (24)，并共同确定激活集合、局部控制族与下一选择历史。当候选集合为空时，式 (24) 仍规定激活集合为空、选择历史不变，但本文不把这个恒等位置收入 $\mathcal E_x^{\mathrm{sel}}$。

由式 (22) 立即得到：

$$
\mathcal C_{j,\theta}
=
\{v\in\mathcal R_j\mid
(v,\theta)\in\mathcal E_x^{\mathrm{node}}\}.
$$

所以同一区域、同一逻辑时间的一个或多个节点事件共同对应一个区域选择事件。区域选择事件是否存在由 $\mathcal C_{j,\theta}$ 是否非空决定，不由 $\mathcal A_{j,\theta}$ 是否非空决定。

每个节点事件包含两个必有的函数作用。若节点被激活，还包含一个完整输出作用。每个区域选择事件包含一个选择作用。取四个两两不同的标签

$$
\mathsf{ActKind}
=
\{\mathrm{prep},\mathrm{select},\mathrm{adopt},\mathrm{full}\}.
$$

对 $(v,\theta)\in\mathcal E_x^{\mathrm{node}}$ 定义：

$$
P_{v,\theta}=(\mathrm{prep},v,\theta),
\qquad
U_{v,\theta}=(\mathrm{adopt},v,\theta).
$$

$P_{v,\theta}$ 取完整时间纤维和旧持久状态，确定 $h_{v,\theta},\widetilde q_{v,\theta},d_{v,\theta}$；$U_{v,\theta}$ 取本次选择结果，确定 $q^{\mathrm{cmp}}_{v,\theta}$ 与 $q_v^{\theta+1}$。对 $(j,\theta)\in\mathcal E_x^{\mathrm{sel}}$ 定义：

$$
S_{j,\theta}=(\mathrm{select},j,\theta).
$$

它确定 $\mathcal A_{j,\theta}$、完整局部控制族以及 $y_j^{\theta+1}$。对每个 $v\in\mathcal A_{j,\theta}$ 定义：

$$
F_{v,\theta}=(\mathrm{full},v,\theta),
$$

它确定式 (26) 的完整输出值。相应的四个有限集合记为：

$$
\begin{aligned}
\mathscr V_x^P
&=\{P_{v,\theta}\mid(v,\theta)\in\mathcal E_x^{\mathrm{node}}\},\\
\mathscr V_x^S
&=\{S_{j,\theta}\mid(j,\theta)\in\mathcal E_x^{\mathrm{sel}}\},\\
\mathscr V_x^U
&=\{U_{v,\theta}\mid(v,\theta)\in\mathcal E_x^{\mathrm{node}}\},\\
\mathscr V_x^F
&=\{F_{v,\theta}\mid v\in\mathcal A_{\rho(v),\theta}\}.
\end{aligned}
$$

令：

$$
\mathscr V_x^{\mathrm{ev}}
=
\mathscr V_x^P\sqcup\mathscr V_x^S
\sqcup\mathscr V_x^U\sqcup\mathscr V_x^F.
$$

四类作用的首标签互异，所以这里的 $\sqcup$ 符合第 1.1 节的不交性要求。

式 (22)--(26) 已经给每个作用坐标唯一赋值。把下列带标签值族定义为**规范作用记录** $\mathcal T_x^{\mathrm{act}}$：

$$
\begin{aligned}
\mathcal T_x^{\mathrm{act}}(P_{v,\theta})
&=(h_{v,\theta},\widetilde q_{v,\theta},d_{v,\theta}),\\
\mathcal T_x^{\mathrm{act}}(S_{j,\theta})
&=(\mathcal A_{j,\theta},
(c_{v,\theta})_{v\in\mathcal C_{j,\theta}},
y_j^{\theta+1}),\\
\mathcal T_x^{\mathrm{act}}(U_{v,\theta})
&=(q^{\mathrm{cmp}}_{v,\theta},q_v^{\theta+1}),\\
\mathcal T_x^{\mathrm{act}}(F_{v,\theta})
&=(f^A_{v,\theta},f^O_{v,\theta}).
\end{aligned}
$$

对有限集合 $K\subseteq\mathscr V_x^{\mathrm{ev}}$，令 $\mathsf{SemRec}(K)$ 为在 $K$ 的每个作用坐标上赋予类型正确函数值的全部记录所成的集合，$\mathcal T_x^{\mathrm{act}}\!\upharpoonright_K$ 表示规范作用记录在这些坐标上的限制。一个 $K$ 上的**联合记录规格**由集合 $\mathsf{JointRec}(K)$ 和一个预先固定的投影组成：

$$
\Pi_K:\mathsf{JointRec}(K)\longrightarrow\mathsf{SemRec}(K).
$$

$\mathsf{JointRec}(K)$ 可以另外携带辅助坐标；$\Pi_K$ 保留 $K$ 的全部规范坐标及其原值，只删除辅助坐标。它不调用参考递归，也不按输入修补函数值。称联合记录 $r\in\mathsf{JointRec}(K)$ **精化到**本次规范记录，当且仅当：

$$
\Pi_K(r)=\mathcal T_x^{\mathrm{act}}\!\upharpoonright_K.
$$

四类作用属于规范记录本身，并描述函数值之间的数学依赖。一次联合求值可以同时实现许多个作用，只要它的联合记录按上述定义精化到这些作用的全部规范坐标。第 8--10 节将依次定义作用之间的依赖、部分记录的封闭证书以及合法的阶段化暴露。

### 6.5 有限性与唯一性定理

定义本次计算产生的全部内部消息与外部输出记录：

$$
M^*=\bigcup_{\theta=0}^{\Theta_{\max}}M_\theta,
\qquad
Z^*=\bigcup_{\theta=0}^{\Theta_{\max}}Z_\theta.
\tag{30}
$$

> [!theorem] 定理 1：完整计算存在、有限且唯一
> 固定第 1--5 节的全部集合、函数与参数，并给定式 (3) 的输入 $x$。则第 6.3 节唯一确定：
>
> - 每个完整时间纤维 $B_{v,\theta}$；
> - 每个候选集合 $\mathcal C_{j,\theta}$；
> - 每个激活集合 $\mathcal A_{j,\theta}$；
> - 每个候选节点的 $h_{v,\theta},\widetilde q_{v,\theta},d_{v,\theta},c_{v,\theta},q^{\mathrm{cmp}}_{v,\theta}$；
> - 每个区域的选择历史序列；
> - 节点事件集合 $\mathcal E_x^{\mathrm{node}}$ 与区域选择事件集合 $\mathcal E_x^{\mathrm{sel}}$；
> - 每个节点的状态序列；
> - 有限内部消息集合 $M^*$；
> - 有限多端口外部输出集合 $Z^*$。

**证明。** 对 $\theta$ 作归纳。

在 $\theta=0$ 时，$M_{<0}=\varnothing$，所以式 (21) 只由已给的 $E_x$ 唯一确定。第 4--5 节给定的 $\operatorname{Agg}$、$\operatorname{Upd}$、三个 $\operatorname{Read}$、$\operatorname{SelStep}$、$\operatorname{Next}$ 与 $\operatorname{Full}$ 都是全函数；已知 $y_j^0$ 后，式 (24) 还同时唯一确定 $\mathcal A_{j,0}$、$(c_{v,0})_{v\in\mathcal C_{j,0}}$ 与 $y_j^1$。因而按式 (22)--(28) 依次求值会给出唯一结果。

假设所有小于 $\theta$ 的结果已经唯一确定，则 $M_{<\theta}$ 唯一。式 (21) 因而唯一；随后所有步骤仍是已给函数的求值，所以时间 $\theta$ 的结果唯一。自然数归纳法给出整个递归的唯一性。

还需证明递归不会产生到达时间超出式 (19) 的消息。任取节点事件 $(v,\theta)$，它所对应的 $B_{v,\theta}$ 至少含有一个到达原子 $z$。若 $z$ 是来自输入位置 $(i,k)$ 的外部输入记录，则 $v=\gamma(i)$ 且 $\theta=\iota_i(k)$。若 $z$ 是内部消息，则它来自某条终点为 $v$ 的边；沿这条消息回到产生它的节点事件，并继续向前追溯。每追溯一条边，节点在固定图中向上游移动一次。由于固定节点图无环，经过有限条边后必定到达某个外部输入记录。因此，每个节点事件的逻辑时间都可以写成：

$$
\iota_i(k)+\Delta(\zeta),
$$

其中 $\zeta$ 是从 $\gamma(i)$ 出发的一条有向路径。它不大于 $\Theta_{\max}$。若这个节点事件沿出边 $a$ 产生消息，就把 $a$ 接到 $\zeta$ 后面；所得仍是从 $\gamma(i)$ 出发的路径，所以这条新消息的到达时间也不大于 $\Theta_{\max}$。固定时间上界、有限节点、有限区域、有限边和有限端口共同推出两类事件集合、两类状态坐标族、消息集合和输出记录集合均有限。$\square$

最终节点状态和最终选择历史分别是：

$$
(q_v^{\Theta_{\max}+1})_{v\in V},
\qquad
(y_j^{\Theta_{\max}+1})_{j\in J}.
$$

把完整计算记录明确定义为：

$$
\mathcal T_x
=
\left(
\begin{aligned}
&(B_{v,\theta})_{\substack{v\in V\\\theta\in[0,\Theta_{\max}+1)}},\\
&\bigl(h_{v,\theta},\widetilde q_{v,\theta},d_{v,\theta},c_{v,\theta},
q^{\mathrm{cmp}}_{v,\theta}\bigr)_{
\substack{j\in J,\ \theta\in[0,\Theta_{\max}+1)\\
v\in\mathcal C_{j,\theta}}},\\
&(\mathcal C_{j,\theta},\mathcal A_{j,\theta})_{
\substack{j\in J\\\theta\in[0,\Theta_{\max}+1)}},\\
&(q_v^\theta)_{
\substack{v\in V\\\theta\in[0,\Theta_{\max}+2)}},\\
&(y_j^\theta)_{
\substack{j\in J\\\theta\in[0,\Theta_{\max}+2)}},\\
&M^*,Z^*,\\
&(f^A_{v,\theta},f^O_{v,\theta})_{
\substack{j\in J,\ \theta\in[0,\Theta_{\max}+1)\\
v\in\mathcal A_{j,\theta}}}
\end{aligned}
\right),
$$

其中时间纤维、本地量、选择和逐坐标输出函数值的时间坐标满足 $0\le\theta\le\Theta_{\max}$，两类状态还包含递归结束后的 $\theta=\Theta_{\max}+1$。称 $\mathcal T_x$ 为输入 $x$ 的完整计算记录。

第 6.4 节的 $\mathcal T_x^{\mathrm{act}}$ 由这个元组中的本地量、选择、状态采用与完整输出坐标唯一导出；以后也把它视为 $\mathcal T_x$ 的规范作用投影。因此，对任意 $K\subseteq\mathscr V_x^{\mathrm{ev}}$，记号 $\mathcal T_x\!\upharpoonright_K$ 与 $\mathcal T_x^{\mathrm{act}}\!\upharpoonright_K$ 表示同一带标签作用记录。

本文把第 1--5 节的数据以及第 6 节所定义的直接语义合称为一个**带区域选择的 TimedDAG 规格**。

### 6.6 多输入、多输出

多输入直接出现在式 (2)--(4) 与式 (20)--(21) 中。一个完整时间纤维可以同时包含：

- 不同端口产生的记录；
- 不同入边产生的消息；
- 外部输入记录与内部消息。

只要这些原子的 $\operatorname{target}$ 与 $\operatorname{time}$ 相同，它们就在式 (21) 的同一个集合中。由式 (4) 的严格递增性，同一输入端口在同一个逻辑时间至多贡献一个位置；不同端口不受这个限制。

多输出直接出现在式 (12) 与式 (28) 中。对每个 $o\in\mathsf O$，可以从 $Z^*$ 取出：

$$
Z_o^*
=
\{(\operatorname{outtime}(z),\operatorname{outvalue}(z))
\mid z\in Z^*,\ \operatorname{outport}(z)=o\}.
$$

整个外部结果是由端口标记的函数族 $(Z_o^*)_{o\in\mathsf O}$。单输入或单输出只是在 $\mathsf I$ 或 $\mathsf O$ 恰有一个元素时得到的特例。

对固定的 $(\theta,o)$，$Z^*$ 中至多有一个值 $y$：端口 $o$ 由式 (2) 指向唯一节点，而该节点的式 (12) 在坐标 $o$ 上只有一个函数值。因此每个 $Z_o^*$ 都是某个有限自然数子集到 $P$ 的函数图。

## 7. 完整手算与局部变式

第 7.1--7.5 节给出一次完整计算，第 7.6--7.9 节再分别考察描述量、选择历史、局部控制与状态延续。所有例子都只使用前面已经定义的对象。

### 7.1 固定集合与图

取：

$$
P=\mathbb N,
\qquad
V=\{s_0,s_1,a,b\},
\qquad
A=\{e_0,e_1\}.
$$

两条边定义为：

$$
\operatorname{src}(e_0)=s_0,
\quad
\operatorname{dst}(e_0)=a,
\quad
\delta(e_0)=5,
$$

$$
\operatorname{src}(e_1)=s_1,
\quad
\operatorname{dst}(e_1)=b,
\quad
\delta(e_1)=4.
$$

这张图只有 $s_0\to a$ 与 $s_1\to b$，所以没有有向环。

取两个输入端口和两个输出端口：

$$
\mathsf I=\{i_0,i_1\},
\qquad
\mathsf O=\{o_a,o_b\},
$$

$$
\gamma(i_0)=s_0,
\quad
\gamma(i_1)=s_1,
\quad
\varepsilon(o_a)=a,
\quad
\varepsilon(o_b)=b.
$$

两个输入端口都只有一个位置：

$$
L_{i_0}=L_{i_1}=1.
$$

具体输入与时间为：

$$
x_{i_0}(0)=2,
\quad
\iota_{i_0}(0)=0,
$$

$$
x_{i_1}(0)=7,
\quad
\iota_{i_1}(0)=1.
$$

取 $J=\{0,1,2\}$，并定义：

$$
\rho(s_0)=0,
\qquad
\rho(s_1)=1,
\qquad
\rho(a)=\rho(b)=2.
$$

于是 $\rho$ 是满射，区域划分为：

$$
\begin{aligned}
\rho^{-1}(\{0\})&=\mathcal R_0=\{s_0\},\\
\rho^{-1}(\{1\})&=\mathcal R_1=\{s_1\},\\
\rho^{-1}(\{2\})&=\mathcal R_2=\{a,b\}.
\end{aligned}
$$

从输入目标节点出发的最长路径是 $s_0\to a$，总时延为 $5$。因此：

$$
\Delta_{\max}=5,
\qquad
\Theta_{\max}=\max\{0,1\}+5=6.
$$

### 7.2 局部函数与选择函数

为简化状态，令每个节点的状态集合都是单点集 $\{*\}$，初态为 $*$，更新函数总返回 $*$。本节第 7.1--7.7 节都取 $\mathsf C_v=\{*\}$，每个候选的控制量为 $*$，并取默认的 $\operatorname{Next}_v(q,q^{\mathrm{cmp}},\theta,h,a,*)=q^{\mathrm{cmp}}$；完整输出忽略最后的单点控制输入。

令每个 $X_v=D_v=\mathbb N$，并定义：

$$
\operatorname{Agg}_v(\theta,B)
=
\sum_{z\in B}\operatorname{value}(z),
$$

$$
\operatorname{Read}^{0}_v(\theta,h)
=
\operatorname{Read}^{-}_v(*,\theta,h)
=
\operatorname{Read}^{+}_v(*,\theta,h)
=h.
$$

各区域都取 $\tau_j=0$ 和 $\kappa_j=0$，并取 $K_0=K_1=K_2=1$。单节点区域 $\mathcal R_0,\mathcal R_1$ 的选择函数在候选非空时选择唯一节点。

本例先令每个 $Y_j=\{*\}$、$y_j^{\mathrm{init}}=*$；每个选择作用的第三个输出仍为 $*$。因此，本例的选择历史不携带额外信息。

区域 $\mathcal R_2$ 取 $K_2=1$，其选择函数选择描述量较大的节点；若相等，固定选择 $a$。这条平局规则使它成为一个确定函数。

对任意函数自变量，$s_0$ 的完整输出函数在边 $e_0$ 上取值 $h$，$s_1$ 的函数在 $e_1$ 上取值 $h$；$a,b$ 的函数分别在端口 $o_a,o_b$ 上取值 $h$。各节点的其他坐标若存在，均取 $\bot$。这就给出了式 (12) 的全函数；实际计算中只有被选节点应用它。

### 7.3 时间 0 与时间 1

实际外部输入集合为：

$$
E_x=
\{(\mathrm{ext},i_0,0,2),
(\mathrm{ext},i_1,0,7)\}.
$$

时间 $0$：

$$
B_{s_0,0}=\{(\mathrm{ext},i_0,0,2)\}.
$$

所以 $s_0$ 是区域 $0$ 的唯一候选并被选择，产生：

$$
m_0=(\mathrm{msg},0,e_0,2),
\qquad
\operatorname{time}(m_0)=0+5=5.
$$

时间 $1$：

$$
B_{s_1,1}=\{(\mathrm{ext},i_1,0,7)\}.
$$

所以 $s_1$ 被选择，产生：

$$
m_1=(\mathrm{msg},1,e_1,7),
\qquad
\operatorname{time}(m_1)=1+4=5.
$$

这里 $m_0,m_1$ 的发送时间不同，但逻辑到达时间相同。

### 7.4 时间 5 的共同选择

由式 (21)：

$$
B_{a,5}=\{m_0\},
\qquad
B_{b,5}=\{m_1\}.
$$

所以区域 $2$ 的完整候选集合是：

$$
\mathcal C_{2,5}=\{a,b\}.
$$

本地内容和描述量为：

$$
h_{a,5}=d_{a,5}=2,
\qquad
h_{b,5}=d_{b,5}=7.
$$

因此：

$$
\mathcal A_{2,5}=\{b\}.
$$

只有 $b$ 应用完整输出函数，于是：

$$
Z_5=\{(\mathrm{out},5,o_b,7)\}.
$$

$o_a$ 仍然是模型定义中的合法输出端口；本次输入只是在该端口没有产生记录。区域 $\mathcal R_2$ 没有发送任何东西，是节点 $b$ 向自己的输出端口产生了值。

至此，式 (30) 与最终状态分别给出：

$$
M^*=\{m_0,m_1\},
\qquad
Z^*=\{(\mathrm{out},5,o_b,7)\},
$$

$$
q_v^7=*\quad(v\in V),
\qquad
y_j^7=*\quad(j\in J).
$$

在 $v\in V$、$\theta\in[0,7)$ 的范围内，非空完整时间纤维恰好是本节已经列出的 $B_{s_0,0},B_{s_1,1},B_{a,5},B_{b,5}$；其余 $B_{v,\theta}$ 全为空。相应地，除 $M_0=\{m_0\},M_1=\{m_1\}$ 与上述 $Z_5$ 外，其余 $M_\theta,Z_\theta$ 都为空。

### 7.5 这个例子已经说明什么

这个例子只使用式 (1)--(30)，并说明：

1. 两个外部输入端口可以有不同输入位置和不同注入时间；
2. 两条路径的总时延可以使它们在同一逻辑时间到达不同节点；
3. 同一区域的这些节点共同形成一个完整候选集合；
4. 选择函数的输入由节点名字标记，不带消息观察次序；
5. 模型有多个输出端口，即使一次具体计算只在其中一部分端口产生记录。

若再增加两个都指向节点 $a$ 的输入端口，并让它们在时间 $5$ 注入，那么两个外部记录会与 $m_0$ 一同出现在 $B_{a,5}$ 中。节点 $a$ 仍只作为一个候选出现，但 $\operatorname{Agg}_a$ 会接收整个三元素集合。

### 7.6 描述量模式与状态采用模式的四种组合

本例只比较一次区域选择中的状态，不重建整个输入与消息图。把第 7.1--7.5 节的单点状态改为下列数值状态，并假定已经得到完整候选集合和本地内容：

$$
\mathcal C_{j,5}=\{a,b\},
\qquad
K_j=1,
$$

旧状态与本地内容分别为：

$$
q_a^5=0,
\quad
q_b^5=6,
\qquad
h_{a,5}=10,
\quad
h_{b,5}=1.
$$

取 $S_a=S_b=X_a=X_b=D_a=D_b=\mathbb N$，并定义：

$$
\operatorname{Upd}_v(q,\theta,h)=q+h,
$$

$$
\operatorname{Read}^{-}_v(q,\theta,h)=q,
\qquad
\operatorname{Read}^{+}_v(\widetilde q,\theta,h)=\widetilde q.
$$

于是候选新状态是：

$$
\widetilde q_{a,5}=10,
\qquad
\widetilde q_{b,5}=7.
$$

本例取 $Y_j=\{*\}$，使选择历史不影响下面的比较。

令选择函数选择描述量较大的一个节点，平局时选择 $a$。若 $\tau_j=-$，它读取旧状态 $(0,6)$，所以：

$$
\mathcal A_{j,5}=\{b\}.
$$

若 $\tau_j=+$，它读取候选新状态 $(10,7)$，所以：

$$
\mathcal A_{j,5}=\{a\}.
$$

再分别代入式 (16)--(17a)，由于本例 $\operatorname{Next}$ 返回计算快照，得到四种状态与选择结果：

| $(\tau_j,\kappa_j)$ | 选择函数读取 | $\mathcal A_{j,5}$ | 状态采用集合 | $(q_a^6,q_b^6)$ | 应用 $\operatorname{Full}$ 的节点 |
| --- | --- | --- | --- | --- | --- |
| $(-,0)$ | $(0,6)$ | $\{b\}$ | $\{b\}$ | $(0,7)$ | $b$ |
| $(-,1)$ | $(0,6)$ | $\{b\}$ | $\{a,b\}$ | $(10,7)$ | $b$ |
| $(+,0)$ | $(10,7)$ | $\{a\}$ | $\{a\}$ | $(10,6)$ | $a$ |
| $(+,1)$ | $(10,7)$ | $\{a\}$ | $\{a,b\}$ | $(10,7)$ | $a$ |

所以 $\tau_j$ 与 $\kappa_j$ 是两项独立数学参数：前者可能改变激活集合，后者可能改变未被选节点的未来状态。附录 S 中的 `pre/post` 与 `SD/BO` 只是这四种数学组合的系统名称。

### 7.7 一个会影响未来选择的历史

只考察区域 $\mathcal R_j=\{a,b\}$，取：

$$
Y_j=\mathbb N^{\{a,b\}},
\qquad
y_j^0(a)=y_j^0(b)=0,
\qquad K_j=1.
$$

定义选择作用在候选非空时选择历史计数最小的候选，计数相等时固定令 $a$ 排在 $b$ 前；随后只把被选节点的计数加一。假设时间 $0$ 与 $2$ 的候选集合都是 $\{a,b\}$，时间 $1$ 的候选集合为空。于是：

$$
\mathcal A_{j,0}=\{a\},
\qquad
(y_j^1(a),y_j^1(b))=(1,0),
$$

$$
y_j^2=y_j^1,
$$

$$
\mathcal A_{j,2}=\{b\},
\qquad
(y_j^3(a),y_j^3(b))=(1,1).
$$

时间 $2$ 的描述量即使与时间 $0$ 相同，旧历史不同也会改变激活集合。这里的 $y_j^\theta$ 是递归状态，不是把全部过去事件原样保存的一份日志。

### 7.8 局部控制如何改变实际发送的值

固定正整数 $n$。给定第 2 节的固定结构，并按第 5.3 节为各区域取容量。令 $P=X_v=S_v=\mathbb R^n$、$D_v=\mathbb R$、$\mathsf C_v=[0,1]$，初始节点状态为零向量，各 $Y_j=\{*\}$、$y_j^{\mathrm{init}}=*$。令聚合为输入值的向量和，空集的和为零向量。用 $h_1,q_1$ 表示向量的第一坐标，并取：

$$
\begin{aligned}
\operatorname{Upd}_v(q,\theta,h)&=q+h,\\
\operatorname{Read}^0_v(\theta,h)&=h_1,\\
\operatorname{Read}^{-}_v(q,\theta,h)
&=\operatorname{Read}^{+}_v(q,\theta,h)=q_1.
\end{aligned}
$$

本例取 $\tau_j=0$，$\kappa_j$ 可以为 $0$ 或 $1$，$\operatorname{Next}$ 默认返回本次计算状态。先在每个区域内固定一个节点全序作为平局规则。对非空 $C$，按描述量从大到小选出 $\min(K_j,|C|)$ 个节点，平局按该全序决定，得到 $A$；同时令：

$$
w_v=1+|d_v|,
\qquad
p_v=\frac{w_v}{\sum_{u\in C}w_u},
\qquad c_v=p_v.
$$

分母严格为正，所以每个 $p_v\in[0,1]$ 都已定义，且 $\sum_{v\in C}p_v=1$。选择函数返回 $(A,(p_v)_{v\in C},*)$。这里 $p_v$ 是确定的数值权重，不是随机抽样。若希望用指数归一化等别的公式，只需将它明确写进选择函数。

再给定全函数 $G_v:S_v\times\mathbb N\times X_v\to\mathbb R^n$，作为节点内部计算，例如 $G_v(q^{\mathrm{cmp}},\theta,h)=h+q^{\mathrm{cmp}}$。对激活节点，记 $g=G_v(q^{\mathrm{cmp}},\theta,h)$，可选择以下两种完整输出：

| 发送方式 | 每个出边与输出端口的值 $\widehat g$ |
| --- | --- |
| HARD | $g$ |
| SOFTP | $h+p_v(g-h)$ |

以上公式对每个 $q^{\mathrm{cmp}},\theta,h,p_v$ 都有定义，因而给出了式 (12) 的全函数。未被选中的节点不应用这个函数，也不因此发送 $h$。同一个局部控制量还可供 $\operatorname{Next}$ 使用，但本例没有把它保存到状态中。

例如，两个候选的描述量分别为 $3,1$，容量为 $1$，则第一节点被选中，控制值为 $4/(4+2)=2/3$。在标量情形 $h=3,q^{\mathrm{cmp}}=6,g=9$ 下，HARD 发送 $9$，SOFTP 发送 $3+(2/3)(9-3)=7$。另一候选会影响第一节点的 $p_v$，因此这个值必须经选择函数显式传给第一节点，不能假装由它的本地 $h$ 单独决定。

### 7.9 按逻辑时间衰减，使用以后清空

先解释一种带时间戳的状态。取 $S_v=\mathbb R\times\mathbb N$，一个状态写作 $(z,\ell)$：$z$ 是在逻辑时间 $\ell$ 记录的数值。固定 $0<\alpha\le1$，并对所有状态与时间定义有效值：

$$
\operatorname{Eff}((z,\ell),\theta)
=\alpha^{\max\{\theta-\ell,0\}}z.
$$

最大值只用于让函数在整个 $S_v\times\mathbb N$ 上有定义；下面从 $(0,0)$ 出发的实际读取都满足 $\ell\le\theta$。指数中的间隔是同一个逻辑时间轴上的 $\theta-\ell$，不是输入序列位置的差，也不是机器等待时间。

令 $P=X_v=D_v=\mathbb R$，输入聚合为值的和，$\mathsf C_v=Y_j=\{*\}$。每个节点初态为 $(0,0)$，每个区域初始选择历史为 $*$。给定固定图与容量，取 $\tau_j=+$、$\kappa_j=1$，并定义：

$$
\operatorname{Upd}_v(q,\theta,h)
=(\operatorname{Eff}(q,\theta)+h,\theta),
$$

$$
\begin{aligned}
\operatorname{Read}^0_v(\theta,h)&=h,\\
\operatorname{Read}^{-}_v(q,\theta,h)
&=\operatorname{Eff}(q,\theta),\\
\operatorname{Read}^{+}_v(q,\theta,h)
&=\operatorname{Eff}(q,\theta).
\end{aligned}
$$

选择函数按描述量选出 $\min(K_j,|C|)$ 个节点，平局按事先固定的节点全序，局部控制恒为 $*$、选择历史保持 $*$。由于 $\kappa_j=1$，所有候选都用当前输入产生的状态作为本次计算快照。令完整输出把 $\operatorname{Eff}(q^{\mathrm{cmp}},\theta)$ 送到每个出边与输出端口；再定义：

$$
\operatorname{Next}_v(q,q^{\mathrm{cmp}},\theta,h,a,*)
=\begin{cases}
(0,\theta),&a=1,\\
q^{\mathrm{cmp}},&a=0.
\end{cases}
$$

于是未被选中的候选继续积累内容，被选中的节点用本次快照产生完整输出，并把零值留给未来。这是“使用后清空”的精确含义：清空只依赖 $a$，不依赖完整输出算出了什么，也不抹掉本次快照。

取 $\alpha=1/2$。设某节点在时间 $2$ 未被选中，采用候选新状态后留下 $(8,2)$，时间 $3,4$ 没有输入，则持久状态表示保持：

$$
q_v^3=q_v^4=q_v^5=(8,2).
$$

到时间 $5$ 收到本地内容 $h=1$ 时，有效旧值是 $(1/2)^3\cdot8=1$，候选新状态为 $(2,5)$。若该节点本次被选中，则：

$$
q^{\mathrm{cmp}}_{v,5}=(2,5),
\qquad
\widehat g=2,
\qquad
q_v^6=(0,5).
$$

这里存储表示在空档没有逐时写入，按时间读取的有效值却已经衰减；把这两个对象混为一谈，就会误以为衰减违反“空纤维保持状态”。这种表示通常称为惰性表示。若把一个切面的状态改写成“该切面的有效值与当前时间”，虽然未来有效值可能相同，得到的表示也已不同；要把它当作另一种实现编码，仍应给出表示之间的映射，而不能说原始记录逐坐标相等。

空档没有节点事件，既不调用 $\operatorname{Next}$，也不自主触发选择或发送。若希望衰减跨过阈值时自行激活，必须另外定义触发输入或新的事件规则，不能从本例自动得到。

## 8. 规范记录的阶段化暴露、封闭下界与候选集合关闭

第 6 节直接使用完整的 $E_x$ 与递归产生的 $M^*$。本节不再定义另一份计算结果，而是在已经固定的完整记录 $\mathcal T_x$ 上增加一项数学数据：各记录进入一条递增子集链的阶段。数学中常把这种递增子集族称为过滤（filtration）；本文称它为 $\mathcal T_x$ 的**阶段化暴露模型**。它只记录规范坐标何时进入子集，不从部分输入重新定义另一份语义。附录 S 才把“进入当前子集”翻译成系统语言中的可见或公开。

本节的 $H_n$ 是已经进入阶段 $n$ 的**内部消息集合**；它与式 (14) 的选择历史状态 $y_j^\theta$ 是两种不同对象。

### 8.1 阶段编号与阶段秩

定义：

$$
\overline{\mathbb N}=\mathbb N\cup\{\infty\},
$$

保留 $\mathbb N$ 上原有的次序，规定每个 $b\in\mathbb N$ 都满足 $b<\infty$，并规定 $\infty\le\infty$。因而每个 $c\in\overline{\mathbb N}$ 都满足 $c\le\infty$。同时规定 $\min\varnothing=\infty$。这里的 $\infty$ 不是逻辑时间，只表示一个元素没有进入所研究的阶段序列。

取两个函数，称为外部记录与内部消息的**阶段秩**：

$$
\alpha_E:E_x\to\overline{\mathbb N},
\qquad
\alpha_M:M^*\to\overline{\mathbb N}.
$$

对每个阶段编号 $n\in\mathbb N$，定义：

$$
E_n
=\{e\in E_x\mid\alpha_E(e)\le n\},
\qquad
H_n
=\{m\in M^*\mid\alpha_M(m)\le n\}.
\tag{31}
$$

由式 (31) 立即得到：

$$
E_n\subseteq E_{n+1},
\qquad
H_n\subseteq H_{n+1}.
$$

反之，任意这样的递增消息链都由 $\alpha_M(m)=\min\{n\mid m\in H_n\}$ 唯一表示。

$\alpha_E$ 与 $\alpha_M$ 是在 $x$ 和 $\mathcal T_x$ 之外另行选择的数据。因此 $H_n$ 不是由 $E_n$ 决定的“可计算消息闭包”；它只是式 (31) 给出的 $M^*$ 子集。

若 $\alpha_E(e)=\infty$ 或 $\alpha_M(m)=\infty$，相应元素不会进入任何有限阶段。称这条记录链**穷尽**完整记录，当且仅当：

$$
\bigcup_{n\in\mathbb N}E_n=E_x,
\qquad
\bigcup_{n\in\mathbb N}H_n=M^*.
$$

这等价于所有阶段秩都有限；由于两个集合有限，也等价于某个 $N$ 满足 $E_N=E_x$ 且 $H_N=M^*$。第 8 节的关闭结论不要求整条链穷尽。

$n$ 只给阶段排序，不传入任何节点函数；它与逻辑时间 $\theta$ 无关。多个元素可以具有相同阶段秩，$\alpha_M$ 也不必随消息的逻辑到达时间递增。

### 8.2 阶段时间纤维

定义阶段 $n$ 的时间纤维：

$$
B^{(n)}_{v,\theta}
=
B_{v,\theta}(E_n,H_n).
\tag{32}
$$

由集合包含关系立即得到：

$$
B^{(n)}_{v,\theta}\subseteq B_{v,\theta}.
$$

这里右边是第 6 节定义的完整纤维。事实上，正时延还给出：

$$
B_{v,\theta}
=
B_{v,\theta}(E_x,M^*),
$$

因为到达时间为 $\theta$ 的消息都在小于 $\theta$ 的时间发送，已经属于式 (21) 使用的 $M_{<\theta}$。因此，$B^{(n)}_{v,\theta}$ 就是完整记录中阶段秩不大于 $n$ 的 $(v,\theta)$ 纤维。

### 8.3 输入端口和边的有效下界

在阶段 $n$，给定两个函数：

$$
\sigma_n^{\mathrm{in}}:\mathsf I\to\overline{\mathbb N},
\qquad
\sigma_n^{A}:A\to\overline{\mathbb N}.
$$

称它们相对于固定完整记录 $\mathcal T_x$ 的当前 filtration 截面 $(E_n,H_n)$ **有效**。把这个关系记为：

$$
\begin{aligned}
&\operatorname{ValidSeal}_{\mathcal T_x}
\left(E_n,H_n;
\sigma_n^{\mathrm{in}},\sigma_n^A\right)
\\
&\quad\Longleftrightarrow
\left\{
\begin{aligned}
\forall e\in E_x\setminus E_n,\qquad
&\operatorname{time}(e)
\ge
\sigma_n^{\mathrm{in}}(\operatorname{inport}(e)),
\\
\forall m\in M^*\setminus H_n,\qquad
&\operatorname{time}(m)
\ge
\sigma_n^{A}(\operatorname{edge}(m)).
\end{aligned}
\right.
\end{aligned}
\tag{33}
$$

附录 S 把 $\sigma$ 称为 seal。式 (33) 才是这个词的数学含义；“有效”不是 $\sigma_n$ 脱离上下文的一元性质。因为 $(E_n,H_n)$ 由 $(\alpha_E,\alpha_M,n)$ 决定，也可以把左边等价地写成：

$$
\operatorname{ValidSeal}_{\mathcal T_x}
(\alpha_E,\alpha_M,n;
\sigma_n^{\mathrm{in}},\sigma_n^A).
$$

不过式 (33) 的真值只使用当前截面与固定完整记录；两个不同阶段秩若在阶段 $n$ 诱导同一个 $(E_n,H_n)$，就给出相同的有效性判断。后文说“有效下界”时，均指上述关系在当前截面成立。

为了把这个全称命题改写成集合覆盖关系，对 $i\in\mathsf I$、$a\in A$ 与 $b\in\overline{\mathbb N}$ 定义：

$$
E_x(i,<b)
=\{e\in E_x\mid
\operatorname{inport}(e)=i,
\ \operatorname{time}(e)<b\},
$$

$$
M^*(a,<b)
=\{m\in M^*\mid
\operatorname{edge}(m)=a,
\ \operatorname{time}(m)<b\}.
$$

于是式 (33) 等价于：

$$
E_x\bigl(i,<\sigma_n^{\mathrm{in}}(i)\bigr)
\subseteq E_n
\quad(i\in\mathsf I),
$$

$$
M^*\bigl(a,<\sigma_n^A(a)\bigr)
\subseteq H_n
\quad(a\in A).
$$

所以 $\sigma_n^A(a)=b$ 只断言：边 $a$ 上到达时间严格小于 $b$ 的全部实际消息都属于 $H_n$。$|H_n|$ 不出现在条件中；一个较早消息不属于 $H_n$，不能由许多个较晚消息属于 $H_n$ 来补偿。

若 $\sigma_n^A(a)=\infty$，上述包含关系要求边 $a$ 上的全部实际消息都属于 $H_n$。

相对于已经定义的完整记录，可以写出每个通道的最大有效下界：

$$
\widehat\sigma_n^{\mathrm{in}}(i)
=
\min\left(
\{\operatorname{time}(e)\mid
e\in E_x\setminus E_n,\ \operatorname{inport}(e)=i\}
\right),
$$

$$
\widehat\sigma_n^A(a)
=
\min\left(
\{\operatorname{time}(m)\mid
m\in M^*\setminus H_n,\ \operatorname{edge}(m)=a\}
\right).
$$

空集的最小值按第 8.1 节取 $\infty$。任意不大于相应 $\widehat\sigma$ 的数都是有效下界；大于它的数无效。这个公式只刻画“有效”一词，不给出仅由当前阶段数据构造下界的方法。

式 (33) 借助完整结果 $M^*$ 判断一个下界是否为真；它是判定有效性的条件，不是一条只由当前阶段数据构造下界的规则。可用证据例如外部输入源给出的完整性条件，或第 9.6 节由上游完成推出的条件；无论采用什么证据，它都必须推出式 (33) 的全称命题，不能只凭当前集合看起来没有新元素便任意填写一个较大的数。

合法的阶段化暴露链还要求这些下界不减小：

$$
\sigma_n^{\mathrm{in}}(i)
\le\sigma_{n+1}^{\mathrm{in}}(i),
\qquad
\sigma_n^A(a)\le\sigma_{n+1}^A(a).
\tag{34}
$$

### 8.4 节点与区域的封闭前沿

定义节点 $v$ 的封闭前沿：

$$
\lambda_n(v)
=
\min
\left(
\{\sigma_n^A(a)\mid a\in\operatorname{In}(v)\}
\cup
\{\sigma_n^{\mathrm{in}}(i)\mid i\in\operatorname{InPort}(v)\}
\right).
\tag{35}
$$

若节点没有入边也没有输入端口，括号内是空集，按第 8.1 节规定有 $\lambda_n(v)=\infty$。这种节点的所有完整时间纤维本来就为空。

定义区域的封闭前沿：

$$
\lambda_n(\mathcal R_j)
=
\min_{v\in\mathcal R_j}\lambda_n(v).
\tag{36}
$$

### 8.5 节点纤维关闭引理

> [!lemma] 引理 2：节点纤维关闭
> 若 $\lambda_n(v)>\theta$，则：
> $$
> B^{(n)}_{v,\theta}=B_{v,\theta}.
> \tag{37}
> $$

**证明。** 已知左边包含于右边。若右边还有一个左边没有的元素 $z$，则分两种情况。

若 $z$ 是来自端口 $i$ 的外部输入记录，则 $z\in E_x\setminus E_n$，且 $\operatorname{time}(z)=\theta$。由式 (35) 与 $\lambda_n(v)>\theta$ 可得 $\sigma_n^{\mathrm{in}}(i)>\theta$；这与式 (33) 要求 $\theta\ge\sigma_n^{\mathrm{in}}(i)$ 矛盾。

若 $z$ 是沿边 $a$ 到达的内部消息，同理得到 $\sigma_n^A(a)>\theta$，又与式 (33) 矛盾。因此不存在这样的 $z$，两集合相等。$\square$

严格不等式不能换成 $\lambda_n(v)\ge\theta$。下界等于 $\theta$ 时，式 (33) 仍允许某个不属于 $E_n\cup H_n$ 的记录恰好具有时间 $\theta$。

### 8.6 区域候选集合关闭定理

定义阶段 $n$ 的候选集合：

$$
\mathcal C^{(n)}_{j,\theta}
=
\{v\in\mathcal R_j
\mid B^{(n)}_{v,\theta}\ne\varnothing\}.
$$

> [!theorem] 定理 3：完整候选集合不会再增加
> 若：
> $$
> \lambda_n(\mathcal R_j)>\theta,
> $$
> 则：
> $$
> \mathcal C^{(n)}_{j,\theta}
> =
> \mathcal C_{j,\theta}.
> \tag{38}
> $$

**证明。** 由式 (36)，区域中的每个 $v$ 都满足 $\lambda_n(v)>\theta$。引理 2 对每个成员给出：

$$
B^{(n)}_{v,\theta}=B_{v,\theta}.
$$

因此对每个 $v\in\mathcal R_j$：

$$
B^{(n)}_{v,\theta}\ne\varnothing
\Longleftrightarrow
B_{v,\theta}\ne\varnothing.
$$

取所有满足条件的节点，便得到式 (38)。$\square$

定理同时处理“有一个消息的节点”和“完整时间纤维为空的节点”。$B^{(n)}_{c,\theta}=\varnothing$ 并不能证明 $c$ 不在完整候选集合；还需要 $\lambda_n(c)>\theta$。

### 8.7 用第 7 节例子检查严格不等式

设某个阶段满足 $m_0\in H_n$ 且 $m_1\notin H_n$。若边 $e_1$ 的有效下界是：

$$
\sigma_n^A(e_1)=5,
$$

这是可能的，因为 $\operatorname{time}(m_1)=5$。此时：

$$
\lambda_n(\mathcal R_2)\le5,
$$

所以不能对时间 $5$ 使用定理 3；当前候选集合 $\{a\}$ 不是完整候选集合。

若在阶段 $n'$，两条边上的全部实际消息都已属于 $H_{n'}$，则可以有效地令：

$$
\sigma_{n'}^A(e_0)=\sigma_{n'}^A(e_1)=\infty.
$$

于是 $\lambda_{n'}(\mathcal R_2)=\infty>5$，定理 3 才证明候选集合等于 $\{a,b\}$。

### 8.8 输入完整、节点状态与选择历史就绪是三个命题

定理 3 只证明时间 $\theta$ 的候选节点和每个候选节点的完整纤维已经确定。若式 (23) 使用 $q_v^\theta$ 或 $\widetilde q_{v,\theta}$，还必须先知道 $q_v^\theta$。

任取一个节点事件子集：

$$
\mathsf{Done}\subseteq\mathcal E_x^{\mathrm{node}},
$$

对 $v\in V$ 与 $\theta\in\mathbb N$ 定义状态前驱集合：

$$
\operatorname{StatePred}(v,\theta)
=
\{(v,r)\in\mathcal E_x^{\mathrm{node}}\mid r<\theta\}.
$$

定义：节点 $v$ 在时间 $\theta$ 的旧状态关于 $\mathsf{Done}$ **就绪**，当且仅当：

$$
\operatorname{StatePred}(v,\theta)
\subseteq\mathsf{Done}.
$$

$\operatorname{StatePred}(v,\theta)$ 索引的式 (25) 按 $r$ 递增组成第 6 节状态序列的完整前缀；上述包含关系成立时，该前缀已经到达唯一的 $q_v^\theta$。

再任取一个区域选择事件子集：

$$
\mathsf{Selected}\subseteq\mathcal E_x^{\mathrm{sel}}.
$$

对区域 $j$ 定义选择历史前驱集合：

$$
\operatorname{HistPred}(j,\theta)
=
\{(j,r)\in\mathcal E_x^{\mathrm{sel}}\mid r<\theta\}.
$$

定义区域 $j$ 在时间 $\theta$ 的旧选择历史关于 $\mathsf{Selected}$ **就绪**，当且仅当：

$$
\operatorname{HistPred}(j,\theta)
\subseteq\mathsf{Selected}.
$$

候选集合为空的时间只应用恒等规则，不产生区域选择事件；因此，上述集合恰好索引所有真正改变或读取历史的较早选择作用。包含关系成立时，这些步骤按时间组成唯一的 $y_j^\theta$。

所以，对区域选择事件 $(j,\theta)\in\mathcal E_x^{\mathrm{sel}}$，一个统一而保守的求值条件是：

1. $\lambda_n(\mathcal R_j)>\theta$；
2. 每个 $v\in\mathcal C_{j,\theta}$ 的旧状态关于当前已完成节点事件集合 $\mathsf{Done}$ 就绪；
3. $\operatorname{HistPred}(j,\theta)\subseteq\mathsf{Selected}$；
4. 每个候选描述量已经按式 (23) 求出；
5. $(j,\theta)$ 尚未应用过式 (24)。

当 $\tau_j=0$ 时，描述量本身不读取节点状态，可以先求描述量，再等待节点状态与选择历史就绪后选择、采用状态和执行完整函数。这种提前求值不会改变式 (21)--(30) 定义的结果。

## 9. 规范记录的合法阶段化暴露

第 6 节按逻辑时间递增定义唯一结果。本节在同一份规范记录上安排四类作用进入递增子集的阶段。每个截面同时携带已经出现的外部记录、内部消息、作用值与封闭下界，因此构成一族带证书的过滤。附录 S.9 再说明具体求值过程怎样精化到这种数学轨迹。

### 9.1 阶段化暴露轨迹

对四类作用分别取阶段秩函数，并定义：

$$
\begin{aligned}
\alpha_P&:\mathscr V_x^P\to\overline{\mathbb N},
&\mathsf P_n&=\{\xi\in\mathscr V_x^P\mid\alpha_P(\xi)\le n\},\\
\alpha_S&:\mathscr V_x^S\to\overline{\mathbb N},
&\mathsf S_n&=\{\xi\in\mathscr V_x^S\mid\alpha_S(\xi)\le n\},\\
\alpha_U&:\mathscr V_x^U\to\overline{\mathbb N},
&\mathsf U_n&=\{\xi\in\mathscr V_x^U\mid\alpha_U(\xi)\le n\},\\
\alpha_F&:\mathscr V_x^F\to\overline{\mathbb N},
&\mathsf F_n&=\{\xi\in\mathscr V_x^F\mid\alpha_F(\xi)\le n\}.
\end{aligned}
\tag{39}
$$

四个集合分别索引已经确定的准备、选择、状态采用与完整输出作用。阶段秩给出作用进入当前过滤截面的先后；逻辑时间仍由作用自身的最后一个坐标给出。

为了使轨迹中的函数值也形式确定，对每个节点事件定义规范准备标签：

$$
\operatorname{Lab}_P(v,\theta)
=
(B_{v,\theta},q_v^\theta,h_{v,\theta},
\widetilde q_{v,\theta},d_{v,\theta}),
$$

对每个区域选择事件定义规范选择标签：

$$
\operatorname{Lab}_S(j,\theta)
=
(\mathcal C_{j,\theta},y_j^\theta,
(d_{v,\theta})_{v\in\mathcal C_{j,\theta}},
\mathcal A_{j,\theta},(c_{v,\theta})_{v\in\mathcal C_{j,\theta}},y_j^{\theta+1}),
$$

对每个节点事件定义规范状态采用标签：

$$
\operatorname{Lab}_U(v,\theta)
=
(q^{\mathrm{cmp}}_{v,\theta},q_v^{\theta+1}),
$$

并对每个激活坐标定义规范完整输出标签：

$$
\operatorname{Lab}_F(v,\theta)
=
(f^A_{v,\theta},f^O_{v,\theta}).
$$

阶段 $n$ 的函数值就是四个标签函数在 $\mathsf P_n,\mathsf S_n,\mathsf U_n,\mathsf F_n$ 上的限制。作用首次进入时增加相应规范标签；第 9.2--9.3 节的条件保证该标签的全部自变量已经属于更早截面。

把一个阶段截面的全部数据打包为：

$$
\begin{aligned}
\Xi_n=\bigl(&E_n,H_n,
\sigma_n^{\mathrm{in}},\sigma_n^A,
\mathsf P_n,\mathsf S_n,\mathsf U_n,\mathsf F_n,\\
&\operatorname{Lab}_P|_{\mathsf P_n},
\operatorname{Lab}_S|_{\mathsf S_n},
\operatorname{Lab}_U|_{\mathsf U_n},
\operatorname{Lab}_F|_{\mathsf F_n}\bigr).
\end{aligned}
$$

第 8 节的记录链、下界函数、式 (39) 的四条作用链及上述标签限制合在一起，称为一条**阶段化暴露轨迹**：

$$
\boldsymbol\Xi=(\Xi_n)_{n\in\mathbb N}.
$$

其中 $\Xi_n$ 是一个 filtration 截面，$\boldsymbol\Xi$ 是整条带阶段索引的轨迹。

每条实际消息都有唯一的源完整输出作用。定义：

$$
g_F:M^*\to\mathscr V_x^F,
\qquad
g_F(m)=F_{\operatorname{src}(\operatorname{edge}(m)),\operatorname{send}(m)}.
$$

定义源作用已经进入截面的消息集合：

$$
M_n^{\mathrm{src}}
=g_F^{-1}(\mathsf F_n),
\qquad
H_n\subseteq M_n^{\mathrm{src}}
\qquad(n\in\mathbb N).
\tag{40}
$$

每条实际输出记录也有唯一的源完整输出作用。令：

$$
g_Z:Z^*\to\mathscr V_x^F,
\qquad
g_Z(z)=
F_{\varepsilon(\operatorname{outport}(z)),\operatorname{outtime}(z)},
$$

并定义当前已经确定的输出集合：

$$
Z_n^F=g_Z^{-1}(\mathsf F_n).
$$

式 (40) 允许一个 $F$ 标签先进入截面，其内部消息随后进入 $H$；每条消息进入 $H$ 时，其源 $F$ 已经属于同一截面。输出由 $Z_n^F$ 随源 $F$ 一起确定。

定义关系：

$$
\operatorname{LegalExposure}_{\mathcal T_x}(\boldsymbol\Xi),
$$

它成立，当且仅当 $\mathsf P_0=\mathsf S_0=\mathsf U_0=\mathsf F_0=\varnothing$，并且对每个阶段 $n$ 都满足：

1. $\operatorname{ValidSeal}_{\mathcal T_x} (E_n,H_n;\sigma_n^{\mathrm{in}},\sigma_n^A)$；
2. 式 (34) 的 seal 单调性；
3. 式 (40) 的源事件约束；
4. 第 9.2 节的 $P,S$ 首次进入条件；
5. 第 9.3 节的 $U,F$ 首次进入条件。

称满足这个关系的 $\boldsymbol\Xi$ 为一条**合法阶段化暴露轨迹**。第 9.5 节给出从状态前缀和消息公开推出新边下界的构造。

称合法暴露轨迹**完整**，当且仅当存在 $N\in\mathbb N$ 使：

$$
\begin{aligned}
E_N&=E_x,
&H_N&=M^*,\\
\mathsf P_N&=\mathscr V_x^P,
&\mathsf S_N&=\mathscr V_x^S,\\
\mathsf U_N&=\mathscr V_x^U,
&\mathsf F_N&=\mathscr V_x^F.
\end{aligned}
$$

在第 8.8 节的状态与历史就绪关系中，以下分别取 $\mathsf{Done}=\{(v,\theta)\mid U_{v,\theta}\in\mathsf U_n\}$ 与 $\mathsf{Selected}=\{(j,\theta)\mid S_{j,\theta}\in\mathsf S_n\}$。

### 9.2 合法准备与合法选择

定义阶段 $n$ 到 $n+1$ 的四个增量：

$$
\Delta\mathsf P_n=\mathsf P_{n+1}\setminus\mathsf P_n,
\qquad
\Delta\mathsf S_n=\mathsf S_{n+1}\setminus\mathsf S_n,
$$

$$
\Delta\mathsf U_n=\mathsf U_{n+1}\setminus\mathsf U_n,
\qquad
\Delta\mathsf F_n=\mathsf F_{n+1}\setminus\mathsf F_n.
$$

每个增量的前提均在截面 $\Xi_n$ 上检查。这样，一条作用边的起点与终点至少分属相邻的两个截面。

作用 $P_{v,\theta}$ 可以属于 $\Delta\mathsf P_n$，当且仅当：

1. $\lambda_n(v)>\theta$；
2. 每个更早的实际节点事件 $(v,r)$ 都满足 $U_{v,r}\in\mathsf U_n$。

进入时，以 $B^{(n)}_{v,\theta}$ 和 $q_v^\theta$ 求出 $h_{v,\theta},\widetilde q_{v,\theta},d_{v,\theta}$。引理 2 保证这些值等于第 6 节的相应坐标。

作用 $S_{j,\theta}$ 可以属于 $\Delta\mathsf S_n$，当且仅当：

1. $\lambda_n(\mathcal R_j)>\theta$；
2. 每个 $v\in\mathcal C_{j,\theta}$ 都满足 $P_{v,\theta}\in\mathsf P_n$；
3. 每个更早的实际选择事件 $(j,r)$ 都满足 $S_{j,r}\in\mathsf S_n$。

这次选择把完整函数族：

$$
(d_{v,\theta})_{v\in\mathcal C_{j,\theta}}
$$

与旧选择历史 $y_j^\theta$ 一次交给式 (24)，由此共同产生激活集合、局部控制族与下一选择历史。

若 $\mathcal C_{j,\theta}=\varnothing$，式 (24) 的恒等分支直接传递历史；相应坐标不属于 $\mathscr V_x^S$。

### 9.3 合法状态采用与完整输出

若 $v\in\mathcal C_{j,\theta}$，作用 $U_{v,\theta}$ 可以属于 $\Delta\mathsf U_n$，当且仅当：

1. $P_{v,\theta}\in\mathsf P_n$；
2. $S_{j,\theta}\in\mathsf S_n$；
3. 每个更早的实际节点事件 $(v,r)$ 都满足 $U_{v,r}\in\mathsf U_n$。

进入时按式 (25a)、(25) 确定本次计算快照与下一持久状态。不同节点拥有不同的状态坐标，因此同一区域的多个 $U$ 可以在选择以后进入不同阶段；同一节点的 $U$ 按逻辑时间递增。

若 $v\in\mathcal A_{j,\theta}$，作用 $F_{v,\theta}$ 可以属于 $\Delta\mathsf F_n$，当且仅当 $U_{v,\theta}\in\mathsf U_n$。它进入时应用式 (26)，并由式 (27)--(28) 派生实际消息与输出。派生消息从这一阶段起属于 $M^{\mathrm{src}}$，再依式 (40) 进入某个当前或后续的 $H$ 截面。

### 9.4 作用值、暂存值与规范坐标

固定 $v\in\mathcal C_{j,\theta}$。纤维关闭以后，$h_{v,\theta}$ 已具有唯一值；旧状态确定以后，$\widetilde q_{v,\theta}$ 与三个描述量也具有唯一值。具体求值过程可以暂存这些值。过滤截面只在相应作用满足第 9.2--9.3 节条件时把它们收入规范标签限制。

由此得到一条简单的封闭原则：

- 后续状态只读取已经属于 $\mathsf U_n$ 的前驱状态；
- 选择只读取已经属于 $\mathsf P_n$ 的完整候选描述族；
- 完整输出只读取已经属于 $\mathsf U_n$ 的计算快照；
- 内部消息只从已经属于 $\mathsf F_n$ 的源作用进入 $H_n$。

### 9.5 有效下界约束消息阶段秩

若有效值 $\sigma_n^A(a)=b$，则式 (33) 要求每个满足：

$$
\operatorname{edge}(m)=a,
\qquad
\operatorname{time}(m)<b
$$

的消息都属于 $H_n$，即 $\alpha_M(m)\le n$。所以这样的消息不可能在更晚阶段首次进入 $H$。

阶段秩不必随逻辑到达时间递增；它只受已经给出的有效下界约束。

### 9.6 边下界怎样由上游完成推出

对 $v\in V$ 与 $b\in\mathbb N$ 定义谓词：

$$
\begin{aligned}
\operatorname{DoneTo}_n(v,b)
\Longleftrightarrow {}&
\lambda_n(v)\ge b\\
&{}\land
\{U_{v,r}\mid(v,r)\in\mathcal E_x^{\mathrm{node}},\ r<b\}
\subseteq\mathsf U_n\\
&{}\land
\{F_{v,r}\in\mathscr V_x^F\mid r<b\}
\subseteq\mathsf F_n.
\end{aligned}
\tag{41}
$$

谓词 $\operatorname{DoneTo}_n(v,b)$ 同时覆盖时间小于 $b$ 的输入纤维、状态采用前缀与实际完整输出前缀。

若 $\operatorname{DoneTo}_n(v,b)$ 成立，则该节点时间小于 $b$ 的全部状态采用与实际完整输出作用均已进入截面。对任意 $a\in\operatorname{Out}(v)$，以后由该节点新增的完整输出作用所产生的消息，其到达时间不小于：

$$
b+\delta(a).
$$

如果还满足：

$$
M_n^{\mathrm{src}}
\cap M^*\bigl(a,<b+\delta(a)\bigr)
\subseteq H_n,
$$

那么：

$$
\sigma_n^A(a)=b+\delta(a)
\tag{42}
$$

是一个有效下界。

事实上，任取 $m\in M^*(a,<b+\delta(a))$，都有 $\operatorname{send}(m)<b$，所以其源 $F$ 属于 $\mathsf F_n$，即 $m\in M_n^{\mathrm{src}}$。附加条件给出 $m\in H_n$，再用式 (33) 即得结论。

式 (42) 把状态采用前缀、源 $F$ 作用和消息进入 $H$ 的三个条件合成一项边下界证书。

### 9.7 一条顺序参考轨迹

> [!proposition] 命题 4：完整合法顺序参考轨迹存在
> 对任意固定规格与输入 $x$，至少存在一条完整合法阶段化暴露轨迹。

**证明。** 令全部外部记录从阶段 $0$ 起属于 $E_n$。从四个空作用集合与 $H_0=\varnothing$ 开始，依次处理 $\theta=0,1,\ldots,\Theta_{\max}$。对每个时间依次用四次阶段转移加入本时间的全部 $P$、全部 $S$、全部 $U$ 和全部实际 $F$。加入 $F$ 后令 $H_{n+1}=M_{n+1}^{\mathrm{src}}$；其余转移保持同一等式。每个阶段都取第 8.3 节由当前 $E_n,H_n$ 定义的最大有效下界 $\widehat\sigma_n^{\mathrm{in}},\widehat\sigma_n^A$。

在加入时间 $\theta$ 的 $P$ 以前，所有更小逻辑时间的 $U$ 与实际 $F$ 都已进入。若一条尚未属于 $H_n$ 的消息满足 $\operatorname{time}(m)\le\theta$，则正时延给出 $\operatorname{send}(m)<\theta$；其源 $F$ 已经进入，又由 $H_n=M_n^{\mathrm{src}}$ 得到 $m\in H_n$，矛盾。因此每条入边的最大有效下界严格大于 $\theta$；外部记录已经全部属于 $E_n$，所以输入端口下界为 $\infty$。每个待加入的 $P$ 因而具有关闭纤维与完整状态前缀。

随后，本时间的全部 $P$ 为 $S$ 提供完整描述族，本时间的 $S$ 为全部 $U$ 提供选择结果，本时间的 $U$ 为实际 $F$ 提供计算快照。第 9.2--9.3 节的条件逐项成立。$E_n,H_n$ 只增不减，最大有效下界满足式 (34)，而 $H_n=M_n^{\mathrm{src}}$ 满足式 (40)。最后在某个阶段 $N$，四类作用全部进入且 $H_N=M^*$。对所有 $n\ge N$ 保持阶段 $N$ 的截面，便得到定义在整个 $\mathbb N$ 上的完整轨迹。$\square$

命题 4 给出一条基准轨迹；其他合法轨迹可以同时加入任意多个已经满足前提的作用。

### 9.8 完整暴露的终值

> [!theorem] 定理 5：规范记录的完整暴露与次序无关
> 任取两条从相同固定规格与输入 $x$ 开始的完整合法阶段化暴露轨迹。它们可以具有不同阶段秩和中间子集，但在各自完整阶段暴露相同的 $E_x,M^*,Z^*$ 以及全部 $P,S,U,F$ 标签；忽略阶段秩以后，终端所暴露的规范记录都是同一个 $\mathcal T_x$。

**证明。** 完整性给出 $E_N=E_x$、$H_N=M^*$ 以及四类作用集合的完整定义域。第 9.1 节把每个阶段的函数值定义为固定规范标签函数在相应子集上的限制，所以完整定义域上的限制就是标签函数本身，并且 $Z_N^F=Z^*$。空纤维和空候选集合分别由式 (25) 与式 (24) 的恒等分支补齐状态坐标。阶段秩只改变限制扩大的先后。$\square$

定理 5 给出固定 $\mathcal T_x$ 上的过滤唯一性。附录 S.9 用精化映射把具体求值记录连接到这里的规范标签与阶段截面。

### 9.9 三个派生的节点级公开视图

从四类作用截面可以取出三个较粗的索引集合：

$$
\begin{aligned}
\mathsf{Prepared}_n
&=\{(v,\theta)\in\mathcal E_x^{\mathrm{node}}
\mid P_{v,\theta}\in\mathsf P_n\},\\
\mathsf{Selected}_n
&=\{(j,\theta)\in\mathcal E_x^{\mathrm{sel}}
\mid S_{j,\theta}\in\mathsf S_n\},\\
\mathsf{Completed}_n
&=\{(v,\theta)\mid
(v,\theta)\in\mathcal E_x^{\mathrm{node}},\\
&\qquad U_{v,\theta}\in\mathsf U_n,
(v\notin\mathcal A_{\rho(v),\theta}
\ \lor\ F_{v,\theta}\in\mathsf F_n)\}.
\end{aligned}
$$

$\mathsf{Prepared}_n$ 表示本地准备标签已经进入，$\mathsf{Selected}_n$ 表示一次区域共同选择的全部输出已经进入，$\mathsf{Completed}_n$ 表示节点的状态采用已经进入，并且激活节点的完整输出也已经进入。它们由 $\Xi_n$ 唯一确定，适合陈述节点级的公开进度。

细粒度截面还保留两项派生视图未表达的信息：$U$ 与 $F$ 可以分属不同阶段，且一条已由 $F$ 产生的消息可以在更晚阶段进入 $H$。因此，状态批、完整输出批和消息公开分别直接使用 $\mathsf U_n,\mathsf F_n,H_n$ 计量。

## 10. 一次计算产生的事件 DAG

第 6.4 节已经从完整记录中取出 $P,S,U,F$ 四类作用。本节用有向边记录它们之间的直接函数自变量依赖。所得图刻画一次具体输入实际产生的因果结构；第 8 节的封闭下界则作为作用进入过滤截面的证书。

### 10.1 函数作用事件的集合

固定输入 $x$ 及其完整记录 $\mathcal T_x$，取第 6.4 节的有限不交并 $\mathscr V_x^{\mathrm{ev}}$ 作为顶点集合。

令：

$$
\mathscr A_x^{\mathrm{ev}}
\subseteq
\mathscr V_x^{\mathrm{ev}}
\times
\mathscr V_x^{\mathrm{ev}}
$$

是恰好由第 10.2--10.4 节所列有序对组成的关系；除这些小节列出的有序对以外，不加入其他边。定义事件图：

$$
\mathscr G_x^{\mathrm{ev}}
=
(\mathscr V_x^{\mathrm{ev}},\mathscr A_x^{\mathrm{ev}}).
$$

### 10.2 同一时间内的依赖边

若 $v\in\mathcal C_{j,\theta}$，令下列三个有序对属于 $\mathscr A_x^{\mathrm{ev}}$：

$$
P_{v,\theta}
\longrightarrow
S_{j,\theta}
\longrightarrow
U_{v,\theta},
\qquad
P_{v,\theta}\longrightarrow U_{v,\theta}.
$$

若 $v\in\mathcal A_{j,\theta}$，再令下列三个有序对属于 $\mathscr A_x^{\mathrm{ev}}$：

$$
U_{v,\theta}\longrightarrow F_{v,\theta},
\qquad
P_{v,\theta}\longrightarrow F_{v,\theta},
\qquad
S_{j,\theta}\longrightarrow F_{v,\theta}.
$$

这些边分别记录式 (18) 在同一逻辑时间内的直接函数自变量依赖：选择读取准备所得的描述量，状态采用读取选择结果与候选新状态，完整输出读取本次计算快照、准备所得的本地内容与选择给出的局部控制量。$S\to F$ 虽已被 $S\to U\to F$ 蕴含，仍作为直接函数自变量依赖保留。

### 10.3 节点状态与选择历史的跨时间依赖

若：

$$
(v,\theta),(v,\theta')\in\mathcal E_x^{\mathrm{node}},
\qquad
\theta<\theta',
$$

并且不存在满足：

$$
\theta<r<\theta',
\qquad
(v,r)\in\mathcal E_x^{\mathrm{node}}
$$

的 $r\in\mathbb N$，则令下列有序对属于 $\mathscr A_x^{\mathrm{ev}}$：

$$
U_{v,\theta}\longrightarrow P_{v,\theta'}.
$$

它表示 $P_{v,\theta'}$ 使用的旧状态由前一个节点事件的状态采用作用 $U_{v,\theta}$ 所给下一持久状态决定；它不读取前一次计算快照。

类似地，若：

$$
(j,\theta),(j,\theta')\in\mathcal E_x^{\mathrm{sel}},
\qquad
\theta<\theta',
$$

并且不存在满足：

$$
\theta<r<\theta',
\qquad
(j,r)\in\mathcal E_x^{\mathrm{sel}}
$$

的 $r\in\mathbb N$，则令：

$$
S_{j,\theta}\longrightarrow S_{j,\theta'}
$$

属于 $\mathscr A_x^{\mathrm{ev}}$。候选集合为空的中间时间只保持历史，所以 $S_{j,\theta'}$ 读取的 $y_j^{\theta'}$ 由前一个实际选择作用确定。这里没有从 $F_{v,\theta}$ 指向后续 $S_{j,\theta'}$ 的 history 边；式 (14) 已经排除了这种直接依赖。不过，过去的完整输出仍可经普通消息影响未来的描述量和选择，例如形成 $F_{v,\theta}\to P_{w,\theta'}\to S_{j,\theta'}$ 的消息依赖路径。

### 10.4 消息依赖

若 $F_{v,\theta}$ 产生消息：

$$
m=(\mathrm{msg},\theta,a,y),
$$

则令下列有序对属于 $\mathscr A_x^{\mathrm{ev}}$：

$$
F_{v,\theta}
\longrightarrow
P_{\operatorname{dst}(a),\theta+\delta(a)}.
$$

一个目标纤维若含多个消息，其准备顶点就可以有多个消息依赖前驱。

### 10.5 事件图无环

先在 $\mathscr V_x^{\mathrm{ev}}$ 上定义时间函数，使每个形式符号的时间都是其第二个下标：

$$
\operatorname{etime}(P_{v,\theta})
=\operatorname{etime}(U_{v,\theta})
=\operatorname{etime}(S_{j,\theta})
=\operatorname{etime}(F_{v,\theta})
=\theta.
$$

再定义阶段函数：

$$
p(P_{v,\theta})=0,
\qquad
p(S_{j,\theta})=1,
\qquad
p(U_{v,\theta})=2,
\qquad
p(F_{v,\theta})=3.
$$

对每个事件顶点 $\xi\in\mathscr V_x^{\mathrm{ev}}$ 定义秩：

$$
\operatorname{rank}(\xi)
=
(\operatorname{etime}(\xi),p(\xi))
\in\mathbb N\times\{0,1,2,3\}.
\tag{43}
$$

这个秩由事件坐标固定，只用于证明无环；阶段化暴露另由 $\alpha_E,\alpha_M,\alpha_P,\alpha_S,\alpha_U,\alpha_F$ 给出。

这里的字典序定义为：

$$
(\theta,p)<_{\mathrm{lex}}(\theta',p')
\Longleftrightarrow
\bigl(\theta<\theta'\bigr)
\qquad\text{或}\qquad
\bigl(\theta=\theta'\ \text{且}\ p<p'\bigr).
$$

按这个次序比较秩。同一时间内的依赖严格增加 $p$；节点状态与选择历史依赖严格增加 $\theta$；消息依赖由 $\delta(a)>0$ 也严格增加 $\theta$。所以每条事件边都严格增加式 (43)。沿有向边不可能回到原秩，因此事件图没有有向环。

由此得到两项不同事实：

1. 固定空间图 $G$ 按第 2.2 节的假设是 DAG；
2. 每次具体输入产生的上述细分事件图也由式 (43) 证明为 DAG。

`TimedDAG` 不是在这两者中二选一。本文的规格以固定空间 DAG 为结构，并且它的每次运行还导出一张依赖于输入的事件 DAG。前者说明允许在哪里传值，后者说明这一次实际发生的函数作用怎样依赖。

### 10.6 区域商图可以有环

只为研究区域布局，定义关系：

$$
Q_\rho
=
\{(j,j')\in J\times J
\mid j\ne j',
\ \exists a\in A:
\rho(\operatorname{src}(a))=j,
\rho(\operatorname{dst}(a))=j'\}.
\tag{44}
$$

称 $(J,Q_\rho)$ 为区域商图。它不是消息图。

$\mathcal R_j$ 本身只是一个节点子集。如果把 $G$ 中起点和终点都位于 $\mathcal R_j$ 的边取出来，就得到 $G$ 在该子集上的诱导子图；由于 $G$ 已经是 DAG，这个诱导子图自动无环，不需要再加一条“区域诱导子图无环”公理。与此不同，式 (44) 把每个节点子集收缩成一个点，所得区域商图可以有环。

即使固定节点图是路径：

$$
u\longrightarrow v\longrightarrow w,
$$

只要 $u,w\in\mathcal R_0$ 且 $v\in\mathcal R_1$，区域商图就含有：

$$
0\longrightarrow1\longrightarrow0.
$$

这不构成函数作用事件图中的有向环。即使某次计算确实沿这两条空间边依次产生消息，每条消息依赖边也会因 $\delta(a)>0$ 而严格增加逻辑时间。若第一条消息由逻辑时间 $\theta$ 的节点事件产生，那么返回区域 $0$ 时，对应的节点事件与区域选择事件具有某个逻辑时间 $\theta'>\theta$。选择历史只给同一个 $j$ 的选择事件增加跨时间边，不增加式 (44) 的跨区域边。本文因此不要求区域商图无环。

### 10.7 因果链与联合求值

同一节点相邻时间的边

$$
U_{v,\theta}\longrightarrow P_{v,\theta'}
$$

规定后一作用读取前一作用留下的持久状态。对一个递增时间集合 $\Theta=\{\theta_1<\cdots<\theta_k\}$，这组边形成一条状态递归。下面先定义这类递归的一般联合契约。

固定非空状态集合 $\mathsf Q$、驱动集合 $\mathsf Z_1,\ldots,\mathsf Z_k$ 和全函数：

$$
T_i:\mathsf Q\times\mathsf Z_i\longrightarrow\mathsf Q
\qquad(1\le i\le k).
$$

从 $q_0\in\mathsf Q$ 与 $(z_i)_{i=1}^k$ 出发，递归定义 $q_i=T_i(q_{i-1},z_i)$。由此得到参考状态转导：

$$
\begin{aligned}
\operatorname{RefState}_{\Theta}
&:\mathsf Q\times\prod_{i=1}^{k}\mathsf Z_i
\longrightarrow\mathsf Q^{\{0,\ldots,k\}},\\
\operatorname{RefState}_{\Theta}
(q_0,(z_i)_{i=1}^{k})
&=(q_i)_{i=0}^{k}.
\end{aligned}
$$

同类型全函数

$$
\operatorname{BatchState}_{\Theta}:
\mathsf Q\times\prod_{i=1}^{k}\mathsf Z_i
\longrightarrow\mathsf Q^{\{0,\ldots,k\}}
$$

满足

$$
\operatorname{BatchState}_{\Theta}
=\operatorname{RefState}_{\Theta}
$$

时，称为这条递归的**精确因果状态契约**。它从左边界状态和整段已经确定的驱动量返回全部中间状态与右边界状态；等式在整个定义域上成立。

因此，事件 DAG 给出规范上的偏序，联合函数给出这张偏序的一种求值方式。一次具体联合调用经过精化以后可以对应许多个 $P,U$，也可以同时对应其后的 $F$。内部的状态边仍保留在规范记录中，而联合调用在外层只占一个求值块。

哪些作用计入主要成本、哪些递归具有状态块、哪些状态与完整输出可以融合，都由附加的函数类与联合求值契约给出。[[timed-dag-chunk-prefill-learning-note|分块预填充教材]] 在本节事件 DAG 上定义这些契约，并分别计量状态批、完整输出批与自适应阶段。

## 11. 在逻辑时间切面停止与继续

### 11.1 完整记录的左前缀

固定 $b\in\mathbb N$，并要求 $0\le b\le\Theta_{\max}+1$。定义 $\mathcal T_x$ 在切面 $b$ 左侧的**规范前缀** $\mathcal T_{x,<b}$，它由以下坐标组成：

1. $q_v^0,\ldots,q_v^b$ 与 $y_j^0,\ldots,y_j^b$；
2. 所有逻辑时间小于 $b$ 的规范 $P,S,U,F$ 标签；
3. 由这些 $F$ 标签按式 (27)--(28) 派生的全部内部消息与外部输出记录。

等价地，第二项取第 10 节中满足 $\operatorname{etime}(\xi)<b$ 的全部函数作用事件及其函数值。由于一个 $S_{j,\theta}$ 同时确定激活集合、局部控制族与 $y_j^{\theta+1}$，这个前缀不会停在一次区域选择的三个输出之间。$\mathcal T_{x,<b}$ 是唯一完整记录 $\mathcal T_x$ 的限制，不依赖某条阶段轨迹是否已经求得它。

### 11.2 跨越切面的内部消息

定义：

$$
W_b
=
\{m\in M^*\mid
\operatorname{send}(m)<b
\le\operatorname{time}(m)\}.
\tag{45}
$$

$W_b$ 中的每条消息都已经由某个满足 $\theta<b$ 的完整输出作用事件 $F_{v,\theta}$ 产生，但它到达的完整时间纤维位于切面右侧。丢掉 $W_b$ 会改变未来某些式 (21)。

### 11.3 切面状态、未来输入和输出前缀

第 6 节的 $q_v^b$ 正好是节点 $v$ 完成所有满足 $\theta<b$ 的状态采用作用事件 $U_{v,\theta}$ 以后、处理逻辑时间 $b$ 的任何节点事件以前的状态。

同样，$y_j^b$ 是区域 $j$ 完成所有满足 $\theta<b$ 的选择作用事件以后、处理时间 $b$ 的任何区域选择事件以前的历史；候选为空的时间只作恒等传递。

定义未来外部输入记录：

$$
E_{\ge b}
=
\{e\in E_x\mid\operatorname{time}(e)\ge b\},
$$

以及已经产生的外部输出前缀：

$$
Z_{<b}
=
\{z\in Z^*\mid\operatorname{outtime}(z)<b\}.
$$

定义切面上的未来计算状态：

$$
Q_b
=
\left(
b,
(q_v^b)_{v\in V},
(y_j^b)_{j\in J},
W_b
\right).
\tag{46}
$$

$Z_{<b}$ 不影响未来节点计算；若要在恢复后重建完整多端口输出记录，则还要保存 $Z_{<b}$ 或保存“这些输出已被外部可靠接收”的等价证据。

### 11.4 分段继续定理

> [!theorem] 定理 6：规范左前缀上的继续等于一次算完
> 从 $Q_b$ 开始，以 $(q_v^b)$ 为初始节点状态、以 $(y_j^b)$ 为初始选择历史，令恢复递归开始时已有消息集合 $M^{\mathrm{res}}_{<b}=W_b$，并只使用 $E_{\ge b}$。对
> $$
> \theta=b,b+1,\ldots,\Theta_{\max}
> $$
> 依次递归，把式 (21) 改写为：
> $$
> B^{\mathrm{res}}_{v,\theta}
> =
> B_{v,\theta}(E_{\ge b},M^{\mathrm{res}}_{<\theta})
> \qquad(b\le\theta\le\Theta_{\max}),
> $$
> 其余步骤仍按第 6.3 节递归；把本次按式 (27) 产生的消息记为 $M_\theta^{\mathrm{res}}$，并在每一步令：
> $$
> M^{\mathrm{res}}_{<\theta+1}
> =M^{\mathrm{res}}_{<\theta}\cup M_\theta^{\mathrm{res}}.
> $$
> 所得第 6.3 节全部时间不小于 $b$ 的后缀坐标——完整时间纤维 $B$，候选节点的 $h,\widetilde q,d,c,q^{\mathrm{cmp}}$，候选集合 $\mathcal C$，激活集合 $\mathcal A$，节点状态 $q$，选择历史 $y$，以及激活节点的 $f^A,f^O$——逐坐标等于 $\mathcal T_x$ 的相应值。重新产生的内部消息恰为
> $$
> \{m\in M^*\mid\operatorname{send}(m)\ge b\},
> $$
> 重新产生的外部输出恰为
> $$
> \{z\in Z^*\mid\operatorname{outtime}(z)\ge b\}.
> $$
> $W_b$ 则恰好提供满足 $\operatorname{send}(m)<b\le\operatorname{time}(m)$ 的边界消息；它们是右侧时间纤维的已有输入，不会被重新产生。

若 $b=\Theta_{\max}+1$，上述递归区间为空；由定理 1 还有 $W_b=\varnothing$，且不存在发送时间或输出时间不小于 $b$ 的记录，定理结论直接成立。

**证明。** 以下设 $b\le\Theta_{\max}$，并对 $\theta=b,\ldots,\Theta_{\max}$ 归纳。归纳中保持消息不变量：

$$
M^{\mathrm{res}}_{<\theta}
=
W_b\cup\bigcup_{r=b}^{\theta-1}M_r.
$$

它在 $\theta=b$ 时由恢复递归的初值成立。时间 $b$ 的旧节点状态与旧选择历史都由式 (46) 给出，并与完整计算相同。

一般地，任取 $\theta\ge b$。完整计算中能进入时间 $\theta$ 纤维的外部记录必属于 $E_{\ge b}$。能进入该纤维的内部消息由式 (29) 满足 $\operatorname{send}(m)<\theta$：其中发送时间小于 $b$ 的消息恰由 $W_b$ 提供，发送时间 $r\in[b,\theta)$ 的消息则由归纳式右边的 $M_r$ 提供。恢复集合刻意不保存发送时间与到达时间都小于 $b$ 的已消费消息，但这些消息不可能成为时间 $\theta$ 的原子。因此，恢复集合与原累积消息集合虽然未必相等，却在构造 $B_{v,\theta}$ 时给出完全相同的原子。

于是式 (21)--(28) 的全部自变量相同，所得 $B,h,\widetilde q,d,c,q^{\mathrm{cmp}},\mathcal C,\mathcal A,q,y,f^A,f^O$ 以及本时间的消息与输出均逐坐标相同。特别地 $M_\theta^{\mathrm{res}}=M_\theta$；代入恢复消息更新式便得到时间 $\theta+1$ 的同一不变量。归纳完成。$\square$

这个定理说明式 (46) 中的持久节点状态、选择历史与跨界消息共同足以从规范前缀 $\mathcal T_{x,<b}$ 的右边继续当前模型。完整切面左边的所有完整输出已经求出，因此不需再保存过去的 $c_{v,\theta}$ 或 $q^{\mathrm{cmp}}_{v,\theta}$；它们留在完整记录中，却不是未来递归所需的持久坐标。若停在某次完整输出尚未求出的中间阶段，就不是这里的完整切面，仍须保存其待用快照和控制量。

## 12. 从规范记录到联合求值的层次

### 12.1 区域前沿由最慢成员决定

由式 (36)，对每个 $v\in\mathcal R_j$：

$$
\lambda_n(\mathcal R_j)\le\lambda_n(v).
$$

例如成员前沿分别为 $100,96,17$ 时，区域前沿是 $17$。前两个节点已经能够证明更远时间的纤维完整，但整个区域在时间不小于 $17$ 处仍可能新增候选成员。

这个最小值决定整个 region 当前可以共同使用的时间范围。

### 12.2 当前已经关闭的区域时间集合

任取 $c_j\in\mathbb N$，并要求 $0\le c_j\le\Theta_{\max}+1$；把它作为本次要研究的时间区间左端点。定义：

$$
\mathcal W_{j,n}
=
\{\theta\in[c_j,\Theta_{\max}+1)
\mid
\theta<\lambda_n(\mathcal R_j),
\ \mathcal C_{j,\theta}\ne\varnothing\}.
\tag{47}
$$

对每个 $\theta\in\mathcal W_{j,n}$，定理 3 已经固定候选集合。这些关闭坐标构成区域联合函数的合法输入范围；状态与选择历史怎样整段求值由第 10.7 节定义的精确因果状态契约给出。

### 12.3 区域拓扑次序

若式 (44) 的 $(J,Q_\rho)$ 无环，可以选择一个区域全序，使每条商图边的起点排在终点以前；这样的全序称为区域拓扑序。其存在性可对 $|J|$ 归纳证明：有限 DAG 至少有一个无入边顶点；先把该顶点排在最前，删除后对剩余图重复。

区域拓扑序允许前一 region 的完整区间输出成为后一 region 的关闭输入，从而构造分层时间块。一般区域商图仍由第 6 节的逻辑时间递归和第 10 节事件 DAG 处理；性能分析使用与该图相容的联合契约。

### 12.4 两种联合求值等式

固定 $k\in\mathbb N_{>0}$。对每个 $\ell\in\{1,\ldots,k\}$，给定集合 $X_\ell,Y_\ell$ 和函数：

$$
f_\ell:X_\ell\to Y_\ell.
$$

再给定两个集合 $W_{\mathrm{in}},W_{\mathrm{out}}$。若要用一个联合函数代替逐项求值，至少要构造三个有明确类型的函数：

$$
\operatorname{Pack}:
\prod_{\ell=1}^{k}X_\ell
\to W_{\mathrm{in}},
$$

$$
\mathcal K:W_{\mathrm{in}}\to W_{\mathrm{out}},
$$

$$
\operatorname{Unpack}:
W_{\mathrm{out}}
\to\prod_{\ell=1}^{k}Y_\ell,
$$

并证明对所有 $(z_\ell)_{\ell=1}^{k}\in\prod_{\ell=1}^{k}X_\ell$：

$$
\operatorname{Unpack}
\left(
\mathcal K(\operatorname{Pack}((z_\ell)_{\ell=1}^{k}))
\right)
=
\left(f_\ell(z_\ell)\right)_{\ell=1}^{k}.
\tag{48}
$$

式 (48) 适合全部输入坐标已经分别确定的逐坐标函数，例如一组 $F$ 作用。状态递归采用第 10.7 节的因果形式。固定非空状态集合 $\mathsf Q$、驱动集合 $(\mathsf Z_\theta)_{\theta\in[b,c)}$ 和全函数：

$$
T_\theta:\mathsf Q\times\mathsf Z_\theta\longrightarrow\mathsf Q
\qquad(b\le\theta<c).
$$

给定左边界状态 $q_b\in\mathsf Q$ 和有序驱动族 $(z_\theta)_{\theta\in[b,c)}\in \prod_{\theta\in[b,c)}\mathsf Z_\theta$，参考递归定义：

$$
q_{\theta+1}=T_\theta(q_\theta,z_\theta),
\qquad b\le\theta<c.
$$

相应的参考转导与精确因果联合函数具有共同类型：

$$
\operatorname{RefState}_{[b,c)},
\operatorname{BatchState}_{[b,c)}:
\mathsf Q\times\prod_{\theta\in[b,c)}\mathsf Z_\theta
\longrightarrow
\prod_{\theta\in[b,c+1)}\mathsf Q.
$$

精确因果状态契约要求：

$$
\begin{aligned}
&\operatorname{BatchState}_{[b,c)}
(q_b,(z_\theta)_{\theta\in[b,c)})\\
&\qquad=(q_\theta)_{\theta\in[b,c+1)}\\
&\qquad=\operatorname{RefState}_{[b,c)}
(q_b,(z_\theta)_{\theta\in[b,c)}).
\end{aligned}
$$

中间状态可以在联合函数内部产生。定理 3 提供关闭输入域，式 (48) 与这个全定义域等式分别提供完整输出批和状态块的精确性见证。

### 12.5 五层研究顺序

可以把后续工作分成五层：

1. **语义层**：第 1--6 节定义映射 $x\mapsto\mathcal T_x$；
2. **作用层**：第 6.4 与第 10 节给出 $P/S/U/F$ 和事件 DAG；
3. **过滤层**：第 8--9 节定义合法暴露、关闭条件和派生进度视图；
4. **契约层**：式 (48)、因果状态等式及分块续篇定义精确联合函数；
5. **成本层**：选定计算模型以后研究块数、工作量、跨度和设备测量。

区域划分给出共同选择边界，函数类给出可用的联合契约。[[timed-dag-chunk-prefill-learning-note|分块预填充续篇]] 以本教材的作用 DAG 和 filtration 为规范基础，构造契约相对的最大前沿递归。

## 13. 定义边界与推广范围

本节说明哪些前提支撑现有证明，以及改变它们会触及什么结论。已完成的推广、仅供讨论的扩展与明确排除的情形分别标明，不构成后续开发清单。

### 13.1 允许固定节点图有环，但仍保持正时延

若删除第 2.2 节“固定图无环”的条件，而仍保持每条边 $\delta(a)>0$，则对任意固定有限时间上界 $B$，按 $\theta=0,\ldots,B$ 的递归仍然唯一，因为当前消息只能影响更大时间。

但一次完整运行可能沿有向环不断产生更大时间的消息，不再具有式 (19) 的全局有限上界。因此要改为研究：给定有限逻辑时间切面，是否能在有限工作后完成该切面。定理 1 的“整个运行有限”部分不能原样保留。

这一推广现已在续篇 [[positive-delay-graph-finite-cut-learning-note|《正时延有向图的有限切面语义》]] 中完成。续篇不改写本文的 TimedDAG 定义，而是以相容的有限切面记录代替全局有限完整记录，并证明封闭下界、分段继续与切面复合的相应结论。

### 13.2 正时延前提不能直接删除

零时延边不在当前设计或研究候选中。这里只用它说明正时延前提的作用：若某条边满足 $\delta(a)=0$，时间 $\theta$ 的完整输出可能立即改变同一时间的输入纤维。式 (29) 失效，事件秩可能从阶段 $3$ 指回同一时间的阶段 $0$。

因此不能只把 $0$ 填入现有 $\delta$ 而继续使用当前递归与证明。

### 13.3 节点内部异步计算

在完整纤维已经确定后，一个节点可以把 $\operatorname{Agg}$、$\operatorname{Upd}$、读取函数和 $\operatorname{Full}$ 分成许多内部步骤。只要存在一个删除这些内部中间量的函数，并且删除后得到的本次快照、下一持久状态、消息和输出仍等于第 4--6 节的函数值，这种分解就没有改变正文语义。

若希望在同一逻辑时间的完整纤维尚未确定前，就根据部分原子发布不可撤销状态或图内消息，则已不再实现本文函数。它需要新的节点语义，并必须重新证明候选集合、消息发射和封闭下界之间的关系。

### 13.4 让完整输出改变后续状态或选择

当前式 (14) 只允许选择作用读取旧历史、逻辑时间和当前描述量，式 (17a) 的 $\operatorname{Next}$ 也只读取声明的控制与本地状态；二者均在不调用 $\operatorname{Full}$ 的条件下确定。若要让 $y_j^{\theta+1}$ 读取某个 $f^A_{v,\theta}$ 或 $f^O_{v,\theta}$，就必须改变式 (14)，并把历史更新从当前选择作用中拆出。

另一种扩展是让完整输出写回仅供该节点未来完整计算读取的私有状态；它也超出了式 (12) 的值域与当前状态规则，不能把这种递归藏进一个看似无状态写回的全函数。若该状态还影响后续区域选择，则增加了更强的控制反馈。这两种扩展都需要重新定义事件边和节点时间块契约，不是本文核心默认。

事件图随后会出现从 $F_{v,\theta}$ 到后续状态或历史更新作用的边。直接语义仍需重新规定更新发生在哪个逻辑时间；第 10 节的依赖图、第 11 节的分段继续，以及“先确定整个时间块的选择、再批量应用 $\operatorname{Full}$”的求值次序都要重新证明。

### 13.5 允许一个节点属于多个选择域

当前满射 $\rho:V\to J$ 保证每个节点恰属一个区域。若一个节点同时参加多个选择，必须定义多个选择结果发生冲突时怎样决定状态采用与完整输出。把 $\rho$ 换成一般关系而不增加冲突规则，不会得到一个完整函数。

## 14. 本文已经证明的结果与尚未证明的结果

本文正文已经给出：

1. 完整输入下有限且唯一的直接语义（定理 1）；
2. 节点时间纤维的关闭条件（引理 2）；
3. 区域候选集合不会再增加的条件（定理 3）；
4. 至少一条完整合法顺序暴露轨迹存在（命题 4）；
5. 规范记录的完整暴露与阶段次序无关（定理 5）；
6. 规范逻辑时间左前缀上的继续等于一次算完（定理 6）；
7. $P/S/U/F$ 四类作用的合法细粒度 filtration，以及由它派生的 Prepared/Selected/Completed 视图；
8. 选择历史的唯一递归、跨时间依赖与切面保存；
9. 每次有限运行的规范值与状态依赖事件图无环；
10. 区域商图允许有环，联合求值按附加契约分析。

仍需单独完成的工作包括：

- 把第 9 节实现成规范记录上的阶段暴露检查器，并以随机合法阶段秩检验式 (33)--(40)；程序求值的正确性按附录 S.9 作精化检验；
- 为更多受限模型与具体实现给出完整坐标映射或精化证明，而不只比较最终输出；
- 找到能推出较大关闭窗口的区域结构定理；
- 对具体神经节点证明式 (48) 或因果状态等式的联合求值契约与复杂度；
- 若改变当前事件触发规则或有限性前提，重新证明有限切面以下的工作有限；保留当前节点规则的正时延有环图已由续篇处理。

## 15. 建议的学习顺序

第一次读第 1--3 节：

1. 独立写出一个 $B_{v,\theta}(E,M)$，确认它怎样保留消息来源，又怎样按目标与时间取出纤维。

第二次读第 4--7 节：

2. 给定一个区域的旧节点状态、旧选择历史与各节点纤维，依次写出 $C,h,\widetilde q,d,(A,(c_v)_{v\in C},y'),O,q^{\mathrm{cmp}},q'$。
3. 手算第 7.1--7.5 节，解释为什么 $m_0,m_1$ 的发送时间不同而到达时间相同。
4. 用第 7.6--7.9 节检查三个区别：描述量读取与状态采用、节点状态与选择历史、本次计算快照与下一持久状态。再说明局部控制量为什么必须由选择函数显式传给节点。

第三次读第 8--9 节：

5. 从式 (33)--(36) 独立重证引理 2 与定理 3。
6. 构造一个完整时间纤维为空的节点，说明为什么仍需该节点的前沿越过 $\theta$，才能把它从本次候选中排除。随后区分“已固定的规范结果”和“这份结果的合法暴露次序”。

第四次读第 10--12 节：

7. 为第 7.1--7.5 节列出 $P,S,U,F$ 事件顶点，并检查节点状态边、选择历史边和消息边都增加式 (43)。
8. 选择一个切面 $b$，写出 $W_b$，检查丢掉它会使哪个未来完整时间纤维缺元素。
9. 最后研究式 (47)--(48)，区分输入已经关闭、联合求值保持结果与联合算法高效这三个命题。

## 16. 可选的相关材料

本文的数学定义不依赖下列材料。只有在已经能够独立手算第 7 节后，才建议按目的查阅：

- [[timed-dag-chunk-prefill-learning-note|以本文为唯一前置的分块预填充、时间块与严格分层区域续篇]]；
- [[positive-delay-graph-finite-cut-learning-note|以本文为唯一前置的正时延有环图、有限切面与强连通分量续篇]]；
- [[settlegraph-learning-note|单次结算 SettleGraph 入门与向本文的受限嵌入]]；
- [[semantics-anchor|TIDE 语义锚点]]。

这些材料中的同名词不能反过来改写本文公式；若两份文档要建立关系，必须给出从一边全部数学坐标到另一边全部数学坐标的函数或关系。

## 附录 H：HST 是附加反向规则，不是普通导数（可选）

正文只定义前向函数值、状态、消息与选择，没有定义训练算法。本附录只解释第 7.8 节向量发送例子的一个常用训练约定；不把它扩张成任意图的训练理论。阅读这里需另熟悉可微标量函数与链式法则，不影响正文的阅读前提。

固定一个已经被选中的节点，令 $h,g\in\mathbb R^n$、$p\in[0,1]$，并给定实常数 $\zeta$。符号 $\operatorname{sg}$ 表示如下附加计算约定：前向把输入原样返回，反向把传给该输入的贡献规定为零。定义：

$$
\rho=1+\zeta(p-\operatorname{sg}(p)),
\qquad
\widehat g=h+\rho(g-h).
$$

前向恒有 $\rho=1$，所以 $\widehat g=g$，与 HARD 完全相同。只按普通实函数看，$p\mapsto p-p$ 的导数是零；这里绝不把它的普通导数改称 $\zeta$。所谓 HST，是为这段表达式另配一个**替代反向规则**。

具体地，设 $u\in\mathbb R^n$ 是从输出传回的向量，内积定义为 $\langle u,w\rangle=\sum_{k=1}^n u_kw_k$。暂把 $h,g,p$ 视为独立输入，上述约定规定传回三者的贡献为：

$$
\bar h=0,\qquad \bar g=u,\qquad
\bar p=\zeta\langle u,g-h\rangle.
$$

再沿生成 $g$ 与 $p$ 的已声明可微运算传播这些贡献。即使 $g$ 本来由 $h$ 算出，$\bar h=0$ 也只指发送表达式的直接路径，不能删除经 $g$ 返回 $h$ 的贡献。离散选中集合本身不按普通导数求导；是否以及怎样给未被选节点梯度、历史写回是否截断梯度，都必须另行声明。

SOFTP 的前向是 $h+p(g-h)$，已经不同于 HARD/HST；在把三个输入视为独立变量时，它有通常的局部导数，不能与上面的替代规则混用。正文的执行顺序、切面继续或 chunk 前向等价也不自动证明反向等价；要比较训练，须固定同一替代规则、状态展开与梯度截断位置，再比较相应反向计算。

## 附录 S：计算机系统词汇与正文数学对象的对应（可选）

本附录只做翻译，不增加正文定理的前提。系统词的含义依赖所选模型；以下对应只适用于本文已经定义的数学对象。

> [!info]- S.1　semantics、logical time、stage 与 visibility
> **semantics（语义）**对应固定规格下由定理 1 唯一确定的映射 $x\mapsto\mathcal T_x$。两个过程“语义相同”表示它们得到相同的 $\mathcal T_x$，或得到事先明确指定的同一投影。
>
> **logical time（逻辑时间）**对应第 2.1 节的 $\theta\in\mathbb N$；内部消息的逻辑到达时间由式 (7) 定义。**machine time / wall clock（机器时间、墙钟时间）**若需记录，要另加坐标 $w\in\mathbb R_{\ge0}$。
>
> 若要区分节点逻辑耗时与边上传递耗时，可以另给 $d_{\mathrm{node}}:V\to\mathbb N$ 与 $d_{\mathrm{edge}}:A\to\mathbb N$，使
> $$
> \delta(a)=d_{\mathrm{node}}(\operatorname{src}(a))+d_{\mathrm{edge}}(a)>0.
> $$
> 正文只使用总和，因此这项分解不改变递归。消息的发送时间仍是源节点事件的逻辑时间；若要另行记录节点完成时间或外部值的逻辑可用时间，还需增加相应坐标或输出端口时延。
>
> **stage（阶段）**对应第 8.1 节的索引 $n$。它排列阶段化暴露轨迹，不是逻辑时间或墙钟时间。
>
> 外部输入记录的 **visibility（可见）**对应 $\alpha_E(e)\le n$，内部消息的 **visibility / publication（可见、公开）**对应 $\alpha_M(m)\le n$。消息还须满足式 (40) 的源 $F$ 作用约束。外部输出记录在阶段 $n$ 已经确定对应 $z\in Z_n^F$。
>
> 本文只有一个全局 $H_n$。若不同消费者可以在不同阶段得到同一消息，必须另加消费者坐标并定义 $H_{c,n}$；这不是当前模型的一部分。

> [!info]- S.2　message、port、bucket 与 channel identity
> **message（消息）**的类型是 $m=(\mathrm{msg},\eta,a,y)\in\mathsf{Msg}$；一次输入实际产生的消息构成 $M^*\subseteq\mathsf{Msg}$。payload 只对应 $y$，发送时间 $\eta$ 和边 $a$ 也是消息坐标。
>
> 在完整语义中，**produced / generated（产生）**对应 $m\in M^*$；在阶段 $n$，源事件已经完成对应 $m\in M_n^{\mathrm{src}}$。它仍不推出 $m\in H_n$。
>
> **input/output port（输入、输出端口）**对应 $\mathsf I,\mathsf O$ 与式 (2)。端口不向拓扑中增加节点；多个端口可以属于同一节点。
>
> **bucket / input bucket / time bucket（输入桶、时间桶）**严格对应式 (8) 的纤维 $B_{v,\theta}(E,M)$。在完整语义里，式 (21) 把它实例化为 $B_{v,\theta}$。
>
> **channel identity（通道身份）**对应内部消息保留的边坐标 $a$；外部输入来源对应端口坐标 $i$。$\operatorname{Agg}_v$ 可以区分它们。数组或队列只是这些有限集合的一种编码。

> [!info]- S.3　region、candidate、selector、active 与 Top-K
> **region（区域）**只对应式 (6) 的节点子集 $\mathcal R_j$。它不持有普通消息和节点状态；由 $j$ 索引的 selector 另有选择历史 $y_j^\theta$。
>
> **candidate（候选）**在完整记录中对应式 (22) 的 $\mathcal C_{j,\theta}$；阶段 $n$ 暂得的候选是第 8.6 节的 $\mathcal C^{(n)}_{j,\theta}$。式 (38) 成立时二者才相等。
>
> **selector（选择器）**对应式 (14) 的一步全函数族。一次调用由区域索引 $j$、完整候选集合 $C$、旧选择历史 $y$、逻辑时间 $\theta$ 与带节点坐标的描述量族 $(d_v)_{v\in C}$ 共同确定；其中 $j,C$ 索引所调用的函数族成员，后三项是该成员的普通自变量。
>
> **selector-history（选择历史）**对应 $y_j^\theta\in Y_j$。它可以编码累计激活次数、移动平均、最近激活位置或预算，但不是程序日志，也不必逐项保留过去。若 $Y_j$ 为单点集，就没有有效历史信息。
>
> **local control / gate（局部控制、门控量）**对应 $c_{v,\theta}\in\mathsf C_v$。`Full` 只读取本节点控制量；它不直接读取整个 region 的描述量或控制族。控制可以在 `Next` 或 selector-history 中产生持久影响，但控制量本身不自动成为 continuation 坐标。
>
> **active node / selected node（激活、选中节点）**对应 $v\in\mathcal A_{j,\theta}$。只有这类节点应用式 (26)；派生集合 $\mathsf{Selected}_n$ 索引已经进入 $\mathsf S_n$ 的区域选择事件。
>
> **Top-K**只是式 (14) 的一种实例。若分数相等，固定平局规则属于函数定义的一部分。
>
> **routing（路由）**可以从实际 active sets 与式 (27) 真正产生的边消息导出。selector 不修改固定边集合 $A$。

> [!info]- S.4　content、pre、post、SD、BO 与 expensive compute
> `content` 对应 $\tau_j=0$ 与式 (11) 的 $d^0$；`pre` 对应 $\tau_j=-$ 与 $d^-$；`post` 对应 $\tau_j=+$ 与 $d^+$。
>
> `SD`（selected-dispatch）对应 $\kappa_j=0$，即式 (16) 中只有 active nodes 采用候选新状态。
>
> `BO`（broadcast-observe）对应 $\kappa_j=1$，即全部 candidates 采用候选新状态，但仍只有 active nodes 执行完整输出函数。
>
> 因此 `content/pre/post` 决定 selector 读取什么；`SD/BO` 决定哪些候选状态用于本次计算。下一持久状态还由 `Next` 单独指定；只有它返回计算快照时，采用与最终保存才是同一状态。
>
> **proposal** 对应候选新状态 $\widetilde q$。**state commit（状态提交）**对应同时给出式 (25a) 本次快照与式 (25) 下一持久状态的作用 $U_{v,\theta}$；选择作用 $S_{j,\theta}$ 同时确定 active set、局部控制族与下一 selector-history。派生的 $\mathsf{Completed}_n$ 要求 $U$ 已进入，并要求 active 坐标的 $F$ 也已进入。**expensive compute / NodeCompute** 由分块教材的成本标记确定，可以对应 $U$、$F$ 或二者的联合块。

> [!info]- S.5　seal、frontier、watermark、closure 与 ready
> **seal（封闭下界）**对应式 (33) 的 $\sigma$。例如边 seal 为 $b$ 精确表示 $M^*(a,<b)\subseteq H_n$，不是“当前队列为空”。
>
> 系统中的 seal 控制记录对应 $\sigma_n$ 到 $\sigma_{n+1}$ 的变化；正文只规定该值何时有效，不规定控制记录的编码或证明算法。
>
> **node frontier** 对应式 (35)；**region frontier** 对应式 (36)。区域前沿是成员前沿的最小值。
>
> **closed / closure（关闭）**对应式 (37) 的纤维相等或式 (38) 的候选集合相等。seal 是推出关闭的前提，二者不是同一个对象。
>
> **watermark** 若表示“输入确定到哪里”，对应某个 $\sigma$ 或 $\lambda$；若表示“节点完成到哪里”，对应式 (41) 的 $\operatorname{DoneTo}_n(v,b)$。必须注明采用哪一种。
>
> **ready（就绪）**对节点状态对应较早 $U$ 全部属于 $\mathsf U_n$，对选择历史对应较早 $S$ 全部属于 $\mathsf S_n$；区域选择还要满足第 9.2 节的纤维关闭与完整 $P$ 描述族条件。
>
> **barrier（屏障）**是实现等待这些数学条件成立的位置，并不要求所有处理器同时停止。

> [!info]- S.6　identifier、schema 与 semantic coordinate
> **identifier / ID（标识符）**在正文中首先就是集合元素本身，例如 $v\in V$、$a\in A$。若程序需要整数，可为本文任一有限对象集合 $X$ 固定单射 $\operatorname{id}_X:X\to\mathbb N$。单射保证不同对象有不同整数。
>
> **schema（数据形状约定）**对应一个积集合以及各坐标的所属集合。例如 $\mathsf{Msg}$ 规定消息具有标签、发送时间、边和值四个坐标。字段名可以改变；删除边坐标则会改变正文对象。
>
> **semantic coordinate（语义坐标）**是在已经说明对象类型后，用于唯一指出该类型中数学对象的有序组。节点事件的语义坐标是 $(v,\theta)$，区域选择事件的语义坐标是 $(j,\theta)$，函数作用事件还包含 $\mathrm{prep}$、$\mathrm{select}$、$\mathrm{adopt}$ 或 $\mathrm{full}$ 标签；其中 $\mathrm{select}$ 同时产生 active set 与下一选择历史。输出位置的坐标是 $(\theta,o)$。
>
> 若三类事件需要共享一个整数 ID 空间，先取三个两两不同的标签 $\mathrm{node},\mathrm{sel},\mathrm{act}$，并定义有限编码域：
> $$
> \begin{aligned}
> \mathsf{EventRef}_x
> ={}&(\{\mathrm{node}\}\times\mathcal E_x^{\mathrm{node}})\\
> &\cup(\{\mathrm{sel}\}\times\mathcal E_x^{\mathrm{sel}})\\
> &\cup(\{\mathrm{act}\}\times\mathscr V_x^{\mathrm{ev}}).
> \end{aligned}
> $$
> 再固定单射：
> $$
> \operatorname{id}_{\mathrm{event},x}:
> \mathsf{EventRef}_x\to\mathbb N.
> $$
> 外层标签先把不同层次的事件变成不同编码对象，随后单射保证不同编码对象不会共用一个整数。这里 $\mathsf{EventRef}_x$ 只是一项编码域，不把三种事件合并成同一种语义作用。这项编码固定在输入 $x$ 的完整事件集合上；阶段 $n$、线程编号、墙钟时刻和本次分块编号都不在编码函数的输入中，所以改变它们不会改变 ID。若还要求不同输入之间共享稳定整数 ID，就需要另外在一个与输入无关的全局坐标域上固定编码；这不是本文语义的前提。

> [!info]- S.7　chunk、prefill、decode、packing 与 fast path
> **input chunk（输入分块）**可以对应 $E_{n+1}\setminus E_n$；同一批进入 $H$ 的消息对应 $H_{n+1}\setminus H_n$。**compute chunk（计算分块）**则是某类事件集合的划分。两者必须区分。
>
> 若 $\mathscr E=\mathscr E_1\cup\cdots\cup\mathscr E_k$ 且各 $\mathscr E_i$ 两两不交，那么 $(\mathscr E_i)$ 是事件集合 $\mathscr E$ 的一种分块；还须说明 $\mathscr E$ 采用节点事件、区域选择事件还是函数作用事件。
>
> **prefill** 通常表示一次联合处理许多已知输入位置；**decode** 通常表示逐个或小批增加输入位置。要声称二者等价，必须比较第 6 节的节点状态、selector-history、内部消息、所有输出端口、候选集合和 active sets，而不只是最后一个张量。
>
> **packing / packed attention** 可以实现式 (48) 的逐坐标联合等式，也可以实现第 12.4 节的因果状态等式。seal 给出合法输入域，联合等式给出逐坐标精确性。
>
> **fast path** 可以在额外图结构和代数条件成立时使用联合函数；一般路径仍须得到第 6 节的规范记录。若还要比较分阶段进展，则另须投影到第 9 节的合法阶段化暴露轨迹。

> [!info]- S.8　runtime、scheduler、workspace、commit 与 trace
> **compute / evaluate（计算、求值）**在正文中只指取已给全函数的函数值；本文没有定义指令集或成本函数。因此“可求值”不包含时间复杂度结论。
>
> **runtime / executor（运行时、解释器）**不是正文定义的数学对象。第 9.1 节的轨迹已经是固定 $\mathcal T_x$ 上的阶段化暴露，不能单凭“程序产生了这样一条轨迹”证明程序算对；实现与规范完整记录之间还必须给出附录 S.9 的全局精化关系。
>
> **scheduler（调度器）**在通过精化投影以后，对应选择哪些合格作用进入第 9.2--9.3 节的 $\Delta\mathsf P_n,\Delta\mathsf S_n,\Delta\mathsf U_n,\Delta\mathsf F_n$。投影后的调度只改变阶段秩，保留 $\mathcal T_x$ 的规范值。
>
> **workspace（临时工作区）**可以保存第 9.4 节提前求出的 $h,\widetilde q,d$。临时变量不等于式 (25) 确定的下一持久状态。
>
> **event completion（事件完成）**对应第 9.9 节的派生成员关系 $(v,\theta)\in\mathsf{Completed}_n$：$U$ 已进入，且 active 坐标的 $F$ 已进入。**completed to $b$（完成到 $b$）**对应式 (41) 的复合谓词。消息进入 $H_n$ 具有独立的阶段秩 $\alpha_M$。
>
> **atomic commit（原子提交）**在本文中的最小要求是：节点状态后继只能读取式 (25) 确定以前的 $q_v^\theta$ 或确定以后的 $q_v^{\theta+1}$；选择历史后继只能读取式 (24) 确定以前的 $y_j^\theta$ 或确定以后的 $y_j^{\theta+1}$。二者都不能读取规格未定义的中间状态。这不声称使用某条特定处理器原子指令。
>
> **semantic trace（语义记录）**对应唯一的 $\mathcal T_x$；**staged exposure trace（阶段化暴露轨迹）**对应第 9.1 节的递增子集与标签限制，同一 $x$ 可以有多条。**execution trace（执行记录）**属于实现另行定义的记录集合。普通语义精化只保留规范坐标；若要比较阶段化暴露，还须另给保留阶段秩与递增子集的精化投影。两种投影都在附录 S.9 定义。

> [!info]- S.9a　continuation、checkpoint 与 resume
> **continuation** 对应式 (46) 的 $Q_b$，其中同时含节点状态、selector-history 与跨界消息。只为继续未来计算而保存的 **checkpoint** 可以是 $Q_b$ 的某种可保存编码。
>
> 若保存与读取函数分别为 $\operatorname{save}$ 和 $\operatorname{load}$，无损编码至少要求：
> $$
> \operatorname{load}(\operatorname{save}(Q_b))=Q_b.
> $$
>
> **resume** 对应从相同 $Q_b$ 与 $E_{\ge b}$ 再执行未来递归。定理 6 给出它与一次算完相同的数学目标。$Q_b$ 并不是规范左前缀 $\mathcal T_{x,<b}$ 的完整编码；若要重建整个语义记录，还须保存该前缀或保存足以重算它的数据。若只要求拼回完整外部输出记录，则至少还须保存 $Z_{<b}$，或保存这些输出已被可靠接收的等价证据。

> [!info]- S.9b　普通语义 refinement
> **refinement（实现精化）**把第 6.4 节的块投影扩展到完整执行记录。先给出调度集合 $\mathsf{Sched}$、程序记录集合 $\mathsf{ImplTrace}$，以及一个包含全部规范记录和可能错误的同形记录的集合 $\mathsf{SemTrace}$。再给出函数：
> $$
> \operatorname{Run}:
> \left(\prod_{i\in\mathsf I}P^{[L_i]}\right)
> \times\mathsf{Sched}
> \to\mathsf{ImplTrace}.
> $$
> 令 $\operatorname{Legal}_{\mathrm{impl}}(x)\subseteq\mathsf{Sched}$ 是实现一侧的容许调度谓词，例如检查类型、因果等待与 seal 证据；它的定义不能预先把“程序记录投影后已经等于规范记录”当作条件。定义相应的程序记录子集：
> $$
> \mathsf{ImplTrace}_{\mathrm{legal}}
> =\{\operatorname{Run}(x,s)\mid
> x\in\prod_{i\in\mathsf I}P^{[L_i]},
> s\in\operatorname{Legal}_{\mathrm{impl}}(x)\}.
> $$
> 只需在这个子集上给出删除线程、墙钟和阶段等实现坐标的普通语义投影：
> $$
> \Pi:\mathsf{ImplTrace}_{\mathrm{legal}}\to
> \mathsf{SemTrace}.
> $$
> 其中每个 $\mathcal T_x$ 都属于 $\mathsf{SemTrace}$。这里的 $\Pi$ 必须是预先固定的遗忘投影：它保留全部规范坐标及其原值，只删除线程、墙钟、缓存、阶段等额外实现坐标。它不得调用参考语义、重新执行递归，或按照输入修补程序记录中的错误值。

> [!info]- S.9c　安全性与可执行性
> 下式是实现的**安全性**目标：
> $$
> \forall x\in\prod_{i\in\mathsf I}P^{[L_i]},\quad
> \forall s\in\operatorname{Legal}_{\mathrm{impl}}(x),
> \qquad
> \Pi(\operatorname{Run}(x,s))=\mathcal T_x.
> $$
> 它单独成立时可能只是因为没有容许调度。还须另列**可执行性**条件：
> $$
> \forall x\in\prod_{i\in\mathsf I}P^{[L_i]},
> \qquad
> \operatorname{Legal}_{\mathrm{impl}}(x)\ne\varnothing.
> $$

> [!info]- S.9d　staged refinement
> $\Pi$ 的值不含第 8--9 节的阶段秩。若还要验证执行记录的分阶段进展，另给一个包含阶段信息的记录集合 $\mathsf{StageTrace}$、投影
> $$
> \Pi_{\mathrm{stage}}:
> \mathsf{ImplTrace}_{\mathrm{legal}}
> \to\mathsf{StageTrace},
> $$
> $\Pi_{\mathrm{stage}}$ 同样必须预先固定；它只比 $\Pi$ 保留更多已有的阶段坐标，不得补写、重算或修正程序记录以制造一条合法轨迹。
>
> 与第 9.1 节的关系记号一致，令：
> $$
> \mathsf{LegalExposure}(x)
> =
> \left\{
> \begin{aligned}
> \boldsymbol\Xi\in\mathsf{StageTrace}\ \bigm| {}&
> \operatorname{LegalExposure}_{\mathcal T_x}(\boldsymbol\Xi),\\
> &\boldsymbol\Xi\text{ 满足第 9.1 节的完整性条件}
> \end{aligned}
> \right\}.
> $$
> 因而这里不是把“合法”重新留给实现解释，而是直接取正文已经定义的轨迹关系。要求：
> $$
> \begin{aligned}
> &\forall x\in\prod_{i\in\mathsf I}P^{[L_i]},\quad
> \forall s\in\operatorname{Legal}_{\mathrm{impl}}(x):\\
> &\qquad
> \Pi_{\mathrm{stage}}(\operatorname{Run}(x,s))
> \in\mathsf{LegalExposure}(x).
> \end{aligned}
> $$
> 这个条件保留阶段信息，不能用普通语义投影 $\Pi$ 代替。

> [!info|page]- S.10　SettleGraph 的可选对应
> 本块帮助比较 SettleGraph 受限规格与本文，不是阅读正文的前置知识。核心 SettleGraph 定义与嵌入见本地 [[settlegraph-learning-note|SettleGraph 教材]]；实验仓库负责声明它采用的具体实例与额外条件。
>
> 本块中的 Token 是 SettleGraph 给一个外部输入位置使用的标签，不是本文的内部消息或节点事件。
>
> 一份完整翻译至少需要逐项给出：
>
> - receiver 到 $v\in V$ 的映射；
> - graph ingress/egress 到 $\mathsf I,\mathsf O$ 的映射；
> - receiver state 到 $S_v$ 的映射；
> - SettleGraph 的聚合序列及函数到 $B_{v,\theta}$ 与 $\operatorname{Agg}_v$ 的映射，包括入口、原边和终端节点标签的对应；
> - selection region 到 $\mathcal R_j$ 的映射；
> - active set、局部控制、SD/BO、本次计算快照、Next、NodeCompute 与 Emit 到式 (24)--(28) 的映射；
> - selector-history 的 owner、初态、更新与切面状态到 $Y_j,y_j^{\mathrm{init}},\operatorname{SelStep}_{j,C},y_j^b$ 的映射；
> - 已生成 `DATA` 到 $m\in M_n^{\mathrm{src}}$、已进入接收侧输入的 `DATA` 到 $m\in H_n$ 的映射；
> - 边 $a$ 上到达时间小于 $b$ 的全部槽位均已返回 `DATA` 或 `CLOSED`，且其中的 `DATA` 均已可见，由此推出 $M^*(a,<b)\subseteq H_n$；单个 Token 的 `CLOSED` 不能独自证明整个时间前缀已封闭。
>
> SettleGraph 对每个 Token、每个 region 等待该 Token 的所有相关边结算；本文允许不同输入位置通过逻辑时间映射落入同一个 $(v,\theta)$ 或 $(j,\theta)$。若一个 SettleGraph profile 没有有效 selector-history，可把它映为单点集合 $Y_j$。HARD 与 SOFTP 可按第 7.8 节映射；HST 的前向等于 HARD，但它的替代反向规则须另按附录 H 比较，不能由前向映射推出训练等价。要证明前者是后者的受限情形，必须给出一种时间编码，使不同 Token 不会发生被禁止的同刻汇合，并比较完整节点状态、selector-history、局部控制、计算快照、消息、输出端口、候选集合和 active sets。本地 [[settlegraph-learning-note|SettleGraph 教材]] 给出核心实例的编码与证明；下游实验中的梯度规则、模型接入与自定义模块仍需各自的明确契约。
