---
type: mathematical-learning-note
status: active-learning
cssclasses:
  - textbook-math
tags:
  - tide
  - settlegraph
  - mathematics
  - learning-note
---

# 单次结算图 SettleGraph：从一次输入到有状态序列

本文面向第一次接触 TIDE 的数学读者，只需集合、函数、有限求和与数学归纳法。我们从一次输入出发，定义消息、记忆、共同选择和输出，再将多次结算连接成有状态序列。

SettleGraph 的消息边严格通向更高区域，边时延与输入日程共同使同一位置的消息在目标节点对齐。因此每个节点对每个输入位置至多处理一次。局部函数、参数顺序和返回值族与 [[timed-dag-region-selector-learning-note|TimedDAG]] 相同；第 10 节证明这是一项结构限制与时间重索引。

第 1--7 节给出定义和手算，第 8--10 节讨论衰减、继续与嵌入，第 11--12 节讨论联合求值和函数保持。系统术语与计算表示集中在附录 A。

## 1. 从一次共同选择开始

设两个节点 $a,b$ 都收到实数 $x$。它们各自产生描述量，一个区域据此选择至多一个节点做完整计算；节点 $c$ 接收其结果。

```text
         a ──┐
输入 x ──┤    ├── c ── 输出
         b ──┘
```

两条边 $a\to c,b\to c$ 始终属于固定图。一次只选择 $b$ 时，$a\to c$ 在该位置确定无值，$b\to c$ 的值由 $b$ 的完整输出函数决定。$c$ 收集两条边上实际存在的值；两条边的有值或无值结果都确定后，这组输入才完整。

节点还可以保存记忆。收到输入、采用新记忆、被选择做完整计算、留下下一次记忆，是分别定义的动作。下面沿这条顺序建立模型。

## 2. 固定集合、图与逻辑时间

### 2.1 值与带名字的坐标族

令 $\mathbb N=\{0,1,2,\ldots\}$，$[L]=\{0,\ldots,L-1\}$，并用 $[b,c)$ 表示满足 $b\le\theta<c$ 的自然数集合。固定非空承载值空间 $P$，另取 $\bot\notin P$，令

$$
P_\bot=P\cup\{\bot\}.
$$

$\bot$ 表示某个输出坐标确定不产生值。标量和向量教材例子分别取 $P=\mathbb R$ 和 $P=\mathbb R^d$。

全函数 $f:X\to Y$ 对每个 $x\in X$ 恰好给出一个值。若 $C$ 为有限集合，则 $\prod_{v\in C}D_v$ 是所有按名字 $v$ 索引的坐标族 $(d_v)_{v\in C}$；空积只含唯一空族 $()$。$Y^E$ 表示所有函数 $E\to Y$，$\mathcal P_{\mathrm{fin}}(X)$ 表示 $X$ 的有限子集。不同名字即使携带相同数值，仍是不同坐标。

### 2.2 消息图与端口

给定有限非空节点集 $V$、有限边集 $A$ 及函数

$$
G=(V,A,\operatorname{src},\operatorname{dst},\delta),
\qquad \operatorname{src},\operatorname{dst}:A\to V,
\qquad \delta:A\to\mathbb N_{>0}.
$$

边 $a$ 把源节点在时间 $\eta$ 产生的值送到目标节点的时间 $\eta+\delta(a)$。边的身份由 $a\in A$ 给出；允许不同边具有相同起点和终点。定义

$$
\operatorname{In}(v)=\{a:\operatorname{dst}(a)=v\},\qquad
\operatorname{Out}(v)=\{a:\operatorname{src}(a)=v\}.
$$

首尾相接的有限边列称为有向游走，允许停在一个节点的零长度游走。无重复节点的游走称为路径。非空闭游走含有有向环；没有有向环的图称为有向无环图，简称 DAG。

给定有限非空输入、输出端口集合 $\mathsf I,\mathsf O$，以及

$$
\gamma:\mathsf I\to V,\qquad \varepsilon:\mathsf O\to V.
$$

端口 $i$ 向 $\gamma(i)$ 提供外部输入，端口 $o$ 接收 $\varepsilon(o)$ 的外部输出。写

$$
\operatorname{InPort}(v)=\gamma^{-1}(\{v\}),\qquad
\operatorname{OutPort}(v)=\varepsilon^{-1}(\{v\}).
$$

端口可以属于任意节点；一个节点可以同时具有入边、输入端口、出边和输出端口。

### 2.3 区域划分与严格次序

固定有限非空集合 $J$ 与满射 $\rho:V\to J$。区域为

$$
\mathcal R_j=\rho^{-1}(\{j\}).
$$

各区域非空、两两不交且并为 $V$。节点各有私有状态；区域另有一份选择历史。

固定区域秩 $\ell:J\to\mathbb N_{>0}$，要求

$$
\operatorname{LegalRegionRank}_{G,\rho}(\ell)
\quad\Longleftrightarrow\quad
\forall a\in A:\quad
\ell(\rho(\operatorname{src}(a)))<\ell(\rho(\operatorname{dst}(a))).
\tag{1}
$$

于是区域内没有消息边，区域之间和节点图都无环。按秩递增排列区域，就得到每条边都向前的区域拓扑序。同秩区域可任意排序。

节点图无环还允许更一般的区域划分。例如路径 $u\to v\to w$ 中把 $u,w$ 放在同一区域，便不满足式 (1)。

### 2.4 输入位置与到达时间对齐

固定有限位置范围 $[L]$，$L\ge1$。对每个输入端口 $i$，固定 $L_i\ge1$ 和严格递增的位置映射

$$
t_i:[L_i]\to[L].
$$

端口可跳过某些位置；第 $k$ 个端口输入属于全局位置 $t_i(k)$。一次输入是 $x=(x_i)_{i\in\mathsf I}$，其中 $x_i:[L_i]\to P$。

令 $r_v=\ell(\rho(v))$，固定整数 $D>\max_v r_v$，并要求

$$
\begin{aligned}
\theta_{v,t}&=Dt+r_v,\qquad \theta_{j,t}=Dt+\ell(j),\\
\delta(a)&=r_{\operatorname{dst}(a)}-r_{\operatorname{src}(a)},\\
\iota_i(k)&=Dt_i(k)+r_{\gamma(i)}.
\end{aligned}
\tag{2}
$$

式 (1)--(2) 合称合法时间编码，记为 $\operatorname{LegalTimeCode}_{G,\rho}(\ell,D)$，其中端口日程也须满足式 (2)。边时延为正。同一区域的节点具有相同处理时间；同一位置的全部处理时间落在 $[Dt,D(t+1))$ 内。

对边 $a:u\to v$，有

$$
\theta_{u,t}+\delta(a)=\theta_{v,t}.
\tag{3}
$$

沿路径相加，中间秩抵消。因此从同一位置出发、到达同一节点的消息具有相同到达时间。这个等式是单次结算的时间条件。严格分层而时延任意时，两条路径可能在不同时刻到达，形成多次处理。

逻辑时间是函数自变量。改变 $D$ 或区域秩可能改变衰减、位置函数等数值；固定编码以后，求值次序可以改变，函数接收的时间保持不变。

## 3. 来源原子与时间纤维

### 3.1 三类带标签记录

取不同形式标签 $\mathrm{ext},\mathrm{msg},\mathrm{out}$，定义

$$
\begin{aligned}
\mathsf{Ext}&=\{(\mathrm{ext},i,k,p):i\in\mathsf I,k\in[L_i],p\in P\},\\
\mathsf{Msg}&=\{(\mathrm{msg},\eta,a,p):\eta\in\mathbb N,a\in A,p\in P\},\\
\mathsf{OutRec}&=\{(\mathrm{out},\theta,o,p):\theta\in\mathbb N,o\in\mathsf O,p\in P\}.
\end{aligned}
$$

外部记录 $e=(\mathrm{ext},i,k,p)$ 的目标、时间和值分别为 $\gamma(i),\iota_i(k),p$。内部消息 $m=(\mathrm{msg},\eta,a,p)$ 的发送时间、边、目标、到达时间和值分别为

$$
\operatorname{send}(m)=\eta,\quad \operatorname{edge}(m)=a,\quad
\operatorname{target}(m)=\operatorname{dst}(a),\quad
\operatorname{time}(m)=\eta+\delta(a),\quad \operatorname{value}(m)=p.
$$

输出记录的时间、端口和值由其后三个坐标给出，记为 $\operatorname{outtime},\operatorname{outport},\operatorname{outvalue}$。输出记录终止于图的外部。

令 $\mathsf{Atom}=\mathsf{Ext}\cup\mathsf{Msg}$，$\mathsf{Atom}_v=\{z\in\mathsf{Atom}:\operatorname{target}(z)=v\}$。实际外部输入为

$$
E_x=\{(\mathrm{ext},i,k,x_i(k)):i\in\mathsf I,k\in[L_i]\}.
$$

### 3.2 完整输入与聚合

给定已确定的消息集合 $M$，定义时间纤维

$$
B_{v,\theta}(E_x,M)
=\{z\in E_x\cup M:\operatorname{target}(z)=v,\ \operatorname{time}(z)=\theta\}.
\tag{4}
$$

它保留原子的端口或边身份以及时间。平行边上的相同值仍贡献两个不同原子。集合本身无排列次序；具体聚合可以按固定标签序读取它。

为每个节点固定独立的非空本地内容空间 $X_v$，以及全函数

$$
\operatorname{Agg}_v:\mathbb N\times\mathcal P_{\mathrm{fin}}(\mathsf{Atom}_v)\to X_v.
\tag{5}
$$

它在实际非空纤维上给出 $h_{v,\theta}=\operatorname{Agg}_v(\theta,B_{v,\theta})$。例如取 $P=\mathbb R$、$X_v=\mathbb R^2$，定义

$$
\operatorname{Agg}_v(\theta,B)
=\left(\sum_{z\in B}\operatorname{value}(z),\ |B|\right).
$$

节点同时取得值的和与来源数，后续函数可分别使用它们。若取 $X_v=P=\mathbb R^d$，则非空纤维的均值也是一个实例；令空集处返回零向量即得到全函数。空集处的补全值不产生节点事件。

在处理位置 $t$ 时，简写 $B_{v,t}=B_{v,\theta_{v,t}}$。定义

$$
\mathcal C_{j,t}=\{v\in\mathcal R_j:B_{v,t}\ne\varnothing\}.
\tag{6}
$$

它是区域的候选集合。一个纤维含多个原子时，该节点仍只出现一次。

## 4. 节点状态、描述量与区域选择

### 4.1 旧状态与候选新状态

每个节点另有非空空间 $S_v,D_v,\mathsf C_v$，分别表示节点状态、选择描述量与局部控制；初态为 $q_v^{\mathrm{init}}$。用方括号区分位置边界与逻辑时间：$q_v^{[t]}$ 是位置 $t$ 开始前的持久状态，$q_v^{[0]}=q_v^{\mathrm{init}}$。

固定

$$
\operatorname{Upd}_v:S_v\times\mathbb N\times X_v\to S_v.
$$

对候选 $v$ 定义

$$
h_{v,t}=\operatorname{Agg}_v(\theta_{v,t},B_{v,t}),\qquad
\widetilde q_{v,t}=\operatorname{Upd}_v(q_v^{[t]},\theta_{v,t},h_{v,t}).
\tag{7}
$$

候选新状态 $\widetilde q$ 是待采用的函数值。状态空间可取单点集 $\{*\}$，使节点不携带有效记忆。

### 4.2 描述量读取

固定全函数

$$
\begin{aligned}
\operatorname{Read}_v^0&:\mathbb N\times X_v\to D_v,\\
\operatorname{Read}_v^-,\operatorname{Read}_v^+&:S_v\times\mathbb N\times X_v\to D_v.
\end{aligned}
$$

区域选择模式 $\tau_j\in\{0,-,+\}$，并令

$$
d_{v,t}=\begin{cases}
\operatorname{Read}_v^0(\theta_{v,t},h_{v,t}),&\tau_j=0,\\
\operatorname{Read}_v^-(q_v^{[t]},\theta_{v,t},h_{v,t}),&\tau_j=-,\\
\operatorname{Read}_v^+(\widetilde q_{v,t},\theta_{v,t},h_{v,t}),&\tau_j=+.
\end{cases}
\tag{8}
$$

三种模式依次读取当前内容、旧记忆、包含本次更新的候选记忆。$+$ 模式本身只选择描述量的自变量。

### 4.3 选择返回的三个坐标

每个区域固定非空历史空间 $Y_j$、初态 $y_j^{\mathrm{init}}$、容量 $1\le K_j\le|\mathcal R_j|$。令 $y_j^{[0]}=y_j^{\mathrm{init}}$，并对 $C\subseteq\mathcal R_j$ 定义

$$
\mathsf{Allowed}_{j,C}=\{A'\subseteq C:|A'|\le K_j\}.
$$

选择全函数族为

$$
\operatorname{SelStep}_{j,C}:Y_j\times\mathbb N\times\prod_{v\in C}D_v
\to\mathsf{Allowed}_{j,C}\times\prod_{v\in C}\mathsf C_v\times Y_j.
\tag{9}
$$

空候选规定 $\operatorname{SelStep}_{j,\varnothing}(y,\theta,())=(\varnothing,(),y)$。实际选择为

$$
(\mathcal A_{j,t},(c_{v,t})_{v\in\mathcal C_{j,t}},y_j^{[t+1]})
=\operatorname{SelStep}_{j,\mathcal C_{j,t}}
\left(y_j^{[t]},\theta_{j,t},(d_{v,t})_{v\in\mathcal C_{j,t}}\right).
\tag{10}
$$

激活集合 $\mathcal A$ 是候选子集，大小至多为 $K_j$，允许为空。每个候选都得到自己的控制 $c_v$，包括未激活候选。下一历史可以记录本次选择及控制摘要；它由旧历史、时间和描述量族确定。

例如 $D_v=\mathbb R$ 时，可在正描述量中选至多 $K_j$ 个最大者，平票按固定节点全序。取 $Y_j=\mathsf C_v=\{*\}$ 就得到无历史、无额外控制的例子。每项具体规则都须对整个声明定义域给出唯一结果。

## 5. 本次计算与下一次记忆

### 5.1 状态采用

区域固定 $\kappa_j\in\{0,1\}$，定义

$$
O_j(C,A')=\begin{cases}A',&\kappa_j=0,\\C,&\kappa_j=1.\end{cases}
$$

对候选 $v\in\mathcal R_j$，计算快照为

$$
q^{\mathrm{cmp}}_{v,t}=\begin{cases}
\widetilde q_{v,t},&v\in O_j(\mathcal C_{j,t},\mathcal A_{j,t}),\\
q_v^{[t]},&\text{其余候选}.
\end{cases}
\tag{11}
$$

$\kappa_j=0$ 让激活者采用更新，$\kappa_j=1$ 让全部候选采用更新。完整输出读取这份快照。

### 5.2 下一持久状态

固定全函数

$$
\operatorname{Next}_v:S_v\times S_v\times\mathbb N\times X_v\times\{0,1\}\times\mathsf C_v\to S_v.
$$

前两个参数依次为旧状态与计算快照。对候选令 $a_{v,t}=\mathbf1[v\in\mathcal A_{j,t}]$，并定义

$$
q^{\mathrm{next}}_{v,t}
=\operatorname{Next}_v(q_v^{[t]},q^{\mathrm{cmp}}_{v,t},\theta_{v,t},h_{v,t},a_{v,t},c_{v,t}),
\qquad q_v^{[t+1]}=q^{\mathrm{next}}_{v,t}.
\tag{12}
$$

非候选保持 $q_v^{[t+1]}=q_v^{[t]}$，没有本次 Next 作用。默认延续为

$$
\operatorname{Next}_v(q,q^{\mathrm{cmp}},\theta,h,a,c)=q^{\mathrm{cmp}}.
$$

也可在 $a=1$ 时返回声明的重置值，其他情形保留快照。本次完整计算仍读取重置前的 $q^{\mathrm{cmp}}$。Next 的定义与求值只使用已声明自变量，不调用或重算 Full。

### 5.3 完整输出族

固定全函数

$$
\operatorname{Full}_v:S_v\times\mathbb N\times X_v\times\mathsf C_v
\to(P_\bot)^{\operatorname{Out}(v)}\times(P_\bot)^{\operatorname{OutPort}(v)}.
\tag{13}
$$

只有激活节点应用它：

$$
(f^A_{v,t},f^O_{v,t})
=\operatorname{Full}_v(q^{\mathrm{cmp}}_{v,t},\theta_{v,t},h_{v,t},c_{v,t}).
\tag{14}
$$

每条出边和输出端口具有独立坐标，可以取不同值，也可以取 $\bot$。若 $f^A_{v,t}(a)=p\in P$，产生 $(\mathrm{msg},\theta_{v,t},a,p)$；若 $f^O_{v,t}(o)=p\in P$，产生 $(\mathrm{out},\theta_{v,t},o,p)$。$\bot$ 坐标不产生记录。

因此某节点可以激活却不发送任何消息，也可以只输出到某个端口。Full 只给出这两个值族；节点下一状态和区域下一历史已在此前确定。

### 5.4 广播是一个局部实例

给定载荷函数 $\operatorname{EmitValue}_v:S_v\times\mathbb N\times X_v\times\mathsf C_v\to P$，定义

$$
\begin{aligned}
\Delta_v(p)&=\big((p)_{a\in\operatorname{Out}(v)},(p)_{o\in\operatorname{OutPort}(v)}\big),\\
\operatorname{Full}_v&=\Delta_v\circ\operatorname{EmitValue}_v.
\end{aligned}
\tag{15}
$$

该节点激活后向每个去向发送同一载荷。例如在 $P=X_v=\mathbb R^d$、$\mathsf C_v=[0,1]$ 时，可以取

$$
\operatorname{EmitValue}_v(q,\theta,h,c)=h+c(G_v(q,\theta,h)-h).
$$

未激活候选保有控制量，但不应用 Full。

### 5.5 接收驱动的记忆实例

令状态保存带时间戳的有限键值序列，取 $\kappa_j=1$ 与默认 Next。每个候选先由 Upd 形成记忆，Read 从候选记忆给出描述量，选择决定哪些节点应用 Full。于是

$$
q^{\mathrm{cmp}}_{v,t}=\widetilde q_{v,t}=q_v^{[t+1]}.
$$

未激活候选同样保留接收更新。

## 6. 一个输入的完整结算

### 6.1 区域递归与边、端口结果

给定位置 $t$ 的外部记录与旧状态 $(q_v^{[t]})_v,(y_j^{[t]})_j$。按区域拓扑序依次：

1. 汇集本区域在 $\theta_{j,t}$ 的外部记录与已由父区域产生的消息，形成完整纤维及候选集合。
2. 对候选应用 Agg、Upd 和所选 Read，得到 $(h,\widetilde q,d)$。
3. 应用 SelStep，得到 $(\mathcal A,(c_v)_{v\in\mathcal C},y^{[t+1]})$。
4. 对候选确定 $(q^{\mathrm{cmp}},q^{\mathrm{next}})$；非候选状态保持。
5. 对激活节点求 Full，展开消息与输出记录。

空候选区域保持历史。非空候选但空激活时，准备、选择及状态作用仍然发生。

对每条边与每个端口定义总有值的结果槽

$$
\begin{aligned}
z_{a,t}&=\begin{cases}
f^A_{v,t}(a),&v=\operatorname{src}(a)\text{ 激活},\\
\bot,&\text{其余情形},
\end{cases}\\
b_{o,t}&=\begin{cases}
f^O_{v,t}(o),&v=\varepsilon(o)\text{ 激活},\\
\bot,&\text{其余情形}.
\end{cases}
\end{aligned}
\tag{16}
$$

它们完整记录本位置每个去向的有值或无值结果。应用可以另选读出函数 $\operatorname{ReadOut}_t:(P_\bot)^{\mathsf O}\to Y$，从这些端口结果得到任务输出。

### 6.2 有限性与唯一性

> [!theorem] 定理 1：单次结算
> 固定第 2--5 节的规格、一次输入及初态。每个位置经过有限次函数作用得到唯一记录；每节点每位置至多有一个准备、状态作用及完整输出作用，每区域每位置至多有一个选择作用。任何区域拓扑序得到相同结果。

**证明。** 按区域顺序归纳。式 (1) 使当前区域的全部父边来自先前区域，式 (3) 使这些消息恰在本位置的处理时间到达，外部记录也按式 (2) 对齐。当前完整纤维和旧状态因此已确定，全部局部全函数依次给出唯一结果。一个区域只处理一次，节点和区域有限，函数作用总数有限。换用另一拓扑序时，逐区域的自变量与既定记录相同，函数值因而相同。对位置继续归纳即得到整个序列。$\square$

这里的有限性计量抽象函数作用。函数的实际工作量由具体算法决定。

### 6.3 规范作用结果与调用标签

在位置 $t$，候选节点具有作用 $P_{v,\theta_{v,t}}=(\mathrm{prep},v,\theta_{v,t})$、$U_{v,\theta_{v,t}}=(\mathrm{adopt},v,\theta_{v,t})$；非空候选区域具有 $S_{j,\theta_{j,t}}=(\mathrm{select},j,\theta_{j,t})$；激活节点另有 $F_{v,\theta_{v,t}}=(\mathrm{full},v,\theta_{v,t})$。其规范结果为

| 作用 | 返回值 |
|---|---|
| $P_{v,\theta_{v,t}}$ | $(h_{v,t},\widetilde q_{v,t},d_{v,t})$ |
| $S_{j,\theta_{j,t}}$ | $(\mathcal A_{j,t},(c_{v,t})_{v\in\mathcal C_{j,t}},y_j^{[t+1]})$ |
| $U_{v,\theta_{v,t}}$ | $(q^{\mathrm{cmp}}_{v,t},q_v^{[t+1]})$ |
| $F_{v,\theta_{v,t}}$ | $(f^A_{v,t},f^O_{v,t})$ |

记录一次准备调用的输入和输出，得到

$$
\operatorname{Lab}_P(v,\theta_{v,t})
=(B_{v,t},q_v^{[t]},h_{v,t},\widetilde q_{v,t},d_{v,t}).
$$

其后三项是 $P$ 的结果；旧状态是调用输入，供后续 Next 再次读取。仅保存三项结果一般无法恢复旧状态。选择调用标签同样保留 $(\mathcal C,y^{[t]},(d_v))$ 与三项结果。$U,F$ 的标签直接使用上表结果。

完整记录 $\mathcal T_x^{\mathrm S}$ 包含全部纤维、候选与激活集合、这四类实际作用及其结果、各位置边界状态和历史、全部消息、输出记录及式 (16) 的槽。未发生作用由支持集的缺席表示。由定理 1，这份记录唯一。

### 6.4 单输入与终端读出的边界实例

令 $V_{\mathrm{in}}=\{v:\operatorname{In}(v)=\varnothing\}$、$V_{\mathrm{out}}=\{v:\operatorname{Out}(v)=\varnothing\}$。有限 DAG 中每个节点都位于某条入口至终端的路径上：不断向前或向后走，有限性与无环性保证在相应边界停止。

为每个入口设输入端口 $i_v$，各端口都在位置 $t$ 注入同一个 $x(t)$；为每个终端设输出端口 $o_v$。取广播式 (15)，令

$$
\mathcal M_{\mathrm{out},t}
=\{(\mathrm{out},\theta_{v,t},o_v,b_{o_v,t}):v\in V_{\mathrm{out}},\ b_{o_v,t}\ne\bot\}.
$$

给定全函数 $\operatorname{Agg}_{\mathrm{out}}:\mathbb N\times\mathcal P_{\mathrm{fin}}(\mathsf{OutRec})\to P$。选择读出时间 $\theta_{\mathrm{out},t}=Dt+r_{\mathrm{out}}$，其中 $r_{\mathrm{out}}=1+\max_v r_v$，并在这个实例中取 $D>r_{\mathrm{out}}$。定义外部读出槽

$$
b_t=\begin{cases}
\operatorname{Agg}_{\mathrm{out}}(\theta_{\mathrm{out},t},\mathcal M_{\mathrm{out},t}),&\mathcal M_{\mathrm{out},t}\ne\varnothing,\\
\bot,&\mathcal M_{\mathrm{out},t}=\varnothing.
\end{cases}
\tag{17}
$$

这是端口记录的一个读出，读出时间指定何时比较这组结果。图内输出仍保留各自的端口与逻辑时间。需要把聚合本身作为节点作用时，可以显式增加一个输出节点及相应正时延边。

> [!corollary] 推论 2：有值输出的充分条件
> 在本边界实例中，若每个非空候选集合都选择至少一个节点，则每个位置至少有一个激活终端，因而 $b_t\in P$。

**证明。** 入口均收到输入，因此存在激活节点。取区域秩最大的激活节点 $v$。若它有出边，广播会使更高区域有候选，该区域也会激活某节点，与最大性矛盾。因此 $v$ 是终端，广播也向其端口输出有值载荷。$\square$

这个证明使用了广播的有值发送。对一般 Full，同样结论需另假设激活者在每条出边有值发送（TotalEmit），且激活终端至少在一个端口输出有值。定理 1 的有限性与唯一性无需这些附加条件。

## 7. 两个输入的手算

取第 1 节的图，区域为 $\mathcal R_0=\{a,b\}$、$\mathcal R_1=\{c\}$，秩为 $1,2$，$D=4$。按第 6.4 节配置入口和终端；输入为 $x(0)=2,x(1)=1$。取 $P=X_v=\mathbb R$，Agg 对非空纤维取均值、空集返回 $0$，输出读出同样取均值。

节点 $a,b$ 的状态、描述量空间为 $\mathbb R$，初态为 $0,1$。控制和区域历史空间均为 $\{*\}$，初始历史为 $*$。取

$$
\begin{aligned}
\operatorname{Upd}(q,\theta,h)&=q+h,\\
\operatorname{Read}^0(\theta,h)&=h,\qquad
\operatorname{Read}^-(q,\theta,h)=\operatorname{Read}^+(q,\theta,h)=q,\\
\operatorname{Next}(q,q^{\mathrm{cmp}},\theta,h,a,*)&=
\begin{cases}0,&a=1,\\q^{\mathrm{cmp}},&a=0,\end{cases}\\
\operatorname{EmitValue}(q^{\mathrm{cmp}},\theta,h,*)&=h+q^{\mathrm{cmp}}.
\end{aligned}
$$

首区域取 $\tau_0=+,\kappa_0=1,K_0=1$，总选最大描述量，平票按 $a,b$ 的顺序。所有候选均得到控制 $*$，历史保持。节点 $c$ 的状态和描述量均为单点集，其状态与读取函数恒返回 $*$，$\operatorname{EmitValue}_c(*,\theta,h,*)=h$；区域容量为 $1$，有候选时总选中，取 $\tau_1=0,\kappa_1=0$。各节点 Full 都由式 (15) 给出。

| 位置 | 旧状态 $(q_a^{[t]},q_b^{[t]})$ | 快照 | 激活者 | 下一状态 | 读出 $b_t$ |
|---|---|---|---|---|---|
| $0$ | $(0,1)$ | $(2,3)$ | $b$ | $(2,0)$ | $5$ |
| $1$ | $(2,0)$ | $(3,1)$ | $a$ | $(0,1)$ | $4$ |

第一个位置，$b$ 在时间 $1$ 沿边 $e_b:b\to c$ 发送 $5$，消息在时间 $2$ 到达；$c$ 的完整纤维是 $\{(\mathrm{msg},1,e_b,5)\}$，输出记录为 $(\mathrm{out},2,o_c,5)$。外部读出在时间 $3$ 得到 $b_0=5$。

首区域两个节点都收到并采用输入，只有一个做 Full；被选节点读取清理前的快照，下一状态为零。若 Next 始终保留快照，第二个位置仍会选择 $b$，说明清理直接参与模型数值。

再把首区域在 $\theta=1$ 的选择改为空集。第一次快照仍为 $(2,3)$，下一状态也为 $(2,3)$，两条边均无消息，$c$ 无候选，$b_0=\bot$。第二次快照为 $(3,4)$，选择 $b$，得到 $b_1=5$。输出槽序列是 $(\bot,5)$。

**练习。** 分别把首区域改成 $\tau_0=-$、$\kappa_0=0$，重算原表；再让激活节点的 Full 在所有坐标取 $\bot$，检查哪些状态作用仍然存在。

## 8. 按统一逻辑时间衰减

取 $S_v=\mathbb R^k\times\mathbb N$，状态 $(z,s)$ 保存值与时间戳。固定 $0<\alpha\le1$，定义全函数

$$
\operatorname{Eff}((z,s),\theta)=\alpha^{\max(0,\theta-s)}z.
\tag{18}
$$

给定 $H_v:X_v\to\mathbb R^k$，令

$$
\operatorname{Upd}_v((z,s),\theta,h)
=(\operatorname{Eff}((z,s),\theta)+H_v(h),\theta).
$$

初态可取 $(z_0,0)$，Next 保留快照或在激活时重置为 $(0,\theta)$。从这个初态出发，每次读取均有 $s\le\theta$，由更新时间递增归纳可得。Read 和 Full 可以通过 Eff 取得衰减后的值。

空档期间保存坐标保持，读取时补算时间差。由于 $\alpha^{a+b}z=\alpha^b(\alpha^az)$，这种表示与逐时衰减具有相同有效值。同节点相邻位置相隔 $D$，对应衰减因子为 $\alpha^D$；从时间 $2$ 到 $10$ 的间隔始终为 $8$。

选择历史也可保存时间戳，在下一次非空候选的选择中读取有效历史。相同保存坐标与相同有效读值是两种比较关系，应用时须指定所比较的对象。

## 9. 整段输入与分段继续

令

$$
\mathsf S=\left(\prod_vS_v\right)\times\left(\prod_jY_j\right),\qquad
S_t=((q_v^{[t]})_v,(y_j^{[t]})_j).
$$

固定端口日程以后，位置 $t$ 的外部输入坐标集为 $I_t=\{(i,k):t_i(k)=t\}$。定理 1 给出一步函数

$$
T_t:\mathsf S\times P^{I_t}\to(P_\bot)^{\mathsf O}\times\mathsf S,
\qquad T_t(S_t,x|_{I_t})=((b_{o,t})_o,S_{t+1}).
\tag{19}
$$

函数下标 $t$ 确定绝对逻辑时间。反复应用它得到区间转导 $T_{p:q}$；空区间返回空输出族与原状态。

> [!theorem] 定理 3：位置边界上的继续
> 对 $p\le m\le q$，从同一 $S_p$ 出发，先求 $T_{p:m}$，再从所得 $S_m$ 求 $T_{m:q}$，拼接输出，结果等于 $T_{p:q}$。

**证明。** 对输入位置归纳，每一步使用同一绝对时间、外部输入和旧状态，故结果相同。$\square$

保存 $(t,S_t)$ 足以从位置边界继续。若需要完整历史，同时保存已产生的记录；相邻段共享的边界状态只识别一次。停在位置内部时，还可能有未到达消息和待用快照，须采用一般逻辑时间切面或阶段记录。

## 10. 嵌入 TimedDAG

### 10.1 同一份规格

取第 2 节的 $G,\mathsf I,\mathsf O,\gamma,\varepsilon,\rho$，式 (2) 的日程与边时延，以及第 3--5 节的全部空间、初态和函数，直接作为 TimedDAG 规格。式 (1) 保证其节点图无环。原子类型、Agg 的集合输入和 Full 的两个返回族都相同。

### 10.2 单位置支撑

> [!lemma] 引理 4：事件时间对齐
> 在这份 TimedDAG 规格中，每个非空纤维的时间都为某个 $\theta_{v,t}$。该纤维中的外部输入与内部消息都属于同一个位置 $t$。

**证明。** 按节点拓扑序归纳。外部记录已满足式 (2)。父节点从位置 $t$ 产生的消息沿式 (3) 到达当前节点的 $\theta_{v,t}$。不同位置对应不同时间，因此不会混入同一纤维。没有外部或父边输入的坐标为空。$\square$

### 10.3 完整记录对应

用 $q_v^\theta,y_j^\theta$ 表示 TimedDAG 的逻辑时间状态。有限输入之后按空纤维规则延拓至 $DL$。位置边界与局部作用的对应为

$$
\begin{aligned}
q_v^{Dt}=q_v^{\theta_{v,t}}&=q_v^{[t]},&
q_v^{\theta_{v,t}+1}=q_v^{D(t+1)}&=q_v^{[t+1]},\\
y_j^{Dt}=y_j^{\theta_{j,t}}&=y_j^{[t]},&
y_j^{\theta_{j,t}+1}=y_j^{D(t+1)}&=y_j^{[t+1]}.
\end{aligned}
\tag{20}
$$

$B,h,\widetilde q,d,c,q^{\mathrm{cmp}},\mathcal C,\mathcal A,f^A,f^O$ 按 $t\mapsto\theta_{v,t}$ 或 $\theta_{j,t}$ 重索引，四类作用的返回值保持原样。有值 $z_{a,t}$ 对应唯一消息 $(\mathrm{msg},\theta_{\operatorname{src}(a),t},a,z_{a,t})$，有值 $b_{o,t}$ 对应唯一输出 $(\mathrm{out},\theta_{\varepsilon(o),t},o,b_{o,t})$；无记录的已确定槽取 $\bot$。

> [!theorem] 定理 5：记录嵌入
> 上述 TimedDAG 的完整记录与 $\mathcal T_x^{\mathrm S}$ 通过式 (20) 及槽的展开一一对应。

**证明。** 引理 4 确定全部可能非空时间。按位置和区域归纳，两边使用相同纤维、旧状态及历史，故每个局部函数的参数和结果相同。空档按恒等规则填充状态。全部消息在本位置区间内到达，位置边界没有遗留消息，于是下一位置的左状态相同。反向限制到这些时间并恢复已确定槽，即得到 $\mathcal T_x^{\mathrm S}$ 的全部坐标。$\square$

本嵌入保留局部函数本身。一般 TimedDAG 可有不对齐路径或交错区域；它们仍使用同一套局部语义。

### 10.4 封闭与切面

单个 $z_{a,t}=\bot$ 表示这个槽无消息。边的封闭下界 $\sigma^A(a)=b$ 表示该边全部到达时间小于 $b$ 的实际消息已经纳入记录。此前位置均已完成、本位置消息均已纳入以后，边的下界可越过 $\theta_{\operatorname{dst}(a),t}$，从而证明目标纤维完整。

一般切面状态为

$$
Q_b=(b,(q_v^b)_v,(y_j^b)_j,W_b),\qquad
W_b=\{m:\operatorname{send}(m)<b\le\operatorname{time}(m)\}.
$$

在位置边界 $b=Dt$，$W_{Dt}=\varnothing$，所以 $Q_{Dt}$ 恰由 $(t,S_t)$ 给出。位置内部切面仍可能具有非空 $W_b$。

共同区间接口 $\Phi_{a,b}(Q_a,E_{[a,b)})=(\mathcal U_{[a,b)},Q_b)$ 返回完整区间记录及右切面，见 [分块教材 §8.1](timed-dag-chunk-prefill-learning-note.md#81-输入分块组合)。取 $a=Dp,b=Dq$ 并投影到端口槽，就得到第 9 节的 $T_{p:q}$。

## 11. 节点级时间批的含义

固定一个位置区间。按区域次序，先取得某区域整段的完整纤维和左状态。由于 Next 与选择历史只依赖 Full 以前的量，可以先按时间产生该区域全部 $P,S,U$，再把每个节点的激活时间收成集合 $\Theta$，共同求其 Full。

相应联合函数使用正典名称

$$
\operatorname{BatchRegionState}_{j,I},\qquad
\operatorname{BatchFull}_{v,\Theta}.
$$

前者返回区域全部实际 $P,S,U$、状态和历史轨迹及右边界；后者返回全部 $F$ 值及展开的消息、输出。若将两者合为区域联合函数，记为 $\operatorname{BatchRegion}_{j,I}$。精确性均为 $\mathcal K(z)=\operatorname{RefBlock}_K(z)$，其中 $K$ 指明覆盖范围，$z$ 包含全部块外自变量。

[分块教材 §6](timed-dag-chunk-prefill-learning-note.md#6-严格分层区域的整块定理) 证明：具有相应精确见证时，整窗口策略每区域至多使用一个状态块，每节点至多使用一个额外 Full 时间批。激活为空时仍保存候选状态与历史；没有 Full 坐标的节点无需 Full 批。

这个结论计量外层调用。状态递归可以在块内顺序进行，块内工作量和并行深度由局部代数结构决定。立即处理当前安全前缀的策略也可能把同节点分成多批；[§6.7](timed-dag-chunk-prefill-learning-note.md#67-对齐路径仍可能被即时前缀分开) 给出时间已经对齐的例子。

## 12. 接入已有模型时的函数保持

本节采用第 6.4 节的共同输入与终端读出实例，并取 $X_v=P=\mathbb R^d$。设已有模型在位置 $t$ 给出 $h_t$，把它送入各入口。固定一个接入实例类 $\mathfrak X$；每个实例指定输入序列、参数、初态和所比较的原模型轨迹。

先要求 $\operatorname{OutputPresent}_{\mathfrak X}$：对类中每个实例、每个位置，终端读出集合非空。推论 2 给出一个充分条件。

令 $B[p]$ 表示保留原子集合 $B$ 的所有来源与时间身份、把各载荷换成 $p$；输出集合 $Z[p]$ 同理。以下三个条件仅在 $\mathfrak X$ 的实际支持和调用域上要求：

1. 对实际非空输入纤维的身份集合，$\operatorname{Agg}_v(\theta_{v,t},B[h_t])=h_t$。
2. 对实际激活调用，$\operatorname{EmitValue}_v(q^{\mathrm{cmp}}_{v,t},\theta_{v,t},h_{v,t},c_{v,t})=h_{v,t}$。
3. 对实际非空终端集合的身份集合，$\operatorname{Agg}_{\mathrm{out}}(\theta_{\mathrm{out},t},Z[h_t])=h_t$。

> [!proposition] 命题 6：共同载荷保持
> 在输出存在及上述三个条件下，每个有值消息载荷都为 $h_t$，且 $b_t=h_t$。

**证明。** 各入口获得 $h_t$。按区域归纳，父消息均为 $h_t$ 时，第一项使本地内容为 $h_t$，第二项与广播使所有新消息仍为 $h_t$。最后由输出存在与第三项得到 $b_t=h_t$。$\square$

均值与归一化加权平均满足相应共同载荷条件。条件只要求实际可达域；也可声明在整个类型域成立的更强充分条件。输出存在必须单独保证，否则全部空选择会使激活调用条件真空成立，却没有有值读出。

若比较还包括原模型状态，另给初态嵌入 $\eta$ 与状态投影 $\pi$，要求

$$
s^{\mathrm{grown}}_0=\eta(s^{\mathrm{base}}_0),\qquad
\pi(s^{\mathrm{grown}}_t)=s^{\mathrm{base}}_t.
$$

新状态可在该投影条件下演化。外部残差 $h_t+\alpha\Delta_t$ 在 $\alpha=0$ 时提供另一种输出保持方式；其状态保持仍须验证。

## 附录 A：系统用语与表示

| 系统名称 | 数学对应 |
|---|---|
| receiver / region | $v\in V$ / $\mathcal R_j$ |
| candidate / active | 非空纤维对应的候选 / 激活集合成员 |
| content / pre / post | 描述量模式 $0,-,+$ |
| SD / BO | $\kappa_j=0$ / $\kappa_j=1$；默认 Next 时分别由激活者／全部候选保存更新 |
| proposal / snapshot / carry | $\widetilde q$ / $q^{\mathrm{cmp}}$ / $q^{\mathrm{next}}$ |
| selector | $\operatorname{SelStep}_{j,C}$，返回激活、控制族与下一历史 |
| DATA / CLOSED | 单个已确定槽的有值结果 / $\bot$ |
| seal / continuation | 时间前缀完整性证明 / $Q_b$ |
| prefill / decode | 整段输入求值 / 逐位置继续 |
| execution checkpoint | $Q_b$ 的可保存表示 |
| parameter checkpoint | 模型参数的可保存表示 |

实现可直接采用 Agg、Upd、Read、SelStep、Next、Full 和四类作用名称。表示空间经预先固定的解码函数得到数学值，再通过遗忘投影删去辅助字段。解码保留确切原子身份、快照和状态；它只解释表示，不调用参考递归来修补结果。比如按固定全序排列的原子序列可无损表示纤维集合，带名字的数组可表示 Full 的两个函数坐标族。

保存和读取执行切面至少满足 $\operatorname{load}(\operatorname{save}(Q_b))=Q_b$。仅保留输出投影或有效衰减值时，须另给相应比较关系。

## 附录 B：发送权重与指定反向规则

本附录使用微分与链式法则。固定激活集合，把 $h,g\in\mathbb R^d,p\in[0,1]$ 视为独立输入。HARD 的载荷为 $g$，SOFTP 为 $h+p(g-h)$，均可经 $\Delta_v$ 形成 Full。

HST 的前向同为 $g$，另附指定反向规则。令 $\operatorname{sg}(p)$ 前向等于 $p$、反向贡献为零，固定实数 $\zeta$，写

$$
\eta=1+\zeta(p-\operatorname{sg}(p)),\qquad
\widehat g=h+\eta(g-h).
$$

前向 $\eta=1$。若输出反向信号为 $u\in\mathbb R^d$，规定三条局部贡献

$$
\bar h=0,\qquad \bar g=u,\qquad
\bar p=\zeta\langle u,g-h\rangle.
$$

再沿生成 $g,p$ 的依赖传播。$g$ 若依赖 $h$，经 $g$ 的贡献仍须汇总。离散激活、状态展开与梯度截断另行指定。前向记录相等只证明前向关系；反向等价须比较同一指定传播规则。
