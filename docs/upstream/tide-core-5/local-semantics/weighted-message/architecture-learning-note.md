---
type: mathematical-learning-note
status: active
cssclasses:
  - textbook-math
tags:
  - tide
  - mathematics
  - local-semantics
  - semantic-anchor
  - learning-note
---

# 带权消息的有状态局部架构

节点收到消息后，先形成候选新状态，再结合当前接收与自身记忆，向所属区域提供描述。区域依据全部候选描述决定各节点的发送强度；获得正发送强度的节点计算完整内容，并向下游广播。全部候选保留本次状态更新，供以后接收使用。

本文定义这条支线的共同架构：消息、接收、状态、描述、选择与广播如何组成一次计算，以及哪些局部条件足以保证整网性质。图、区域和逻辑时间沿用 [SettleGraph](../../settlegraph-learning-note.md)、[TimedDAG](../../timed-dag-region-selector-learning-note.md) 与 [PositiveDelayGraph](../../positive-delay-graph-finite-cut-learning-note.md)。

第 1—4 节依次定义所需对象，并在 §4.3 写出完整计算过程。第 5 节说明如何填入具体函数；第 6—7 节分别研究数值性质和精确求值。记忆、评分与发送的具体公式由各模块教材给出，本文的过程与条件对这些函数统一适用。

## 1. 消息、输入与候选

### 1.1 图、载荷与候选

固定有限节点集 $V$、区域划分 $V=\bigsqcup_j\mathcal R_j$、有限输入端口集 $\mathsf I$ 与输出端口集 $\mathsf O$，以及具有独立身份和正整数时延的有向边。记 $\rho(v)$ 为节点所属区域，$\operatorname{In}(v),\operatorname{Out}(v)$ 为入边与出边集合，$\operatorname{InPort}(v),\operatorname{OutPort}(v)$ 为归属该节点的输入与输出端口集合。逻辑时间为 $\theta\in\mathbb N$。边 $a$ 在时间 $\eta$ 发出的消息于 $\eta+\delta(a)$ 到达。

每个输入原子保留来源身份和载荷。外部原子写作 $(\mathrm{ext},i,k,p)$，其中 $i$ 是输入端口、$k$ 是该端口的输入序号、$p$ 是载荷；内部原子写作 $(\mathrm{msg},\eta,a,p)$，其中 $\eta$ 是发送时间、$a$ 是边。节点 $v$ 在 $\theta$ 的完整原子集合记为 $B_{v,\theta}$，称为输入纤维。只有 $B_{v,\theta}\ne\varnothing$ 时产生节点作用；区域的候选集合为

$$
C=\{v\in\mathcal R_j:B_{v,\theta}\ne\varnothing\}.
$$

固定内容宽度 $d\ge1$，载荷空间取 $P=\mathbb R^{1+d}$，载荷 $(m,z)$ 含强度 $m$ 与内容 $z\in\mathbb R^d$。记 $x_+=\max(x,0)$。接收使用 $m_+$，实际发送的强度非负。缺席记为 $\bot$。数值比较使用

$$
\pi_P(m,z)=(m_+,m_+z),\qquad \pi_P(\bot)=0.
\tag{1}
$$

存在的零强度消息仍产生输入原子；缺席不产生原子。两者的数值投影相同，原始事件记录可以不同。

### 1.2 共同数值函数

接收与后续模块使用下列归一化及逐分量函数。

固定 $\epsilon>0$，对 $p$ 维向量定义

$$
\mathcal N_p(x)=\frac{x}{\sqrt{\epsilon+\|x\|_2^2/p}},\qquad
\sigma(x)=\frac1{1+e^{-x}},\qquad \operatorname{SiLU}(x)=x\sigma(x).
\tag{2}
$$

向量上的标量函数逐分量作用，$\odot$ 表示逐分量乘法，$[x;y]$ 表示向量拼接。

## 2. 接收聚合

### 2.1 来源分类与聚合

为节点 $v$ 固定有限类别集 $\mathsf L_v$ 和来源分类函数

$$
\operatorname{SrcClass}_v:
 (\{\mathrm{ext}\}\times\operatorname{InPort}(v))
 \sqcup(\{\mathrm{msg}\}\times\operatorname{In}(v))\longrightarrow\mathsf L_v.
$$

将完整输入集合 $B$ 中的原子逐个编号。第 $i$ 个原子的载荷记为 $(m_i,z_i)$，来源 $s_i$ 取其 $(\mathrm{ext},\text{输入端口})$ 或 $(\mathrm{msg},\text{入边})$，类别为 $a_i=\operatorname{SrcClass}_v(s_i)$。分类不使用输入序号或发送时间；原子仍保留完整身份，相同类别的不同原子各参与一次聚合。

取 $\alpha_{\mathrm{agg}}\ge0$，对每个类别 $a\in\mathsf L_v$ 固定在有界集上有界的连续函数 $F_{v,a}:\mathbb R^d\to\mathbb R^d$。聚合空间取 $X_v=\mathbb R^{1+d}$，定义

$$
\begin{aligned}
r^{\mathrm{in}}&=\sum_i(m_i)_+,\\
u&=\begin{cases}
\displaystyle\frac{\sum_i(m_i)_+z_i+
 \alpha_{\mathrm{agg}}\sum_i(m_i)_+F_{v,a_i}(z_i)}{r^{\mathrm{in}}},&r^{\mathrm{in}}>0,\\
0,&r^{\mathrm{in}}=0,
\end{cases}\\
\operatorname{Agg}_v(\theta,B)&=(r^{\mathrm{in}},u).
\end{aligned}
\tag{3}
$$

这里 $r^{\mathrm{in}}$ 是接收强度之和，$u$ 是聚合内容；写出节点下标时记为 $r_v^{\mathrm{in}},u_v$。全部输入强度同乘正数时，$u$ 保持，$r^{\mathrm{in}}$ 按比例变化。空集同样得到 $(0,0)$；无输入节点由正典规则保持状态。

### 2.2 来源函数与参数共享

取正整数宽度 $p_a$ 以及矩阵 $A_{v,a}\in\mathbb R^{p_a\times d}$、$B_{v,a}\in\mathbb R^{d\times p_a}$ 和向量 $e_{v,a}\in\mathbb R^{p_a}$。一种完整的来源函数是

$$
F_{v,a}(z)=B_{v,a}\operatorname{SiLU}(A_{v,a}\mathcal N_d(z)+e_{v,a}),
\tag{4}
$$

参数均有限。令 $j=\rho(v)$，以下绑定给出四种聚合；共享矩阵的宽度须相同：

| 绑定 | 参数 |
|---|---|
| 区域共享 | $A_{v,a}=A_j,B_{v,a}=B_j,e_{v,a}=e_{j,a}$ |
| 节点独立 | $A_{v,a}=A_v,B_{v,a}=B_v$，嵌入按节点和类别指定 |
| 来源独立 | $A_{v,a}=A_{j,a},B_{v,a}=B_{j,a},e_{v,a}=e_{j,a}$ |
| 节点与来源适配 | $A_{v,a}=A_j+L^A_{v,a}R^A_{v,a}$，$B_{v,a}=B_j+L^B_{v,a}R^B_{v,a}$ |

最后一行的因子取相容形状与指定秩。聚合在压缩前引入来源差异；聚合后的函数只能使用已保留的 $u$。相同来源类别、载荷多重集和参数才保证聚合值相同。$\alpha_{\mathrm{agg}}=0$ 给出加权均值。

## 3. 状态与记忆描述

### 3.1 接收更新与状态保留

每个节点选择非空状态空间 $S_v$、初态 $q_v^{\mathrm{init}}\in S_v$ 和函数 $\operatorname{Upd}_v:S_v\times\mathbb N\times X_v\to S_v$。给定旧状态 $q$ 和聚合 $h=(r^{\mathrm{in}},u)$，候选状态为

$$
\widetilde q=\operatorname{Upd}_v(q,\theta,h).
\tag{5}
$$

本次选择的激活指示记为 $a\in\{0,1\}$，发送控制记为 $c\in[0,1]$。本支线的共同采用规则是 $\kappa_j=1$：全部候选都以 $q^{\mathrm{cmp}}=\widetilde q$ 作为本次计算状态，并保留更新：

$$
\operatorname{Next}_v(q,q^{\mathrm{cmp}},\theta,h,a,c)=q^{\mathrm{cmp}}.
\tag{6}
$$

将 Next 的返回值记为下一持久状态 $q^{\mathrm{next}}$。因此记忆写入不等待区域选择，未激活候选也保留自己的更新。Full 的返回值经消息影响以后的作用。Next 与区域历史不读取或重算本次 Full。

### 3.2 从状态取得记忆描述

节点可以结合本次接收与自身记忆参与区域分工。Agg 已将当前原子集合 $B_{v,\theta}$ 变为 $(r^{\mathrm{in}},u)$，节点状态 $q$ 保存历次接收形成的信息。本节规定当前内容特征与记忆描述的接口，使不同状态形式都能为 Describe 提供构造节点描述所需的信息。

固定正整数描述宽度 $d_s$。记忆模块提供当前内容特征函数 $p_v:\mathbb R^d\to\mathbb R^{d_s}$，从聚合内容 $u$ 提取特征。模块另提供两个向量 $M_{v,\mathrm{short}},M_{v,\mathrm{long}}\in\mathbb R^{d_s}$ 和两个非负统计量 $W_{v,\mathrm{short}},W_{v,\mathrm{long}}$，组成记忆描述函数

$$
\operatorname{Mem}_v(q,\theta,u)
 =(M_{v,\mathrm{short}},M_{v,\mathrm{long}},W_{v,\mathrm{short}},W_{v,\mathrm{long}}),
\tag{7}
$$

$p_v(u)$ 与两个记忆向量具有相同宽度，便于构造当前内容与记忆的交互。Mem 从指定状态取得信息，也可以使用当前 $u$ 查询该状态；用于读取的状态由下一节的描述模式指定。两个向量承载记忆内容，统计量可表达不同时间尺度下的接收积累；short、long 是两个通道的名称，各模块定义其数值含义。

本文规定这些函数的输入与返回形状。第 5 节列出的记忆教材给出具体函数，内部状态可以保存序列、矩阵或其他对象。后续 Describe 使用 $p_v(u)$、Mem 的返回值以及当前接收，构造区域所需的节点描述；具体字段由区域评分与发送分配的需求决定，可以承载相关性评分，也可以承载调节发送总量或阈值的证据。

### 3.3 节点描述与状态快照

Describe 将当前接收与记忆信息组合为传给区域的节点描述。为每个节点指定描述空间 $D_v$，以所选记忆描述函数和当前特征函数为部件，记这组描述函数为

$$
\operatorname{Describe}_v[\operatorname{Mem}_v,p_v]:
 S_v\times\mathbb N\times X_v\longrightarrow D_v.
$$

方括号中是组成模型时选定的函数；圆括号中是一次求值的输入。给定 $(q,\theta,h)$，其中 $h=(r^{\mathrm{in}},u)$，Describe 使用 $p_v(u)$、$\operatorname{Mem}_v(q,\theta,u)$ 及当前接收，得到

$$
d_v=\operatorname{Describe}_v[\operatorname{Mem}_v,p_v]
 (q,\theta,h)\in D_v.
$$

这些信息只来自本节点；区域随后同时使用所有候选的 $d_v$。绑定上述函数后，简记为 $\operatorname{Describe}_v(q,\theta,h)$。

具体的 Describe 与区域选择须配套定义。节点可以先将当前内容与记忆变为局部评分，也可以同时提供区域匹配或区域控制所需的向量。当前特征与记忆描述是构造 $d_v$ 的依据，$D_v$ 规定传给区域的描述结构。

正典 Read 按描述模式 $\tau_j\in\{0,-,+\}$ 选取状态，再调用已绑定的 Describe：

$$
\begin{aligned}
\operatorname{Read}_v^+(\widetilde q,\theta,h)&=\operatorname{Describe}_v(\widetilde q,\theta,h),\\
\operatorname{Read}_v^-(q,\theta,h)&=\operatorname{Describe}_v(q,\theta,h),\\
\operatorname{Read}_v^0(\theta,h)&=\operatorname{Describe}_v(q_v^{\mathrm{init}},\theta,h).
\end{aligned}
\tag{8}
$$

选择描述模式 $\tau_j=+$ 时，Describe 可看见本次写入；选 $-$ 时，Describe 使用旧记忆并显式接收当前 $u$；选 $0$ 时只使用当前内容与固定初态。三者的 Full 均读取 $\widetilde q$。状态模块必须定义在查询时刻如何读取旧状态，特别是逻辑时间空档内的衰减与过期。

为写出完整过程，将描述模式指定的状态记为

$$
q_v^{\mathrm{read}}=
\begin{cases}
q_v^{\mathrm{init}},&\tau_j=0,\\
q_v,&\tau_j=-,\\
\widetilde q_v,&\tau_j=+.
\end{cases}
$$

因此三种 Read 均可写成 $d_v=\operatorname{Describe}_v(q_v^{\mathrm{read}},\theta,h_v)$。这里 $q_v^{\mathrm{read}}$ 仅表示本次读取的状态；本次完整内容使用 $q_v^{\mathrm{cmp}}=\widetilde q_v$，未来接收使用 $q_v^{\mathrm{next}}=\widetilde q_v$。

## 4. 分配、完整计算与发送

### 4.1 区域选择

区域历史空间取 $Y_j=\{*\}$，其中 $*$ 是唯一元素；控制空间取 $\mathsf C_v=[0,1]$。给定候选描述 $d_v\in D_v$，所选分配函数确定每个候选的发送强度 $c_v$；激活集合为 $\mathcal A=\{v\in C:c_v>0\}$。区域返回

$$
\operatorname{SelStep}_{j,C}(*,\theta,(d_v)_{v\in C})
 =(\mathcal A,(c_v)_{v\in C},*).
\tag{9}
$$

空候选返回 $(\varnothing,(),*)$。固定整数容量 $1\le K_j\le|\mathcal R_j|$，要求每次返回的激活集合满足 $|\mathcal A|\le K_j$。SelStep 可以先求评分再分配强度，也可以直接从描述确定控制量；具体函数须给出所有候选集合上的返回值。

区域控制可从描述族指定本次发送强度之和，或计算共同阈值，再由评分完成分配。这些中间量属于同一次 SelStep；描述需要提供足够的内容与记忆证据，节点上限、总量范围和激活容量各按所选函数绑定。完整定义见区域教材的[指定总量](region-allocation-learning-note.md#2-指定总量的一次分配)与[区域阈值](region-allocation-learning-note.md#412-区域阈值分配)。

### 4.2 完整内容与广播

完整内容由函数 $G_v:S_v\times\mathbb N\times\mathbb R^d\to\mathbb R^d$ 给出。该函数使用本节点状态和当前聚合内容，返回消息中的 $d$ 维内容。它可以先读取记忆，再作内容变换；它与 Mem 允许共享参数。定义

$$
\begin{aligned}
\operatorname{EmitValue}_v(q^{\mathrm{cmp}},\theta,(r^{\mathrm{in}},u),c)
 &=\begin{cases}(c,G_v(q^{\mathrm{cmp}},\theta,u)),&c>0,\\(0,0),&c=0,\end{cases}\\
\Delta_v(p)&=\big((p)_{a\in\operatorname{Out}(v)},(p)_{o\in\operatorname{OutPort}(v)}\big),\\
\operatorname{Full}_v&=\Delta_v\circ\operatorname{EmitValue}_v.
\end{aligned}
\tag{10}
$$

只有激活节点调用 Full。它返回逐出边与逐输出端口两个值族，每个坐标均为同一载荷。数值贡献为 $(c,cG_v)$，完整内容及其残差都受同一个发送强度控制。广播在每条边复制载荷。

若节点 $v$ 的出度为 $d_v^+$，区域本次发送强度之和为 $S=\sum_vc_v$，内部边上的总强度为 $\sum_vd_v^+c_v$。节点激活数、区域强度和广播工作是三个量。

### 4.3 一次区域作用的完整过程

固定区域 $j$、时间 $\theta$，给定全部输入纤维和各节点旧状态。由第 1 节得到候选集合 $C$。对每个 $v\in C$，先计算

$$
\begin{aligned}
h_v&=\operatorname{Agg}_v(\theta,B_{v,\theta}),\\
\widetilde q_v&=\operatorname{Upd}_v(q_v,\theta,h_v),\\
d_v&=\operatorname{Describe}_v(q_v^{\mathrm{read}},\theta,h_v).
\end{aligned}
$$

$q_v^{\mathrm{read}}$ 按 §3.3 从初态、旧状态或本次新状态中取得。区域收齐这些描述后计算

$$
(\mathcal A,(c_v)_{v\in C},*)
=\operatorname{SelStep}_{j,C}(*,\theta,(d_v)_{v\in C}).
$$

随后全部候选采用并保存更新：

$$
q_v^{\mathrm{cmp}}=\widetilde q_v,\qquad
q_v^{\mathrm{next}}
=\operatorname{Next}_v(q_v,\widetilde q_v,\theta,h_v,
 \mathbf1[v\in\mathcal A],c_v)=\widetilde q_v.
$$

仅对 $v\in\mathcal A$ 求式 (10) 的 Full。每条出边收到 $(c_v,G_v(\widetilde q_v,\theta,u_v))$，并在 $\theta+\delta(a)$ 成为下游输入原子。输出端口在当前时间记录同一载荷。无输入节点保持原状态；候选中未激活的节点保留更新，供未来的 Read 和 Full 使用。

### 4.4 图的输出读出

节点发送产生逐端口记录；图的输出由这些记录上的观察函数取得。

固定有限输出坐标集合 $\Lambda\subseteq\mathbb N\times\mathsf O$，从完整记录 $Z$ 取这些坐标的载荷 $(m_i,z_i)$，定义

$$
\operatorname{ReadOut}_\Lambda(Z)=
 \frac{\sum_i(m_i)_+z_i}{\max(1,\sum_i(m_i)_+)}.
\tag{11}
$$

空集读出零，单个单位载荷原样读出。读出在图外汇集，不增加图内作用。正强度输出的存在与读出向量非零分别判断，内容可以相消。

### 4.5 一组贯穿例子

取三个候选节点 $a,b,c$，当前输入分别为 $(0.5,1),(0.5,1)$、$(1,-1)$ 和 $(0,9)$。内容宽度为 $1$，聚合取 $\alpha_{\mathrm{agg}}=0$，得到 $(1,1),(1,-1),(0,0)$。三个节点均经过 Upd、Read 和 Next。

先固定共同的记忆与内容计算，给定本次新状态上的 $G_a=2,G_b=-2$。下面比较两种发送绑定；它们都令 $c_a,c_b>0$、$c_c=0$，故只调用 $a,b$ 的 Full。若两条消息同时到达使用 $\alpha_{\mathrm{agg}}=0$ 的同一下游，则

$$
r^{\mathrm{in}}=c_a+c_b,\qquad
u=\frac{2c_a-2c_b}{c_a+c_b}.
$$

| 发送绑定 | 控制 $(c_a,c_b,c_c)$ | $a,b$ 发出的载荷 | 下游 $(r^{\mathrm{in}},u)$ |
|---|---|---|---|
| [指定总量，$B=1.4$](region-allocation-learning-note.md#25-三个候选的一次计算) | $(0.95,0.45,0)$ | $(0.95,2),(0.45,-2)$ | $(1.4,5/7)$ |
| [节点增益与固定阈值，$k=1$](region-allocation-learning-note.md#52-更换阈值与节点增益) | $(0.75,0.25,0)$ | $(0.75,2),(0.25,-2)$ | $(1,1)$ |

若直接读出这两个端口，式 (11) 分别给出 $5/7$ 与 $1$。区域教材推导两行的控制；[KV §5.1](kv-memory-learning-note.md#51-从接收到输出)用同一组接收、记忆和内容参数完成两种绑定。选择其中一行后，沿相应控制继续计算。

## 5. 如何组成一个模型

一份组合依次指定下列对象：

| 对象 | 必须给定的内容 |
|---|---|
| 图与时间 | 节点、区域、边身份及时延、端口、输入日程、输出坐标 |
| 接收 | 内容宽度、来源分类与 Agg 参数 |
| 记忆 | 状态、初态、Upd、Mem、当前特征 $p_v$ |
| 完整内容 | 历史读取及随后变换组成的 $G_v$，与记忆模块的参数共享关系 |
| 描述与评分 | 描述模式、Describe 与 SelStep 使用的共同 $D_v$，局部和区域评分 |
| 发送 | 所选分配函数及其参数、正典容量 $K_j$ |
| 延续 | 本文 Next 与广播 Full，状态的时钟、窗口及边界恢复 |

相连节点共用载荷宽度；同一区域的描述须能进入同一个 SelStep。不同状态宽度可先映到共同描述宽度 $d_s$。各记忆教材定义的 KV 与循环模块可在不同区域组合；同一区域异构时还须统一描述和匹配宽度。所有函数对其声明空间都是确定的全函数。

本文规定共同的函数签名与计算顺序。各教材实现这些函数，并按下表使用已给定的接口：

| 教材 | 填入的函数或对象 | 接入共同过程的位置 |
|---|---|---|
| [区域评分与发送分配](region-allocation-learning-note.md) | $D_v$、Describe、SelStep 及其参数 | 使用当前接收、$p_v(u)$ 与 Mem 构造 $d_v$，再从区域描述族得到 $(\mathcal A,(c_v),*)$ |
| [带权 KV](kv-memory-learning-note.md) | $S_v,q_v^{\mathrm{init}}$、Upd、Mem、$p_v,G_v$ | 以带权序列保存接收，实现当前内容特征、记忆描述与完整内容 |
| [基础循环记忆](recurrent-memory/affine-delta-learning-note.md)、[Gated DeltaNet](recurrent-memory/gated-delta-learning-note.md)、[SSD](recurrent-memory/ssd-learning-note.md) | 同上 | 以矩阵及辅助状态保存接收，实现同型的当前内容特征、记忆描述与完整内容 |
| [固定逻辑时间短卷积](recurrent-memory/causal-convolution-learning-note.md) | Conv、ConvWrite | 作为 Gated DeltaNet 与 SSD 更新、读取中的部件 |

记忆教材给出 Mem 与当前特征 $p_v$，区域教材给出 Describe 如何组合它们；选定后得到 §3.3 的 $\operatorname{Describe}_v[\operatorname{Mem}_v,p_v]$。更换区域评分或发送函数时，Upd 与 $G_v$ 仍按各自定义计算。式 (7) 的两个描述通道是这里复用评分族的接口；采用其他描述形式时须同时指定相容的 Describe 与 SelStep。

三个图层级使用同一组 Agg、Upd、Read、SelStep、Next、Full。正时延图允许反馈；图无环时得到 TimedDAG；进一步满足区域严格分层和到达对齐时得到 SettleGraph。[全连接分层](../../topologies/fully-connected-layered.md)与 [HB-Lattice](../../topologies/hb-lattice.md)分别给出独立的图与时间定义；[完整实例](../../README.md#具体拓扑与完整实例)将它们与本支线的局部函数绑定。

## 6. 数值性质的组合

### 6.1 比较对象与局部条件

第 5 节的类型和容量条件使组合成为正典中的合法模型。下面另外给出数值性质的充分条件；每个命题单独使用其前提。

固定有限正时延图、有限切面 $B$、外部输入槽及其原始时间、参数和初态。外部槽按端口与输入序号对齐，内部消息按边与发送时间对齐。缺席槽的 $\pi_P$ 取零；因此消息比较的是式 (1) 的 $(m_+,m_+z)$。

为每个状态模块指定有效状态投影

$$
e_{v,\theta}=\pi_{v,\theta}(q_v).
$$

它在固定比较时刻取值于固定有限维向量空间。例如，带权序列可以按原始写入时间对齐，比较窗口内的加权条目；循环状态可以比较同一时刻的有效矩阵和摘要。精确继续仍保存原始状态。

记忆与内容模块满足以下四项条件：

1. **有界性。** 在固定有限时间内，有界的接收内容和强度产生有界的有效状态、Mem、当前特征与 $G_v$。比较温度族时使用统一的界。
2. **零接收中性。** 插入一次零接收，保持当前及任意相同后续接收下的有效状态和读取结果。没有接收时保存状态保持。
3. **更新连续性。** 旧有效状态收敛，且
   $r^{\mathrm{in},(n)}\to r^{\mathrm{in}}$、$u^{(n)}$ 有界。若 $r^{\mathrm{in}}>0$，另要求 $u^{(n)}\to u$。则更新后的有效状态及其随后在固定时刻的投影收敛；$r^{\mathrm{in}}=0$ 时极限为该槽没有接收的状态。
4. **读取连续性。** 有效状态与当前内容收敛时，Mem、当前特征与 $G_v$ 收敛。三种 Read 分别使用 §3.3 指定的状态。

接收趋零时，聚合内容 $u$ 可以保持非零，甚至不收敛。因此第三项要求本次状态改变量随接收消失；第四项只在当前内容收敛处使用。各记忆教材定义自己的 $\pi_{v,\theta}$，并在附录中验证这四项条件。

### 6.2 零消息与正温度输入

在描述模式指定的状态上求 Read，再按 §4.3 调用 SelStep，得到由这些状态及聚合输入决定的控制族 $(c_v)_{v\in C}$。为比较不同候选集合，把缺席节点的控制补为零。对这一复合函数分别提出两个条件：

- **零接收条件：** $r_v^{\mathrm{in}}=0$ 时 $c_v=0$；增加或删除这样的候选，不改变其余节点的控制。
- **正温度连续条件：** 有效状态收敛，接收强度收敛，聚合内容有界且在正极限接收处收敛；所用温度各自趋向正值时，补齐后的控制族收敛到极限输入的控制族，且
  $r_v^{\mathrm{in},(n)}\to0\Rightarrow c_v^{(n)}\to0$。

**命题 1：零消息删除。** 若记忆满足零接收中性，区域控制满足零接收条件，则增加或删除零强度消息保持消息数值投影、有效状态和 ReadOut。显式计算零控制的 Full 及其零广播，也保持这些数值。

**命题 2：正温度输入连续性。** 若记忆与内容满足 §6.1 的四项条件，区域控制满足上述两个条件，则外部输入的 $\pi_P$ 收敛、内容一致有界、所用温度各自趋向正值时，切面以前的消息投影、有效状态及 ReadOut 收敛到极限输入的运行。

两个命题都比较数值；零原子的存在可以改变原始事件记录。命题 2 的发送步骤使用

$$
c_v^{(n)}\to0,\quad \sup_n\|G_v^{(n)}\|<\infty
\quad\Longrightarrow\quad
\|(c_v^{(n)},c_v^{(n)}G_v^{(n)})\|
\le c_v^{(n)}(1+\|G_v^{(n)}\|)\to0.
$$

当控制的极限为正时，正温度连续条件保证接收极限为正，读取连续性给出内容收敛。两种情形合起来覆盖全部发送坐标。

### 6.3 局部分配极限与整网极限

固定同一外部输入、初态、记忆与内容函数。比较一列区域控制函数与一组极限控制函数；每组函数分别满足第 5 节的类型和容量条件。温度等分配参数允许随序列变化。

**命题 3：局部控制收敛推出整网收敛。** 假设 §6.1 的四项条件成立，各组控制满足零接收条件。在切面以前的每个区域时间槽，若局部输入按 §6.2 的方式趋向极限运行的输入时，控制族也趋向极限函数的控制族，则整网消息投影、有效状态与 ReadOut 收敛到极限运行。

这一前提包含极限运行中没有候选的时间槽：趋于消失的接收必须产生趋零的控制。它还允许零极限接收处的内容只保持有界。固定一个局部输入时的温度极限，需要满足这些随上游变化的输入条件，才能用于整网归纳。

**推论：充分小正温度的精确运行。** 进一步，设一族控制由各区域正温度参数化。在极限运行的每次实际区域选择中，固定该次原始输入后，存在正温度界，使界内的控制精确等于极限控制。则存在共同正界，使全部区域温度位于该界内时，整网在切面以前的实际作用、原始状态、消息、端口记录和继续状态都精确相同。

推论比较同一固定输入、参数与有限时间范围。区域分配教材的[指定总量极限](region-allocation-learning-note.md#72-指定总量的零温分配)、[区域阈值配对](region-allocation-learning-note.md#74-区域阈值与单位强度硬选择)和[节点增益与固定阈值的硬化](region-allocation-learning-note.md#75-节点增益与固定阈值的硬化)分别验证所选函数如何满足本节条件。各自的评分间隔、线性最优性与正则条件由该教材给定；训练过程中参数的变化另行分析。

### 6.4 从局部条件到组合结论

| 要判断的性质 | 记忆与内容需要验证 | 描述与区域选择需要验证 | 结论所在 |
|---|---|---|---|
| 合法的确定计算 | 函数类型与全函数性 | 描述类型、控制范围、激活容量 | 第 5 节与正典递归 |
| 零消息可删除 | 零接收中性 | 零接收条件 | 命题 1 |
| 正温度输入连续 | §6.1 四项条件 | 零接收与正温度连续条件 | 命题 2 |
| 分配极限传递到整网 | 同上，界在比较族上一致 | 在极限运行输入处满足命题 3 的局部条件 | 命题 3 |
| 小正温度即精确相同 | 原始函数固定 | 固定极限运行输入时，控制在小正温度精确相同 | §6.3 推论 |
| 精确分段继续或联合求值 | 保存完整状态、返回完整作用值 | 保留区域选择所需的全部输入和结果 | 第 7 节 |

具体发送规则的适用情况由[区域分配 §8](region-allocation-learning-note.md#8-发送规则如何满足架构条件)逐项验证。不同区域可选择不同记忆与发送函数，每个组合结论按本表分别核对。

## 7. 切面与联合求值

### 7.1 保存什么即可继续

切面 $b$ 表示所有逻辑时间小于 $b$ 的作用已经完成。其继续状态沿用正典：

$$
Q_b=(b,(q_v^b)_v,(y_j^b)_j,W_b),
\tag{12}
$$

其中 $q_v^b$ 是该切面的持久节点状态，$y_j^b=*$，$W_b$ 保存此前发出、在 $b$ 或以后到达的消息。后续仍使用原始逻辑时间、位置函数及同一组参数。

沿用正时延图教材的区间函数 $\Phi_{a,b}$：给定一个可达切面 $Q_a$ 和区间外部输入 $E_{[a,b)}$，它按 §4.3 运行，返回该区间的完整记录 $\mathcal U_{[a,b)}$ 与右切面 $Q_b$。因此

$$
\begin{aligned}
\Phi_{a,b}(Q_a,E_{[a,b)})&=(\mathcal U_{[a,b)},Q_b),\\
\Phi_{b,c}(Q_b,E_{[b,c)})&=(\mathcal U_{[b,c)},Q_c)
\end{aligned}
$$

蕴含

$$
\Phi_{a,c}(Q_a,E_{[a,c)})
=(\mathcal U_{[a,b)}\odot_b\mathcal U_{[b,c)},Q_c).
$$

$\odot_b$ 沿用正典的记录粘合：区间内部的事件与新消息并合，共同边界状态只保留一份，左段的在途消息不重复登记为右段新消息。这就是精确分段继续。其依据是完整切面状态与确定递归，无需第 6 节的连续性假设。

保存原始状态可以直接使用这个等式。若只保存有效状态投影 $\pi_{v,b}(q_v^b)$，须另外证明它足以恢复所要求的后续记录；第 6 节的数值比较本身不提供这一恢复结论。

### 7.2 一次求出多步时必须返回什么

固定区域 $j$ 和区间 $I=[b,c)$，给定左节点状态 $(q_v^b)_{v\in\mathcal R_j}$、区域历史 $*$，以及

$$
(B_{v,\theta})_{v\in\mathcal R_j,\ \theta\in I}.
$$

这里每个纤维都是完整的，包含确定为空的纤维。按时间逐次执行 §4.3 的接收、描述、选择与状态保留，得到区域参考函数 $\operatorname{RefRegionState}_{j,I}$。其返回值包括这些作用的时间标签、全部中间值、控制与激活集合、状态轨迹及右边界状态；正典分别将准备、选择、状态采用标为 P、S、U。

称一个联合函数为 $\operatorname{BatchRegionState}_{j,I}$，要求它在上述全部合法输入上满足

$$
\operatorname{BatchRegionState}_{j,I}
=\operatorname{RefRegionState}_{j,I}.
$$

再固定节点 $v$ 的实际激活时刻集 $\Theta$，给定各时刻的
$(q_{v,\theta}^{\mathrm{cmp}},\theta,h_{v,\theta},c_{v,\theta})$。令 $K^F_{v,\Theta}$ 为这些 Full 作用坐标，$\operatorname{RefBlock}_{K^F_{v,\Theta}}$ 为逐次求 Full 并展开消息、端口记录的函数。完整输出联合函数要求

$$
\operatorname{BatchFull}_{v,\Theta}
=\operatorname{RefBlock}_{K^F_{v,\Theta}}.
$$

两个等式均要求保留全部正式作用值与原始时间。它们分别对应[分块教材](../../timed-dag-chunk-prefill-learning-note.md)的区域状态契约和完整输出契约。

KV 的条目前缀、循环记忆的仿射复合分别构造这些联合函数。完整内部纤维可能仍需由上游 Full 产生；何时能取得这些输入，由图和因果依赖决定。函数等式保证结果相同，计算量与并行性还需分析具体构造。

## 附录 A：组合定理的证明

### A.1 有限槽与聚合

固定切面以前，每条边的每个发送时间至多有一个消息槽，外部输入槽也有限。把缺席按零投影补齐，得到共同有限比较空间。正时延使时间 $\theta$ 的内部输入只来自更早发送。

若输入投影收敛，则接收强度收敛。正极限强度的单条内容可由 $(m,mz)$ 相除恢复；零极限强度的有界内容和来源项对加权和的贡献趋零。故 $r^{\mathrm{in},*}>0$ 时 $u\to u^*$；$r^{\mathrm{in}}\to0$ 时 $u$ 仍有界。有限槽、发送强度至多为 $1$ 及局部有界性，沿时间给出统一的有限时间界。

### A.2 零消息与正温度

**命题 1。** 零消息不改变 Agg 的加权和。仅由零输入产生的候选因零接收中性保持有效状态，区域的零接收条件使它的控制为零且不改变其他控制。零控制载荷的数值投影为零。沿时间归纳，增删这些原子及其后继零广播保持全部所述数值；原始事件支持可以不同。$\square$

**命题 2。** 假设更早消息投影与状态已收敛。A.1 和更新连续性给出本次有效状态收敛；正极限接收处的当前内容也收敛。区域控制条件给出 $c_v\to c_v^*$。若 $c_v^*>0$，接收极限为正，读取连续性给出 $G_v\to G_v^*$；若 $c_v^*=0$，有界性给出 $(c_v,c_vG_v)\to0$。故本次消息投影收敛。Next 保留更新，启动下一时间的归纳；ReadOut 的有限和与正分母保持收敛。$\square$

### A.3 局部极限的传递

**命题 3。** 使用 A.2 的同一归纳，以本命题的局部控制收敛假设替换正温度连续条件。空槽和仅有趋零接收的槽由补零控制与零接收条件处理；其余槽由更新、读取和控制收敛处理。固定有限时间内只有有限步，故得到全部消息、有效状态与读出的收敛。$\square$

### A.4 小正温度的精确性

**证明推论。** 极限运行在切面以前的实际区域选择事件集合 $E$ 有限。对每个事件取使控制精确相同的正温度界 $\delta_e$，令 $\tau_*=\min_{e\in E}\delta_e$；$E$ 为空时取任意正界。

沿时间归纳。初态和外部输入相同；若此前原始状态和实际消息相同，则当前纤维、候选、Upd 和 Read 相同。温度界使控制和激活集合相同，Next 与 Full 遂给出相同状态和记录。空槽两边均不产生候选。于是全部正式作用、消息、端口记录以及切面状态相同。$\square$

## 附录 S：名称、配置与计算工作

### S.1 配置对应

配置名标记组成模型时的函数选择。下表说明职责与定义归属；模块的候选取值、默认值和参数生效范围由对应附录给出。

| 配置名 | 数学选择与定义归属 |
|---|---|
| memory_family | 选择第 5 节所列记忆模块的状态、Upd、Mem、当前特征与内容函数绑定；模块附录给出具体参数 |
| memory_read | 选择该记忆模块的 Mem 读取函数；两个通道的含义由模块定义 |
| read_mode | 本文 §3.3 的正典描述模式 $\tau_j\in\{0,-,+\}$ |
| local_score、region_score、region_key | 选择局部评分、区域评分与 att 所需的记忆 key，见[区域分配附录 S.1](region-allocation-learning-note.md#s1-配置名称) |
| selection_mode | 选择完整发送函数及其描述绑定，见同一配置表 |
| region_control_mode、region_evidence、region_control_width | 选择区域控制、其证据构造及宽度，见同一配置表 |
| send_gain、query_weight、eta、total_fraction 等 | 所选区域函数的组成部件或参数；生效范围见同一配置表 |

其余参数包括宽度、矩阵、偏置、共享关系、窗口、时钟与位置。参数空间和状态空间随这些绑定确定。KV §5.1 的共同绑定为 memory_read=sum、local_score=cat、region_score=ind 和 Read$^+$；发送按例子中的分支选择。函数名称对应数学定义；下游可按配置构造模型，其已支持形状和求值内核由实现声明。

配置改变可以改变参数形状、前向函数和历史状态。形状相同也不自动使旧状态等于新模型从头计算的状态；继续、重算或显式转换各自给出比较关系。

### S.2 工作量

设 $\mathcal E_C,\mathcal E_A,\mathcal E_S$ 分别为候选、激活和区域选择的时间坐标，$\mathcal M$ 为内部消息。按不重复归属的工作代价，有

$$
\mathcal W=\sum_{(v,\theta)\in\mathcal E_C}
 (C^{\mathrm{Agg}}+C^{\mathrm{Upd}}+C^{\mathrm{Read}})_{v,\theta}
 +\sum_{(j,\theta)\in\mathcal E_S}C^{\mathrm{Sel}}_{j,\theta}
 +\sum_{(v,\theta)\in\mathcal E_A}C^{\mathrm{Full}}_{v,\theta}
 +\sum_{m\in\mathcal M}C_m^{\mathrm{msg}}.
$$

共享计算计一次，缓存与读出工作计入对应项。零发送先省去本次 Full 和广播；下游若没有其他输入，还会省去整次准备。局部支持数与端到端工作之间还隔着拓扑、邻域重叠和状态成本。并行跨度另由依赖偏序决定。

区域控制的节点证据构造归入 Read，包括所选拼接投影或分组编码；区域汇集、控制头和分配求解归入 SelStep。固定控制无需计算证据；其余准备、状态更新与 Full 按实际绑定计量。
