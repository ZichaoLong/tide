---
type: mathematical-model-instances
status: active
cssclasses:
  - textbook-math
tags:
  - tide
  - mathematics
  - instances
  - weighted-message
  - hb-lattice
---

# HB-Lattice 上的带权消息实例

本文在 [HB-Lattice 拓扑](../topologies/hb-lattice.md)上绑定[共同局部架构](../local-semantics/weighted-message/architecture-learning-note.md)，给出 KV、稠密发送与混合记忆实例。候选传播、发送强度和计算量由图与所选函数共同决定；[全连接分层实例](fully-connected-layered-weighted-message.md)提供另一种拓扑下的组合。

## 1. 完整的 KV 基础绑定

### 1.1 图与局部选择

取拓扑定义中的 $b,h,n,N$、区域关系 $\mathcal H_\ell$ 与偏移集 $\mathcal D_\ell$，沿用节点、区域、边 $e_{\ell,\omega,\omega',s,a}$、端口及日程，$L=2h+1,D=L+2$。全部实例在一次运行中保持空间图和时间。

以下定义一族 KV 基线，记忆描述和区域评分允许逐区域选择。表中的函数由共同架构及模块教材给定。

| 对象 | 基线固定的选择 | 仍需给定的参数及范围 |
|---|---|---|
| 图与时间 | 本节的 HB-Lattice 拓扑 | 全图的 $b,h,n,N$；逐层固定的区域关系与横向偏移集 |
| 载荷与聚合 | $P=X_v=\mathbb R^{1+d}$，区域共享 Agg | 全图宽度 $d$；逐区域的聚合参数，来源映射见 §1.3 |
| 记忆与内容 | KV 空缓存初态、KV 内容函数 | 逐区域的状态宽度与共享基座；逐节点的适配器、摘要、检索与 Full 参数 |
| 描述与选择 | Read$^+$、拼接局部评分、固定增益、二次正则约束分配 | 逐区域选择时间摘要或内容检索、独立或区域 Attention 评分；逐节点评分参数及固定增益；逐区域 $k_j,\eta_j,\tau_{\mathrm{sel},j}$，$1\le k_j\le n$；增益满足共同有限上界 |
| 状态采用与发送 | 全部候选保留更新、单点区域历史、广播 Full、$K_j=n$ | 使用共同 Next，无额外状态采用参数 |
| 窗口与位置 | §1.2 的位置尺度与衰减换算 | 逐区域的窗口、短期衰减、Read 与 Full 旋转参数 |
| 外部边界 | §1.4 的单位入口和残差读出 | 内容序列 $h_t$、读出矩阵与尺度 |

本节给出基础绑定所需的选择、来源映射与外部边界。每个完整参数取值确定一个模型；第 2 节给出具体取值及运行，第 3 节推导传播性质，第 4 节给出其他局部选择及各例的最终绑定。相连节点共用载荷宽度，描述与评分按共同架构的组合条件检查相容性。

### 1.2 KV 的位置、窗口与衰减

采用 [KV 教材的时间换算](../local-semantics/weighted-message/kv-memory-learning-note.md#附录-b输入位置与逻辑时间的换算)。对第 $\ell$ 层节点取 $o_v^{\mathrm{pos}}=\ell,s_v^{\mathrm{pos}}=D$，即

$$
\operatorname{Pos}_{v_{\ell,\omega,s}}(\theta)=\frac{\theta-\ell}{D},
\qquad \operatorname{Pos}_{v_{\ell,\omega,s}}(Dt+\ell)=t.
\tag{1}
$$

为每区域 $j$ 选择输入位置窗口 $H_j\in\mathbb N_{\ge1}\cup\{\infty\}$ 和短期率 $\bar\beta_{j,\mathrm{short}}>0$，绑定 $T_v=DH_j$、$\beta_{v,\mathrm{short}}=\bar\beta_{j,\mathrm{short}}/D$，长期率为零。窗口覆盖最近 $H_j$ 个输入位置或全部接收历史。Read 与 Full 的旋转维数及频率分别声明，均使用式 (1) 的位置坐标。

### 1.3 来源类别

输入来源类别为 $\mathrm{in}$。对每条边 $e:v\to v'$，令

$$
\operatorname{SrcClass}_{v'}(\mathrm{msg},e)=v,
\qquad
\operatorname{SrcClass}_{v_{1,\varnothing,s}}(\mathrm{ext},i_s)=\mathrm{in}.
\tag{2}
$$

每个接收区域的类别集为其全部潜在父节点身份，第一层加入口类别。该映射覆盖全部原子标签，消息仍保留原始边身份与时间。不同节点的入邻域通常不同，共享 Agg 参数也会得到不同的内容与历史。

### 1.4 输入、读出与任务

给定只依赖输入前缀的 $h_t\in\mathbb R^d$，在所有入口注入 $x_{i_s}(t)=(1,h_t)$。记完整输出记录为 $Z$，采用共同架构的 ReadOut。取 $W_{\mathrm{out}}\in\mathbb R^{d\times d}$、$\alpha_{\mathrm{out}}\ge0$，定义输出槽与读出

$$
\begin{aligned}
\Lambda_t&=\{(Dt+L,o_s):s\in\mathbb Z_n\},\\
O_t&=\operatorname{ReadOut}_{\Lambda_t}(Z),\qquad
 y_t=h_t+\alpha_{\mathrm{out}}W_{\mathrm{out}}O_t,
\end{aligned}
\tag{3}
$$

在 $Dt+L+1$ 比较已经确定的输出槽。通用 ReadOut 对全部终端缺席取零；本实例每位置均有正强度终端输出，见 §3，内容相消仍可使 $O_t=0$。读出发生在图外。独立语言模型可用 token 嵌入作为 $h_t$，以 $y_t$ 接任务读出。

## 2. 三层图上的分叉、汇合与记忆

取 $b=2,h=1,n=3,N=2$，使用基础区域关系 $\mathcal H_\ell^0$ 与偏移 $\mathcal D_\ell=\{0,1\}$。三层区域依次为

$$
(1,\varnothing),\qquad (2,0),(2,1),\qquad (3,\varnothing).
$$

这里 $0,1$ 表示长度一的地址。每个源节点向每个相连目标区域的横向坐标 $s,s+1\bmod3$ 广播。日程间隔 $D=5$，两个输入为 $h_0=1,h_1=-1$，各位置的三层处理时间为 $1,2,3$ 和 $6,7,8$。

采用时间摘要、Read$^+$、独立评分和二次正则约束分配，全部区域取 $k_j=1,\eta_j=0,\tau_{\mathrm{sel},j}=1/8$，固定增益为 $1$，正典容量 $K_j=3$。取 $d=d_s=H_{\mathrm{att}}=d_h=r_a=1$，$\epsilon=1$，聚合残差尺度为零。KV 取无限窗口、无位置旋转、零距离偏置，短期衰减率为 $\log2/D$；共享基座为 $[0;\sqrt2]$，适配器为零，摘要投影为 $\sqrt2$。因此在内容 $u=\pm1$ 处，写入的 value 和摘要都是 $\pm1$。Full 取 $\alpha_{\mathrm{att}}=\alpha_{\mathrm{ff}}=0$，故 $G_v=u_v$；记忆参与区域评分。其余未使用的参数取零。

评分按横向坐标 $0,1,2$ 排列。第一层取

$$
(s_0,s_1,s_2)=
\left(\tfrac12,\tfrac34-M_{v_{1,\varnothing,1},\mathrm{long}},-2\right),
$$

另外三个区域分别取固定评分

$$
s_{(2,0)}=(2,1,0),\qquad
s_{(2,1)}=(0,1,2),\qquad
s_{(3,\varnothing)}=(0,2,0).
$$

这些常数和线性记忆评分均可由拼接评分实现，取评分隐藏宽度 $d_r=2$，使用 $\operatorname{SiLU}(x)-\operatorname{SiLU}(-x)=x$ 表达线性项，其余宽度取 $1$。每个区域只在本次实际候选上使用相应评分。

位置 $0$ 的第一层全部节点收到 $(1,1)$，各写入一条权重为 $1$ 的记忆，新状态的长期摘要为 $1$。Describe 返回各节点的 $(w,\mu,s^{\mathrm{loc}})$，SelStep 在评分 $(1/2,-1/4,-2)$ 下给出 $(1,0,0)$。只有横向坐标 $0$ 执行 Full，向两个子区域中的坐标 $0,1$ 各发送 $(1,1)$。

两个子区域先更新这两个候选的记忆，再分别激活坐标 $0$ 与 $1$。它们向最后一层广播；坐标 $1$ 汇合来自两个区域的消息，坐标 $0,2$ 各收到一条。最后一层的接收强度为 $(1,2,1)$，聚合内容均为 $1$，截断写入权重均为 $1$。评分选择坐标 $1$，终端收到 $(1,1)$。

位置 $1$ 的第一层写入 $-1$ 后，长期摘要变为零，评分为 $(1/2,3/4,-2)$，改由坐标 $1$ 激活。完整路径如下。接收强度族和控制族按 $0,1,2$ 排列；表中非候选的接收强度补零，仅用于展示。

| 位置 | 区域 | 候选横向坐标 | 接收强度族 | 控制族 |
|---|---|---|---|---|
| $0$ | $(1,\varnothing)$ | $\{0,1,2\}$ | $(1,1,1)$ | $(1,0,0)$ |
| $0$ | $(2,0)$ | $\{0,1\}$ | $(1,1,0)$ | $(1,0,0)$ |
| $0$ | $(2,1)$ | $\{0,1\}$ | $(1,1,0)$ | $(0,1,0)$ |
| $0$ | $(3,\varnothing)$ | $\{0,1,2\}$ | $(1,2,1)$ | $(0,1,0)$ |
| $1$ | $(1,\varnothing)$ | $\{0,1,2\}$ | $(1,1,1)$ | $(0,1,0)$ |
| $1$ | $(2,0)$ | $\{1,2\}$ | $(0,1,1)$ | $(0,1,0)$ |
| $1$ | $(2,1)$ | $\{1,2\}$ | $(0,1,1)$ | $(0,0,1)$ |
| $1$ | $(3,\varnothing)$ | $\{0,1,2\}$ | $(1,1,2)$ | $(0,1,0)$ |

每次有效评分的首位间隔都大于温度 $1/8$，本例的正温度二次分配已经给出上述零或一控制。所有实际消息的内容均为当前 $h_t$，所以 $O_t=h_t$；取 $W_{\mathrm{out}}=\alpha_{\mathrm{out}}=1$，有 $y_t=2h_t$。

节点 $v_{2,0,1}$ 在时间 $2$ 接收但未激活，保存 $+1$；在时间 $7$ 再写入 $-1$ 后激活，其长期摘要为零。同区域的坐标 $2$ 在时间 $2$ 没有接收，时间 $7$ 才写入第一条，长期摘要为 $-1$。这两种状态来自不同接收历史。第一层坐标 $1$ 的当前评分也使用了两次输入的摘要；两位置的激活者不同，沿既有空间边形成了不同的实际路径。

## 3. 区域到达与节点稀疏

固定第 1 节的图、单位入口及广播规则。各区域可分别采用区域教材的三种约束分配、直接 Attention、固定总强度 Attention、总量缩放 Attention、按比例分配或硬选择，并满足所选函数的参数条件。以下先给出这些组合共有的区域传播，再分别研究发送总量与节点覆盖。

### 3.1 每个区域都有正发送

[拓扑的邻域定义](../topologies/hb-lattice.md#2-相邻层邻域)使源区域的每个节点都连向每个相连目标区域，且 $\mathcal D_\ell$ 非空。因此一个区域只要有正强度发送，每个相连目标区域就都有有效候选。第一层收到单位输入，非空选择沿递归区域连接传播，故每个输入位置的所有区域均有节点接收并激活，终端输出存在。

这一归纳只要求有正接收的区域产生正发送。每个区域至少保留一个激活节点，因此每位置的 Full 次数至少为区域数；尚不能据此确定每个节点的接收强度或候选覆盖。

### 3.2 发送总量与单节点接收

若全图各区域均采用节点增益与固定阈值分配，还满足每区发送强度之和至少为 $1$。第一层有 $M_j=n$；若某父区域发送强度之和至少为 $1$，相连目标区域所有节点的接收强度之和至少为 $1$，并有

$$
M_j=\sum_{v\in\mathcal R_j}\min(1,\gamma_v r_v^{\mathrm{in}})
\ge\min\left(1,\sum_{v\in\mathcal R_j}r_v^{\mathrm{in}}\right)=1,
$$

其中未接收节点按 $r_v^{\mathrm{in}}=0$ 计。总和下限于是为 $1$，逐层归纳成立。

两个区域控制候选使用接收下界。在同一归纳中，
$W_j^{\mathrm{recv}}=\sum_v\min(1,r_v^{\mathrm{in}})
\ge\min(1,\sum_vr_v^{\mathrm{in}})=1$，
所以 $b_j^{\mathrm{recv}}=1$，区域阈值与指定总量同样至少发送总强度 $1$。这三个约束候选可以在各区域混用并保持该结论；单节点仍可能弱接收。其他发送绑定的结论分别为：

| 发送绑定 | 发送强度结论 |
|---|---|
| 全图各区域均用硬选择 | 单位入口逐层产生单位强度消息，每区发送总强度至少为 $1$ |
| 某区域使用固定总强度 Attention | 由 §3.1，该区域有正接收，故发送总强度为 $1$ |
| 直接 Attention 或其他混合组合 | §3.1 保证每区有正发送；总强度按各区域所选函数计算 |

区域内的候选可只占一小部分，具体节点随上游选择与邻域变化。正温度运行中，单节点接收强度和写入权重可以小于 $1$；KV 节点的缓存长度也可不同。全图采用三种约束分配时，单节点弱接收与区域发送总量至少为 $1$ 可以同时成立，区域内可调整的分配范围由[区域分配的可行集与求解](../local-semantics/weighted-message/region-allocation-learning-note.md#41-约束分配) 判断。

### 3.3 全部节点激活的条件

若全部区域在正温度使用固定总强度 Attention、直接 Attention、总量缩放 Attention、按比例分配、bern 或 ent 的 $\alpha=1$，第一层全部节点都发送，零偏移边让下一层全部节点都有正接收，归纳得到全图所有节点均激活；消息仍只沿既定 HB 边传播。quad 与 ent 的 $\alpha>1$ 可以在正温度通过精确零减少这里的激活数量。各选择函数的写入仍按实际 $w(r^{\mathrm{in}})$；发送强度另由所选函数决定。

## 4. 局部选择与配置实例

本节各例共用第 1 节的来源分类、区域共享 Agg、单位入口、读出、Read$^+$、Next 与广播。下表列出最终的记忆、评分和发送选择；三层算例与七层配置分别使用自己的图尺寸。

| 实例 | 记忆与描述读取 | 局部／区域评分 | 发送与容量 |
|---|---|---|---|
| §4.1 三层发送比较 | 各区域 KV、sum | cat、ind，候选评分相等 | 比较直接、固定总强度、总量缩放 Attention 与约束 Shannon；$K_j=4$ |
| §4.2 七层 KV | 层 $1$—$3$ 为 qry，其余层 sum | cat；层 $4$ 为 ind，其余 att、记忆与统计 key | constrained、quad、$k_j=2,K_j=64$ |
| §4.3 七层混合记忆 | 层 $4$ 为循环记忆、state；其余层的 KV 读取按上一行 | cat；层 $4$ 为 ind，其余 att、记忆与统计 key | constrained、quad、$k_j=2,K_j=64$ |
| §4.4 区域控制 | 七层 KV，或仅在层 $4$ 使用循环记忆、state | cat；层 $4$ 为 ind，其余 att、记忆与统计 key；区域控制描述 | 指定总量或区域阈值，可选各自配对函数；$k_j=2$，容量按所选函数 |
| §4.5 内容 key | 同上一行的两种记忆布局 | cat；选定区域 att、内容 key，其余区域按层取 ind／att；区域控制描述 | 指定总量或区域阈值，可选各自配对函数；$k_j=2$，容量按所选函数 |

§4.2 的表提供七层图、共同尺寸及 KV 参数，§4.3 的表提供循环参数，§4.4 的控制表提供总量或阈值参数。每份实例直接指定这些部件的组合，其余矩阵、偏置与尺度在第 1 节及所选模块的范围内给定。参数形状按最终函数检查。

这些选择也可组成其他模型；内容 key 可用于任一相容的 att 评分。HB 的节点接收由邻域与上游激活共同确定，本节的例子分别展示强度衰减、固定时间上的稀疏记忆和区域发送调节，组合性质见第 5 节。

### 4.1 稠密发送的强度传播

本节在同一小图上比较发送规则。取 $b=2,h=1,n=4$、$\mathcal H_\ell=\mathcal H_\ell^0$、$\mathcal D_\ell=\{0\}$，各层区域数为 $1,2,1$；固定增益 $\gamma=1$，令每区域各候选评分相等。该评分条件可由独立评分和相同的局部评分常数实现。记忆取第 1 节的 KV、sum，聚合与外部边界使用共同绑定；发送容量取 $K_j=4$。

先令全图采用直接 Attention，其总发送强度是各候选上限的加权平均。单位入口给出：

| 层 | 每节点接收强度及上限 $r_v^{\mathrm{in}}=\mu_v$ | 每节点发送强度 $c_v$ | 每区域总发送强度 |
|---|---|---|---|
| $1$ | $1$ | $1/4$ | $1$ |
| $2$ | $1/4$ | $1/16$ | $1/4$ |
| $3$ | $1/8$ | $1/32$ | $1/8$ |

第二层每节点接收一个父节点；第三层每节点汇合两个父区域中对应节点的发送。所有节点均激活，终端有正强度输出，同时总强度逐层减小。基础连接的展开侧在相等评分、相同节点上限、$\gamma=1$ 时，有 $\mu_{\ell+1}=(|\mathcal D_\ell|/n)\mu_\ell$。传播到输出的正性与单位强度保障因而各有条件。

保持图、单位入口、增益及相等评分，改用固定总强度 Attention。uniform 与 mu 在各区域内均分权重，得到：

| 层 | 每节点接收强度 $r_v^{\mathrm{in}}$、写入权重 $w_v$ 及 $\mu_v$ | 每节点发送强度 $c_v$ | 每区域总发送强度 |
|---|---|---|---|
| $1$ | $1$ | $1/4$ | $1$ |
| $2$ | $1/4$ | $1/4$ | $1$ |
| $3$ | $1/2$ | $1/4$ | $1$ |

此时每个区域的发送总强度均为 $1$，各节点的写入权重仍取实际接收强度。

两张表共同说明：这个小图上，非空传播可以伴随不同的强度变化。总量缩放 Attention 在相同算例中也逐区域发送总量 $1$；$k_j=1$ 的约束 Shannon 具有与第二张表相同的前向。这个等值依赖相等评分。例如第二层 $M_j=1$，Shannon 的分配由约束固定，固定总强度 Attention 在评分不等时仍可重新分配。

### 4.2 七层配置与参数形状

采用[拓扑 §4 的七层形状](../topologies/hb-lattice.md#4-图规模)：$b=2,h=3,n=64$，$\mathcal H_\ell=\mathcal H_\ell^0$，$\mathcal D_\ell=\{0,1\}$。连接更丰富的比较形状取偏移 $\{0,1,2,3\}$。

各尺寸按其计算位置给定：

| 符号 | 对应对象 | 取值 |
|---|---|---|
| $d$ | 消息与聚合的内容宽度 | $512$ |
| $p_a$ | 聚合残差中的隐藏宽度 | $32$ |
| $H_{\mathrm{att}},d_h$ | KV Attention 头数、每头宽度，乘积为 $d$ | $8,64$ |
| $r_a$ | 节点 KV 适配器的秩上限 | $32$ |
| $d_s$ | 每个记忆描述向量及当前摘要特征的宽度 | $32$ |
| $d_r$ | 局部评分与增益网络的隐藏宽度 | $32$ |
| $d_{\mathrm{sel}}$ | 区域 query 与 key 的匹配宽度 | $32$ |
| $d_f$ | Full 前馈网络的隐藏宽度 | $2048$ |

各区域取 $k_j=2$，采用区域共享聚合、拼接局部评分、固定增益和二次正则约束分配。记忆描述与区域评分按下表选择：

| 层 | 记忆与评分 | $H_j$ | Full 旋转 | Read 旋转 |
|---|---|---|---|---|
| $1,2,3$ | 内容检索描述、区域 Attention 评分 | $512$ | 每头完整 $64$ 维 | $d_s=32$ 维全部旋转 |
| $4$ | 时间摘要、独立评分 | $\infty$ | NoPE | 时间摘要无需检索旋转 |
| $5,6,7$ | 时间摘要、区域 Attention 评分 | $512$ | 每头部分 $32$ 维 | 时间摘要无需检索旋转 |

对所有旋转族取基数 $10^4$、频率 $(10^4)^{-2(a-1)/d^{\mathrm{rot}}}$，其中 $1\le a\le d^{\mathrm{rot}}/2$；Full 的距离偏置取零。其余标量和矩阵由配置给出。这个形状在展开侧采用内容检索与区域匹配，中间层保留全历史，收拢侧采用摘要与区域匹配；也可逐区域代入已定义的其他组合。

### 4.3 KV 与循环记忆的组合

在选定区域替换记忆时，采用相应模块的状态空间、零初态、Upd、Mem、当前特征与 $G_v$；保留图、时间、来源和外部边界，描述宽度与区域评分保持相容。区域之间可以选择不同模块。

本例的七层图与共同尺寸取 §4.2 的表。中间层 $4$ 的每个区域选择下表的一种循环记忆，取 $d_s=32,d_f=2048$、memory_read=state；其余层取 §4.2 表中的 KV 读取、窗口与位置。全部区域取 cat、fixed 增益、二次正则约束分配、$k_j=2,K_j=64$；层 $4$ 使用 ind，其余层使用 att 评分及记忆与统计 key。

| 记忆 | 状态形状与参数 |
|---|---|
| [affine／delta](../local-semantics/weighted-message/recurrent-memory/affine-delta-learning-note.md) | $d_k=d_m=64$，两种时间尺度，短期率 $\bar\lambda_{\mathrm{short}}/D$，$\bar\lambda_{\mathrm{short}}>0$，长期率零 |
| [gated_delta](../local-semantics/weighted-message/recurrent-memory/gated-delta-learning-note.md) | $4$ 头，$d_k=64,d_m=128$，矩阵被动率零，qk_eps=$10^{-6}$ |
| [ssd](../local-semantics/weighted-message/recurrent-memory/ssd-learning-note.md) | $4$ 头、$1$ 组，$d_k=64,d_m=128$，矩阵被动率零 |

Gated DeltaNet 与 SSD 共同取 output_norm_eps=$10^{-5}$，摘要短期率 $\log2/D$、长期率零，conv_size=4、conv_stride=$D$。全部有限矩阵、偏置、门参数、卷积系数及残差尺度按所选模块定义给出。新的 Mem 与当前特征接入原评分构造；消息宽度仍为 $512$，描述宽度仍为 $32$。循环区域使用自身的时钟与状态参数，KV 的窗口和 RoPE 字段只用于其余 KV 区域。

HB 的节点可能在一些位置没有接收。短卷积仍按 $t,t-1,t-2,t-3$ 取槽，缺失位置为零，正接收历史写入 $wz$；后续位置不因缺失而前移。GDN 的保留门 $g_h$、误差写入门 $\beta_h$ 与 SSD 的有效步长 $d_h$ 由当前接收及内容决定；跨越的逻辑时间通过各自的被动衰减另行作用。此绑定使弱事件消失时仍满足对应模块的连续性引理。

候选传播和消息计数公式只依赖图与实际发送，§3 和第 6 节的相应结论继续成立。更换记忆可以改变评分、激活者和实际路径，计数按新运行确定。缓存长度只描述 KV 区域；循环矩阵与短卷积具有固定大小，并在全部正接收候选上更新。区域 query 只要求描述类型相容，不要求不同区域有相同状态空间。

按层取得完整驱动后，调用各区域自己的状态与 Full 契约。带短卷积的模块先求卷积，随后求矩阵递归。消息宽度、描述类型和拓扑时间均匹配，组合是合法 SettleGraph，并保留正典切面继续。数值性质应用共同架构 §6，各记忆分别提供局部引理。

### 4.4 指定总量与区域阈值

本例的七层图与共同尺寸取 §4.2 的表。记忆布局选择七层 KV，或仅在层 $4$ 使用循环记忆；KV 的读取、窗口和位置使用 §4.2 的表，循环的 state 与参数使用 §4.3。全部区域取 cat，层 $4$ 使用 ind，其余层使用 att 评分及记忆与统计 key。发送参数取 gamma_max=$2$、region_control_width=$32$、$d_r=32$、quad、$k_j=2$、$K_j=64$，各区域给定正温度。Describe 使用[区域分配 §2.3](../local-semantics/weighted-message/region-allocation-learning-note.md#23-节点评分与节点描述)、[§3.6.3](../local-semantics/weighted-message/region-allocation-learning-note.md#363-扩展节点描述)的相应 $D^{\mathrm{reg,ind}}$／$D^{\mathrm{reg,att}}$，att 固定 query_weight=received，逐节点 send_gain 不参与。

| 发送函数 | region_control_mode=fixed | region_control_mode=context |
|---|---|---|
| regional_total | total_fraction=$1/2$，零证据 | 给定节点证据构造与总量头 |
| regional_threshold | eta=$0$，零证据 | 给定节点证据构造与阈值头 |

context 分支逐区域选择 region_evidence=cat／grouped，未另声明时取 cat；grouped 沿用 $d_{\mathrm{sel}}=32$，给定区域教材 §3.2 的证据编码与 §3.4.1 的输出参数，局部评分与区域评分按本节的层别给定。fixed 不使用 region_evidence。每个区域选择一行及一种控制形式，并给定对应全部有限参数，即得到完整绑定。指定总量按区域教材 §2.2 产生 $B_j$ 并调用 §4.1.1；区域阈值按该教材 §4.1.2 产生 $\eta_j$ 并完成分配；其余函数按共同架构计算。

由 §3.2，各区域 $b_j^{\mathrm{recv}}=1$，而节点接收与 $\mu_v^{\max}$ 可不相同。指定总量的固定比例给
$B_j=(1+\min(2,\sum_v\mu_v^{\max}))/2$；区域头能否调节取决于区间是否有余量。阈值候选允许总和在同一区间内由评分与阈值共同确定。局部弱接收的数值例子见[区域分配 §5.3](../local-semantics/weighted-message/region-allocation-learning-note.md#53-弱接收的发送范围)。

两个候选均保持接收趋零时本节点发送趋零。正温度支持取决于正则；零温比较保留各自的区域头及描述，按 §5.3 区分目标。控制证据不会增加节点记忆或区域持久状态。

显式求值各自的配对运行时，全图采用指定总量可取 regional_total_zero，保持 $K_j=64$；全图采用区域阈值可取 regional_threshold_hard，改取 $K_j=k_j=2$。两者均保留对应描述、评分与区域头，移除温度和正则参数。混合不同配对函数时，发送总量与极限条件按所选组合另行核对。

### 4.5 内容 key 与局部评分

本例的七层图与共同尺寸取 §4.2 的表，记忆布局选择七层 KV，或仅在层 $4$ 使用循环记忆。KV 的读取、窗口与位置取 §4.2，循环的 state 与参数取 §4.3。所有区域使用 cat 局部评分，隐藏宽度 $d_r=32$。

在选定区域取 region_score=att、region_key=content、alpha_loc=alpha_sel=$1$、query_weight=received，$d_r=d_{\mathrm{sel}}=32$；描述使用 $D^{\mathrm{reg,att}}$，给定[区域分配 §3.6](../local-semantics/weighted-message/region-allocation-learning-note.md#36-区域-attention-评分)的当前内容投影、独立内容 key 编码，以及该教材 §3.6.4 的区域 query 参数。其余区域中，层 $4$ 使用 ind 与 $D^{\mathrm{reg,ind}}$，其他层使用 att、记忆与统计 key 及 $D^{\mathrm{reg,att}}$；所有 att 的 query 权重取 received。

每个区域的发送独立选择指定总量或区域阈值，控制参数使用 §4.4 的控制表：gamma_max=$2$、$d_{\mathrm{reg}}=32$、$k_j=2$，fixed 使用表中的固定值；context 逐区域选择 cat／grouped 证据，并给定证据及头参数，未另声明时取 cat。正温度分配取 quad、$K_j=64$；显式使用配对函数时，指定总量零温分配仍取 $K_j=64$，区域阈值硬选择取 $K_j=2$。各函数的温度参数按其定义给定。

这组绑定让共同 query 匹配节点本次取得的记忆内容，接收与写入统计仍可经局部评分和区域控制影响发送。给定全部有限参数后得到完整模型，按第 5 节核对组合及新运行的评分条件。

## 5. 组合条件与分段继续

### 5.1 类型与容量

相连节点共用消息宽度，各区域的描述与评分类型相容，图与时间满足单次结算条件。硬选择取正典容量 $K_j=k_j$，其余发送取 $K_j=n$，即可代入 SettleGraph 的确定递归。

### 5.2 精确分段继续

本节适用于上述相容的 KV 与循环组合，以及各区域已选定的发送函数。精确分段继续按同一个固定模型比较，包括使用固定总强度 Attention 的组合。

给定位置区间 $[p,q)$ 的左切面 $Q_{Dp}$ 及完整端口输入，可以逐层求值。先取得该层所有区域的完整驱动纤维，再按区域调用所选模块的 BatchRegionState 和逐节点 BatchFull。KV 使用其教材 §6 的前缀与读取构造，循环使用相应教材 §4 的卷积准备及仿射复合，由此确定下一层所需的消息。基础 affine／delta 无卷积步骤。同层区域之间没有消息依赖，层内多个区域可独立求值。稀疏边只改变需要汇集的父节点集合。

在 $Dq$ 处没有跨界消息，继续保留节点状态、单点历史和原始位置，带短卷积的状态还包含原始时间槽。切在位置内部时须保留 $W_b$。Read 的内容检索和 Full 均只读取各自快照及原始时间，整段、连续多 chunk 与逐位置求值满足同一精确记录契约。查询、缓存与消息的工作量由第 6 节计量。

### 5.3 数值性质的附加条件

固定 $\mathcal H_\ell,\mathcal D_\ell$、区域、时间和局部配置后，数值性质应用[共同架构 §6](../local-semantics/weighted-message/architecture-learning-note.md#6-数值性质的组合)。KV 与循环模块分别验证状态条件，区域控制按区域分配 §8 核对。有限时间的组合结论覆盖本实例的稀疏邻域。

全图采用指定总量时，使用[区域分配 §7.2](../local-semantics/weighted-message/region-allocation-learning-note.md#72-指定总量的零温分配)：在各次线性最优解唯一的条件下收敛到自身零温运行；quad 与固定 $\alpha>1$ 的 ent 同样可在充分小正温度精确达到它。该运行可能发送小数强度，不能据外部单位输入推出后续节点上限二元。

全图采用区域阈值时，单位输入软硬关系使用[区域分配 §7.4](../local-semantics/weighted-message/region-allocation-learning-note.md#74-区域阈值与单位强度硬选择)；全图采用节点增益与固定阈值分配时使用[§7.5](../local-semantics/weighted-message/region-allocation-learning-note.md#75-节点增益与固定阈值的硬化)。若全部区域采用 quad 或固定 $\alpha>1$ 的 ent，且对应硬运行满足评分条件，则在本实例的单位外部输入、固定参数和有限时间范围内，充分小正温度使完整记录及切面继续状态精确等于硬运行。所需温度界按该次硬运行确定。

混合不同发送函数时，逐区域验证共同架构的局部极限条件。

硬选择每个区域每位置激活 $1$ 至 $k_j$ 个候选，激活者随输入和历史改变。既有空间边可在其他位置继续使用。连接较丰富的 HB 和较稀疏的 HB 各自在自己的固定图上使用同一结论。永久删边改变空间图；在同一图上，局部选择决定各次实际使用的边。

## 6. 候选传播与全图成本

### 6.1 候选与消息计数

记 $\mathcal C_\ell(t),\mathcal A_\ell(t)$ 为位置 $t$ 在第 $\ell$ 层全部区域的候选、激活集合，$I_\ell(t)$ 为该层存在外部注入的节点集合，$N^+(v)$ 为出邻域。本文 $I_1(t)$ 为第一层全体节点，其余为空。各选择函数都按正发送强度广播，消息的存在与激活对应，因此

$$
\begin{aligned}
\mathcal C_{\ell+1}(t)
 &= I_{\ell+1}(t)\cup
    \bigcup_{v\in\mathcal A_\ell(t)}N^+(v),\\
|\mathcal C_{\ell+1}(t)|
 &\le |I_{\ell+1}(t)|+\Delta_\ell|\mathcal A_\ell(t)|,\\
\#\mathrm{message}_{\ell\to\ell+1}(t)
 &=\sum_{v\in\mathcal A_\ell(t)}|N^+(v)|,
\end{aligned}
\tag{4}
$$

其中 $\Delta_\ell$ 为该层最大出度。显式保留零 Full 及零广播会产生额外候选。满足[共同架构命题 1](../local-semantics/weighted-message/architecture-learning-note.md#62-零消息与正温度输入)的组合保持消息投影、有效状态和 ReadOut，候选与消息计数须按实际计算规则重算。

含固定总强度 uniform Attention 的混合组合须保留消息存在掩码：缺席槽只作张量填充，不产生真实候选。把缺席改为存在的零强度消息，可能改变下游份额与前向结果；各发送函数的适用范围见[性质对照](../local-semantics/weighted-message/region-allocation-learning-note.md#81-零接收与正温度连续性)。

### 6.2 硬选择与稠密运行的计数

若全图各区域均采用硬选择，则有

$$
|\mathcal B_\ell|\le|\mathcal A_\ell(t)|\le
 \sum_{\omega\in\mathcal B_\ell}k_{(\ell,\omega)},
\qquad
|J|\le\#\mathrm{Full}(t)\le\sum_{j\in J}k_j.
\tag{5}
$$

对 §4.2 的七层形状，各区域改用 $k_j=2$ 的硬选择时，每位置 Full 数在 $22$ 至 $44$ 之间。第一层全体 $64$ 个节点准备，其中一至两个激活；第二层的两个区域都会到达，总候选至多为 $2\times4=8$。后续各区的节点覆盖由实际选择及邻域重叠决定，所有区域始终有非空候选。

同一形状若全部区域采用固定总强度 Attention、直接 Attention、总量缩放 Attention、按比例分配、bern 或 ent 的 $\alpha=1$，正温度下每位置为 $1408$ 个候选、$1408$ 次 Full、$3584$ 条内部消息。发送强度即使很小，这些工作仍然存在；各节点的写入权重可小于 $1$。混合选择时按式 (4) 逐层计数，不能直接套用全稠密或硬运行数字。

### 6.3 一次发送对下游准备的影响

若下一层有 $1000$ 个节点、当前有 $10$ 个激活者且没有其他注入，全连接会使下一层 $1000$ 个节点全部准备；出度至多 $4$ 时，候选至多为 $40$。减少一次上游发送可能同时省去下游的聚合、记忆投影与写入、Read、选择、Full 和进一步发送。

在其他发送者保持不变的条件下，撤去 $v$ 的发送会直接移除的候选恰为

$$
N^+(v)\setminus\left(
 I_{\ell+1}(t)\cup
 \bigcup_{u\in\mathcal A_\ell(t)\setminus\{v\}}N^+(u)
\right).
\tag{6}
$$

因此父邻域重叠和其他注入会削弱这项节约。式 (6) 是保持其他发送者不变的覆盖比较；实际选择须保留本区域的至少一个激活者，撤去唯一激活者时必须另选节点，其邻域也须计入。后续内容、评分和选择也可能改变，整图总工作量不能由单个发送强度的变化直接判定为单调下降。

当前已收到消息的节点，即使未激活，仍完成本次准备并保留写入。Read、选择与下游工作按共同架构附录 S.2 分类，具体状态和算术成本由所选记忆模块确定。

本实例的训练对照见 [可选研究问题与对照](../memos/training/weighted-message-research-questions.md)，图、局部函数与继续记录的程序对应见 [实现核对](../reference/weighted-message-implementation-checks.md)。

## 附录 S：配置对应

第 1 节采用 memory_family=kv、read_mode=+、local_score=cat、send_gain=fixed、selection_mode=constrained、regularizer=quad。第 4.2 节按层给出 memory_read=sum／qry 和 region_score=ind／att；query_weight 未另声明时取 mu，att 的 region_key 未另声明时取 content_stats。第 4.3 节中间层逐区域选 memory_family=affine／delta／gated_delta／ssd、memory_read=state，其余绑定按表给出。

固定尺度的稠密区域 Attention 使用 region_score=att、alpha_loc=0、alpha_sel=1 及固定正 allocation_temperature。发送可取 unit_attention、direct_attention、pooled_attention 或约束 Shannon；它们在弱接收处按第 4.1 节产生不同结果。unit_attention 的 allocation_base 为 uniform／mu，direct_attention 为 mu／received，pooled_attention 固定 mu。

第 4.4 节使用 regional_total／regional_threshold，配套区域描述与 region_control_mode=fixed／context；固定控制分别读取 total_fraction 或 eta，上下文控制读取区域头并按 region_evidence 选择 cat／grouped 证据。region_control_width=$32$，gamma_max=$2$；att 使用 received，send_gain、allocation_base 不参与。其余取值和参数生效范围统一见[区域分配附录 S.1](../local-semantics/weighted-message/region-allocation-learning-note.md#s1-配置名称)。

第 4.5 节在继承的区域控制配置上采用 content key 与非零局部评分，完整替换字段由该节给出；记忆与区域证据继续独立绑定。
