---
type: mathematical-index
status: active
tags:
  - tide
  - mathematics
  - semantic-anchor
---

# 数学符号与函数对应索引

本页汇集四篇正典、局部函数、拓扑与完整实例的共同名称。局部特化选择空间与函数，拓扑文档定义图与时间，实例引用这些定义并给定边界与参数。各定义的推导与证明由所列教材给出。

## 定义归属与适用条件

| 教材 | 结构或函数条件 | 所承担的定义与结论 |
|---|---|---|
| [SettleGraph](../settlegraph-learning-note.md) | 严格区域分层，$\theta_{v,t}=Dt+r_v$，$\delta(a)=r_{\mathrm{dst}(a)}-r_{\mathrm{src}(a)}$，端口日程对齐 | 单次结算、位置继续与记录嵌入 |
| [TimedDAG](../timed-dag-region-selector-learning-note.md) | 有限无环消息图、有限输入及严格递增端口日程 | 唯一有限记录、规范作用、合法暴露、关闭与切面继续 |
| [PositiveDelayGraph](../positive-delay-graph-finite-cut-learning-note.md) | 有限正时延消息图，输入局部有限，允许空间有环 | 相容有限切面、区间组合与强连通分量求值 |
| [分块求值](../timed-dag-chunk-prefill-learning-note.md) | 固定图语义并给出精确联合契约 | 状态块、输出批、契约相对阶段与整窗口求值 |
| [共同局部架构](../local-semantics/weighted-message/architecture-learning-note.md) | 带权载荷、接收驱动状态、标量控制与广播 | 完整计算过程、模块组合、局部条件到整网性质的推导、精确继续与联合求值 |
| [区域分配](../local-semantics/weighted-message/region-allocation-learning-note.md) | 选定评分、节点上限、区域控制与发送函数 | Describe 与 SelStep 的具体定义、求解与导数；指定总量的线性最优极限，区域阈值、节点增益与固定阈值的单位输入硬化 |
| [KV](../local-semantics/weighted-message/kv-memory-learning-note.md)、[基础循环](../local-semantics/weighted-message/recurrent-memory/affine-delta-learning-note.md)、[Gated DeltaNet](../local-semantics/weighted-message/recurrent-memory/gated-delta-learning-note.md)、[SSD](../local-semantics/weighted-message/recurrent-memory/ssd-learning-note.md) | 各自的状态、更新、描述与内容函数 | 完整局部绑定、状态性质及共同组合定理所需的假设 |
| [固定时间短卷积](../local-semantics/weighted-message/recurrent-memory/causal-convolution-learning-note.md) | 原始时间槽与带权投影历史 | Gated DeltaNet／SSD 共用的部件，弱事件与快照读取引理 |
| [全连接拓扑](../topologies/fully-connected-layered.md)、[HB-Lattice 拓扑](../topologies/hb-lattice.md) | 节点、区域、边、端口与日程 | 独立图定义、时间对齐、单次结算与空间规模 |
| [全连接实例](../instances/fully-connected-layered-weighted-message.md)、[HB-Lattice 实例](../instances/hb-lattice-weighted-message.md) | 所选拓扑、局部函数、参数、初态与输入读出 | 完整模型绑定、条件性的传播保障、联合求值与实际工作量 |

SettleGraph 向 TimedDAG 保留图对象、端口和局部函数，重索引位置坐标并补齐空档状态；TimedDAG 记录逐切面限制后得到 PositiveDelayGraph 的相容记录族。分块求值研究固定语义下的联合函数。

局部架构是这些正典允许的一条特化支线。[共同架构 §6](../local-semantics/weighted-message/architecture-learning-note.md#6-数值性质的组合)给出局部条件与组合结论；[区域分配 §7—8](../local-semantics/weighted-message/region-allocation-learning-note.md#7-温度极限与硬选择)验证具体发送函数的适用范围。

## 基础对象与时间

| 符号 | 含义 |
|---|---|
| $V,A,\operatorname{src},\operatorname{dst},\delta$ | 节点、独立消息边身份、端点与正整数时延 |
| $\mathsf I,\mathsf O,\gamma,\varepsilon$ | 输入／输出端口及其所属节点 |
| $\operatorname{In}(v),\operatorname{Out}(v),\operatorname{InPort}(v),\operatorname{OutPort}(v)$ | 节点的入边、出边、输入端口与输出端口集合 |
| $J,\rho,\mathcal R_j$ | 区域标识、节点归属与区域成员 |
| $\theta,t,\iota_i(k)$ | 逻辑时间、输入位置、端口位置的注入时间 |
| $P,P_\bot$ | 消息承载值空间及增加缺席标记后的空间 |
| $\mathsf{Atom}_v,B_{v,\theta}$ | 可到达节点的带来源原子、完整时间纤维 |
| $\mathcal C_{j,\theta},\mathcal A_{j,\theta}$ | 候选集合、激活集合 |
| $S_v,X_v,D_v,\mathsf C_v,Y_j$ | 节点状态、本地内容、描述量、局部控制、选择历史空间 |
| $q,\widetilde q,q^{\mathrm{cmp}},q^{\mathrm{next}}$ | 旧状态、候选状态、计算快照、下一持久状态 |
| $\tau_j,\kappa_j,K_j$ | 描述模式、状态采用模式、激活数上界 |

SettleGraph 用 $\theta_{v,t}=Dt+r_v$ 与端口日程对齐；$q_v^{[t]}=q_v^{Dt}$、$y_j^{[t]}=y_j^{Dt}$ 是位置边界状态。其简写 $h_{v,t}$ 等均表示时间 $\theta_{v,t}$ 的同名坐标。一般图的状态上标 $\theta$ 始终为逻辑时间。

局部支线的正典容量 $K_j$ 由所选发送函数绑定：基础硬选择及区域阈值的配对硬函数取 $k_j$，其余已定义函数取区域大小，包含指定总量的零温函数。$k_j$ 在约束分配和按比例分配中限制发送强度之和，在前两种硬函数中限制激活数。选择温度 $\tau_{\mathrm{sel}}$ 与描述模式 $\tau_j$ 分别命名。

## 局部支线的符号与定义归属

### 共同接口与状态

| 符号或函数 | 含义与唯一的定义入口 |
|---|---|
| $(m,z),\pi_P,(r^{\mathrm{in}},u),\operatorname{Agg},\operatorname{SrcClass}_v$ | [共同架构 §1—2](../local-semantics/weighted-message/architecture-learning-note.md#1-消息输入与候选)：载荷、数值贡献、接收聚合及从带标签端口／入边到有限类别的映射 |
| $p_v,\operatorname{Mem}_v$ | [共同架构 §3.2](../local-semantics/weighted-message/architecture-learning-note.md#32-从状态取得记忆描述)：当前内容特征与记忆描述接口，由各记忆模块实现 |
| $\operatorname{Describe}_v[\operatorname{Mem}_v,p_v],d_v\in D_v$ | [共同架构 §3.3](../local-semantics/weighted-message/architecture-learning-note.md#33-节点描述与状态快照)：以 Mem 与当前特征为部件的描述函数；绑定后简记 Describe，具体构造由区域教材给出 |
| $q_v^{\mathrm{read}}$ | [共同架构 §3.3](../local-semantics/weighted-message/architecture-learning-note.md#33-节点描述与状态快照)：描述模式指定的初态、旧状态或新状态 |
| $G_v,\operatorname{EmitValue},\Delta$ | [共同架构 §4.2](../local-semantics/weighted-message/architecture-learning-note.md#42-完整内容与广播)：完整内容、载荷与逐边／逐端口广播；$G_v$ 由具体模块绑定 |
| $\operatorname{ReadOut}$ | [共同架构 §4.4](../local-semantics/weighted-message/architecture-learning-note.md#44-图的输出读出)：有限输出坐标上的读出 |

### 区域描述、评分与分配

| 符号或函数 | 含义与定义入口 |
|---|---|
| $w_v,\mu_v^{\max}$ | [区域分配 §2.1](../local-semantics/weighted-message/region-allocation-learning-note.md#21-接收与发送上限)：截断接收量与固定最大倍率给出的节点发送上限 |
| $\operatorname{Score}_v,s_v^{\mathrm{loc}}$ | [区域分配 §2.3](../local-semantics/weighted-message/region-allocation-learning-note.md#23-节点评分与节点描述)：从当前接收、当前特征与记忆描述产生实数评分的函数及其输出 |
| $f_v$ | [区域分配 §3.1](../local-semantics/weighted-message/region-allocation-learning-note.md#31-拼接评分)：供拼接评分、拼接区域证据和拼接上下文增益使用的特征 |
| $E_v^{h,g}$ | [区域分配 §3.2](../local-semantics/weighted-message/region-allocation-learning-note.md#32-分组匹配评分)：当前内容、记忆、统计的分组编码；score、reg、key 用途默认参数独立，key 只使用 mem 组 |
| $s_v^{\mathrm{cat}},s_v^{\mathrm{match}}$ | [区域分配 §3.1—3.3](../local-semantics/weighted-message/region-allocation-learning-note.md#33-所选局部评分)：$\operatorname{Score}_v$ 的两种具体构造所得评分；match 独立编码内容、记忆与统计量 |
| $x_v^{\mathrm{sel}},K_v^{\mathrm{sel}}$ | [区域分配 §3.6](../local-semantics/weighted-message/region-allocation-learning-note.md#36-区域-attention-评分)：区域 Attention 评分所需的当前内容投影与记忆 key；后者选择 content_stats／content，宽度相同 |
| $\Phi_v,z_v^{\mathrm{reg}}$ | [区域分配 §2.2](../local-semantics/weighted-message/region-allocation-learning-note.md#22-由节点特征确定发送总量)：从当前接收、当前特征与记忆描述产生区域证据的函数及其输出；cat／grouped 的具体构造见[§3.4](../local-semantics/weighted-message/region-allocation-learning-note.md#34-区域证据)，固定控制取零证据 |
| $D_v^{\mathrm{reg,ind}},D_v^{\mathrm{reg,att}}$ | [§2.3](../local-semantics/weighted-message/region-allocation-learning-note.md#23-节点评分与节点描述)的四元组与[§3.6.3](../local-semantics/weighted-message/region-allocation-learning-note.md#363-扩展节点描述)的六元组：供指定总量与区域阈值使用的节点描述 |
| $\gamma_v,\mu_v,D_v^{\mathrm{ind}},D_v^{\mathrm{att}}$ | [区域分配 §4.1.3](../local-semantics/weighted-message/region-allocation-learning-note.md#413-节点增益与固定阈值分配)：固定或拼接上下文增益、接收相关发送量与三元组／五元组描述 |
| $\operatorname{Describe}_v[\operatorname{Mem}_v,p_v]$ | [区域分配 §5.1](../local-semantics/weighted-message/region-allocation-learning-note.md#51-从所选函数得到完整返回)：选定各部件后，组装所需描述的完整函数 |
| $Q_j^{\mathrm{sel}},s_v$ | [区域分配 §3.6.4](../local-semantics/weighted-message/region-allocation-learning-note.md#364-query-与最终评分)：区域 query 与最终评分；ind 直接取 $s_v=s_v^{\mathrm{loc}}$，区域控制的 att 按 $w_v$ 汇集 query |
| $W^{\mathrm{recv}},M^{\max},b^{\mathrm{recv}},U$ | [区域分配 §2.1](../local-semantics/weighted-message/region-allocation-learning-note.md#21-接收与发送上限)：截断接收之和、节点上限之和，以及允许发送总量的下界与上界 |
| $\bar z_j^{\mathrm{reg}},f_j^{\mathrm{reg}},\rho_j,B$ | [区域分配 §2.2](../local-semantics/weighted-message/region-allocation-learning-note.md#22-由节点特征确定发送总量)：接收加权证据、区域头输入、区间内比例与指定总量；$B=b^{\mathrm{recv}}+\rho_j(U-b^{\mathrm{recv}})$ |
| $H_j,\eta_j$ | 区域头的具体函数见[§3.5](../local-semantics/weighted-message/region-allocation-learning-note.md#35-区域头函数)，将其用于共同阈值见[§4.1.2](../local-semantics/weighted-message/region-allocation-learning-note.md#412-区域阈值分配) |
| $c_v,\mathcal A$ | [区域分配 §1](../local-semantics/weighted-message/region-allocation-learning-note.md#1-输入与返回)：发送强度（本支线的控制量）与正发送节点组成的激活集合 |
| $M,S$ | $M=\sum_v\mu_v$ 为节点上限之和，见[§4.1.3](../local-semantics/weighted-message/region-allocation-learning-note.md#413-节点增益与固定阈值分配)；$S=\sum_vc_v$ 为实际发送总量，见[§4](../local-semantics/weighted-message/region-allocation-learning-note.md#4-发送规则的其他选择) |
| $\mathcal F(\mu;\underline S,\overline S),\operatorname{Alloc}_\tau$ | [区域分配 §4.1.1](../local-semantics/weighted-message/region-allocation-learning-note.md#411-共同优化问题与正则)：以节点上限和总量范围为参数的共同可行集与正则分配；指定总量取两端均为 $B$ |
| $c^{\mathrm{hard}}$ | [区域分配 §4.4.1](../local-semantics/weighted-message/region-allocation-learning-note.md#441-按评分选择节点)：区域阈值与节点增益的硬函数共用排序规则，分别代入各自评分及上限 |
| $c^0$ | [区域分配 §4.4.2](../local-semantics/weighted-message/region-allocation-learning-note.md#442-指定总量的零温分配)：指定总量的零温分配，按评分依次填入节点允许的发送量 |
| $F(\mu),b,k,\eta,(t_v),\tau_{\mathrm{sel}}$ | [区域分配 §4.1.3](../local-semantics/weighted-message/region-allocation-learning-note.md#413-节点增益与固定阈值分配)：节点增益与固定阈值分配的可行集、总量下限 $b=\min(1,M)$、上限、阈值、调整后评分和温度 |
| $a_v,Z,p_v^{\mathrm{sel}}$ | [Attention 份额](../local-semantics/weighted-message/region-allocation-learning-note.md#42-通过-attention-份额分配)：三种 Attention 发送规则共同使用的基权重、归一化分母与内部份额 |
| $\lambda,T$ | [区域分配 §6.1—6.2](../local-semantics/weighted-message/region-allocation-learning-note.md#61-一个标量乘子)：求解器乘子与当前由等式固定的总量；$T$ 可为起作用的上下界或模型指定的 $B$ |
| $x_v^{\mathrm{resp}},a_v^{\mathrm{resp}},e_v^{\mathrm{resp}}$ | [区域分配 §6.2](../local-semantics/weighted-message/region-allocation-learning-note.md#62-约束分配的反向)：实际响应点 $t_v-\lambda$ 及在此处对自变量、发送上限的响应导数 |
| $x_v^{\mathrm{bracket}}$ | [区域分配附录 S.2](../local-semantics/weighted-message/region-allocation-learning-note.md#s2-标量求解与隐式反向)：用于构造求根区间的逆响应点 |

### 记忆模块与共用部件

| 符号或函数 | 含义与定义入口 |
|---|---|
| $\xi_i,T_v,\chi_{v,\theta},\operatorname{Pos},\operatorname{Rot}$ | [KV §1—2](../local-semantics/weighted-message/kv-memory-learning-note.md#1-状态与接收写入)：原始写入时间、窗口、遮罩、位置与旋转 |
| $H_a,Z_a,W_a,s$ | [基础循环 §1](../local-semantics/weighted-message/recurrent-memory/affine-delta-learning-note.md#1-状态与逻辑时间)：两种时间尺度的矩阵、摘要、写入量与保存时间 |
| $(A,B),\circledast$ | [基础循环 §4.1](../local-semantics/weighted-message/recurrent-memory/affine-delta-learning-note.md#41-仿射复合)：矩阵变换 $H\mapsto HA+B$ 的系数对与先后复合；各循环模块分别给出系数 |
| $\mathcal B,\operatorname{Conv},\operatorname{ConvWrite},L_{\mathrm{conv}},\delta_{\mathrm{conv}}$ | [短卷积 §1](../local-semantics/weighted-message/recurrent-memory/causal-convolution-learning-note.md#1-缓存卷积与写入)：缓存、读取、写入、抽头数与抽头的逻辑时间间距 |
| $H_h,Q_h,k_h,\nu_h,g_h,\beta_h$ | [Gated DeltaNet §1—2](../local-semantics/weighted-message/recurrent-memory/gated-delta-learning-note.md#1-输入参数与状态)：逐头矩阵、查询、key、value、保留门与误差写入门 |
| $H_h,B_g,C_g,\nu_h,d_h,g_h,D_h$ | [SSD §1—3](../local-semantics/weighted-message/recurrent-memory/ssd-learning-note.md#1-尺寸状态与时钟)：逐头矩阵、组内写入与读取向量、内容、有效步长、保留门与直通系数 |
| $\operatorname{Mix},R,\operatorname{Mem}$ | 两份完整循环教材 §3：记忆内容混合、矩阵查询及区域描述；记忆门与发送增益 $\gamma_v$、控制 $c_v$ 分别定义 |
| $\pi_{\mathrm{KV}},\pi_{\mathrm{rec}},\pi_{\mathrm{gdn}},\pi_{\mathrm{ssd}}$ | 各记忆教材的有效状态投影，用于数值性质比较；后两者还包括 $\pi_{\mathrm{conv}}$ |

记忆教材中的 $M_a^{\mathrm{sum}},M_a^{\mathrm{qry}},M_a^{\mathrm{state}}$ 标记不同读取候选；$M_a$ 表示本次按已选方式得到的描述值。所选的两个向量与共同写入量统计组成 $\operatorname{Mem}_v(q,\theta,u)$ 的返回。state 的具体通道由各模块定义，基础循环与 Gated DeltaNet／SSD 分别使用自己的矩阵读取构造。

### 配置与名称

配置查询按定义归属定位。具体取值、默认选择和不适用字段由对应教材附录集中给出；实例只记录自己选定的值。

| 要查询的选择 | 定义入口 |
|---|---|
| memory_family、read_mode 与完整组合 | [共同架构附录 S.1](../local-semantics/weighted-message/architecture-learning-note.md#s1-配置对应)；read_mode 对应描述模式，指定 Read 所用的快照 |
| memory_read、记忆尺寸、窗口和位置 | 各记忆教材附录；memory_read 指定该模块的 Mem 读取函数 |
| local_score、region_score、region_key、region_evidence | [区域分配附录 S.1](../local-semantics/weighted-message/region-allocation-learning-note.md#s1-配置名称)；分别选择局部评分、区域评分、记忆 key 与区域证据 |
| selection_mode、区域控制、增益、温度与正则 | 同一[配置表](../local-semantics/weighted-message/region-allocation-learning-note.md#s1-配置名称)，同时列出配对函数、描述类型与容量 |
| conv_stride 及其时间单位 | [短卷积附录 S](../local-semantics/weighted-message/recurrent-memory/causal-convolution-learning-note.md#附录-s配置表示与继续) |

SettleGraph 的 $r_v$ 保持节点秩的含义；带权消息的接收强度统一记为 $r_v^{\mathrm{in}}$。边集沿用 $A$，激活集合沿用 $\mathcal A$。

## 局部函数

类型定义见 [TimedDAG §4--5](../timed-dag-region-selector-learning-note.md#4-节点的局部函数)、[SettleGraph §3--5](../settlegraph-learning-note.md#3-来源原子与时间纤维) 与 [正时延图 §1](../positive-delay-graph-finite-cut-learning-note.md#14-内部消息输出和局部函数)。

| 名称 | 定义域与值域 |
|---|---|
| $\operatorname{Agg}_v$ | $\mathbb N\times\mathcal P_{\mathrm{fin}}(\mathsf{Atom}_v)\to X_v$ |
| $\operatorname{Upd}_v$ | $S_v\times\mathbb N\times X_v\to S_v$ |
| $\operatorname{Read}_v^0$ | $\mathbb N\times X_v\to D_v$ |
| $\operatorname{Read}_v^-,\operatorname{Read}_v^+$ | $S_v\times\mathbb N\times X_v\to D_v$ |
| $\operatorname{SelStep}_{j,C}$ | $Y_j\times\mathbb N\times\prod_{v\in C}D_v\to\mathsf{Allowed}_{j,C}\times\prod_{v\in C}\mathsf C_v\times Y_j$ |
| $\operatorname{Next}_v$ | $S_v\times S_v\times\mathbb N\times X_v\times\{0,1\}\times\mathsf C_v\to S_v$ |
| $\operatorname{Full}_v$ | $S_v\times\mathbb N\times X_v\times\mathsf C_v\to(P_\bot)^{\operatorname{Out}(v)}\times(P_\bot)^{\operatorname{OutPort}(v)}$ |

所有局部函数均为确定的全函数。只有非空纤维产生节点事件，$\mathcal C_{j,\theta}=\{v\in\mathcal R_j:B_{v,\theta}\ne\varnothing\}$。描述模式 $\tau_j\in\{0,-,+\}$ 选择当前内容、旧状态或候选状态；采用模式 $\kappa_j$ 决定哪些候选状态成为计算快照。非候选保持节点状态；非空候选即使没有激活，仍有准备、选择与状态作用。

$\mathsf{Allowed}_{j,C}=\{A'\subseteq C:|A'|\le K_j\}$。空候选返回 $(\varnothing,(),y)$。Next 的调用为 $\operatorname{Next}_v(q,q^{\mathrm{cmp}},\theta,h,a,c)$；Full 的调用为 $\operatorname{Full}_v(q^{\mathrm{cmp}},\theta,h,c)$。Next 与下一选择历史只使用 Full 以前确定的量，不调用或重算 Full。节点状态与区域历史分别归属，固定参数可以共享。

三层均保留上述类型。广播实例另选 $\operatorname{EmitValue}_v:S_v\times\mathbb N\times X_v\times\mathsf C_v\to P$，并取 $\operatorname{Full}_v=\Delta_v\circ\operatorname{EmitValue}_v$。[共同架构 §3—5](../local-semantics/weighted-message/architecture-learning-note.md#3-状态与记忆描述)给出函数绑定与描述空间相容性，区域分配教材给出 SelStep 的数值函数。记忆模块分别提供状态、Upd、Mem 和 Full 所用的内容函数。

## 作用、结果与调用标签

定义见 [TimedDAG §6.4](../timed-dag-region-selector-learning-note.md#64-节点事件区域选择事件与四类作用)。

| 作用坐标 | 规范结果 $\mathcal T_x^{\mathrm{act}}$ | 调用标签 |
|---|---|---|
| $P_{v,\theta}=(\mathrm{prep},v,\theta)$ | $(h,\widetilde q,d)$ | $\operatorname{Lab}_P=(B,q,h,\widetilde q,d)$ |
| $S_{j,\theta}=(\mathrm{select},j,\theta)$ | $(\mathcal A,(c_v)_{v\in\mathcal C},y')$ | $\operatorname{Lab}_S=(\mathcal C,y,(d_v)_{v\in\mathcal C},\mathcal A,(c_v)_{v\in\mathcal C},y')$ |
| $U_{v,\theta}=(\mathrm{adopt},v,\theta)$ | $(q^{\mathrm{cmp}},q^{\mathrm{next}})$ | 同规范结果 |
| $F_{v,\theta}=(\mathrm{full},v,\theta)$ | $(f^A,f^O)$ | 同规范结果 |

$\operatorname{res}_P,\operatorname{res}_S$ 取标签的结果三元组。Next 所需旧状态来自显式状态边界或完整 Lab_P，准备结果三元组一般不包含可恢复的旧状态。

事件图统一记为 $\mathscr G_x^{\mathrm{ev}}=(\mathscr V_x^{\mathrm{ev}},\mathscr A_x^{\mathrm{ev}})$；有限切面在各符号增加 $<b$ 限制。$\operatorname{own}$ 映射作用到节点或选择历史归属，用于静态依赖图；$\operatorname{cown}$ 映射联合契约到 $(\mathrm{node},v)$ 或 $(\mathrm{region},j)$。两者定义域不同，其归属值通过 $v\leftrightarrow(\mathrm{node},v)$、$s_j\leftrightarrow(\mathrm{region},j)$ 对应。

## 封闭、切面与区间

| 名称 | 数学含义 |
|---|---|
| $\operatorname{ValidSeal},\sigma^{\mathrm{in}},\sigma^A$ | 当前集合覆盖所有时间严格小于相应下界的输入／到达消息 |
| $\lambda(v),\lambda(\mathcal R_j)$ | 节点全部输入下界的最小值、区域成员前沿的最小值 |
| $\operatorname{DoneTo}(v,b)$ | 输入前缀关闭，时间小于 $b$ 的状态与完整输出作用完成 |
| $\operatorname{PublishedDoneTo}(v,b)$ | 进一步要求相应源消息全部纳入当前记录 |
| $M_{<b}^{\mathrm{send}}$ | 发送时间小于 $b$ 的消息；其中可含尚未到达的消息 |
| $W_b$ | $\operatorname{send}(m)<b\le\operatorname{time}(m)$ 的跨界消息 |
| $Q_b$ | $(b,(q_v^b)_v,(y_j^b)_j,W_b)$ |
| $\Phi_{a,b}$ | $(Q_a,E_{[a,b)})\mapsto(\mathcal U_{[a,b)},Q_b)$，定义在可达且相容的输入域 $\operatorname{Adm}_{a,b}$ |
| $\odot_b$ | 两段记录粘合；共享状态边界识别一次，新消息按发送区间归属 |

纤维关闭要求 $\lambda(v)>\theta$。Completed 只要求作用完成，PublishedDone 还要求消息齐备；PDG 的强部分截面与 TimedDAG 的细粒度暴露通过这些条件对应。

区间接口在 [TimedDAG §11.5](../timed-dag-region-selector-learning-note.md#115-区间转导与记录粘合)、[分块 §8.1](../timed-dag-chunk-prefill-learning-note.md#81-输入分块组合) 与 [PDG §7](../positive-delay-graph-finite-cut-learning-note.md#7-区间转导与复合) 同义。关闭证据与数值输入分别给出。空区间记录只含共同边界状态，是粘合的单位元。

## 联合求值

定义见 [分块 §2--3](../timed-dag-chunk-prefill-learning-note.md#2-作用块外部边界与精确联合求值) 与 [PDG §10.2](../positive-delay-graph-finite-cut-learning-note.md#102-有限区间的联合求值契约)。

| 名称 | 数学含义 |
|---|---|
| $K,K(\omega)$ | 潜在作用范围、某次实例的实际作用支持 |
| $\mathsf{AdmIn}(K),\mathsf{BlockOut}(K)$ | 全部合法外部自变量、包含实际支持集的完整返回记录 |
| $\operatorname{RefBlock}_K$ | 该边界唯一确定的参考联合函数 |
| $\operatorname{ExpandFull}_{v,\Theta}$ | 将 Full 值族按作用、边与端口展开成规范记录 |
| $\operatorname{BatchFull}_{v,\Theta}$ | 已知 $(q^{\mathrm{cmp}},\theta,h,c)$ 族上的 Full 见证 |
| $\operatorname{BatchState}_{K^U_{v,\Theta}}=\operatorname{BatchState}_{v,\Theta}$ | 节点状态轨迹及范围中实际 $P/U$、允许的局部 $S$ |
| $\operatorname{BatchNode}_{K^{UF}_{v,\Theta}}=\operatorname{BatchNode}_{v,\Theta}$ | 节点状态与完整输出 |
| $\operatorname{BatchRegionState}_{K^R_{j,I}}=\operatorname{BatchRegionState}_{j,I}$ | 区域 $P/S/U$、状态与历史轨迹 |
| $\operatorname{BatchRegion}_{K^{RF}_{j,I}}=\operatorname{BatchRegion}_{j,I}$ | 区域 $P/S/U/F$ 及其消息与输出 |
| $\operatorname{Acts}(B,z)$ | 契约调用在边界 $z$ 上返回的实际作用支持 |
| $\Lambda_v^U(I;\cdot),\Lambda_v^F(I;\cdot)$ | 实际状态作用时间、实际完整输出时间 |
| $R,m_v^U,m_v^F,m_v^{UF},m_j^R$ | 自适应阶段数、各类节点批数与区域块数 |

各见证均满足 $\mathcal K(z)=\operatorname{RefBlock}_K(z)$，并遵守范围的进入依赖与关闭条件。范围包含未发生坐标时，返回的实际支持集明确给出缺席。区域状态与区域状态加输出契约都可属于契约种类 $R$，覆盖范围区分二者。

PDG 增加 Full 解释参数 $\mathbf F$，写作 $\operatorname{RefBlock}_K^{\mathbf F}$ 等；固定解释后省略上标。$\Lambda_v^F(I;\mathbf F,\omega)$ 的上标 $F$ 始终表示作用种类。PDG 的性能比较固定其余局部函数，属于分块教材共同规格族的特例。批数和阶段数同时依赖规格、输入、策略和契约族。

TimedDAG §10.7、§12.4 的 StatePath 只返回一般递归的状态轨迹；BatchState 返回完整规范块，其状态投影才是该轨迹。

## 附录：实现对应的验证关系

实现可沿用上述名称与参数次序。具体表示通过固定解码取得数学值，再通过遗忘投影取得规范记录：

$$
\text{实现表示}\xrightarrow{\operatorname{Decode}}
\text{数学值及辅助字段}\xrightarrow{\Pi}\text{规范记录}.
$$

Decode 忠实恢复所表示的原子、状态、控制和快照；Pi 只删除辅助坐标。两者都不调用参考递归来补写或修正结果。局部函数核对解码后函数值，联合求值核对实际支持集及全部返回坐标，层级嵌入按明确的限制与重索引比较。

切面继续比较同一 $Q_b$ 与后续输入所得记录；各记忆的有效状态投影、外部读出及指定反向规则分别采用自身证明条件。有效投影可以丢弃原始状态细节，不能代替无损解码。KV 与各循环教材附录给出状态证明和表示；区域分配教材给出求解与反向目标。[实现核对](weighted-message-implementation-checks.md)整理位置、短卷积、原模型对应、软硬极限和分段继续的检查范围；浮点容差和有限验证结果由具体实现声明。

下游按[实现核对 §1.3](weighted-message-implementation-checks.md#13-从定义到验证证据)维护固定定义、配置与代码、测试及运行证据的对应记录。对应须覆盖参数、状态快照、返回结构与适用条件，并支持从代码和测试回到定义；本索引提供数学入口，具体代码路径与验证结果由实现仓库维护。
