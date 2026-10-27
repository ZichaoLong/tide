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

# 基础循环记忆：仿射写入与 Delta 写入

本模块用固定大小的矩阵与摘要保存历史。每次接收对状态作一次仿射更新，当前内容再查询该状态。它与[带权 KV](../kv-memory-learning-note.md)共用[局部架构](../architecture-learning-note.md)的载荷、聚合、描述、区域分配、Next 和广播 Full。

下面完整定义两种接收更新，配置名为 affine 和 delta。固定当前输入时，两者都对旧矩阵作仿射变换；affine 衰减旧矩阵后累加当前写入，delta 按当前关联预测的误差修正矩阵。[Gated DeltaNet](gated-delta-learning-note.md)与 [Mamba-2／SSD](ssd-learning-note.md)分别定义包含短卷积、多头、具体门和输出归一化的完整候选；它们使用独立的 memory_family 名称。

本文各节依次填入架构的一次节点作用：

| 本文定义 | 在共同计算中的用途 |
|---|---|
| §1 的状态与初态，§2 的 Upd | 从旧状态和本次聚合形成 $\widetilde q_v$，全部候选保留 |
| §2 的当前特征 $p_v(u)$，§3.1 的 Mem | 从当前内容取得特征，从描述模式指定的状态取得记忆描述，供 Describe 使用 |
| §3.2 的 $G_v$ | 区域返回 $c_v$ 后，激活节点在 $\widetilde q_v$ 上计算完整内容 |

Describe 如何使用这些记忆量，以及 SelStep 如何形成评分和控制，见[区域教材 §2](../region-allocation-learning-note.md#2-指定总量的一次分配)。

## 1. 状态与逻辑时间

固定内容宽度 $d$ 和正整数 $d_k,d_m,d_s$，为节点保存

$$
q=\bigl(s,(H_a,Z_a,W_a)_{a\in\{\mathrm{short},\mathrm{long}\}}\bigr),
\quad H_a\in\mathbb R^{d_m\times d_k},\quad Z_a\in\mathbb R^{d_s},\quad W_a\ge0.
\tag{1}
$$

$s\in\mathbb N$ 是上次正接收的时间。初态为时间零和全部零张量。取衰减率 $\lambda_{\mathrm{short}}>0,\lambda_{\mathrm{long}}=0$，对查询时间 $\theta$ 定义 $\Delta=(\theta-s)_+$ 和有效状态

$$
H_a^-(q,\theta)=e^{-\lambda_a\Delta}H_a,\quad
Z_a^-(q,\theta)=e^{-\lambda_a\Delta}Z_a,\quad
W_a^-(q,\theta)=e^{-\lambda_a\Delta}W_a.
\tag{2}
$$

可达状态总有 $s\le\theta$；正部使函数在全部声明空间上有定义。没有输入时保持保存元组，下一次读取按实际时间差衰减。$e^{-\lambda(a+b)}=e^{-\lambda b}e^{-\lambda a}$ 保证分段读取与一次读取相同。逻辑空档不额外产生节点事件。

## 2. 接收更新

### 2.1 投影与门参数

两种更新共用 key、value、当前内容特征和保留比例；Delta 更新还使用误差写入比例。

当前聚合为 $(r^{\mathrm{in}},u)$，令 $w=\min(1,(r^{\mathrm{in}})_+)$，并定义 $\operatorname{softplus}(x)=\log(1+e^x)$。取矩阵 $K_v\in\mathbb R^{d_k\times d}$、$V_v\in\mathbb R^{d_m\times d}$、$P_v\in\mathbb R^{d_s\times d}$，向量 $a_v^{\mathrm{retain}},a_v^{\mathrm{write}}\in\mathbb R^d$ 和标量 $b_v^{\mathrm{retain}},b_v^{\mathrm{write}}$。用共同归一化定义

$$
\begin{aligned}
\widehat k&=K_v\mathcal N_d(u),\quad
k=\widehat k/\max(1,\|\widehat k\|_2),\quad
\nu=V_v\mathcal N_d(u),\quad p_v(u)=P_v\mathcal N_d(u),\\
g&=\exp[-w\operatorname{softplus}((a_v^{\mathrm{retain}})^\top\mathcal N_d(u)+b_v^{\mathrm{retain}})],\quad
\beta=w\sigma((a_v^{\mathrm{write}})^\top\mathcal N_d(u)+b_v^{\mathrm{write}}),
\end{aligned}
\tag{3}
$$

式 (3) 的 $p_v$ 实现架构的当前内容特征，同一特征用于下面的摘要写入。全部参数有限，$\|k\|\le1$、$0<g\le1$、$0\le\beta\le w$。$g$ 为共同保留比例，$\beta$ 供 Delta 写入使用；两者依赖当前已知接收，不读取旧状态或区域选择。

### 2.2 矩阵更新的两个候选

记 $I$ 为 $d_k$ 维单位矩阵。当 $r^{\mathrm{in}}>0$ 时，选择下述一种更新，对两个时间尺度分别作用：

$$
\begin{array}{ll}
\text{affine：}&\widetilde H_a=gH_a^-+w\nu k^\top,\\[2pt]
\text{delta：}&\widetilde H_a=gH_a^-(I-\beta kk^\top)+\beta\nu k^\top.
\end{array}
\tag{4}
$$

Delta 形式也写作 $\widetilde H_a=gH_a^-+\beta(\nu-gH_a^-k)k^\top$：先预测当前 key 对应的 value，再按误差修正。其线性系数由当前输入决定。

### 2.3 完成 Upd 的返回

选定矩阵更新后，两种模块均按下式更新摘要与权重，并返回完整状态：

$$
\begin{gathered}
\widetilde Z_a=Z_a^-+wp_v(u),\qquad \widetilde W_a=W_a^-+w,\\
\operatorname{Upd}_v(q,\theta,(r^{\mathrm{in}},u))=
 \bigl(\theta,(\widetilde H_a,\widetilde Z_a,\widetilde W_a)_a\bigr).
\end{gathered}
\tag{5}
$$

$r^{\mathrm{in}}\le0$ 时直接定义 Upd 返回原 $q$。正接收后的状态由所有候选保留，未激活也写入。弱接收时 $g\to1,\beta\to0,w\to0$，所以遗忘和写入共同趋于中性；只缩小新 value 而保持固定遗忘比例通常不具备这一性质。

## 3. 描述与完整内容

### 3.1 两种记忆描述

本节用时间摘要或矩阵查询实现架构的 Mem 接口，为 Describe 提供本节点的记忆信息；评分与发送分配怎样使用这些信息，由区域教材定义。

给定一次读取的 $(q,\theta,u)$，两种方式共用写入量统计 $W_{v,a}=W_a^-(q,\theta)$。矩阵查询另取 $Q_v^r\in\mathbb R^{d_k\times d}$、$L_{v,a}\in\mathbb R^{d_s\times d_m}$，定义 $Q_v^{\mathrm{read}}(u)=Q_v^r\mathcal N_d(u)$。两种描述向量为

$$
\begin{aligned}
M_a^{\mathrm{sum}}(q,\theta,u)&=\frac{Z_a^-(q,\theta)}{\max(1,W_a^-(q,\theta))},\\
M_a^{\mathrm{state}}(q,\theta,u)&=
 L_{v,a}H_a^-(q,\theta)Q_v^{\mathrm{read}}(u).
\end{aligned}
\tag{6}
$$

在组成模型时固定 sum 或 state；两个时间尺度采用同一种方式。对 $a\in\{\mathrm{short},\mathrm{long}\}$，所选描述及完整返回为

$$
\begin{aligned}
M_a&=\begin{cases}
M_a^{\mathrm{sum}}(q,\theta,u),&\text{选择时间摘要（sum）},\\
M_a^{\mathrm{state}}(q,\theta,u),&\text{选择矩阵查询（state）},
\end{cases}\\
\operatorname{Mem}_v(q,\theta,u)
 &=(M_{\mathrm{short}},M_{\mathrm{long}},W_{\mathrm{short}},W_{\mathrm{long}}).
\end{aligned}
$$

sum 按时间汇集独立摘要，state 让当前内容分别查询两份矩阵。读取方式与 §2 的 affine／delta 更新可独立组合；选定后，每次调用在给定状态与时间上使用同一组函数。

在描述模式指定的状态上求这个四元组。[区域教材定义的 Describe](../region-allocation-learning-note.md#51-从所选函数得到完整返回)使用它、式 (3) 的 $p_v(u)$ 及当前接收 $(r^{\mathrm{in}},u)$ 计算 $d_v$。区域 SelStep 随后将完整描述族变为控制 $c_v$ 与激活集合。

这里 $W_a$ 是衰减后的接收写入量之和；Delta 会覆盖或抵消矩阵中的部分内容，故 $W_a$ 不表示可恢复的记忆条数，也不度量矩阵存储的精确信息量。

### 3.2 内容函数

固定前馈宽度 $d_f$，取 $Q_v^F\in\mathbb R^{d_k\times d}$、$O_{v,a}\in\mathbb R^{d\times d_m}$、$W_v^{\mathrm{gate}},W_v^{\mathrm{up}}\in\mathbb R^{d_f\times d}$、$W_v^{\mathrm{down}}\in\mathbb R^{d\times d_f}$ 及非负残差尺度 $\alpha_{\mathrm{mem}},\alpha_{\mathrm{ff}}$，令

$$
\begin{aligned}
R_a&=H_a^-(q,\theta)Q_v^F\mathcal N_d(u),\\
x&=u+\alpha_{\mathrm{mem}}\sum_aO_{v,a}R_a,\\
G_v(q,\theta,u)&=x+\alpha_{\mathrm{ff}}W_v^{\mathrm{down}}
 [\operatorname{SiLU}(W_v^{\mathrm{gate}}\mathcal N_d(x))
 \odot W_v^{\mathrm{up}}\mathcal N_d(x)].
\end{aligned}
\tag{7}
$$

在空初态上读取时，$R_a=0$、$x=u$，$G_v$ 仍保留当前内容与前馈残差。Full 在新状态 $\widetilde q$ 上使用此 $G_v$，经共同 EmitValue 和广播返回。描述模式可为 $+$、$-$ 或 $0$；分别使用新、旧或初始状态取得描述。单点区域历史和默认 Next 与 KV 相同。

### 3.3 一个完整单节点例子

取所有宽度为 $1$、$\epsilon=1$，投影 $K=V=P=Q^r=Q^F=\sqrt2$，聚合取均值。输入为 $(1,1)$，则 $k=\nu=p=Q=1$。令增益为 $1$、发送为任一单有效候选时取 $c=\mu$ 的函数，采用 Read$^+$、$\alpha_{\mathrm{mem}}=1,\alpha_{\mathrm{ff}}=0$，$O_{\mathrm{long}}=1,O_{\mathrm{short}}=0$。

affine 从零写入 $H_{\mathrm{long}}=1$，Full 得到 $G=2$，发送 $(1,2)$。delta 若取 $\beta=1/2$，则写入 $H_{\mathrm{long}}=1/2$，Full 得到 $G=3/2$。两者的摘要写入均为 $Z=W=1$。之后没有输入时不产生发送；在未来时刻读旧状态，长期矩阵保持，短期矩阵按逻辑时间衰减。

## 4. 精确联合求值与继续

### 4.1 仿射复合

固定矩阵宽度 $d_m,d_k$、步数 $N\in\mathbb N$、初态 $H_0\in\mathbb R^{d_m\times d_k}$，以及系数序列 $A_i\in\mathbb R^{d_k\times d_k},B_i\in\mathbb R^{d_m\times d_k}$，$1\le i\le N$。仿射递归为

$$
H_i=H_{i-1}A_i+B_i.
\tag{8}
$$

一个系数对 $(A,B)$ 表示变换 $H\mapsto HA+B$。先使用第一对、再使用第二对，定义复合

$$
(A_1,B_1)\circledast(A_2,B_2)
 =(A_1A_2,B_1A_2+B_2).
\tag{9}
$$

**命题。** 式 (9) 表示上述先后作用，且满足结合律。记 $I$ 为 $d_k$ 维单位矩阵，单位元为 $(I,0)$，空前缀取此值。对 $0\le i\le N$，依次复合前 $i$ 对得到 $(\bar A_i,\bar B_i)$，则 $H_i=H_0\bar A_i+\bar B_i$。

**证明。** 两次作用得到 $(HA_1+B_1)A_2+B_2=H(A_1A_2)+B_1A_2+B_2$。三个系数对无论按哪种括号复合，均得到

$$
(A_1A_2A_3,\ B_1A_2A_3+B_2A_3+B_3).
$$

这证明结合律；代入 $(I,0)$ 验证单位元，再按 $i$ 归纳即得前缀公式。$\square$

若各系数可在矩阵递归之前由完整驱动输入和左边界计算，就可通过任意保持先后次序的结合方式求各个前缀。本文两种基础更新、Gated DeltaNet 与 SSD 的矩阵递归均使用这一性质。系数若还依赖未知前态或区域选择，相应的系数生成与联合算法需另行证明；图中自适应消息的完整驱动仍须由外层依赖确定。

### 4.2 两种基础更新的系数

固定某节点的一列正接收时刻 $\theta_1<\cdots<\theta_N$ 及完整聚合输入，记 $\Delta_i$ 为距上次正接收的时间差。对每个时间尺度 $a$，式 (8) 的系数为：

| 更新 | $A_{a,i}$ | $B_i$ |
|---|---|---|
| affine | $e^{-\lambda_a\Delta_i}g_iI$ | $w_i\nu_i k_i^\top$ |
| delta | $e^{-\lambda_a\Delta_i}g_i(I-\beta_ik_ik_i^\top)$ | $\beta_i\nu_i k_i^\top$ |

这些系数由当前已知输入及原始时间决定。摘要 $Z,W$ 同样是已知系数的仿射变换，因此可用逐次递归或前缀复合得到每个中间状态。零强度事件和空纤维保留原始元组，正事件更新时间戳；读取时应用式 (2)。

### 4.3 完整记录与继续

给定区域窗口的完整驱动纤维，先计算这些状态与每次读取，再计算区域评分和分配，最后在激活快照上求 Full。得到与逐次运行完全相同的 P/S/U/F 结果和右边界。返回完整记录后才构成正典的 BatchRegionState 与 BatchFull 契约。

继续保存 $q$ 的时间戳和完整 $Q_b$。按逻辑时间衰减的半群性质保证分块边界不重置衰减。把衰减改为“每次候选事件衰减一次”会得到另一种函数，零消息也可能额外推进遗忘。

## 5. 有效状态与局部性质

<a id="5-连续性与组合"></a>

在比较时刻 $\theta$，定义有效状态投影

$$
\pi_{\mathrm{rec},v,\theta}(q)=(H_a^-(q,\theta),Z_a^-(q,\theta),W_a^-(q,\theta))_a.
\tag{10}
$$

原始时间戳可以因微弱写入改变，投影比较相同时刻的有效状态。

**引理。** 从零初态开始，在固定有限时间和有界接收内容、强度下，affine 与 delta 满足[共同架构 §6.1](../architecture-learning-note.md#61-比较对象与局部条件)的有界性、零接收中性、更新连续性和读取连续性。描述与内容函数使用本文给定的绑定。证明见附录 A。

该引理验证本模块的记忆与内容条件。与其他记忆混合时，各节点分别使用相应引理；发送侧条件按[区域教材 §8 的性质对照](../region-allocation-learning-note.md#8-发送规则如何满足架构条件)核对，再应用架构的组合命题。各候选的极限比较及其条件由表中 §7.2、§7.4—7.5 分别给出。

矩阵状态的时间信息由式 (2) 提供；增加位置机制须给出相容的状态与查询定义。

## 附录 A：状态性质的证明

**有界性。** 归一化使 $k,\nu,p,Q$ 有界。$\|I-\beta kk^\top\|_2\le1$，因为 $\|k\|\le1$、$0\le\beta\le1$；$g$ 和逻辑衰减均至多为 $1$。两种更新均有 $\|\widetilde H_a\|_F\le\|H_a^-\|_F+Cw$。摘要和权重也每次增加有界量。有限时域只含有限次接收，故状态、描述与内容有界。此界与区域选择温度无关。

**零与弱接收。** 零接收不改保存状态。$r^{\mathrm{in}}\to0$ 时，$w\to0,g\to1,\beta\to0$，即使 $u$ 只保持有界，新的有效矩阵、摘要和权重都趋于旧状态在当前时刻的有效值。把时间戳改为当前时刻后再向未来读取，与保留旧时间戳的极限一致，依据指数衰减的半群律。因此插入微弱事件的影响在任意固定未来时刻趋零，插入零事件无数值影响。

**读取。** 有效状态与当前内容收敛时，式 (3)—(7) 的连续函数给出 Mem、当前特征与完整内容的收敛。正接收处更新连续，零接收处由前段得到中性极限；旧状态、新状态和零初态按相同逻辑时间读取，三种 Read 模式均满足共同条件。局部引理成立。$\square$

## 附录 S：配置与实现对应

memory_family=affine／delta 分别绑定式 (4)；memory_read=sum／state 绑定式 (6)。配置另给 $d_k,d_m,d_s$、衰减率、投影、门控与 Full 参数，描述模式与区域分配沿用共同名称。默认 memory_read=sum、read_mode=+。

每节点保存两个 $d_m\times d_k$ 矩阵及摘要、权重和时间戳，大小与接收次数无关。单次矩阵更新和读取的主要工作为 $O(d_md_k)$，投影与前馈另计。显式复合稠密 $A$ 可能比顺序更新昂贵；Delta 的低秩结构可用于专门联合算法。结合律提供精确重排依据，实际工作量、存储和并行收益由具体算法给出。

有限时间内的数值连续性不等于长程梯度稳定。更新到未来的导数包含 $A_i$ 的乘积；内容查询提供直接读取当前矩阵的路径，历史信息仍经过状态压缩和更新。实现核对包括有效时间衰减、弱事件、零事件、完整中间状态、顺序与复合结果，以及跨切面的记录恢复。
