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

# 带权 KV：写入、描述与内容读取

本模块把节点状态定义为带原始时间和写入权重的 KV 序列。接收发生时保存信息，Read 提取候选描述，激活后的 Full 再读取完整内容。聚合、状态采用和广播使用[局部架构教材](architecture-learning-note.md)的共同函数，评分与发送使用[区域分配教材](region-allocation-learning-note.md)。本文具体给出架构的 $S_v,q_v^{\mathrm{init}}$、Upd、Mem、当前特征 $p_v$ 与内容函数 $G_v$。

阅读顺序对应一次节点作用：§1 定义收到聚合后怎样写入；§2 定义查询位置；§3 计算供 Describe 使用的记忆量；区域选择给出 $c_v$ 后，§4 为激活节点计算 $G_v$。第 5 节完成架构中 $a,b,c$ 的同一算例。

固定内容宽度 $d$。消息 $(m,z)$、聚合 $h=(r^{\mathrm{in}},u)$、归一化 $\mathcal N$ 及数值投影 $\pi_P(m,z)=(m_+,m_+z)$ 均沿用共同定义。各参数在一次运行中固定。

## 1. 状态与接收写入

### 1.1 保存的条目

固定记忆描述宽度 $d_s$、KV 适配秩 $r_a$、Attention 头数 $H_{\mathrm{att}}$ 和每头宽度 $d_h$，满足 $H_{\mathrm{att}}d_h=d$。

节点状态空间 $S_v$ 是全部有限序列

$$
q=((\xi_i,k_i,\nu_i,p_i,w_i))_{i=1}^N,
\quad \xi_i\in\mathbb N,\quad
k_i,\nu_i\in\mathbb R^d,\quad
p_i\in\mathbb R^{d_s},\quad w_i\ge0.
\tag{1}
$$

$k_i$ 是匹配记忆所用的 key，$\nu_i$ 是读取时被加权的 value。它们分别分成 $H_{\mathrm{att}}$ 个 $d_h$ 维子向量，记为 $k_{i,h},\nu_{i,h}$；$p_i$ 是供 Mem 汇集的摘要特征，$w_i$ 是独立的写入权重，$\xi_i$ 是原始逻辑时间。初态为空；实际运行中每节点每逻辑时间至多写一项。

### 1.2 写入投影与当前特征

令 $j=\rho(v)$ 为节点所属区域，记 $I_p$ 为 $p$ 维单位矩阵。区域共享基座 $W_j^{\mathrm{KV}}\in\mathbb R^{2d\times d}$，节点各有 $U_v\in\mathbb R^{2d\times r_a}$、$V_v^{\mathrm{ad}}\in\mathbb R^{r_a\times d}$ 与 $W_v^S\in\mathbb R^{d_s\times d}$。定义

$$
\begin{aligned}
[k_v(u);\nu_v(u)]
  &=(W_j^{\mathrm{KV}}+U_vV_v^{\mathrm{ad}})\mathcal N_d(u),\\
p_v(u)&=W_v^S\mathcal N_d(u),
\qquad w(r^{\mathrm{in}})=\min(1,(r^{\mathrm{in}})_+).
\end{aligned}
\tag{2}
$$

式 (2) 的 $p_v$ 实现架构的当前内容特征。写入时将 $p_v(u)$ 保存为条目中的 $p_i$，使当前内容与历史摘要使用同一投影；第 3 节再从保存的条目实现 Mem。

共享基座给出区域共同的 KV 投影，节点差异由秩至多为 $r_a$ 的乘积提供。相同输入的基座投影可共同求值，各节点仍保存私有条目。取 $r_a=d$、$W_j^{\mathrm{KV}}=0$、$V_v^{\mathrm{ad}}=I_d$ 与 $U_v=[W_v^K;W_v^V]$，即可表示任意独立的 K、V 投影。式 (2) 定义写入；Attention 读取由 §4.1 给出。

### 1.3 窗口与接收更新

为每节点固定窗口 $T_v\in\mathbb N_{\ge1}\cup\{\infty\}$，以原始逻辑时间计量。定义遮罩

$$
\chi_{v,\theta}(\xi)=\mathbf1[0\le\theta-\xi<T_v].
$$

$q|_{v,\theta}$ 表示按原序保留 $\chi_{v,\theta}(\xi_i)=1$ 的条目，$\oplus$ 表示向序列追加一项。接收更新是

$$
\widetilde q=\operatorname{Upd}_v(q,\theta,(r^{\mathrm{in}},u))
=\begin{cases}
q|_{v,\theta}\mathbin{\oplus}
  (\theta,k_v(u),\nu_v(u),p_v(u),w(r^{\mathrm{in}})),&r^{\mathrm{in}}>0,\\
q,&r^{\mathrm{in}}\le0.
\end{cases}
\tag{3}
$$

每次正接收只写一条，KV 内容不预乘 $w$；接收强度为零时保留原序列。是否执行本节点 Full 不参与写入决定。

所有读取同样使用 $\chi_{v,\theta}$。没有候选或只有零强度输入时，保存序列可以含已过期条目，但它们不会再被读出；下一次正接收再清理。有限 $T_v$ 下，由每逻辑时间至多写一项，实际保存序列始终至多含 $T_v$ 项。$T_v=\infty$ 保留全部接收历史，在任意有限时间范围内仍只有有限项。历史覆盖范围取决于该节点实际收到的内容。

窗口按原始写入时间定义。一条强度趋零的输入会产生权重趋零的新条目；窗口内已有条目的保留仍由各自的写入时间决定。

## 2. 位置坐标

内容检索和完整内容读取都可以用位置参与匹配。先定义共同的位置坐标与旋转矩阵族；两种读取随后分别选择自己的宽度、频率和投影。

为每个节点固定位置函数

$$
\operatorname{Pos}_v:\mathbb N\to\mathbb R.
$$

逻辑时间 $\theta$ 决定消息到达、状态顺序和窗口成员；$\operatorname{Pos}_v(\theta)$ 为 Attention 提供位置坐标。本文使用的仿射族为

$$
\operatorname{Pos}_v(\theta)
=\frac{\theta-o_v^{\mathrm{pos}}}{s_v^{\mathrm{pos}}},
\qquad o_v^{\mathrm{pos}}\in\mathbb R,\quad s_v^{\mathrm{pos}}>0.
$$

$o_v^{\mathrm{pos}}=0,s_v^{\mathrm{pos}}=1$ 直接编码逻辑时间。按输入位置换算的固定时间网格见附录 B。一般图中，同一次接收可以汇合不同输入位置的消息；这里编码当前聚合的处理时间和历史条目的写入时间，无需给混合内容指定唯一来源位置。

位置函数在整条继续运行中固定，不随接收次数、激活次数或分段边界改变。更一般的固定实值位置函数也可用于下文定义与有限切面证明；具体组合须给出其完整定义。

固定偶数 $0\le d^{\mathrm{rot}}_{v,h}\le d_h$、坐标配对与有限频率 $\omega^{\mathrm{rot}}_{v,h,a}$。矩阵之间的 $\oplus$ 表示直和，定义

$$
\begin{aligned}
\operatorname{Rot}_{v,h}(p)
 &=I_{d_h-d^{\mathrm{rot}}_{v,h}}
   \oplus\bigoplus_{a=1}^{d^{\mathrm{rot}}_{v,h}/2}
   \begin{pmatrix}
   \cos(\omega^{\mathrm{rot}}_{v,h,a}p)&-\sin(\omega^{\mathrm{rot}}_{v,h,a}p)\\
   \sin(\omega^{\mathrm{rot}}_{v,h,a}p)&\cos(\omega^{\mathrm{rot}}_{v,h,a}p)
   \end{pmatrix}.
\end{aligned}
$$

旋转维数为零时 $\operatorname{Rot}=I_{d_h}$；旋转全部或部分坐标分别得到完整或部分 RoPE。旋转维数独立于评分网络的宽度。

数学状态中的 $k_i$ 保持未旋转形式，$\nu_i,p_i$ 也不施加旋转。上述旋转满足

$$
\operatorname{Rot}_{v,h}(p)^\top\operatorname{Rot}_{v,h}(q)
=\operatorname{Rot}_{v,h}(q-p),
$$

故旋转后的匹配依赖相对坐标，并包含各频率对应的相位信息。第 3、4 节分别为记忆描述和完整内容选定查询与匹配参数。

## 3. 记忆描述

本节实现架构的 Mem 接口。时间摘要汇集已保存的摘要特征，内容检索用当前内容查询已保存的 K/V；两种方式为 Describe 提供同型的记忆描述，供其构造评分、区域证据或增益所需的字段。

### 3.1 两种读取共用的时间权重

两种描述都要确定哪些条目参与，以及这些条目积累了多少写入权重。短期、长期通道分别采用以下时间权重和统计量。

对每节点取 $\beta_{v,\mathrm{long}}=0$、$\beta_{v,\mathrm{short}}>0$，均按逻辑时间计量。固定节点后省略部分下标，对 $a\in\{\mathrm{short},\mathrm{long}\}$ 定义

$$
\begin{aligned}
\omega_i^a(\theta)&=\chi_{v,\theta}(\xi_i)w_i
 e^{-\beta_{v,a}(\theta-\xi_i)_+},\\
W_a(q,\theta)&=\sum_i\omega_i^a(\theta).
\end{aligned}
$$

### 3.2 时间摘要

时间摘要（sum）按上述权重汇集保存的摘要特征：

$$
M_a^{\mathrm{sum}}(q,\theta)=
 \frac{\sum_i\omega_i^a(\theta)p_i}{\max(1,W_a(q,\theta))}.
\tag{4}
$$

短期摘要突出近期接收，长期摘要覆盖整个窗口；分母使最后一条弱记忆消失时摘要也趋零。摘要本身只按时间和写入权重汇集，当前内容在后续评分中与它交互。

### 3.3 当前内容检索

取偶数 $0\le d_v^{\mathrm{rot,read}}\le d_s$、坐标配对及有限频率 $\omega_{v,a}^{\mathrm{rot,read}}$。$\operatorname{Rot}_v^{\mathrm{read}}$ 使用 §2 的旋转矩阵族，宽度换为 $d_s$，采用这些独立参数；零旋转维数给出内容匹配。它沿用同一 $\operatorname{Pos}_v$，与 Full 的位置配置分别选择。

取节点矩阵 $W_v^{\mathrm{RQ}},W_v^{\mathrm{RK}},W_v^{\mathrm{RV}}\in\mathbb R^{d_s\times d}$，定义

$$
\begin{aligned}
Q_v^{\mathrm{read}}(u)&=W_v^{\mathrm{RQ}}\mathcal N_d(u),\qquad
K_{v,i}^{\mathrm{read}}=W_v^{\mathrm{RK}}k_i,\qquad
V_{v,i}^{\mathrm{read}}=W_v^{\mathrm{RV}}\nu_i,\\
\ell_{v,i}^{\mathrm{read}}&=
\frac{
 [\operatorname{Rot}_v^{\mathrm{read}}(\operatorname{Pos}_v(\theta))Q_v^{\mathrm{read}}(u)]^\top
 [\operatorname{Rot}_v^{\mathrm{read}}(\operatorname{Pos}_v(\xi_i))K_{v,i}^{\mathrm{read}}]
}{\sqrt{d_s}},\\
M_a^{\mathrm{qry}}(q,\theta,u)&=
\begin{cases}
\displaystyle \min(1,W_a)
 \frac{\sum_i\omega_i^a e^{\ell_{v,i}^{\mathrm{read}}}V_{v,i}^{\mathrm{read}}}
      {\sum_i\omega_i^a e^{\ell_{v,i}^{\mathrm{read}}}},&W_a>0,\\
0,&W_a=0.
\end{cases}
\end{aligned}
\tag{5}
$$

两个时间尺度共用本次匹配分数。内容检索（qry）使用共同时间权重，并按当前内容与记忆的匹配重新分配读取份额。宽度与检索量允许接近 Full，成本按实际配置计量。

### 3.4 Mem 的返回值与状态快照

在组成模型时，为节点选定 sum 或 qry。给定一次读取的 $(q,\theta,u)$，两个通道采用同一种读取方式：对 $a\in\{\mathrm{short},\mathrm{long}\}$，取

$$
M_a=
\begin{cases}
M_a^{\mathrm{sum}}(q,\theta),&\text{选择时间摘要（sum）},\\
M_a^{\mathrm{qry}}(q,\theta,u),&\text{选择内容检索（qry）}.
\end{cases}
$$

共同统计量仍由 §3.1 给出，完整返回为

$$
\operatorname{Mem}_v(q,\theta,u)
 =(M_{\mathrm{short}},M_{\mathrm{long}},W_{\mathrm{short}}(q,\theta),W_{\mathrm{long}}(q,\theta)).
$$

sum 使用式 (2) 的摘要投影，当前内容在后续评分中与摘要交互；qry 从已写入的 K/V 再投影，并由当前内容参与检索。两者改变描述向量的来源，保持相同的四元组形状与写入量统计。

Read$^+$ 在新状态 $\widetilde q$ 上求 Mem，Read$^-$ 在旧状态上按当前时间读取，Read$^0$ 使用空序列。Full 始终在 $\widetilde q$ 上读取。[区域教材的 Describe](region-allocation-learning-note.md#51-从所选函数得到完整返回)使用这些 Mem 返回值、式 (2) 的当前特征 $p_v(u_v)$ 及接收 $(r_v^{\mathrm{in}},u_v)$，计算节点描述 $d_v$；SelStep 收齐区域描述后得到 $c_v$ 与激活集合。sum／qry 改变记忆信息的取得方式，区域评分和发送函数仍可分别选择。

## 4. 激活后的内容读取

区域选择完成后，只有 $c_v>0$ 的节点执行本节内容函数。它读取本次计算状态 $\widetilde q_v$；§3 中给区域提供描述的读取已在选择前完成。

### 4.1 带权 Attention

对每个头固定 $W^Q_{v,h}\in\mathbb R^{d_h\times d}$，定义当前 query 为 $Q_{v,h}(u)=W^Q_{v,h}\mathcal N_d(u)$。另取非负距离偏置斜率 $b^{\mathrm{pos}}_{v,h}$，为完整内容读取定义匹配分数

$$
\begin{aligned}
\ell_{i,h}&=\frac{
 [\operatorname{Rot}_{v,h}(\operatorname{Pos}_v(\theta))Q_{v,h}(u)]^\top
 [\operatorname{Rot}_{v,h}(\operatorname{Pos}_v(\xi_i))k_{i,h}]}{\sqrt{d_h}}\\
 &\quad{}-b^{\mathrm{pos}}_{v,h}
   (\operatorname{Pos}_v(\theta)-\operatorname{Pos}_v(\xi_i))_+.
\end{aligned}
\tag{6}
$$

下面三种配置共用这一匹配公式：

| 配置 | 旋转维数 | 距离偏置 |
|---|---|---|
| NoPE | $d^{\mathrm{rot}}_{v,h}=0$ | $b^{\mathrm{pos}}_{v,h}=0$ |
| 完整 RoPE | $d^{\mathrm{rot}}_{v,h}=d_h$，要求 $d_h$ 为偶数 | 通常为零 |
| 部分 RoPE | $0<d^{\mathrm{rot}}_{v,h}<d_h$，旋转维数为偶数 | 通常为零 |

NoPE 的 Full 匹配使用内容，时间窗口、摘要和状态顺序仍保留时间信息。令 $\Delta p=\operatorname{Pos}_v(\theta)-\operatorname{Pos}_v(\xi_i)$，可选的线性距离偏置在 softmax 中乘入 $e^{-b^{\mathrm{pos}}_{v,h}\Delta p_+}$，是一种距离衰减；它与 RoPE 的旋转可以分别配置。

用这些匹配分数，从本次计算状态 $q$ 中读取窗口内的记忆：

$$
\begin{aligned}
\widehat w_i&=\chi_{v,\theta}(\xi_i)w_i,\qquad
W=\sum_i\widehat w_i,\qquad a(W)=\min(1,W),\\
R_{v,h}(q,\theta,u)
&=\begin{cases}
\displaystyle a(W)
 \frac{\sum_i\widehat w_i e^{\ell_{i,h}}\nu_{i,h}}
      {\sum_i\widehat w_i e^{\ell_{i,h}}},&W>0,\\
0,&W=0.
\end{cases}
\end{aligned}
\tag{7}
$$

$\ell_{i,h}$ 是当前内容与第 $i$ 条记忆的匹配分数。$\widehat w_i$ 是本次读取中该条目的有效权重，$W$ 是这些权重之和。$R_{v,h}$ 先对记忆的 value 作归一化加权平均，再乘 $a(W)$。

窗口内有效权重全为 $1$ 时，得到通常的多头 Attention 读取。最后一条弱记忆消失时，$a(W)$ 使读取趋零；单位记忆有 $a(1)=1$。实际匹配采用本节的位置与内容公式；零旋转和零距离偏置恢复纯内容匹配。

### 4.2 内容函数

固定前馈宽度 $d_f$，取 $W_v^O\in\mathbb R^{d\times d}$、$W_v^{\mathrm{gate}},W_v^{\mathrm{up}}\in\mathbb R^{d_f\times d}$、$W_v^{\mathrm{down}}\in\mathbb R^{d\times d_f}$ 以及非负残差尺度 $\alpha_{\mathrm{att}},\alpha_{\mathrm{ff}}$，定义

$$
\begin{aligned}
x_v&=u+\alpha_{\mathrm{att}}W_v^O
       [R_{v,1};\ldots;R_{v,H_{\mathrm{att}}}],\\
G_v(q,\theta,u)
 &=x_v+\alpha_{\mathrm{ff}}W_v^{\mathrm{down}}
 \left[
 \operatorname{SiLU}(W_v^{\mathrm{gate}}\mathcal N_d(x_v))
 \odot W_v^{\mathrm{up}}\mathcal N_d(x_v)
 \right].
\end{aligned}
\tag{8}
$$

式 (7) 负责读取历史，式 (8) 将读出与当前内容合成，再作预归一化的 SwiGLU 前馈残差。$G_v$ 接入[共同 Full](architecture-learning-note.md#4-分配完整计算与发送)，发送 $(c_v,G_v)$；下一状态仍取接收后的 $\widetilde q$。取 $\alpha_{\mathrm{att}}=0$ 得到只在 Full 使用前馈残差的变式，KV 仍可供 Read 选择使用；若再删除记忆写入，需改用相应无记忆模块定义。

## 5. 完整例子

### 5.1 从接收到输出

本例为[架构 §4.5](architecture-learning-note.md#45-一组贯穿例子)的两种发送绑定提供共同的记忆与内容函数。先计算接收、记忆及评分，再分别使用指定总量、节点增益与固定阈值分配，最后代入同一 Full。

取共享聚合、Read$^+$、时间摘要、cat 局部评分和 ind 区域评分。区域 $\{a,b,c\}$ 无内部边，初始缓存为空，$d=d_s=H_{\mathrm{att}}=1$、$\epsilon=1$、$\alpha_{\mathrm{agg}}=0$，Attention 只按内容匹配。本次 $a$ 收到 $(0.5,1),(0.5,1)$，$b$ 收到 $(1,-1)$，$c$ 收到 $(0,9)$。三个节点都是候选。令 key、query 为零，value 与摘要投影均为 $\sqrt2$，故单位内容的归一化投影仍是 $\pm1$。

本次新状态上，两种时间摘要均只读到当前写入；因此

$$
\begin{aligned}
\operatorname{Mem}_a(\widetilde q_a,\theta,u_a)&=(1,1,1,1),\\
\operatorname{Mem}_b(\widetilde q_b,\theta,u_b)&=(-1,-1,1,1),\\
\operatorname{Mem}_c(\widetilde q_c,\theta,u_c)&=(0,0,0,0).
\end{aligned}
$$

这些四元组依次为短期描述、长期描述、短期写入量和长期写入量，正是 Describe 的记忆输入。

评分网络输出常数 $s=(2,1,100)$，可由零输出权重与相应偏置实现。共同数据为

| 节点 | 聚合 $(r^{\mathrm{in}},u)$ | 新写入 $(k,\nu,p,w)$ | 截断接收 $w$ | 评分 $s$ |
|---|---|---|---|---|
| $a$ | $(1,1)$ | $(0,1,1,1)$ | $1$ | $2$ |
| $b$ | $(1,-1)$ | $(0,-1,-1,1)$ | $1$ | $1$ |
| $c$ | $(0,0)$ | 不追加 | $0$ | $100$ |

**发送绑定。** 两种绑定均使用 quad、正典容量 $K_j=3$，先取 $\tau_{\mathrm{sel}}=2$：

- 指定总量按[区域教材 §2.5](region-allocation-learning-note.md#25-三个候选的一次计算)取 $\gamma_{\max}=2,d_{\mathrm{reg}}=1,k_j=2$，使用该例的上下文控制：证据为 $(-1/2,-1/2,0)$，区域头给出 $\rho_j=0.4$。此时 $\mu^{\max}=(1,1,0)$、$B=1.4$；Describe 返回四元组，SelStep 得到 $(0.95,0.45,0)$。
- 节点增益与固定阈值按[区域教材 §5.2](region-allocation-learning-note.md#52-更换阈值与节点增益)取固定 $\gamma_v=1,\eta_j=0,k_j=1$。此时 $\mu=(1,1,0)$；Describe 返回三元组，SelStep 得到 $(0.75,0.25,0)$。

**共同的内容与返回。** 三个候选均保留接收后的状态。令 $W_v^O=1,\alpha_{\mathrm{att}}=1,\alpha_{\mathrm{ff}}=0$；$a,b$ 激活时，Attention 分别读出 $1,-1$，得到内容 $G_a=2,G_b=-2$。节点 $c$ 在两种绑定中上限均为零，始终不调用 Full。

记同刻端口的 ReadOut 为 $O$，则 $O=(2c_a-2c_b)/\max(1,c_a+c_b)$。固定各自参数、只改变温度，得到：

| 发送绑定 | $\tau_{\mathrm{sel}}$ | 控制 $(c_a,c_b,c_c)$ | $a,b$ 的端口载荷 | $O$ |
|---|---|---|---|---|
| 指定总量，$B=1.4$ | $2$ | $(0.95,0.45,0)$ | $(0.95,2),(0.45,-2)$ | $5/7$ |
| 同上 | $0.5$ | $(1,0.4,0)$ | $(1,2),(0.4,-2)$ | $6/7$ |
| 节点增益与固定阈值，$k=1$ | $2$ | $(0.75,0.25,0)$ | $(0.75,2),(0.25,-2)$ | $1$ |
| 同上 | $0.5$ | $(1,0,0)$ | 仅 $(1,2)$ | $2$ |

若发送给同一采用 $\alpha_{\mathrm{agg}}=0$ 的下游，其聚合为 $(r^{\mathrm{in}},u)=(c_a+c_b,O)$，随后开始自己的接收流程。指定总量的零温分配在本例给出第二行；节点增益与固定阈值的硬选择给出第四行。两者的比较条件分别见区域教材 §7.2、§7.5。

### 5.2 未激活时保留的记忆

再取同一区域的两个节点 $a,b$，各有输入与输出端口，没有内部边。各输入端口在时间 $1,4$ 分别收到 $(1,1),(1,-1)$，取 $T_a=T_b\ge4$。沿用上一例的投影与 Full，每次读出同刻两个端口的输出；采用节点增益与固定阈值的硬选择，取固定 $\gamma_v=1$、$k_j=1,\eta_j=0$，评分改为

$$
s_a=0.5,\qquad
s_b=0.75-M_{\mathrm{long}}^{\mathrm{sum}}(\widetilde q_b,\theta).
$$

这属于[拼接评分](region-allocation-learning-note.md#31-拼接评分)的网络：$\operatorname{SiLU}(x)-\operatorname{SiLU}(-x)=x$，两个隐藏单元即可表达该线性项。

| 位置 | 两节点新写入的 $(\xi,\nu,p,w)$ | $M_{\mathrm{long}}$ | $(s_a,s_b)$ | 激活者 | 终端 $O$ |
|---|---|---|---|---|---|
| $0$ | $(1,1,1,1)$ | $1$ | $(0.5,-0.25)$ | $a$ | $2$ |
| $1$ | $(4,-1,-1,1)$ | $0$ | $(0.5,0.75)$ | $b$ | $-1$ |

第一次 $b$ 未激活，但保留 $+1$ 的记忆。第二次它的 Attention 读出 $(1-1)/2=0$，残差内容为 $-1$；若误删第一次未激活时的写入，Attention 会返回 $-1$，Full 内容变成 $-2$。本次选择与内容计算因此使用了前次未激活时保留的记忆。

<a id="6-局部批量契约"></a>

## 6. 联合求值与分段继续

固定区域、时间窗口 $I=[b,c)$，给定可达左状态以及区域全部节点、全部时刻的完整输入纤维，包括确定为空的纤维。按如下方式构造联合函数：

1. 对非空纤维应用 Agg，对正接收计算 K/V、摘要特征和写入权重。
2. 按原始时间构造条目前缀。正接收时过滤并追加，零接收或空纤维时保留原序列，恢复每个时刻的精确状态。
3. 按所选描述模式取得旧状态、新状态或空状态的 Mem，形成描述与区域分配，返回各候选的计算状态和下一状态。
4. 收集每节点正式激活时刻 $\Theta$，在对应快照上联合求 Full，再按边和端口展开结果。

记上述状态联合函数为 $\operatorname{BatchRegionState}^{\mathrm{KV}}_{j,I}$，完整计算联合函数为 $\operatorname{BatchFull}^{\mathrm{KV}}_{v,\Theta}$。前者返回实际 P/S/U 作用、全部值及右边界，后者返回全部正式 Full 值与消息、输出记录。参考函数 $\operatorname{RefRegionState}_{j,I}$ 和 $\operatorname{RefBlock}_{K^F_{v,\Theta}}$ 沿用[分块教材](../../timed-dag-chunk-prefill-learning-note.md)，其中 $K^F_{v,\Theta}$ 是节点 $v$ 在时刻集 $\Theta$ 上的 Full 作用块。

写入不依赖本地选择或 Full，因此第 2 步等于逐次递归；第 3 步在相同输入上求同一函数；第 4 步使用同一快照。由此得到

$$
\operatorname{BatchRegionState}^{\mathrm{KV}}_{j,I}=\operatorname{RefRegionState}_{j,I},\qquad
\operatorname{BatchFull}^{\mathrm{KV}}_{v,\Theta}=\operatorname{RefBlock}_{K^F_{v,\Theta}}.
\tag{9}
$$

时间摘要可用窗口和及衰减累计，Read 与 Full 的查询在各自因果窗口上求值。输入窗口已知和各计算可并行是分别判断的条件。

分段继续保存共同切面 $Q_b=(b,(q_v^b)_v,(y_j^b)_j,W_b)$。KV 保留原始写入时间，后续查询沿用全局逻辑时间与固定位置函数。任意有限切分均恢复同一记录；位置内部切面还保留在途消息。SettleGraph 对齐的位置边界没有在途消息。

## 7. 有效状态与局部性质

固定有限切面 $B$，按原始写入时间对齐可达状态中的条目，记时间 $\xi$ 的条目为 $(\xi,k_\xi,\nu_\xi,p_\xi,w_\xi)$；未写入的槽各分量补零。在比较时刻 $\theta$ 使用

$$
\pi_{\mathrm{KV},v,\theta}(q)=
 (\chi_{v,\theta}(\xi)w_\xi,\chi_{v,\theta}(\xi)w_\xi k_\xi,
  \chi_{v,\theta}(\xi)w_\xi\nu_\xi,\chi_{v,\theta}(\xi)w_\xi p_\xi)_{0\le\xi\le B}.
\tag{10}
$$

微弱正接收可能触发清理，零接收保留原列表；过期项在该投影中均为零，之后不会重新可读。有效状态比较允许裸列表不同；同一函数的精确求值仍恢复原始序列。

**引理。** 从空缓存开始，在固定有限时间和有界接收内容、强度下，KV 更新、sum／qry 描述与式 (8) 的内容函数满足[共同架构 §6.1](architecture-learning-note.md#61-比较对象与局部条件)的有界性、零接收中性、更新连续性和读取连续性。三种 Read 模式均成立。证明见附录 A。

<a id="7-有效状态与正温度连续性"></a>
<a id="8-单位输入的整网软硬极限"></a>

该引理验证记忆与内容一侧的条件。与所选发送函数组合时，按[区域教材 §8 的性质对照](region-allocation-learning-note.md#8-发送规则如何满足架构条件)核对发送侧条件，再应用共同架构的组合命题；各候选的极限比较及其条件由表中 §7.2、§7.4—7.5 分别给出。与循环模块混合时，各节点分别验证自己的记忆条件。

## 附录 A：局部性质的证明

### A.1 有界性

$\|\mathcal N_p(x)\|\le\sqrt p$，故写入的 K/V 与摘要特征有界；有限时间内每节点至多写有限项，权重不超过 $1$。时间摘要的分母至少为 $1$；检索和 Attention 是有界 value 的凸组合再乘至多为 $1$ 的系数，位置旋转保持范数。式 (8) 的残差在有界 $u$ 上有界。所得界与区域选择温度无关。

### A.2 更新与时间推进

零接收保留原序列。正接收处，$r^{\mathrm{in}},u$ 收敛使本次写入的加权坐标连续；接收趋零时，$w(r^{\mathrm{in}})\to0$，即使 $u$ 只保持有界，新条目的 $(w,wk,w\nu,wp)$ 也趋零。旧条目按固定时间遮罩读取；清理只删除当前及未来均不可读的项。因此更新后的有效状态连续，且在以后各个固定时刻继续成立。

### A.3 描述与内容读取

时间摘要是有效加权坐标的有限和除以正连续分母，因而连续。Attention 的有效总权重有正极限时，正权重条目的 K/V 可由加权坐标恢复，匹配连续；零权重项因有界指数消失，分母趋正。总权重 $W\to0$ 时，读取范数至多为 $\min(1,W)\max_i\|\nu_i\|$，趋零。qry 逐项使用相同论证。固定位置函数、旋转和窗口保持这些结论。

所以有效状态与当前内容收敛时，Mem、当前特征和 $G_v$ 收敛。Read$^-$ 在当前时间遮罩下读旧状态，Read$^0$ 读空状态，Read$^+$ 读更新后的状态，均满足同一条件。结合 A.1—A.2，局部引理成立。$\square$

## 附录 B：输入位置与逻辑时间的换算

设节点 $v$ 可能接收的时间位于固定网格 $\theta_t=Dt+o_v^{\mathrm{pos}}$，其中 $t\in\mathbb N$ 表示输入位置，$D\ge1$ 为整数，$o_v^{\mathrm{pos}}\in\mathbb N$。取 $s_v^{\mathrm{pos}}=D$，则 $\operatorname{Pos}_v(\theta_t)=t$；在位置 $t'$ 写入的 key 使用位置 $t'$。SettleGraph 的分层日程可令 $o_v^{\mathrm{pos}}$ 等于节点所在区域的秩。

在该网格的查询时间上，有限逻辑窗口 $T_v$ 至多覆盖 $\lceil T_v/D\rceil$ 个输入位置。为该节点选择输入位置窗口 $H\in\mathbb N_{\ge1}\cup\{\infty\}$ 和按输入位置计量的短期衰减率 $\bar\beta_{\mathrm{short}}>0$，绑定

$$
T_v=DH,\qquad
\beta_{v,\mathrm{short}}=\bar\beta_{\mathrm{short}}/D.
$$

于是窗口条件与短期衰减分别为

$$
\begin{aligned}
0\le\theta_t-\theta_{t'}<T_v
 &\iff 0\le t-t'<H,\\
e^{-\beta_{v,\mathrm{short}}(\theta_t-\theta_{t'})_+}
 &=e^{-\bar\beta_{\mathrm{short}}(t-t')_+}.
\end{aligned}
$$

有限 $H$ 覆盖含当前位置的最近 $H$ 个输入位置，$H=\infty$ 覆盖全部接收历史。某位置没有接收时不写入，后续条目仍使用原始位置。距离偏置的斜率按 Pos 的输入位置单位给定。改变日程间隔 $D$ 时，同步调整位置尺度、逻辑窗口和衰减率即可保持这些输入位置尺度；分段继续保持原有 $t,\theta$。

## 附录 S：状态表示与工作量

memory_family=kv 绑定本模块的状态、Upd、Mem、当前特征和默认内容函数；memory_read=sum／qry 分别选择[第 3 节的时间摘要与内容检索](#3-记忆描述)。描述模式、区域评分和发送函数使用共同配置名称。

每个保存状态和 Full 快照须能解码为定义的有限序列。共享存储和不可变窗口视图可以减少复制；解码须恢复尚未触发清理的过期条目身份。有效状态投影相等不能替代原始记录相等。

key 可存原始形式，也可存按原始写入位置旋转后的形式，后者使用同一固定配置的逆旋转解码。qry 的投影从原始 key 取得，再施加自己的旋转；Full 已旋转的 key 通常不能直接代入。可保存检索辅助投影，仍须说明其与数学状态的对应。

带权 Attention 在正权重上使用匹配分数加 $\log\widehat w_i$ 的稳定 softmax，再乘总权重系数；空记忆直接返回零。内容与权重分开保存。频率或局部函数改变后，旧状态包含旧计算结果，重新旋转不能一般地恢复新模型的历史。

有效条目数为 $N$ 时，普通写入工作约为 $O(d^2+dr_a+dd_s)$；时间摘要朴素读取 $O(Nd_s)$，可用累计减少重复扫描；qry 的投影和检索约为 $O(dd_s+Nd_s)$，未保存辅助投影时另计重投影；Full 约为 $O(d^2+dd_f+Nd)$。有限窗口的每节点存储为 $O(T_v(2d+d_s))$。这些工作按[共同成本式](architecture-learning-note.md#s2-工作量)分别计入候选准备和激活计算。
