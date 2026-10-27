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

# Gated DeltaNet 的带权局部记忆

本模块用多头矩阵保存关联记忆。当前内容经投影、短卷积和归一化形成 query、key 与 value；保留门先衰减旧矩阵，写入门再沿当前 key 修正预测误差。当前 query 读取更新后的矩阵，经输出归一化与门控得到内容。

本文完整绑定[共同架构](../architecture-learning-note.md)的状态、Upd、Mem、当前特征与内容函数 $G_v$。Agg、三种 Read、区域评分与发送、Next 和广播 Full 沿用共同定义。短卷积使用[固定逻辑时间规则](causal-convolution-learning-note.md)。[基础循环教材](affine-delta-learning-note.md)的 delta 保留其简洁定义；本文另外给出包含多头、短卷积和输出门的候选。

本文各节依次填入架构的一次节点作用：

| 本文定义 | 在共同计算中的用途 |
|---|---|
| §1 的状态与初态，§2 的 Upd | 从旧状态和本次聚合形成 $\widetilde q_v$，全部候选保留 |
| §1 的当前特征 $p_v(u)$，§3.2 的 Mem | 从当前内容取得特征，从描述模式指定的状态取得记忆描述，供 Describe 使用 |
| §3.3 的 $G_v$ | 区域返回 $c_v$ 后，激活节点在 $\widetilde q_v$ 上计算完整内容 |

Describe 如何使用这些记忆量，以及 SelStep 如何形成评分和控制，见[区域教材 §2](../region-allocation-learning-note.md#2-指定总量的一次分配)。

## 1. 输入、参数与状态

固定节点 $v$，省略其参数下标。内容宽度为 $d$，描述宽度为 $d_s$；头数为 $n_{\mathrm{head}}\ge1$，每头的 key、value 宽度分别为 $d_k,d_m\ge1$。头编号为 $h=1,\ldots,n_{\mathrm{head}}$。当前聚合为 $(r^{\mathrm{in}},u)$，定义

$$
w=\min(1,(r^{\mathrm{in}})_+),\qquad x=\mathcal N_d(u),\qquad
p_v(u)=P_vx,\quad P_v\in\mathbb R^{d_s\times d}.
\tag{1}
$$

式 (1) 的 $p_v$ 实现架构的当前内容特征，也用于后面的摘要写入。$\mathcal N,\sigma,\operatorname{SiLU}$ 使用共同架构 §1.2 的定义。另取 $\epsilon_{\mathrm{qk}},\epsilon_{\mathrm{out}}>0$，定义稳定的 L2 归一化和带缩放的 RMS 归一化：

$$
\operatorname{L2}_{\epsilon}(z)=\frac{z}{\sqrt{\epsilon+\|z\|_2^2}},\qquad
\operatorname{RMS}_{\rho,\epsilon}(z)
 =\rho\odot\frac{z}{\sqrt{\epsilon+\|z\|_2^2/p}},
\quad z,\rho\in\mathbb R^p.
\tag{2}
$$

前者的范数至多为 $1$，后者的 $\rho$ 是可学习缩放向量。零向量处两者都有定义。

保存状态

$$
q=\bigl(s,(H_h)_{h=1}^{n_{\mathrm{head}}},\mathcal B,
          (Z_a,W_a)_{a\in\{\mathrm{short},\mathrm{long}\}}\bigr),
\quad H_h\in\mathbb R^{d_m\times d_k},\quad
Z_a\in\mathbb R^{d_s},\quad W_a\ge0.
\tag{3}
$$

$s\in\mathbb N$ 为上次正接收时间；$\mathcal B$ 保存下节定义的投影历史，宽度为 $n_{\mathrm{head}}(2d_k+d_m)$。初态为 $s=0$、全部零张量与空缓存。$Z_a,W_a$ 为供 selector 使用的接收摘要与写入量，独立于矩阵中的关联内容。

取矩阵的被动衰减率 $\lambda_h\ge0$，摘要衰减率 $\lambda_{\mathrm{short}}>0,\lambda_{\mathrm{long}}=0$。对查询时间 $\theta$ 定义

$$
H_h^-=e^{-\lambda_h(\theta-s)_+}H_h,\qquad
Z_a^-=e^{-\lambda_a(\theta-s)_+}Z_a,\qquad
W_a^-=e^{-\lambda_a(\theta-s)_+}W_a.
\tag{4}
$$

这里 $H_h^-,Z_a^-,W_a^-$ 表示当前查询时间的有效值。默认 $\lambda_h=0$：矩阵的保留与误差修正只在正接收时发生，具体门由 §2.2 定义。正的 $\lambda_h$ 另提供按逻辑时间流逝的遗忘。空档不产生额外事件，保存状态不变。

## 2. 投影与接收更新

### 2.1 投影、短卷积与查询

对每头取矩阵 $U_h^Q,U_h^K\in\mathbb R^{d_k\times d}$、$U_h^V\in\mathbb R^{d_m\times d}$。将各头原始投影依次拼接成 $z(x)$；给该向量绑定短卷积的抽头数、步长、系数和偏置。按相同顺序拆分卷积结果，定义

$$
\begin{aligned}
z(x)&=[(U_h^Qx,U_h^Kx,U_h^Vx)_{h=1}^{n_{\mathrm{head}}}],\\
[(\widehat Q_h,\widehat k_h,\nu_h)_h]
 &=\operatorname{SiLU}(\operatorname{Conv}(\mathcal B,\theta,z(x))),\\
Q_h&=d_k^{-1/2}\operatorname{L2}_{\epsilon_{\mathrm{qk}}}(\widehat Q_h),
\qquad k_h=\operatorname{L2}_{\epsilon_{\mathrm{qk}}}(\widehat k_h).
\end{aligned}
\tag{5}
$$

$Q_h$ 是本节点的记忆查询，与区域的 $Q_j^{\mathrm{sel}}$ 分别定义。当前 $Q,k,\nu$ 使用旧缓存；短卷积的快照等式保证从本次更新后的缓存求得相同值。

### 2.2 保留门与写入门

为每头取向量 $a_h^\Delta,a_h^\beta\in\mathbb R^d$ 和标量 $b_h^\Delta,b_h^\beta,A_h^{\log}$，定义 $\operatorname{softplus}(t)=\log(1+e^t)$ 以及

$$
\begin{aligned}
A_h^{\mathrm{rate}}&=e^{A_h^{\log}}>0,\qquad
\Delta_h=\operatorname{softplus}((a_h^\Delta)^\top x+b_h^\Delta),\\
g_h&=\exp(-wA_h^{\mathrm{rate}}\Delta_h),\qquad
\beta_h=w\sigma((a_h^\beta)^\top x+b_h^\beta).
\end{aligned}
\tag{6}
$$

$g_h$ 为保留比例，$\beta_h$ 为误差写入比例；$0<g_h\le1$、$0\le\beta_h\le w$。$\Delta_h$ 是输入决定的门参数，逻辑时间差已经由式 (4) 单独定义。两者均不使用区域控制 $c_v$，也不读取旧矩阵。

### 2.3 完整接收更新

现在将矩阵修正、摘要写入和卷积缓存更新组成 Upd 的返回值。记 $I$ 为 $d_k$ 维单位矩阵。当 $r^{\mathrm{in}}>0$ 时，定义

$$
\begin{aligned}
\bar H_h&=g_hH_h^-,\\
\widetilde H_h
 &=\bar H_h+\beta_h(\nu_h-\bar H_hk_h)k_h^\top
  =g_hH_h^-(I-\beta_hk_hk_h^\top)+\beta_h\nu_hk_h^\top,\\
\widetilde Z_a&=Z_a^-+wp_v(u),\qquad \widetilde W_a=W_a^-+w,\\
\widetilde{\mathcal B}&=\operatorname{ConvWrite}(\mathcal B,\theta,w,z(x)),\\
\operatorname{Upd}_v(q,\theta,(r^{\mathrm{in}},u))
 &=\bigl(\theta,(\widetilde H_h)_h,\widetilde{\mathcal B},
                 (\widetilde Z_a,\widetilde W_a)_a\bigr).
\end{aligned}
\tag{7}
$$

$r^{\mathrm{in}}\le0$ 时 Upd 返回原 $q$。全部正接收候选保留矩阵、卷积和摘要更新，未激活者也保留。当前 key 所对应的旧预测为 $\bar H_hk_h$，写入修正其与 $\nu_h$ 的差；其他 key 的关联随方向重叠而受影响。

式 (7) 的每头矩阵更新沿用[基础 Delta 式 (4)](affine-delta-learning-note.md#2-接收更新)的误差修正形式，$H_a^-,g,\beta,k,\nu$ 分别对应这里的 $H_h^-,g_h,\beta_h,k_h,\nu_h$。本文另用投影历史的短卷积形成特征，采用自己的归一化与门参数，并保存一组多头矩阵。基础模块的下标 $a$ 表示两种时间尺度，本文的 $h$ 表示头；它们的状态数量与读取规则各按自身定义。

## 3. 矩阵读取、记忆描述与完整内容

Mem 为区域提供记忆描述，$G_v$ 为激活节点提供完整内容。它们都可查询本节点的矩阵；先定义共同读取，再分别构造两个接口所需的结果。

### 3.1 共同矩阵读取

在任意待读状态 $q$ 上，使用其缓存与当前 $u$ 按式 (5) 形成查询，定义拼接读出

$$
R(q,\theta,u)=[(H_h^-(q,\theta)Q_h(q,\theta,u))_h]
 \in\mathbb R^{n_{\mathrm{head}}d_m}.
\tag{8}
$$

参数 $(q,\theta,u)$ 明确指定读取的状态、时间与当前内容。Mem 按描述模式选择 $q$；Full 使用本次新状态。只有快照与参数相同的计算才能复用同一次读出。

### 3.2 记忆描述

Mem 返回两个描述向量和两个写入量统计。这里有两种取得描述向量的方式：sum 使用接收摘要，state 另使用 §3.1 的矩阵查询。先定义候选向量，再组装完整返回。

给定 $(q,\theta,u)$，以下有效状态与读取均在这组输入上求值。对 $a\in\{\mathrm{short},\mathrm{long}\}$ 定义时间摘要；矩阵查询另取 $L^{\mathrm{read}}\in\mathbb R^{d_s\times n_{\mathrm{head}}d_m}$，得到

$$
\begin{aligned}
M_a^{\mathrm{sum}}&=Z_a^-/\max(1,W_a^-),\\
(M_{\mathrm{short}}^{\mathrm{state}},M_{\mathrm{long}}^{\mathrm{state}})
 &=(M_{\mathrm{short}}^{\mathrm{sum}},L^{\mathrm{read}}R(q,\theta,u)).
\end{aligned}
$$

在组成模型时固定 sum 或 state。所选向量和 Mem 的完整返回为

$$
\begin{aligned}
(M_{\mathrm{short}},M_{\mathrm{long}})
 &=\begin{cases}
(M_{\mathrm{short}}^{\mathrm{sum}},M_{\mathrm{long}}^{\mathrm{sum}}),&\text{选择 sum},\\
(M_{\mathrm{short}}^{\mathrm{state}},M_{\mathrm{long}}^{\mathrm{state}}),&\text{选择 state},
\end{cases}\\
\operatorname{Mem}_v(q,\theta,u)
 &=(M_{\mathrm{short}},M_{\mathrm{long}},W_{\mathrm{short}}^-,W_{\mathrm{long}}^-).
\end{aligned}
\tag{9}
$$

两种方式共用短期描述和写入量统计；选择 state 时，long 描述由当前内容查询多头矩阵取得。基础循环的 state 分别查询两种时间尺度的矩阵，本文使用一组多头矩阵，其历史保留与修正由 $\lambda_h,g_h,\beta_h$ 共同决定。$W_a$ 始终表示相应衰减规则下的接收写入量，不能由它推出矩阵还保存多少信息。

Read$^+$／$^-$／$^0$ 分别选择新状态、旧状态和零初态，在该状态上求式 (9)。[区域教材定义的 Describe](../region-allocation-learning-note.md#51-从所选函数得到完整返回)使用这个 Mem 返回值、式 (1) 的 $p_v(u)$ 及当前接收 $(r^{\mathrm{in}},u)$ 计算 $d_v$；SelStep 再从区域描述族形成评分及控制 $c_v$。默认使用 state 描述与 Read$^+$。

### 3.3 完整内容

区域从描述得到正控制 $c_v$ 后，Full 在新状态上求以下内容函数。矩阵查询使用 §3.1 的共同读取，后续变换在本节定义。

取 $U_h^o\in\mathbb R^{d_m\times d}$、$\rho_h\in\mathbb R^{d_m}$ 以及 $O\in\mathbb R^{d\times n_{\mathrm{head}}d_m}$。记 $R_h$ 为式 (8) 的第 $h$ 个分块，定义记忆内容混合

$$
\operatorname{Mix}(q,\theta,u)
 =O\bigl[(\operatorname{RMS}_{\rho_h,\epsilon_{\mathrm{out}}}(R_h)
                \odot\operatorname{SiLU}(U_h^o x))_h\bigr].
\tag{10}
$$

每头先做 RMS 归一化，再乘输出门，最后拼接投影。输出门决定内容分量，不决定节点是否激活。

固定前馈宽度 $d_f$，取 $W^{\mathrm{gate}},W^{\mathrm{up}}\in\mathbb R^{d_f\times d}$、$W^{\mathrm{down}}\in\mathbb R^{d\times d_f}$ 和非负残差尺度，完成内容函数：

$$
\begin{aligned}
x_F&=u+\alpha_{\mathrm{mem}}\operatorname{Mix}(q,\theta,u),\\
G_v(q,\theta,u)&=x_F+\alpha_{\mathrm{ff}}W^{\mathrm{down}}
 [\operatorname{SiLU}(W^{\mathrm{gate}}\mathcal N_d(x_F))
      \odot W^{\mathrm{up}}\mathcal N_d(x_F)].
\end{aligned}
\tag{11}
$$

Full 在本次新状态上求 $G_v$，正控制时向每条出边与输出端口发送 $(c_v,G_v)$。保留门 $g_h$、误差写入门 $\beta_h$ 和输出门 $\operatorname{SiLU}(U_h^ox)$ 分别控制旧矩阵保留、关联修正和输出内容；发送增益 $\gamma_v$ 与发送强度 $c_v$ 使用共同区域规则。

### 3.4 一个标量例子

取单头、所有宽度和抽头数为 $1$，矩阵被动衰减为零，从零状态接收 $(1,1)$。选择有限参数，使当前 $k=Q=1/2,\nu=2,g=\beta=1/2,p_v(u)=1$；这些值均在上述函数的可取范围内。于是 $\widetilde H=1/2$、$R=1/4$，两个摘要都有 $\widetilde Z=\widetilde W=1$。

再取 $\epsilon_{\mathrm{out}}=1,\rho=O=1,U^ox=1$、$\alpha_{\mathrm{mem}}=1,\alpha_{\mathrm{ff}}=0$，有 $G_v=1+\operatorname{SiLU}(1)/\sqrt{17}$。单有效候选若由约束分配得到 $c=\mu=1$，就广播 $(1,G_v)$。同样的第二次门与 key、value 会把矩阵更新为 $23/32$，展示预测误差对重复写入的修正。

## 4. 联合求值与切面继续

给定某节点的完整聚合输入序列，先按原始时间形成所有投影与卷积，再得到各次门、key、value。它们不依赖未知的旧矩阵，故第 $i$ 次正接收的每头更新可以写成

$$
\begin{aligned}
H_{h,i}&=H_{h,i-1}A_{h,i}+B_{h,i},\\
A_{h,i}&=e^{-\lambda_h(\theta_i-s_{i-1})}g_{h,i}
                  (I-\beta_{h,i}k_{h,i}k_{h,i}^\top),\qquad
B_{h,i}=\beta_{h,i}\nu_{h,i}k_{h,i}^\top.
\end{aligned}
\tag{12}
$$

$s_{i-1}$ 为此前保存时间。每头的系数代入[基础教材 §4.1 的仿射复合](affine-delta-learning-note.md#41-仿射复合)，即可按其结合律求得全部中间矩阵；摘要同样为仿射更新。再按描述模式形成描述、区域选择和 Full，构成共同架构 §7 的联合契约。实际的高效分块算法还需利用 $I-\beta kk^\top$ 的结构；显式形成并乘稠密转移矩阵可能更昂贵。

继续保存式 (3) 的完整状态，以及正典切面中的区域历史和在途消息。卷积保持原始时间，矩阵与摘要保持上次写入时间。位置空档、切分长度或延后求 Full 都不重置这些量。每个 Full 使用自己的计算快照。

## 5. 有效状态与组合性质

在比较时刻 $\theta$ 定义

$$
\pi_{\mathrm{gdn},v,\theta}(q)
 =\bigl((H_h^-)_h,(Z_a^-,W_a^-)_a,
                 \pi_{\mathrm{conv},\theta}(\mathcal B)\bigr).
\tag{13}
$$

**引理。** 从零初态开始，固定有限时间范围与有限参数，本文模块满足[共同架构 §6.1](../architecture-learning-note.md#61-比较对象与局部条件)的四项局部条件。证明见附录 A。

该引理验证本模块的记忆与内容条件。与其他记忆混合时，各节点分别使用相应引理；发送侧条件按[区域教材 §8 的性质对照](../region-allocation-learning-note.md#8-发送规则如何满足架构条件)核对，再应用架构的组合命题。各候选的极限比较及其条件由表中 §7.2、§7.4—7.5 分别给出。

## 附录 A：局部性质的证明

**有界性。** 固定参数时，$x=\mathcal N_d(u)$ 有界，故原始投影有界。短卷积引理使卷积、$Q,k,\nu$ 和门参数有界。$\|k\|\le1$ 且 $0\le\beta\le1$，矩阵 $I-\beta kk^\top$ 的特征值在 $[0,1]$ 内。因此

$$
\|\widetilde H_h\|_F\le\|H_h^-\|_F+Cw.
$$

有限次接收使矩阵、摘要和写入量有界。RMS 的正分母、有限投影和前馈函数给出 Mem 与 $G_v$ 的有限时间界。所用函数不含区域选择温度，这些界可在共同定理的温度族上统一。

**零与弱接收。** 零接收返回原状态。若 $r^{\mathrm{in}}\to0$，则 $w\to0,g_h\to1,\beta_h\to0$；当前内容只需有界。式 (7) 的矩阵趋于 $H_h^-$，摘要与写入量趋于旧有效值。短卷积引理使新增缓存槽趋零且旧槽不移动。时间戳虽变为 $\theta$，指数衰减满足半群律，故在任意固定未来时间的有效投影也趋向无此事件的状态。

**正接收与读取。** 正极限接收处，当前内容及旧有效状态收敛；卷积、归一化、门和全部代数运算连续，给出更新与读取的收敛。零初态、旧状态、新状态按各自快照读取；式 (5) 的当前卷积可在新快照精确重建。由此验证四项局部条件。$\square$

## 附录 S：配置、原模型对应与工作量

memory_family=gated_delta 选择本文完整函数；memory_read=sum／state 选择式 (9)。配置还给出 num_heads、head_k_dim、head_v_dim、描述及前馈宽度、conv_size、conv_stride、归一化常数、被动衰减率、全部参数与残差尺度。两种归一化的常数分别为 qk_eps、output_norm_eps；共同输入归一化沿用架构的 $\epsilon$。

[Gated DeltaNet 论文 §3.4](https://arxiv.org/html/2412.06464v2#S3.SS4)给出投影、短卷积、L2 归一化、门控 Delta 及输出门的构成。从零矩阵与空缓存开始，在时间 $\theta_i=\theta_0+i\delta_{\mathrm{conv}}$ 连续接收、每次权重为 $1$、$\lambda_h=0$ 时，本文的 Mix 恢复相应的普通序列记忆混合器：对照采用相同的归一化输入 $x$、投影、卷积系数与偏置、门、缩放和头排列。$w$ 对门与历史写入的调制、逻辑时间空档、selector 摘要和式 (11) 的外部内容绑定是这里明确给出的 Tide 选择。

一个具体实现对应为 [FLA GatedDeltaNet](https://github.com/fla-org/flash-linear-attention/blob/9d478facb282410dc150051d5f57205cd3812f31/fla/layers/gated_deltanet.py)，取相同 Q/K/V 头数、开启短卷积与 SiLU 输出门、allow_neg_eigval=False。对照时 $b_h^\beta=0$，输出 RMS 的缩放向量在各头共享，卷积偏置按同一 conv_bias 开关取值。其[循环核](https://github.com/fla-org/flash-linear-attention/blob/9d478facb282410dc150051d5f57205cd3812f31/fla/ops/gated_delta_rule/fused_recurrent.py)取 $\epsilon_{\mathrm{qk}}=10^{-6}$ 和 query 缩放 $d_k^{-1/2}$；输出 RMS 常数取 $10^{-5}$。本对应限定这些开关和实数公式，浮点内核的容差另验。整个外部语言模型的嵌入、层排列、训练配方和任务读出由其架构另行定义。

矩阵状态大小为 $O(n_{\mathrm{head}}d_md_k)$，另加卷积槽与 $O(d_s)$ 摘要；都不随接收次数增长。每次正接收的 Delta 更新需 $O(n_{\mathrm{head}}d_md_k)$，投影与卷积另计。state 描述又包含矩阵查询，Full 可复用同快照的中间量。降低本次激活数可省输出变换、前馈和广播，当前候选的状态更新仍属于 Upd。

本文有限时间连续性与转移系数范数界不提供长程任务梯度的下界；门、压缩与覆盖的训练取舍见[设计总览](../../../memos/design/weighted-message-design.md#2-保存历史与分配信用)，程序对应见[实现核对](../../../reference/weighted-message-implementation-checks.md#4-记忆模块)。
