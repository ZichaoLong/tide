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

# Mamba-2／SSD 的带权局部记忆

本模块给出一套具体的输入相关状态空间更新。每个头用一个标量控制整个矩阵的保留比例，当前输入决定写入向量与读取向量；完整内容还包含当前输入的直通项和输出门。相同递归可以展开为带衰减的历史加权和，这一联系称为状态空间对偶（State Space Duality，SSD）。

本文采用 Mamba-2 的 Mamba2Simple 计算形状，完整绑定[共同架构](../architecture-learning-note.md)的状态、Upd、Mem、当前特征与 $G_v$。短卷积使用[固定逻辑时间规则](causal-convolution-learning-note.md)，Agg、Read、区域分配、Next 和广播 Full 沿用共同定义。SSM 是一类状态空间模型的统称；这里每个矩阵的尺寸、门、离散递归和读出均由下文确定。

本文各节依次填入架构的一次节点作用：

| 本文定义 | 在共同计算中的用途 |
|---|---|
| §1 的状态与初态，§2 的 Upd | 从旧状态和本次聚合形成 $\widetilde q_v$，全部候选保留 |
| §2 的当前特征 $p_v(u)$，§3.2 的 Mem | 从当前内容取得特征，从描述模式指定的状态取得记忆描述，供 Describe 使用 |
| §3.3 的 $G_v$ | 区域返回 $c_v$ 后，激活节点在 $\widetilde q_v$ 上计算完整内容 |

Describe 如何使用这些记忆量，以及 SelStep 如何形成评分和控制，见[区域教材 §2](../region-allocation-learning-note.md#2-指定总量的一次分配)。

## 1. 尺寸、状态与时钟

固定节点，内容宽度为 $d$、描述宽度为 $d_s$。取头数 $n_{\mathrm{head}}$、每头内容宽度 $d_m$、状态宽度 $d_k$，以及整除头数的组数 $n_{\mathrm{group}}$，均为正整数。内部内容宽度为 $d_{\mathrm{inner}}=n_{\mathrm{head}}d_m$。每头归属

$$
\operatorname{grp}(h)=1+\left\lfloor\frac{h-1}{n_{\mathrm{head}}/n_{\mathrm{group}}}\right\rfloor,
\quad 1\le h\le n_{\mathrm{head}}.
\tag{1}
$$

同一组共享本次写入和读取的 $d_k$ 维向量，各头仍有独立矩阵、内容向量与门参数。

保存状态

$$
q=\bigl(s,(H_h)_h,\mathcal B,(Z_a,W_a)_{a\in\{\mathrm{short},\mathrm{long}\}}\bigr),
\quad H_h\in\mathbb R^{d_m\times d_k},\quad
Z_a\in\mathbb R^{d_s},\quad W_a\ge0.
\tag{2}
$$

$s\in\mathbb N$ 为上次正接收时间，$\mathcal B$ 是宽度 $d_{\mathrm{inner}}+2n_{\mathrm{group}}d_k$ 的投影历史缓存。初态为时间零、零张量和空缓存。摘要 $Z_a$ 与接收写入量 $W_a$ 为区域描述提供辅助信息。

取矩阵被动衰减率 $\lambda_h\ge0$、摘要率 $\lambda_{\mathrm{short}}>0,\lambda_{\mathrm{long}}=0$。在查询时间 $\theta$ 定义

$$
H_h^-=e^{-\lambda_h(\theta-s)_+}H_h,\qquad
Z_a^-=e^{-\lambda_a(\theta-s)_+}Z_a,\qquad
W_a^-=e^{-\lambda_a(\theta-s)_+}W_a.
\tag{3}
$$

默认 $\lambda_h=0$，矩阵只在正接收时更新。正的 $\lambda_h$ 为 Tide 的逻辑时间空档增加被动遗忘；无接收时保存状态保持，未来读取应用式 (3)。

## 2. 投影、门与写入

### 2.1 当前特征与输入投影

给定聚合 $(r^{\mathrm{in}},u)$，令 $w=\min(1,(r^{\mathrm{in}})_+)$、$x_0=\mathcal N_d(u)$。取矩阵 $P_v\in\mathbb R^{d_s\times d}$，以 $p_v(u)=P_vx_0$ 实现架构的当前内容特征，同一特征用于下面的摘要写入。

取输入投影矩阵
$U\in\mathbb R^{(2d_{\mathrm{inner}}+2n_{\mathrm{group}}d_k+n_{\mathrm{head}})\times d}$，按以下顺序拆分其输出：

$$
Ux_0=[z^o,(\widehat\nu_h)_h,(\widehat B_g)_g,(\widehat C_g)_g,(\widehat\Delta_h)_h],
\tag{4}
$$

其中 $z^o\in\mathbb R^{d_{\mathrm{inner}}}$，$\widehat\nu_h\in\mathbb R^{d_m}$，$\widehat B_g,\widehat C_g\in\mathbb R^{d_k}$，$\widehat\Delta_h\in\mathbb R$。这里 $g$ 为组编号。

### 2.2 短卷积

给拼接向量 $z^{\mathrm{conv}}=[(\widehat\nu_h)_h,(\widehat B_g)_g,(\widehat C_g)_g]$ 绑定短卷积参数，定义

$$
[(\nu_h)_h,(B_g)_g,(C_g)_g]
 =\operatorname{SiLU}(\operatorname{Conv}(\mathcal B,\theta,z^{\mathrm{conv}})).
\tag{5}
$$

只有这三个分支经过短卷积。输出门 $z^o$ 和步长分支 $\widehat\Delta_h$ 直接由当前输入决定。$B_g,C_g$ 使用上述值，不另作 L2 归一化。

### 2.3 有效步长与保留比例

取有限标量 $A_h^{\log},b_h^\Delta$，以 $\operatorname{softplus}(t)=\log(1+e^t)$ 定义

$$
A_h^{\mathrm{rate}}=e^{A_h^{\log}},\quad
\Delta_h=\operatorname{softplus}(\widehat\Delta_h+b_h^\Delta),\quad
d_h=w\Delta_h,\quad g_h=e^{-A_h^{\mathrm{rate}}d_h}.
\tag{6}
$$

$d_h$ 是本次输入决定的有效步长，$g_h$ 是保留比例；它们与逻辑时间差分别定义。弱接收同时减少遗忘与写入。

### 2.4 完整接收更新

将矩阵、摘要和卷积缓存更新组成 Upd。正接收时令

$$
\begin{aligned}
\widetilde H_h&=g_hH_h^-+d_h\nu_hB_{\operatorname{grp}(h)}^\top,\\
\widetilde Z_a&=Z_a^-+wp_v(u),\qquad \widetilde W_a=W_a^-+w,\\
\widetilde{\mathcal B}&=\operatorname{ConvWrite}(\mathcal B,\theta,w,z^{\mathrm{conv}}),\\
\operatorname{Upd}_v(q,\theta,(r^{\mathrm{in}},u))
 &=\bigl(\theta,(\widetilde H_h)_h,\widetilde{\mathcal B},
                    (\widetilde Z_a,\widetilde W_a)_a\bigr).
\end{aligned}
\tag{7}
$$

$r^{\mathrm{in}}\le0$ 时返回原 $q$。所有正接收候选保留本次更新，包含未激活节点。本文的离散写入系数为 $d_h$；若用连续系统的精确零阶保持系数 $(1-e^{-A_h^{\mathrm{rate}}d_h})/A_h^{\mathrm{rate}}$ 替换它，就定义了另一个数值函数。

式 (7) 与[基础仿射式 (4)](affine-delta-learning-note.md#2-接收更新)同样由旧矩阵的标量保留与当前外积写入组成。按代数位置，基础模块的 $g,w,\nu,k$ 对应这里的 $g_h,d_h,\nu_h,B_{\operatorname{grp}(h)}$。本文重新定义这些量：$d_h=w\Delta_h$ 可大于 $1$，$B_g$ 经短卷积取得且不作 L2 归一化，保留比例与写入系数共用 $\Delta_h$。读取另用 $C_g$，输出还含直通项和门，均在下一节定义。

## 3. 矩阵读取、记忆描述与完整内容

Mem 为区域提供记忆描述，$G_v$ 为激活节点提供完整内容。它们都可查询本节点的矩阵；先定义共同读取，再分别构造两个接口所需的结果。

### 3.1 共同矩阵读取

在待读状态的缓存上按式 (5) 取得当前读取向量，定义

$$
R_h(q,\theta,u)=H_h^-C_{\operatorname{grp}(h)},\qquad
R(q,\theta,u)=[(R_h)_h]\in\mathbb R^{d_{\mathrm{inner}}}.
\tag{8}
$$

参数 $(q,\theta,u)$ 明确指定读取的状态、时间与当前内容。Mem 按描述模式选择 $q$；Full 使用本次新状态。只有快照与参数相同的计算才能复用同一次读出。

### 3.2 记忆描述

Mem 返回两个描述向量和两个写入量统计。这里有两种取得描述向量的方式：sum 使用接收摘要，state 另使用 §3.1 的矩阵查询。先定义候选向量，再组装完整返回。

给定 $(q,\theta,u)$，以下有效状态与读取均在这组输入上求值。对 $a\in\{\mathrm{short},\mathrm{long}\}$ 定义时间摘要；矩阵查询另取 $L^{\mathrm{read}}\in\mathbb R^{d_s\times d_{\mathrm{inner}}}$，得到

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

两种方式共用短期描述和写入量统计；选择 state 时，long 描述由当前内容查询多头矩阵取得。这与 Gated DeltaNet 的通道安排相同；基础循环的 state 则分别查询两种时间尺度的矩阵。本文矩阵按保留比例 $g_h$、有效步长 $d_h$ 和被动衰减更新；$W_a$ 统计接收写入量，矩阵的遗忘、相消和压缩并不由它精确计量。

Read$^+$／$^-$／$^0$ 分别在新、旧和初始状态上求式 (9)。[区域教材定义的 Describe](../region-allocation-learning-note.md#51-从所选函数得到完整返回)使用这个 Mem 返回值、§2 的 $p_v(u)$ 及当前接收 $(r^{\mathrm{in}},u)$ 计算 $d_v$；SelStep 再从区域描述族形成评分及控制 $c_v$。默认使用 state 描述与 Read$^+$。本次卷积可以从旧缓存或新缓存得到相同值，矩阵读取仍按指定状态进行。

### 3.3 直通项、输出门与前馈

区域从描述得到正控制 $c_v$ 后，Full 在新状态上求以下内容函数。矩阵查询使用 §3.1 的共同读取，后续变换在本节定义。

给每头取标量 $D_h$，给完整内部宽度取缩放向量 $\rho\in\mathbb R^{d_{\mathrm{inner}}}$、常数 $\epsilon_{\mathrm{out}}>0$ 和矩阵 $O\in\mathbb R^{d\times d_{\mathrm{inner}}}$。定义

$$
\begin{aligned}
y&=[(R_h+D_h\nu_h)_h],\qquad \bar y=y\odot\operatorname{SiLU}(z^o),\\
\operatorname{Mix}(q,\theta,u)
 &=O\left(\rho\odot\frac{\bar y}
  {\sqrt{\epsilon_{\mathrm{out}}+\|\bar y\|_2^2/d_{\mathrm{inner}}}}\right).
\end{aligned}
\tag{10}
$$

$D_h\nu_h$ 是当前卷积特征的直通项；Mem 的矩阵描述不含此项。这里先乘输出门，再对整个内部宽度做 RMS 归一化，对应所选的 Mamba2Simple 形状。Gated DeltaNet 教材则在每头归一化之后乘门，两套顺序由各自函数确定。

固定 $d_f$，取 $W^{\mathrm{gate}},W^{\mathrm{up}}\in\mathbb R^{d_f\times d}$、$W^{\mathrm{down}}\in\mathbb R^{d\times d_f}$ 及非负残差尺度，完成

$$
\begin{aligned}
x_F&=u+\alpha_{\mathrm{mem}}\operatorname{Mix}(q,\theta,u),\\
G_v(q,\theta,u)&=x_F+\alpha_{\mathrm{ff}}W^{\mathrm{down}}
 [\operatorname{SiLU}(W^{\mathrm{gate}}\mathcal N_d(x_F))
               \odot W^{\mathrm{up}}\mathcal N_d(x_F)].
\end{aligned}
\tag{11}
$$

Full 在新状态上求此内容，以共同规则广播 $(c_v,G_v)$。输入步长、保留门和输出门参与记忆与内容计算，区域增益及 $c_v$ 控制传播。

### 3.4 一个标量例子

取单头、单组、各宽度和抽头数为 $1$，从零状态接收 $(1,1)$，令矩阵被动衰减为零。选取有限参数使 $\Delta=1/2,A^{\mathrm{rate}}=2\log2,\nu=2,B=1,C=3,D=1/4$，则 $g=1/2$、$\widetilde H=1$、$R=3$、$y=7/2$。同样的下一次接收得到矩阵 $3/2$，展示标量遗忘与关联叠加。

令 $p_v(u)=1,z^o=1,\rho=O=\epsilon_{\mathrm{out}}=1$、$\alpha_{\mathrm{mem}}=1,\alpha_{\mathrm{ff}}=0$。第一次接收的两个摘要均有 $\widetilde Z=\widetilde W=1$，完整内容为 $G_v=1+\bar y/\sqrt{1+\bar y^2}$，其中 $\bar y=(7/2)\operatorname{SiLU}(1)$。单有效候选由约束分配得到 $c=1$ 时，广播 $(1,G_v)$。

## 4. 历史展开、联合求值与继续

给定完整聚合输入序列，先在原始时间上求所有投影与短卷积。对第 $i$ 次正接收，记 $s_{i-1}$ 为此前保存时间，定义

$$
\zeta_{h,i}=e^{-\lambda_h(\theta_i-s_{i-1})}g_{h,i},\qquad
J_{h,i}=d_{h,i}\nu_{h,i}B_{\operatorname{grp}(h),i}^{\top}.
\tag{12}
$$

于是 $H_{h,i}=\zeta_{h,i}H_{h,i-1}+J_{h,i}$。记 $I$ 为 $d_k$ 维单位矩阵，这对应系数对 $(\zeta_{h,i}I,J_{h,i})$；复合与前缀求值使用[基础教材 §4.1 的仿射复合](affine-delta-learning-note.md#41-仿射复合)。从零矩阵出发可展开为

$$
R_{h,i}
 =\sum_{j=1}^i d_{h,j}
       \left(\prod_{\ell=j+1}^i\zeta_{h,\ell}\right)
       \nu_{h,j}\bigl(B_{\operatorname{grp}(h),j}^{\top}
                     C_{\operatorname{grp}(h),i}\bigr).
\tag{13}
$$

空积取 $1$。每条历史写入通过中间保留比例到达当前，当前 $C$ 与历史 $B$ 的内积决定读出方向。该权重一般不归一化，也可有符号；它与 softmax KV Attention 的归一化加权和有不同定义。

已知驱动输入后，逐次递归、仿射前缀或保持同一公式的 SSD 分块算法可以求得中间矩阵。随后计算所需 Read、区域选择与 Full，即给出完整的联合契约。高效内核还需匹配头、分组、卷积时间和门参数；仿射性本身不消除全图自适应消息的依赖。

分段继续保存完整矩阵、摘要、时间戳和卷积缓存，使用原始逻辑时间，并保留正典切面的区域历史和跨界消息。指数衰减的半群律、卷积的固定时间槽与递归的确定性保证整段与分段得到相同记录。参数和状态形状变化后的迁移另行定义。

## 5. 有效状态与组合性质

定义

$$
\pi_{\mathrm{ssd},v,\theta}(q)
 =\bigl((H_h^-)_h,(Z_a^-,W_a^-)_a,
                  \pi_{\mathrm{conv},\theta}(\mathcal B)\bigr).
\tag{14}
$$

**引理。** 固定有限参数，从零初态开始，本文模块在固定有限时间内满足[共同架构 §6.1](../architecture-learning-note.md#61-比较对象与局部条件)的有界性、零接收中性、更新连续性和读取连续性。证明见附录 A。

该引理验证本模块的记忆与内容条件。与其他记忆混合时，各节点分别使用相应引理；发送侧条件按[区域教材 §8 的性质对照](../region-allocation-learning-note.md#8-发送规则如何满足架构条件)核对，再应用架构的组合命题。各候选的极限比较及其条件由表中 §7.2、§7.4—7.5 分别给出。

## 附录 A：局部性质的证明

**有界性。** 有限参数和输入归一化给出有界原始投影。短卷积引理给出有界的 $\nu,B,C$；有限的步长投影使 $0\le d_h\le Cw$。因 $0<g_h\le1$、被动衰减至多为 $1$，式 (7) 给出

$$
\|\widetilde H_h\|_F\le\|H_h^-\|_F+C'w.
$$

有限次接收使矩阵、摘要和统计量有界。读出、直通项与输出门有界，RMS 的分母为正，前馈残差在有界输入上有界，得到描述和内容的界。此界不依赖区域温度。

**零与弱接收。** 零接收返回原状态。$r^{\mathrm{in}}\to0$ 时，$w,d_h\to0$、$g_h\to1$，写入项、摘要增量和缓存新增槽均趋零；遗忘趋于中性。投影内容只需有界。更新后矩阵趋于旧状态在当前时间的有效值，指数衰减半群律保证未来时刻仍有同一极限。缓存由短卷积引理处理。

**正接收与读取。** 当前内容和旧有效状态收敛时，投影、卷积、门、乘加与带正分母的归一化连续，故正接收更新及 Mem、$G_v$ 收敛。旧、新、零初态的读取均使用定义的时间与快照，满足共同条件。$\square$

## 附录 S：配置、原模型对应与工作量

memory_family=ssd 选择本文函数，memory_read=sum／state 选择式 (9)。配置给出 num_heads、head_v_dim、state_dim、num_groups，分别对应 $n_{\mathrm{head}},d_m,d_k,n_{\mathrm{group}}$；另给描述和前馈宽度、conv_size、conv_stride、output_norm_eps、被动衰减率、全部参数和残差尺度。

[Mamba-2 论文](https://arxiv.org/abs/2405.21060)研究此类标量转移与结构化历史计算的联系。本文具体对照 [Mamba2Simple](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba2_simple.py)：相同归一化输入 $x_0$，输入投影按 z、x、B、C、dt 拆分，卷积作用于 x/B/C，标量 dt 使用 softplus，写入为 dt 乘外积，输出含逐头 D 直通。参考实现中的 x 对应本文 $\nu$。

从零矩阵与空缓存开始，在时间 $\theta_i=\theta_0+i\delta_{\mathrm{conv}}$ 连续接收、每次权重为 $1$、$\lambda_h=0$，并匹配参数、头分组、卷积系数与偏置、$\epsilon_{\mathrm{out}}=10^{-5}$ 时，Mix 恢复该参考混合器的实数公式。对应选择 bias=False、learnable_init_states=False、无 dt 额外裁剪；其 norm_before_gate=False 对应式 (10) 的先门控后归一化。内部宽度满足参考 expand 与 d_model 的乘积关系。其他 Mamba 变式可另作完整绑定。

Tide 增加接收权重对步长和卷积历史的调制、逻辑时钟、描述摘要与外部内容残差；式 (11) 选择了供节点使用的前馈组合。原模型的整网结构与训练结果由其独立架构决定。

每节点矩阵大小为 $O(n_{\mathrm{head}}d_md_k)$，另加固定范围卷积和摘要。一次接收的更新、一次矩阵读取各需 $O(n_{\mathrm{head}}d_md_k)$，分组减少 B/C 投影宽度而不删除各头状态；投影、卷积和前馈另计。所有正接收候选都更新状态，Full 稀疏只省相应内容计算与广播。有限时间连续性不保证无限时间或完整参数梯度稳定；比较见[设计总览](../../../memos/design/weighted-message-design.md#2-保存历史与分配信用)，验证范围见[实现核对](../../../reference/weighted-message-implementation-checks.md#4-记忆模块)。
