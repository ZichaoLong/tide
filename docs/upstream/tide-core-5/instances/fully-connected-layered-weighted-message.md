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
---

# 全连接分层图上的带权消息实例

本文在[全连接分层拓扑](../topologies/fully-connected-layered.md)上绑定带权消息的局部函数，给出 KV、稠密 Attention 和混合记忆实例，并推导各自的传播、继续与计算量。局部函数的定义与证明分别见[共同架构](../local-semantics/weighted-message/architecture-learning-note.md)、[区域分配](../local-semantics/weighted-message/region-allocation-learning-note.md)及各记忆教材。第 4.3 节包含基础循环、Gated DeltaNet 和 SSD 的完整替换绑定。

## 1. 完整的 KV 基础绑定

### 1.1 图与局部选择

取拓扑定义中的深度 $L$、区域宽度 $n$ 和输入长度 $N$，沿用节点 $v_{\ell,s}$、区域、边 $a_{\ell,s,r}$、端口及日程，$D=L+2$。全部实例保持这份图与时间绑定。

以下先定义一族 KV 基线。表中的函数均由共同架构及模块教材给定；尺寸与参数在一次模型绑定中固定。

| 对象 | 基线固定的选择 | 仍需给定的参数及范围 |
|---|---|---|
| 图与时间 | 本节的全连接分层拓扑 | 全图的 $L,n,N$ |
| 载荷与聚合 | $P=X_v=\mathbb R^{1+d}$，区域共享 Agg | 全图宽度 $d$；逐区域的聚合宽度、矩阵、来源嵌入与残差尺度；来源映射见 §1.3 |
| 记忆与内容 | KV 空缓存初态、时间摘要、KV 内容函数 | 同区域的状态宽度与共享 KV 基座；逐节点的适配器、摘要与 Full 参数 |
| 描述与选择 | Read$^+$、拼接局部评分、独立评分、固定增益、二次正则约束分配 | 逐节点评分参数及固定增益；逐区域的 $k_j,\eta_j,\tau_{\mathrm{sel},j}$，其中 $1\le k_j\le n$；增益满足共同有限上界 |
| 状态采用与发送 | 全部候选保留更新、单点区域历史、广播 Full、$K_j=n$ | 使用共同 Next，无额外状态采用参数 |
| 窗口与位置 | §1.2 的位置尺度与衰减换算 | 逐区域的窗口、短期衰减与旋转参数，见 §4.2 |
| 外部边界 | §1.4 的单位入口和残差读出 | 内容序列 $h_t$、读出矩阵与尺度 |

本节给出基础绑定所需的选择、来源映射与外部边界。每个完整参数取值确定一个模型；第 2 节给出具体取值及运行，第 3 节推导传播性质，第 4 节给出其他局部选择及各例的最终绑定。相连节点共用载荷宽度，描述与评分按共同架构的组合条件检查相容性。

### 1.2 KV 的位置、窗口与衰减

采用 [KV 教材的时间换算](../local-semantics/weighted-message/kv-memory-learning-note.md#附录-b输入位置与逻辑时间的换算)。对区域 $\ell$ 中的节点取 $o_v^{\mathrm{pos}}=\ell,s_v^{\mathrm{pos}}=D$，即

$$
\operatorname{Pos}_{v_{\ell,s}}(\theta)=\frac{\theta-\ell}{D}.
$$

为区域 $\ell$ 选择输入位置窗口 $H_\ell\in\mathbb N_{\ge1}\cup\{\infty\}$ 和短期率 $\bar\beta_{\ell,\mathrm{short}}>0$，绑定 $T_v=DH_\ell$、$\beta_{v,\mathrm{short}}=\bar\beta_{\ell,\mathrm{short}}/D$，长期率为零。于是查询和 key 按输入位置编码，窗口覆盖最近 $H_\ell$ 个位置或全部接收历史。Read、Full 的旋转及可选距离偏置逐区域声明，见 §4.2。

### 1.3 来源类别与参数共享

来源类别按区域定义。第一层只使用入口类别 $\mathrm{in}$，令 $\operatorname{SrcClass}_{v_{1,s}}(\mathrm{ext},i_s)=\mathrm{in}$。对 $\ell>1$，类别为上一层节点身份，令

$$
\operatorname{SrcClass}_{v_{\ell,r}}(\mathrm{msg},a_{\ell-1,s,r})
=v_{\ell-1,s}.
$$

同一父节点通向不同接收者的消息因此具有相同类别；原始原子仍保留各自的边身份与时间。区域内共享聚合参数和 KV 基座，各节点分别具有适配器、摘要、评分与 Full 参数，私有状态分别保存。

### 1.4 输入与连续读出

设 $h_t\in\mathbb R^d$ 是只依赖输入前缀的表示序列。全部输入端口在位置 $t$ 接收相同单位强度载荷：

$$
x_{i_s}(t)=(1,h_t),\qquad 1\le s\le n.
\tag{1}
$$

完整输出记录记为 $Z$。采用局部架构教材 §4 的同一个 $\operatorname{ReadOut}$，固定

$$
\begin{aligned}
\Lambda_t&=\{(Dt+L,o_s):1\le s\le n\},\\
O_t&=\operatorname{ReadOut}_{\Lambda_t}(Z),\\
y_t&=h_t+\alpha_{\mathrm{out}}W_{\mathrm{out}}O_t,
\qquad W_{\mathrm{out}}\in\mathbb R^{d\times d},\quad
\alpha_{\mathrm{out}}\ge0.
\end{aligned}
\tag{2}
$$

终端端口记录的时间为 $Dt+L$。在 $Dt+L+1$ 比较这组已经确定的槽，取得图外读出 $O_t$；该读出不产生新的图内作用或输出记录。通用 ReadOut 在全部终端缺席时定义为零；本实例的单位输入与非空选择保证每位置都有正强度终端输出，证明见 §3。内容相消仍可能使数值 $O_t=0$。

若采用 SettleGraph §6.4 的派生单槽记法，可在 $Z_{\Lambda_t}$ 非空时令 $b_t=(1,O_t)$，为空时令 $b_t=\bot$，并取 $\pi_{\mathrm{out}}(1,z)=z,\pi_{\mathrm{out}}(\bot)=0$。这个派生读出由端口记录确定。

式 (2) 可以接入已有因果主干。独立语言模型可令 $h_t$ 为 token 嵌入，以 $y_t$ 接任务读出。由位置顺序与私有状态递归，$y_t$ 只依赖截至位置 $t$ 的输入。

## 2. 两层图上的两次输入

取 $L=n=N=2$，故 $D=4$。两个位置的输入内容为 $h_0=1,h_1=-1$；第一层在时间 $1,5$ 接收，第二层在时间 $2,6$ 接收。采用 §1.1 的 KV 基线，并给定以下参数：

- $d=d_s=H_{\mathrm{att}}=d_h=r_a=p_a=d_f=1$，评分隐藏宽度 $d_r=2$，归一化常数 $\epsilon=1$；聚合残差尺度为零，因此 $u$ 为按消息强度加权的均值。
- 窗口无限，旋转与距离偏置为零，短期衰减率为 $\log2/D$。共享 KV 基座为 $[0;\sqrt2]$，适配器为零，摘要投影为 $\sqrt2$。于是 key 为零，写入的 value 与摘要均为
  $f(u)=\sqrt2\,u/\sqrt{1+u^2}$。
- Full 的 query 为零、输出投影为 $1$，$\alpha_{\mathrm{att}}=1,\alpha_{\mathrm{ff}}=0$。所有实际写入的权重均为 $1$，故完整内容等于当前 $u$ 加历史 value 的平均值。
- 各区域固定增益为 $1$，采用 Read$^+$、时间摘要、独立评分和二次正则约束分配；$k_j=1,\eta_j=0,\tau_{\mathrm{sel},j}=1/2$，正典容量 $K_j=2$。

第一层两节点的评分为

$$
s_{1,1}=\tfrac12,\qquad
s_{1,2}=\tfrac34-M_{v_{1,2},\mathrm{long}},
$$

第二层评分固定为 $(2,1)$。这些函数均可由拼接评分实现：常数用输出偏置，线性记忆项用
$\operatorname{SiLU}(x)-\operatorname{SiLU}(-x)=x$ 的两个隐藏单元。其余未使用的参数取零。

在位置 $0$，第一层两节点各收到 $(1,1)$，聚合为 $(1,1)$。Upd 各写入一条 $(\xi,k,\nu,p,w)=(1,0,1,1,1)$。新状态上的 Mem 为 $(1,1,1,1)$；Describe 分别返回 $(w,\mu,s^{\mathrm{loc}})=(1,1,1/2)$、$(1,1,-1/4)$。SelStep 给出 $c=(1,0)$。两节点都保留新缓存，只有 $v_{1,1}$ 执行 Full，读出 $1$，得到 $G=2$，向第二层的两个节点分别广播 $(1,2)$。

第二层两节点因此都聚合到 $(1,2)$。令 $a=f(2)=\sqrt{8/5}$，则两者写入的 value 与摘要均为 $a$，完整内容为 $2+a$。评分 $(2,1)$ 给出 $c=(1,0)$，终端读出 $O_0=2+a$。

位置 $1$ 继续使用各自缓存。下表中，同层两节点具有相同的记忆；$R$ 表示本例的标量 Attention 读出，控制族按节点横向坐标 $1,2$ 排列。

| 位置、层 | 聚合内容 $u$ | 新写入的 $\nu=p$ | $M_{\mathrm{long}}=R$ | 控制族 $c$ | 激活节点的完整内容 $G=u+R$ |
|---|---|---|---|---|---|
| $0,1$ | $1$ | $1$ | $1$ | $(1,0)$ | $2$ |
| $0,2$ | $2$ | $a$ | $a$ | $(1,0)$ | $2+a$ |
| $1,1$ | $-1$ | $-1$ | $0$ | $(1/4,3/4)$ | $-1$ |
| $1,2$ | $-1$ | $-1$ | $(a-1)/2$ | $(1,0)$ | $(a-3)/2$ |

位置 $1$ 的第一层评分为 $(1/2,3/4)$，两个节点均执行 Full，分别广播 $(1/4,-1)$ 与 $(3/4,-1)$。第二层每个节点收齐这两条消息，得到 $(r^{\mathrm{in}},u)=(1,-1)$；终端读出为 $O_1=(a-3)/2$。取 $W_{\mathrm{out}}=\alpha_{\mathrm{out}}=1$，最终表示按式 (2) 为 $y_t=h_t+O_t$。

第一层的第二个节点在位置 $0$ 未激活，仍保存了 $+1$。这条记忆使它在位置 $1$ 的长期摘要与 Attention 读出都为零，并参与当前评分；若漏掉该次写入，读出会变为 $-1$。本例的 $k_j=1$ 限制发送强度之和，位置 $1$ 的第一层仍有两个正控制量。

## 3. 每层的接收、写入与输出

### 3.1 传播条件与逐层归纳

固定第 1 节的图、单位入口和广播规则。各区域可分别采用区域教材的三种约束分配、直接 Attention、固定总强度 Attention、总量缩放 Attention、按比例分配或硬选择，并满足所选函数的参数条件。下面先证明这些组合共有的接收性质，再说明对状态和发送的推论。

在每个输入位置 $t$，所有层的全部节点都会收到正强度消息。先看使用节点增益描述的函数。第一层接收强度为 $1$，故 $w=\mu=1$。若某层全体节点有 $\mu=1$，则节点增益与固定阈值分配的发送强度之和在 $[1,k_j]$ 内，直接 Attention、固定总强度 Attention 和总量缩放 Attention 的总和均为 $1$，按比例分配的总和为 $\min(k_j,n)\ge1$，硬选择激活 $1$ 至 $k_j$ 个节点并各自产生单位强度载荷。每个下一层节点收到全部激活者的广播，接收强度至少为 $1$，于是仍有 $w=\mu=1$。逐层归纳即可到达终端。

两个区域控制候选使用相同的归纳：单位截断接收给出 $\mu^{\max}=1$、$b^{\mathrm{recv}}=1$、$U=k_j$，实际总量在 $[1,k_j]$ 内。下层接收仍至少为 $1$，所以它们也保持各节点的单位写入权重。

同一归纳覆盖各自的配对函数：指定总量的零温分配保持 $B\in[1,k_j]$，区域阈值的硬选择在单位上限下至少发送一个单位。各层可分别选用这些函数。

### 3.2 对状态与发送的推论

同层节点的接收强度相同，且每个节点每位置均有 $w=\mu=1$。该结论只用到接收强度求和与发送规则，适用于相容的 KV 或循环记忆。

对采用共同窗口 $H_\ell$、从空缓存开始的 KV 区域，各节点写入时间相同。在位置 $t$ 的接收更新之后，有限窗口的缓存长度均为 $\min(t+1,H_\ell)$，全历史缓存长度为 $t+1$。区域共享 Agg 的基线还具有相同的聚合内容；KV 仍可因节点的适配器和私有投影不同而分化。采用节点独立聚合时，聚合内容也可不同。激活选择不改变本层的写入次数。

在 §3.1 的全部组合中，即使采用上下文增益，接收强度仍至少为 $1$，$\mu=1$ 已由上界截断确定；增益网络不会改变本实例的前向结果。

采用约束分配或硬选择的层取 $k_j=1$ 时，该层每位置的发送强度之和恒为 $1$，共同阈值 $\eta_j$ 不影响选择。温度与评分差异决定约束分配的激活集合，硬选择在该层恰激活一个节点。直接 Attention 与固定总强度 Attention 在本实例中始终发送总强度 $1$。

## 4. 局部选择与配置实例

本节各例共用第 1 节的来源分类、区域共享 Agg、单位入口、读出、Read$^+$、Next 与广播。下表列出最终的记忆、评分和发送选择；各例分别给定尺寸与参数，无须逐例叠加配置。

| 实例 | 记忆与描述读取 | 局部／区域评分 | 发送与容量 |
|---|---|---|---|
| §4.1 稠密 Attention | 各区域 KV、sum | cat；选定区域 att、记忆与统计 key，其他区域 ind | 选定区域使用该节四种稠密函数之一，其他区域 constrained；$K_j=n$ |
| §4.2 八层 KV A、B | 各区域 KV、sum；窗口与旋转按 A 或 B | cat、ind | constrained、quad、$k_j=1,K_j=4$ |
| §4.3 八层混合记忆 | 区域 $4,8$ 使用循环记忆、state；其余区域使用 B 的 KV、sum | cat、ind | constrained、quad、$k_j=1,K_j=4$ |
| §4.4 区域控制 | 全部采用 B 的 KV，或在区域 $4,8$ 使用循环记忆、state | cat、ind；区域控制描述 | 指定总量或区域阈值，可选各自配对函数；$k_j=2$，容量按所选函数 |
| §4.5 内容 key | 同上一行的两种记忆布局 | cat；选定区域 att、内容 key，其余区域 ind；区域控制描述 | 指定总量或区域阈值，可选各自配对函数；$k_j=2$，容量按所选函数 |

§4.2 的窗口与位置表提供 KV 参数，§4.3 的表提供循环参数，§4.4 的控制表提供总量或阈值参数。复用其中一组参数时，其余选择由上表和当前实例直接确定。所有未取具体值的矩阵、偏置与尺度，在第 1 节及所选模块的范围内给定；参数形状按最终函数检查。

记忆、评分与发送还可组合成其他实例。内容 key 可用于任一相容的 att 评分；本节用区域控制组合展示其作用。全连接图的共同接收与传播结论见第 3 节，各发送规则的数值性质按第 5 节判断。

### 4.1 固定尺度的稠密 Attention 组合

在选定区域将独立评分替换为区域 Attention 评分，Describe 同时改为五元组，采用 region_key=content_stats，给定相应投影与 query 参数，取 $\alpha_j^{\mathrm{loc}}=0,\alpha_j^{\mathrm{sel}}=1$。发送选择直接 Attention、固定总强度 Attention、总量缩放 Attention 或 $k_j=1$ 的约束 Shannon，采用固定正温度与正典容量 $K_j=n$。各区域的记忆取 KV、sum，局部评分取 cat、增益取 fixed；未选区域使用第 1 节的 ind 与二次正则约束分配，其余参数按共同范围给定。

四种选择均由 §3 得到 $w_v=\mu_v=1$，所以在这些区域共同满足

$$
c_v=\frac{e^{s_v/\tau_{\mathrm{sel}}}}{\sum_{u\in\mathcal R_j}e^{s_u/\tau_{\mathrm{sel}}}}.
$$

这些组合具有相同的前向和评分导数；两种 query 权重也相同。固定评分而独立改变 $\mu$ 时，各函数的导数仍按自己的定义计算。按比例分配则直接均分。

固定参数、正温度与有限输入长度，上述四种等值组合关于内容序列 $(h_t)$ 连续：将选定区域统一表示为直接 Attention，其余区域仍用基线的二次正则约束分配，便可应用共同架构命题 2。这里的输入域是给定端口日程上的单位强度内容序列。

采用这组绑定的区域，其全部 $n$ 个节点执行 Full，发送强度是区域 query 对节点 keys 的 Attention 权重，总和为 $1$。各节点继续保存自己的 KV；广播之后，下一层每个节点收到该区域全部父节点的加权内容。本组合的训练与推理可以始终保持同一正温度。

进一步在其余区域采用其他已定义发送函数时，接收与写入仍满足 §3；连续性按第 5 节核对新的组合。Full 和消息工作按 §6.2 计数。

### 4.2 KV 窗口与位置的配置

本节所有区域使用 KV。每个区域内共用窗口与位置参数，区域间允许不同。需要确定的字段及其作用为：

| 配置项 | 区域 $\ell$ 的数学绑定 |
|---|---|
| 记忆范围 | $H_\ell$，各节点取 $T_v=DH_\ell$ |
| 位置坐标 | $o_v^{\mathrm{pos}}=\ell,s_v^{\mathrm{pos}}=D$，故 $\operatorname{Pos}_v(Dt+\ell)=t$ |
| Full 旋转范围 | 各头取偶数 $d^{\mathrm{rot}}_\ell\le d_h$；零、全部、部分分别对应 NoPE、完整和部分 RoPE |
| 旋转频率 | 对 $d^{\mathrm{rot}}_\ell>0$，可取 $\omega^{\mathrm{rot}}_{v,h,a}=b_\ell^{-2(a-1)/d^{\mathrm{rot}}_\ell}$，$b_\ell>1$，$1\le a\le d^{\mathrm{rot}}_\ell/2$ |
| Read 检索位置 | 采用内容检索时，独立声明其旋转维数和频率，仍使用相同 $\operatorname{Pos}_v$；时间摘要无需检索旋转 |
| 摘要衰减 | $\beta_{v,\mathrm{short}}=\bar\beta_{\ell,\mathrm{short}}/D$，$\beta_{v,\mathrm{long}}=0$ |
| 距离偏置 | 基线取 $b^{\mathrm{pos}}_{v,h}=0$ |

以下两份完整配置都使用 KV、时间摘要、cat 局部评分、ind 区域评分、fixed 增益与二次正则约束分配。共同取 $L=8$、$n=4$、$d=512$、$H_{\mathrm{att}}=8$、$d_h=64$、$k_\ell=1$，有旋转时取 $b_\ell=10^4$；其余参数按第 1 节的范围给定。

| 配置 | 区域 | $H_\ell$ | $d^{\mathrm{rot}}_\ell$ | 意义 |
|---|---|---|---|---|
| A：全历史 RoPE | $1,\ldots,8$ | $\infty$ | $64$ | 在全部接收历史上使用完整 RoPE |
| B：窗口 RoPE 与全历史 NoPE | $1,2,3,5,6,7$ | $512$ | $64$ | 近期 KV 与完整 RoPE |
| B：窗口 RoPE 与全历史 NoPE | $4,8$ | $\infty$ | $0$ | 全历史 KV 与内容匹配 |

将某区域的 $64$ 改为 $32$ 即得部分 RoPE 变式。全历史始终指该节点接收并保存的聚合历史；上游稀疏选择仍决定哪些信息进入该历史。配置 B 在区域间组合不同的记忆范围与位置函数。

各区域采用不同位置机制仍使用同一个载荷空间。本文的两个配置均使用 KV。循环与 KV 的异构组合见第 4.3 节。

### 4.3 KV 与循环记忆的组合

在选定区域替换记忆时，采用相应模块的状态空间、零初态、Upd、Mem、当前特征及内容函数 $G_v$；保留图、时间、来源和外部边界，描述宽度与区域评分保持相容。区域之间可以选择不同模块。

本例取 $L=8,n=4,d=512$。区域 $4,8$ 各选择下表的一种循环记忆，取 $d_s=32,d_f=2048$、memory_read=state；其余区域取 KV、sum、$H_{\mathrm{att}}=8,d_h=64$，窗口与旋转使用 §4.2 的 B 行。所有区域使用 cat、ind、fixed 增益和二次正则约束分配，$k_j=1,K_j=4$；其余参数按共同范围给定。

| 记忆 | 状态形状 | 时间与附加参数 |
|---|---|---|
| [affine／delta](../local-semantics/weighted-message/recurrent-memory/affine-delta-learning-note.md) | 两个 $d_m\times d_k$ 矩阵，$d_k=d_m=64$ | 矩阵和摘要的短期率为 $\bar\lambda_{\mathrm{short}}/D$，$\bar\lambda_{\mathrm{short}}>0$，长期率为零 |
| [gated_delta](../local-semantics/weighted-message/recurrent-memory/gated-delta-learning-note.md) | $4$ 头，$d_k=64,d_m=128$ | 矩阵被动率为零，qk_eps=$10^{-6}$，output_norm_eps=$10^{-5}$ |
| [ssd](../local-semantics/weighted-message/recurrent-memory/ssd-learning-note.md) | $4$ 头、$1$ 组，$d_k=64,d_m=128$ | 矩阵被动率为零，output_norm_eps=$10^{-5}$ |

Gated DeltaNet 与 SSD 的摘要短期率取 $\log2/D$、长期率为零，conv_size=4、conv_stride=$D$。在节点时间 $Dt+\ell$，卷积依次读取输入位置 $t,t-1,t-2,t-3$；每位置均有单位写入权重且从空缓存开始时，得到普通因果卷积。GDN 的保留门 $g_h$、误差写入门 $\beta_h$ 与 SSD 的有效步长 $d_h$ 依各自教材计算，不另乘逻辑时间间隔 $D$。循环区域使用自身的时钟与状态参数，KV 的窗口和 RoPE 字段只用于其余 KV 区域。

各矩阵、偏置、卷积系数、缩放和残差尺度按所选模块形状给定有限值。新的 Mem 与当前特征接入原评分构造，Describe 的参数形状按 $d_s=32$ 给定；消息宽度仍为 $512$。这些选择确定一族完整模型，可把两个替换区域选为同一模块，也可分别使用 Gated DeltaNet 与 SSD。

第 3 节的单位接收归纳只依赖发送规则，继续成立；各候选的 $w=\mu=1$。更换记忆会改变内容、评分与具体激活者。同层缓存长度的结论仅用于 KV；循环状态的大小不随接收次数增长。各循环矩阵与卷积都在全部正接收候选上更新。

联合求值使用 KV 前缀或对应循环模块的仿射复合；带短卷积的模块先按完整驱动计算卷积，再求矩阵递归。按层取得完整输入后，可应用第 6.1 节的整窗口策略。相同载荷、相容描述和原有图时间使组合成为合法 SettleGraph。数值性质按共同架构 §6 应用，各记忆分别提供局部引理。

### 4.4 指定总量与区域阈值

本例取 $L=8,n=4,d=512$。记忆布局选择全部 B 型 KV，或仅在区域 $4,8$ 使用循环记忆；KV 的 sum、窗口、位置与头尺寸见 §4.2，循环的 state 与参数见 §4.3。全部区域取 cat 局部评分、ind 区域评分，使用[区域分配 §2.3](../local-semantics/weighted-message/region-allocation-learning-note.md#23-节点评分与节点描述)的 $D^{\mathrm{reg,ind}}$，取 $d_r=d_{\mathrm{reg}}=32$、gamma_max=$2$、quad、$k_j=2$、$K_j=4$，各区域给定正温度。局部评分与区域头按该宽度给定全部有限参数，发送使用区域控制描述中的节点上限。

每个区域选择下列一行；context 使用同一行对应的独立区域头，固定控制返回零证据：

| 发送函数 | 固定控制 | 上下文控制 |
|---|---|---|
| regional_total | region_control_mode=fixed、total_fraction=$0.4$ | region_control_mode=context，给定节点证据构造与总量头参数 |
| regional_threshold | region_control_mode=fixed、eta=$0$ | region_control_mode=context，给定节点证据构造与阈值头参数 |

region_control_width=$32$。context 分支逐区域选择 region_evidence=cat／grouped，未另声明时取 cat。grouped 另取 $d_{\mathrm{sel}}=32$，给定区域教材 §3.2 的证据编码与 §3.4.1 的输出参数；局部评分继续使用 cat。fixed 不使用 region_evidence。两个发送候选分别绑定区域教材 §4.1.1、§4.1.2，保留全部候选的更新和共同广播。各行配合所选参数是一份完整模型，不增加区域持久状态。

由第 3 节，每次区域都有 $b^{\mathrm{recv}}=1,U=2$。指定总量的固定比例给出 $B=1.4$；上下文比例使 $B=1+\rho_j$。节点上限虽然全部为 $1$，区域头仍能改变发送总量和分配。阈值头在总量约束松弛时也能起作用。若把 $k_j$ 改回 $1$，两个头都失去调节总量的空间。

KV 写入时间与循环更新时钟继续满足 §3。指定总量按教材 §7.2 比较自身零温分配，所选节点可以发送小数强度；区域阈值的单位输入硬化按 §7.4 核对。显式求值时，分别将 selection_mode 改为 regional_total_zero 或 regional_threshold_hard，保留本节对应的描述、评分与区域头，移除温度和正则参数；容量分别取 $K_j=4$、$K_j=k_j=2$。

### 4.5 内容 key 与局部评分

本例取 $L=8,n=4,d=512$，记忆布局为全部 B 型 KV，或仅在区域 $4,8$ 使用循环记忆。KV 的 sum、窗口、位置与头尺寸取 §4.2 的 B 行，循环的 state 与参数取 §4.3。所有区域使用 cat 局部评分，隐藏宽度 $d_r=32$。

在选定区域取 region_score=att、region_key=content、alpha_loc=alpha_sel=$1$、query_weight=received，$d_r=d_{\mathrm{sel}}=32$。描述使用 $D^{\mathrm{reg,att}}$，给定[区域分配 §3.6](../local-semantics/weighted-message/region-allocation-learning-note.md#36-区域-attention-评分)的当前内容投影、独立内容 key 编码，以及该教材 §3.6.4 的区域 query 参数。其余区域取 ind，描述使用 $D^{\mathrm{reg,ind}}$。

每个区域的发送独立选择指定总量或区域阈值，控制参数使用 §4.4 的控制表：gamma_max=$2$、$d_{\mathrm{reg}}=32$、$k_j=2$，fixed 使用表中的固定值；context 逐区域选择 cat／grouped 证据，并给定证据及头参数，未另声明时取 cat。正温度分配取 quad、$K_j=4$；显式使用配对函数时，指定总量零温分配仍取 $K_j=4$，区域阈值硬选择取 $K_j=2$。各函数的温度参数按其定义给定。

这组绑定让内容 key 与共同 query 匹配，接收与写入统计仍可经局部评分影响分配；统计对记忆与区域控制的作用由所选函数决定。给定全部有限参数后得到完整模型，按第 5 节核对组合及新运行的评分条件。

## 5. 组合成立的条件

图与时间满足拓扑教材的单次结算条件。所有相连节点共用 $P=\mathbb R^{1+d}$，各记忆提供相容的描述，区域分配返回控制与激活集合；因此所选函数可以代入正典递归。除硬选择取 $K_j=k_j$ 外，各发送函数均取 $K_j=n$。数值性质分别使用下表的条件。

| 对象或性质 | 本实例的绑定与依据 |
|---|---|
| 输出比较 | §1.4 声明有限槽集合和连续数值读出，缺席组读出零 |
| 正温度连续性 | 各区域使用约束分配、直接 Attention、总量缩放 Attention 或按比例分配时，按共同架构 §6.2 的条件使用，各记忆模块验证其状态假设 |
| 非空传播 | §3 保证每层全体节点接收并写入，每位置有正强度终端输出 |
| 温度极限 | 指定总量按[区域分配 §7.2](../local-semantics/weighted-message/region-allocation-learning-note.md#72-指定总量的零温分配)，区域阈值按[§7.4](../local-semantics/weighted-message/region-allocation-learning-note.md#74-区域阈值与单位强度硬选择)，节点增益与固定阈值按[§7.5](../local-semantics/weighted-message/region-allocation-learning-note.md#75-节点增益与固定阈值的硬化)选定比较运行并核对唯一性 |
| 正温度精确性 | 满足所选比较的条件且全部区域取 quad 或固定 $\alpha>1$ 的 ent 时，充分小正温度使完整记录和切面与该极限运行相同 |

给定上述配置即可直接应用共同的 P/S/U/F 递归。硬化所需的评分间隔取决于参数与输入，须在所考察硬运行上核对。拓扑允许相同局部函数在不同尺寸下组合，各性质按自己的前提使用。

区域控制的比较保留同一 Describe 与控制头。混合不同发送候选时，按共同架构逐区域核对所选极限函数；指定总量的零温函数具有自己的控制幅度和容量约定。

## 6. 整窗口求值与架构成本

### 6.1 从局部契约到整段求值

给定目标窗口的完整驱动纤维，KV 教材 §6 的前缀与读取构造、各循环教材 §4 的卷积准备及仿射复合分别提供精确的 BatchRegionState 与逐节点 BatchFull。基础 affine／delta 无卷积步骤，每个区域使用自己所选模块的契约。

取 $0\le p\le q\le N$，输入位置区间 $[p,q)$ 对应逻辑时间窗口 $I=[Dp,Dq)$。给定其完整端口输入及左切面，第一层的窗口纤维已确定；其状态块给出全部快照和选择，逐节点 Full 批给出发往第二层的全部消息。依次沿区域顺序处理，就能为下一层提供完整窗口纤维。局部写入不依赖本节点选择或 Full，区域内部也没有消息边，因此可以直接应用 [分块教材 §6](../timed-dag-chunk-prefill-learning-note.md#6-严格分层区域的整块定理)。

所得策略每区域至多使用一个状态块，每节点至多使用一个额外 Full 时间批。这个结论计量外层调用；块内的写入、描述与完整内容计算仍随窗口长度增长。KV 的 query 和 key 分别使用原始处理时间与写入时间，循环状态使用原始逻辑时钟；各自的联合算法保持这些时间与快照。

从输入位置 $q$ 继续时，保留 $Q_{Dq}$ 及原有时间编码，下一位置仍为 $q$。任意连续分块与逐位置求值均能恢复同一记录与边界状态；全历史 KV 区域保留完整接收历史，有限窗口 KV 保留可恢复的窗口状态；循环区域保存矩阵、摘要、时间戳及适用的卷积缓存。位于位置内部的一般切面同样可以继续，边界须包含跨界消息。计算表示与解码分别见所选记忆教材。

### 6.2 共享计算与宽度成本

相邻层全连接与广播 Full 使同一区域的节点面对相同上游载荷；§1.3 的来源绑定使其类别也一致。在区域共享 Agg 的 KV 基线中，区域聚合与共享 KV 基座投影可各计算一次，各节点仍分别保存 KV，计算适配器、记忆描述、评分及被选择的 Full。循环区域可复用相同的聚合，其状态投影与更新按循环模块计算。采用节点独立或不同的来源适配时，须重新判断哪些中间量相同。

若全图各区域均采用硬选择，则每个输入位置满足

$$
\begin{aligned}
L\le\#\mathrm{Full}&\le\sum_{j=1}^{L}k_j,\\
\#\mathrm{candidate}&=Ln,\\
n(L-1)\le\#\mathrm{internal\ message}&\le n\sum_{j=1}^{L-1}k_j.
\end{aligned}
\tag{3}
$$

§3 保证每层都有正强度发送，下一层全部 $n$ 个节点始终成为候选并写入。区域宽度增加时，准备、状态与消息成本仍可能增长。工作分类见共同架构附录 S.2，具体状态与算术成本按所选记忆模块计算；KV 窗口按 §1.2 换算，循环状态大小由矩阵宽度确定。

若全部区域采用固定总强度 Attention、直接 Attention、总量缩放 Attention、按比例分配、bern 或 ent 的 $\alpha=1$，正温度下每位置有 $Ln$ 次 Full 和 $(L-1)n^2$ 条内部消息。可稀疏的约束分配按实际激活数量计数；全部区域使用硬选择时满足式 (3)。两种局部评分及内容检索的准备成本另计，不能由发送强度之和推算。

整段求值使用各模块的精确联合算法；逐位置继续仍承担候选准备、记忆读取和消息工作。本实例减少激活时可减少 Full 和消息条数，各层候选与写入次数保持不变。

## 附录 S：配置对应

基线使用 memory_family=kv、memory_read=sum、read_mode=+、local_score=cat、region_score=ind、send_gain=fixed、selection_mode=constrained、regularizer=quad。所有节点的初态、来源与矩阵形状由第 1 节确定；位置绑定对应 KV 教材的 Pos、窗口与旋转配置。

第 4.1 节在选定区域使用 region_score=att、region_key=content_stats、alpha_loc=0、alpha_sel=1 和固定 allocation_temperature。四种等值发送分别取 direct_attention、unit_attention、pooled_attention，以及 constrained／ent／ent_alpha=1／k=1。第 4.2 节的 A、B 继承基线的评分与发送；第 4.3 节继承 B，再替换区域 $4,8$ 的 memory_family、memory_read 及对应参数形状。组合采用第 4.1 节时另声明相应区域。完整取值与生效范围统一见[区域分配附录 S.1](../local-semantics/weighted-message/region-allocation-learning-note.md#s1-配置名称)。

第 4.4 节使用 regional_threshold／regional_total 及其配套描述，region_control_mode 选择 fixed／context，region_control_width=$32$、gamma_max=$2$、$k_j=2$。固定分支分别读取 eta 或 total_fraction，context 使用区域头并按 region_evidence 选择 cat／grouped 证据；send_gain 与 allocation_base 不参与。参数形状和无效字段按区域分配附录 S.1 核对。

第 4.5 节在继承的区域控制配置上采用 content key 与非零局部评分，完整替换字段由该节给出；其余 att 配置未声明时使用 content_stats。

实现核对见[局部架构实现对应](../reference/weighted-message-implementation-checks.md)，任务与训练比较见[可选研究问题](../memos/training/weighted-message-research-questions.md)。
