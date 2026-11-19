---
type: mathematical-instance-family
status: active
tags: [tide, backbone, capacity, parameter-sharing]
---

# 主干比较绑定与尺寸阶梯

本页将 FC、HB-Lattice、HB-Exchange、HB-Multiscale 接到统一尺寸和参数归属，供下游选择实例。目标是用整套候选覆盖小规模至约 256M 的主干、不同普通区预算及更大的容量；不要求每个小实例都有约 3% 的预算比例，也不为对准参数数目而使用特殊的小维度。这里规定模型与计数，不执行训练。

不同候选覆盖不同需求组合。小总参、大维度、多节点，以及更多交换与更强局部性之间的取舍，连同需求如何推动新拓扑、Hub 和时间设计，保存在[主干候选的形成与取舍](backbone-design.md)。后续可据此新增构造或绑定，已有尺寸并非唯一扩展路线。

## 1. 空间图、时间与局部函数

| 空间图 | 端点定义 | 时间绑定 |
|---|---|---|
| FC | [全连接分层](../topologies/fully-connected-layered.md)，每层一区 | Settle，原周期 $D=M+2$ |
| HB-Lattice | [二叉展开收拢](../topologies/hb-lattice.md)，基础区域关系、偏移 $\{0,1\}$ | Settle，原周期 $D=M+2$ |
| HB-Exchange | [位交换图](../topologies/hb-exchange.md)，展开深度与交换轮数可分开 | Settle，$D=M+1$ |
| HB-Multiscale | [多跨度宽图与窄图](../topologies/hb-multiscale.md) | 单位边时延的 HB-Timed、跨度时延的 HB-Multiscale-Settle |

$M$ 为包含首尾的层数。Settle 保持当前正典的 $D>M$；Timed 主周期 $T=4$。同一空间图改变时延不改变参数数目，但改变聚合、状态写入和分配。LH-DAG／PDG 保留为一般语义探索，不进入本页规模表。

HB 继承[多跨度主绑定](hb-multiscale-weighted-message.md)与 [Settle 共同绑定](hb-settle-weighted-message.md)的全部函数和初态，再按本页替换尺寸、预算和参数归属。主表使用 group 分区；可选 hub 为非末层每位置 4 个、独立 H 区，末层只有 D。无 hub 时保留全部普通边。

**FC 的统一绑定。** 复用上述主绑定的载荷、KV／SSD 状态与零初态、Agg、Read$^+$、encoded／self_attention 评分、Next 和 Full；全图只有 D 角色，来源类为 ext、DD。每个首层节点有入口 $(1,E_Dx_t)$，末层全部节点有输出端口。时间、Pos、窗口、衰减与卷积步长按 Settle 规则，$P_{\mathrm{in}}=D=M+2$。参数按[三档整体绑定](parameter-profiles.md)的节点／区域／层级归属，区域为 D 类型。这是一份新的完整比较绑定；[旧 FC 教学实例](fully-connected-layered-weighted-message.md)的混合参数归属保留，不套用本页计数。

三组记忆仍为 K（全部 KV/sum）、S（全部 SSD/state）、X（D 用 KV，H 用 SSD）；无 hub 时 X 与 K 重合，表中去重。接收机制由独立的 agg_mode 选择，agg_profile 进一步绑定尺寸规则：来源残差、两种轻量自注意力及加宽自注意力，完整替换见 [HB §1.1](hb-multiscale-weighted-message.md#11-接收自注意力的完整替换)。主表分别列出四个配置并计入各自全部参数，计数见本页 §5.1。FC 同样可以替换，来源类仍为 ext、DD。

## 2. 合理维度与参数归属

全图使用相同内容宽度 $d$，不按深度或节点数自动缩小：

$$
d\in\{64,128,192,256,384,512,768\},\qquad
d_s=d/4,\qquad d_z=d/2,\qquad d_f=4d.
$$

接收尺寸按 agg_profile：来源残差及两种轻量自注意力取 $p_a=d/8$；加宽自注意力取 $d_A=p_a=d$。轻量自注意力的 $d_A$ 固定为 64 或 128。所有自注意力每头宽度 32，头数为 $d_A/32$。这些尺寸使用既有局部函数，无需新增语义；来源 MLP 的 $p_a$ 与 Full 的 $d_f$ 是不同阶段的宽度。

描述与区域评分统一取 region_profile=encoded_self_attention，按 [HB §1.2](hb-multiscale-weighted-message.md#12-共同编码与区域候选关系)绑定：$d_z=d/2$、$H_R=d/64$、每头 32 维、区域 FFN 宽度 $d_{\mathrm{rf}}=d$。区域块一层，主约束分配不启用总量头。这里扩大的是候选比较的表示，与接收 attention 的 $d_A$ 分别设置。

KV 每头宽度 32，头数 $d/32$，旋转频率及零距离偏置沿用主绑定。SSD 内部宽度 $d$，$d/32$ 个头，每头内容与状态宽度均 32，1 组，conv_size=4。其余固定增益、温度、窗口、时钟换算、残差尺度与归一化 epsilon 沿用主绑定；SSD 的可训练 norm 尺度仍计参数。

参数采用 independent、region_shared、layer_shared 三档整组归属，HB 优先区域共享。两个共享档默认适配秩 $r_{\mathrm{ad}}=8$，同族可选 4、16；矩阵实际秩取 $\min(r_{\mathrm{ad}},a,b)$。它限制节点私有修正，不限制完整矩阵的秩或消息宽度；状态始终私有。共享说明、KV 基座例外及训练梯度归约均以参数绑定页为准。FC 每层一区，相同适配秩下两个共享档等价，实验无需重复；单节点组仍保留基座加适配器，三档参数量不保证严格有序。

## 3. 二次幂预算与稠密对照

每区普通节点数取

$$
n\in\{8,16,32,64,128\},\qquad
k_D\in\{1,2,4,8,16,32,64\}\cap[1,n],
$$

分配函数与数值预算分别选定：主候选为 `quad`、`shannon`、`pool`；前两者只从上述数值选 k，不按目标百分比反推其他整数，pool 不使用 k。默认 $k_D=\max(1,n/32)$ 在上述尺寸中总为合法整数；它只是已有主绑定的默认档，不限制其他档。

| $n$ | 数值预算 | 最小 $k_D/n$ | 接近常用比例的自然档 |
|---:|---|---:|---|
| 8 | 1、2、4、8 | 12.5% | 12.5%、25%、50%，另有 pool |
| 16 | 1、2、4、8、16 | 6.25% | 6.25%、12.5%、25%、50%，另有 pool |
| 32 | 1、2、4、8、16、32 | 3.125% | 3.125%、6.25%、12.5%、25%、50%，另有 pool |
| 64 | 1、2、4、8、16、32、64 | 1.5625% | 含 3.125%、12.5%、25%、50%，另有 pool |
| 128 | 1、2、4、8、16、32、64 | 0.78125% | 含 3.125%、12.5%、25%、50%，另有 pool |

| allocation_profile | D 区完整发送函数 | 总量与边界 | 活跃代理的 D 比例 |
|---|---|---|---|
| quad，默认 | role_bounded_quad | 普通边界 $(0,\mu)$，interval、lower_base=received、固定 $\eta=0$ | $k_D/n$ |
| shannon | role_bounded_shannon | 与 quad 相同的边界、总量区间和阈值 | 1 |
| pool | [pooled_attention](../local-semantics/weighted-message/region-allocation-learning-note.md#423-总量缩放-attention) | $B=\min(1,\sum\mu)$、先验 $a=\mu$，不使用 k、eta、regularizer | 1 |

三者均使用同一 encoded 描述、self_attention 评分、固定增益 2 和温度 1，$K_j=n$。约束分配有 $b=\min(1,\sum w)$、$U=\min(k,\sum\mu)$、$L=b$；普通边界下 $b\le U$。pool 使用区域教材 §3.7.4 的节点增益编码描述，不附加角色界，也不保证逐节点 $c\le\mu$。Shannon 与 pool 的所有正接收普通候选均正发送，不强制不存在消息的节点作用。H 区在三种 D 分配下均保留 received 下界的 role_bounded_quad、$k_H=2$ 及边界 $(\mu_h/4,\mu_h)$。这三个分配对照不增加参数。

$k_D=n$ 的 quad、约束 Shannon 与 pool 仍是不同函数。quad 可能稀疏；Shannon 保留与 quad 相同的可行集，正则还可能改变总强度；pool 同时改变总强度及逐节点边界。它们不构成只改变支持大小的实验。主表不使用 mixed；另做 mixed 对照时须按角色分配核对合并预算。

## 4. 拓扑尺寸与窄图

| 空间图 | 已登记的层数／轮廓 |
|---|---|
| FC | 4、6、8、9、10、12、16、24、32 层；$S=M$ |
| Lattice | 3、5、7、9、11、13 层；$S=3\cdot2^{(M-1)/2}-2$ |
| Exchange 原阶梯 | $m=1,\ldots,6$，$R=m$，4、7、10、13、16、19 层 |
| Exchange 窄平台 | $(m,R)=(1,3),(1,4),(1,5),(1,10),(2,5),(2,9)$ |
| Multiscale 原宽图 | 5、5-dd4、7、9、13、15、17 层；25／33 层延长中央平台 |
| Multiscale 窄图 | compact-5-dd4、compact-7、compact-9、compact-9-two-inputs、compact-15 |

$S=\sum_\ell W_\ell$，普通节点数 $N_D=nS$；全位置可选 4 hub 时，$N_H=4(S-W_{M-1})$。原宽图与大区域尺寸均保留。25／33 层宽图从 17 层序列的索引 8 前插入 8／16 个宽度 16 的层，置换按新层号重算。更大模型可以超过 256M，表中标记预算范围而不自动缩小 d。

窄图的完整宽度与连线见拓扑定义。层宽缩到 1／2 不破坏严格区域分层；相位覆盖依赖时延分拆，不要求每层至少 4 个位置。固定 $T=4$、单位边时延时，基础普通子图从 7 层覆盖全部相位，15 层起每相位至少两种时延。5-dd4 需要新增普通 DD4 边。基础 5 层不作为完整四相位覆盖点。

以下为来源残差接收的部分 independent 点，普通 K、hub 为 S；参数单位 M=$10^6$：

| 空间候选 | $n,h$ | 总节点 | $d=64$ | $d=128$ |
|---|---|---:|---:|---:|
| 窄九层 | 8,0 | 128 | 9.33 | 37.27 |
| 窄九层 | 16,0 | 256 | 18.53 | 74.02 |
| 窄九层 | 32,0 | 512 | 36.92 | 147.51 |
| 窄九层 | 32,4 | 572 | 41.47 | 165.08 |
| 双入口位置窄九层 | 32,4 | 712 | 51.61 | 205.47 |
| 窄十五层 | 32,4 | 788 | 57.14 | 227.43 |
| Exchange $m=1,R=5$，8 层 | 32,4 | 500 | 36.25 | 144.29 |
| Exchange $m=1,R=10$，13 层 | 32,4 | 860 | 62.35 | 248.20 |
| 原宽九层 | 32,4 | 1712 | 124.12 | 494.08 |
| 原宽十七层 | 32,4 | 5456 | 395.67 | 1574.75 |

这些不是训练效果排名。窄图减少并行容量与路径冗余；重复合作、摘要钩子是否有用以及扩大后如何分工，继续作为[设计问题](backbone-design.md)保留。

## 5. 主干参数与活跃代理

总参数计 Agg、实际来源嵌入、记忆投影、描述／评分、区域头、Full／FFN，以及 SSD 的可训练向量和卷积抽头。仅计启用函数；固定增益、预算、阈值、时钟、初态不计。输入嵌入、入口投影、任务头、Pronounce、优化器与 KV／SSD 运行状态不计。

在 source_residual 下，独立 K 单节点不含来源嵌入与区域头的参数量为

$$P_K(d)=17.5d^2+2.5d+1.$$

两个共享档分别按实际区域／层级中的唯一基座及节点私有适配器计数，不把重复使用的权重多计。KV 查询的 $W^Q_{v,h}$ 逐数学头分别适配；把多个头打包后改用一份低秩适配器会改变绑定，不是实现上的无差别优化。KV 独立配置的零基座和单位因子固定，不增加冗余可训练参数。

令 $B$ 为按所选区域／层级范围去重的共享基座、共享向量与区域头，$P_D,P_H$ 为 D/H 私有参数总数，则

$$
P_{\mathrm{total}}=B+P_D+P_H,\qquad
P_{\mathrm{active,proxy}}=B+\frac{k_D}{n}P_D+P_H.
$$

上式的活跃代理用于 quad；Shannon 与 pool 用比例 1。independent 中 $B$ 只有区域网络。估算假定各区均有正接收、H 全部发送；来源嵌入按普通参数总数的相同比例估计。quad 的 k 不是硬支持上限；降低 k 不减总参数。准备阶段仍可能访问未选节点的参数，因此该代理不是实际触及参数、FLOPs 或每输出 token 工作量，Timed 也可能每输入产生多次作用。保持这一简明口径，真实活跃与系统性能由下游测量。

计数中的节点评分已替换为共同编码与读出，没有叠加原 cat／key 参数；区域网络每份 $P_R=2d^2+2d$，归属见[参数绑定](parameter-profiles.md)。其工作发生在选择前，不能随 Full 的 $k_D/n$ 折减。

### 5.1 接收自注意力的参数与准备成本

两种轻量 incoming_self_attention 保持原来源函数的 $p_a=d/8$，增加四个投影和每个来源类的 $d$ 维嵌入，逐节点／共享组的增量见[三档参数绑定 §4](parameter-profiles.md#4-规模与通信含义)。全节点使用同一 $d_A$ 时，独立配置的增量为

$$
\Delta P=4Ndd_A+d\sum_v|\mathsf L_v|.
$$

这里 $N=N_D+N_H$。例如 $d=128,d_A=64$ 时，每节点四个投影增加 $0.032768$M，另计新来源嵌入；$d_A=128$ 时投影增加 $0.065536$M。两个轻量配置保持 $p_a=16$；加宽配置则取 $d_A=p_a=128$。

加宽配置还将来源 MLP 从 $p_0=d/8$ 改为 $p_a=d$。独立配置相对来源残差的完整增量为

$$
\Delta P_{\mathrm{wide}}
=N\bigl(4d^2+2d(d-p_0)\bigr)
 +(2d-p_0)\sum_v|\mathsf L_v|.
$$

其中 $2d(d-p_0)$ 来自两个 MLP 矩阵，来源类项包括原嵌入的加宽与新增 attention 嵌入。两个共享档也同步扩大基座、来源嵌入和适配器；具体形状见参数绑定页。加宽 K 单节点不含来源嵌入与区域头的参数量为 $23.25d^2+2.5d+1$。例如 $d=128$ 时，相比来源残差每节点增加 $0.094208$M 矩阵参数，每个实际来源类另增 240 个参数。

[公共计数源](../reference/backbones/capacity.py)的 count 接收 agg_profile，直接返回所选配置完整的 total／base／D／H 及节点数；CSV 和面板也使用这一结果，无需另加增量。保留的 incoming_attention_increment 只计算注意力新增的四个投影与来源嵌入，适用于来源 MLP 宽度不变的比较；不能用它代替加宽配置的完整计数。§5 的活跃代理沿用同一比例口径。

这些参数在候选准备中使用。实际工作须按每次正消息数 $n_+(v,\theta)$ 计四投影、$n_+^2$ 成对读取和逐条来源变换；主要算术项分别为 $O(n_+dd_A)$、$O(n_+^2d_A)$、$O(n_+dp_a)$。Full 预算比例不能说明省去了多少接收工作。加宽增加的参数量也不能直接当作额外时延。记录实际到达数分布、候选作用次数、准备与 Full 时间，才能评估 Hub 扇入及异时输出的物理成本。

## 6. 交互与复算

[交互面板](../explorers/backbone-atlas/README.md)提供拓扑、数学公式和规模筛选；[完整参数 CSV](../explorers/backbone-atlas/generated/capacity.csv)保留全部尺寸、独立／共享、K/S/X 和二次幂预算的估算，[生成范围表](../explorers/backbone-atlas/generated/capacity-summary.md)便于逐图阅读。CSV 和页面的参数单位均为百万参数，$1\mathrm{M}=10^6$；内部计数与范围判断保留精确值。所有行由[计数源](../reference/backbones/capacity.py)重建。

profile 绑定空间轮廓与时间，例如 `multiscale-compact-9-timed` 和 `multiscale-compact-9-settle` 分别列行，二者的 `spatial_profile` 均为 `multiscale-compact-9`。同一接收配置下它们参数量相同，消息汇合、状态写入和作用次数可以不同。当前 41 个空间轮廓展开为 55 个含时间绑定的轮廓，与合法尺寸、记忆、参数归属和四个接收配置组合后有 225,400 份参数配置；CSV 再按三个 allocation_profile 分行为 676,200 行，预算阶梯仍是列。页面去重保存参数配置，分配选择独立控制预算及活跃代理。约束行尚未选定 k，各行也不代表已经规定了下游训练与读出。

| CSV 列 | 含义 |
|---|---|
| `region_profile`、`local_score`、`region_score` | 当前共同编码与候选关系绑定 encoded_self_attention，以及 encoded／self_attention 函数 |
| `descriptor_width`、`region_attention_heads`、`region_attention_head_dim`、`region_ffn_width` | 描述宽度 $d_z$、区域头数、每头宽度和区域 FFN 隐藏宽度 |
| `allocation_profile`、`selection_mode_D`、`selection_mode_H` | 三种分配对照标识及 D/H 使用的实际函数；无 H 时 H 字段留空 |
| `lower_base_D`、`lower_base_H`、`total_mode_D`、`total_mode_H` | 约束分配显式使用 received／interval；pool 的 D 字段和无 H 的 H 字段留空 |
| `agg_profile`、`agg_mode` | 接收尺寸配置标识、接收函数；四个配置见 HB §1.1，每行已计入其完整参数 |
| `agg_hidden_width` | 来源 MLP 的隐藏宽度 $p_a$ |
| `agg_attention_width`、`agg_attention_heads`、`agg_attention_head_dim` | 接收注意力的总宽度 $d_A$、头数 $H_A$、每头宽度 $d_h$；来源残差的三列均为 0，表示未启用注意力 |
| `profile`、`spatial_profile` | 含时间绑定的轮廓标识、共用的空间轮廓标识；完整参数配置还需结合 n/h/d、记忆、共享档、秩及 agg_profile |
| `family`、`timing` | 空间图族 fc／lattice／exchange／multiscale；时间绑定 timed／settle |
| `depth`、`widths` | 包含首尾的层数；各层位置数，以分号分隔 |
| `positions`、`max_width` | 全图位置总数、最宽层的位置数；位置内可有 D 区及 H 区 |
| `expansion`、`rounds` | Exchange 的展开深度 m、宽段交换轮数 R；其他图留空 |
| `input_period` | 当前主周期：Timed 为 4；Settle 按图族采用 M+1 或 M+2 |
| `n`、`h`、`d` | 每个 D 区节点数、每个非末层位置的 H 节点数、全图内容维度；当前表使用 group 分区和全位置／无 hub |
| `memory`、`sharing`、`rank` | K／S／X、三档参数归属、共享适配秩；independent 的 rank 为 0 |
| `nodes_D`、`nodes_H` | 全图两类节点总数 |
| `total_M`、`base_M` | $P_{\mathrm{total}}/10^6$、$B/10^6$；base 的组成依参数归属，见 §5 |
| `private_D_M`、`private_H_M` | $P_D/10^6$、$P_H/10^6$；两类节点私有参数总量 |
| `within_256M` | 精确总参数是否不超过 256,000,000 |
| `active_k1_M` 至 `active_k64_M` | §5 的活跃参数代理；quad 使用 k/n，Shannon 为 total_M；k>n 或 pool 行留空 |
| `active_pool_M` | 仅 pool 行有值，等于 total_M；约束分配行留空 |

CSV 使用各候选主周期。面板另保留 Timed 与 Settle 同周期的对照设置，显示对应输入周期并在图与规模表之间双向传递；它属于 Timed 的另一输入日程。修改日程后的完整实验仍须按实例声明位置函数、衰减、窗口和卷积抽头。生成数据中的整数计数供计算使用，CSV 的 _M 列保存纯数字，便于直接筛选和统计。

对应核对为[紧凑图的相位与合作](../examples/hb_compact_checks.py)、[参数形状与预算](../examples/backbone_capacity_checks.py)、[接收自注意力](../examples/incoming_attention_checks.py)、[区域自注意力](../examples/regional_attention_checks.py)与[角色分配](../examples/role_bounded_allocation_checks.py)。面板可单选或并列比较四个接收配置，另选三种分配，并显示接收及区域尺寸；规模限制按完整总参筛选。上述证据不包含整网训练或吞吐结果。真正实现后还应按唯一 Parameter 对象核对主干总参。
