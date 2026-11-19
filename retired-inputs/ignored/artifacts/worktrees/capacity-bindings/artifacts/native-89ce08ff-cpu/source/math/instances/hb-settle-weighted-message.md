---
type: mathematical-instance
status: active
tags: [tide, hb, settlegraph, weighted-message]
---

# HB-Settle 三种空间图的共同主绑定

本页将三个 Settle 空间候选接到同一组局部函数，用于后续实现和比较。原有 [HB-Lattice 教学实例](hb-lattice-weighted-message.md)仍按原定义运行；本页是另一份明确采用新主绑定的实例，不覆盖旧例子。

为什么保留三种空间图、同端点时间对照与各自原生图分别比较什么，见[主干记录 §5](backbone-design.md#5-settle-的交换与三档参数绑定)。

## 1. 空间候选与时间

| 名称 | 全部消息边与分区的来源 | 数值起点 | 时间 |
|---|---|---|---|
| Lattice | [原 HB 拓扑](../topologies/hb-lattice.md)，基础递归关系、偏移 $\{0,1\}$ | $b=2,h=3,n=32$；7 层，22 区，704 节点 | $D=L+2=9$，保留原周期 |
| Multiscale-Settle | [多跨度图](../topologies/hb-multiscale.md)原样复用端点，group 分区 | 9 层，$n=32,h=4$，非末层全位置 hub；92 区，1712 节点 | 跨度时延，$D=M+1=10$ |
| Exchange-Settle，独立 Settle 设计的优先候选 | [位交换图](../topologies/hb-exchange.md) | $m=3,n=32,h=0$；10 层，46 区，1472 节点 | 相邻层时延 1，$D=M+1=11$ |

Exchange 可在非末层每位置启用 4 个 hub，分为 D 区与 H 区，成为 91 区、1652 节点的增强对照；末层只有 D。原 7／10／13 层阶梯保留，另有独立选择平台轮数的窄图；$n=8/16/32/64/128$、二次幂预算与内容维度由[比较绑定](backbone-comparison.md)统一给出，不增加局部函数种类。多跨度图的其他 hub 掩码与分区继续按其原定义；第一轮无需全部展开。

三个候选均取 $r_v=\ell+1$（此处层号从零起），$\delta(e)=r_{\mathrm{dst}}-r_{\mathrm{src}}$，输入为 $Dt+1$，保留当前 $D>\max r_v$。$M$ 是包含首尾的层数，旧图的 $L=M$；$D$ 是本页输入周期，$T$ 是 Timed 讨论中输入周期的名称。它们不代表三种 Settle 语义。原图一基层号与面板零基层号只差地址记法。

## 2. 完整的局部绑定

继承[HB 主实例](hb-multiscale-weighted-message.md)的载荷、尺寸、初态、Agg 函数形式、Upd、Mem、$p_v,G_v$、Read$^+$、encoded／self_attention 评分、received 下界的 role_bounded_quad、Next 和广播。D 区可按统一比较绑定替换为角色 Shannon 或 pool，H 区保持角色 quad。记忆主候选仍至多三个：K 全 KV、S 全 SSD、X 普通 KV／hub SSD；无 hub 时 X 与 K 相同。Exchange 无 hub 默认 K，含 hub 默认 X。参数优先采用[区域共享配置](parameter-profiles.md) region_shared，另保留 independent 与 layer_shared 两档对照；Agg 及其他部件均按同一整体配置绑定。

具体继承及替换如下，未列出的有限矩阵形状、固定尺度和温度等逐项沿用 HB 主实例 §1—3。

| 对象 | 本页绑定 |
|---|---|
| Agg | 默认 source_residual；可按 [HB §1.1](hb-multiscale-weighted-message.md#11-接收自注意力的完整替换)完整替换为 incoming_self_attention，使用本表来源类及相同三档参数归属 |
| 来源类别 | Lattice：ext、DDL；Exchange：ext、DDX，以及启用时的 DH1、HH1、HD1；Multiscale：沿用原边族 |
| 普通区 | $k_D=\max(1,\lceil n/32\rceil)$，$K=n$，普通边界 $(0,\mu_v)$ |
| 增强图 H 区 | $h=4,k_H=2,K=4$，hub 边界 $(\mu_h/4,\mu_h)$；共同角色分配 |
| 区域历史 | 单点；全部候选保留 Upd；零控制不求 Full |
| 时间 | $P_{\mathrm{in}}=D$；$\operatorname{Pos}_v(\theta)=(\theta-r_v)/D$，KV 窗口 $128D$，短期率 $\log2/(8D)$，SSD conv_stride=$D$ |
| 输入 | 首层全部现存节点各有入口，$(1,E_{\mathrm{role}(v)}x_t)$；无 hub 只需 $E_D$ |
| 输出 | 末层 D 的逐端口原始记录 $Z$；需要单槽时使用共同 ReadOut，全部缺席返回零 |

本页 Lattice 按边族分类，与旧教学实例按父节点身份分类不同；这是为统一比较而明确选择的局部绑定。输入投影、任务头和读出规则在各对照中固定或明确说明变化。Pronounce 的具体语言任务组织留给下游。

## 3. 求值、传播与比较范围

三图均严格区域分层。同源路径对齐，已知输入窗口时可按区域拓扑序取得完整上游驱动，再应用 KV／SSD 的精确块见证。窗口边界仍保存节点状态；输入位置之间的完整间隔切面没有在途消息。改变时间日程不要求设备按数值间隔空转。

对表中普通节点独立成区的三个主配置，每个源 D 都连接到每个基础后继位置中的至少一个 D；正接收的普通区由共同分配下限产生正发送，故沿基础展开收拢关系，每个输入位置都有普通终端输出。这个结论不保证每个终端节点激活、每条交换边发送或所有来源内容保留。Hub 增强保持这条独立普通通路。

空间候选比较用于研究对齐计算中的交换组织：原 Lattice 通过展开传播共同背景、收拢汇合分支；位交换让不同分支的新中间结果更早相遇；多跨度 Settle 保留跨层捷径，适合与 Timed 做端点一致的比较。三图默认规模不同，表中的配置不直接构成同资源实验。

机制对照固定同一多跨度空间图、参数配置和局部函数，分别训练 Timed 与 Settle 版本。只改时延时，两侧共同取 $D>M$ 的输入周期，并固定相同输入日程、Pos、窗口、衰减和卷积抽头。各自实用时间配置的对照则按 HB 主实例 §2 换算，并注明还改变了局部时钟。

架构对照可分别选择原生 Settle 与 Timed 的合适图，在任务、可见输入前缀、读出能力、训练预算及资源口径相近时比较。异时版可能更早给出有用结果，质量可能升也可能降；逻辑短路径不自动保证物理低时延。同参数量也不等于同工作量：作用、写入和 Full 次数可能不同。下游应联合观察质量、实际延迟、吞吐与实际计算／通信量，不在核心添加累计 token 活跃预算。
