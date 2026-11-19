---
type: mathematical-instance
status: active
tags: [tide, lh, weighted-message]
---

# LH 探索图的局部绑定与有限继续

取 [LH-DAG 或 LH-PDG](../topologies/lh-exploration.md)，层次深度 $d_{\mathrm{tree}}=3,n=32$、$4\times8$ 网格、542 节点。内容宽度取 $d_{\mathrm{model}}=128$，对应被引用局部教材中的 $d$。采用 [HB 的三组主绑定](hb-multiscale-weighted-message.md#2-两类记忆与三个主候选) K／S／X；优先 X，即卫星有限窗口 KV、hub 固定状态 SSD。两图使用同一组局部参数，只在 PDG 中启用反馈边。

本候选保留区内连接与可选反馈，用于探索 HB 严格区域分层之外的语义；参考来源与形成理由见[主干记录](backbone-design.md)。

## 1. 完整替换项

继承 HB 实例 §1—2 的载荷宽度、归一化、记忆尺寸、空／零初态、全部 Upd、Mem、$p_v,G_v$、Read$^+$、encoded 与 self_attention 评分、received 下界的角色二次分配、Next 和广播。局部时间取 $P_{\mathrm{in}}=4$、$\operatorname{Pos}_v(\theta)=(\theta-1)/4$、conv_stride=1，窗口、衰减和全部有限矩阵的范围依 HB 绑定。下表补全 LH 特有部分。

| 对象 | LH 绑定 |
|---|---|
| Agg | 默认 source_residual；incoming_self_attention 按 [HB §1.1](hb-multiscale-weighted-message.md#11-接收自注意力的完整替换)替换，来源类别和参数分组取本表；同刻联合处理不读取在途或未来消息 |
| 来源类别 | ext、hub 层次、hub 到卫星、卫星到 hub、网格邻域、I→O 桥、O→I 反馈；类别由静态边决定 |
| 区域 | 默认底层 hub 与本组卫星混合，上层 hub 各自单区，两侧分别划分 |
| 上层 hub | $k=K=1$，固定通过 $c_h=\mu_h$ |
| 底层混合区 | $k=k_D+1$，$k_D=\max(1,\lceil n/32\rceil)$，$K=n+1$；hub 为 $\beta=1/4,\alpha=1/k$，卫星为普通边界 |
| 拆区对照 | 每组卫星一区，$k_D,K=n$；底层 hub 单一区，$k=K=1$、固定通过；网格区内消息保留 |
| 参数 | 采用[三档整体参数绑定](parameter-profiles.md)；默认 layer_shared，层级组以侧 I/O、层次、角色等区分，卫星属于层次 $d_{\mathrm{tree}}$；region_shared 再按实际区域拆组，independent 逐节点／区域独立；全部状态私有 |
| 输入 | 根入口 $(1,Ex_t)$，$E$ 为有限 $d_{\mathrm{model}}\times d_{\mathrm{in}}$ 矩阵，载荷属于 $\mathbb R^{1+d_{\mathrm{model}}}$；日程 $4t+1$ |
| 输出 | O 侧根的逐逻辑时间端口记录，无额外核心读出状态 |

固定阈值为零、增益为 2、正温度为 1；所有区域历史单点。区域预算和角色界满足全输入可行条件。底层 hub 只有成为正接收候选时有保障；若只有卫星收到消息，该次选择没有 hub 描述，卫星在实际候选中分配。

底层混合区可用同边界的 role_bounded_shannon 作稠密倾向的对照，此时卫星和 hub 的分配一起改变；固定通过的上层 hub 不配置评分网络。若要采用 pool 而保留 hub 待遇，则使用表中的拆区配置，仅将卫星区替换为 pooled_attention。三种选择均保留全部候选更新，不自动加入规模表或穷尽所有组合。

这些替换项与被引用的共同函数一起构成完整模型族。有限参数值由实例配置给定；不需要历史公平计数、旧 selector、完整 token 来源标签或自主时钟驱动。

## 2. 数学见证与范围

可以从图中抽取根桥、经过一级 hub 的桥、经过底层卫星和网格的一条路径，分别核对时延 $1,3,2d_{\mathrm{tree}}+3,2d_{\mathrm{tree}}+4$。设一次输入在时刻 1，输出根可以在 2 先收到直达，再在 4 收到一级往返贡献。它们是两次接收更新；某个时刻两条贡献重合时，则先聚合再写一次。不会因来自同一输入位置就自动合并所有时刻。

PDG 的根反馈给出 $H^I_{0,0}(1)\to H^O_{0,0}(2)\to H^I_{0,0}(3)$。在切面 3 处比较时，2 时刻发出的回传仍是待处理消息；继续到 3 才参与聚合。不能只保存节点记忆而删除这条消息。[参考核对](../examples/hb_lh_checks.py)在有限区间使用显式标量接收、状态累加与广播函数验证区间分割；它是继续语义的计算见证，不冒充上述 KV／SSD 整网实现。

在真正的主绑定中，弱接收、未选中写入、Read 快照和记忆继续由各模块的公式保证；正接收 hub 的有限通过由角色边界保证。运行中是否有每相位输出、数值稳定性、训练效果和设备利用率均须实测。

LH-DAG 可以使用一般 TimedDAG 的规范关闭和分块框架，但区内因果消息会让后续驱动依赖当前区域较早的计算。LH-PDG 还必须处理跨切面的反馈。两者的定位是更一般语义探索，不替换严格区域分层 HB 作为 prefill 主候选。语言读出头的具体设计交给下游。
