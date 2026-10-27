---
type: mathematical-learning-note
status: active
cssclasses:
  - textbook-math
tags:
  - tide
  - mathematics
  - local-semantics
  - learning-note
---

# 固定逻辑时间上的短卷积

短卷积用当前内容和有限段历史形成新的特征。本文定义一个带接收权重的因果卷积部件，供 [Gated DeltaNet](gated-delta-learning-note.md) 与 [SSD](ssd-learning-note.md) 的局部记忆使用。它保存投影后的历史，时间坐标采用[共同架构](../architecture-learning-note.md)的逻辑时间 $\theta\in\mathbb N$。

在完整记忆模块中，当前聚合先作投影，Conv 将它与已保存的投影历史合成特征；接收更新用 ConvWrite 保存本次历史。节点描述和完整内容读取也可使用 Conv。本文返回卷积特征与缓存，矩阵更新、Mem 和 $G_v$ 由使用它的记忆教材继续定义。

## 1. 缓存、卷积与写入

### 1.1 保存的时间槽

固定特征宽度 $p\ge1$、抽头数 $L_{\mathrm{conv}}\ge1$、抽头的逻辑时间间距 $\delta_{\mathrm{conv}}\ge1$，后两者均为整数。抽头指卷积所读取的一个时间偏移。最远偏移为

$$
T_{\mathrm{conv}}=(L_{\mathrm{conv}}-1)\delta_{\mathrm{conv}}.
\tag{1}
$$

缓存 $\mathcal B$ 是从非负整数时间到 $\mathbb R^p$ 的有限映射，至多有 $T_{\mathrm{conv}}+1$ 项；初态为空。缺失及负时间的读取均定义为 $\mathcal B[\xi]=0$。同一节点同一时间的消息已经由 Agg 汇集，至多写一个条目。

### 1.2 当前卷积

取有限向量 $b^{\mathrm{conv}},\omega_0,\ldots,\omega_{L_{\mathrm{conv}}-1}\in\mathbb R^p$，以 $\odot$ 表示逐分量乘法。给定当前投影 $z\in\mathbb R^p$，定义

$$
\operatorname{Conv}(\mathcal B,\theta,z)
 =b^{\mathrm{conv}}+\omega_0\odot z
  +\sum_{\ell=1}^{L_{\mathrm{conv}}-1}
      \omega_\ell\odot\mathcal B[\theta-\ell\delta_{\mathrm{conv}}].
\tag{2}
$$

每个分量有自己的卷积系数，称为逐通道卷积。当前项直接使用 $z$，历史项读取缓存；本部件不含非线性，使用它的模块另定义 SiLU 等变换。

### 1.3 写入与有效缓存

给定接收权重 $0\le w\le1$，定义缓存更新

$$
\operatorname{ConvWrite}(\mathcal B,\theta,w,z)=
\begin{cases}
\left.\mathcal B\right|_{[\max(0,\theta-T_{\mathrm{conv}}),\theta)}
 \cup\{\theta\mapsto wz\},&w>0,\\
\mathcal B,&w=0.
\end{cases}
\tag{3}
$$

这里 $\left.\mathcal B\right|_I$ 表示只保留时间属于 $I$ 的条目。历史保存的是 $wz$。当前卷积的 $z$ 不乘 $w$；记忆模块在写入与遗忘处另使用 $w$，发送由区域规则决定。因而微弱接收仍可形成一个有方向的当前特征，其持久影响由这些权重控制。

在时刻 $\theta$ 比较缓存时，使用补零向量

$$
\pi_{\mathrm{conv},\theta}(\mathcal B)
 =\bigl(\mathcal B[\xi]\bigr)_{\max(0,\theta-T_{\mathrm{conv}})\le\xi\le\theta}.
\tag{4}
$$

无接收时保存元组不变，读取只按原始时间取槽。过期条目对当前和未来卷积均无贡献。

## 2. 快照读取与等间隔序列

### 2.1 更新前后的读取

可达状态按递增逻辑时间更新。式 (3) 保留当前卷积所需的全部旧槽，式 (2) 又显式使用当前 $z$，故

$$
\operatorname{Conv}(\operatorname{ConvWrite}(\mathcal B,\theta,w,z),\theta,z)
 =\operatorname{Conv}(\mathcal B,\theta,z).
\tag{5}
$$

因此 Read$^+$ 和 Full 可以从更新后的快照重建本次卷积，Read$^-$ 可以从旧快照重建；式 (2) 中当前投影只计入一次。这一等式只涉及卷积，更新前后的记忆矩阵仍可不同。保存到 $\theta-T_{\mathrm{conv}}$ 的端点使延后求值的 Full 也能使用本次快照。

### 2.2 与通常短卷积的对应

从空缓存开始，当接收时间为 $\theta_i=\theta_0+i\delta_{\mathrm{conv}}$、每次 $w_i=1$ 时，记 $z_i$ 为当前投影、$z_i=0$ 对 $i<0$，$\mathcal B_i^-$ 为本次写入前的缓存，有

$$
\operatorname{Conv}(\mathcal B_i^-,\theta_i,z_i)
 =b^{\mathrm{conv}}+\sum_{\ell=0}^{L_{\mathrm{conv}}-1}\omega_\ell\odot z_{i-\ell}.
\tag{6}
$$

这就是通常的补零因果短卷积。若某位置没有接收，该位置的历史值为零；后续位置的坐标保持。一般异步图可以在不等间隔的时间接收，式 (2) 仍给出确定的读取规则。

例如取标量、$L_{\mathrm{conv}}=3,\delta_{\mathrm{conv}}=2$，系数为 $(1,2,3)$、偏置为零。在时间 $0,2$ 分别收到 $(w,z)=(1,4),(\varepsilon,5)$，时间 $4$ 的当前投影为 $6$。卷积值为

$$
6+2(5\varepsilon)+3(4)=18+10\varepsilon.
$$

删除时间 $2$ 的事件后得到 $18$；$\varepsilon\to0$ 时两者趋同。

## 3. 弱接收与固定时间

**引理。** 在固定有限时间范围、有界投影输入和递增时间下，缓存的有效投影有界；零权重写入保持缓存；插入权重趋零、投影保持有界的事件，其对当前及固定未来时间的有效缓存影响趋零。旧有效缓存与当前投影收敛时，卷积值收敛。

**证明。** 每槽值为某次 $wz$，且 $w\le1$，有限个槽给出有界性。零权重直接返回原缓存。对趋弱事件，新槽 $wz\to0$；被清除的槽早于 $\theta-T_{\mathrm{conv}}$，当前与未来均不再读取。其余槽没有移动。以绝对时间对齐并补零后，所有保留坐标收敛。随后固定时间的读取与清理只是这些坐标的投影，后续正接收写入是连续的乘法。式 (2) 为有限和，遂得到读取连续性。$\square$

按接收次数推进的队列采用另一种时间。设队列只保存最近一项，原来为 $a\ne0$；插入微弱事件后替换成 $\varepsilon z$。下一事件读到的历史从 $a$ 变成 $\varepsilon z\to0$。即使新值乘了权重，旧值被移走的影响仍未趋零。故这类队列需要自己的性质分析；本引理使用固定时间槽。

## 附录 S：配置、表示与继续

配置名 conv_size、conv_stride 分别对应抽头数 $L_{\mathrm{conv}}$ 与抽头的逻辑时间间距 $\delta_{\mathrm{conv}}$，卷积权重和偏置逐模块声明。在逐逻辑时间排列的数组上，$\delta_{\mathrm{conv}}$ 对应通常卷积接口的 dilation；每个查询时间均可产生输出，输出采样的 stride 为 $1$。

若节点时间网格为 $\theta_t=Dt+\ell$，其中 $D\ge1,\ell\ge0$ 为固定整数、$t\in\mathbb N$ 为输入位置，取 $\delta_{\mathrm{conv}}=D$，则按输入位置排列的抽头间距和输出采样间隔均为 $1$。这些参数对应说明读取位置，带权数值仍按式 (2)—(3) 求值。两种记忆教材均使用当前项加历史槽的同一规则。

一般整数到达最多需保存 $T_{\mathrm{conv}}+1$ 个槽；在间距为 $\delta_{\mathrm{conv}}$ 的单一时间网格上最多为 $L_{\mathrm{conv}}$ 个槽。读取使用 $L_{\mathrm{conv}}$ 个抽头，主要算术量为 $O(pL_{\mathrm{conv}})$。不同余数的到达历史不能在一般图中合并成一条按接收次数推进的队列。

精确分段继续保存原始时间、式 (3) 定义的缓存和卷积参数。实现采用其他存储表示时，固定解码须恢复该缓存及作用记录中的历史快照，包括尚未由正接收清理的过期条目。若只比较未来数值读取，可以按有效投影丢弃这些条目；这一比较允许原始保存状态不同。

式 (5) 允许 Read$^-$、Read$^+$ 与 Full 在当前输入和相应快照匹配时复用本次卷积值。实现是否实际共享中间量，由求值方式决定。固定时间、边界状态和参数相同，分段读取与整段读取相同。
