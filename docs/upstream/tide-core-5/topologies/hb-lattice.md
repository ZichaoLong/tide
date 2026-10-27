---
type: mathematical-topology-definition
status: active
cssclasses:
  - textbook-math
tags:
  - tide
  - mathematics
  - topology
  - hb-lattice
---

# HB-Lattice 拓扑与时间

区域递归展开，到中间层达到最大数量，再逐层收拢；相邻层节点按固定横向邻域连接。每层可有多个互不相交的区域。本构造给出 [SettleGraph](../settlegraph-learning-note.md) 的图与时间数据，记忆、选择和发送函数由实例绑定。

## 1. 递归展开与收拢

取整数分支数 $b\ge2$、展开深度 $h\ge1$、区域宽度 $n\ge1$，令总层数 $L=2h+1$。记 $\mathbb Z_n=\{0,\ldots,n-1\}$，节点内部的横向坐标按模 $n$ 相加。第 $\ell$ 层的区域地址为长度 $q_\ell$ 的字：

$$
q_\ell=h-|\ell-(h+1)|,\qquad
\mathcal B_\ell=\{0,\ldots,b-1\}^{q_\ell},
\qquad 1\le\ell\le L.
\tag{1}
$$

长度零只有空字 $\varnothing$。区域与节点定义为

$$
\begin{aligned}
J&=\{(\ell,\omega):1\le\ell\le L,\ \omega\in\mathcal B_\ell\},\\
V&=\{v_{\ell,\omega,s}:1\le\ell\le L,\ \omega\in\mathcal B_\ell,\ s\in\mathbb Z_n\},\\
\rho(v_{\ell,\omega,s})&=(\ell,\omega),\qquad
\mathcal R_{(\ell,\omega)}=\{v_{\ell,\omega,s}:s\in\mathbb Z_n\}.
\end{aligned}
\tag{2}
$$

各区域互不相交。区域秩取其层号，故 $r_{v_{\ell,\omega,s}}=\ell$。同层不同区域具有相同秩；所有消息边都严格增加秩。节点全序取 $(\ell,\omega,s)$ 的字典序，用于选择平票。

## 2. 相邻层邻域

对每个 $1\le\ell<L$，固定非空横向偏移集 $\mathcal D_\ell\subseteq\mathbb Z_n$，要求 $0\in\mathcal D_\ell$。先定义递归区域连接关系

$$
\mathcal H_\ell^0=
\begin{cases}
\{(\omega,\omega a):\omega\in\mathcal B_\ell,\ 0\le a<b\},&\ell\le h,\\
\{(\omega a,\omega):\omega\in\mathcal B_{\ell+1},\ 0\le a<b\},&\ell>h.
\end{cases}
\tag{3}
$$

取一个已声明的固定关系
$\mathcal H_\ell^0\subseteq\mathcal H_\ell\subseteq\mathcal B_\ell\times\mathcal B_{\ell+1}$。
基础形状取 $\mathcal H_\ell=\mathcal H_\ell^0$；增加区域连接或横向偏移可以得到连接更丰富的形状。

对每个 $(\omega,\omega')\in\mathcal H_\ell$、$s\in\mathbb Z_n$、$a\in\mathcal D_\ell$，放置一条有独立身份的边

$$
e_{\ell,\omega,\omega',s,a}:
 v_{\ell,\omega,s}\longrightarrow
 v_{\ell+1,\omega',s+a\bmod n},\qquad
\delta(e_{\ell,\omega,\omega',s,a})=1.
\tag{4}
$$

这些是全部消息边，层内没有消息边。偏移按集合给出，同一对端点不会因重复偏移产生重边。对基础形状的第 $\ell\to\ell+1$ 层连接，展开侧源节点出度为 $b|\mathcal D_\ell|$、目标节点入度为 $|\mathcal D_\ell|$；收拢侧分别为 $|\mathcal D_\ell|$、$b|\mathcal D_\ell|$。横向零偏移和递归关系使每个节点都位于某条入口到终端的路径上。

$|\mathcal D_\ell|\ll n$ 给出局部邻域。增大偏移集改变空间上允许的连接；在固定形状上改变局部门控则改变本次实际发送。取全部区域对和全部偏移可得到相邻层全连接，同时保留多区域选择。

## 3. 端口与时间

第一层与最后一层均只有一个区域。取 $\mathsf I=\{i_s:s\in\mathbb Z_n\}$、$\mathsf O=\{o_s:s\in\mathbb Z_n\}$，端口归属为

$$
\gamma(i_s)=v_{1,\varnothing,s},\qquad
\varepsilon(o_s)=v_{L,\varnothing,s}.
$$

固定输入长度 $N\ge1$。对位置 $t\in\{0,\ldots,N-1\}$，令 $D=L+2$，

$$
\theta_{v_{\ell,\omega,s},t}=Dt+\ell,\qquad
\iota_{i_s}(t)=Dt+1.
\tag{5}
$$

各边时延等于端点秩差，端口日程对齐，满足 SettleGraph 的单次结算条件。消息汇合仍属于同一输入位置，每节点每位置至多处理一次。同层多个区域各自结算。

每条内部边严格增加层号，所有入口到终端的路径均含 $L-1$ 条边。对任意相容的局部绑定，位置 $t$ 的所有可能作用均落在 $[Dt,D(t+1))$，位置边界没有跨界消息。取位置内部的切面时，按正典保存尚未到达的消息。

## 4. 图规模

区域数、节点数和边数分别为

$$
|J|=b^h+2\sum_{a=0}^{h-1}b^a,\qquad
|V|=n|J|,\qquad
|A|=n\sum_{\ell=1}^{L-1}|\mathcal H_\ell|\,|\mathcal D_\ell|.
\tag{6}
$$

区域数由各层地址长度得到。对每个相连区域对，$n$ 个源节点各有 $|\mathcal D_\ell|$ 条边，给出边数公式。

取 $b=2,h=3,n=64$，各层区域数为 $1,2,4,8,4,2,1$，共有 $22$ 个区域、$1408$ 个节点。基础关系 $\mathcal H_\ell^0$ 与偏移 $\{0,1\}$ 给出 $3584$ 条边；展开侧最大出度为 $4$，收拢侧为 $2$。偏移改为 $\{0,1,2,3\}$ 时共有 $7168$ 条边。

以上计量固定空间图。一次运行中的候选、激活与消息数量由输入和局部函数决定；区域是否接收消息、输出的存在与取值及实际计算量须在完整绑定中推导。
