---
type: mathematical-topology-definition
status: active
cssclasses:
  - textbook-math
tags:
  - tide
  - mathematics
  - topology
  - fully-connected-layered
---

# 全连接分层拓扑与时间

本构造每层只有一个区域，相邻层节点全连接，端口日程使同一输入位置的消息逐层对齐。节点、区域、边和时间采用 [SettleGraph](../settlegraph-learning-note.md) 的定义。局部函数、状态与输入载荷由使用本拓扑的实例指定。

## 1. 节点、区域与消息边

取正整数深度 $L$、区域宽度 $n$，定义

$$
\begin{aligned}
J&=\{1,\ldots,L\},\\
V&=\{v_{\ell,s}:1\le\ell\le L,\ 1\le s\le n\},\\
\rho(v_{\ell,s})&=\ell,\qquad
\mathcal R_\ell=\{v_{\ell,1},\ldots,v_{\ell,n}\}.
\end{aligned}
\tag{1}
$$

对 $1\le\ell<L$ 和 $1\le s,r\le n$，设边 $a_{\ell,s,r}$ 从 $v_{\ell,s}$ 指向 $v_{\ell+1,r}$，时延为 $1$。这些边组成全部消息边。固定按 $(\ell,s)$ 字典序排列的节点全序，用于选择平票。$L=1$ 时消息边集为空。

每个第一层节点具有输入端口 $i_s$，每个最后一层节点具有输出端口 $o_s$：

$$
\mathsf I=\{i_1,\ldots,i_n\},\quad
\mathsf O=\{o_1,\ldots,o_n\},\qquad
\gamma(i_s)=v_{1,s},\quad \varepsilon(o_s)=v_{L,s}.
$$

## 2. 输入位置与处理时间

固定输入长度 $N\ge1$，位置范围为 $[N]=\{0,\ldots,N-1\}$。第 $\ell$ 个区域的秩为 $\ell$，令 $D=L+2$，并定义

$$
\theta_{v_{\ell,s},t}=Dt+\ell,\qquad
\iota_{i_s}(t)=Dt+1,\qquad t\in[N].
\tag{2}
$$

相邻层消息在 $Dt+\ell+1$ 到达，恰为下一层的处理时间。严格区域秩、边时延与端口日程满足 SettleGraph 的合法时间编码，每节点每位置至多处理一次。

## 3. 结构性质与局部绑定

图共有 $L$ 个区域、$Ln$ 个节点和 $(L-1)n^2$ 条内部边。非末层节点的出度为 $n$，非首层节点的入度为 $n$；$L=1$ 时节点同时具有输入和输出端口。

任意入口到终端的路径经过 $L-1$ 条边，时延总和为 $L-1$。对位置 $t$，首层只能在 $Dt+1$ 接收外部输入；逐层加上单位时延，所有可能到达第 $\ell$ 层的消息均在 $Dt+\ell$ 到达。是否实际产生消息由 Full 的返回值决定。因此时间对齐与单次结算对相容的局部函数均成立。

位置 $t$ 的全部图内作用落在 $[Dt,D(t+1))$，位置边界没有跨界消息；位于位置内部的切面仍须按正典保存在途消息。给定承载值空间、各节点及区域的状态、局部函数、容量、初态和输入载荷，即得到一份完整 SettleGraph 绑定。输出的存在与取值，以及实际计算量，由这些局部选择共同决定。
