# TIDE 实验语义、命名与数学符号

> 状态：新实验的规范性文档。
>
> 本文只定义“模型实际怎样计算”和“实验名称怎样反映计算图”。实验晋级、结果报告组织和 checkpoint 保留策略另行讨论。
>
> 第 4 节复杂拓扑中标为“首个设置”或“推荐”的部分是待逐项核验的候选默认值，核验台帐见 [`experiment-semantics-review-ledger.md`](experiment-semantics-review-ledger.md)。

## 阅读入口：先看完整图景

本文只从更上层研究计划继承 **TIDE** 这个名字，其余内容均可独立阅读。本文研究插入现有 Transformer 的 **GraphBranch**：对每个 Token，消息沿固定边到达 AggregatePorts，receiver nodes 准备可选状态与轻量读出，局部 selector 和传播 profile 决定状态提交与 active nodes，active nodes 完成计算并经 `EmitPolicy` 继续发送，终端端口最终返回一个同维 hidden；它相对入口产生的变化再合入 base 的 always-on 路径。第 1 节定义外部边界，第 2 节定义共用组件与策略，第 3、4 节分别组成单层特例和 HB-Lattice。

## 1. Base block 与 GraphBranch 顶层边界

### 1.1 Base 与顶层接口符号

本节只引入理解 base block 和 GraphBranch 接入位置所需的符号；内部共用组件、单层特例和 HB-Lattice 的符号分别在第 2、3、4 节首次使用时定义。

| 符号 | 含义 |
| --- | --- |
| \(b\) | micro-batch 中的序列编号；正文通常省略这一维 |
| \(\ell\) | base Transformer block 编号 |
| \(j\) | GraphBranch 插入位置（site）编号；site \(j\) 所在的 base block 记为 \(\ell(j)\) |
| \(t\) | 序列中的 Token 位置 |
| \(x_{\ell,t}\) | 实际送入第 \(\ell\) 个 base block 的 hidden |
| \(u_{\ell,t}\) | 当前 block 完成 Attention residual merge 后的 hidden |
| \(v_{\ell,t}\) | 当前 block 完成原 dense MLP residual merge 后的 hidden |
| \(y_{\ell,t}\) | 当前 block 连同可选 GraphBranch 最终送往下一个 block 的 hidden |
| \(h^{\mathrm{in}}_{j,t}\) | site \(j\) 的 GraphBranch 实际入口 hidden |
| \(b_{\mathcal G,j,t}\) | GraphBranch 返回的完整 hidden |
| \(\Delta_{\mathcal G,j,t}\) | GraphBranch 相对入口产生的 residual |

后文始终按“作用”复用符号：各类归一化写成 \(N\)，私有状态写成 \(s/S\)，状态操作写成 \(\operatorname{Update}\)、\(\operatorname{Read}^{\mathrm{sel}}\) 和 \(\operatorname{Read}^{\mathrm{ffn}}\)，激活节点的完整计算统一写成 \(\operatorname{NodeCompute}\)。相同基本符号表示组件承担相同作用，不表示共享参数或采用相同算法；\(\operatorname{Read}^{\mathrm{ffn}}\) 是当前默认 node 模板内部给昂贵 FFN 路径使用的状态读出。

### 1.2 Base Qwen3 block

令 \(N_A\) 和 \(N_F\) 分别表示 Attention 前与 MLP 前、逐 Token 执行的归一化操作；当前 Qwen3 base block 中二者都实现为 RMSNorm。\(A_\ell\) 表示 causal self-attention，\(F_\ell\) 表示原 dense SwiGLU MLP。先把第 \(\ell\) 个 block 到位置 \(t\) 为止的输入前缀记为：

$$
X_{\ell,\le t}
:=(x_{\ell,0},x_{\ell,1},\ldots,x_{\ell,t}).
$$

\(N_A(X_{\ell,\le t})\) 表示分别归一化前缀中的每个 Token，方括号后的下标 \(t\) 表示取 self-attention 在当前位置的输出。一个原始 Pre-Norm Qwen3 block 对位置 \(t\) 计算：

$$
u_{\ell,t}
=x_{\ell,t}
+\left[
A_\ell\!\left(N_A(X_{\ell,\le t})\right)
\right]_t,
$$

$$
v_{\ell,t}
=u_{\ell,t}
+F_\ell\!\left(N_F(u_{\ell,t})\right).
$$

因此 \(x_{\ell,t}\) 是实际送入第 \(\ell\) 个 base block 的当前 Token hidden；对 \(\ell>0\)，有 \(x_{\ell,t}=y_{\ell-1,t}\)，而 \(y_{\ell-1,t}\) 已经包含上一层可能存在的 GraphBranch merge。

Dense 基线没有 receiver，直接令：

$$
y_{\ell,t}=v_{\ell,t}.
$$

### 1.3 GraphBranch 的单入口、单出口契约

每个 site 在原有 base computation 之外只接入一个 GraphBranch，记为 \(\mathcal G_j\)。**GraphBranch** 是整个单入口、单出口模块的专名。**placement** 表示 GraphBranch 相对当前 base block 的接入位置；对当前 Token，四种 placement 的入口分别是：

$$
h^{\mathrm{in}}_{j,t}
=
\begin{cases}
v_{\ell,t}, & \text{POST},\\
x_{\ell,t}, & \text{PARBLK},\\
x_{\ell,t}, & \text{PARATTN},\\
u_{\ell,t}, & \text{PARMLP},
\end{cases}
\qquad \ell=\ell(j).
$$

GraphBranch 内部可以采用第 3 节的单层特例，也可以采用第 4 节的 HB-Lattice；它对外始终只返回一个同维 hidden：

$$
b_{\mathcal G,j,t}
=\mathcal G_j\!\left(h^{\mathrm{in}}_{j,t}\right),
\qquad
\Delta_{\mathcal G,j,t}
=b_{\mathcal G,j,t}-h^{\mathrm{in}}_{j,t}.
$$

这里的 \(\mathcal G_j\) 省略了逐序列持久状态；第 2、3 节会展开状态接口与具体读写顺序。无论内部多复杂，placement 只看见入口 \(h^{\mathrm{in}}\)、完整输出 \(b_{\mathcal G}\) 和唯一 residual \(\Delta_{\mathcal G}\)。

若 placement 的 always-on 输出记为 \(b^0_{j,t}\)，GraphBranch 与 base 的边界统一使用 **RESIDUAL_ADD**：

$$
\operatorname{BoundaryMerge}
\left(h^{\mathrm{in}}_{j,t};b^0_{j,t},b_{\mathcal G,j,t}\right)
=b^0_{j,t}+\left(b_{\mathcal G,j,t}-h^{\mathrm{in}}_{j,t}\right)
=b^0_{j,t}+\Delta_{\mathcal G,j,t}.
$$

它保留 always-on 路径，只叠加 GraphBranch 相对共同入口产生的变化。第 2.5、3.5 节会把这种外部边界与 GraphBranch 内部的消息聚合分开。

### 1.4 四种 placement

四种 placement 只改变 \(h^{\mathrm{in}}\)、always-on 输出和 residual 合入位置，不改变 GraphBranch 的内部接口。

#### 1.4.1 POST：完整 block 后串联

先按第 1.2 节得到完整 base block 输出 \(v_{\ell,t}\)，再令：

$$
h^{\mathrm{in}}_{j,t}=v_{\ell,t},
\qquad
y_{\ell,t}
=v_{\ell,t}+\Delta_{\mathcal G,j,t}
=b_{\mathcal G,j,t}.
$$

~~~text
x → Attention → u → 原 dense MLP → v → GraphBranch → y
~~~

GraphBranch 能看到当前 block 的 Attention 和原 MLP 结果。POST 是串联结构，也是最直接的首个实现位置。

#### 1.4.2 PARBLK：与完整 block 并列

GraphBranch 和完整 base block 都从 \(x_{\ell,t}\) 开始，最后在 block 出口合并：

$$
h^{\mathrm{in}}_{j,t}=x_{\ell,t},
\qquad
y_{\ell,t}
=v_{\ell,t}+\Delta_{\mathcal G,j,t}.
$$

~~~text
          ┌→ 完整 base block → v ─────┐
x ────────┤                            + → y
          └→ GraphBranch → Δ_G(x) ────┘
~~~

GraphBranch 看不到当前 block 的 Attention 或 MLP 结果，也不改变它们的输入；两条路径可以并行执行。

#### 1.4.3 PARATTN：与 Attention 并列

GraphBranch 与 Attention 都读取 \(x_{\ell,t}\)。先在 Attention residual 位置合并，再让原 dense MLP 读取合并后的表示：

$$
h^{\mathrm{in}}_{j,t}=x_{\ell,t},
\qquad
u'_{\ell,t}=u_{\ell,t}+\Delta_{\mathcal G,j,t},
$$

$$
y_{\ell,t}
=u'_{\ell,t}+F_\ell\!\left(N_F(u'_{\ell,t})\right).
$$

~~~text
          ┌→ self-attention ─┐
x ────────┤                   + → u' → 原 dense MLP → y
          └→ GraphBranch ────┘
~~~

PARATTN 只说明 GraphBranch residual 的接入位置，不限制 GraphBranch 内部只能使用 Attention。

#### 1.4.4 PARMLP：与 MLP 并列

Attention residual 先得到 \(u_{\ell,t}\)；原 dense MLP 与 GraphBranch 都读取这个共同输入，最后在 MLP residual 位置合并：

$$
h^{\mathrm{in}}_{j,t}=u_{\ell,t},
\qquad
y_{\ell,t}
=v_{\ell,t}+\Delta_{\mathcal G,j,t}.
$$

~~~text
x → self-attention → u
                      ├→ 原 dense MLP ─┐
                      └→ GraphBranch ── + → y
~~~

GraphBranch 能看到当前 Attention 的结果，但看不到当前原 MLP 的结果，也不改变原 MLP 的输入。本文统一使用 **PARMLP**；**PARFFN** 指同一个 placement。原 dense MLP 是 always-on 路径，GraphBranch 是与它并列的稀疏有状态主旁路。

#### 1.4.5 直接比较与初始化

| Placement | \(h^{\mathrm{in}}\) | always-on 输出 | 看见当前 Attention | 看见当前原 MLP | 改变原 MLP 输入 | GraphBranch merge 后 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| **POST** | \(v\) | \(v\) | 是 | 是 | 否 | 直接得到 \(y\) |
| **PARBLK** | \(x\) | \(v\) | 否 | 否 | 否 | 直接得到 \(y\) |
| **PARATTN** | \(x\) | \(u\) | 否 | 否 | 是 | 得到 \(u'\)，再执行原 MLP |
| **PARMLP** | \(u\) | \(v\) | 是 | 否 | 否 | 直接得到 \(y\) |

若 GraphBranch 初始化为 identity，则 \(b_{\mathcal G}=h^{\mathrm{in}}\)、\(\Delta_{\mathcal G}=0\)，四种 placement 都保持原 base 函数不变。离开初始点后，它们的前向耦合、梯度路径和有效深度不同，不能视为同一架构。语义上保留四种 placement；实现可以先从 POST 开始。

## 2. GraphBranch 内部的共用组件与策略接口

单层特例和 HB-Lattice 使用同一套组件与策略接口；具体拓扑只决定它们的数量、固定连接和配置映射。数据面始终传递 \(d_{\mathrm{model}}\) 维的**完整 hidden**。

### 2.1 共用角色一览

| 类别 | 共用角色 | 职责 |
| --- | --- | --- |
| 数据图 | `GraphInputPort` | 唯一入口端点；把 \(h^{\mathrm{in}}\) 沿固定出边发送 |
| 数据图 | 固定边 | 静态规定完整 hidden 可以从哪个 sender 传到哪个端口 |
| 数据图 | `AggregatePort` | 收集固定 parents 实际送达的消息，合成一个完整 hidden |
| 数据图 | receiver node | 持有独立参数与可选私有状态，active 时产生完整 hidden |
| 数据图 | `GraphOutputPort` | 唯一出口端点和终端 AggregatePort；聚合最终消息并返回 \(b_{\mathcal G}\) |
| 执行策略 | `MessageAggregate` | 规定每个 AggregatePort 怎样合并实际收到的消息 |
| 执行策略 | `ReceiverNodeTemplate` | 规定 receiver 内部的状态、昂贵计算、归一化和 residual |
| 控制 | selector / region | 在一个固定局部 region 的 reached receivers 中选择 active set |
| 控制 | propagation profile | 规定哪些 reached receivers 提交状态、哪些执行完整计算 |
| 执行策略 | `EmitPolicy` | 把 active receiver 的完整输出变成沿固定出边发送的消息 |
| 仅训练 | `BalancePolicy` | 根据选择事件产生辅助均衡 loss，不改变推理数据流 |

这些都是两种拓扑共用的角色。`GraphInputPort` 和 `GraphOutputPort` 各只有一个，均不是 receiver node；`GraphOutputPort` 同时是一个 AggregatePort。第 1.3 节的 GraphBranch 单入口、单出口契约与 `BoundaryMerge` 也对所有内部拓扑一致，但位于 GraphBranch 与 base 的外部边界。

### 2.2 边界端口、固定边与消息聚合

`GraphInputPort` 是数据图的 source：它不聚合消息、不持有状态，直接把 GraphBranch 入口 \(h^{\mathrm{in}}\) 沿全部固定出边发送。

每个 receiver node 前都有一个输入 AggregatePort。令端口 \(a\) 在 Token \(t\) 实际收到的非空消息集合为 \(\mathcal M_{a,t}\)，则：

$$
h_{a,t}
=\operatorname{MessageAggregate}_a(\mathcal M_{a,t}).
$$

收到至少一条消息的 receiver 才是 **reached**。一条消息时聚合退化为恒等操作；没有消息时端口不产生输出，后面的 receiver 也不执行。`GraphOutputPort` 后面不隐含 receiver node，它直接把聚合结果作为 \(b_{\mathcal G}\) 返回。

AggregatePort 不持有 receiver 私有状态，不参加 selector，也不执行 `NodeCompute`。一个 receiver 向多个 children 发送消息只是固定边的 fan-out，不需要额外的“发散点”。

### 2.3 Receiver node

**receiver node** 是固定拓扑上持有独立参数和可选私有状态的计算节点。它通常包含一个记忆/状态模块和一个昂贵计算模块；拓扑只依赖下面的稳定契约，不依赖节点内部采用 EMA、Gated DeltaNet、Attention 还是其他实现。

| 契约项 | 输入 | 结果 |
| --- | --- | --- |
| 入口准备 | 聚合后的完整 hidden \(h\) | receiver 本地消息 \(m\) |
| 状态 proposal | 旧状态 \(s^-\) 与 \(m\) | 临时状态 \(\widetilde s=\operatorname{Update}(s^-,m)\) |
| selector 读出 | \(m\) 与指定时刻的状态 | 轻量 \(r^{\mathrm{sel}}\) |
| 状态提交 / Observe | \(s^-\)、\(\widetilde s\) 与传播 profile | 下一 Token 可见的持久状态 \(s\) |
| 激活计算 | \(h\)、\(m\) 与已提交状态 \(s\) | 同维完整 hidden \(g=\operatorname{NodeCompute}(h,m,s)\) |

`Update` 只产生 proposal；只有 commit 把 proposal 保存为持久状态时，当前消息才算被该节点 **Observe**。receiver 不自行决定是否 active，也不在内部乘 selector 概率。它只向 selector 提供轻量读出；较大的状态读出和昂贵计算只由 active nodes 执行。

当前默认 `ReceiverNodeTemplate` 采用 Pre-Norm 双 residual，但模板可以替换；只要继续满足轻量 selector 读出、状态提交和完整 hidden 输出契约，拓扑、selector 与消息聚合就不需要改变。

### 2.4 Selector、传播 profile 与 EmitPolicy

**selector** 与一个固定的局部 receiver 集合（region）关联。对当前 Token，它只在其中已经 reached 的 receivers 中产生 soft probabilities \(p\) 和 hard active set \(\mathcal A\)。selector 读取当前内容、presence 标记以及 receivers 本地生成的低维向量、范数或历史统计，不读取 receiver 的完整私有状态，也不执行 receiver 的昂贵计算。

**candidate** 只是当前参加选择的 reached receiver；被选中的 candidate 称为 **active**。二者都是 receiver 在当前 Token 的运行时角色，不是新的组件类型。

selector 是控制模块，不是拓扑节点或发散点：固定边决定消息能到达哪里，selector 只决定 reached receivers 中哪些可以继续完整计算和发送。

selector 读取状态的时刻有三种：

| 时序 | selector 使用的 receiver 信息 | 自然兼容的有状态 profile |
| --- | --- | --- |
| **Content-only** | 当前本地消息的轻量读出 | SD、BO |
| **Pre-update state** | 当前消息到来前的旧状态读出 | SD、BO |
| **Post-update state** | 当前消息产生的状态 proposal 读出 | BO |

N 没有 receiver 私有状态，因此只使用 content-only。

传播 profile 决定哪些 reached nodes 提交状态、哪些 nodes 执行完整计算：

| Profile | 状态提交 / Observe | 完整 `NodeCompute` 与发送 |
| --- | --- | --- |
| **N（stateless）** | 无私有状态 | active nodes |
| **SD（selected-dispatch）** | active nodes | active nodes |
| **BO（broadcast-observe）** | 全部 reached nodes | active nodes |

active receiver 得到完整输出 \(g_{v,t}\) 后，由节点外部的 `EmitPolicy` 产生实际消息：

$$
\widehat g_{v,t}
=\operatorname{EmitPolicy}_v(h_{v,t},g_{v,t},p_{v,t}).
$$

同一个 \(\widehat g_{v,t}\) 被复制到该 receiver 的全部固定出边。selector 概率对当前主任务前向或梯度的直接作用统一放在 `EmitPolicy`，不在 receiver 内部或 `MessageAggregate` 中重复使用。`BalancePolicy` 只根据选择事件产生训练期辅助 loss，不改变推理数据流。

### 2.5 每个 Token 的共用数据流

不论拓扑深浅，一个 Token 都重复下面的局部过程：

~~~text
GraphInputPort(h_in) → 固定边
  → AggregatePort / MessageAggregate → reached receiver 的完整入口 h
  → receiver 本地消息，以及当前 selector 时序所需的轻量 Read^sel / proposal
  → region selector + propagation profile → active set 与状态提交
  → active receiver 的 NodeCompute
  → EmitPolicy → 固定边
      ├→ 后续 receiver 的 AggregatePorts（若有则重复上述局部过程）
      └→ GraphOutputPort / MessageAggregate → b_G
~~~

content-only、pre-update 和 post-update 只改变 selector 前的轻量阶段顺序；N、SD、BO 只改变状态提交范围。拓扑则决定端口、receivers、regions 和固定边怎样排列：

| 拓扑形态 | 组件排列 |
| --- | --- |
| **单层特例** | 输入端口连接一层并列 receivers；一个 selector 处理这些候选；输出端口聚合 active receivers 的消息 |
| **HB-Lattice** | receivers 放在有序 Lines 中；每个 Line 划分局部 regions；固定边支持多父、多子和受限跨 Line 直通 |

第 3 节用单层特例展开 selector 时序、状态提交、节点模板和 Emit/聚合公式；第 4 节保持上述角色与接口不变，只增加 Plan、Line、波前和多父消息。

## 3. 单层特例：用最小拓扑展开共用接口

本节不引入新组件，只把第 2 节的共用角色放进一个最小拓扑并给出完整公式：输入端口连接一层并列 receiver nodes，一个 selector 负责这些候选，active nodes 的消息直接进入输出端口。

### 3.1 拓扑与局部符号

把 GraphBranch 的唯一输入端口和终端 AggregatePort 分别记为 `GraphInputPort` 与 `GraphOutputPort`。这个单层样例有 \(R\) 个并列 receiver nodes：输入端口沿 \(R\) 条固定边发送同一个 \(h^{\mathrm{in}}_{j,t}\)，所以每个 receiver 的输入 AggregatePort 都只收到一条消息，聚合后仍是 \(h^{\mathrm{in}}_{j,t}\)。active nodes 的最终消息再由 `GraphOutputPort` 聚合成 \(b_{\mathcal G,j,t}\)。

拓扑形状为：

~~~text
GraphInputPort(h_in)
  ├→ AggregatePort → receiver 0 ─┐
  ├→ AggregatePort → receiver 1 ─┤
  ├→ ...                         ├→ GraphOutputPort → b_G
  └→ AggregatePort → receiver R-1┘
          一个 selector 在这 R 个 reached candidates 中选择
~~~

“单层”表示任一入口到出口路径只经过一个 receiver node，不表示 GraphBranch 总共只有一个 node。一个 receiver node 内部可以串行执行多个子层；只要它仍作为一个拓扑节点接收和返回完整 hidden，这些内部子层就不会变成新的拓扑节点。

例如，这个结构包含 4 个 nodes 且采用 Top-1 时，每个 Token 只选其中一个做完整计算；这 4 个 nodes 并列存在，并不顺序串行。

本节使用以下局部下标和集合：

| 符号 | 含义 |
| --- | --- |
| \(i\) | 当前候选 receiver node 的编号 |
| \(R\) | 并列 receiver node 总数 |
| \(K_{\mathrm{act}}\) | 当前激活的 receiver node 数量 |
| \(\mathcal A_{j,t}\) | active receiver node 集合 |
| \(\mathcal O_{j,t}\) | 当前 Token 实际 Observe 消息的 receiver node 集合 |
| \(s_{j,t}^{(i)}\)、\(S_{j,t}\) | receiver node \(i\) 的私有状态，以及这 \(R\) 个 nodes 的全部私有状态 |
| \(g_{j,t}^{(i)}\) | active receiver node \(i\) 完成 `NodeCompute` 后的完整 hidden |
| \(\widehat g_{j,t}^{(i)}\) | \(g_{j,t}^{(i)}\) 经 `EmitPolicy` 处理后实际发送的完整 hidden |
| \(\operatorname{Inbox}_{\mathrm{out},j,t}\) | `GraphOutputPort` 当前实际收到的消息集合 |

### 3.2 入口消息与私有状态

下面在单层特例中展开这套契约。receiver node \(i\) 的输入 AggregatePort 只收到共同入口 \(h^{\mathrm{in}}_{j,t}\)，所以聚合结果仍是它；selector 使用自己的归一化 \(N_{\mathrm{sel}}\)，node \(i\) 使用自己独立的入口归一化 \(N_{R,i}\)：

$$
\mu_{j,t}=N_{\mathrm{sel}}\!\left(h^{\mathrm{in}}_{j,t}\right),
\qquad
m_{j,t}^{(i)}=N_{R,i}\!\left(h^{\mathrm{in}}_{j,t}\right),
\quad i=0,1,\ldots,R-1.
$$

\(\mu_{j,t}\) 是 selector 的公共内容消息，\(m_{j,t}^{(i)}\) 是 receiver \(i\) 的本地入口消息。各 receiver 只在本地使用 \(m_{j,t}^{(i)}\)，并只向 selector 发送第 3.3 节定义的轻量 \(\operatorname{Read}^{\mathrm{sel}}\)；selector 不读取所有 receivers 的完整入口消息。

使用相同 \(\epsilon\) 的 RMSNorm 时，各归一化的无参数 RMS 统计相同，可以只计算一次：

$$
\bar h^{\mathrm{in}}_{j,t}
=\frac{h^{\mathrm{in}}_{j,t}}
{\operatorname{RMS}(h^{\mathrm{in}}_{j,t})},
$$

再分别应用互不共享的可学习 scale：

$$
\mu_{j,t}=g_{\mathrm{sel}}\odot\bar h^{\mathrm{in}}_{j,t},
\qquad
m_{j,t}^{(i)}=g_{R,i}\odot\bar h^{\mathrm{in}}_{j,t}.
$$

对于有状态的 SD 和 BO，在处理 Token \(t\) 之前，把这 \(R\) 个 nodes 的完整私有状态统一记为：

$$
S_{j,t-1}
:=\left(
s_{j,t-1}^{(0)},
s_{j,t-1}^{(1)},
\ldots,
s_{j,t-1}^{(R-1)}
\right).
$$

这里的 \(s_{j,t}^{(i)}\) 只表示“receiver \(i\) 的完整私有状态”，不预先限定它是 EMA 向量、GDN 矩阵，还是在内部额外包含历史激活记录。具体内部结构不改变框架符号。

### 3.3 三种 selector 时序

单层特例的 selector 是位于全部候选 nodes 外部的打分模块。它接收公共内容消息 \(\mu_{j,t}\) 和各 node 的轻量 \(r^{\mathrm{sel}}\)，输出全部候选的 logits；三种语义只改变 node 在什么时刻生成这份轻量读出：

| Selector 语义 | 打分时可读取 | SD | BO |
| --- | --- | --- | --- |
| **Content-only** | 公共消息 \(\mu_{j,t}\) 与各 receiver 从当前本地消息发出的轻量读出 | 自然兼容 | 自然兼容 |
| **Pre-update state** | \(\mu_{j,t}\) 与 receivers 从旧状态发出的轻量读出 | 自然兼容；选完后只更新 active receivers | 自然兼容；选完后更新全部 receivers |
| **Post-update state** | \(\mu_{j,t}\) 与 receivers 从更新后状态发出的轻量读出 | 严格 SD 不兼容 | 天然兼容；全部更新后再选择 |

先定义当前本地消息更新后的临时状态，以及三种时刻的轻量 receiver 读出：

$$
\widetilde s_{j,t}^{(i)}
=\operatorname{Update}_i(s_{j,t-1}^{(i)},m_{j,t}^{(i)}),
$$

$$
r_{j,t,\mathrm{content}}^{\mathrm{sel},(i)}
=\operatorname{Read}_i^{\mathrm{sel}}(m_{j,t}^{(i)}),
$$

$$
r_{j,t,\mathrm{pre}}^{\mathrm{sel},(i)}
=\operatorname{Read}_i^{\mathrm{sel}}(s_{j,t-1}^{(i)},m_{j,t}^{(i)}),
\qquad
r_{j,t,\mathrm{post}}^{\mathrm{sel},(i)}
=\operatorname{Read}_i^{\mathrm{sel}}(\widetilde s_{j,t}^{(i)},m_{j,t}^{(i)}).
$$

三种打分统一写成：

$$
a_{j,t}
=
\begin{cases}
\operatorname{Score}\!\left(
\mu_{j,t},
\left(r_{j,t,\mathrm{content}}^{\mathrm{sel},(k)}\right)_{k=0}^{R-1}
\right),
& \text{content-only},\\[4pt]
\operatorname{Score}\!\left(
\mu_{j,t},
\left(r_{j,t,\mathrm{pre}}^{\mathrm{sel},(k)}\right)_{k=0}^{R-1}
\right),
& \text{pre-update state},\\[4pt]
\operatorname{Score}\!\left(
\mu_{j,t},
\left(r_{j,t,\mathrm{post}}^{\mathrm{sel},(k)}\right)_{k=0}^{R-1}
\right),
& \text{post-update state}.
\end{cases}
$$

$$
p_{j,t}=\operatorname{softmax}(a_{j,t}),
\qquad
\mathcal A_{j,t}=\operatorname{TopKIndex}(p_{j,t},K_{\mathrm{act}}),
\qquad 1\le K_{\mathrm{act}}\le R.
$$

\(\operatorname{TopKIndex}(p,K)\) 返回概率 \(p\) 中最大的 \(K\) 个候选下标。

Top-1 是 \(K_{\mathrm{act}}=1\) 的特例，此时继续记：

$$
c_{j,t}=\arg\max_i p_{j,t}^{(i)},
\qquad
\mathcal A_{j,t}=\{c_{j,t}\}.
$$

每个直接 receiver 在局部执行 \(N_{R,i}\) 和 \(\operatorname{Read}^{\mathrm{sel}}\)，只向 selector 提供小向量、范数或历史激活统计等少量标量；\(\operatorname{Score}\) 一次输出全部 logits，可以逐候选独立打分，也可以联合处理这些轻量读出。\(\widetilde s\) 是同一个 \(s\) 在“已经为当前消息计算 proposal、尚未选择和 commit”阶段的临时值。Pre 与 Post 不是包含关系：如果 \(\operatorname{Update}\) 会覆盖、压缩或遗忘旧状态，post-update state 未必还能恢复 pre-update state；若 selector 同时读取二者，应另行明确声明。

历史激活也可以作为 \(s\) 的内部内容。当前 Token 的激活结果只能影响以后 Token，因此会形成时间维上的因果递归；这不妨碍在一个 chunk 内用 scan、Torch 算子或专用 kernel 执行，但相同前缀的整段预填充（prefill）、分块预填充与逐 Token 解码（decode）必须得到相同结果。

### 3.4 传播 profile、状态提交与 receiver node compute

#### 3.4.1 Observe 范围与状态提交

令 \(\mathcal O_{j,t}\) 表示当前消息实际完成 commit、因而真正 Observe 的 receiver node 集合，\(\mathcal A_{j,t}\) 表示执行较大读出和昂贵 FFN 的 active receiver nodes。三种传播 profile 的规则是：

| Profile | \(\mathcal O_{j,t}\)：哪些 receiver nodes Observe 当前消息 | 哪些 receiver nodes 继续执行较大读出与昂贵 FFN |
| --- | --- | --- |
| **N（stateless）** | 无状态，不计算 proposal，也不 Observe | \(\mathcal A_{j,t}\)；没有状态读出 |
| **SD（selected-dispatch）** | \(\mathcal A_{j,t}\) | \(\mathcal A_{j,t}\) |
| **BO（broadcast-observe）** | 全部 \(R\) 个 receivers | \(\mathcal A_{j,t}\) |

本表先按单层特例书写：全部 \(R\) 个候选都已收到共同入口。第 4 节推广到多层拓扑时，只把 BO 的“全部 \(R\) 个 nodes”改成“当前真正 reached 的全部 nodes”。

设计定位上，BO 是以后走向一般 Graph 的主要候选 profile，因为全部 reached 节点都可以先计算 proposal、供 post-update selector 读取，再统一 commit / Observe；N 和 SD 则分别保留为从无状态 MoE 与 selected dispatch 出发的 matched controls（匹配对照）。这个定位是实验主轴，不预先代表 BO 已被证明更优。

Content-only 和 pre-update state 都先完成选择，再按上表提交状态：

$$
s_{j,t}^{(i)}
=
\begin{cases}
\operatorname{Update}_i\!\left(s_{j,t-1}^{(i)},m_{j,t}^{(i)}\right),
& i\in\mathcal O_{j,t},\\[4pt]
s_{j,t-1}^{(i)},
& i\notin\mathcal O_{j,t}.
\end{cases}
$$

Post-update state 与 BO 的顺序是“全部 receivers 计算 Update proposal → selector → 全部 commit / Observe”，所以直接有：

$$
s_{j,t}^{(i)}=\widetilde s_{j,t}^{(i)},
\qquad i=0,1,\ldots,R-1.
$$

严格的 SD 不自然兼容 post-update state，因为选择发生前还不知道哪些 receivers 可以 Update。N 没有 receiver 私有状态，因此当前规范下只使用 content-only。

状态提交完成后的完整状态统一记为：

$$
S_{j,t}
:=\left(
s_{j,t}^{(0)},
s_{j,t}^{(1)},
\ldots,
s_{j,t}^{(R-1)}
\right).
$$

#### 3.4.2 Active NodeCompute 与当前默认模板

选择与状态提交完成后，active receiver node 的稳定输出契约统一写成：

$$
g_{j,t}^{(i)}
=\operatorname{NodeCompute}_{j,i}\!\left(
h^{\mathrm{in}}_{j,t},m_{j,t}^{(i)},s_{j,t}^{(i)}
\right),
\qquad i\in\mathcal A_{j,t}.
$$

其中无状态 N 忽略状态参数。当前默认的 **Pre-Norm 双 residual 模板**按下面的方式展开。在有状态的 SD 和 BO 中，每个 active node 先在局部执行较大的 \(\operatorname{Read}^{\mathrm{ffn}}\)：

$$
\rho_{j,t}^{(i)}
=\operatorname{Read}_{i}^{\mathrm{ffn}}\!\left(
s_{j,t}^{(i)},m_{j,t}^{(i)}
\right),
\qquad
\rho_{j,t}^{(i)}\in\mathbb R^{d_{\mathrm{model}}}.
$$

无论 selector 使用 content-only、pre-update state 还是 post-update state，\(\operatorname{Read}^{\mathrm{ffn}}\) 都读取已经提交当前 Token 后的状态。它直接返回 hidden 维 residual；Attention output projection、EMA/GDN 状态投影等实现细节都包含在该操作内部。无状态的 N 不执行状态读出，对每个 active receiver 令：

$$
\rho_{j,t}^{(i)}=0.
$$

每个 active receiver node 先把状态/上下文读出加回未归一化的 residual stream，再执行一个 Pre-Norm FFN。令 \(N_{F,i}\) 表示 receiver node \(i\) 的 FFN 前归一化：

$$
u_{j,t}^{(i)}
=h^{\mathrm{in}}_{j,t}+\rho_{j,t}^{(i)},
$$

$$
z_{j,t}^{(i)}
=N_{F,i}\!\left(u_{j,t}^{(i)}\right),
$$

$$
g_{j,t}^{(i)}
=u_{j,t}^{(i)}
+E_{j,i}\!\left(z_{j,t}^{(i)}\right),
\qquad i\in\mathcal A_{j,t}.
$$

因此，当前默认有状态 node 模板的 residual 顺序是：

~~~text
入口 h
  → 记忆/上下文读出 residual：u = h + ρ
  → 昂贵 FFN residual：output = u + E(N_F(u))
~~~

这直接对标 Pre-Norm Transformer block：记忆模块的读出承担第一个 residual 子层，昂贵 FFN 承担第二个 residual 子层；\(N_{R,i}\) 和 \(N_{F,i}\) 分别是两个子层的入口归一化。记忆模块内部可以采用 EMA、GDN、Attention 等实现。N 没有第一个 residual 子层，节点输出退化为 \(h+E_i(N_{F,i}(h))\)，但仍可计算供 selector 使用的本地消息和轻量读出。

Pre-Norm 双 residual 是当前默认 `ReceiverNodeTemplate`，不是 receiver node 的永久定义。以后可以替换状态模块、昂贵计算、执行顺序、归一化或 residual 组合；只要继续满足第 2.3 节的轻量读出、状态提交和完整 hidden 输出契约，selector、拓扑执行与外部汇聚语义都无需改变。每个实验必须记录模板的精确公式和初始化。

\(N_{\mathrm{sel}}\)、各 \(N_{R,i}\) 与各 \(N_{F,i}\) 的可学习参数互不共享，也不与 base block 共享。这两个 residual 子层合计仍只算一个 receiver node；只有该节点的完整输出继续进入另一个 receiver node 时，拓扑才增加一层。

在单层特例中，各 receiver nodes 共享入口和 selector，但各自拥有独立的参数与可选状态。\(g_{j,t}^{(i)}\) 是 node \(i\) 完成 `NodeCompute` 后的完整输出。全部候选 nodes 都执行本地入口归一化和轻量 \(\operatorname{Read}^{\mathrm{sel}}\)；每个 active node 再执行一次较大读出和一次昂贵 FFN。

至此，selector 与候选 nodes 已产生 soft probabilities \(p_{j,t}\)、active node set \(\mathcal A_{j,t}\)、提交后的状态 \(S_{j,t}\) 和各 active nodes 的完整输出 \(g_{j,t}^{(i)}\)。第 3.5 节再定义这些输出怎样变成消息并由输出 AggregatePort 聚合。若概率还写入历史状态，必须另外声明写回与梯度规则。

### 3.5 EmitPolicy 与 MessageAggregate

active receiver node \(i\) 先产生完整输出 \(g_{j,t}^{(i)}\)，节点外部的 `EmitPolicy` 再把它变成实际发送的消息 \(\widehat g_{j,t}^{(i)}\)。selector 概率对当前主任务前向或梯度的直接作用统一放在这里，不放进 receiver node 或 `MessageAggregate`。

首个推荐设置是 **EMIT-HST**：

$$
\xi_{j,t}^{\mathrm{emit},(i)}
=1+\zeta_{j,t}^{\mathrm{ST}}
\left(
p_{j,t}^{(i)}-\operatorname{sg}(p_{j,t}^{(i)})
\right),
\qquad i\in\mathcal A_{j,t},
$$

$$
\widehat g_{j,t}^{(i)}
=h^{\mathrm{in}}_{j,t}
+\xi_{j,t}^{\mathrm{emit},(i)}
\left(g_{j,t}^{(i)}-h^{\mathrm{in}}_{j,t}\right).
$$

\(\operatorname{sg}\) 表示 stop-gradient：前向值不变，反向梯度为零；ST 表示 straight-through（直通估计）。因此 EMIT-HST 前向恒有 \(\widehat g=g\)，但主任务梯度仍可通过 \(p\) 返回 selector。令 \(\mathcal L_{\mathrm{LM}}\) 表示第 6 节定义的语言模型主损失，则本次 Emit 的直接梯度是：

$$
\left.
\frac{\partial\mathcal L_{\mathrm{LM}}}
{\partial p_{j,t}^{(i)}}
\right|_{\mathrm{Emit}}
=\zeta_{j,t}^{\mathrm{ST}}
\left\langle
\frac{\partial\mathcal L_{\mathrm{LM}}}
{\partial\widehat g_{j,t}^{(i)}},
g_{j,t}^{(i)}-h^{\mathrm{in}}_{j,t}
\right\rangle.
$$

这里把 hard active set 视为常量，离散的 Top-1 / Top-K 成员选择本身仍不求导。Top-1 的首个设置取 \(\zeta^{\mathrm{ST}}=1\)；Top-K 可先取 \(\zeta^{\mathrm{ST}}=1/|\mathcal A|\) 控制每次选择的梯度尺度。该值只改变反向，必须写入实验设置。

下表中的 \(h_i\) 表示 receiver \(i\) 经输入 AggregatePort 得到的完整入口；在本节的单层样例中就是 \(h^{\mathrm{in}}_{j,t}\)。可对照的 `EmitPolicy` 为：

| Policy | 实际 Emit | selector 从主任务得到的梯度 |
| --- | --- | --- |
| **EMIT-HARD** | \(\widehat g_i=g_i\) | 不通过 Emit 返回 |
| **EMIT-HST** | 前向 \(\widehat g_i=g_i\) | 通过上面的 delta Hard-ST 返回 |
| **EMIT-SOFTP** | \(\widehat g_i=h_i+p_i(g_i-h_i)\) | 通过 soft \(p_i\) 返回，同时改变前向强度 |
| **EMIT-CUSTOM** | 由实验明确 | 由实验明确 |

每个 active node 只计算一次 \(\widehat g\)，再把同一消息沿全部固定出边发送；未激活节点不 Emit。

所有 AggregatePort 复用同一个 `MessageAggregate` 接口。令端口 \(a\) 当前实际收到的非空消息集合为 \(\mathcal M_{a,t}\)，其中 \(y_{k,t}\) 是来源 \(k\) 发来的完整 hidden，则：

$$
\operatorname{MessageAggregate}_a(\mathcal M_{a,t})
=\sum_{(k,y_{k,t})\in\mathcal M_{a,t}}
\alpha_{a,k,t}y_{k,t},
$$

$$
\alpha_{a,k,t}\ge0,
\qquad
\sum_{(k,y_{k,t})\in\mathcal M_{a,t}}
\alpha_{a,k,t}=1.
$$

首个设置 **AGG-MEAN** 对实际到达的消息取均值；**AGG-LEARNED** 可以用端口自己的轻量 \(\operatorname{MergeScore}_a\) 产生归一化权重。当前规范不把 sender 的 selector 概率再次交给 `MessageAggregate`，避免同一个 \(p\) 被重复使用。一个端口只收到一条消息时，聚合自然退化为恒等操作；没有消息时不产生输出，receiver 因而不 reached，而 `GraphOutputPort` 在有效执行中必须至少收到一条消息。

单层样例的输出 inbox 与 GraphBranch 输出为：

$$
\operatorname{Inbox}_{\mathrm{out},j,t}
=\left\{
(i,\widehat g_{j,t}^{(i)})
\mid i\in\mathcal A_{j,t}
\right\},
$$

$$
b_{\mathcal G,j,t}
=\operatorname{MessageAggregate}_{\mathrm{out}}
\left(\operatorname{Inbox}_{\mathrm{out},j,t}\right).
$$

Top-1 时输出聚合只收到一条消息，因此直接返回该消息。Top-K 使用 AGG-MEAN 时，由于各 node 共享入口 \(h\)，有：

$$
\frac1{|\mathcal A|}
\sum_{i\in\mathcal A}\widehat g_i
=h+
\frac1{|\mathcal A|}
\sum_{i\in\mathcal A}(\widehat g_i-h),
$$

所以不会重复加入公共输入。第 4 节的 receiver 多父输入与 GraphBranch 最终输出继续使用完全相同的 `MessageAggregate`；区别只在于聚合结果后面是否还接 receiver node。

第 1.3 节的 `BoundaryMerge` 仍只处理 GraphBranch 与 base 的 **RESIDUAL_ADD** 边界，不属于 `MessageAggregate`。对当前默认 node 模板，identity 初始化要求 \(\rho_i=0\) 且 FFN residual 为零；其他模板必须给出自己的 identity 条件。若所有实际消息都保持共同入口，AGG-MEAN 也返回该入口，进而得到 \(\Delta_{\mathcal G}=0\)。

### 3.6 状态生命周期

以下规则适用于本文所有 receiver nodes。每条独立序列都从空状态开始：EMA、GDN 和 SSM 状态置零，Attention 历史为空，历史激活计数清零。padding 等无效 Token 不计算 Update proposal、不 Observe，也不进入路由辅助 loss。

这里的 chunk 是一次前向接收的连续 Token 片段。同一逻辑序列跨 chunk 时继承状态值，不同序列之间清零；默认在每个 chunk 边界 detach，状态继续前传，但梯度只在当前 chunk 内传播。对同一有效前缀，整段 prefill、任意分块和逐 Token decode 应在数值误差范围内得到相同的逐 Token 输出与最终状态。

至此，单层特例的主干规范计算语义已经完整。若要继续理解多层拓扑，可以直接进入第 4 节；第 3.7 节只是可独立查阅的状态模块样例库。

### 3.7 可选参考：Receiver node 的状态模块样例

第 2.3、3.3 和 3.4 节中的 \(s\)、\(\operatorname{Update}\) 和 \(\operatorname{Read}^{\mathrm{sel}}\) 是所有 receiver nodes 共用的状态交互接口；\(\operatorname{Read}^{\mathrm{ffn}}\) 是当前默认 node 模板使用的较大状态读出。本节只展开状态模块样例，昂贵 FFN \(E\) 保持不变。下面的样例用于建立设计空间，不表示它们已经通过 TIDE 实验，也不预设哪一种必然最好。状态实现与 selector 时序是两个独立坐标：content-only 的 \(\operatorname{Read}^{\mathrm{sel}}\) 只读取当前本地消息，pre/post state 则额外读取对应时刻的状态；\(\operatorname{Read}^{\mathrm{ffn}}\) 不受这一选择影响。

#### 3.7.1 一览

| 样例 | 主要保留什么 | 典型消费者 | 主要特点 |
| --- | --- | --- | --- |
| **历史激活** | 激活次数、最近激活位置、概率或局部预算 | selector | 最轻量；记录控制历史，不直接保存内容语义 |
| **EMA（指数移动平均）** | 一个固定长度的低通内容摘要 | selector / FFN | 简单、稳定，但不同历史会持续混合 |
| **Gated DeltaNet（GDN）** | 固定大小的 key-value 关联矩阵 | selector / FFN | 可以按 query 关联读取，并按预测误差写入 |
| **Kimi Delta Attention（KDA）** | 带细粒度门控的 delta-rule 矩阵状态 | selector / FFN | delta-rule 家族的近期增强，门控更细但实现更复杂 |
| **SSM（state-space model）/ Mamba-2** | 固定大小的状态空间递归状态 | selector / FFN | 与 delta-rule 不同的成熟有界状态路线 |
| **Attention** | 完整历史、局部窗口或压缩后的 key/value | selector / FFN | 设计空间大；信息保留与状态/计算成本由具体实现决定 |

两类读出都在 receiver 局部完成：\(\operatorname{Read}^{\mathrm{sel}}\) 通常只输出低维投影、范数或历史统计，\(\operatorname{Read}^{\mathrm{ffn}}\) 则在内部完成必要的 output projection，并统一输出 hidden 维 residual。“典型消费者”只是常见用法，不是硬限制。

#### 3.7.2 历史激活

历史激活可以记录每个候选被选中的次数、距上次激活的 Token 数、soft probability 的移动平均或剩余局部预算、历史 selector 打分。本次选择只能在 selector 决策完成后写回，因此只影响以后 Token。若它只服务于 selector，则对应的 \(\rho_i=0\)。

#### 3.7.3 EMA（指数移动平均）

EMA\(D\) 把收到的内容压缩成一个长度为 \(D\) 的固定向量：

$$
s_{j,t}^{(i)}\in\mathbb R^D,
\qquad
o_{j,t}^{(i)}
=\tanh\!\left(W_i^{\mathrm{obs}}m_{j,t}^{(i)}+b_i^{\mathrm{obs}}\right).
$$

它对统一接口的实现为：

$$
\operatorname{Update}_i^{\mathrm{EMA}}(s_{j,t-1}^{(i)},m_{j,t}^{(i)})
=\lambda_i\odot s_{j,t-1}^{(i)}
+(1-\lambda_i)\odot o_{j,t}^{(i)},
$$

$$
\operatorname{Read}_i^{\mathrm{ffn,EMA}}(s_{j,t}^{(i)},m_{j,t}^{(i)})
=W_i^{\mathrm{out}}s_{j,t}^{(i)}.
$$

EMA 是最简单的内容记忆基线：新观察按 \(1-\lambda_i\) 写入，旧状态按 \(\lambda_i\) 保留。EMA128 就是 \(D=128\)。

#### 3.7.4 Gated DeltaNet 与 KDA

Gated DeltaNet（GDN）把同一个框架状态 \(s\) 实现为固定大小的关联矩阵：

$$
s_{j,t}^{(i)}\in\mathbb R^{K\times V}.
$$

这里先抽取 gated delta-rule 的核心状态语义，不默认复制完整开放模型 block 中的短卷积、输出门或其他外围结构；若实验加入这些部件，必须单独声明。

调用 \(\operatorname{Update}\) 或 \(\operatorname{Read}^{\mathrm{ffn}}\) 时，receiver \(i\) 从本地消息 \(m_{j,t}^{(i)}\) 生成归一化的 query/key、value 和写入门：

$$
q_{j,t}^{(i)}=N_q(W_i^q m_{j,t}^{(i)}),
\qquad
k_{j,t}^{(i)}=N_k(W_i^k m_{j,t}^{(i)}),
\qquad
\nu_{j,t}^{(i)}=W_i^\nu m_{j,t}^{(i)},
$$

$$
\eta_{j,t}^{(i)}
=\sigma\!\left((w_i^\eta)^\top m_{j,t}^{(i)}+b_i^\eta\right),
\qquad
\gamma_{j,t}^{(i)}
=\exp\!\left[
-\exp(\alpha_i)\,
\operatorname{softplus}\!\left((w_i^\gamma)^\top m_{j,t}^{(i)}+b_i^\gamma\right)
\right].
$$

其中 \(N_q,N_k\) 表示 query/key 的向量归一化，\(\gamma\) 控制旧状态保留量，\(\eta\) 控制本次误差写入量，\(\alpha_i\) 是可学习的衰减参数。

GDN 先衰减旧状态，再只写入当前 value 与已有预测之间的误差：

$$
s_{j,t,\mathrm{decay}}^{(i)}
=\gamma_{j,t}^{(i)}s_{j,t-1}^{(i)},
\qquad
e_{j,t}^{(i)}
=\nu_{j,t}^{(i)}
-(k_{j,t}^{(i)})^\top s_{j,t,\mathrm{decay}}^{(i)},
$$

$$
\operatorname{Update}_i^{\mathrm{GDN}}(s_{j,t-1}^{(i)},m_{j,t}^{(i)})
=s_{j,t,\mathrm{decay}}^{(i)}
+\eta_{j,t}^{(i)}k_{j,t}^{(i)}(e_{j,t}^{(i)})^\top,
$$

$$
\operatorname{Read}_i^{\mathrm{ffn,GDN}}(s_{j,t}^{(i)},m_{j,t}^{(i)})
=W_i^{\mathrm{out}}
\left[(q_{j,t}^{(i)})^\top s_{j,t}^{(i)}\right].
$$

因此 GDN 比 EMA 多了“按 key 写入、按 query 读取”的结构。它已经被开放权重的 [Qwen3-Next](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) 和 [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) 系列直接采用，是很强的现代参考点，但这不证明它对 TIDE receiver 必然最优。

[Kimi Linear](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base) 使用的 Kimi Delta Attention（KDA）同样为 delta rule 引入细粒度门控，并公开了训练权重与 chunk/recurrent kernel。它可以作为 GDN 之后的增强候选；代价是状态更新、参数匹配和 kernel 移植都更复杂，因此不必在第一轮同时实现。

#### 3.7.5 Attention 状态

Attention receiver 可以把实际 Observe 到的 key/value 作为状态 \(s\)，再用当前 query 执行普通 Attention。下面以保留最近 \(W\) 次 Observe 为例：

$$
q_{j,t}^{(i)}=N_q(W_i^q m_{j,t}^{(i)}),
\qquad
k_{j,t}^{(i)}=N_k(W_i^k m_{j,t}^{(i)}),
\qquad
\nu_{j,t}^{(i)}=W_i^\nu m_{j,t}^{(i)}.
$$

$$
\mathcal H_{j,t}^{(i)}
:=\{\tau\le t\mid i\in\mathcal O_{j,\tau}\},
\qquad
\mathcal W_{j,t}^{(i)}
:=\operatorname{Last}_W(\mathcal H_{j,t}^{(i)}),
$$

\(\operatorname{Last}_W\) 表示按时间保留集合中最近的至多 \(W\) 个位置。

$$
s_{j,t}^{(i)}
=\left((k_{j,\tau}^{(i)},\nu_{j,\tau}^{(i)})\right)_{
\tau\in\mathcal W_{j,t}^{(i)}}.
$$

令 \(\mathbf K_{j,t}^{(i)}\) 和 \(\mathbf V_{j,t}^{(i)}\) 分别表示状态中按时间堆叠的 key 和 value，key 维度记为 \(d_k\)，则：

$$
\operatorname{Read}_i^{\mathrm{ffn,Attn}}(s_{j,t}^{(i)},m_{j,t}^{(i)})
=W_i^{\mathrm{out}}
\left[
\operatorname{softmax}\!\left(
\frac{q_{j,t}^{(i)}\mathbf K_{j,t}^{(i)\top}}{\sqrt {d_k}}
\right)\mathbf V_{j,t}^{(i)}
\right].
$$

对应的 \(\operatorname{Read}^{\mathrm{sel}}\) 可以只在 receiver 局部输出该向量的范数和少量历史激活统计。

实际实现也可以保留完整历史，或使用分层/稀疏选择、压缩 key/value、固定记忆槽位。完整历史的状态和读取成本随上下文增长；其他方案成本更可控，但会引入不同的信息选择。实验应如实记录实际状态量、读取成本和被保留的历史范围。

#### 3.7.6 其他有界状态路线与当前定位

SSM（state-space model）/ Mamba-2 是另一类重要的固定状态候选，开放权重的 [Falcon-H1](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Base) 已采用 Transformer 与 Mamba 的混合结构；RWKV-7、Lightning Attention 等也提供了可参考的递归或线性注意力状态。它们证明“有界 recurrent state”有多条成熟路线，但不必全部进入首轮 TIDE 实现。

当前更合适的定位是：历史激活用于最轻量的 selector 控制，EMA 作为简单内容基线，GDN 作为第一种先进关联记忆锚点，Attention 保留为可按预算选择的宽泛设计族；KDA 和 Mamba / structured state-space duality（SSD）则是增强或跨家族候选。这只是帮助建立全局观，不是固定实验顺序。维度和状态量必须在名称中明确，例如 **GDN-K32-V32** 有 \(32\times32=1024\) 个状态标量，不能与 EMA128 当作等状态量对照。

## 4. HB-Lattice：多层固定波前

**HB-Lattice** 是本文对一种受限多层拓扑的专名，不依赖外部文档定义。它完整复用第 2 节的两个边界端口、AggregatePort / `MessageAggregate`、receiver node / template、selector、propagation profile、`EmitPolicy` 和 `BalancePolicy`，只增加描述固定边排列的 Plan、多个 Lines 与逐 Line 波前执行。

### 4.1 Line、region 与波前

HB-Lattice 把 receiver nodes 静态放进有序的 **Lines** \(L_0,L_1,\ldots,L_D\)。Line 是 GraphBranch 内部的逻辑执行步：对同一个 Token，\(L_d\) 的接收、选择、状态提交、完整计算和发送全部结算后，才开始 \(L_{d+1}\)。每个 Line 的节点数可以先在扩展期增大，在平台期保持不变，再在收拢期缩小；扩展、平台和收拢统称 Line 的 **phase**。

每个 Line 又被静态划分成若干互不重叠的 **selector regions**。一个 region 只有一个 selector；它只在本 region 当前 reached 的 nodes 中选择 active nodes。一个 receiver node 只属于一个 Line 和一个 region，但可以有多个固定 parents 和 children。

selector region 是控制面的静态分组，不是数据图中的节点。固定边决定消息能到达哪些 AggregatePorts，AggregatePort 决定哪些 receivers reached，selector 才在这些候选中决定哪些 active。一个 receiver 可以沿多条固定出边发送同一消息，但不需要额外的“发散点”。

一个最小心智样例是：

~~~text
GraphInputPort
L0: {0}          reached 时 forced-active
L1: {1,2}        一个 selector region
L2: {3,4}        一个 selector region
GraphOutputPort  只聚合消息，不是 receiver node

edges: Input→0；0→1, 0→2；1→3, 1→4；2→3, 2→4；3→Output, 4→Output
~~~

对一个 Token，`GraphInputPort` 先把 \(h^{\mathrm{in}}\) 送到节点 0；节点 0 结算后沿两条固定边发送同一消息。L1 的 selector 在 reached 的 1、2 中选择，实际 Emit 的消息决定 L2 中哪些 nodes reached；节点 3、4 都有两个固定 parents，各自在自己的 AggregatePort 聚合父消息。L2 结算后，`GraphOutputPort` 再用相同的 `MessageAggregate` 聚合最终消息并返回 \(b_{\mathcal G}\)。

端口、节点、边、Lines 和 regions 的完整静态列表称为 **`HBLatticePlan`**，后文简称 **Plan**。数学上 HB-Lattice 是分层 DAG，但本规范的执行器只处理下面的受限结构：

- GraphBranch 只有一个 `GraphInputPort` 和一个 `GraphOutputPort`，二者都不是 receiver node；
- 节点被静态分配到有序 Line \(L_0,L_1,\ldots,L_D\)，每个节点只属于一个 Line；
- Line 内没有消息依赖；除输入、输出边外，普通边只连接相邻 Line；
- 递归扩展节点可以沿显式声明的镜像直通边跳过中间 Lines，把消息送到对应的收拢节点；
- 平台期各 Line 使用相同的空间坐标集合，每对相邻平台 Line 的连接可以分别指定；
- 每个 Line 被划分为固定、不重叠的 selector regions，每个节点只由所属 region 的 selector 决定是否 active；
- 每个 receiver 前有一个输入 AggregatePort，`GraphOutputPort` 本身是终端 AggregatePort；
- 同一 Token 在一个端口只聚合一次消息，在一个 receiver node 只更新一次状态并至多执行一次昂贵计算。

因此近期实现不需要任意拓扑排序、同层依赖、异步 event queue、有环执行或一般 DAG 接口。

Plan 中的计算 nodes 都是第 2 节定义的 receiver nodes；AggregatePorts 与 GraphBranch 边界端口属于另一种语义类型。一个 Token 实际激活的 nodes 和实际发送消息的固定边共同形成 **active subgraph**。fan-in 由目标 AggregatePort 处理，fan-out 只是 sender 的固定出边数量，不另外引入发散或分支计算节点。

receiver 的输入 AggregatePort 由该 receiver 及其固定入边唯一确定，Plan 不把它重复列成计算 node；`GraphInputPort` 与 `GraphOutputPort` 则作为边界端点显式保存。

单层特例与 HB-Lattice 的对应关系如下：

| 位置 | 单层特例 | HB-Lattice |
| --- | --- | --- |
| GraphBranch 输入 | `GraphInputPort` 沿固定边发送 \(h^{\mathrm{in}}_{j,t}\) | 完全相同 |
| receiver node 入口 | 每个输入 AggregatePort 只收到同一个 \(h^{\mathrm{in}}_{j,t}\) | 每个 reached node 的 AggregatePort 可收到不同父消息并得到 \(h_{v,t}\) |
| 当前候选 | 固定的 \(R\) 个 receiver nodes | 当前 region 中实际 reached 的 receiver nodes |
| selector | 一个 region 在共享入口的 reached nodes 中选择 | 每个 Line 的各 regions 分别在本地 reached nodes 中选择 |
| 状态与节点计算 | 第 2.3、3.3、3.4 节的契约与模板 | 复用相同契约，只替换节点入口 |
| receiver node 输出 | 经 `EmitPolicy` 发到 `GraphOutputPort` | 经 `EmitPolicy` 沿固定边发到下游端口 |
| 消息聚合 | 输出 AggregatePort 使用 `MessageAggregate` | receiver 输入端口与输出端口使用同一接口 |
| GraphBranch 出口 | `GraphOutputPort` 聚合后得到 \(b_{\mathcal G,j,t}\) | 完全相同 |

本节使用以下局部符号：

| 符号 | 含义 |
| --- | --- |
| \(d\) | GraphBranch 内部的波前 Line 编号 |
| \(r\) | 一个 Line 内的 selector region 编号 |
| \(v\) | 当前 receiver node；每个 node 只属于一个 Line |
| \(w\) | \(v\) 的一个固定 parent node |
| \(Q\) | 平台期每个 Line 共享的空间坐标集合 |
| \(\operatorname{Inbox}_{v,t}\) | 节点 \(v\) 在当前 Token 实际收到的父消息集合 |
| \(\operatorname{Inbox}_{\mathrm{out},t}\) | `GraphOutputPort` 当前实际收到的最终消息集合 |
| \(q_{v,t}\) | 节点 \(v\) 是否收到至少一条父消息的 reached 标记 |
| \(\mathcal C_{d,r,t}\) | region \((d,r)\) 当前实际 reached 的候选集合 |
| \(h_{v,t}\) | 节点 \(v\) 聚合 inbox 后的完整入口 hidden |
| \(s_{v,t}\) | 节点 \(v\) 的 receiver 私有状态；site 下标在本节省略 |
| \(\mathcal A_{d,r,t}\) | region \((d,r)\) 当前选出的 active 节点集合 |
| \(g_{v,t}\) | active receiver node 完成计算、尚未发送的完整 hidden |
| \(\widehat g_{v,t}\) | active receiver node 经 sender 输出策略实际发送的完整 hidden |

### 4.2 两层拓扑接口

第一层是执行层，由规范化的 Plan、`HBLatticeExecutionConfig` 和逐 Line 运行二者的 `WavefrontExecutor` 组成。Plan 只回答“谁与谁连接、谁由哪个 region 选择”，至少完整列出：

~~~text
HBLatticePlan
├── GraphInputPort / GraphOutputPort：唯一边界端口及其固定边
├── Lines：每层的 phase、节点、坐标和 selector regions
├── adjacent edges：扩展、平台和收拢的相邻 Line 边
├── mirror map / edges：扩展与收拢节点的对应关系及逐节点直通开关
├── forced-active 节点：reached 时必定 active
└── edge class：input / tree / local / shortcut / mirror / output
~~~

其中 input / output 表示 GraphBranch 边界端口的固定边，tree 表示扩展或收拢树边，local / shortcut 分别表示平台期局部边与长程边，mirror 表示跨中间 Lines 的镜像直通边。

`HBLatticeExecutionConfig` 则回答 reached 节点怎样计算。除第 2 节已经定义的 propagation profile、`ReceiverNodeTemplate`、state 和 selector 时序外，它还配置逐 region 最多激活数 \(K^{\max}\) 与下面四个外部接口：

| 接口 | 职责 |
| --- | --- |
| `Selector` | 在一个 region 当前 reached 的 nodes 中产生概率与 active set |
| `MessageAggregate` | 在任意 AggregatePort 把实际收到的消息合成一个完整 hidden |
| `EmitPolicy` | 把 active node 的完整输出变成发给固定 children 的消息 |
| `BalancePolicy` | 只在训练时根据选择事件产生辅助均衡 loss |

这些接口可以全局统一，也可以按端口、node 或 region 映射到不同配置；第 4.3 至 4.5 节按执行顺序给出精确定义。

扩展树和收拢树可以采用不均匀但有界的叉数；平台期每对 Line 的邻接也可以不同。`WavefrontExecutor` 只消费已展开的 Plan 和执行配置，不负责猜测树形、空间邻接或镜像关系。

Plan 载入时必须检查 Line 顺序、region 唯一归属、边类型与端点、`GraphInputPort` 到 `GraphOutputPort` 的静态可达性、镜像对应关系，以及声明的 fan-in/fan-out 和 region-size 上界。

第二层由一个或多个 `TopologyBuilder` 组成：

~~~text
TopologyBuilder(config) → HBLatticePlan
~~~

builder 可以生成规则树、逐坐标混合、重复空间 Graph 或其他 HB-Lattice 模板。每个正式实验同时保存最终 Plan 的规范化内容与哈希、完整执行配置，以及 builder 名称、版本和配置；完整计算语义由 Plan 与执行配置共同决定，不能只看生成器名称。

### 4.3 Inbox、reached 与 MessageAggregate

设节点 \(v\in L_d\) 的固定父节点集合为 \(P(v)\)，并用 \(\mathrm{in}\) 表示 `GraphInputPort`。当前 Token 上，只有已经激活、完成计算并 Emit 的父节点才会发送完整 hidden。节点 \(v\) 的实际 inbox 为：

$$
\operatorname{Inbox}_{v,t}
=\left\{
(w,\widehat g_{w,t})
\mid w\in P(v),\ w\text{ 在 Token }t\text{ 已 Emit}
\right\}
\cup
\left\{
(\mathrm{in},h^{\mathrm{in}}_{j,t})
\mid \mathrm{in}\to v\text{ 是固定边}
\right\}.
$$

因此，任何与 `GraphInputPort` 直接相连的 receiver 都能收到外部入口消息，不需要一个专门的入口 receiver。镜像直通消息可以在较早的 Line 产生，但只保存在目标 inbox 中；目标 Line 到来、所有可能父节点都已经结算后，才把“未到达”和“尚未到达”区分开。令：

$$
q_{v,t}=\mathbf 1[\operatorname{Inbox}_{v,t}\ne\varnothing]
$$

表示节点是否 reached。若 \(q_{v,t}=0\)，节点不参加当前选择，不更新状态，也不输出；若收到一条或多条消息，其输入 AggregatePort 就先执行一次与消息到达顺序无关的 `MessageAggregate`：

$$
h_{v,t}
=\operatorname{MessageAggregate}_v(\operatorname{Inbox}_{v,t}).
$$

首个设置继续使用第 3.5 节的 **AGG-MEAN**：对实际到达的消息均匀平均，并把到达数量或有界 source-presence mask 作为额外轻量信息。归一化聚合不会因消息数量变化而重复放大公共 hidden，也能在各节点初始化为 identity 时保持 identity。以后若使用 **AGG-LEARNED**，由当前 AggregatePort 自己的 \(\operatorname{MergeScore}\) 产生归一化权重。

`MessageAggregate` 不使用各 sender 的 selector 概率：不同消息的概率可能来自不同 regions，数值不能直接比较；只有一条消息时，归一化权重又恒为 1，无法给上游 selector 提供梯度。selector 的主任务梯度统一放在 sender 的 `EmitPolicy`。

### 4.4 Region selector 与节点计算

节点 \(v\) 用自己的入口归一化产生本地消息：

$$
m_{v,t}=N_{R,v}(h_{v,t}),
$$

再按第 3.3 节的 content-only、pre-update 或 post-update 时序产生轻量 \(\operatorname{Read}^{\mathrm{sel}}\)。同一 region 的节点可能拥有不同的 \(h_{v,t}\)，因此一般 HB-Lattice 不要求存在单层特例中由共同入口产生的单一 \(\mu\)；region selector 只联合处理各 reached nodes 发来的轻量读出和 presence 信息。

对 region \(\mathcal R_{d,r}\subseteq L_d\)，当前候选集合为：

$$
\mathcal C_{d,r,t}
=\{v\in\mathcal R_{d,r}\mid q_{v,t}=1\}.
$$

region selector 可以联合处理本 region 的轻量读出，但只为 reached nodes 产生有效概率。令 \(\tau\in\{\mathrm{content},\mathrm{pre},\mathrm{post}\}\) 表示本实验选择的 selector 时序。对未 reached 节点，约定其读出以零占位并由 \(q_{v',t}=0\) mask 掉，不在 receiver 端实际计算：

$$
a_{v,t}
=\left[
\operatorname{Score}_{d,r}
\left((r_{v',t,\tau}^{\mathrm{sel}},q_{v',t})_{v'\in\mathcal R_{d,r}}\right)
\right]_v,
\qquad v\in\mathcal C_{d,r,t},
$$

$$
(p_{v,t})_{v\in\mathcal C_{d,r,t}}
=\operatorname{softmax}
\left((a_{v,t})_{v\in\mathcal C_{d,r,t}}\right).
$$

selector 返回当前候选集合 \(\mathcal C_{d,r,t}\)、soft probabilities 和 hard active set \(\mathcal A_{d,r,t}\)。不同 regions 的 selector 参数默认不共享；若实验共享，必须在执行配置中声明。

若 \(\mathcal C_{d,r,t}=\varnothing\)，该 region 选择空集；否则必须满足：

$$
1\le |\mathcal A_{d,r,t}|
\le \min(K_{d,r}^{\max},|\mathcal C_{d,r,t}|).
$$

全激活只是令所有 reached nodes 都进入 \(\mathcal A_{d,r,t}\) 的特例。任何需要固定通过的 receiver node 都可以在 Plan 中声明为 forced-active，含义是“只要 reached 就必定 active”，不能让未 reached 节点凭空激活；GraphBranch 的输入、输出端口不参加选择。

传播 profile 在 HB-Lattice 中统一解释为：

- **N**：无状态，只有 active nodes 执行完整节点计算；
- **SD**：只有 active nodes commit / Observe 并执行完整节点计算；
- **BO**：全部 reached nodes commit / Observe，只有 active nodes 执行较大读出和昂贵计算。

因此 BO + post-update state 的顺序是“全部 reached nodes 聚合并计算 Update proposal → 各 region 选择 → 全部 reached nodes commit / Observe → active nodes 执行”；SD 仍只自然兼容 content-only 和 pre-update state。一个节点即使收到多个父消息，也只在其输入端口完成一次 `MessageAggregate` 后计算一次 proposal。

active 节点沿用第 2.3、3.4 节的完整输出契约，只把单层特例的共同入口 \(h^{\mathrm{in}}_{j,t}\) 换成本节点入口 \(h_{v,t}\)：

$$
g_{v,t}
=\operatorname{NodeCompute}_{v}(h_{v,t},m_{v,t},s_{v,t}).
$$

在当前默认的 Pre-Norm 双 residual 模板中，有状态节点在状态提交后读取：

$$
\rho_{v,t}
=\operatorname{Read}_{v}^{\mathrm{ffn}}(s_{v,t},m_{v,t}),
$$

无状态的 N 令 \(\rho_{v,t}=0\)。随后计算：

$$
u_{v,t}=h_{v,t}+\rho_{v,t},
$$

$$
g_{v,t}
=u_{v,t}+E_v\!\left(N_{F,v}(u_{v,t})\right).
$$

这里的 \(g_{v,t}\) 是尚未应用 `EmitPolicy` 的完整节点输出。其他 `ReceiverNodeTemplate` 可以采用不同内部公式，但必须返回相同形状的完整 hidden。

### 4.5 Emit、输出聚合与 Line barrier

#### 4.5.1 Receiver Emit 与 GraphOutputPort

每个 active receiver 复用第 3.5 节的 `EmitPolicy`：

$$
\widehat g_{v,t}
=\operatorname{EmitPolicy}_v
\left(h_{v,t},g_{v,t},p_{v,t}\right),
\qquad v\in\mathcal A_{d,r,t}.
$$

首个设置使用 **EMIT-HST**，并按 region 记录 \(\zeta_{d,r,t}^{\mathrm{ST}}\)；EMIT-HARD、EMIT-SOFTP 与 EMIT-CUSTOM 的前向和梯度语义也完全沿用第 3.5 节。一个 active receiver 只产生一次 \(\widehat g_{v,t}\)，再把同一消息复制到全部固定出边。未激活节点不执行昂贵计算，也不 Emit。

用 \(\mathrm{out}\) 表示 `GraphOutputPort`。与它相连的 receiver 消息形成最终 inbox：

$$
\operatorname{Inbox}_{\mathrm{out},t}
=\left\{
(v,\widehat g_{v,t})
\mid v\to\mathrm{out}\text{ 是固定边，且 }v\text{ 已 Emit}
\right\}.
$$

全部 Lines 结算后，输出 AggregatePort 使用同一个 `MessageAggregate` 接口：

$$
b_{\mathcal G,j,t}
=\operatorname{MessageAggregate}_{\mathrm{out}}
\left(\operatorname{Inbox}_{\mathrm{out},t}\right).
$$

`GraphOutputPort` 不更新状态、不参加 selector，也不执行 `NodeCompute`；它只把最终消息聚合成 GraphBranch 输出。若某个拓扑需要在输出前再做一次 receiver 计算，就应显式放置一个 receiver node，并让它继续 Emit 到输出端口。

#### 4.5.2 Line 结算、训练事件与接口边界

训练时，每个 region 把一次选择的 \((\mathcal C,p,\mathcal A)\) 连同 site、Line、region、序列和 Token 标识交给 `BalancePolicy`；这些记录不改变推理前向。第 6.2 节定义首个 **BAL-AVAIL-SOFT**，即按当前可达候选计算的 soft 均衡目标。

至此，一个 Line 的数据流可以概括为：

~~~text
固定入边 → AggregatePort / MessageAggregate → receiver 本地轻量阶段
      → selector 时序与 profile 协同完成 Update proposal、选择与状态提交
      → active receiver node compute
      → EmitPolicy → 固定出边 → 下游 AggregatePorts

(C, p, A) 与位置标识 → BalancePolicy（仅训练）
~~~

BO + post-update state 的 Update proposal 计算位于 selector 之前，状态提交位于选择之后；其他 profile 和 selector 时序按第 3.3、3.4 节执行。只有当前 Line 的 inbox、选择、状态提交、完整节点计算和 Emit 全部结算后，才开始下一个 Line。实现可以做等价的批处理或流水线，但不得改变这套规范计算语义。全部 Lines 结算后才执行输出 AggregatePort。

`MessageAggregate` 是所有 AggregatePorts 共用的消息聚合接口；`EmitPolicy` 是 sender 端的前向与主任务梯度接口；`BalancePolicy` 只产生训练期辅助 loss；`BoundaryMerge` 只处理 GraphBranch 与 base 的边界。它们职责互不替代。selector 概率由 `EmitPolicy` 使用后，不在 `MessageAggregate` 中再次使用；若把 \(p\) 写入历史状态，必须单独记录是否 stop-gradient。

当节点初始化为严格 identity 时，\(g_v-h_v=0\)，EMIT-HST 的主任务 selector 梯度也为零。首轮训练可以让 balance loss 先提供路由牵引，并使用全激活或较大的 \(K^{\max}\) 做短 warmup；节点离开 identity 后，主任务梯度才会逐渐进入 selector。

这四个接口分别配置，首个 HB-Lattice 实现建议采用：

| 接口 | 首个设置 | 可替换方向 |
| --- | --- | --- |
| `Selector` | reached candidates 上的 masked Top-1/Top-K | 全激活或其他局部选择 |
| `EmitPolicy` | EMIT-HST | EMIT-HARD、EMIT-SOFTP 或后续 surrogate |
| `MessageAggregate` | AGG-MEAN | AGG-LEARNED 或按端口配置 |
| `BalancePolicy` | BAL-AVAIL-SOFT | BAL-NONE 或另行定义的统计目标 |

它们是可独立替换的接口，不是整个框架唯一允许的实现。

### 4.6 两类标准 TopologyBuilder

第一类 builder 使用 \(B\) 叉扩展、逐坐标平台混合和镜像收拢。设扩展深度为 \(D_{\mathrm{up}}\)，平台 hop 数为 \(P\)，最大宽度为 \(W=B^{D_{\mathrm{up}}}\)，Line 宽度为：

$$
1,B,\ldots,B^{D_{\mathrm{up}}},
\underbrace{B^{D_{\mathrm{up}}},\ldots,B^{D_{\mathrm{up}}}}_{P\text{ 个额外 Line}},
B^{D_{\mathrm{up}}-1},\ldots,B,1.
$$

例如 \(B=2,D_{\mathrm{up}}=2,P=2\) 时，可以用二进制空间地址写成：

~~~text
L0:  ε
L1:  0, 1
L2:  00, 01, 10, 11
L3:  00', 01', 10', 11'       # 混合第一位
L4:  00'', 01'', 10'', 11''   # 混合第二位
L5:  0', 1'
L6:  ε'
~~~

扩展时追加一位；第一个平台 hop 令 \((a,b)\to(0,b),(1,b)\)，第二个 hop 令 \((a,b)\to(a,0),(a,1)\)；收拢时删除一位。builder 另生成 `GraphInputPort` 到 \(L_0\) 以及 \(L_6\) 到 `GraphOutputPort` 的边。\(L_0\to L_6\)、\(L_1\to L_5\) 的对应节点可以逐个开启镜像直通。样例可以把 \(L_1\) 和 \(L_5\) 各划成一个二节点 region，把每个宽度为 4 的 Line 划成 \(\{00,01\}\)、\(\{10,11\}\) 两个 regions；首尾两个 singleton receiver nodes 强制激活。相同地址出现在不同 Line 时仍表示不同节点，默认不共享参数或状态。

对称 \(B\) 叉模板中，扩展节点最多连接 \(B\) 个 children 和一个镜像节点，平台节点的入度、出度为 \(B\)，收拢节点最多接收 \(B\) 个深层 parents 和一个镜像 parent；这些上界不随平台宽度或长度增长。

第二类 builder 使用统一空间有向 Graph。设平台坐标集合为 \(Q\)，空间图为 \(G_{\mathrm{space}}=(Q,E_{\mathrm{space}})\)，则每个相邻平台 hop 生成：

$$
E_d
=\left\{
((d,q),(d+1,q'))
\mid(q,q')\in E_{\mathrm{space}}
\right\}.
$$

同一空间图可以在所有平台 hop 重复，也可以由 builder 为不同 hop 产生不同 \(E_d\)。即使 \(G_{\mathrm{space}}\) 自身有环，逐 Line 展开后的 HB-Lattice 仍然无环。

一个空间图可以同时包含固定大小的局部邻域和每节点固定数量的长程 shortcut。长程边宜采用置换或其他入度、出度同时有界的规则，避免宽度增长时形成高入度枢纽。平台 shortcut 只跨一个逻辑 Line、但跨越较远空间坐标；镜像直通则跨越多个逻辑 Line。两类边必须分别标记、记录成本并支持独立消融。

## 5. Dense 与标准 mixture-of-experts（MoE）基线

### 5.1 DENSE

DENSE 使用原 block：

$$
y=v
=u+F_\ell(N_F(u)).
$$

### 5.2 MOE

M8 是每个 site 有 8 个 experts 的简写；正式名称分别用 **MOE-R8** 和 **I** 表示 expert 数与 site 数。它把每个 expert 初始化为原 dense MLP 的副本，每个有效 Token 只执行一个 expert，不设 capacity、不丢 Token、也不 reroute。这里 capacity 指一个 expert 在当前 batch 最多接收多少 Token；本设置不设上限，因此被选 Token 不会因 expert 过载而跳过，也不会改送其他 expert。

对插在 block \(\ell=\ell(j)\) 的 MoE site \(j\)，计算为：

$$
m_{j,t}=N_F(u_{\ell,t}).
$$

$$
a_{j,t}=W_{\mathrm{moe}}m_{j,t},
\qquad
p_{j,t}=\operatorname{softmax}(a_{j,t}),
$$

$$
c_{j,t}=\arg\max_i p_{j,t}^{(i)}.
$$

$$
z_{j,t}^{(c_{j,t})}=m_{j,t}.
$$

$$
y_{\ell,t}
=u_{\ell,t}
+E_{j,c_{j,t}}\!\left(z_{j,t}^{(c_{j,t})}\right).
$$

这里的 \(m,a,p,c,z,E\) 与第 3 节使用同一组角色符号。M8 直接合并被选 expert 的输出，不乘 soft 概率；它没有 receiver 私有状态。

它与 PARMLP 处于相同的 block 接口，但语义不同：

- MOE 用一个 routed expert 替换原 dense MLP；
- PARMLP 保留原 dense MLP，再增加一个并列 GraphBranch residual。

当 PARMLP 的 GraphBranch 采用第 3 节的单层特例时，也可以直观地看作一种 shared-expert MoE：原 dense MLP 是 always-on shared expert，receiver nodes 是 routed experts。

## 6. 实际训练时的损失函数

令 \(\mathcal T\) 表示一个 micro-batch 中所有有效目标 Token 的 \((b,t)\) 集合，\(N_T=|\mathcal T|\)；\(w_{b,t}\) 是目标 Token，\(P_\theta(w_{b,t}\mid w_{b,<t})\) 是模型给它的条件概率。自回归语言模型损失为：

$$
\mathcal L_{\mathrm{LM}}
=-\frac{1}{N_T}
\sum_{(b,t)\in\mathcal T}
\log P_\theta(w_{b,t}\mid w_{b,<t}).
$$

路由辅助项使用的 Token 集合略有不同。当前单层特例中，每个 site 的 selector 都处理同一个集合 \(\mathcal V\)；标准 MoE 中，对应的是每个 site 的 router。\(\mathcal V\) 包含 attention mask 标记为有效、实际经过相应选择模块的全部 \((b,t)\) 位置，\(N_V=|\mathcal V|\)。它与 receiver node 或 expert \(i\) 无关，不是候选 \(i\) 实际被选中的 Token 集。balance loss 不要求单个 Token 均匀选择所有候选，而是避免整个 micro-batch 长期集中到少数 nodes 或 experts。令 \(\mathcal I\) 表示所有 routed sites，\(I=|\mathcal I|\)。

每个 site 独立计算 balance loss，再在 sites 间等权平均。统计范围是当前 micro-batch；梯度累积只累积各 micro-batch 的梯度，不预先把多个 micro-batches 合并成 global-batch balance loss。

### 6.1 单层特例中 N、SD、BO 的 receiver balance loss

对 site \(j\) 的 \(R\) 个 receivers，平均 softmax 概率为：

$$
\bar p_{j,i}
=\frac{1}{N_V}
\sum_{(b,t)\in\mathcal V}p_{j,b,t}^{(i)}.
$$

当前单层结构的 N、SD、BO 共同使用；它也是 **BAL-AVAIL-SOFT** 在全部候选始终 reached 时的特例：

$$
\mathcal L_{\mathrm{bal}}^{\mathrm{receiver}}
=\frac{1}{I}
\sum_{j\in\mathcal I}
\sum_{i=0}^{R-1}
\left(\bar p_{j,i}-\frac1R\right)^2.
$$

它约束的是平均 soft 概率，不直接约束 \(\operatorname{TopKIndex}\) 后各 receiver 真正执行了多少次；Top-1 时，后者就是 \(\arg\max\) 的选择次数。因此它鼓励均衡，但不能严格保证 hard active counts 均衡。

N、SD、BO 的实际反向传播目标都是：

$$
\boxed{
\mathcal L_{\mathrm{train}}^{\mathrm{N/SD/BO}}
=\mathcal L_{\mathrm{LM}}
+\omega_{\mathrm{receiver}}\mathcal L_{\mathrm{bal}}^{\mathrm{receiver}}
}.
$$

\(\omega_{\mathrm{receiver}}\ge0\) 由实验设置记录；本文定义的单层 receiver 目标不含第 6.3 节的 MoE router z-loss。

### 6.2 HB-Lattice 的 region balance loss

HB-Lattice 中每个 region 只处理实际 reached 的节点。对 site \(j\)、Line \(d\)、region \(r\)，固定节点集合记为 \(\mathcal R_{j,d,r}\)，并令：

$$
\mathcal V_{j,d,r}
=\left\{
(b,t)\mid\mathcal C_{j,d,r,b,t}\ne\varnothing
\right\},
\qquad
N_{j,d,r}=|\mathcal V_{j,d,r}|
$$

表示该 region 在当前 micro-batch 中真正发生选择的 Token 事件。selector 只在 \(\mathcal C_{j,d,r,b,t}\) 内做 masked softmax；未 reached 节点不进入当前候选集合，也不能被 balance loss 当作本次本应选择的候选。

这里的 \(\mathcal C_{j,d,r,b,t}\) 就是第 4.4 节的 \(\mathcal C_{d,r,t}\) 补回 site \(j\) 和序列 \(b\) 下标。

首个 `BalancePolicy` 使用 **BAL-AVAIL-SOFT**。对 \(N_{j,d,r}>0\) 的 region，约定未 reached 时 \(p_{j,d,r,b,t}^{(v)}=0\)，并定义节点 \(v\) 实际得到的平均 soft mass：

$$
\bar p_{j,d,r,v}
=\frac{1}{N_{j,d,r}}
\sum_{(b,t)\in\mathcal V_{j,d,r}}
\mathbf 1[v\in\mathcal C_{j,d,r,b,t}]
p_{j,d,r,b,t}^{(v)}.
$$

在相同可达性（availability）条件下，若每次都在当前 reached candidates 中均匀选择，节点 \(v\) 应得到的基准 mass 为：

$$
\bar p_{j,d,r,v}^{\mathrm{avail}}
=\frac{1}{N_{j,d,r}}
\sum_{(b,t)\in\mathcal V_{j,d,r}}
\frac{
\mathbf 1[v\in\mathcal C_{j,d,r,b,t}]
}{
|\mathcal C_{j,d,r,b,t}|
}.
$$

region 的 loss 为：

$$
\mathcal L_{\mathrm{bal},j,d,r}^{\mathrm{avail}}
=
\sum_{v\in\mathcal R_{j,d,r}}
\left(
\bar p_{j,d,r,v}
-\bar p_{j,d,r,v}^{\mathrm{avail}}
\right)^2.
$$

令 \(\mathcal Z\) 表示当前 micro-batch 中至少出现过一次 \(|\mathcal C_{j,d,r,b,t}|\ge2\) 的 region 实例，则：

$$
\mathcal L_{\mathrm{bal}}^{\mathrm{HB}}
=\frac{1}{|\mathcal Z|}
\sum_{(j,d,r)\in\mathcal Z}
\mathcal L_{\mathrm{bal},j,d,r}^{\mathrm{avail}},
$$

$$
\boxed{
\mathcal L_{\mathrm{train}}^{\mathrm{HB}}
=\mathcal L_{\mathrm{LM}}
+\omega_{\mathrm{HB}}\mathcal L_{\mathrm{bal}}^{\mathrm{HB}}
}.
$$

\(\omega_{\mathrm{HB}}\ge0\) 由实验设置记录；若 \(\mathcal Z=\varnothing\)，约定 \(\mathcal L_{\mathrm{bal}}^{\mathrm{HB}}=0\)。这个 reduction 先在每个 region 内计算，再对本 micro-batch 中真正出现过竞争选择的 regions 等权平均；forced-active 或始终只有一个候选的 singleton region 不稀释 loss。

在这个 policy 中，reached mask、\(\mathcal C\)、\(\bar p^{\mathrm{avail}}\) 和 hard active set 都视为 stop-gradient；balance 梯度只通过当前 region 的 \(p\) 返回 selector。

这个目标只比较“在同样已经 reached 的候选范围内，selector 是否长期偏向某些节点”：

- 若所有候选始终 reached，则 \(\bar p_v^{\mathrm{avail}}=1/R\)，退化为第 6.1 节的单层目标；
- 若某次只有一个候选 reached，该事件的实际 \(p\) 与均匀基准都为 1，不会产生无法完成的均衡要求；
- 从未 reached 的节点在 \(\bar p\) 和 \(\bar p^{\mathrm{avail}}\) 中都为 0。

例如一个 region 只有 A、B 两个节点：第一个 Token 只 reached A，第二个 Token 同时 reached A、B，则两个事件的均匀基准分别是 \((1,0)\) 和 \((1/2,1/2)\)，micro-batch 基准为 \(\bar p^{\mathrm{avail}}=(3/4,1/4)\)，而不是强行要求 \((1/2,1/2)\)。

它在 micro-batch 的平均值上约束分布，不强迫每个 Token 的 selector 概率都均匀，因此仍允许按内容形成专业化。它也不负责修复上游路由或 topology 造成的 reach starvation；那是独立问题。

同时记录 hard active share：

$$
\bar f_{j,d,r,v}
=\frac{1}{N_{j,d,r}}
\sum_{(b,t)\in\mathcal V_{j,d,r}}
\frac{
\mathbf 1[v\in\mathcal A_{j,d,r,b,t}]
}{
|\mathcal A_{j,d,r,b,t}|
}.
$$

\(\bar p\) 是可导的 soft mass，\(\bar p^{\mathrm{avail}}\) 是 availability 基准，\(\bar f\) 是实际 active slots 的份额；三者应一起报告。另按所有有效 site-Token 事件分别记录每个节点的 reached、Observe、active 和 Emit rate，用来区分：

| 现象 | 首先检查 |
| --- | --- |
| 节点很少 reached | topology 与上游路径选择 |
| reached 后总是落选 | region selector 的 \(\bar p,\bar f\) |
| Observe 少 | reached 情况与 N/SD/BO profile |
| Emit 少或计算量失衡 | active set、forced-active 与实际执行 |

**BAL-NONE** 可作为无辅助均衡的消融；其他 opportunity-normalized 或跨 micro-batch 方案统一写 **BAL-CUSTOM**，并在实验设置中给出完整公式、统计范围和 reduction。

### 6.3 标准 MoE（M8）的 balance loss 与 router z-loss

M8 使用不同的 Switch-style balance loss。令 \(p_{j,b,t}^{(i)}\) 为 MoE router 的 softmax 概率，\(c_{j,b,t}\) 为硬 Top-1 expert，定义：

$$
\bar p_{j,i}
=\frac{1}{N_V}
\sum_{(b,t)\in\mathcal V}p_{j,b,t}^{(i)},
\qquad
f_{j,i}
=\frac{1}{N_V}
\sum_{(b,t)\in\mathcal V}
\mathbf 1[c_{j,b,t}=i].
$$

其中 \(f_{j,i}\) 是 expert \(i\) 真正收到的 Token 比例。M8 使用：

$$
\mathcal L_{\mathrm{bal}}^{\mathrm{MoE}}
=\frac{1}{I}
\sum_{j\in\mathcal I}
R\sum_{i=0}^{R-1}
\operatorname{sg}(f_{j,i})\,\bar p_{j,i}.
$$

其中 \(\operatorname{sg}\) 表示 stop-gradient（停止梯度）：前向值不变，反向梯度为零。\(f_{j,i}\) 来自不可导的硬路由，梯度只通过 \(\bar p_{j,i}\) 返回 router。完全均衡时，\(\mathcal L_{\mathrm{bal}}^{\mathrm{MoE}}=1\)，而 receiver balance loss 完全均衡时等于 0，所以两种 `balance_loss` 的原始数值不能直接比较。

沿用第 5.2 节，MoE router 收到的消息、expert 输入和 router logits 为：

$$
m_{j,t}=N_F(u_{\ell(j),t}),
\qquad
z_{j,t}^{(c_{j,t})}=m_{j,t},
\qquad
a_{j,t}=W_{\mathrm{moe}}m_{j,t}.
$$

M8 还使用 router z-loss，限制 logits 的整体尺度：

$$
\mathcal L_z
=\frac{1}{I N_V}
\sum_{j\in\mathcal I}
\sum_{(b,t)\in\mathcal V}
\left[
\log\sum_{i=0}^{R-1}\exp(a_{j,b,t}^{(i)})
\right]^2.
$$

因此 M8 的实际反向传播目标是：

$$
\boxed{
\mathcal L_{\mathrm{train}}^{\mathrm{M8}}
=\mathcal L_{\mathrm{LM}}
+\omega_{\mathrm{MoE}}\mathcal L_{\mathrm{bal}}^{\mathrm{MoE}}
+\omega_z\mathcal L_z
}.
$$

\(\omega_{\mathrm{MoE}},\omega_z\ge0\) 由实验设置记录。

> **备注：**M8 采用的是成熟、可靠且便于对照的经典 MoE 基线，但不是所有先进 MoE 统一采用的唯一方案。

| 机制或路线 | 当前定位 | 代表性采用情况 |
| --- | --- | --- |
| **Switch-style balance loss** | 常见的标准基线，但不是唯一推荐路线 | [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)、[OLMoE](https://huggingface.co/allenai/OLMoE-1B-7B-0924) 使用；[Qwen3](https://huggingface.co/Qwen/Qwen3-235B-A22B) 使用 global-batch 变体 |
| **Router z-loss** | 常用的可选稳定项，但采用并不统一 | [ST-MoE](https://arxiv.org/abs/2202.08906) 推荐，OLMoE 使用 |
| **其他负载均衡路线** | 用动态 bias、分位数校准或系统级 dispatch 替代或补充经典辅助损失 | [DeepSeek-V3/R1](https://arxiv.org/abs/2412.19437)：动态 expert bias；[Kimi K3](https://github.com/MoonshotAI/Kimi-K3)：Quantile Balancing；[GLM-5.2](https://huggingface.co/zai-org/GLM-5.2)/5.3：`noaux_tc`，5.3 沿用 5.2 base；[MiniMax-Text-01](https://huggingface.co/MiniMaxAI/MiniMax-Text-01)：GShard-style auxiliary loss + global token dispatch |

这里的动态 expert bias 和 Quantile Balancing 都是训练期均衡；Kimi K3 的最终 bias 在推理时冻结，不等于第 6.4 节的推理期负载感知 selector。

DENSE 没有 router，实际目标只有 \(\mathcal L_{\mathrm{LM}}\)。训练日志中的 `loss` 是包含上述辅助项的总损失，`lm_loss` 只表示 Token 预测损失；跨架构比较模型质量时应使用验证集 `lm_loss` 或 perplexity，而不是直接比较总 `loss` 或两种定义不同的 `balance_loss`。

### 6.4 训练期均衡与推理期负载感知

| 机制 | 训练时 | 推理时 | 作用 |
| --- | --- | --- | --- |
| **训练期 balance loss** | 加入训练目标 | 不再计算 | 让模型学出较均衡的路由倾向，但不保证推理时始终均衡 |
| **负载感知 selector** | 作为前向规则参与训练 | 继续使用 | 根据当前序列的路由历史动态调整后续选择 |

下面给出一个最简单的样例，实际实现可根据训练和推理情况调整。

每个 receiver 可以把近期激活负载作为一个历史标量发给 selector。令 \(\operatorname{load}_{j,b,t}^{(i)}\) 表示这一标量：

$$
a_{j,b,t}^{(i)}
=\left[\operatorname{Score}(\cdots)\right]_i
-\kappa_{\mathrm{load}}\,\operatorname{load}_{j,b,t-1}^{(i)},
$$

完成选择后更新：

$$
\operatorname{load}_{j,b,t}^{(i)}
=\lambda_{\mathrm{load}}\,\operatorname{load}_{j,b,t-1}^{(i)}
+(1-\lambda_{\mathrm{load}})\mathbf 1[i\in\mathcal A_{j,b,t}].
$$

其中 \(\kappa_{\mathrm{load}}\ge0\)，\(0\le\lambda_{\mathrm{load}}<1\)。

同一前向规则可以同时用于训练和推理。训练期 balance loss 只留下学到的均衡倾向；动态负载路由则形成更强的闭环反馈，但也会引入跨 Token 递归，并可能造成路由振荡或增加训练难度。

## 7. 规范命名

### 7.1 科学条件名

单层特例采用：

~~~text
<TRAIN>-<PLACEMENT>-<PROFILE>-R<WIDTH>-I<SITES>-H<DEPTH>-<STATE>-<SELECTOR>-K<ACTIVE>-<EMIT>-<AGG>-<BAL>
~~~

非平凡 HB-Lattice 使用：

~~~text
<TRAIN>-<PLACEMENT>-<PROFILE>-R<WIDTH>-I<SITES>-H<DEPTH>-T<TOPO_ID>-<STATE>-<SELECTOR>-K<ACTIVE>-<EMIT>-<AGG>-<BAL>
~~~

这里的 **H** 和 **T** 都只是可读索引。H 表示一个 GraphBranch 从入口到出口最多顺序经过多少个 receiver nodes，由固定单层结构或最终 Plan 推导；短名称中的 H1、H2 分别表示最大深度为 1、2。H 不是拓扑名称，也不是独立配置。

T 中的 `TOPO_ID` 索引已展开 Plan，不代替 manifest（完整实验配置记录）中的 Plan 与规范化哈希。除本文固定的单层特例外，任何结构即使同样是 H1，也必须提供 T；非平凡 HB-Lattice 始终必须提供 T。

字段定义如下：

| 字段 | 允许值或形式 | 含义 |
| --- | --- | --- |
| TRAIN | PT / CPT / FT / SFT | 初始化与训练阶段 |
| PLACEMENT | POST / PARBLK / PARATTN / PARMLP | GraphBranch 的输入与 residual 返回位置 |
| PROFILE | N / SD / BO | 状态接收与稀疏计算语义 |
| R | R4、R8、R16、RVAR 等 | 单层特例的候选总数，或 HB-Lattice 非平凡 selector region 的候选数摘要；不统一时用 RVAR |
| I | I1、I4、I8 等 | 一个 Token 顺序经过的插入位置数 |
| H | H1、H2 等 | 从固定结构或 Plan 推导的最大 receiver node 深度；AggregatePorts 不计入 |
| T | T\<TOPO_ID\> | 已展开 topology 的索引；只有本文固定的单层特例省略 |
| STATE | NONE、EMA128、GDN-K32-V32、ATTN-FULL、ATTN-W128、ATTN-COMP 等 | 状态结构和必要尺寸 |
| SELECTOR | SEL-CONTENT / SEL-PRE / SEL-POST | 第 3.3 节定义的 selector 输入时序 |
| K | K1 / K2 / KALL / KVAR | 单层特例每次激活的候选数，或 HB-Lattice 各 region 的 \(K^{\max}\) 摘要；不统一时用 KVAR |
| EMIT | EMIT-HARD / EMIT-HST / EMIT-SOFTP / EMIT-CUSTOM / EMIT-VAR | 第 3.5 节定义的 receiver sender Emit 语义 |
| AGG | AGG-MEAN / AGG-LEARNED / AGG-CUSTOM / AGG-VAR | 第 3.5、4.3 节定义的 AggregatePort 消息聚合；不统一时用 VAR |
| BAL | BAL-AVAIL-SOFT / BAL-NONE / BAL-CUSTOM / BAL-VAR | 第 6.1、6.2 节定义的训练期路由均衡；不改变推理前向 |

**SEL-CONTENT**、**SEL-PRE** 和 **SEL-POST** 分别表示 \(\operatorname{Read}^{\mathrm{sel}}\) 只读取 receiver 当前本地消息、额外读取旧状态或额外读取更新后状态。单层特例的 \(\operatorname{Score}\) 还读取由共同入口产生的公共 \(\mu\)；HB-Lattice region 不要求存在单一公共 \(\mu\)，只联合处理 reached nodes 的轻量读出和 presence 信息。名称不限定打分采用线性层、MLP 或其他实现；精确读出、打分公式以及状态中是否包含历史激活记录仍由 manifest 和实验设置保存。

如果历史激活记录会影响 selector 或输出，它就是模型前向语义的一部分，不能隐藏在同一个纯 EMA/GDN 条件名下。具体实现确定后，应在 **STATE** 中增加明确的复合状态标签；记录维度、衰减、写回规则等细节再放入 manifest。

**K** 只表示 selector 激活多少个候选，**EMIT** 只表示 active receiver 怎样产生发送消息，**AGG** 只表示 AggregatePort 怎样合并实际收到的消息。三者不能互相代替。

单层输出聚合、HB receiver 多父输入和 HB 最终输出都使用同一个 **AGG** 接口；AGG-MEAN 在单消息端口自然退化为 identity。selector 概率的直接前向或梯度作用由 **EMIT** 承担，当前 `MessageAggregate` 不再次读取它。GraphBranch 与 backbone 的 RESIDUAL_ADD 已由 placement 固定，不属于 AGG。若同一实验的 Emit 或 Aggregate policy 不统一，使用对应的 **VAR**，并在 manifest 中列出逐 node 或逐端口设置。

**BAL-AVAIL-SOFT** 在单层特例中退化为第 6.1 节的固定候选均衡，在 HB-Lattice 中使用第 6.2 节的 availability 基准。**BAL-CUSTOM** 和 **BAL-VAR** 必须附完整公式与聚合范围。

TRAIN 的含义必须严格区分：

- **PT**：随机初始化后做自回归预训练；
- **CPT**：加载预训练 checkpoint，继续做语言模型目标训练；
- **FT**：加载预训练 checkpoint，使用不同于基础自回归预训练的下游任务目标；
- **SFT**：FT 中特指有监督的指令或输入输出微调。

TRAIN 描述 base 权重与训练目标；新增 GraphBranch 及其 receiver nodes 的初始化方式由实验设置单独记录。

口语中的“finetune”不能直接写入正式名称：如果实际仍是 FineWeb 或领域语料上的自回归语言模型训练，应记为 CPT；只有训练目标确实改变时才记为 FT 或 SFT。

读完字段定义后，三个完整例子分别是：

~~~text
CPT-PARMLP-BO-R8-I4-H1-EMA128-SEL-POST-K1-EMIT-HST-AGG-MEAN-BAL-AVAIL-SOFT
PT-POST-SD-R8-I4-H1-EMA128-SEL-PRE-K1-EMIT-SOFTP-AGG-MEAN-BAL-AVAIL-SOFT
PT-POST-BO-R2-I4-H7-THBL2D2P2CMIR-GDN-K32-V32-SEL-POST-K1-EMIT-HST-AGG-MEAN-BAL-AVAIL-SOFT
~~~

### 7.2 R、I、H 与 K 不得混用

- **R8** 表示单层特例有 8 个候选，或 HB-Lattice 的非平凡 selector regions 统一有 8 个候选；不表示模型共有 8 个 receiver nodes。
- **I8** 表示每个 Token 顺序经过 8 个插入位置，不表示 Transformer 只有 8 个 blocks。
- **H2** 只摘要一个插入位置内部的最大 receiver node 深度为 2；它由 Plan 推导，不能唯一确定拓扑。
- **K2** 表示每次局部选择激活两个候选；在 HB-Lattice 中，它表示各非平凡 regions 的 \(K^{\max}=2\)。**KALL** 表示全部当前候选都 active；不同 regions 不统一时使用 **KVAR**。
- **AGG** 不携带 K；例如 **K2-EMIT-HST-AGG-MEAN** 表示最多激活两个候选，各自用 Hard-ST Emit，端口再均匀聚合实际消息。

receiver node 内部串行的状态/上下文 residual 与 FFN residual 合计仍算一层；只有该 node 的完整输出继续进入下一层 receiver node 时，H 才增加。GraphBranch 输入、输出端口以及任意 AggregatePort 都不增加 H。

例如 **R4-I8-H1-K1** 表示 8 个顺序插入位置，每处采用固定单层特例，有 4 个候选且激活 1 个。它不是 8 层递归。

如果不同插入位置、Line 或非平凡 selector region 采用不同宽度，短名字中使用 **RVAR**，并在 manifest 和报告中列出完整宽度；forced-active 的 singleton receiver region 与所有 AggregatePorts 不参与 R 的摘要。除本文固定的单层特例外，平台期、多父边、镜像直通以及任何其他结构差异都不能靠 R/H 推断，必须同时给出 **TOPO_ID** 和完整 Plan。

### 7.3 具体 run 实例名

科学条件之外，真实 run 还需要模型、seed 和尝试编号：

~~~text
<MODEL>-<scientific-condition>-s<SEED>-r<ATTEMPT>
~~~

例如：

~~~text
q3-06b-cpt-parmlp-bo-r8-i4-h1-ema128-sel-post-k1-emit-hst-agg-mean-bal-avail-soft-s42-r1
~~~

模型 checkpoint、数据 revision、精确 block 编号、Token 预算、学习率、dtype、设备和代码 commit 仍由 manifest 保存，不强行塞进短名字。名称是可读索引，不代替完整实验设置。

### 7.4 基线名称

Dense 与标准 MoE 不使用 TIDE placement/profile 字段：

~~~text
PT-DENSE
CPT-DENSE
PT-MOE-R8-I4
CPT-MOE-R8-I4
~~~

MOE 的精确插入 block、Top-K、capacity、token-drop、shared expert 和路由辅助项必须在完整设置中声明。

## 8. 名称之外仍必须明确的语义

即使规范名称相同，每个正式设置仍要明确记录：

- 精确插入 block 编号；
- 不同 sites、Lines 和节点之间是否共享参数；默认不共享；
- 若使用 HB-Lattice，记录完整 `HBLatticePlan`、规范化哈希、`HBLatticeExecutionConfig`，以及 TopologyBuilder 的名称、版本和配置；
- 若使用 HB-Lattice，记录每条 Line 的 phase、节点与 region 划分，`GraphInputPort` / `GraphOutputPort` 的连接，每条边的端点和 input/tree/local/shortcut/mirror/output 类别，以及逐节点镜像直通开关；
- 若使用 HB-Lattice，记录最大 fan-in/fan-out、region 大小与 forced-active 节点；
- 每个 selector 的 active 数规则，以及逐 region 的 \(K^{\max}\)；
- 每个 AggregatePort 的 `MessageAggregate` policy、source-presence 与任何 `MergeScore` 的精确公式；
- `EmitPolicy` 的精确公式，以及 EMIT-HST 的 \(\zeta^{\mathrm{ST}}\)；
- `ReceiverNodeTemplate` 的精确 `NodeCompute` 公式、内部执行顺序、归一化、residual、昂贵计算与 identity 初始化条件；
- \(N_{\mathrm{sel}}\)、\(N_{R,i}\) 与 \(N_{F,i}\) 的精确实现和初始化；
- \(\operatorname{Read}^{\mathrm{sel}}\)、\(\operatorname{Read}^{\mathrm{ffn}}\) 与 \(\operatorname{Score}\) 的精确公式、输出维度以及是否包含历史激活记录；
- Update proposal、selector、commit / Observe 和历史激活写回的精确顺序，以及写入 \(p\) 时是否 stop-gradient；
- GraphBranch 与 backbone 的 RESIDUAL_ADD 公式以及任何额外缩放；
- `BalancePolicy`、各辅助 loss 的公式、系数与 reduction；
- 状态初始化、有效 Token mask、跨 chunk 的 carry/reset 与梯度 detach 规则；
- 辅助 loss 的 Token 范围、site/Line/region 聚合范围、reached mask 处理以及是否跨 micro-batch 或设备统计；
- reached、Observe、active、Emit、soft mass 与 hard share 等诊断量的分母和聚合范围；
- 每个 Token 实际执行多少次 `MessageAggregate`、本地入口归一化、轻量 selector 读出、Update proposal、Observe commit、较大状态读出、昂贵 FFN 和 Emit；
- 初始化怎样保持或改变 base 函数；
- MOE 是否有 expert capacity、token drop 或 reroute。

这些项目不会全部进入短名字，但它们决定两个 run 是否构成真正的匹配对照。
