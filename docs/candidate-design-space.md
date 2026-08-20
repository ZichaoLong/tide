# TIDE 候选设计空间

> 状态：研究与实现候选，不代表已有训练结论。
>
> 本文回答“候选结构可以怎样组成、各机制分别解决什么问题”。当前任务、实验顺序与验收标准见 [README](../README.md) 和 [实验协议](experiment-protocol.md)。

## 1. 先区分四类内容

| 类别 | 当前内容 | 地位 |
| --- | --- | --- |
| TIDE 结构要求 | 固定空间拓扑、单节点成本有界、可达容量增长、有界度多跳扩展 | 长期目标必须满足 |
| 当前稳定骨架 | checkpoint 中性生长、always-on backbone、fixed merge | 近期默认采用，不主张对所有 TIDE 架构都必要 |
| 主要候选机制 | 私有状态、`broadcast-observe`（BO）、交叉汇聚、backbone reinjection | 用来检验和缓解下游历史覆盖问题 |
| 直接对照 | `selected-dispatch`、无延迟状态、无交叉边、flat MoE | 用来判断收益到底来自哪里 |

工作流 B 会围绕 `broadcast-observe` 这条主要验证轴组织第一批候选与配对实验，但它的真实任务是找到能够训练和扩展的完整候选，而不是先把某一种机制写成已经成立的结论。

## 2. 当前结构骨架

### 2.1 有界度的局部分支

考虑一个输入输出空间相同的父模块。它包含 always-on 主分支 $B_0$ 和有限个候选 residual 分支 $B_1,\ldots,B_b$。跨模型规模始终要求：

$$
b\leq b_{\max},
$$

其中 $b_{\max}$ 不随模型总容量增长。无状态简写下，selector 为输入 $x$ 选择集合 $A(x)$，固定 merge 为：

$$
T(x)=B_0(x)+\sum_{j\in A(x)}g_j(x)B_j(x).
$$

“固定”表示分支入口、出口、边和 merge 位置预先写入模型结构；动态变化的是本次激活哪些固定分支，而不是临时增加或删除边。

这个骨架提供四项近期好处：

1. 原 checkpoint 主路径可以完整保留。
2. 新分支可以通过零输出或经证明等价的 clone、缩放与 merge，在初始化时保持原函数。
3. selector、router 和 merge 只处理数量有上界的直接邻居，不会随总容量变成越来越大的中心节点。
4. 当前分支的显式路径身份在 fixed merge 处结束，控制影响范围比较容易测量。

### 2.2 容量怎样继续增长

每个具体模型都是有限 Graph，并声明本实例的最大深度和最长串行路径。跨模型规模扩展时，统一保持有界的是每层 fan-in、fan-out、局部候选数和 Top-K；递归深度或空间直径可以增长。

如果入口数量、fan-out 和最大传播深度都在所有规模上保持固定，可达节点总数也会存在固定上限。因此，容量增长最终必须表现为更多传播层级或更大空间范围。

规则递归只是近期容易从 checkpoint 生长的工程起点。Line、lattice、mesh、多尺度 backbone 和其他有界度局部 DAG 也可以表达同一个长期方向。

### 2.3 长短路径与 fixed merge

候选分支可以是单个 Attention、FFN、SSM，也可以是有限串联模块或另一个满足相同入口、出口和 merge 契约的递归模块：

```text
             +-> Attention --------------------+
input -------+                                  +-> fixed merge -> output
             +-> Attention -> FFN --------------+
```

长短路径来自本次有限 DAG 中不同的串联深度。Fixed merge 会等待本次已激活分支完成；它不是让短路径先提交，再让长路径在以后 Token 回来修改旧输出。

## 3. 消息、状态、计算和发送必须分开

一个 receiver 的局部语义至少包含四个不同决定：

1. **Receive**：它是否收到当前消息。
2. **Update**：收到后是否更新私有状态。
3. **Activate / Read**：是否执行 Attention、FFN、SSM readout 等昂贵计算。
4. **Emit**：是否产生输出并继续向下游发送。

不能用一个“节点是否激活”掩盖这四件事，否则既无法解释模型，也无法分别计算消息、状态和重计算成本。

### 3.1 两种主要传播 profile

| Profile | 语义 | 主要用途 |
| --- | --- | --- |
| `selected-dispatch` | 先选择 active children；只有被选节点收到、更新、计算并继续发送 | 最接近普通 MoE，是 BO 的直接反事实 |
| `broadcast-observe` | active sender 向全部固定直接 children 发送；receivers 先执行声明的 Observe / Update，只有少数节点做昂贵计算并继续发送 | 把消息可见性和状态写入机会，与昂贵激活分开 |

BO 的最小流程是：

```text
active sender
-> send to every declared direct child
-> every actual receiver observes and applies Update
-> local selector chooses a small active set
-> active receivers run expensive compute and emit
-> inactive receivers retain the declared state but emit nothing
```

只有 active sender 的固定直接下游收到消息。BO 不会绕过未激活祖先，也不会让所有深层节点自动看到全局历史。

### 3.2 Update 不等于昂贵计算

对节点 $v$ 在 Token 位置 $t$，令 $M_{v,t}$ 为实际收到的消息集合，$S_{v,t}$ 为它的单序列私有状态，可以写成：

$$
S_{v,t}^{+}=U_v(S_{v,t},M_{v,t}),
$$

$$
a_{v,t}=1\Longrightarrow y_{v,t}=F_v(S_{v,t}^{+},M_{v,t}).
$$

$U_v$ 可以是 KV 写入、SSM accumulator、有限窗口或低维 summary 更新，也可以退化为无持久效果的操作。只有 $a_{v,t}=1$ 的节点执行 $F_v$ 并继续发送。

如果 Update 的成本接近完整 Attention/FFN，跳过昂贵计算就可能没有系统收益。因此必须分别记录消息投递、状态更新、昂贵计算和继续发送的成本。

### 3.3 多父节点

两种传播 profile 都可以用于多父局部 DAG：

- `selected-dispatch` 让每个父节点只向自己选中的 children 发送。
- BO 让每个 active parent 沿全部声明的局部出边发送。

多父节点必须额外定义 inbox 何时完整、多条消息怎样归约、空消息怎样处理、状态何时提交，以及多个父节点的预算怎样仲裁。BO 与这种固定局部接收方式结合得比较自然，但多父拓扑并不在语义上强迫使用 BO。

## 4. 下游历史覆盖稀释

### 4.1 这里担心的到底是什么

多级 `selected-dispatch` 可能使分支路径越往下游，有状态节点接收到的历史 Token 或消息越少，因而其私有状态覆盖的上下文历史越来越窄。本文把这个待验证现象称为：

> **分支路径下游的上下文历史覆盖稀释**，简称“下游历史覆盖稀释”。

它不表示 parent hidden 被机械切成几份。当前 hidden 仍可能已经整合完整前缀中的任务相关信息；选择性的局部历史也可能形成有价值的专门化，而不一定伤害质量。

因此必须分别测量：

- receiver 收到的消息和 Token 覆盖率；
- 不同 receiver 私有状态的历史重叠；
- 当前 hidden 中的任务相关信息；
- 历史覆盖差异是否真的改变质量、状态利用和后续路由。

### 4.2 四类候选机制

如果下游历史覆盖稀释真实存在并且伤害学习，当前有四类重要候选：

1. **私有状态与 Update/Compute 分离**：节点先保存收到的信息，昂贵 readout 是否执行另行决定。
2. **`broadcast-observe`**：未被选中做昂贵计算的直接下游，仍获得当前消息和状态写入机会。
3. **分支间交叉汇聚**：不同路径在最终 fixed merge 之前交换已经处理过的局部信息，扩大局部 source coverage。
4. **Backbone reinjection 或周期性共同接口**：重新注入稳定公共表示，避免深层分支长期只依赖狭窄路径历史。

它们可以单独或组合使用。Always-on backbone 和 fixed merge 提供稳定起点与明确收拢位置；前四类机制则用来检验、缓解或绕开下游历史覆盖问题。

### 4.3 BO 能缓解什么、不能保证什么

BO 能保证 active parent 的全部固定直接 children 获得本次 Observe / Update 机会，因此可以缓解 selected-only data exposure。但它不能自动解决：

| 问题 | BO 是否自动解决 |
| --- | --- |
| 实际收到消息节点的状态/数据饥饿 | 可以缓解 |
| 节点长期不执行昂贵计算 | 不能保证 |
| 状态写入没有有效梯度或以后从未读出 | 不能保证 |
| 未激活祖先造成的更深层无消息 | 不能解决 |
| 输入分布漂移和专门化失败 | 不能保证 |

因此，第一批实验会把 BO 作为主要验证轴，同时保留相同拓扑和 route 下的 `selected-dispatch` 对照，以及 state freeze、clear、shuffle、no-read、reset 等 knockout。

### 4.4 交叉汇聚的两个不同作用

交叉连接至少包含两个容易混淆的概念：

- `cross-coupling`：分支交换摘要后继续保持各自路径身份，主要扩大局部历史来源。
- `fixed convergence/reset`：多路信号回到共同接口并结束旧路径身份，主要限制控制寿命。

交叉汇聚可能增加更短的信息和梯度路径，也可能增加动态输入组合、多路归因、等待和通信成本。因此它是重要候选，而不是已有结论。

## 5. 门控与 selector 是独立坐标

每个候选至少使用下面五个坐标描述：

$$
\mathcal C=(\text{门控范围},\ \text{传播 profile},\ \text{状态生命周期},
\ \text{selector 输入},\ \text{selector 决策}).
$$

### 5.1 门控范围

- **Leaf-Gated**：只有末级叶子可以跳过昂贵计算，非叶层级始终开启。
- **Receiver-Gated**：部分内部 receiver 也可以不做昂贵计算和不继续发送，因此一次选择可以裁剪更深子树。

门控范围不决定传播 profile。Leaf-Gated 和 Receiver-Gated 都可以使用 `selected-dispatch` 或 BO。

### 5.2 Selector 输入

局部 selector 可以按候选需要读取：

- 当前内容摘要；
- receiver 更新后的 semantic-state summary；
- 逐序列历史负载、恢复量或预算；
- 静态节点类型、层级和容量信息。

物理 batch、实时设备负载和 chunk 切分不能进入模型路由语义。只要历史状态会改变输出，它就必须逐序列隔离、可保存、可恢复并可重放。

### 5.3 Selector 决策

可比较的规则包括 fixed/hash route、content-only scoring、soft mixture、hard Top-K、语义候选后再做负载筛选，以及累计信号超过阈值后激活。

这些规则目前都是实验坐标。自由度越高，越需要更强的配对反事实、状态语义和 `prefill = decode` 验证。

## 6. 三种距离不要混写

| 距离 | 含义 | 主要风险 |
| --- | --- | --- |
| 数值语义距离 | 中间数值贡献到最终 loss 经过多少计算 | 普通深网络也会很长 |
| 控制寿命 | 一次离散选择在多远范围内继续限制后续可达路径 | 路由漂移和长路径信用分配 |
| 状态读写延迟 | 某次私有状态写入到以后真正读出的 Token 间隔 | 延迟信用、detach 和状态归因 |

Always-on backbone 能保留短而连续的公共梯度路径，但不能保证稀疏分支获得有效梯度。Fixed merge 能限制显式控制寿命，但不会删除已经写入私有状态的跨 Token 影响。

## 7. 其他候选的位置

### 7.1 Flat MoE

Flat MoE 是成熟强基线。它通常在单个子层内从完整专家集合做一次选择并立即 merge，不验证有界度多跳传播，也通常没有专家私有的跨 Token 延迟状态。

### 7.2 Head-Wise MoE

Head-Wise MoE 把 hidden 按 Head 或 Group 切分，在组内使用较小 expert pool，再 concat 或 mixer。它主要检验局部因子化 MoE 的质量与通信价值，属于工作流 A 的可选后续，不自动验证 BO、私有状态或多跳传播。

### 7.3 Line、lattice、mesh 与一般局部 DAG

这些结构可以提供更自然的空间扩展和交叉汇聚。近期先用规则分支获得可训练性和失败诊断证据，再按具体问题引入更多拓扑自由度；这是一项工程选择，不表示规则递归是唯一必经形式。

## 8. 每个候选必须记录什么

实验配置至少要明确：

- branch grammar 与静态 receiver 拓扑；
- 最大 fan-in、fan-out、传播深度和局部 Top-K；
- 门控范围与传播 profile；
- Receive、Update、Read、Activate、Emit 的顺序；
- 私有状态类型、生命周期、空消息规则和 chunk 边界；
- selector 输入、决策与 active/message budget；
- always-on backbone、交叉边和 fixed merge 位置；
- 参数、FLOPs、消息、状态和物理放置成本。

一个配置同时改变多个坐标时，可以支持“这个完整候选出现正面信号”，但不能单独证明其中某个部件必要或有效。
