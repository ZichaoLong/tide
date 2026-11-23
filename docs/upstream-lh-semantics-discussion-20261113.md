# 参考 LH 的图、时间与区域语义：带到数学上游的讨论材料

日期：2026-11-13。本文可单独携带到上游新会话，不依赖实现会话历史或运行目录。
它区分用户意图、参考源码事实、固定教材已有能力与尚待决定的定义；不是已经采用的新规范。

## 1. 本次希望讨论什么

用户希望设计一类具有少量常亮枢纽（hub）、大量稀疏卫星节点、不同长度消息路径的网络，
参考旧 LH 的结构与批处理经验，但局部记忆和区域分配以 Tide 数学定义为锚点。
请先明确它与 TimedDAG、PositiveDelayGraph（PDG）的关系，再判断是否需要补充具体局部候选。
不要求复刻旧 Selector，不预先要求新的网络可以改写成 SettleGraph。

实现仓库正在另外讨论 FC、HB 及“HB 跨区域、跳层边变体”的直接计算样板。
真异步 LH 的实现暂不展开，等待这里的数学对齐；这些 SettleGraph 样板不应被当成参考 LH。
这里“异步”指消息在不同逻辑时间到达，不等于主机线程、设备流的异步执行。

## 2. 固定来源与阅读入口

| 材料 | 身份与位置 |
| --- | --- |
| 实现仓库 | `graph-execution-foundation`，调查源码 `9afe84773afb72214e80b615cc704667945f0ccb` |
| 已采用数学 | `ZichaoLong/ObsidianVault` 的提交 `0770593838a37c790a568226930567ed60f99b83`，目录 `20-tide-decentralized-neural-network` |
| 数学离线副本 | 实现仓库 `docs/upstream/tide-core-5/`；`docs/upstream.json` 记录哈希，`python scripts/check_upstream.py` 核对 |
| 参考 LH | 本机 `/mi/data2T/zlong/home/llm/lh`，HEAD `5fd237d40c9880ccb6e511e4bf20799c7022fd1e` |

上游阅读入口（路径相对上述数学目录）：

- `positive-delay-graph-finite-cut-learning-note.md`：正时延图、有限切面及继续。
- `timed-dag-region-selector-learning-note.md`：时间纤维、候选、区域历史与完整局部作用。
- `timed-dag-chunk-prefill-learning-note.md`：联合求值与适用条件。
- `settlegraph-learning-note.md`：严格区域分层、时间对齐与嵌入。
- `local-semantics/weighted-message/architecture-learning-note.md`：共同状态保留、描述、选择和广播。
- 同目录 `region-allocation-learning-note.md`、`kv-memory-learning-note.md` 与 `recurrent-memory/`：具体候选。

参考 LH 已有其他工作区修改；本次只读以下文件，四者均与该 HEAD 一致：

| 文件 | 关键入口 |
| --- | --- |
| `PyConnectome/Graph.py` | `HierarchyHubsAndPoints`、`HierarchyGraph`、`IOGraph`、`GraphData.create_from` |
| `Connectome/cpp/src/Selector.cpp` | `NaiveSelector`、`select`、`gather_select_info_select_by_tensor` |
| `Connectome/cpp/src/CortexNet.cpp` | `IOCortexNet::think_single_step`、`think`、节点 `chals` 配置分组 |
| `Connectome/cpp/test/test-cortexnet.cpp` | 历史测试入口、selector 实参、逐 token 调用 |

这些源码提供设计事实，不自动成为 Tide 的数学规范，也不证明二者局部数值等价。

## 3. 用户希望保留的架构意图

1. 少量枢纽形成持续可用的输入到输出通路。层次节点数可类似 `1,2,4,8`，数量仅为示意。
2. 枢纽附近分布较多小节点，例如每组 32 个，只有少量小节点执行完整内容计算。
3. 枢纽可使用固定维度循环记忆，避免常亮节点的全历史 KV 随上下文持续增长。
4. 小节点与枢纽有两种待比较的区域组织：
   - 同区域，共用本次分配信息，但枢纽具有强制参与规则，小节点竞争稀疏份额；
   - 分属不同区域，只通过实际消息交互，枢纽与卫星分别采用区域规则。
5. 同一源输入可沿短路径和长路径到达同一节点，保留不同逻辑时间的作用。
6. 是否保留反馈、哪些边双向、输入输出如何配对，还需在上游明确。

“常亮”“约 3% 活跃”“最短通路”都需要定义分母、触发条件和保证范围。
结构上存在路径，不保证该次运行沿途确实产生消息；接收、候选、Full、KV 写入也不是同一计数。

## 4. 参考 LH 实际做了什么

### 4.1 图结构并非区域层链

`HierarchyHubsAndPoints` 生成多层 hub，最基层 hub 各带 `localnum` 个叶节点。
因此现有生成器并非“每一级的每个 hub 都带 32 个卫星”的固定实现。
`HierarchyGraph` 加入几何邻域与跨层连接；`IOGraph` 按配置定向，输出侧图是输入侧图的转置。
`bidirection` 还会保留部分双向连接。

`GraphData.create_from` 明确生成：

- `ioA`：全部输入侧节点连接到对应输出侧节点；
- `oiA`：输出侧 hub 连接回对应输入侧 hub。

所以完整输入侧／输出侧组合图含反馈环，不因某一侧采用分层结构就成为 DAG。
把其数值模块映射到 Tide 仍需逐项定义，不能只复制这张图便宣称局部语义一致。

### 4.2 一次 token 调用内还有多个逻辑步

`think_single_step` 的桥接读取传入的旧 `iacts/oacts`，之后才计算并返回 `newiacts/newoacts`：

```text
iinputs = oibridge(old_oacts) + 本步外部输入
oinputs = iobridge(old_iacts)
new_iacts = inet(old_iacts, iinputs, input_states, input_selector)
new_oacts = onet(old_oacts, oinputs, output_states, output_selector)
```

这为把传播解释成正逻辑时延提供了直接依据。不能只凭一处注释把输入到输出桥当作零时延。
`think` 在首步注入当前 token，随后再运行 `n_layer-1` 个内部步，收集每步的输出 hub 信号，
交给 `Pronounce`。token 编号、内部逻辑时间、输出收集窗口因此必须分开。

### 4.3 “常亮”与 Selector

`NaiveSelector::select` 让指定 hub 的已接收信号直接通过。
无输入的 hub 仍没有信号；这不是无输入也自主执行的心跳机制。

- `with_lead_point=true`：上层 hub 直接通过；最基层 hub 与本组叶节点一起竞争。
- `with_lead_point=false`：最基层 hub 也直接通过；只在叶节点间竞争。
- 原测试 `test-cortexnet.cpp` 使用 `with_lead_point=true`，不能描述成所有 hub 都无条件常亮。

旧选择器优先照顾历史选择次数较少、接收次数较多的节点，再比较信号范数和稳定编号。
这是带计数历史的简化规则；不等于当前带权区域候选中的评分、总量与约束优化。
`chals` 的两组配置还允许上层 hub 与最基层 hub／叶节点使用不同局部模块。

## 5. 与当前固定数学的关系

### 5.1 不同到达时间不能靠改延迟消除

考虑源 hub 在时刻 `t` 产生的一次消息：

```text
h0 --delay 1----------------> h1       到达 t+1
h0 --delay 1--> s --delay 1--> h1       到达 t+2（若 s 实际发送）
```

无反馈时可形成非对齐 TimedDAG；`h1` 有两个潜在作用时间。
若再有 `s -> h0`，则出现反馈，需要 PDG。每条边仍需正时延。
把直达边改为 delay 2 可以使这两个路径对齐，但改变了运行的时间和状态读取，不是等价优化。
同一节点、同一逻辑时间的多条消息先聚合；不同时刻的消息不能因源自同一 token 就合并。

当前实现中曾称作 LH 的 `python/tidegraph/planning/topology.py::lh_inspired` 则是：

- 区域层宽 `1,2,4,8,4,2,1`，共 22 个区域；每区 32 节点是 704 节点，每区 8 节点是 176 节点；
- 节点没有“强制常亮 hub”与“卫星”的身份区别；
- 添加向更高层的跨区域边和跳层边，跳过两秩的边使用 delay 2，继续保持时间对齐。

它应称作“HB 跨区域、跳层边变体”，仍属 SettleGraph。它可作为候选，但不替代上述异步结构。

### 5.2 正典可表达，与现成候选已定义，是两回事

一般正典允许自定义区域 `SelStep` 和区域历史空间；不要求所有区域使用同一种记忆。
已有带权支线则明确采用全部候选保留更新、`Y_j={*}`、以 `c_v>0` 决定 Full 支持的绑定。
一般引擎能够表达某种策略，不代表固定教材已经定义了这种策略的具体公式与导数。

现有函数可提供一个较简单的不同区域方案：hub 单独成区，采用
`selection_mode=unit_attention, allocation_base=uniform`，单候选存在时发送强度为 1。
这个规则连“收到真实零强度原子”的单候选也会发送；完全没有输入仍不产生候选。
卫星区域另外使用稀疏分配。它不共享同一区域的 query、预算与本次选择信息，
所以不能把它当成“同区强制 hub”的等价替代。

## 6. 需要在上游作出的具体选择

### A. 常亮的定义、区域归属与预算

请明确保证是“成为候选就必选”“正接收就必选”，还是“每个逻辑步都执行”。
最后一种需要明确驱动来源；不能用缺席消息冒充真实零消息。
若 hub 与卫星同区且要求强制选择，请定义：

- 必选子集如何与本次真实候选集相交；hub 的发送强度是固定 1、保底值还是可学习值；
- hub 是否计入发送总量 `k` 和活动数量上限 `K`，剩余预算如何交给卫星；
- 低接收、零容量、预算不足时的可行性与返回；不能用一个很大的评分代替所有输入上的必选保证；
- 同分规则、连续性适用范围、硬选择或最优解映射的一阶反向。

区域本次只能使用同一逻辑时间的候选描述。若 hub 本次无输入、卫星有输入，
是否还希望使用 hub 的过去证据？若希望，须显式定义状态和读取，不能假设当前描述自然存在。

### B. 稀疏 Full 与稀疏接收／写入

共同架构 §3.1 规定所有候选先更新并保留状态，未选中者也保留；KV 写入不等待选择。
因此 hub 每步广播给全部卫星时，即使只有约 3% 执行 Full，也可能全部发生接收与记忆更新。
请区分希望减少的是 Full、候选准备、KV 写入、消息量，还是其中几项。
有限 KV 窗口、固定维度记忆、稀疏入边各有不同作用；把未选中者写入删除会改变局部语义。

所有窗口、位置编码、短卷积沿原始逻辑时间定义，不能压紧稀疏事件后使用接收次数替代。
hub／卫星使用不同记忆是可行的设计方向，但发送载荷和区域所需描述必须有相容定义。

### C. 历史均衡是否成为正式候选

参考 LH 的历史接收／选择计数没有自动包含在 `Y_j={*}` 的带权区域候选中。
若采用历史均衡，请定义计数对象、每样本状态、更新时机、是否衰减、空事件行为和初态；
还需说明计数如何影响评分或预算、是否求导，以及对整段联合求值带来的时间依赖。
不能把一个统计惩罚就视作所有输入上的负载均衡保证。

### D. 输入、输出、反馈与自回归

请明确 token 的输入端口及日程、每 token 的内部步数、合法输出槽与读出函数。
迟到消息在下一 token 或下一窗口是否继续处理？反馈何时停止，哪些 pending 必须跨切面保留？
teacher-forcing 的已知输入窗口与依赖上一生成输出的新 token，批处理条件不同。
因果依赖还包括选择和消息存在性；不能只对一条实际运行轨迹上的数值消息作追溯。

## 7. 建议上游讨论形成的交付

先给一个小型、完整、可手算的实例，再作参数化推广。至少明确：

1. 节点／区域／端口／边及正时延；无反馈与有反馈版本分别归类。
2. 初态、各类记忆、来源分类与共享、Read 快照、分配规则、完整发送与读出。
3. 已有规则可直接组合之处，与确需新增定义之处；列出适用条件和一阶反向约定。
4. 小轨迹见证：长短路径不同到达、同刻合并、缺席／零强度、hub 必选预算、未选中写入、跨窗迟到消息。
5. 稀疏性分母和工作量：节点总数、候选数、Full 数、接收／写入数、消息数分别统计。
6. 可并行／联合求值的条件与精确继续状态，不强制每个数学事件对应单独 Tensor 或调用。

不需要在数学讨论中指定 Python/C++ 调度架构，不要求训练出有竞争力的模型。
若定义变化，请形成可固定采用的版本；实现仓库不会无记录地跟随上游可变 HEAD。

## 附录：参考源码字节身份

以下 SHA256 对应 §2 四个参考 LH 文件；本次未修改参考仓库：

```text
917e1cdc0d3b4fb3bc2bccbe60516ab10498caac8e0073892af377ace94fb84a  PyConnectome/Graph.py
a7f56fd8e3d203d7b75d09bb6d54c7e04145ff8c3b01ae9b6f67b7cd0de6a570  Connectome/cpp/src/Selector.cpp
7cd9fa49c4308092461f20d6fd2c5b0b9299d6895d9cea85229df4111c82615e  Connectome/cpp/src/CortexNet.cpp
72af85c478400f789d80aad5476023fdacad44f0c3d25aaeec2db6c4d7413a53  Connectome/cpp/test/test-cortexnet.cpp
```
