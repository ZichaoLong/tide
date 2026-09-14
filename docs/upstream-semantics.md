# 上游语义引用与本地对应

fractal-latcarf 是 20-tide 的 SettleGraph 实验平台。上游负责抽象对象、函数依赖、数学教材和一般定理；本仓库负责具体函数选择、模型接入、参数与状态配置，以及实现、训练和性能证据。本页说明继承关系，不复制一份可以独立修改的核心定义。

## 固定的采用版本

| 字段 | 采用值 |
| --- | --- |
| 上游仓库 | [ZichaoLong/ObsidianVault](https://github.com/ZichaoLong/ObsidianVault) |
| 不可变提交 | `facf1afc696673a80e49b1e327bb1d058273893b` |
| 语义版本 | `tide-core-2` |
| 语义锚点路径 | `20-tide-decentralized-neural-network/semantics-anchor.md` |
| 锚点原始文件 SHA-256 | `36a482c14b359994bf61bfbb3b7f98456df9effdc59abc5e468c3e5247b0732b` |
| 采用的模型族 | SettleGraph；通过上游声明的逻辑日程与投影，对应 TimedDAG 的受限实例 |

这些值保存在 [tide-core.lock.json](upstream/tide-core.lock.json)。哈希针对该提交下锚点的原始文件字节，不能用网页渲染内容计算。锁文件固定文档依赖；它不声称已有执行器、Plan、fixture 或 checkpoint 已能解析该文件，也不修改这些工件现有的格式。

本次采用的教材均固定在同一提交：

- [语义锚点与仓库分工](https://github.com/ZichaoLong/ObsidianVault/blob/facf1afc696673a80e49b1e327bb1d058273893b/20-tide-decentralized-neural-network/semantics-anchor.md)
- [SettleGraph 教材](https://github.com/ZichaoLong/ObsidianVault/blob/facf1afc696673a80e49b1e327bb1d058273893b/20-tide-decentralized-neural-network/settlegraph-learning-note.md)
- [TimedDAG 与区域选择](https://github.com/ZichaoLong/ObsidianVault/blob/facf1afc696673a80e49b1e327bb1d058273893b/20-tide-decentralized-neural-network/timed-dag-region-selector-learning-note.md)
- [TimedDAG 的节点级时间批](https://github.com/ZichaoLong/ObsidianVault/blob/facf1afc696673a80e49b1e327bb1d058273893b/20-tide-decentralized-neural-network/timed-dag-chunk-prefill-learning-note.md)
- [正时延 Graph 与有限切面](https://github.com/ZichaoLong/ObsidianVault/blob/facf1afc696673a80e49b1e327bb1d058273893b/20-tide-decentralized-neural-network/positive-delay-graph-finite-cut-learning-note.md)

本仓库采用 SettleGraph，不因此宣称已实现一般 TimedDAG 或有环 Graph 的执行器。上游后续提交也不会自动成为当前实验的依据。

## 共同边界与对象映射

共同边界包括固定有限图、完整输入确定后选择、active 属于 candidates、每份可变状态唯一归属，以及不依赖 Full 结果的状态延续。固定参数可以共享。非候选节点不自主更新或发送；时间衰减按统一逻辑时间解释保存值与时间戳，不生成空纤维事件。

下表把[本地实验文档](experiment-semantics-and-naming.md)中的操作对应到上游角色。具体公式、输入域和记录关系仍须在每个声明的实验实例中确定。

| 本地对象 | 上游对应与需要保持的含义 |
| --- | --- |
| receiver、固定边、region、reached candidates | 节点、消息边、区域划分，以及由非空输入确定的候选集合；区域共同选择的依赖必须完整声明 |
| `DATA` / `CLOSED`、父消息收齐 | 存在的数据值与已确定无数据的结算结果；`CLOSED` 不是消息数值，不参加 Agg |
| Aggregate、入口归一化 | 从完整输入得到本地内容的纯函数；严格采用上游 SettleGraph 时，Agg 只读取按固定顺序排列的有值序列，不额外读取缺失消息的边身份 |
| Update 与 content/pre/post Read | Upd 候选新状态及对应的描述量读取模式 |
| Score、Top-K、概率及历史 | SelStep 给出 active set、每候选局部控制量和下一 selector-history；soft 概率作为数值控制，不自动表示随机选择 |
| Observe/commit、状态采用 | 确定本次计算快照；它与下一持久状态是两个角色 |
| 下一持久状态、选择统计写回、选后清理 | 通过 Next 或区域 SelStep 声明；若写回只由本次选择及已知量决定，可以在 Full 返回前确定 |
| NodeCompute 与 Emit | 可合成为 Full 的输出函数；本次完整计算读取快照与本节点控制量，不回写持久状态 |
| 终端消息聚合与 Base placement | 对声明输出的应用层读出及模型组合；具体张量公式由本地定义 |
| 跨 Token、跨 chunk 的持久状态 | 同一序列的延续边界；完整逻辑切面还涉及绝对时间与跨界消息，不能只按一个 KV 容器判断对应 |

“在完整计算后写回”若仅表示程序先后，不构成 Full 反馈；是否属于当前核心要看写回函数的自变量。Next 不能调用或重算 Full。以 Full 结果改写本节点状态或 selector-history，再影响以后计算或共同选择，以及更宽的有状态 NodeChunk，只在上游备忘讨论，本仓库未通过本页采用这些扩充。Full 输出沿正常消息边进入后续区域，则属于既有数据流。零时延边和隐式共同读写可变 KV 不属于本仓库采用范围。

Token 位置、HB Line 层级与设备执行次序分别记录；向上游解释时须给出统一逻辑时间日程。局部函数忽略逻辑时间时，可以选择合法编码证明前向对应；启用时间衰减后，必须固定编码与时间差。按 Observe 次数更新的 EMA/GDN 和最近若干次 Observe 的窗口，是另外明确的状态更新规则，不能在迁移中静默改成按经过时间衰减。

## 本地选择、限制与扩展

| 项目 | 与上游的关系 | 本地应声明什么 |
| --- | --- | --- |
| 单输入输出 hidden、Base checkpoint 与四种 placement | 模型接入实例 | 维度、接入位置、参数装载和函数保持初始化范围 |
| 有界度、有界局部成本、单层或 HB 拓扑 | 实验构型与成本目标 | 对哪些规模变量保持上界；不将某种拓扑升格为一般 Tide 公理 |
| N/SD/BO 与标准 timing 组合 | 状态采用规则的实例及本地子集 | 当前标准 SD 不含 post 是本地限制；上游能表达 SD/post，不代表现有资格范围已包含它 |
| 只读有值序列的 Aggregate、同一 hidden 复制到全部出边 | 上游 SettleGraph 输出函数的受限选择 | 完整公式、值序列顺序及广播范围 |
| 读取父 edge ID 或终端 node ID 的 Aggregate | 本地扩展；一般 TimedDAG 的带标记原子可以容纳这种信息，但当前上游 SettleGraph 的 Agg 不读取身份 | 逐公式登记所读身份、缺失消息规则、参数成本和所采用的 TimedDAG 映射；相应 fixture 不标为严格 SettleGraph 实例 |
| EMA、GDN、窗口 Attention 等 | 具体状态函数 | 首状态、shape、更新、读取、清理和成本 |
| 显式固定参数共享 | 上游允许的参数关系 | 操作到逻辑参数身份的映射；状态 owner 仍然独占 |
| HARD/HST/SOFTP | 控制量与输出函数的选择 | HARD/HST 前向相同，HST 的替代反向独立声明；SOFTP 同时影响前向值 |
| LM/balance loss、mask、detach | 应用和训练契约 | 统计范围、梯度规则与序列边界；不从前向等价推出训练等价 |
| 输入提供 K 等本地扩展接口 | 超出上游 SettleGraph 规格内固定 $K_j$ 的实例 | 控制输入在何时确定、怎样进入函数，以及对应实现和证据范围 |
| 额外跨区域控制依赖 | 仅增加等待顺序时可由更严格的逻辑日程表达；传递额外数据时属于本地扩展 | 区分调度约束与函数新增自变量，登记数据来源和记录坐标 |

本仓库保留上述已有本地扩展，但它们不修改 tide-core-2，也不能借用上游 SettleGraph 的同名定理。每个 Plan、fixture 和实验应分别说明自己落在严格采用子集还是某项本地扩展中；本地 `core-v1` 的资格名称本身不构成上游符合性标签。以后再增加变化时，应列出改变的函数、自变量、状态所有权、时间或观察契约，再判定它是合法实例、较窄限制，还是需要另立版本的语义扩展。`CUSTOM` 名称或通过一个本地 validator，都不能替代这个判断。

## 几种版本各指什么

| 名称 | 版本对象 | 与其他版本的关系 |
| --- | --- | --- |
| `tide-core-2` | 上游抽象接口、依赖边界和教材 | 本页固定的共同语义；不是软件完成度 |
| 本地 `core-v1` | 已声明的实现与资格子集 | 保留其固定 K、配置及测试范围，不因采用 tide-core-2 自动变成另一个子集 |
| 本地 `extension-v2` | 本地扩展接口和资格进入范围 | 不是上游第二版；其中的固定参数共享可仍是 tide-core-2 的合法实例 |
| Plan schema 1 / 2 | 本地规范化图与参数引用的表示格式 | schema 1 保持既有独立参数范围，schema 2 显式声明参数共享；不能代替数学语义版本 |
| parameter schema v1 / v2 | 参数身份、slots 与绑定的描述格式 | 对应独立参数或共享身份的记录需要，仍须与 Plan、具体公式及代码提交一起解释 |

旧 run 和 fixture 保留原有版本、哈希和证据身份。更新本文或锁文件不自动改写其解释，也不表示已有序列化工件新增了语义版本字段。

## 符合性需要分别说明的层面

| 层面 | 要回答的问题 |
| --- | --- |
| 文档采用 | 固定来源是否明确，本地限制与扩展是否已区分？本页和锁文件处理这一层。 |
| 数学实例对应 | 所选函数、时间编码、输出及延续状态是否落在上游定义内？引用上游定理时是否满足其前提？ |
| 数值执行一致 | 在声明输入域、dtype、归约顺序与容差下，输出、路由、状态、消息和继续执行是否符合本地实例？ |
| 训练契约一致 | HST、stop-gradient、detach、初态、参数共享和实际反向目标是否保持声明的梯度关系？ |
| 已取得的工程与实验结果 | 哪个代码提交、fixture 和运行证据支持哪项正确性、性能或学习结论？ |

这些层面需要不同证据，没有从文档采用到实现通过的自动推论。上游节点级时间批定理也不直接证明控制扫描低 span、设备吞吐或训练收益。已有结果以[开发验证状态](executor-equivalence-development-status.md)中绑定的提交和运行范围为准，正式资格按[测试契约](equivalence-test-contract.md)与[资格计划](core-v1-qualification-plan.md)分别判定。

## 更新与历史追溯

采用新上游提交时，应检查定义、状态时序、逻辑日程与比较对象的变化，更新本页和锁文件，并说明哪些已有实验实例保持原义、哪些需要修订或重新验证。保留旧记录原先绑定的提交；不把文档改版当成旧实验的新资格。

[采用 tide-core-2 前的实验语义备份](archive/experiment-semantics-and-naming-pre-tide-core-2.md)仅用于解释历史材料，**不是现行规范**。当前入口见[文档导航](README.md)，当前具体实验定义见[实验语义文档](experiment-semantics-and-naming.md)。
