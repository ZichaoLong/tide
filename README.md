# fractal-latcarf：SettleGraph 实验平台

本仓库是 [20-tide](https://github.com/ZichaoLong/ObsidianVault/blob/facf1afc696673a80e49b1e327bb1d058273893b/20-tide-decentralized-neural-network/README.md) 的下游实验平台，研究从已有语言模型 checkpoint 生长出的 SettleGraph：具体模块与拓扑怎样选择，执行器是否保持同一模型，以及局部选择、私有状态和 BO 是否带来可复现的训练或推理价值。

20-tide 负责 Graph、TimedDAG、SettleGraph 的抽象定义、数学教材和一般定理；本仓库负责具体实验实例、模型接入、实现、等价性验证与测量。当前采用上游 **tide-core-2**，固定提交为 `facf1afc696673a80e49b1e327bb1d058273893b`。采用范围、对象映射和版本区别见 [上游语义引用](docs/upstream-semantics.md)，不可变来源见 [语义锁文件](docs/upstream/tide-core.lock.json)。

## 从哪里开始

| 想了解什么 | 入口 |
| --- | --- |
| SettleGraph 是什么，怎样包含于 TimedDAG | [固定版本的上游 SettleGraph 教材](https://github.com/ZichaoLong/ObsidianVault/blob/facf1afc696673a80e49b1e327bb1d058273893b/20-tide-decentralized-neural-network/settlegraph-learning-note.md) |
| 本仓库的模型实际计算什么 | [SettleGraph 实验实例、局部公式与命名](docs/experiment-semantics-and-naming.md) |
| 实现边界和执行路径 | [实现与等价性验证计划](docs/settlegraph-implementation-plan.md) |
| 如何判定等价，以及怎样形成资格证据 | [等价性测试契约](docs/equivalence-test-contract.md)、[core-v1 资格计划](docs/core-v1-qualification-plan.md) |
| 哪些检查实际完成过 | [执行器等价性开发验证状态](docs/executor-equivalence-development-status.md) |
| 怎样开展 checkpoint 与 BO 实验 | [checkpoint 实验路线](docs/checkpoint-to-bo-experiment-roadmap.md) |

完整文档分工见 [docs 导航](docs/README.md)。这些入口按问题组织，不要求先完成全部工程资格工作才能阅读教材或开展明确标注范围的探索。

## 当前实现与证据边界

仓库已有 token-major 与 region-major eager reference、固定 K `core-v1` 的通用 packed executor，以及单层和 HB 的拓扑特化执行器。Plan schema 2 的显式固定参数共享属于本地扩展范围；可变状态仍按 owner 独占。具体支持集合、参数和状态格式以实现计划为准。

[开发验证状态](docs/executor-equivalence-development-status.md)记录了提交 `5712e66f1cf51a85360e6507839c2fe443aa81ae` 上 CPU FP64/FP32 的受控差分检查，包括输出、状态、路由、trace 和指定 VJP 目标。记录的成功终态仍是 `qualification=false`；它不覆盖所有合法配置，也不因本次采用上游语义而自动成为 tide-core-2 的实现符合性证明。

完整正确性与性能资格、真实 Base/Qwen 的端到端接入、selector-history、可学习首状态及完整训练恢复等仍有未完成范围。已有探索性性能记录不能证明当前 packed decode 加速，也不能替代目标设备、低精度或训练实验的独立证据。最新的范围说明和后续工作保留在状态文档与计划中。

## 研究问题

- 新结构能否从保持原 checkpoint 行为的初始化开始，稳定学到有用的变化？
- 私有状态是否被以后计算使用，BO 相比 SD 的额外写入是否有价值？
- 内容、旧状态和候选新状态的选择规则，以及多父汇合和局部拓扑，分别带来什么影响？
- 稀疏计算节省的工作，能否覆盖状态、选择、消息、packing 和通信成本？

固定拓扑、有界局部成本和可达容量增长是本平台关注的构型目标；单层 receivers、HB、具体状态模块和训练损失是可比较的实验选择。当前没有证据证明 SettleGraph 或 BO 优于 Dense、Flat MoE 等基线，也不预设 HB 是唯一或最优拓扑。

采用上游前的语义原文保存在 [历史备份](docs/archive/experiment-semantics-and-naming-pre-tide-core-2.md)，仅用于追溯旧记录，**不作为现行规范**。
