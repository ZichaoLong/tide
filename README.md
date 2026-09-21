# Tide

本分支 `graph-execution-foundation` 实现 CPU PyTorch/LibTorch 的图执行与等价性验证。
开发与中断接续从 [AGENTS.md](AGENTS.md)、[当前进度](docs/STATUS.md) 开始；
[路线图](docs/ROADMAP.md) 保留完整任务，[架构导航](docs/architecture.md) 定位代码，
[本地语义约定](docs/semantics.md) 和 [上游锁定](docs/upstream.json) 界定能力。
下文保留 Tide 研究总览；具体实现状态以本分支的验证证据为准。

本分支当前运行入口（CPU FP64/FP32，选择已有的匹配 Torch 环境）：

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/build.py --jobs 2
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/verify.py --device cpu --dtype both --output-dir artifacts/my-verification
./build/tidegraph-smoke --device cpu --dtype float64
python scripts/status.py
```

`--output-dir` 必须是新目录。验证包括 PyTorch/LibTorch streaming、TimedDAG 前沿、
SettleGraph 编码及拓扑特化；当前局部模块包括 EMA、identity、对角选择性 SSM、
Linear Attention、Gated DeltaRule、事件 GQA/window attention、tanh FFN 和 SwiGLU。
[原生 streaming cursor](docs/streaming-cursor.md) 可跨窗口持有状态和消息队列。
Packed 训练目前使用 [逐事件语义反传图](docs/packed-autograd.md) 保留梯度连接，
其额外计算单独计数；`no_grad` / `inference_mode` 不需要这部分重放。
具体覆盖以 [验证证据](docs/ROADMAP.md) 为准；LH 数值对齐等仍见 [后续工作](docs/module-extension-plan.md)。
源码无需导入任何本机 skill 文件；CMake 从所选 Python 的 Torch 查找 LibTorch。

Tide 是一条研究线，研究如何在固定消息拓扑上组织局部选择、节点状态和稀疏计算，并把这些语义落实为可验证的模型与执行器。

本仓库的 `main` 分支是项目总入口。它维护研究对象、上游语义、项目导航和共同验证约定；具体实现、实验路线与测量结果在独立分支或独立仓库中维护。

## 从哪里开始

| 想了解什么 | 入口 |
| --- | --- |
| Tide 的数学对象和教材地图 | [20-tide 总入口](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/README.md) |
| 当前语义版本、对象层级和上下游分工 | [语义锚点](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/semantics-anchor.md) |
| 从一次输入到有状态序列 | [SettleGraph 教材](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/settlegraph-learning-note.md) |
| TimedDAG 的完整定义和区域选择 | [TimedDAG 教材](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/timed-dag-region-selector-learning-note.md) |
| 节点级时间批与 chunk prefill | [TimedDAG 分块预填充教材](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/timed-dag-chunk-prefill-learning-note.md) |
| 允许正时延反馈环的图 | [PositiveDelayGraph 教材](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/positive-delay-graph-finite-cut-learning-note.md) |
| 已有的 SettleGraph 实现实验平台 | [`fractal-latcarf` 分支](https://github.com/ZichaoLong/tide/tree/fractal-latcarf) |

初次接触数学对象时，可以从 SettleGraph 开始；需要一般定义时直接阅读 TimedDAG；需要反馈环时阅读 PositiveDelayGraph。各教材都包含自己的阅读前提和定义顺序。

## Tide 研究什么

Tide 关注以下相互关联的问题：

- 固定的局部消息拓扑怎样表达有界邻接、区域选择和节点级状态。
- 选择、状态更新、完整计算和消息发送怎样组成确定的因果记录。
- 逐时间执行、chunk prefill 和不同物理调度怎样保持同一份规范语义。
- 状态更新与 Full 计算何时可以跨多个逻辑时间联合求值，以及联合求值的代数条件是什么。
- 稀疏激活节省的昂贵计算能否覆盖 selector、状态、消息、packing、通信和负载不均衡成本。
- 局部状态、控制寿命和跨时间信用如何影响训练稳定性与可解释性。

这些是研究问题和验证目标。具体模型是否满足它们，需要由相应实现和实验给出证据。

## 数学与语义基础

当前上游语义版本为 `tide-core-3`。详细定义、例子、命题和证明位于 [ObsidianVault 的 20-tide 目录](https://github.com/ZichaoLong/ObsidianVault/tree/master/20-tide-decentralized-neural-network)；本仓库不复制另一份“唯一语义定义”。

| 对象 | 作用 | 主要教材 |
| --- | --- | --- |
| `PositiveDelayGraph` | 固定消息图允许空间反馈环；严格正时延保证每个有限逻辑时间切面具有确定的有限记录 | [正时延 Graph 的有限切面语义](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/positive-delay-graph-finite-cut-learning-note.md) |
| `TimedDAG` | 固定消息图为无环图；统一逻辑时间、时间纤维、区域选择、状态和事件依赖 | [带区域选择的 TimedDAG](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/timed-dag-region-selector-learning-note.md) |
| `SettleGraph` | TimedDAG 的受限实例；区域依赖严格有序，每个输入位置完成一次结算 | [单次结算图 SettleGraph](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/settlegraph-learning-note.md) |
| chunk prefill | 在 TimedDAG 的作用依赖上定义状态块、完整输出批、联合节点块、区域块和最大前沿递归 | [TimedDAG 的分块预填充](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/timed-dag-chunk-prefill-learning-note.md) |

共同语义固定几个边界：消息边具有严格正整数时延；状态由唯一节点持有；选择历史由唯一区域持有；`Next` 在本次 `Full` 以前确定下一状态；没有输入的节点不会自主激活或发送。具体教材还规定端口、控制量、事件记录、continuation 和合法联合求值的完整类型。

`SettleGraph` 可以在明确的时间和边界编码下嵌入 `TimedDAG`。`TimedDAG` 的无环固定图和 `PositiveDelayGraph` 的有环固定图属于同一语义谱系，但有限切面、区域商图和时间批的结论分别以各自教材的条件为准。

## 节点级时间批与 prefill

节点级时间批同时关注状态更新和完整输出的批量处理。分析时需要分开记录：

1. 外层需要多少个因果联合块或自适应阶段；
2. 每个节点、每个 region 的状态块、Full 块和联合块数量；
3. 联合函数内部的工作量、并行深度、存储和设备代价。

分块教材以 `P/S/U/F` 作用和固定的联合求值契约为基础，在线构造相对于契约的最大前沿。它保证当前因果可取块尽早进入，并在固定宏作用图上达到相应的最少自适应阶段；总块数和全局最小批次数是另一项目标。严格分层的拓扑在额外契约成立时可以得到更强的整块结论。

因此，“前向结果能批量计算”“状态递推具有 scan 或其他低跨度结构”“硬件 kernel 实际获得收益”是三项分别验证的命题。任何实现都应同时报告其采用的联合契约和未覆盖的状态、控制与输出依赖。

## 实现与实验项目

### `fractal-latcarf`

[fractal-latcarf](https://github.com/ZichaoLong/tide/tree/fractal-latcarf) 是当前独立的 SettleGraph 实验平台，负责：

- 具体模块公式、拓扑和参数/状态配置；
- eager、packed 及适用的拓扑特化执行器；
- 从模型 checkpoint 接入 SettleGraph 的实现；
- 等价性、梯度、状态 continuation、训练和性能验证；
- 实验记录、资格边界和后续工程路线。

该分支的 [上游语义引用](https://github.com/ZichaoLong/tide/blob/fractal-latcarf/docs/upstream-semantics.md) 和 [开发验证状态](https://github.com/ZichaoLong/tide/blob/fractal-latcarf/docs/executor-equivalence-development-status.md) 是实现工作的入口。分支中的某个测试通过，不会自动成为 Tide 数学教材的定理，也不会扩大其他分支的支持范围。

其他实现或验证任务可以建立新的分支或仓库。它们应在自己的 README 中说明采用的上游提交、语义版本、对象族、局部公式、状态规则、实现范围和证据位置。

## 新项目的共同约定

一个新的 Tide 实现或实验项目，建议在开始编码前写清以下内容：

### 语义采用

- 上游仓库和不可变 revision；
- semantic version；
- 采用的对象族，例如 `TimedDAG`、`PositiveDelayGraph` 或 `SettleGraph`；
- 继承的函数、事件、状态和 continuation 规则；
- 超出上游边界的 local extensions。

### 具体实例

- 节点、边、端口、region 和逻辑时间域；
- `P/S/U/F` 或对应作用的成本标记；
- selector 输入、候选、active set、局部控制和历史更新；
- `Next`、`Full`、Aggregate、Emit 以及空输入行为；
- 状态 owner、状态容量、清空/衰减规则和 chunk detach 边界；
- 物理 placement、设备、dtype 和允许的实现变体。

### 验证证据

至少分别检查：

- 单 Token、full prefill、任意声明的 chunk continuation 和 decode；
- 输出、内部消息、route、状态、selector-history 和必要的 trace；
- 参数、入口 hidden、可微初始状态和 optimizer step 的梯度；
- `None`、connected-zero、nonzero gradient 的契约分类；
- 参考执行器、联合执行器和硬件路径的数值容差与离散结构；
- 性能测量中的工作量、状态/消息成本、吞吐、延迟、显存和通信。

验证结果应绑定源码 revision、输入/Plan identity、运行环境和未覆盖范围。数值相同不能单独证明状态所有权、事件依赖、反传边界或 continuation 语义相同。

## 分支和仓库职责

主分支保持稳定的导航与共同约定，不把某一个模型路线、硬件后端或实验假设写成 Tide 的默认实现。一个项目若需要较长的代码、专用依赖、运行记录或快速变化的实验计划，应使用独立分支或仓库，并保留自己的状态文档。

上游 20-tide 维护抽象接口、数学教材、一般定理和研究备忘；下游实现选择具体公式、尺寸、初始化、placement、kernel、训练配置和实验对照。上游教材中的复杂度或批量结论，需要下游为所采用的联合契约和调用域提供实现证据。

## 当前边界

当前项目没有一个已经被所有分支共同实现的 Tide runtime，也没有一个默认的神经网络架构。以下内容都需要具体项目单独验证：

- Tide 相对于 Dense Transformer、Mamba 或标准 MoE 的质量和性能优势；
- 一般有环图、跨设备 allocator、多卡节点调度和大规模稀疏 kernel；
- 任意 stateful selector 的高性能 chunk prefill；
- 低精度、CUDA、NPU、分布式训练、完整 Base checkpoint 和真实部署；
- 任何具体拓扑或状态 profile 的训练稳定性、节点专门化和负载均衡。

“局部连接”“正时延”“最大前沿”是数学或设计条件，不单独推出物理通信便宜、训练可行或硬件加速。

## 研究备忘与历史

[20-tide 研究备忘索引](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/memos/README.md) 汇集数学补充、selector 与局部记忆、训练诊断、脑科学与统计力学背景，以及 LH 历史。

其中：

- 正典教材定义对象、接口和已证明结论；
- 数学备忘提供局部推导与候选机制；
- 训练诊断记录尚待实验检验的风险和假设；
- 背景调查提供设计动机，不构成模型或性能证据；
- [LH 历史](https://github.com/ZichaoLong/ObsidianVault/blob/master/20-tide-decentralized-neural-network/memos/history/lh-history.md)记录早期实现和设计来源，不代表当前实现能力。

Tide 的目录名称和“去中心化”一词保留历史动机。当前数学对象和实验能力以语义锚点、具体项目文档及其可追溯证据为准。
