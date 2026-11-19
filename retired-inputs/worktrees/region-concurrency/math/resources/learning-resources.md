---
type: learning-resource-index
status: optional-reading
tags:
  - tide
  - resources
  - learning
---

# 按问题选读的学习资源

从 [数学教材](../README.md) 出发，按遇到的问题选择本页书目与练习。

## 1. 选择材料

| 想理解的问题 | 中文入口 | 英文入口 | 重点 |
| --- | --- | --- | --- |
| DAG、可达性与 SCC | 《算法导论》图算法；《离散数学及其应用》按需 | Jeff Erickson, *Algorithms*；CLRS | 关系、路径、拓扑序、缩点 |
| 同一张图为什么能有不同语义 | Lee–Seshia《嵌入式系统导论：CPS 方法》 | *Introduction to Embedded Systems*；Van Roy–Haridi, *Concepts, Techniques, and Models of Computer Programming* | 状态机、组合、逻辑时间、通信模型 |
| 为什么有 DAG 仍可能很慢 | 陈国良《并行算法的设计与分析》；《结构化并行程序设计》 | Grama 等 *Introduction to Parallel Computing*；Blelloch 的 scan 文献 | work/span、归约、prefix、通信 |
| 为什么 kernel 形状很重要 | 《深入理解计算机系统》；《大规模并行处理器编程实战》 | CSAPP；*Programming Massively Parallel Processors* | cache、带宽、tiling、profiling |
| 如何在不改语义时优化 | Cooper–Torczon《编译器设计》；《程序分析原理》 | *Engineering a Compiler*；*Static Program Analysis* | def-use、SSA、数据流分析、translation validation |
| 如何证明过去不会再收到消息 | 《流式系统》；《数据密集型应用系统设计》 | *Streaming Systems*；DDIA；Naiad | 时间、seal、进展、状态与恢复 |
| 模型与梯度怎样工作 | 邱锡鹏《神经网络与深度学习》；《动手学深度学习》 | Simon Prince, *Understanding Deep Learning*；D2L | 自动微分、RNN、Attention、优化 |
| 图计算怎样学到东西 | 《图神经网络导论》 | Hamilton, *Graph Representation Learning* | 消息传递、归纳偏置与实验对照 |

中文与英文都是可独立选择的入口，不必成套同时读。章节号随版次变化，按主题定位更可靠。

## 2. 可选的本地阅读副本

个人阅读用 PDF 可自行存放于 `20-tide-decentralized-neural-network/appendices/books/`。这些副本不随仓库分发；目录是否存在、有哪些书籍，以本机文件为准。

可按 CPS、计算模型、并行处理器、结构化并行、编译器、静态分析、流系统与数据系统等主题选择阅读材料。相同书名的不同文件可能是不同版本或加工本；按实际阅读需要选择，不凭名称推断内容相同。

## 3. 不写程序也能完成的练习

1. 给四个节点、三条正时延边和两个输入端口，手算每个非空时间纤维；改变物理到达顺序，说明哪些数学对象不变。
2. 给同一区域两个候选，分别使用内容、旧状态、候选状态评分；比较激活集合与状态采用。
3. 加入选择历史和一次 Next 清理，分别列出本次 Full 的读出快照与下次持久状态。
4. 在一条跨越切面的边上留一条消息，比较保存/丢弃它后继续运行的差异。
5. 构造空间 DAG，但区域商图有环；解释为什么这没有让实际事件倒流。
6. 构造一个正时延自环：整体不停止，每个有限切面仍只有有限作用。
7. 在 $u\to v\to w$ 上把 $\{u,w\}$ 收成一个块，说明任意 DAG 分区收缩为何可能有环。
8. 手算三个仿射转移的两种括号顺序，区分“结合律成立”和“组合便宜”。

## 4. 想通过实现加深理解时

每次只实现一个清楚问题，不必先搭通用执行系统：

- **图算法**：两种拓扑排序、小图可达性、SCC 及缩点；用暴力可达性作独立对照。
- **教材 interpreter**：逐 $\theta$ 计算时间纤维、选择、状态与消息；比较两段继续与一次运行。
- **Kernel lab**：cached attention 与 causal bulk；标量/对角仿射递推与 scan；比较输出与终态。
- **控制与批次**：为一个区域给定窗口内完整输入纤维与左边界状态，按[局部联合契约](../local-semantics/weighted-message/architecture-learning-note.md#7-切面与联合求值)先求状态和选择，再收集逐节点 Full 的输入；观察控制依赖长度与 Full 批数。完整外部序列未必已经确定内部纤维，分层实例须取得上游 Full 结果后再处理下一层。
- **成本测量**：固定输入、硬件与 dtype，分别测算术、packing、通信及总时间，不只比较 launch 次数。

开始哪一个取决于当下问题。Python/NumPy 足以完成多数语义小例子；PyTorch、C++ 或设备 kernel 在确有需求时再加入。

## 5. 一页学习记录

对一个概念，可以分别写下：自己的定义、一个完整例子、命题前提、简短证明、去掉前提的反例，以及仍不能推出什么。若实现了程序，再附输入域、独立 oracle、测试结果和成本。

六种证据应分清：数学证明、模型模拟关系、复杂度分析、有限测试、硬件测量、训练实验。它们各自有价值，但不能彼此替代。与 AI 协作时，可以先自己手算，再请它找反例或检查漏洞；无需把读书进度转成强制交付流程。

## 6. 免费入口与一手材料

- [Erickson: Algorithms](https://jeffe.cs.illinois.edu/teaching/algorithms/)
- [Lee–Seshia: Introduction to Embedded Systems](https://ptolemy.berkeley.edu/books/leeseshia/)
- [Static Program Analysis](https://cs.au.dk/~amoeller/spa/)
- [Blelloch: Prefix Sums](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)
- [Naiad](https://www.cs.princeton.edu/courses/archive/fall22/cos418/papers/naiad.pdf)
- [神经网络与深度学习](https://nndl.ai/)
- [动手学深度学习](https://zh.d2l.ai/)
- [Understanding Deep Learning](https://udlbook.github.io/udlbook/)
- [Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/)
- [机器学习系统：设计和实现](https://openmlsys.github.io/)

外部术语背景另见 [编译器与 dataflow](../memos/background/compiler-and-dataflow.md)。书籍和论文帮助理解前提，不替代本仓库对具体对象的定义。
