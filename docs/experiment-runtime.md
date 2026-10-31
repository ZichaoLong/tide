# 公共实验运行与返回契约

`tidegraph run/check/matrix/job/results` 使用安装包 `tidegraph.experiment`。
模型来自公共 planning/materialize；参考仍由独立 Python 标量调度或独立 C++ Streaming 调度执行。
候选先运行，之后的参考重放不向候选提供 ready、选择、路由或批次。
图层级视图校验见 [experiment-validation](experiment-validation.md)；运行记录另列实际调度器。
同一 ranked 实例的 PDG/TimedDAG/Settle 视图不会伪装成三个独立特化。

## 请求、运行和验收

完整展开配置、请求/生效执行值、模型/工作负载/任务/执行身份、安装文件与实际构建 hash
分别保存。`validation=strict|explore|none` 是验收策略，不改变数学或调度。
严格失败仍非零退出；explore 只接纳有限浮点超差且完整离散比较通过。
非有限、结构、求解器、通信或容量异常都失败；未知异常只记录后重新抛出。
`none` 明确不运行独立参考，不能把执行成功写成数值通过。

每窗从静态 task 输入生成新的独立样例；CPU 输入的真实广播别名保留为同一根，
按实际节点所有者可微复制。训练在该窗 loss、VJP、有限梯度检查与一次统一 optimizer 后 detach。
固定默认平方输出诊断 loss、SGD 与 FP16 master 策略不变；新任务配方拥有新的 task 身份。
推理不创建 optimizer 或扫描参数梯度。
每窗也检查任务自身的张量状态是否有限，不要求任务带有 `task.module`；
`validation=none` 和按需返回都不能绕过该运行检查。

CPU 线程区分 ATen/BLAS intraop、interop 与调度 workers。
CUDA 运行显式关闭 TF32；NPU 显式关闭矩阵/卷积 HF32，并记录生效值。
输入材料按同一 seed 生成 FP32，再按请求显式转换；没有隐式精度或算法替代。

`packing=scalar|packed` 控制公共 eager 的 Aggregate/Full/Read 批路径与一阶图构造策略。
scalar 仍可使用局部函数内部 ATen、共享计算和标量公式已有的优化，不是关闭所有优化。
`auto` 仅解析为既有路径：Python streaming 为 scalar，其他可用 eager 为 packed；不另算算法。
`return_packing` 只改变完整返回的搬运，不能与数值 packing 混淆。
resident 使用其公共设备库，拒绝 eager scalar/return-packed 覆盖。

## 完整与按需返回

| return_mode | 返回内容 | 独立资格与计时 |
| --- | --- | --- |
| complete | 完整 state、history、pending、控制、物理消息和端口输出的 CPU 记录 | 可在该流程完成完整独立比较；保留旧完整返回语义 |
| outputs | 实际端口输出及 cut，状态仍归运行 owner | 独立配置；运行要求 validation=none，先用 check 单独验完整记录 |
| metrics | 实际工作计数/cut及声明的任务标量，状态仍归运行 owner | 独立配置；不把少返回的成本与完整返回混算加速比 |

Eager 仍构建内部完整记录；按需返回不声称已消除内部对象或 autograd 图。
resident 仍保留完整 device journal；按需返回避免不需要的 CPU 解码，不能把它当成 journal 容量压缩。
resident inference 的 metrics 不额外计算诊断 loss；训练始终计算所声明的 loss/根/optimizer。

Python运行将所需端口输出按窗保存为 `outputs-N.pt`（metrics无此文件），不将整个诊断轨迹长期落盘；输出持久化成本单列。

窗口时间包含输入准备/传输、调度、packing、计算、有限检查、所选返回及同步；
训练另包含反向与 optimizer。构造、warmup、checkpoint 写盘、独立参考和恢复检查分别计时。
profile/工作形状观察用独立运行，不作为正式吞吐。样例数 B、原始逻辑时间、实际算子行数分别记录。

## 续接与停止

每个完整窗口在计时外保存新的 checkpoint generation。
新 state/task/manifest 经 fsync、原子发布与 latest 指针替换后，才退役本次 writer 的上一 generation。
强停可恢复最后已发布的边界；进行中的窗口不承诺保存。磁盘预算须包含新旧两代，写失败保留旧边界。
加载前校验模型/任务/执行、文件 hash、安装和 binary 身份；新输出目录不覆盖旧运行。
Python 实验恢复检查按顺序关闭旧 owner 后建立新 owner，完整解码和下一窗逐值相等，
另比较参数、梯度、optimizer 槽和 task 状态。跨平台 checkpoint 可移植性是另一项资格。

`check smoke` 是显式小模型的一窗独立推理及恢复；related 增加相关独立根与两窗训练；
full 增加声明的多根、非整位置 raw-state 根和三窗训练。具体库存和未测项写入结果。
这些是选定有限配置的安装基座检查；完整仓库/多卡 resident 门禁仍见 [verification](verification.md)，
不能由安装 smoke 或运行退出码替代。

## 普通复制、blocking 与 CPU 中转

`execution.communication` 固定于每个实验进程，Python/C++ 共用同一含义：
ordinary 使用普通 Torch/ATen `.to`；blocking 在显式跨设备复制前后同步两端；
cpu_bridge 在不同加速器间显式执行 accelerator→CPU→accelerator 并同步端点。
它们不使用新通信协议，不宣称直接 P2P、NCCL 或原生跨卡原子。
原生跨卡 resident 仍走原硬件能力门禁，不能回退成主机中转。

实现位于 [communication.py](../python/tidegraph/communication.py)、
[communication.cpp](../cpp/src/communication.cpp)、两端 transfer 与 weighted placement copy。
每条自定义 VJP 边保存实际策略；None/零梯度、原始版本和 int64 元数据均保留。
低层未配置调用保持 ordinary；实验进程通过显式 `TIDE_COMMUNICATION_POLICY` 传递给 C++ worker。
不同策略使用独立进程，不能在活跃图之间修改环境。

计数覆盖公共 placement/message 复制：logical_bytes、tensor_leg_bytes、复制次数与端点同步次数。
这些是 API 张量字节与显式两跳计数，**不是 PCIe 实际流量或底层 P2P 证据**。
初始化、返回、allocator 内部复制和厂商隐式通信另有计时/profile，不混入该计数的完整性声明。

[人类操作手册](human-workflow.md)给出安装、锁、预算、作业与报告流程；
[逐卡预算](experiment-placement.md)区分静态银行、动态估计、resident 物理布局和真机资格。

`measurement.observation=packing|profile`要求一个测量窗，使用独立进程，
复用公共packing计数与厂商profiler；`none`为正式测量。观察结果即使独立比较通过，
也不具有吞吐资格，HTML/CSV不把其秒数显示为正式吞吐。standalone支持packing，
设备时间线用外部profiler或同配置Python/native独立profile。
完整返回的逐样例活动、消息及raw KV长度在计时后记录；B仍是独立样例数。
`resources.live_results`预算设备同时保留的结果，`host_results`另计比较/恢复的CPU副本；
按实际窗口数自动保留足够主机记录，不能把已解码CPU记录重复算成连续设备结果。
CANN所需`PYTHONPATH`可由站点显式给出；持久任务不继承临时源码检出的该变量。
