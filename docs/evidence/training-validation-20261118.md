# 训练交付：集成检查、隔离安装与身份

本轮生产包、命名训练预设、路径诊断、完整计时和证据已集成。数学固定
`4f5b9f20a4030d68ee66ae2d26e34659a5e37234`，108文件核验通过。
本页的测试、产物、全部本轮作业终态、归档核对和用户文件哈希见
[JSON](training-validation-20261118.json)。数值失败仍由
[路径](training-paths-20261118.md)、[连续恢复](training-continuous-20261118.md)
及本页安装结果分别保留。

## 源码与检查

生产 `src/tide` tree 为 `d3e8a6cb40eceaf487dc642a4283d58f90d5726f`。
CPU完整套件、后续定向检查、已安装wheel和交付代码的生产树相同；之后的提交增加
harness、配置、计时修复和文档，不能把较早的完整测试计数改标成后来全树测试。

| 检查 | 实际冻结身份 | 结果 |
|---|---|---|
| 完整CPU套件 | `ae247dee` | 625 passed，1 warning，370.70秒 |
| 入口、安装判定、路径、Attention见证、continuation定向集成 | `32ab13a5` | 83 passed，70.56秒 |
| forward-only异步拼接计时回归 | `f8235fe4`对应工作树 | 2 passed，0.09秒 |
| 当前配置目录元数据 | `5e61ec74`，同生产树 | passed；5510个预算合法性检查 |

完整套件的warning为Torch profiler周期事件提示，不是算子执行失败。新增候选的
独立检查包括仿射递推值/全部输入VJP、SSD literal oracle、零/一衰减、空写入恢复、
大int64时钟、共享BPTT、多步与恢复。目录检查不构造模型张量，不证明整个菜单的
所有组合都具有数值或硬件资格；正式范围仍按各测点记录。

最后的缓存计时审阅发现：`run_stream`已同步每次前向/flush，但最终`torch.cat`
在推理与含图前向下缺少完成同步。`f8235fe4`修复此边界，训练分支保持原有更新同步。
两个模拟异步设备的回归检查验证拼接等待确实进入complete时间；仅重测受影响的
两组缓存programme。旧原始记录及原边界保留，新的正式表见
[模式对照](training-modes-20261118.md)，不把新边界追认给旧结果。

## 新wheel与公开入口

从冻结 `32ab13a5` 归档的**单独副本**离线构建wheel，再安装到全新私有venv。
公开Ascend/Python基线只读，禁用vendor autoload；检查实际使用Python3.12.13、
Torch2.12.0+cpu。wheel SHA256：
`5fafa17e72d4ccfeca45cfd78e317faadcad5e6faacb524ec355bf4094c29399`。

从`/`移除PYTHONPATH并用Python `-I`运行；21个安装Python文件与wheel/冻结源逐一
匹配，最终加载的21个Tide模块也全部匹配。每次恢复在独立解释器再次检查安装身份。
原355文件归档运行前后未变；构建只写副本。examples/configs是checkout工具，
wheel安装`tide`包，不谎称把示例文件也打进wheel。

安装harness使用`--execution-variant training`，即prefix摘要、Reception局部重算、
local Attention auto；batch路径为相容区域batch4，placed路径为两个逻辑CPU组、
batch1和per_device。共同使用compact-9、n2/hubs2、d64、X、Timed period3、terminal
Hubs及显式时钟，B1/L4，`2+2`分段保留完整梯度、flush后detach。三更新后保存，
原进程和新进程分别执行第四次更新。它是机制检查，不是性能测量或双NPU资格。

| 安装案例 | 推理/含图前向最大绝对差 | 新进程第四次梯度 | 新进程第四次参数/optimizer/state |
|---|---:|---|---|
| batched FP32 | 4.7684e-6 | 精确 | 精确 |
| batched BF16 | 0 | 精确 | 精确 |
| placed FP32 | 9.5367e-6 | 原阈值内 | 原阈值内 |
| placed BF16 | 0 | 154个张量超差，最大1.46294e-3 | 140个张量超差，最大1.46294e-4 |

四例保存数据、配置和下一前向精确，dtype、None梯度语义及实际step3→4保留；
batched的204个、placed的409个活跃参数张量实际更新，全部数据有限。
恢复阈值仍为`atol=1e-6, rtol=2e-3`。模式比较额外用0/0观察逐位相同，
故两项FP32非精确使案例标记finite_deviation；这不能当成一般FP32容差失败，
也不能把placed BF16的真实恢复超差改为通过。服务exit0允许记录有限偏差，
安装内部总体状态仍为`finite_deviation`。

同一新安装环境完成了无参数CPU三步入口、六个NPU命名预设的显式设备dry-run，
以及公开入口上的Settle合批/prefix/Reception两步训练、精确保存/下一前向、
阈内配对更新和另一个新进程从保存点的实际更新。NPU dry-run不初始化设备，
只是验证配置；NPU资格来自独立真机记录。入口的合成数据/有限检查边界不同于性能
设施，不能用入口elapsed替换表中吞吐。

安装作业使用background.slice、CPU affinity84–87、ATen/interop1线程、900秒外界限、
24 GiB采样进程树RSS界限，实际101.4秒。没有修改公共包、参考或下游环境。

## 保存、终态与复现

JSON逐一记录本轮`training-*`服务的真实退出状态、MainPID和已跟踪后代身份，
并对每份冻结归档的Git tracked blobs与声明commit逐一核验。检查时无活跃Tide任务或
仍存活的已记录后代。保留两个错误CLI尝试、Timed BF16结构比较失败、预算停止及
所有有限超差，不能用外层通过覆盖内部结果。

四份用户未跟踪文档SHA256与起点一致，未纳入提交。没有删除本轮checkpoint、anchor、
trace、wheel、私有安装或代理工作树；这些仍是复现依赖。没有触及参考仓库、公共环境
或下游发布；下游验收不由本轮仓库检查替代。

本轮完整模型仍是PyTorch；既有C++17/LibTorch区域评分切片继续保留原资格。
CPU/NPU有实际记录，CUDA、peer DMA/HCCL、全模型C++、编译/resident训练没有新增资格。
上游主绑定菜单、可选Hub和时钟可表达不等于全部组合已穷举；数学快照中的非选定实例
不是当前能力清单。

复现入口是[训练建议](../execution-recommendations.md)和本轮各证据JSON中的冻结命令。
安装JSON保存了完整driver文本/哈希、构建/安装/入口命令与manifest；使用新的输出目录，
离线构建副本、`--no-deps --no-index`安装、`-I`从`/`验证。硬件实验重新依
[执行契约](../execution-flows.md)准入，不沿用过去的设备占用许可。
