# 有界共同原点 KV 位置与一般图异步资格

数值实现固定`fb83ee91cc34ba49bf00465916d6e7e7a7eacdec`，承接a605的位置策略并修正构建调用类型；
数学固定`0770593838a37c790a568226930567ed60f99b83`。
本报告另记录**55a3074a**的一般图双卡后继，两份源码资格分开。
[机器记录](rope-qualification-20261112.json)含逐项结果、实际命令、构建/包身份、失败见证及来源hash；
[42点性能与诊断](rope-performance-20261112.md)另列，成本完成不等于严格等价。

| 范围 | 实际终态与范围 |
| --- | --- |
| fb83 Python-owned native CPU | 281 passed、9 原FP32 xfail；实际290个JUnit testcase |
| fb83 standalone SDK CPU | column、owner、实际CLI及3个CTest通过 |
| fb83 standalone SDK NPU | 五记忆×mixed-a/b/c共15项通过，独立Streaming/raw/VJP/两个SGD窗口 |
| fb83 LH176非空左态 | CPU FP64 task/messages/raw三族数值投影通过；NPU FP32 messages/raw两族严格有限数值失败 |
| fb83 独立安装 | 位置小图/LH176 × CPU/NPU × Python/native，共8项三步AdamW及新进程精确恢复通过 |
| 55a 一般图双NPU | delta-pdg、ssd-dag、kv-pdg-initial × Python/native，6格通过；显式每流队列、完整字段/根、四次AdamW及恢复 |

## 语义与代码对应

[窗口位置契约](../weighted-tensor-windows.md)采用`bounded-common-origin-v1`，默认执行身份为v6。
固定教材KV的旋转恒等式`Rot(p)ᵀRot(q)=Rot(q−p)`允许在同一整数原点下分别旋转Q、K后矩阵乘，
避免为每个query展开一份全部历史key。raw key及原始逻辑时间不变，距离/可见性仍单独用int64差值。
这改变浮点求值顺序，不保证与逐对相对旋转逐位相同。

共同原点首先在int64域内相减。FP32/FP64整数跨度分别须≤2^24/2^53，且最大绝对相位P满足
`8eP ≤ sqrt(e)`（e为相应dtype的epsilon）。该限制是表示/相位误差的工程界，不是输出或路由
误差证明。超界沿原相对展开，FP16保持原half路径；并非放宽容差或捕获计算异常后换算法。
旧v5列身份需按契约显式迁移，原记录接口与完整观察能力分别声明。

代码为[Python位置求值](../../python/tidegraph/weighted/column_position_ops.py)、
[C++位置求值](../../cpp/src/weighted_column_positions.cpp)，与此前的相容节点Read/Full分组组合。
[独立位置检查](../../tests/test_column_position_ops.py)共17项，覆盖字面相对旋转公式、巨大绝对时钟、
非空/异形查询、伴随、None/零、整数/相位超界和half明确后备；原有一般端口、共享、训练/恢复等检查
包含在此次281项通过中。`test_query_sdk_capability.py`的9项检查实际interface/schedule及输出列/完整记录分支；
不能把旧record计数器套在列路径上。

## 构建、CLI与原失败

本机CANN9.2、aarch64/GCC10.3.1、Python3.12.13、Torch2.12.0+cpu/TorchNPU2.12.0.post3；
standalone另用LibTorch2.12.0及NPU SDK。两个fb83核心分别固定源码构建，随后生成wheel并独立安装。
原a605两次构建都在计数器`std::string→const char*`调用处失败，没有运行数值资格；fb83修正后重建通过，
旧failed终态保留。CPU九个xfail延续此前一般端口FP32见证，未改阈值、未计入passed。

SDK CPU实跑`column-check`、`column-checkpoint-check`和`weighted-run-check`，
三个CTest为`weighted-comparison-check/weighted-run-check/weighted-roots-check`。
SDK NPU的实际命令仅为`column-check --device=npu:0 --dtype=float32`，15组合全部通过；
**不包含NPU的column-checkpoint-check新进程owner资格**。跨进程NPU恢复由下述安装消费者证明。

## 非空LH与独立安装恢复

LH材料仍为176节点、22区域、1058主体边、B4/L16、d64、14,902,694可训练参数。
这是异构textbook配置：56节点qry、56节点sum并带部分位置频率、64中间节点sum且window0；
区域为14个att与8个ind。此前报告列出的首节点qry/att设置不能代表全图，完整划分见性能报告和JSON。

非空投影继续每节点3条字面左记忆、起始位置2^40，独立公式→公共接口→默认入口分别解码实际结果。
CPU FP64的task/messages/raw三族均在`atol=1e-10/rtol=1e-8`通过数值投影；
仍分别有20/26/20处None与连接零差异，全部已连接梯度数值为零。不据此认证带已有动量或
weight decay的直接公式/候选优化器轨迹等价，也没有把上一轮七族全部重新认证给fb83。

NPU FP32两族在`atol=1e-6/rtol=1e-5`严格失败。候选/活动支持、raw整数域通过，值与梯度有限，
但下列数值字段不满足原阈值。public/direct与default/direct比较记录相同。

| 超差张量/字段数 | messages根 | raw根 |
| --- | ---: | ---: |
| controls | 119 | 119 |
| raw浮点字段 | 2127 | 2127 |
| 端口输出 | 8 | 8 |
| 所选根的值 | 1688 | 2127 |
| 梯度 | 2771 | 2794 |

首端口见证最大绝对差约2.66552e-4；输入梯度见证分别0.03515625/0.06709290。
这些是保留的有限数值失败，没有在此证明原因仅为舍入，也不以计数比上一轮多或少判断策略更准确。
NPU生命周期外层作业因此仍为failed；其内部四个安装消费者各自passed。

独立安装使用`core-rope-install`，八项均实际选择`settle-columns-v1`、报告新position policy且没有后备。
位置小图逐步对照CPU参考；LH计划图检查实际输出任务/梯度、参数、raw/pending/history/ledger及AdamW槽。
每项三步AdamW，并以三个新进程分别执行连续三步、前两步保存、载入第三步；最终对象精确比较
`atol=rtol=0`。这些使用默认初态/起始位置，不是非空2^40投影的重验。
消费者deterministic=true的策略被保留，不扩大到未测的执行策略。

## 55a 一般图双卡：6格而非任意图证明

本节源码固定`55a3074a441d77720cf00e44433bd702cd09e175`，不是fb83位置策略资格。
实际物理1与9，前者空闲、后者经当前对话共享许可准入；payload/scoring FP32、mixed-c、workers1、packing auto。
显式`TASK_QUEUE_ENABLE=1/PER_STREAM_QUEUE=1`，`ASCEND_LAUNCH_BLOCKING`未设置。

| 配置（均Python/native通过） | 图/记忆 | 独立raw根数 | cut处pending | 更新/恢复 |
| --- | --- | ---: | ---: | --- |
| delta-pdg | 3节点、5边、2区域，反馈PDG，Read+ | 24 | 4 | 四次AdamW、精确第四步、CPU交接 |
| ssd-dag | 3节点、4边、2区域，非对齐DAG，Read+ | 30 | 2 | 同上 |
| kv-pdg-initial | 3节点、5边、2区域，反馈PDG，Read0绑定初态 | 30 | 4 | 同上 |

每格在独立进程中运行，内部恢复构造新owner后在**同一进程**比较第四步；不是跨进程checkpoint资格。
检查包含完整记录、第一输出/Describe各字段/control独立根、raw根、retained cut、实际跨卡传输、
三次AdamW后恢复第四次，以及CPU checkpoint交接。pytest/字段支持与线程池、容量、通用模块能力不能相互外推。
[普通队列EE1017](query-lifecycle-20261112.md)仍是另一策略的失败，不被这六格成功改写；没有新resident或跨卡吞吐结论。

## 复现与本报告的边界

在固定实现按[开发](../development.md)和[资格](../verification.md)建立新构建/安装目录，
沿[上一轮计划生成命令](query-lifecycle-20261112.md#在新检出中复现)重建LH材料。
CPU全定向文件清单、原SDK命令、生命周期driver及六格argv均保存在JSON；只替换新目录和设备准入，
不恢复旧queue或依赖旧raw目录。位置小图入口为`examples/consumer_core5/tensor_window.py --positions`，
LH为`planned_windows.py --plan ...`，非空根为`check_lh_tensor.py --left-entries 3 --start-position 1099511627776`。

本报告没有全尺度的严格等价结论，也没有新增CUDA/GPU、resident整图、1B训练或多卡性能资格。
有限数值失败保持独立身份；已完成的模块、安装恢复与显式异步策略证据可以用于各自声明范围。
