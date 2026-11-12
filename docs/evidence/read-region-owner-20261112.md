# Read/Describe 字段 owner 到区域计算

实现固定 **c6afc20c**，数学固定 **0770593838a37c790a568226930567ed60f99b83**。
本轮只重写主机 KV Read/Describe→区域的实际字段发布、装配和伴随；没有宣称整个 KV/区域/Full/续接已贯穿。
完整配置、构建、安装资格、原失败、20点成本和复现命令在[机器记录](read-region-owner-20261112.json)。

## 实现与独立检查

Python/native 每字段只发布一个真实行 owner，节点返回其独立只读根视图；区域对齐时借用同一存储，
不齐时仅保留真实 incidence 所需的 gather。区域伴随每字段整体回填，再返回有结构作用的节点根视图。
借用的线性 VJP 连接传入的真实根、保护原始版本；没有直接使用 storage base 绕过公开根，也不合并因果相关输出。
不同存储、重叠、重排或不同类型不满足借用条件。空列与单行即使 is_contiguous=true，原stride仍可能不能扩大；
因此借用采用经过形状证明的标准连续stride，另有零尾维、空首列和异常单行stride反例。

实现定位：`weighted/column_blocks.py`、`tensor_kernel.py`、`regional_packing.py`；
C++ `weighted_column_blocks.h`、`weighted_kv_tensor_vjp.cpp`、`weighted_regional_packing.cpp`。
[接口](../weighted-tensor-windows.md)、[只读所有权](../tensor-publication.md)、[总对应](../semantic-map.md)保持原数学/图/checkpoint身份。
直接 FC/LH 公式没有改变；真实Read生产者到区域消费者的见证用普通stack/cat和跨节点多项式作独立对照，
检查原输入与每个字段根、连接零/None、共享参数、版本及无数值重拼。SDK现有 `weighted-kv-tensor` 含对应实际消费者检查。

| 固定资格 | 结果与范围 |
| --- | --- |
| Python-owned / standalone core | 两套CANN9.2构建通过；新二进制，新wheel，320安装文件逐字节匹配固定源码 |
| CPU FP32/FP64 定向 | **245 passed / 7 原 xfail / 2 原严格 failed**，0 errors；原作业failed/exit1保持 |
| SDK CPU | 7项通过：KV tensor/prepare、Read transport、Aggregate、region transport、columns、checkpoint |
| SDK NPU | 五种记忆×三种放置，共15组合通过；独立streaming、raw状态、VJP和两窗SGD |
| CPU/NPU安装消费者 | 各Python/native×positions、mixed-a、LH176，共12个父调用全部通过；三步AdamW、whole3/prefix2/新进程resumed1 |

两项严格失败仍为 `test_general_ports_messages_states_and_gradients` 的Python FP32 regional_total / regional_threshold，
根 `nodes.0.kernel.base[7,1]`；差8.851289749145508e-6，原atol1e-6/rtol1e-5，未更改阈值或标成通过。
同材料FP64诊断及None/零/完整观察见[上一固定记录](source-bank-20261112.md)；这次复现同坐标和同差值，不重复完整数值工程。
小型positions/mixed-a每步独立核对483数值叶/162离散叶；LH176安装消费者核对输出、raw continuation/pending/history/ledger、
None/零梯度、参数和全部AdamW槽的精确恢复，不是该规模的独立数学资格。

构建用CPU0–159/160–319、各512GiB/2400秒；检查CPU0–7/8–15/16–19/24–27、各256GiB、900–1800秒。
SDK NPU物理11，安装NPU物理1，均映射逻辑npu:0；Python与独立LibTorch工具链分离。
源码/构建/NPU运行和CUDA资格不混称；本轮未做CUDA、多卡或resident检查。

## 有限成本

配置/材料与[72e1d22c](source-bank-20261112.md#成本设置与计时边界)一致：LH176为22区域、14,902,694参数，
B4/L16/d64、KV混合sum/qry与区域att/ind、regional_total/quad、原始staggered输入 **61位置**。
FC16为4×4、B8/L128/d128、KV qry、regional_total/quad、总5,092,788参数和 **1024位置**。
LH为输出平方loss/SGD；FC为固定下游任务/head/AdamW/clip。

CPU Torch1线程、affinity0–7；NPU主机2线程、affinity32–35、物理11。
数值计算在NPU，host日程、events在CPU，即原mixed-c有效放置，**不是resident**。
每配置一个独立进程、一次reset warmup和一次测量；20点均exit0。主构建已结束；
安装资格可在不同CPU/物理NPU1并行，外部争用未完全追踪，不宣称独占或峰值。
完整训练包含必要输入准备/搬运、packing、前向/loss、反向、有限性检查、optimizer、清理与同步；FC另含clip。
构造/reset单列。此次没有profile/census；host phase不是可相加的设备时间。

每格 **秒 / 输入位置每秒**：

| 设备 | LH入口 | 无grad推理 | 含grad前向 | 完整训练 |
| --- | --- | --- | --- | --- |
| CPU | 直接PyTorch | 0.204585 / 298.2 | 0.238555 / 255.7 | 0.749660 / 81.4 |
| CPU | 公共native | 0.508174 / 120.0 | 0.553846 / 110.1 | 1.255592 / 48.6 |
| CPU | 默认native | 0.602624 / 101.2 | 0.681386 / 89.5 | 1.419332 / 43.0 |
| NPU | 直接PyTorch | 0.177075 / 344.5 | 0.198387 / 307.5 | 0.621882 / 98.1 |
| NPU | 公共native | 1.200854 / 50.8 | 1.279951 / 47.7 | 2.276100 / 26.8 |
| NPU | 默认native | 1.336712 / 45.6 | 1.439747 / 42.4 | 2.471804 / 24.7 |

| FC16 默认native完整训练 | 秒 | 输入位置每秒 |
| --- | ---: | ---: |
| CPU | 1.156975 | 885.1 |
| NPU | 0.301235 | 3399.3 |

全部20点的初始参数SHA、实际工作与旧记录匹配；18个LH点还匹配实际输入字节SHA。
FC未记录运行时输入字节SHA，不能以固定配方补称字节核对。LH每点8368候选/2572 selected/15340边访问，
公共/默认94区域列调用/176状态列调用；直接公式Full算11264行后mask，公共只算2572行，公开根/观察职责也不同。
FC CPU/NPU selected分别8344/8345，设备间路由不声称严格一致。三模式职责和吞吐分母不得混用为生成token速度。

相对72e1d22c，LH默认训练CPU **1.405757→1.419332秒**、NPU **2.902243→2.471804秒**；
公共NPU **2.708257→2.276100秒**。FC NPU **.291365→.301235秒**。NPU物理卡由1变11，
一次测量不能将全部变化归因于改动，CPU/FC未显示改善。NPU默认LH仍约为本轮直接公式的 **3.97倍**。
采用这个已验证的相邻阶段存储边界；**主要架构差距未闭合，不追加同一局部优化循环**。

仍存在prepare/raw-bank到Read/Full的重装配、区域候选分组及default continuation的物化成本。
这轮没有隔离它们的全部耗时，不能从未测的剩余项推导必然加速或理论上限。直接公式也有较窄职责；
后继需要选择贯穿这些阶段的typed owner/伴随边界，而不是因历史mixed预设或旧接口保留拆分。
本轮后按用户要求停止并重新对齐；P1—P6不因此标成全部完成。

## 重建

从固定提交按[开发入口](../development.md)构建两套匹配库和独立wheel；运行机器记录中的pytest/CTest/SDK argv。
安装消费者从source外执行 `examples/consumer_core5/tensor_window.py --positions`、`--placement mixed-a` 及
`planned_windows.py --plan PLAN`，各选择Python/native和CPU/NPU。LH plan完整存于机器记录，也可用上一报告的固定recipe重建。
成本用跟踪的 `benchmark_lh_tensor.py` 和 `benchmark_dense_query.py`；每点命令、runtime、工作量、原文件hash与原资格失败均有记录。
替换历史绝对路径/输出目录即可，不以旧artifacts存在为前提。FC下游仍以固定commit只读archive，不采mutable HEAD。
