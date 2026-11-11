# 张量窗口整链：固定资格与 FC 三层成本

生产实现 `4b9ca5483b405e915af6f7fea1e42f96e325fb3e`，诊断及实际固定运行
`01cde3bd837f82ba9f5ed424f41b039f1747e908`；数学为 core-5 `0770593`。
[机器记录](tensor-windows-20261111.json.gz)保存逐项配置/参数/安装/构建身份、命令、结果hash、
三个任务终态、六个安装消费者与真实profile；阅读不依赖在线原始运行目录。
实现对应见[张量窗口契约](../weighted-tensor-windows.md)，后续 KV 位置扩展不属于本次性能源码。

默认 eager 的 `run_positions/run_ports` 按能力采用 `settle-columns-v1`：输入、聚合、
KV/Read、区域求解与VJP、Full和消息共同保留B×L列。节点私有状态有类型，独立物理边仍各参与一次。
普通列根/输入/参数/末状态与完整独立事件根分开；后者仍可使用记录接口。
旧消费者若先把输入拆成事件，须按契约迁移才能使用本次窗口路径。

## 资格范围

固定 CPU 232 passed、13具名FP32数值xfail。覆盖：KV三快照/真实时间窗口、四循环和卷积、
一般Settle端口、同秩独立区域、重复边、缺失/零消息、共享参数、十发送/代表评分、完整字段、
一阶VJP与原始状态续接、容量/非法输入/空样本，以及一般调度回归。相同FP32材料提升FP64
另行通过；不改原阈值、None/zero、离散支持或原始严格失败。

安装后、独立cwd的CPU FP32/FP64 × Python/native四组，NPU FP32两组：三步AdamW对独立
标量参考检查通过；新进程精确恢复比较输出、输入/参数梯度、参数、optimizer槽和raw状态通过。
standalone LibTorch SDK的 `tidegraph-column-check` 在CPU FP64通过独立streaming、raw状态、
VJP和两个SGD窗口。Python-owned NPU二进制与standalone SDK分别构建，不混用wheel库。
复用开发二进制前逐项校验固定C++源码hash和二进制hash，机器记录保留原始构建来源。

这些是相关局部/小图资格；B8/L128完整FC数值投影、全部独立根、多步训练或学习效果未在本次
成本进程检查。旧严格失败保持原身份，成本完成不把它们改为通过。
原dev1编译、dev2收集及dev4 SDK缺Ninja的失败保留；SDK后续成功独立记录。

## 配置与完整成本

B8/L128 = 8个独立样本、每个128个输入位置，共1024位置。FP32、d128、summary32、H2、
FFN512、KV qry、Read+、cat/ind、regional_total/context/cat/quad、k2、温度0.25。
FC4为4层×1节点，FC16为4层×4节点；FC16核心4,564,404参数，含embedding/head为5,092,788。
窗口3072原始逻辑时间。材料来自只读下游固定`0439efd2`，预热后重置参数及optimizer；
同设备的默认/直接FC初始和计时前参数hash相同。Transformer是不同数学函数的对应4/16块，
只作成本参照。直接FC是独立PyTorch张量实现，不能把命令名中的native当成其计算客户端。

每配置独立进程一次预热、一次测量。完整训练含输入准备/上传、调度与packing、forward/loss、
backward、finite checks、clip、AdamW、清理和最终同步；构造、profile另列。
CPU ATen/interop均1，CPU池80–111；NPU ATen2/interop1、host池0–15，队列分配空闲物理2。
NPU为mixed-c（主机推进，payload/Read/选择在NPU），不是resident。两个CPU池不重叠，
未排除外部负载；一次有限测量不代表统计稳定、最优线程数或任意尺度。

| 图/设备 | 路径 | 无梯度推理 s | 含grad前向 s | 完整训练 s | 训练位置/s |
| --- | --- | ---: | ---: | ---: | ---: |
| FC4/cpu | 默认 native | 0.1130 | 0.1415 | 0.3554 | 2881.1 |
| FC4/cpu | 默认 PyTorch | 0.1336 | 0.1642 | 0.3735 | 2741.3 |
| FC4/cpu | 直接 FC | 0.1032 | 0.1372 | 0.3818 | 2681.9 |
| FC4/cpu | Transformer | 0.0425 | 0.0672 | 0.2453 | 4174.5 |
| FC4/npu:0 | 默认 native | 0.0663 | 0.0672 | 0.1718 | 5958.9 |
| FC4/npu:0 | 默认 PyTorch | 0.0922 | 0.0942 | 0.2084 | 4913.4 |
| FC4/npu:0 | 直接 FC | 0.0740 | 0.0813 | 0.1901 | 5385.3 |
| FC4/npu:0 | Transformer | 0.0030 | 0.0039 | 0.0134 | 76264.3 |
| FC16/cpu | 默认 native | 0.4489 | 0.5009 | 1.2863 | 796.1 |
| FC16/cpu | 默认 PyTorch | 0.5943 | 0.5523 | 1.2824 | 798.5 |
| FC16/cpu | 直接 FC | 0.4197 | 0.4944 | 1.3706 | 747.1 |
| FC16/cpu | Transformer | 0.1663 | 0.1953 | 0.7457 | 1373.2 |
| FC16/npu:0 | 默认 native | 0.2212 | 0.2288 | 0.6788 | 1508.5 |
| FC16/npu:0 | 默认 PyTorch | 0.3056 | 0.3134 | 0.7571 | 1352.5 |
| FC16/npu:0 | 直接 FC | 0.2640 | 0.2952 | 0.6978 | 1467.5 |
| FC16/npu:0 | Transformer | 0.0089 | 0.0112 | 0.0428 | 23952.9 |

吞吐为1024/完整秒数，是输入位置吞吐；本轮没有测新接口TTFT，不能用秒数除1024冒充。
CPU/NPU各另有一个低层columns训练点1.3036/0.7077秒，默认导出raw continuation没有重现旧的大幅差距。
本测量FC16 native/直接FC训练比为CPU0.9385、NPU0.9727；接近同一成本，不能据单次略快宣称普遍优胜。

FC16 CPU三Tide路径均16,384候选、8,344 Full、25,060内部边消息；NPU均16,384/8,345/25,064。
同设备工作量相同不证明逐事件数值相同；CPU/NPU离散轨迹有差异，不能称跨设备严格等价吞吐。
默认native/Python实际16个state列调用和4个区域列调用。NPU native峰值allocated484.24MiB，
直接FC402.94MiB、Transformer342.61MiB；reserved/RSS另见记录，不由单一峰值推断总资源上限。

## 差距解释与取舍

独立B2/L32 profile（64位置；不按比例外推）显示：

| 路径 | CPU可见图节点 | CPU反向节点执行 | NPU设备kernel记录 |
| --- | ---: | ---: | ---: |
| 默认native窗口 | 4,001 | 3,773 | 19,472 |
| 默认PyTorch窗口 | 3,481 | 3,253 | 16,352 |
| 直接FC | 5,338 | 5,278 | 16,224 |
| Transformer16 | 730 | 730 | 2,298 |

旧查询块默认CPU可见含私有图约10,163节点；本次列链没有隐藏逐事件nested-grad图来伪装减少。
不同设备/语言census覆盖重叠，不把总数相加。保存张量信息、连接边和具体节点类型在机器记录。
完整步表来自无profile进程；profile核数和host范围时间不能直接加总成完整耗时。

FC与Transformer差距仍大：Tide有全部候选的KV写入和两次Read、评分/区域分配、选择后的Full，
以及按节点参数的多次投影和检查；Transformer仅每块一次Attention与FFN。本轮NPU实际
Transformer命中16次融合Attention前后向，带权Tide和直接FC都没有Flash kernel。
不能把可求导log写入权重改成bool或detach以换融合，也不能把全部差额归给数学要求。
窗口NPU仍有大量allocation、index/copy、标量检查和optimizer提交；这些是后续具名优化边界。
本轮已回答“通用模块/接口是否必然比直接FC慢很多”：在该配置/观察范围内并非如此。
采用这条按能力分派的公共路径，保留一般图/完整事件根；不继续重复旧逐行装配优化。

## 复现与剩余范围

按[开发指南](../development.md)在固定源码构建/安装；`examples/consumer_core5/tensor_window.py`
运行三步训练/新进程恢复，native显式传匹配库。CPU两dtype；NPU合法分配后使用FP32。
`run_fc_comparison.py --suite windows`重建50点有限成本，`--suite window-profiles`重建8项诊断；
传目标设备、线程、固定下游repo、匹配native库和新输出目录。每设备单独调用。
`collect_fc_suites.py`使用显式suite/job/consumer/JUnit收集，详见[复现导航](../reproduction.md#可重用的三层对照收集)。

当前无本轮新TTFT、宽FC/176节点HB-LH、1B/9B、mixed-a/b、resident或多卡速度结论；
GPU真机继续延后。native循环列、位置项及其他迁移按ROADMAP逐项推进。
全部P1—P6和全仓审阅尚未完成，不能由本阶段FC结果推断所有限制解除。
