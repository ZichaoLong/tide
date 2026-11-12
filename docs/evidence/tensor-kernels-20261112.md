# KV整组数值边界：固定资格、44点成本与组合失败

实现 `6011d5992802cfa3f304e3dca7b6742e8136915d`；数学 `0770593`。完整配置、二进制身份、命令、检查结果和数据在
[机器记录](tensor-kernels-20261112.json)。本报告不包括之后的C++写入、Read传输、聚合或resident控制图修正。

## 采用的改动与资格

KV Read/Describe/Full在G×B×Q/K上计算，真实共享参数保留单矩阵；短/长Read共用QK。
Python正写入同样整组计算。元数据int64、实际选择Full、未乘c的G、各原始根与版本守卫保持。
resident改为实际bank矩阵、设备分组GEMM及块dW；其静态控制图在组合处仍失败。

- 两套独立core/resident构建成功；SDK三个定向CTests通过。
- 固定CPU资格272通过、9原严格数值xfail、4指标断言失败。4项仍查询已替换的旧bucket；
  后继测试改为真实tensor bucket。原失败保留，不能把整个固定suite称为通过。
- CPU FP64/NPU FP32 × Python/native，共8个安装消费者通过三步AdamW及新进程精确恢复。
  四个小型位置消费者还通过独立标量数值检查；LH176消费者的资格是实际训练/完整raw恢复，不是数学oracle。
- resident投影、head、KV更新、Aggregate、区域、残差、query、Full及read-selection共9项真机通过；
  第10项block-kv-messages在录入时失败。厂商日志明确EE1019、`stream task buffer full`，
  原因是bank×logR备选体静态展开过多，不是已运行GEMM的数值错误或HBM不足。
  其依赖的后续整链/恢复/规模资格未执行；新Python resident库仅构建成功。

## 同材料有限成本

CPU1线程，NPU主机2线程；NPU物理5，独立队列锁。其他实验占用其他卡，主机非独占。
每点新进程一次reset warmup、一次测量；profile另开进程。共享条件和单点结果不构成稳定峰值结论。
默认native是Python调用C++/LibTorch，主机日程，CPU或mixed-c；不是常驻或standalone计时。

FC16：16节点/4区域，B8/L128/d128/H2/FFN512，1024实际输入位置，固定TL-REQ-004任务。
推理只做最后位置head并返回首token；含grad前向/训练包含整序列head/loss。
训练含必要准备、packing、反向、检查、clip、AdamW、清理及同步。

| 路径 | CPU推理 / 含grad前向 / 训练（秒） | NPU推理 / 含grad前向 / 训练（秒） |
| --- | --- | --- |
| 直接FC | 0.410539 / 0.495872 / 1.400383 | 0.279876 / 0.326503 / 0.771525 |
| 默认native | 0.433849 / 0.451158 / 1.172892 | 0.167590 / 0.175172 / 0.363573 |
| 默认Python | 0.524984 / 0.574380 / 1.222852 | 0.289157 / 0.311538 / 0.559371 |

FC公共native训练CPU/NPU为1.136538/0.348373秒。默认native训练吞吐CPU/NPU为
873.1/2816.5位置每秒；吞吐=1024/完整秒数。NPU优于CPU，且本测量默认native快于直接FC。
这不把直接实现当硬件极限：它仍按B循环，算法和接口都可继续改进。
Transformer16沿用[原三层报告](fc-three-layer-comparison-20261111.md)原身份：
训练CPU/NPU 0.7660/0.04267秒。其数学、控制、记忆和共享拓扑不等价，只提供计算效率参照；本轮未重测。

LH176：教材LH启发拓扑，176节点/22区域/1058体边，d64、B4/L16、staggered-v1，61实际输入位置，
14,902,694可训练参数。实际8368候选、2572 Full、15340边访问；直接实现物理执行11264个Full行后mask。
所有模式包含输入准备、完整前向/输出loss及finite检查，训练另含SGD(momentum .3)、反向/检查/同步。

| 路径 | CPU推理 / 含grad前向 / 训练（秒） | NPU推理 / 含grad前向 / 训练（秒） |
| --- | --- | --- |
| 直接公式 | 0.202943 / 0.272252 / 0.754625 | 0.201852 / 0.210401 / 0.632201 |
| 公共native | 0.572435 / 0.627774 / 1.437753 | 1.530797 / 1.587047 / 3.542038 |
| 默认native | 0.652923 / 0.745274 / 1.595441 | 2.057216 / 1.741265 / 3.595338 |
| 默认Python | 1.494731 / 1.583731 / 2.766918 | 2.643018 / 3.164061 / 5.444566 |

LH默认native训练CPU从历史2.3187秒到本次1.5954秒；NPU仍明显慢于直接公式。
直接与默认具有不同Full物理工作、root/解码责任；两者差额不能全部命名为调度开销。

## 加法与静态/动态诊断

直接公式保留region×B×L主张量。新KV局部数值图已经实现同样的组内算术，
但源消息仍为每个接收节点装配均值，阶段间字段发布后再组装区域输入，raw终态/公共返回另有责任。
固定33文件数据流审阅包含循环位置、实际形状与复制，附于机器记录。

LH CPU诊断含私图：直接Python共11192图节点；默认native的Python外图14858节点，
加native私图后的native统计23333节点。两套观测重叠，不可相加。
默认native仍有2066个IndexCopyBackward节点；新Read/Full实际各22个数值组。
算子/计算图计数解释工作来源，不单独证明耗时占比；完整profile汇总保留在机器记录。

后继已选择两个可验证边界：接收节点×行×物理来源的一次均值/伴随，及resident设备bank游标复用
同一容量分支体。前者消除逐源index-copy图，后者减少静态任务录入且保留最小桶；资格必须用新提交。
后续未完成工作仅在[ROADMAP](../ROADMAP.md)，本报告只保留6011d599事实。

## 从干净源码复现

检出本实现，按[development](../development.md)分别构建Python-owned和SDK，显式选择新build目录。
运行`tests/test_tensor_kernel.py tests/test_kv_tensor_batches.py tests/test_settle_columns.py`及相关字段/root检查；
SDK运行`ctest -R 'weighted-(columns|kv-tensor|column-checkpoint)' --output-on-failure`。
CPU完整定向列表及原失败见机器记录；不能删去失败后重称原资格通过。

成本入口`benchmark_lh_tensor.py --case many-lh --profile textbook --batch 4 --length 16
--level direct|public|default --client python|native --device cpu|npu:0 --mode inference|grad-forward|training`，
原生路径加匹配`--native-library`。FC用`benchmark_dense_query.py --baseline fc-tensor|window|columns
--phase-fences step --batch 8 --length 128`和固定下游Git对象。诊断用`profile_lh_tensor.py --census`。
安装消费者用`examples/consumer_core5/tensor_window.py --positions`及`planned_windows.py`。
按[资源/运行契约](../execution-flows.md)分配设备、固定源码并启动有限后台服务；完整参数由记录恢复。
