# 性能经验与适用条件

本页保存可复用的经验，当前工作安排只在[ROADMAP](ROADMAP.md)。每条经验须给出
固定源码、输入/计时条件、独立正确性证据和实际结果；源码推断、profile假设、容量估算、
实测内存与吞吐结论分别记录。后续实现产生的正、负结果均更新此页并链接小型固定证据。
历史来源只用于提出候选改进，不继承其数学资格或默认重新运行历史实验。

## 本仓库 core-3

| 经验 | 来源与实际结果 | 迁移到 core-5 的条件 |
| --- | --- | --- |
| 局部批量化不保证整图提速 | [批量State/Read](evidence/batched-selection-cpu-20261010.md)：Read局部3.16×、Attention局部2.96×；完整Add单次3.811→4.091秒，RSS增加，外部负载未隔离 | 同时测装配、反向、optimizer与同步；不能用局部倍率替代完整训练 |
| 私有冻结参数可借用，公开快照仍须隔离 | [4467493](evidence/resident-projection-borrow-20261003.md)：D512/B8双卡每卡allocator峰值减少296274432字节；完整梯度/恢复通过 | 参数版本与所有者存活、未消费窗口阻止发布；不把动态状态或外部可变参数当冻结bank |
| 复用通信arena减少存储 | [a72868f](evidence/resident-shared-packets-20261002.md)：同配置每卡减少268437504字节；144次通信迭代未减少 | 发送复用须等接收完成，收发分区、tail/None/零与稳定归约顺序保持；内存收益不自动是吞吐收益 |
| 准入扣费与实际分配不同 | [219719d](evidence/consumer-private-bank-capacity-20261003.md)：只去掉重复保留扣费，既有allocator结果完全相同 | 新清单分列逻辑参数、物理副本、借用、各生命周期峰值；不能把估算降低报道为实测降低 |
| 总HBM通过不代表内部缓冲通过 | [中型选择](evidence/native-selection-20261011.md)：Add B8反向packet拒绝；B4通过；Attention仍在cache-adjoint边界拒绝 | 分区与组件预算分别审计；物理切分保持逻辑batch、Attention和optimizer边界 |
| CPU/mixed/resident胜负依赖规模、函数、客户端 | [大规模](evidence/selection-review-20261009.md)、[终态](evidence/selection-terminal-20261010.md)、[中型](evidence/native-selection-20261011.md) | 旧近似平票失败仍保留；D64 CPU结论不外推D2048，不能将不同轨迹称等价加速 |
| 混合执行的小算子、scalar、切卡与同步成本真实存在 | [完整profile](evidence/native-selection-20261011.md)：Add mixed有大量SetDevice/同步；selected-clear阻止自由时间合批 | core-5观察全部候选保留状态，不能照搬旧clear后备；测真实批长和缺席物理行 |
| 版本限定的运行时缓存行为须显式记录 | [CANN缓存](evidence/resident-cann-cache-policy-20261004.md)：指定旧栈cache-off解除copy错误，默认失败保留 | 不修改公共环境，不把旧栈开关当所有版本的优化；构建、运行和错误证据分别固定 |

## 参考仓库

LH 固定历史 `a10fdb1883fccd63ec21e36cc9cffa294c63c9e2` 的
`Connectome/cpp/src/BatchHidden.cpp` 合并多样本KV读取并矩阵化回收梯度；
`CortexNet.cpp` 使用CSR邻接、按来源合并出边投影。历史缓存采用其自身激活时钟，
不可代替core-5原始逻辑时间。背景、计时与模型口径见[LH对照契约](lh-scale-benchmark.md)、
[原始测试](lh-original-test.md)。当前LH工作树有已有修改，不能作为该固定版本的替身。

fractal-latcarf 固定 `4883483dcd3f4e73bfceecb4cba877d87063cefd` 的
`src/tide/packed.py` 使用公式/形状分组和区域消息生命周期；
`_record_group_linear_1d` 为保持aarch64 FP32算术边界仍逐记录计算。
连续buffer、编译循环与实际批量GEMM须分别计数；独立调度和None/连接零不能在迁移时丢失。

## Core-5 已有观察

[ceae7bf有限测量](evidence/core5-weighted-performance-20261028.md)只覆盖宽4/32、3/8节点、
单卡FP32全活动发送；HB/KV medium有102604个可训练参数，SSD为116220。
参数按真实别名去重；这不是旧core-3的94.68亿/175.21亿规模。
resident存在逐节点队列填充、物理行多于实际作用、窗口反向程序构造及head/归约开销。
它们是有源码/profile依据的优化方向，尚无新优化收益结论。

后续每份小型证据应回答：改了什么、什么条件允许改、完整结果/梯度是否保持、
哪项实际工作或内存减少、完整调用是否变快、哪些失败和未覆盖范围仍保留。
至少包含输入/源码/构建身份、单独profile与非profile流程、原始测量次数及资源争用。
