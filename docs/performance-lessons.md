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

## 本轮迁移与负结果

实现`2da3f2c`、诊断/设施`0503593`的固定证据在
[参数匹配与迁移报告](evidence/core5-performance-transfer-20261028.md)。数学版本没有变化。

| 经验 | 本次实施、证据与适用条件 |
| --- | --- |
| 复用实际存储须同时处理发布 | 冻结`ParameterBankPool`按TensorImpl/切片/设备/版本借用，同值独立参数不合并；canonical发布去重，所有使用的VJP仍归约。CPU别名/切片/版本/预算反例与设备完整训练/恢复分别验收，实际allocator观测另列；延续core-3的借用及真实扣费经验 |
| 阶段共同输入可以只复制一次 | ready数值副本由逐节点改为逐阶段，前向/逆向保持自己的布局与依赖；新增字节计数，不把内存布局连续或估算减少当作实际吞吐结果 |
| 低秩路径避免构造稠密临时矩阵 | eager合法共享KV计算`U(Vx)`，标量/未共享路径保留组合矩阵作独立锚点；只复用相同参数和输入，状态仍私有。原HB medium同材料CPU单次391.432→365.834ms通过完整对照；不能由此外推大D或逐项归因 |
| 少分配不保证更快 | 同原HB medium的NPU构造allocated减少187392字节、测量峰值减少374784字节，reserved峰值不变；完整严格对照通过，但单次2576.159→2795.223ms约慢8.5%。这是真实存储小收益和计时负结果，不能宣称设备加速 |
| 初始化和材料成本需要独立观察 | 移除随后废弃的旧随机权重初始化；v2不展开大型JSON。92.62亿参数真实材料可构造，导出约366s、C++构造约184s、RSS约46.93GiB；仍不是训练峰值或吞吐资格 |
| 匹配参数必须匹配口径 | 旧core-3减去embedding/head后匹配主体，通过真实FFN宽度而非无用权重；同参数数量不意味着同函数、同激活量或同训练成本。五族的key/value/FFN尺寸和误差都保留 |
| 稀疏发送不等于所有计算消失 | HB64的实际候选/Full由32/32变为20/16；ranked64即使k=区域大小也只有344/479 Full。局部候选、消息、物理行、反向和optimizer各自计数，严格失败候选不参与等价加速声明 |
| 打包接口不保证形成大矩阵批 | ranked64 profile仍有479次State、最大State batch=1，22905次mm；调度/Query/装配/反向成本显著。下一步应研究同形私有模块的真实BMM及反向程序复用，而不是仅合并buffer或按内核数猜瓶颈 |
| 验证失败也应保存测量事实 | 原ranked64 Python的result未附metrics，但独立metrics.json已持久保存；0503593让最终失败记录也保留同一metrics列表。失败时间可以解释成本，不能认证正确性或胜出 |
| 大profile本身可能昂贵 | 单个ranked64 trace约818MiB、profile RSS约16.25GiB；应优先较小代表案例，采集/解析预算与真实训练峰值分开。profile独立进程不进入吞吐对照 |
| 数值规模必须独立检验 | 旧/新同材料D64都严格失败；新FP32梯度和更新参数也有差异，显式FP64两步诊断通过。quad大共同偏移的手算反例在FP64也拒绝，不能把所有失败都归结为dtype不够或删除残差守卫 |
| 验证子进程也要隔离scratch | NPU产生kernel_meta令并行CUDA构建最终源码身份检查拒绝。0503593把设备子进程cwd设为输出目录，保留原失败并独立核验同内容构建；不放宽源码洁净检查或掩盖新文件 |

这些改动处理了可证明的复制、冻结存储、共享低秩和初始化冗余。逐节点物理行、未共享KV的
稠密组合矩阵及窗口反向程序构造仍有余地。当前证据不支持“低垂果实已经基本耗尽”；后续改进
继续以独立数值资格、实际工作量和完整调用结果决定保留与否。
