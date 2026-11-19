# 训练、前向与真实缓存的四类联合比较 · 2026-11-18

五份零状态作业完成35个观察、175个正常计时样本；末尾同步修复后的两份缓存programme完成10个子项、30个实际历史续接段。三组完整训练中，Tide的时间和allocated峰值均高于A/B；具体倍数和模式差别见下表。比较同时约束block/深度与总/活跃参数，模型数学不同，不比较模型质量。
全部正常样本、三段逐调用结果、实际dtype/operator/device、匹配、终态、原始文件SHA和命令见 [JSON](training-modes-20261118.json.gz)。本页不新增数值等价或checkpoint恢复资格；finite与这些资格分别记录。

数学固定 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`（此前108文件核验通过）。完整训练测量源码/设施为 `2c011b8a4f703b5aa50ab4107bb50085e3786c56`，零状态其它模式与旧缓存为 `22527cf472fc4d6ea9ca7ce22528b7e4ca9cc96a`；正式缓存测量为 `f8235fe48ff794f0e5ac60bf7565d7f918479e6c`。整理基于成本证据提交 `0e0c9989`，不替代各测量身份。

## 模式、精度与计时边界

零状态各组为两次预热、五次采样，每次forward重新建立私有状态。inference关闭梯度；grad_forward建立完整图但不反向/更新；train包括zero_grad、准备/packing、forward、必要Timed flush、全部返回值readout/loss、backward、AdamW foreach与同步/标量。训练预热也更新参数，随后五个样本继续更新参数；其私有状态仍逐样本清零。输入传输、模型构造、语义/路径审计、预热和checkpoint不计时。这些零状态窗口不是逐位置decode，未测首输出时延。

本页全部零状态Transformer实际使用chunk256、window128；缓存programme使用chunk128、window128。FC是n8×4层、32节点、d128、K内存、independent、region_batch1。HB是compact-9、n8+hubs4、188节点/9层、d128、X内存、region_shared，使用region_batch4、Reception局部重算、SSD摘要prefix，区域整体重算关闭。HB组合来自训练工作，本页不称其推理最优；Transformer的有限候选依据见 [成本报告](training-costs-20261118.md)，也不构成全局最优基线。

FP32、BF16各自同精度比较。BF16矩阵输出确实在NPU观测到BF16，FP32组矩阵输出为FP32；FC的addmm/bmm/matmul及 `npu.npu_fusion_attention_v2`、HB及基线的matmul/native Attention等实际调度记录在JSON。这些是排除于正常计时外的模式路径审计，不是性能份额或缓存子项的逐算子trace。BF16的主参数、参数梯度和AdamW矩槽仍FP32，K/V为BF16，摘要/SSD、归一化与分配为FP32；FP32路径保持FP32，HF32 matmul/conv关闭。环境为torch 2.12.0+cpu、torch_npu 2.12.0.post3、CANN 9.2.0，实际算子位置是npu:0，物理卡另列。

零状态表峰值是预热后重置、正常样本后读取、后审计前冻结的累计allocated峰值；包含持久参数/优化器及输入等存活张量。缓存表取三个逐段重置allocated峰值的最大值，包含实际状态；该峰值读取在计时外finite检查之后。reserved缓存、状态payload、RSS不是同一指标，原记录分别保留。

## 四类口径与实际匹配

| 类别 | 同时匹配的结构约束 | 同时匹配的参数约束 |
| --- | --- | --- |
| A | 节点/局部block总数 | 总参数 |
| B | 节点/局部block总数 | 实际有效触及参数 |
| C | 区域层数/串行深度 | 总参数 |
| D | 区域层数/串行深度 | 实际有效触及参数 |

总参数统计实际唯一Parameter，Tide包含入口投影；Transformer包含主干blocks与final RMSNorm，均不含任务头、词表embedding或为凑数而添加的无效权重。活跃量包含入口、接收、memory prepare、regional和Full的语义有效owner/group/source参数切片；共享参数在每样本每时刻去重。Settle按输入步跨层去重，Timed按逻辑tick跨区域去重且包含flush；它不是非零梯度、物理访存、FLOPs或输出因果归属。教材k/n代理仍保留，不用作B/D目标，k/n也不是硬活动数量上限。

每种模式的匹配先固定初态、输入、精度和初始参数，再独立审计；训练审计建图但不更新。固定匹配后，预热及计时中的训练活动变化不会改变基线构造器。缓存匹配另按B1、seed751的真实初始prefill/decode/flush审计，不用训练B/L或种子的活跃量替代。不同宽度输入共享逐特征种子前缀，完整hash自然不同。
JSON另列每输入位置的消息、候选、Full、实际Attention对数、查询packing的selected/executed行及NodeLinear multiply-add；大规模逐样本时刻数组保留在SHA标识的原件，参数去重不消除这些重复工作。

| 匹配组 | 总参数 | 教材代理 | 实际活跃均值 | 均值逻辑轴 | 样本时刻slot/输入位置 | 累计活跃/输入位置 |
| --- | --- | --- | --- | --- | --- | --- |
| FC零状态：train/inference/grad_forward | 9,334,304 | 1,280,324 | 4,205,138.000000 | input_step | 32,768/32,768 | 4,205,138.000 |
| Settle零状态：train | 15,694,224 | 12,103,328 | 14,463,720.623047 | input_step | 6,144/6,144 | 14,463,720.623 |
| Timed零状态：train | 15,694,224 | 12,103,328 | 10,881,349.113622 | tick | 6,240/1,536 | 44,205,480.774 |
| Timed零状态：inference/grad_forward | 15,694,224 | 12,103,328 | 10,882,076.528365 | tick | 2,080/512 | 44,208,435.896 |
| FC缓存：inference | 9,334,304 | 1,280,324 | 4,173,344.000000 | input_step | 136/136 | 4,173,344.000 |
| Timed缓存：inference | 15,694,224 | 12,103,328 | 10,942,018.215580 | tick | 552/136 | 44,411,720.993 |

FC三个零状态模式的活跃目标相同；Timed其它两个模式的目标也相同，均已逐模式核实。Timed训练的6,240 slot为12×(128×4+8)，其它前向2,080 slot为4×(128×4+8)，缓存552 slot为136×4+8。表中Timed约1,090万是每逻辑tick均值，不能写成每输入位置活跃量。

所有结构block偏差均为0；固定该block数后，在d为16倍数且d≥32、FFN为8倍数且2d≤FFN≤6d的合法集合中取参数最近点，平局优先FFN=4d。下列参数偏差是(实际−目标)/目标，四类都同时保留两个约束。

| 组/类别 | block目标→实际 | d/FFN/head维度 | 参数目标 | 实际参数 | 参数偏差 |
| --- | --- | --- | --- | --- | --- |
| FC零状态三模式 A | 32→32 | 160/392/32 | 9,334,304.000 | 9,308,320 | -0.27837% |
| FC零状态三模式 B | 32→32 | 80/440/16 | 4,205,138.000 | 4,203,600 | -0.03657% |
| FC零状态三模式 C | 4→4 | 432/1224/16 | 9,334,304.000 | 9,335,088 | +0.00840% |
| FC零状态三模式 D | 4→4 | 288/832/32 | 4,205,138.000 | 4,205,088 | -0.00119% |
| Settle零状态train A | 188→188 | 80/240/16 | 15,694,224.000 | 15,671,760 | -0.14314% |
| Settle零状态train B | 188→188 | 64/312/32 | 14,463,720.623 | 14,366,272 | -0.67375% |
| Settle零状态train C | 9→9 | 368/1088/16 | 15,694,224.000 | 15,692,624 | -0.01019% |
| Settle零状态train D | 9→9 | 320/1248/64 | 14,463,720.623 | 14,475,200 | +0.07937% |
| Timed零状态train A | 188→188 | 80/240/16 | 15,694,224.000 | 15,671,760 | -0.14314% |
| Timed零状态train B | 188→188 | 64/216/32 | 10,881,349.114 | 10,901,056 | +0.18111% |
| Timed零状态train C | 9→9 | 368/1088/16 | 15,694,224.000 | 15,692,624 | -0.01019% |
| Timed零状态train D | 9→9 | 320/832/64 | 10,881,349.114 | 10,880,960 | -0.00358% |
| Timed零状态inference/grad_forward A | 188→188 | 80/240/16 | 15,694,224.000 | 15,671,760 | -0.14314% |
| Timed零状态inference/grad_forward B | 188→188 | 64/216/32 | 10,882,076.528 | 10,901,056 | +0.17441% |
| Timed零状态inference/grad_forward C | 9→9 | 368/1088/16 | 15,694,224.000 | 15,692,624 | -0.01019% |
| Timed零状态inference/grad_forward D | 9→9 | 320/832/64 | 10,882,076.528 | 10,880,960 | -0.01026% |
| FC缓存inference A | 32→32 | 160/392/32 | 9,334,304.000 | 9,308,320 | -0.27837% |
| FC缓存inference B | 32→32 | 112/240/16 | 4,173,344.000 | 4,193,392 | +0.48038% |
| FC缓存inference C | 4→4 | 432/1224/16 | 9,334,304.000 | 9,335,088 | +0.00840% |
| FC缓存inference D | 4→4 | 240/1128/16 | 4,173,344.000 | 4,172,400 | -0.02262% |
| Timed缓存inference A | 188→188 | 80/240/16 | 15,694,224.000 | 15,671,760 | -0.14314% |
| Timed缓存inference B | 188→188 | 64/216/32 | 10,942,018.216 | 10,901,056 | -0.37436% |
| Timed缓存inference C | 9→9 | 368/1088/16 | 15,694,224.000 | 15,692,624 | -0.01019% |
| Timed缓存inference D | 9→9 | 240/1368/16 | 10,942,018.216 | 10,942,800 | +0.00714% |

## 完整AdamW训练：五次更新的正常计时

### FC BF16 B128/L256

| 对象 | 中位s | 实际输入位置/s | allocated峰值GiB | 样本range/median |
| --- | --- | --- | --- | --- |
| Tide | 0.780468 | 41,985.04 | 24.1258 | 0.93% |
| A | 0.391510 | 83,696.50 | 8.6582 | 0.12% |
| B | 0.319427 | 102,583.58 | 6.5783 | 0.14% |
| C | 0.187776 | 174,506.10 | 3.5436 | 0.37% |
| D | 0.079299 | 413,222.81 | 2.2723 | 0.64% |

### HB Settle FP32 B24/L256

| 对象 | 中位s | 实际输入位置/s | allocated峰值GiB | 样本range/median |
| --- | --- | --- | --- | --- |
| Tide | 2.857489 | 2,150.14 | 35.9224 | 1.36% |
| A | 0.598656 | 10,262.99 | 8.8267 | 2.37% |
| B | 0.591192 | 10,392.56 | 9.2309 | 1.39% |
| C | 0.081500 | 75,386.68 | 2.1107 | 0.37% |
| D | 0.055614 | 110,476.67 | 2.0627 | 1.19% |

### HB Timed FP32 B12/L128

| 对象 | 中位s | 实际输入位置/s | allocated峰值GiB | 样本range/median |
| --- | --- | --- | --- | --- |
| Tide | 4.238573 | 362.39 | 35.2947 | 0.56% |
| A | 1.214859 | 1,264.34 | 2.3427 | 2.63% |
| B | 1.210969 | 1,268.41 | 1.9295 | 1.27% |
| C | 0.058430 | 26,287.68 | 0.6843 | 4.12% |
| D | 0.058318 | 26,338.40 | 0.5322 | 5.50% |

| 组 | Tide/A时间比 | Tide/B时间比 | 七次更新后另审计的活跃均值 |
| --- | --- | --- | --- |
| FC BF16 B128/L256 | 1.993 | 2.443 | 4,416,272.000 |
| HB Settle FP32 B24/L256 | 4.773 | 4.833 | 14,350,460.248 |
| HB Timed FP32 B12/L128 | 3.489 | 3.500 | 10,858,519.061 |

Tide在这些完整训练点尚未追平A/B，C/D更快。参数相近并不等于数学工作相同：Tide还有逐边接收、区域分配、记忆摘要/矩阵递推、padding与细粒度调度；C/D将参数集中在少量宽block。[成本报告](training-costs-20261118.md)已用B8/L256 BF16 Settle的保存张量、工作量、分阶段诊断及真实硬件轨迹解释主要方向，并测量了区域合批、Reception重算、SSD摘要prefix和Transformer chunk候选。该成本点与这里精度/批量不同，不把它的耗时份额或收益直接套到本页三组，也不把剩余差距全部归于一个kernel。

## 零状态推理与含梯度前向

### FC BF16 B128/L256

| 对象 | inference s | 输入位置/s | 峰值GiB | grad_forward s | 输入位置/s | 峰值GiB |
| --- | --- | --- | --- | --- | --- | --- |
| Tide | 0.375089 | 87,360.71 | 5.0656 | 0.377761 | 86,742.64 | 24.1678 |
| A | 0.160150 | 204,608.30 | 0.3019 | 0.164115 | 199,664.90 | 8.5371 |
| B | 0.166182 | 197,181.54 | 0.2258 | 0.169311 | 193,537.45 | 6.5033 |
| C | 0.097711 | 335,357.08 | 0.8078 | 0.097959 | 334,508.92 | 3.3685 |
| D | 0.033880 | 967,166.39 | 0.5334 | 0.034251 | 956,692.40 | 2.1502 |

### HB Timed FP32 B4/L128

| 对象 | inference s | 输入位置/s | 峰值GiB | grad_forward s | 输入位置/s | 峰值GiB |
| --- | --- | --- | --- | --- | --- | --- |
| Tide | 1.166351 | 438.98 | 1.4112 | 1.325101 | 386.39 | 12.1825 |
| A | 0.217475 | 2,354.30 | 0.0821 | 0.262114 | 1,953.35 | 0.7809 |
| B | 0.216553 | 2,364.32 | 0.0603 | 0.259354 | 1,974.14 | 0.6450 |
| C | 0.010944 | 46,783.22 | 0.0945 | 0.013680 | 37,425.66 | 0.2304 |
| D | 0.010968 | 46,679.61 | 0.0696 | 0.013115 | 39,040.14 | 0.1767 |

记录图可大幅增加保存内存，前向耗时接近不代表完整训练成本接近。Timed D inference的五样本range/median为11.64%，Tide grad_forward为9.51%，所有样本都保留；有限样本不支持小差异的最优性判断。

## 真实缓存prefill/decode：同步修复后的主表

每子项为B1、prefill128一次chunk128调用、随后8次单位置decode；一次独立零状态推理预热后，三个测量段连续沿用私有状态，输入使用同一合成张量作为下一段。CLI `--updates 3` 在这里是三段推理，optimizer参数owner为0，没有任何更新。第一段从零开始；后两段prefill也携带历史，因此跨段中位数不是三次独立的零状态重复。

Transformer只计算新query，使用原位置RoPE与逐层真实K/V历史，不重算前缀；window128在下一次调用前保留127个历史位置。冻结源码对位置/缓存长度作校验，记录的payload和retained storage也等于2×层数×127×宽度×dtype字节。通用continuation的kv_retained字段只识别Tide KVState，Transformer的null不能解读成没有缓存。FC Tide记录K容量128；Timed Tide记录K容量512，这是容量而非有效写入数。

稳定decode每段舍弃第一个decode，时延取余下7次的中位数、吞吐取7/sum(7次时间)，再分别取三段指标的中位数；吞吐不是1/中位时延。首输出列仅取初始段，在同步chunk边界确认主干输出可用，支持性检查在计时后；它没有任务头或生成token，不能称LM TTFT。Timed每段末尾另执行8个flush tick，输出552 slot，吞吐仍按136个实际输入位置计算。

### FC BF16，B1、128+8位置

| 对象 | prefill ms | prefill位置/s | 初段首主干输出ms | 稳定decode ms | 稳定decode位置/s | 最大峰值GiB |
| --- | --- | --- | --- | --- | --- | --- |
| Tide | 71.405 | 1,792.60 | 70.314 | 51.382 | 19.564 | 0.0797 |
| A | 43.883 | 2,916.82 | 43.011 | 42.840 | 23.280 | 0.0713 |
| B | 43.470 | 2,944.54 | 43.053 | 42.397 | 23.585 | 0.0506 |
| C | 5.817 | 22,002.60 | 5.692 | 5.608 | 178.254 | 0.0698 |
| D | 5.947 | 21,525.02 | 5.769 | 5.685 | 174.546 | 0.0486 |

### HB Timed FP32，B1、128+8位置

| 对象 | prefill ms | prefill位置/s | 初段首主干输出ms | 稳定decode ms | 稳定decode位置/s | 最大峰值GiB |
| --- | --- | --- | --- | --- | --- | --- |
| Tide | 698.129 | 183.35 | 657.767 | 298.145 | 3.340 | 0.5446 |
| A | 444.389 | 288.04 | 441.471 | 393.100 | 2.543 | 0.1189 |
| B | 466.621 | 274.31 | 464.989 | 413.362 | 2.410 | 0.0954 |
| C | 22.164 | 5,775.00 | 21.977 | 19.851 | 50.400 | 0.0974 |
| D | 22.664 | 5,647.81 | 22.669 | 20.098 | 49.769 | 0.0771 |

完整返回时间包含输出拼接的最终同步。下表分别列输入窗口、flush、两者同步后的边界与最终complete；各列是各指标的三段中位数，不能把不同段的中位数相加。完整吞吐取每段136/complete后再取中位数。

| 组/对象 | 输入窗口s | flush s | 输入+flush s | complete s | 完整输入位置/s |
| --- | --- | --- | --- | --- | --- |
| FC Tide | 0.482920 | 0.000000 | 0.482921 | 0.482988 | 281.581 |
| FC A | 0.388075 | 0.000000 | 0.388076 | 0.388140 | 350.389 |
| FC B | 0.382161 | 0.000000 | 0.382162 | 0.382228 | 355.808 |
| FC C | 0.050774 | 0.000000 | 0.050775 | 0.050834 | 2,675.394 |
| FC D | 0.051906 | 0.000000 | 0.051907 | 0.051964 | 2,617.209 |
| Timed Tide | 3.090223 | 0.341615 | 3.431261 | 3.431367 | 39.634 |
| Timed A | 3.593327 | 0.000000 | 3.593328 | 3.593454 | 37.847 |
| Timed B | 3.786344 | 0.000000 | 3.786345 | 3.786460 | 35.917 |
| Timed C | 0.180929 | 0.000000 | 0.180930 | 0.181034 | 751.241 |
| Timed D | 0.183796 | 0.000000 | 0.183797 | 0.183895 | 739.553 |

这个Timed B1点的Tide稳定decode中位为298.145ms，A/B分别为393.100/413.362ms；模式和粒度会改变排序，不能把训练表延伸为所有场景的统一结论。Tide在该点的prefill更慢、峰值更高，C/D仍更快。这是固定配置与有限历史长度的观察，不推断长上下文、服务并发或任务质量。

全部正式缓存子项的三段输出/参数/状态finite，状态在段边界detached。FC与Transformer位置为136、272、408；Timed Tide完整flush后为138、276、414。这里只是推理状态续接，不能替代跨更新训练、checkpoint恢复或恢复后更新证据。

## 终态、资源与保留失败

| 正式作业 | 物理NPU | CPU affinity / 线程 | 外层时限/RSS预算 | UTC时段 | 终态 |
| --- | --- | --- | --- | --- | --- |
| training-joint-fc-bf16 | 4 | 120-123 / 4 | 2400s / 100GiB | 10:00:58–10:03:47 | passed |
| training-joint-settle-fp32 | 5 | 124-127 / 4 | 2400s / 100GiB | 10:00:58–10:03:58 | passed |
| training-joint-timed-fp32 | 14 | 128-131 / 4 | 2400s / 100GiB | 10:00:58–10:03:59 | passed |
| training-other-modes-fc | 7 | 132-135 / 4 | 2400s / 100GiB | 09:56:04–10:04:13 | passed |
| training-other-modes-timed | 4 | 136-139 / 4 | 2400s / 100GiB | 09:56:05–09:58:21 | passed |
| training-prefill-decode-fc-synced | 1 | 108-111 / 4 | 2400s / 80GiB | 10:09:51–10:10:46 | passed |
| training-prefill-decode-timed-synced | 7 | 112-115 / 4 | 2400s / 80GiB | 10:09:51–10:12:00 | passed |

各作业使用冻结源码、background.slice服务与协作队列独占身份 `exclusive_cooperative_queue; external jobs not controlled`。这是协作队列的声明身份；外部作业及主机负载不受项目控制，不能由队列身份推定设备物理绝对独占。JSON保留host load、时间、物理卡和队列信息，跨卡/跨时段仍为不同测量。RSS是进程树采样预算而非cgroup硬内存限制；零状态任务另外检查48GiB allocated预算。缓存子项实际device-budget参数、子项300s时限和RSS留在原记录中，不从其它任务推定。launch中的active是提交时快照，终态以status及programme子项为准。

两个旧 `training-prefill-decode-{fc,timed}-fixed` programme均completed、各5子项exit0，但旧run_stream仅同步各输入/flush阶段，最终torch.cat后没有显式同步；其 `input_plus_flush_seconds` 保留阶段边界，`complete_seconds` 只保证拼接入队及主机开销，不能作完整推理时间。该问题已由正式测量提交修复并重跑两个小programme。旧文件与SHA、全部样本作为legacy保留在JSON，正式主表使用同步修复后的结果。

更早的 `training-prefill-decode-fc` 和 `training-prefill-decode-timed` 均failed/exit2，错误为 `programme.py: error: unrecognized arguments: --seconds 1600`。这是programme命令行不支持该参数，没有生成模型结果；外层作业时限本身受支持。两份launch/status/queue/stderr身份和错误均保留，不重写成模型或数值失败。

## 复现与剩余边界

采用JSON中对应job的冻结source提交、launch.command和environment，复现实验沿用对应设备队列与有限资源边界。原始入口如下，JSON还校验每个子结果/config哈希：

- training-joint-fc-bf16（原路径 `../../artifacts/training-joint-fc-bf16/joint.json`，已退役）
- training-joint-settle-fp32（原路径 `../../artifacts/training-joint-settle-fp32/joint.json`，已退役）
- training-joint-timed-fp32（原路径 `../../artifacts/training-joint-timed-fp32/joint.json`，已退役）
- training-other-modes-fc（原路径 `../../artifacts/training-other-modes-fc/modes.json`，已退役）
- training-other-modes-timed（原路径 `../../artifacts/training-other-modes-timed/modes.json`，已退役）
- training-prefill-decode-fc-synced（原路径 `../../artifacts/training-prefill-decode-fc-synced/results/programme.json`，已退役）
- training-prefill-decode-timed-synced（原路径 `../../artifacts/training-prefill-decode-timed-synced/results/programme.json`，已退役）
- training-prefill-decode-fc-fixed（原路径 `../../artifacts/training-prefill-decode-fc-fixed/results/programme.json`，已退役）
- training-prefill-decode-timed-fixed（原路径 `../../artifacts/training-prefill-decode-timed-fixed/results/programme.json`，已退役）

整理校验重算175个正常样本的中位数/吞吐/范围、20个新旧子结果的三段稳定decode与内存指标，验证9份成功外层作业、4份programme的20个子项终态及2份CLI失败，并检查匹配偏差、初始参数/输入、活跃分母、配置及原始文件SHA。这里没有再次启动NPU或为文档新增模型测试。

本页覆盖指定有限FC/HB点，不覆盖全部B/L、精度/拓扑交叉或推理最优搜索；Settle的其它两种前向模式未在本批新增。合成机制loss不用于评价下游任务效果，finite不构成FP32/BF16误差资格；已有超差保留。后续性能工作应按 [比较契约](../benchmark-contract.md)、[精度](../precision.md)、[验证](../verification.md) 与 [执行](../execution-flows.md) 继续保持数学、更新/恢复语义和明确计时边界。
