# Reception：FP32 连续训练、容量与 A–D 三模式对照

本页采用真实 continuing private state 的完整 AdamW 更新，训练以 FP32 为主。
吞吐分子均为实际输入位置；Timed 空 tick 和 drain 输出不充作输入。
[精简 JSON](reception-training-20261118.json)保存源码、命令、SHA256、资源、
原阈值恢复结果、容量及三模式主表。局部优化和双方成本见[性能归因](reception-performance-20261118.md)，
跨实现数值资格见[Reception 数值报告](reception-numerics-20261118.md)。

## 一百次连续更新

稳定吞吐取第3–100次更新的输入位置总数除以完整时间总和。计入准备、packing、
forward/readout/loss、必要 Timed flush、backward、AdamW、detach 和同步；
构造、输入生成/传输、热身、finite 审计及 checkpoint 保存/比较另计。
每个窗口保留真实历史，更新间 detach，不把一百次零状态启动称作连续训练。
Timed 训练的每个有限 loss 窗口包含 flush；它与不在中途 flush 的连续推理是不同负载。

| 配置 | B/L | 输入位置/s | 峰 allocated/reserved GiB | 配对恢复更新 | 源码 |
|---|---|---:|---:|---|---|
| FC-K 9.33M | 128/256 | 30333.16 | 37.361/39.576 | within_threshold | `8ce51410` |
| HB source 494.11M Settle | 4/128 | 267.40 | 33.840/35.617 | within_threshold | `6570d061` |
| HB light64 17.88M Settle | 24/256 | 1618.21 | 38.537/44.904 | within_threshold | `6570d061` |
| HB source 15.69M Settle | 24/256 | 2643.76 | 39.277/43.508 | within_threshold | `8ce51410` |
| HB source 15.69M Timed | 12/128 | 369.74 | 42.193/47.248 | within_threshold | `f3e6661c` |

全部表内更新的输出、参数、梯度、optimizer、私有状态有限，窗口之间状态已 detach；
各自 final_continuation 保存 optimizer 实际 step、KV retained slots 和 slot limits。
保存数据和下一前向的 exact 检查，与原阈值内的下一实际更新分别记录；
within_threshold 不等于逐位相同，更不消除跨候选有限超差。
参数和 AdamW 槽保持 FP32。64 GiB NPU 的 sampled allocated 预算为48 GiB；
reserved 另列，运行时仍有 Torch 外的占用。此预算不是硬隔离，也不包含下游任务头。
各作业通过协作队列独占准入，仍共享主机且不控制外部任务；精简 JSON 保留物理卡、
请求 affinity 及原记录提供的实际 affinity；缺失的观测保持 null，不以请求值冒充实测。

## 新进程真实恢复

下表从上述第100次更新保存点，在新解释器加载同一 checkpoint 并执行第101次更新。
加载和首个恢复更新均包含阶段同步，排除运行时初始化和 checkpoint SHA pass；
没有新建原 owner 参照，所以这些时间不是额外的数值等价通过。
文件系统缓存未控制；新进程不等于冷磁盘。

| 配置 | 加载秒 | 第101次更新秒 | 更新峰 GiB | 状态 |
|---|---:|---:|---:|---|
| FC-K 9.33M | 0.833 | 2.600 | 36.962 | completed_finite |
| HB source 494.11M Settle | 7.287 | 3.526 | 33.999 | completed_finite |
| HB light64 17.88M Settle | 1.197 | 6.112 | 38.696 | completed_finite |
| HB source 15.69M Settle | 1.134 | 4.119 | 39.322 | completed_finite |
| HB source 15.69M Timed | 1.469 | 6.178 | 42.069 | completed_finite |

各记录保留 checkpoint 生产源码、文件 SHA256、恢复者源码、状态位置、
更新前后 optimizer step 和 increment_errors。冷恢复较稳定更新慢时，不能把其吞吐替代稳定训练。

## 有限 batch 搜索与内存余量

这些是每步从零私有状态开始的五样本搜索，不能与上面的持续吞吐直接混成同一个测点。
增加 batch 以真实 sample 轴为单位；没有用节点数或 sequence 长度替代 batch。

| 配置 / 接收重算 | B/L | 输入位置/s | allocated/reserved GiB |
|---|---|---:|---:|
| FC / False | 32/256 | 27707.96 | 7.911/9.031 |
| FC / False | 64/256 | 35268.56 | 15.640/17.916 |
| FC / False | 128/256 | 36158.17 | 31.348/35.516 |
| HB light64 / True | 12/256 | 1433.72 | 18.217/20.406 |
| HB light64 / True | 24/256 | 1651.11 | 35.971/39.865 |
| HB light64 / False | 8/256 | 1366.71 | 22.439/23.262 |
| HB light64 / False | 16/256 | 1721.82 | 44.146/46.004 |
| HB 0.5B / True | 1/128 | 96.41 | 14.332/15.801 |
| HB 0.5B / True | 2/128 | 186.32 | 19.451/20.941 |
| HB 0.5B / True | 4/128 | 273.64 | 29.532/32.174 |

Light64 不重算 B16 的短测吞吐较高，但 allocated 已44.1GiB；
其余量小于 B24 接收重算。选主力默认时优先采用通过连续100更新的后者；
这不是宣称任何任务下都应重算。两种候选、整区域重算和原 block 路径全部保留。
0.5B B1→2→4 的短测持续增长，在本轮预定 B4 边界停止；不声称已经找到全局最优 batch。

## 三种接收的四类联合匹配

共同 FP32 B8/L128，compact-9 n8/h4、d128、X、region_shared，Settle；
batch4、Reception 重算、prefix+doubling、fused QKV+结构 padding mask。
每种接收及每种模式独立审计实际初态活动后冻结 A–D；Transformer query chunk256，
原生/auto Attention。A/B 匹配188个 block，C/D匹配9层；参数量分别匹配总量和实际有效触及均值。
相同输入特征前缀、固定种子；每模式 warmup2、计时5。训练的后续活动另记，
计时过程中不按新活动重建对照。Transformer 是不同数学函数，不据此比较模型质量。

| 接收 | 总参数 M | 实际触及 M（训练初态） | Full/候选槽位 |
|---|---:|---:|---:|
| incoming_light_128 | 19.281 | 18.034 | 135903/192512 |
| incoming_light_64 | 17.880 | 16.638 | 136171/192512 |
| incoming_wide | 20.520 | 19.256 | 135140/192512 |

下表单位全部是输入位置/s；无梯度推理、含梯度前向、完整训练分别计时。
含梯度前向不包含 backward/optimizer；本表均从零私有状态开始，不能替代缓存 decode。

| 接收 | 模式 | Tide | A | B | C | D |
|---|---|---:|---:|---:|---:|---:|
| incoming_light_128 | inference | 2213.55 | 4701.24 | 4656.49 | 85770.15 | 85831.30 |
| incoming_light_128 | grad_forward | 2039.10 | 3808.57 | 3789.81 | 75257.45 | 75101.94 |
| incoming_light_128 | train | 873.94 | 1634.03 | 1636.33 | 32592.38 | 33229.28 |
| incoming_light_64 | inference | 2367.20 | 4643.99 | 4614.19 | 85876.03 | 93958.17 |
| incoming_light_64 | grad_forward | 2108.35 | 3784.96 | 3761.54 | 75370.80 | 75731.41 |
| incoming_light_64 | train | 954.89 | 1677.96 | 1676.02 | 34109.42 | 34052.00 |
| incoming_wide | inference | 2203.24 | 4756.08 | 4762.01 | 96605.13 | 96422.84 |
| incoming_wide | grad_forward | 2020.69 | 3827.56 | 3814.16 | 76962.01 | 77331.02 |
| incoming_wide | train | 868.85 | 1652.49 | 1677.62 | 33903.95 | 34085.55 |

所有实际 constructor、block/参数目标与相对偏差、allocated/reserved、五个原始样本和
range/median 均在配套 JSON。A/B 的差距没有因 Tide 合批而消失；
相近参数和 block 数仍不约束乘加、动态索引、SSD、接收 Attention、padding 或串行深度。
C/D 宽浅矩阵的吞吐更高；这也是其对照意义，不能视为同一数学函数的替代实现。
具体成本证据见[双边归因](reception-performance-20261118.md)。

## 0.5B 主力 B4 的四类完整训练对照

FP32 B4/L128、HB profile9/n32+h4、X、independent、source_residual；
region batch4、Reception 重算、prefix+doubling，与上面所选 0.5B 训练配置一致。
本表每次从零私有状态开始，warmup2、完整训练5样本；不同于100步真实历史吞吐。
Transformer chunk128，原生/auto Attention；每个对象在独立子进程计时。
总参数 494.111M，初态实际有效触及均值 145.360M，
Full/候选槽位 168675/876544。四类同用本负载初态审计，计时中不重匹配。

| 对象 | block数 / 宽度 / FFN | 实际参数 M | 目标参数偏差 | 输入位置/s | allocated/reserved GiB | range/median |
|---|---|---:|---:|---:|---:|---:|
| tide | 1712 节点 / d128 | 494.111 | — | 293.76 | 29.532/31.914 | 1.64% |
| transformer_A | 1712 / 128 / 584 | 496.562 | +0.4962% | 89.42 | 18.274/18.715 | 4.08% |
| transformer_B | 1712 / 80 / 248 | 145.999 | +0.4399% | 90.94 | 8.182/8.424 | 3.26% |
| transformer_C | 9 / 2032 / 6296 | 494.107 | -0.0007% | 4711.94 | 9.258/9.883 | 0.22% |
| transformer_D | 9 / 1072 / 3592 | 145.358 | -0.0015% | 10275.27 | 2.788/2.867 | 10.76% |

A/B 均1712层，C/D均9层；总量与实际触及的目标分别匹配。
这张表比较同 B/L 与精度的不同数学主干，不是同函数精度验证，也不认证 Transformer 全局最优。
原始样本、初态活动、后续训练工作、参考记录和有限搜索停止原因保存在精简 JSON；
正常计时与 0.5B B1 的侵入式成本诊断分开解释。

## 复现

使用每行记录的冻结源码及 `launch.json` 命令，而不是把新 HEAD 追认为旧测量。
命令入口分别为 `benchmarks.search`、`benchmarks.continuous`、`benchmarks.restore` 和
`benchmarks.run`，输出需要新的目录；按[执行契约](../execution-flows.md)在 detached
background.slice 内完成资源准入。真实记录位于 `artifacts/agg-{train,restore,modes,...}-*`，
精简 JSON 给出逐文件路径和 SHA256。加载配置入口见[训练建议](../execution-recommendations.md)。
