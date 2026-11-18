# ROADMAP 9B/17B 实际规模与容量，2026-11-18

本报告为 2026-11-18T05:06:02.805123+00:00 的实测快照；当前状态 `running_snapshot`。已完成行来自实际结果文件，
未完成项保留原状态。[机器记录](roadmap-scale-large-20261118.json)包含固定源码、命令、
准入与 NUMA 策略、原始计时/工作量/匹配参数/存储以及失败。预算不作为执行成功证明。
这是阶段证据；CPU17B 尚未完成，后续结果另行更新。

本组全部源码与设施固定为 `0992b467584c1f603314ed04d8efe8596a14f576`，数学为 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`。
9B/17B 指 HB Multiscale-X Settle、source_residual/quad/independent、d768、B1/L8：
实际唯一参数分别为 **8,778,358,224** 和 **16,707,220,176**。静态分区、运输原始失败及
小型梯度/续接/恢复资格见[放置报告](roadmap-placement-20261118.md)；
[0.5B/1B 报告](roadmap-scale-small-20261118.md)不替代本篇的大规模实测。

## 生命周期、资源与待测项

| 任务 | 实际状态 | 物理 NPU | CPU affinity / NUMA | 进程树峰值/界限 GiB |
|---|---|---|---|---|
| scale-npu-hb-9b-v16 | 外层完成 | 9 | 176-191 / --membind=4 | 41.792 / 160.000 |
| scale-two-npu-hb-17b-v16 | 外层完成 | 7,11 | 192-207 / --interleave=4,5 | 68.192 / 192.000 |
| scale-four-npu-hb-9b-train-v16 | 外层完成 | 1,5,7,11 | 208-223 / --membind=5 | 34.678 / 160.000 |
| scale-cpu-hb-17b-inference-v17 | 运行中 | 尚未准入 / CPU | 240-279 / --membind=6 | 77.040 / 192.000 |
| scale-cpu-hb-17b-train-v17 | 等待前置任务 | 尚未准入 / CPU | 240-279 / 尚未进入 workload | 0.015 / 384.000 |

NPU 经协作队列整组准入，未使用共享卡借用；队列外任务不受控制。CPU 是共享主机上的固定
affinity/NUMA 观察。9B 单卡绑定 NUMA4；17B 两卡主机内存 interleave 4/5；9B 四卡训练
绑定 NUMA5 并等待 17B 两卡任务结束。17B CPU 推理绑定 NUMA6、CPU 240–279，要求
全机可用内存至少 700 GiB 且 node6 free+file LRU+reclaimable slab 至少 160 GiB，RSS
界限 192 GiB。CPU 训练等待此推理与 Attention 测量结束，使用同一 CPU affinity、
内存 interleave NUMA6/7；要求两个节点各自上述 free+reclaimable 至少 210 GiB、全机
至少 700 GiB，RSS 界限 384 GiB。这些是准入观察，不是预留。全部超时界限在 JSON。

这些大规模记录保留原 0992 启动环境身份：当时 PYTHONPATH 仅指向 frozen source。
后续 `ca6199d4` 修复本地 launcher 对模块 CANN/TBE 路径的覆盖，模型 src/tide Git tree
与 0992 相同，公共环境未改动。新环境的实际导入与小型 NPU 复核见[放置报告](roadmap-placement-20261118.md)：
auto/BF16 完成，synchronized/direct FP32 各保留事件句柄失败。旧成功结果没有重标为新环境实测，
活动大任务也未被修改。

原 `scale-npu-hb-9b-v12`、`scale-two-npu-hb-17b-v12`、
`scale-four-npu-hb-9b-train-v14`、`scale-cpu-hb-17b-v15/v16` 在计算前为重排 NUMA 资源停止。
保留原终止状态、signal/exit、cleanup 和 `superseded.json`；它们既不是硬件失败，也不是通过。

预定范围为：9B 单 NPU FP32/BF16 推理 Tide+A–D；17B 两 NPU 两精度推理；9B 四 NPU
两精度各两次连续 AdamW；17B CPU 经 16/32/40 线程有限校准后，两精度推理 Tide+A–D，
以及继承所选线程数的 BF16 两次连续 AdamW。未完成或被界限阻止的项目不列作性能样本。

## 已完成 A–D 比较

A/B 同时匹配节点 block 数与总/实际有效触及参数；C/D 同时匹配层深与总/实际有效触及参数。
匹配以同精度、同模式、同输入/初参/零状态的独立审计固定，使用真实参与计算的 Transformer
权重。异宽输入共享特征前缀。不同数学主干的用时比不说明任务质量。

每行实际包含一个 warmup 和三个计时样本；完整样本与匹配目标/偏差在 JSON。

| 任务 / 精度 | 主体 / 模式 | 参数 | 中位数 ms | 输入位置/s | peak allocated GiB |
|---|---|---:|---:|---:|---:|
| scale-npu-hb-9b-v16 / fp32 | tide / inference | 8,778,358,224 | 887.250 | 9.017 | 32.980 |
| scale-npu-hb-9b-v16 / fp32 | transformer_A / inference | 8,778,632,400 | 951.778 | 8.405 | 33.507 |
| scale-npu-hb-9b-v16 / fp32 | transformer_B / inference | 2,433,394,080 | 931.879 | 8.585 | 9.125 |
| scale-npu-hb-9b-v16 / fp32 | transformer_C / inference | 8,778,353,328 | 55.042 | 145.345 | 32.725 |
| scale-npu-hb-9b-v16 / fp32 | transformer_D / inference | 2,433,606,400 | 13.005 | 615.141 | 9.088 |
| scale-npu-hb-9b-v16 / bf16 | tide / inference | 8,778,358,224 | 970.698 | 8.241 | 33.284 |
| scale-npu-hb-9b-v16 / bf16 | transformer_A / inference | 8,778,632,400 | 1070.909 | 7.470 | 33.507 |
| scale-npu-hb-9b-v16 / bf16 | transformer_B / inference | 2,441,860,512 | 1085.393 | 7.371 | 9.144 |
| scale-npu-hb-9b-v16 / bf16 | transformer_C / inference | 8,778,353,328 | 88.793 | 90.097 | 35.792 |
| scale-npu-hb-9b-v16 / bf16 | transformer_D / inference | 2,441,877,456 | 20.716 | 386.178 | 9.591 |

| 任务 / 精度 / 类别 | block 目标 / 实际 | 参数目标 / 实际 | 参数偏差 | width / FFN |
|---|---|---|---:|---|
| scale-npu-hb-9b-v16 / fp32 / A | 848 / 848 | 8778358224.000 / 8778632400 | 0.003123% | 720 / 3832 |
| scale-npu-hb-9b-v16 / fp32 / B | 848 / 848 | 2433608656.375 / 2433394080 | -0.008817% | 416 / 1744 |
| scale-npu-hb-9b-v16 / fp32 / C | 5 / 5 | 8778358224.000 / 8778353328 | -0.000056% | 11408 / 36088 |
| scale-npu-hb-9b-v16 / fp32 / D | 5 / 5 | 2433608656.375 / 2433606400 | -0.000093% | 6400 / 16816 |
| scale-npu-hb-9b-v16 / bf16 / A | 848 / 848 | 8778358224.000 / 8778632400 | 0.003123% | 720 / 3832 |
| scale-npu-hb-9b-v16 / bf16 / B | 848 / 848 | 2441873371.875 / 2441860512 | -0.000527% | 416 / 1752 |
| scale-npu-hb-9b-v16 / bf16 / C | 5 / 5 | 8778358224.000 / 8778353328 | -0.000056% | 11408 / 36088 |
| scale-npu-hb-9b-v16 / bf16 / D | 5 / 5 | 2441873371.875 / 2441877456 | 0.000167% | 4976 / 26080 |

以下比率为同一结果文件的 Tide 中位数 / 对应已完成基线中位数；小于 1 表示 Tide 用时较短。

| 任务 / 精度 / 类别 | Tide / baseline 时间 | Tide / baseline peak |
|---|---:|---:|
| scale-npu-hb-9b-v16 / fp32 / A | 0.9322× | 0.9842× |
| scale-npu-hb-9b-v16 / fp32 / B | 0.9521× | 3.6143× |
| scale-npu-hb-9b-v16 / fp32 / C | 16.1196× | 1.0078× |
| scale-npu-hb-9b-v16 / fp32 / D | 68.2230× | 3.6287× |
| scale-npu-hb-9b-v16 / bf16 / A | 0.9064× | 0.9933× |
| scale-npu-hb-9b-v16 / bf16 / B | 0.8943× | 3.6399× |
| scale-npu-hb-9b-v16 / bf16 / C | 10.9321× | 0.9299× |
| scale-npu-hb-9b-v16 / bf16 / D | 46.8578× | 3.4704× |

## 已完成多设备/CPU 容量执行

| 任务 / 精度 / 模式 | 完整秒数 | 每卡 timed peak allocated GiB | 构造/转移秒数 | 结果 |
|---|---|---|---:|---|
| scale-two-npu-hb-17b-v16 / bf16 / inference | 1.492621, 1.458641, 1.478270 | npu:0: 42.980, npu:1: 21.351 | 166.585 | completed |
| scale-two-npu-hb-17b-v16 / fp32 / inference | 1.255810, 1.293946, 1.283656 | npu:0: 42.406, npu:1: 20.778 | 166.064 | completed |
| scale-four-npu-hb-9b-train-v16 / bf16 / capacity_train | 5.230806, 2.987303 | npu:0: 49.353, npu:1: 49.638, npu:2: 24.852, npu:3: 22.114 | 87.126 | completed |
| scale-four-npu-hb-9b-train-v16 / fp32 / capacity_train | 4.874603, 2.926335 | npu:0: 51.899, npu:1: 52.178, npu:2: 25.842, npu:3: 22.804 | 87.271 | completed |

真实更新后的存储（GiB）与状态推进如下，分设备明细、dtype、optimizer step 标量在 JSON。

| 任务 / 精度 / update | 状态位置 | 参数 / 梯度 / optimizer payload GiB | 最大单设备 combined payload GiB | private state MiB | finite |
|---|---|---|---:|---:|---|
| scale-four-npu-hb-9b-train-v16 / bf16 / 1 | 0→8 | 32.702 / 32.702 / 65.404 | 44.440 | 32.225 | True |
| scale-four-npu-hb-9b-train-v16 / bf16 / 2 | 8→16 | 32.702 / 32.702 / 65.404 | 44.448 | 56.831 | True |
| scale-four-npu-hb-9b-train-v16 / fp32 / 1 | 0→8 | 32.702 / 32.702 / 65.404 | 44.446 | 50.225 | True |
| scale-four-npu-hb-9b-train-v16 / fp32 / 2 | 8→16 | 32.702 / 32.702 / 65.404 | 44.460 | 92.831 | True |

推理容量每次从零状态开始，计时包含前向、readout、必要的 Timed flush 和全部设备同步，
构造/输入转移单列；这不构成缓存 decode 或首次服务输出时延测量。
训练使用 `placement_adamw(foreach=False)`，每次完整墙钟时间包含 zero_grad、准备、
packing、前向/flush/loss、反向、跨设备运输、AdamW、detach、最终同步及期间的阶段记录 I/O。
私有状态在两次更新之间续接；每个真实更新前以相同参数/输入/状态做独立实际工作审计。
完整有限值扫描、实际参数/梯度/moment/状态存储及采样参数变化见证放在计时外。
容量模式只有一份放置模型，不分配大规模等价性参考或大 checkpoint。

两次容量更新没有额外 warmup；第一步建立 AdamW slots，第二步状态与实际工作可以变化。
其时间差不是同轨迹优化收益，也不构成稳定训练吞吐。容量脚本用 seed807 和普通随机输入，
A–D benchmark 用 seed1107 和特征配对输入，因此 17B CPU 表与两卡容量表不构成同初参/
同输入的设备速度比。17B NPU 的 A–D 尚未测量。

FP32/BF16 均用 FP32 参数/master、梯度及 AdamW moments；BF16 只改变声明的计算和
部分 KV 存储精度。实际 peak allocated 与 reserve、RSS、可见 payload 和 retained storage
是不同口径。计时峰值在审计图释放后重置，但 reserve 可保留审计缓存；CPU RSS 覆盖整个
进程生命周期，不能当作某个 Transformer 单独的峰值。

## 原始失败与剩余范围

当前未出现已记录的内部失败；运行中/等待状态不等于通过。

当前整层放置下 17B 即使五卡也有单层 dense AdamW 持久预算超过 60 GiB，故不把未执行的
17B NPU 训练列为通过。CPU 替代只有实际完成记录才能建立容量结论。大规模 checkpoint/
恢复后更新、17B CPU FP32 训练、单独计时的含梯度前向、大规模 A–D 训练比较、长窗口/
decode 和任意多步轨迹不在这些大模型作业的实测范围。
小型 FP32/BF16 数值失败仍以原报告为准，finite 标签不取消它们。

复现使用 JSON 的完整固定源码命令、新输出路径、有限资源界限与当次授权的协作准入，见
[执行契约](../execution-flows.md)和[放置入口](../placement.md)。读取本报告无需重启计算。
