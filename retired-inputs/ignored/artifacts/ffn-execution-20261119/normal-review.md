常规计时只读汇总。未完成套件标为 pending；下列已有数值只来自已完成且文件哈希匹配的子任务。

吞吐按每个 repeat 的总输入位置数 / 所有 normal 完整时间之和计算；median/range 汇总 repeat，不把 sample 中位数当配对 repeat。校准只有一个 repeat。阶段均值采用互不重叠边界及显式剩余时间；首输出、stable decode 和其他子项不重复加总。

| 套件 | 状态 | 已完成/计划子任务 | warm / normal / repeat | CPU worker affinity |
|---|---|---:|---|---|
| ffn-medium-cpu-timing-v1 | complete | 12/12 | 2/5/3 | 240-271 |
| ffn-large-cpu-timing-v1 | complete | 6/6 | 1/2/3 | 0-31 |
| ffn-large-npu-timing-v1 | complete | 6/6 | 2/5/3 | 224-227 |
| ffn-long-npu-timing-v2 | complete | 6/6 | 1/2/3 | 224-227 |
| ffn-cpu16-calibration-v1 | complete | 2/2 | 2/5/1 | 240-255 |
| ffn-cpu32-calibration-v1 | complete | 2/2 | 2/5/1 | 240-271 |
| ffn-medium-npu-timing-v1 | complete | 15/15 | 2/5/3 | 224-227 |

长 decode 选择 `ffn-long-npu-timing-v2`。v1 因前置 Transformer 诊断误传 `--config` 而在进入 queue 前取消，无模型结果；CPU TF v1 参数检查失败、NPU TF v1 取消及各自 replacement 身份也保留在 JSON，均不并入 normal 数字。

CPU32 中尺度 worker 为 240–271，大尺度为 0–31；另做的阶段诊断使用 120–151，其时间不进入这里的 normal 分母。监督器 affinity 调整也不改写 worker 身份。当前约 3GHz/performance governor、worker 页超过 99.5% 在各自本地 NUMA 的只读快照，并未解释旧时间差；不能把 NUMA3 诊断单次 TF 约 3.64s 与 NUMA6 校准约 10.94s 混成正式加速，或自动归因频率/远端内存。

中尺度 Transformer：32 层，d496、FFN2048、31×16 heads、window128，129,039,856 参数；大尺度固定参考：32 层，d4352、FFN15208、68×64 heads、window128，8,778,371,328 参数。大尺度 Tide KV128 为 7,268,408,784 参数；该固定 Transformer 是辅助参考，不能替代 A–D。

| 套件 / 对象 | 每 repeat 完整 positions/s | median [min,max] | 同 repeat 吞吐比 / Tide baseline（median） | allocated/reserved peak GiB | normal-end RSS GiB |
|---|---|---|---:|---|---:|
| ffn-medium-cpu-timing-v1 / baseline | r0=90.258, r1=94.076, r2=93.050 | 93.050 [90.258,94.076] | 1.000 | CPU 无 device allocator | 30.921 |
| ffn-medium-cpu-timing-v1 / W | r0=75.071, r1=78.856, r2=78.988 | 78.856 [75.071,78.988] | 0.838 | CPU 无 device allocator | 29.085 |
| ffn-medium-cpu-timing-v1 / P | r0=78.962, r1=78.805, r2=80.174 | 78.962 [78.805,80.174] | 0.862 | CPU 无 device allocator | 26.671 |
| ffn-medium-cpu-timing-v1 / Transformer32 | r0=258.396, r1=233.510, r2=239.461 | 239.461 [233.510,258.396] | 2.573 | CPU 无 device allocator | 9.578 |
| ffn-large-cpu-timing-v1 / baseline | r0=19.410, r1=18.410, r2=19.220 | 19.220 [18.410,19.410] | 1.000 | CPU 无 device allocator | 40.665 |
| ffn-large-cpu-timing-v1 / Transformer32 | r0=33.240, r1=33.200, r2=33.268 | 33.240 [33.200,33.268] | 1.731 | CPU 无 device allocator | 37.191 |
| ffn-large-npu-timing-v1 / baseline | r0=115.493, r1=115.180, r2=115.682 | 115.493 [115.180,115.682] | 1.000 | npu:0: 33.491/35.854 | 28.889 |
| ffn-large-npu-timing-v1 / Transformer32 | r0=792.025, r1=789.943, r2=789.494 | 789.943 [789.494,792.025] | 6.858 | npu:0: 34.866/37.639 | 34.540 |
| ffn-long-npu-timing-v2 / baseline | r0=26.919, r1=26.585, r2=26.964 | 26.919 [26.585,26.964] | 1.000 | npu:0: 33.507/36.193 | 28.890 |
| ffn-long-npu-timing-v2 / Transformer32 | r0=198.870, r1=199.590, r2=198.893 | 198.893 [198.870,199.590] | 7.388 | npu:0: 34.898/37.670 | 34.542 |
| ffn-cpu16-calibration-v1 / baseline | r0=85.039 | 85.039 [85.039,85.039] | 1.000 | CPU 无 device allocator | 28.091 |
| ffn-cpu16-calibration-v1 / Transformer32 | r0=208.501 | 208.501 [208.501,208.501] | 2.452 | CPU 无 device allocator | 7.278 |
| ffn-cpu32-calibration-v1 / baseline | r0=94.556 | 94.556 [94.556,94.556] | 1.000 | CPU 无 device allocator | 28.639 |
| ffn-cpu32-calibration-v1 / Transformer32 | r0=187.657 | 187.657 [187.657,187.657] | 1.985 | CPU 无 device allocator | 7.342 |
| ffn-medium-npu-timing-v1 / baseline | r0=961.595, r1=883.923, r2=883.702 | 883.923 [883.702,961.595] | 1.000 | npu:0: 22.739/23.777 | 4.428 |
| ffn-medium-npu-timing-v1 / W | r0=942.152, r1=882.153, r2=875.783 | 882.153 [875.783,942.152] | 0.991 | npu:0: 21.517/22.520 | 4.435 |
| ffn-medium-npu-timing-v1 / P | r0=821.560, r1=747.421, r2=761.036 | 761.036 [747.421,821.560] | 0.854 | npu:0: 20.598/21.494 | 4.457 |
| ffn-medium-npu-timing-v1 / WP | r0=788.020, r1=718.749, r2=707.665 | 718.749 [707.665,788.020] | 0.813 | npu:0: 19.273/20.291 | 4.465 |
| ffn-medium-npu-timing-v1 / Transformer32 | r0=4593.017, r1=4501.196, r2=4448.307 | 4501.196 [4448.307,4593.017] | 5.034 | npu:0: 6.644/7.670 | 4.256 |

中尺度 CPU 每 repeat 的完整训练均值（秒）。forward window 含准备和窗口调度，prefill 是其子项；其他边界时间由完整时间减去前三项得到。均值可加，子项不再次相加。

| 对象/repeat | forward window | backward | optimizer | 其他边界 | 完整均值 | 同 repeat 吞吐比 |
|---|---:|---:|---:|---:|---:|---:|
| P/r0 | 5.467645 | 20.275698 | 0.191098 | 0.002141 | 25.936583 | 0.874846 |
| P/r1 | 5.475369 | 20.335109 | 0.176445 | 0.001436 | 25.988358 | 0.837665 |
| P/r2 | 5.454718 | 19.871827 | 0.216436 | 0.001412 | 25.544392 | 0.861628 |
| Transformer32/r0 | 2.789798 | 4.866637 | 0.268162 | 0.001213 | 7.925810 | 2.862865 |
| Transformer32/r1 | 3.074459 | 5.412732 | 0.281997 | 0.001307 | 8.770495 | 2.482132 |
| Transformer32/r2 | 3.167935 | 5.130084 | 0.252764 | 0.001747 | 8.552531 | 2.573479 |
| W/r0 | 7.313456 | 19.797036 | 0.168803 | 0.001422 | 27.280716 | 0.831742 |
| W/r1 | 7.180032 | 18.613048 | 0.176386 | 0.001996 | 25.971462 | 0.838209 |
| W/r2 | 7.161216 | 18.569881 | 0.195429 | 0.001480 | 25.928007 | 0.848880 |
| baseline/r0 | 4.992913 | 17.502816 | 0.192418 | 0.002374 | 22.690520 | 1.000000 |
| baseline/r1 | 5.008386 | 16.585657 | 0.173095 | 0.002388 | 21.769526 | 1.000000 |
| baseline/r2 | 4.873655 | 16.942321 | 0.191865 | 0.001915 | 22.009756 | 1.000000 |

| 推理套件 / 对象 | 首主干输出 repeat-median 的中位数 s | prefill positions/s 中位数 | stable decode positions/s 中位数 | 独立 batch 请求/s 中位数 |
|---|---:|---:|---:|---:|
| ffn-large-cpu-timing-v1 / baseline | 6.426261 | 38.520 | 3.843 | 0.133 |
| ffn-large-cpu-timing-v1 / Transformer32 | 3.681484 | 68.439 | 6.506 | 0.231 |
| ffn-large-npu-timing-v1 / baseline | 0.742524 | 337.104 | 18.449 | 0.802 |
| ffn-large-npu-timing-v1 / Transformer32 | 0.129124 | 1962.448 | 137.114 | 5.486 |
| ffn-long-npu-timing-v2 / baseline | 0.746970 | 335.178 | 18.436 | 0.070 |
| ffn-long-npu-timing-v2 / Transformer32 | 0.129106 | 1965.587 | 137.226 | 0.518 |

训练每 normal 更新为 B16×128=2048 个输入位置；5 normal 的 repeat 分母为 10240 / Σ完整秒。大 CPU 是 1 warm + 2 normal × 3 repeat；短 NPU 是 2 warm + 5 normal × 3 repeat；长 NPU 是 1 warm + 2 normal × 3 repeat。短推理每请求 B8×(128+16)=1152 位置，稳定 decode 为 B8×15=120 位置；长推理每请求 3072 位置，稳定 decode 为 B8×255=2040 位置。warmup 不入 normal 分母。

CPU/NPU 同形初始数据检查：
- medium tide/baseline：input=True，initial parameters=True，shape=True。
- medium tide/weight_access：input=True，initial parameters=True，shape=True。
- medium tide/packing：input=True，initial parameters=True，shape=True。
- medium transformer/baseline：input=True，initial parameters=True，shape=True。
- large tide/baseline：input=True，initial parameters=True，shape=True。
- large transformer/baseline：input=True，initial parameters=True，shape=True。

16/32 线程校准（各仅一个 repeat，不代表全局线程最优）：
- baseline：85.039 / 94.556 positions/s，32/16 比 1.112。
- Transformer32：208.501 / 187.657 positions/s，32/16 比 0.900。

W/P 原有限数值失败保留，normal 完成不代表严格等价。训练经过 warmup 后，参数、优化器及私有状态轨迹不同，r0 实际活动也不完全相同；这些是固定形状、初始数据与 batch 的有限执行比较。诊断 work 不是每个计时更新的跟踪；ledger 的阶段峰值累计，不能相加，allocator residual 也不能全部叫临时工作区。

CPU allocator 空值使用独立的 RSS 记录；RSS 含进程生命周期内构造及 host snapshot，不称 HBM。NPU normal allocator、诊断 ledger、最终 RSS 和监督器进程树 RSS 分开保留。训练 allocator 在 run_stream 返回后采样，包含该函数里计时外的有限性检查，先于另做的 continuation 检查。

每个 repeat 的原始 normal、阶段均值可加性、同 repeat 比值、r0 work/matrix/padding 和完整 ledger、源码/输入指纹及原资格记录见 `normal-review.json`。首输出是主干可用输出，不是完整语言模型 TTFT；此证据不评估下游任务质量。

复现汇总：`python artifacts/ffn-execution-20261119/collect_normal.py`。仅读取现有证据并原子写入汇总，不运行模型或设备。
