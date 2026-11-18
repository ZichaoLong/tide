# LibTorch 区域切片：NPU 资格与正常性能 · 2026-11-18

[JSON](native-regional-performance-20261118.json) 保留 120 个正常计时观察、600 个样本及 84 个末步数值比较；
三组作业均完成，80/84 个末步比较在原阈值内。完整 Tide 仍由 PyTorch 实现；本页仅涉及区域评分 Attention+FFN。
独立二进制固定到 `89ce08ff858724648152790502f8c65d8d8b1222`，CPU 构建/公式见[既有记录](native-regional-cpu-20261118.md)。

## 硬件与数值边界

NPU 同 stream 检查 28/29 通过：两精度、三模式、1/2/4 workers、三步更新、新进程恢复及第四步。
唯一失败为 BF16 四 worker 恢复后的 `v_weight` 梯度，最大差 `7.62939453125e-6`；原 `atol=2e-6, rtol=2e-5` 未变。
另外八个 pool-stream 训练/恢复观察全部在原阈值内，覆盖 FP32/BF16、2/4 workers、三步后恢复第四步。
pool 的对照是已检查的同布局 Python 三步及 native 串行恢复结果；这不是多 stream 的完整 Tide 资格，也不覆盖跨 NPU transport。
原同 stream 失败没有因 pool 通过而改标；有限 CPU 重放未隔离该梯度超差，见[舍入调查](capacity-rounding-20261118.md)。

较大正常性能形状在五次更新后另出现四个严格比较失败：CPU BF16/4 workers，NPU BF16/1、2、4 workers。
CPU 最大输出差 `0.00814360`，另四个参数梯度张量超差；NPU 最大输出差分别 `0.00374132`、`0.00374132`、`0.00434518`，
各参数/梯度/AdamW moments 本身仍在原阈值内。这些都是有限值，执行完成与数值通过分列；不能称这些 BF16 训练吞吐为严格等价加速。
只读末步参数检查发现 CPU 最多约 `1.90e-6`、NPU 最多约 `8.95e-7` 的 FP32 参数差，并观察到若干参数 BF16 cast 不同。
这是五步后的状态，不能倒推第五次前向的首因；未进一步隔离每条训练轨迹，不能把全部超差归为同一个原因。

## 负载和计时

所有任务共享一组区域参数；相容节点在区域内批量处理，外层 worker 执行独立区域任务。两形状为：

| 名称 | 区域任务数 | 节点/区域 | B | L | 描述宽度 |
|---|---:|---:|---:|---:|---:|
| medium | 4 | 8 | 8 | 128 | 64 |
| large | 4 | 16 | 16 | 128 | 128 |

三组实验的 fixture SHA256 完全一致。每个观察是独立进程；两次预热使用丢弃的 owner，随后新的 owner 记录五次同步采样。
训练含 zero_grad、四任务 forward/loss/backward、一次共享 AdamW 和最终设备同步；首个采样仍含该 owner 的首次 slot 建立。
不含 fixture/model 加载、线程池创建、输出序列化及事后数值比较。训练样本连续改变参数，不能当 IID；无梯度推理与含梯度前向分别测量。
下面是五次样本中位数，单位毫秒。`†` 表示 range/median>25%，保留而不用于稳定排序；它也可能反映首次 slot 成本，非统计置信区间。

### CPU

| 形状 / 精度 / 模式 | Python串行 inner1 | C++串行 inner1 | C++2任务 inner1 | C++4任务 inner1 | Python串行 inner4 | C++串行 inner4 |
|---|---:|---:|---:|---:|---:|---:|
| medium / float32 / inference | 59.785 | 57.994 | 29.074 | 28.906 | 72.045† | 22.347† |
| medium / float32 / grad_forward | 65.186 | 61.940 | 31.873 | 23.807† | 40.245† | 34.204† |
| medium / float32 / train | 178.600 | 176.727 | 91.460 | 80.243 | 256.615† | 107.999† |
| medium / bfloat16 / inference | 154.193 | 156.293 | 80.062 | 73.334† | 132.087† | 136.590† |
| medium / bfloat16 / grad_forward | 73.638 | 69.847 | 36.541 | 33.779 | 80.777† | 68.284† |
| medium / bfloat16 / train | 190.170 | 186.632 | 98.261 | 80.270† | 187.685† | 203.416† |
| large / float32 / inference | 749.960 | 698.311 | 392.747 | 311.603 | 394.344 | 381.927 |
| large / float32 / grad_forward | 757.365 | 739.692 | 406.425 | 308.543 | 402.771 | 411.899 |
| large / float32 / train | 2237.102 | 2234.978 | 1209.615 | 870.141 | 1276.067 | 1351.920 |
| large / bfloat16 / inference | 1714.786 | 1720.827 | 905.245 | 666.646 | 890.289 | 912.259 |
| large / bfloat16 / grad_forward | 845.182 | 831.141 | 448.198 | 374.068 | 447.821 | 482.278 |
| large / bfloat16 / train | 2318.437 | 2339.722 | 1254.817 | 1032.077 | 1213.176 | 1214.561 |

### NPU

| 形状 / 精度 / 模式 | Python串行 inner1 | C++串行 inner1 | C++2任务 inner1 | C++4任务 inner1 |
|---|---:|---:|---:|---:|
| medium / float32 / inference | 3.575 | 2.774 | 2.700† | 2.849 |
| medium / float32 / grad_forward | 4.316 | 2.738 | 2.664 | 2.789 |
| medium / float32 / train | 13.882 | 8.498 | 7.813† | 7.561 |
| medium / bfloat16 / inference | 4.502 | 4.091 | 4.223 | 3.940 |
| medium / bfloat16 / grad_forward | 5.486 | 4.215 | 4.194 | 4.189 |
| medium / bfloat16 / train | 15.573 | 10.069† | 9.834† | 9.794† |
| large / float32 / inference | 5.201 | 5.180 | 5.153 | 5.608 |
| large / float32 / grad_forward | 5.322† | 5.295 | 5.363 | 5.646 |
| large / float32 / train | 14.825 | 14.413 | 15.192† | 15.594 |
| large / bfloat16 / inference | 7.095 | 7.090 | 7.046† | 7.574† |
| large / bfloat16 / grad_forward | 7.452† | 7.446 | 7.283 | 7.659† |
| large / bfloat16 / train | 17.640 | 18.035 | 17.206† | 17.906 |

CPU 相同 inner1 下，C++ 串行与 Python 串行的大形状训练时间接近；外层区域并发能利用更多 CPU。
同样四 CPU 预算，大形状 FP32 的 C++ 四任务约 870 ms，Python 串行 inner4 约 1276 ms；没有测 Python 线程池。
较小形状的 inner4 多个观察波动很大，不给语言或线程策略的普遍排序。
NPU medium/FP32 训练 Python/C++串行/四任务约 13.882/8.498/7.561 ms；large/FP32 则约 14.825/14.413/15.594 ms。
这支持在小算子形状研究提交成本；并发没有随形状增大而持续获益，不能据局部结果替换整个模型。

## 资源与复现

CPU 使用 affinity 76–79，inner1 与 inner4 两组串行衔接；NPU 正常性能在物理1、host affinity120–123，
pool 也是独占队列准入的物理1、host affinity140–143。CPU/内存/文件系统仍是共享主机资源。
每个 C++ 子进程保留实际编译源、ABI、库加载路径、current/pool stream IDs，明确无 Python runtime 库依赖。
BF16 projection/FFN dtype 由执行 guard 检查；参数/梯度/AdamW、Attention乘积和标量 readout 为 FP32。
没有原生逐算子 placement trace 或设备 allocator 峰值；RSS 由外围进程树采样，不能当显存。没有 CUDA 构建或真机资格。

本阶段保留 PyTorch 主实现与可选 LibTorch 区域实验；同 stream 默认、pool 单列实验，没有自动选择语言或完整 native backend。
复现先按[native入口](../native-regional.md)构建固定源码，使用新输出目录和当次准入。原始命令在 JSON `runs[].launch.command`；
冻结 driver 为 `artifacts/native-performance-driver-v1.py`、`artifacts/native-performance-inner4-driver-v1.py`，其完整字节另随本证据提交到
`native-regional-performance-driver-v1.py` 与 `native-regional-performance-inner4-driver-v1.py`；运行时必须位于记录源码根并由 launcher 传入新输出。
同 stream/pool 的逐案例命令、日志、输入和对照位置在相应 JSON 中，CPU只读末步参数检查不构成新模型资格。
