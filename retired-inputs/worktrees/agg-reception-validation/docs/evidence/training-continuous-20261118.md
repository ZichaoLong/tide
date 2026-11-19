# 二十步连续训练与恢复后的真实更新

六个负载各完成二十次真实 AdamW 更新，保留跨更新私有历史、每次更新后 detach；
随后完成配对 checkpoint 检查，并在独立新进程执行一次真实更新。
详细逐步内存、optimizer计数、KV容量、哈希、命令和原失败见
[JSON](training-continuous-20261118.json)。训练推荐见
[使用建议](../execution-recommendations.md)。这里验证训练机制，不评价任务质量。

## 连续负载与结果

预热为一次单独推理。每次更新的 prefill chunk 等于完整 L，decode 为0，
Timed 每次输入窗口后完整 flush。稳态吞吐是更新3–20的**总输入位置/总完整秒数**；
中位时间单列。完整时间含拼接、loss、反向、AdamW、detach和同步。
所有峰值取二十步最大值，allocated 与 reserved 是不同量，不能相加。

| 配置 | B/L | 中位秒 | 输入位置/秒 | allocated GiB | reserved GiB | 末位置 |
|---|---|---:|---:|---:|---:|---:|
| FC-K BF16 | 128/256 | 0.855 | 38249 | 25.69 | 29.94 | 5120 |
| Settle-X FP32，三项组合 | 24/256 | 2.961 | 2075 | 38.29 | 43.00 | 5120 |
| Settle-X BF16，三项组合 | 32/256 | 3.947 | 2078 | 41.84 | 50.63 | 5120 |
| Timed-X FP32，三项组合 | 12/128 | 4.827 | 317 | 41.65 | 46.39 | 2600 |
| HB 0.5B FP32，三项组合 | 1/128 | 1.323 | 96.1 | 15.37 | 16.51 | 2560 |
| HB 0.5B BF16，三项组合 | 1/128 | 1.403 | 90.6 | 13.18 | 14.64 | 2560 |

三项组合为 `region_batch_size=4`、`ssd_summary_backend="prefix"`、
`checkpoint_reception=True`；FC-K保留原计算策略。FC/0.5B冻结 `ae247dee`，
Settle/Timed冻结 `22527cf4`，二者 `src/tide` tree 相同。
Settle BF16在本点的持续吞吐基本相当，reserved反而多7.63 GiB，因此优先推荐FP32。
0.5B也优先FP32，BF16保留为内存备选。0.5B的本轮负载与旧chunk64+decode8不同，
不能把与旧报告的巨大差值全部归因于新优化。

六组逐步输出、参数、梯度、optimizer和私有状态均有限；更新后状态无残留计算图。
所有活跃参数的AdamW step逐次1→20。FC/Settle/0.5B的KV槽长度填满且不超过128；
Timed填满且不超过512。Timed每更新128个输入位置，flush另推进2个输入周期，
即8个逻辑tick；`state.position`按输入周期计，故20步末位置2600。
flush输出槽不计为输入token。完整逐步记录保留了这种区别。

## 配对恢复与独立进程执行分别判定

同进程配对先保存参数、私有状态、完整非空AdamW，再由原owner和重新构造的owner
对同一下一窗口各执行真实更新。六组保存数据、下一前向与相关状态均逐位相同。
Settle两精度、Timed FP32、0.5B两精度的梯度、更新参数、optimizer和状态在原容差内。

**FC-K BF16保留 `finite_deviation`：** 27个参数梯度张量超差，最大绝对差
`6.103515625e-5`、最大scaled error22.55；3个AdamW一阶moment张量超差，
最大绝对差`6.103538908e-6`。输入梯度、更新参数在阈内，更新参数最大绝对差
`7.4505805969e-7`；更新状态精确。梯度/参数/slot阈值保持
`atol=1e-6, rtol=2e-3`，输入梯度仍使用原设施的`1e-6/1e-2`。
精确数据和前向没有证明相同反向轨迹，后续独立进程执行成功也不抹掉本失败。
本轮未逐算子解释这个大batch检查的所有梯度差异；小型同owner/跨实现检查见
[路径证据](training-paths-20261118.md)，此前有界恢复调查仍保持各自作用范围。

独立进程从20步checkpoint恢复，无模型/更新预热，运行seed752相同输入的第21次更新：

| 配置 | 载入与同步秒 | 首次恢复更新秒 | 结果 |
|---|---:|---:|---|
| FC-K BF16 | 0.748 | 2.245 | completed_finite |
| Settle-X FP32 | 1.146 | 4.692 | completed_finite |
| Settle-X BF16 | 1.144 | 5.966 | completed_finite |
| Timed-X FP32 | 1.286 | 6.998 | completed_finite |
| 0.5B FP32 | 7.914 | 2.697 | completed_finite |
| 0.5B BF16 | 7.815 | 2.907 | completed_finite |

这些进程核对了非空optimizer、活动参数step增量、浮点有限值、私有状态与输入哈希；
它们**只声明独立进程恢复和真实更新完成**，没有保存原owner全部第21步张量作数值配对，
因此不能赋予新的严格恢复等价资格。首次更新包括冷缓存/首次算子准备，不能替代稳态吞吐。

三份Settle/Timed独立恢复的命令将`--checkpoint-source`误传为`ae247dee`，
真实生产者是各自continuous记录中的`22527cf4`。JSON明确记录该标签纠正、真实生产源码、
checkpoint文件SHA256及二者相同的生产树；原始命令和结果均未改写。FC/0.5B的原标签正确。

## 内存余量与资格边界

设备总内存64 GiB。48 GiB界限检查采样allocated，不是reserved硬上限；
推荐应结合reserved和框架外占用。入场前设备空闲基底约2.8–3.3 GiB，
是资源快照，不是与上述峰值同时测得的HBM高水位。尤其Settle BF16 B32的
reserved已50.63 GiB，不能根据41.84 GiB allocated声称仍有约22 GiB真实余量。
FP32推荐点为下游数据、任务头和临时工作区保留更大空间；接入真实任务后需再次测量。
当前结果不保证任意后续轨迹、任意输入或无限步数不会OOM。

作业使用冻结源、background.slice、协作队列独占NPU准入、4个固定CPU线程，
80–100 GiB采样进程树RSS和有限超时。其他项目仍共享主机。没有修改公共环境或下游。
复现入口为`benchmarks.continuous`与`benchmarks.restore`，完整命令在JSON；
实际推荐参数可从命名预设加载。模型仍为PyTorch，单卡CPU/NPU资格不推广到CUDA、
peer DMA/HCCL或编译训练，也不改变多卡路径已保留的数值失败。
