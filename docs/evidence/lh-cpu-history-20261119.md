# 历史 LH CPU 吞吐定位

本附件回应本线程明确提出的历史查询，仅复核 Git 中保留的口述记录和旧实测报告。
未运行旧模型、恢复旧实现或重新认证历史数值；不并入当前 decode capacity 矩阵。
以下 `commit:path:line` 均指退役前完整快照
`728d19a470d16d392d5dd6a2452161d1a3e9959e`，可用 `git show COMMIT:PATH` 读取。

## 用户口述：原始未知项保留

记录日期为 2026-09-22；共同条件是旧 Intel CPU、56 核、batch 512。
下表吞吐和 batch-step 时间只是对口述均摊时间的算术换算，不是仓库新增实测。

| 口述配置 | 参数口述 / d / 名义活跃比例 | nograd 均摊 ms/sample-token | 换算总吞吐 sample-token/s | 换算秒/batch-step | 原 grad 列 ms/token |
|---|---|---:|---:|---:|---:|
| wide | 约“8.8B” / 2048 / 1/32 | 3.3 | 约 303.0 | 1.6896 | 3.6，阶段未确认 |
| narrow | 约“8.5B” / 128 / 1/64 | 29 | 约 34.48 | 14.848 | 136，阶段未确认 |

口述的 224 / 57,344 节点数可能不含 hub，且可能是 inet/onet 复用的单个静态图。
精确 Intel 型号、dtype、拓扑、权重、实际 selector 预算、token 步数/长度、缓存年龄及
reset 策略未恢复。dtype 仅记得是 LH 默认，FP32 未获确认；grad 究竟指含梯度前向
还是单独反向也未确认，不能写成完整训练吞吐。数百 token 内未明显变慢只是口述的
有限观察，不能解释为固定时间 Attention 或长上下文资格。

精确出处：

- 条件与两行口述：`728d19a470d16d392d5dd6a2452161d1a3e9959e:docs/lh-scale-benchmark.md:19`、`728d19a470d16d392d5dd6a2452161d1a3e9959e:docs/lh-scale-benchmark.md:23`、`728d19a470d16d392d5dd6a2452161d1a3e9959e:docs/lh-scale-benchmark.md:24`。
- 用户确认的 batch/步数归一化：`728d19a470d16d392d5dd6a2452161d1a3e9959e:docs/lh-scale-benchmark.md:28`。
- 硬件、dtype、grad 阶段及历史窗口未知项：`728d19a470d16d392d5dd6a2452161d1a3e9959e:docs/lh-scale-benchmark.md:34`。

后续记录称用户认为原“8.8B”可能是 Add。重建 Add 的 9,468,020,899 个参数除以
1024³ 为 8.818，但这不证明原口述使用了同一图或配置，也不把口述参数数目修正为精确值。
出处：`728d19a470d16d392d5dd6a2452161d1a3e9959e:docs/add-scale-comparison.md:3`。

另有“旧 Intel Attention 17 ms”的简短提及，缺少足够配置和阶段信息，本附件不据其
换算可比较吞吐。出处：
`728d19a470d16d392d5dd6a2452161d1a3e9959e:docs/evidence/add-scale-comparison.md:67`。

## 后续本机原 LH 实测：保留各自身份

这些结果是原报告记录的 CPU 测量，不能当作上表口述的精确复现。

| 旧实测配置/模式 | 实际参数 | B / 步数与采样 | CPU 资源 | sample-token/s | 秒/batch-step |
|---|---:|---|---|---:|---:|
| 原 LH Add，无梯度前向 | 9,468,020,899 | B512；12 步，前 4 步预热；3 个独立进程 | HiSilicon aarch64，56 物理核 | 171.698 | 2.9820 |
| 原 LH Add，含梯度前向 | 9,468,020,899 | 同上；保留窗口内状态和前向图 | 同上 | 140.286 | 3.6497 |
| 原 LH Attention，无梯度前向 | 17,269,426,339 | B512；全部 100 次 think | HiSilicon aarch64，160 核预算 | 37.5457 | 13.6367 |

### Add 的统计、软件与计时

固定 d2048、V50304、每 cortex 224 leaves + 8 hubs，名义 leaf 选择 1/32；
Add/all-softmax、retention 0.99、SiLU/RMS body Full、selected clear，每 token 两个
body tick。源码身份为报告中的 `0c053ebc7f8b3be03220c54aebf56502f5f0f6f1`，
原 LH 使用报告注明的 original-a10 快照；本附件没有重新认证该源码。
Torch/LibTorch 2.10.0+cpu、FP32；绑核 160–215，实际 ATen/OpenMP/OpenBLAS 均报 56，
interop 报 320 不代表 320 个活跃工作线程。主机共享，其他负载未完全控制。

主指标是三个独立进程各自测量窗口均值的中位数。索引 4–11 的同步 forward 墙钟计时
包含 embedding、body、vocabulary head；排除构造、token-ID 生成、旧 logits 销毁、
有限性/checksum 检查和日志，内部 RAII/phase/operator 计时关闭。
含梯度前向没有 backward、optimizer、zero_grad 或 detach，不能称完整训练。

精确出处：

- 冻结源码：`728d19a470d16d392d5dd6a2452161d1a3e9959e:docs/evidence/add-scale-comparison.md:3`；original-a10 快照说明：`728d19a470d16d392d5dd6a2452161d1a3e9959e:docs/evidence/add-scale-comparison.md:121`。
- 形状/参数与模块：`728d19a470d16d392d5dd6a2452161d1a3e9959e:docs/evidence/add-scale-comparison.md:11`。
- 硬件/软件与实际线程池：`728d19a470d16d392d5dd6a2452161d1a3e9959e:docs/evidence/add-scale-comparison.md:23`。
- 计时和 grad-forward 边界：`728d19a470d16d392d5dd6a2452161d1a3e9959e:docs/evidence/add-scale-comparison.md:36`。
- 三进程统计及结果：`728d19a470d16d392d5dd6a2452161d1a3e9959e:docs/evidence/add-scale-comparison.md:47`；两行结果：`728d19a470d16d392d5dd6a2452161d1a3e9959e:docs/evidence/add-scale-comparison.md:53`、`728d19a470d16d392d5dd6a2452161d1a3e9959e:docs/evidence/add-scale-comparison.md:55`。

### Attention 的统计、软件与计时

原 LH 版本 `a10fdb1883fccd63ec21e36cc9cffa294c63c9e2`，冻结 runner
`6ade84dffa2101bb18aa66afd12d167cdad0816e`。d2048、B512、selectnum 1、
100 次 think；使用重建并保留 seed7 的图，历史图字节和 RNG 状态不可得。
Torch 2.10.0+cpu、FP32，绑核 160–319；ATen/OpenMP 160、OpenBLAS 1。

37.5457/s 来自完整 100 步总量/Think 总时间，并非独立重复样本中位数。
计时沿用原整数毫秒 Think timer，包含原内层计时打印及首步状态初始化；构造和
teardown 属于外层进程时间。第 4–99 步均摊为 27.1745 ms/sample-token，不能与
全 100 步的 26.6342 ms/sample-token 混用。第 4–11 步与第 80–99 步还分别为
24.5820 / 28.5503 ms/sample-token；路由、缓存、占用和主机条件均可能变化，
不是受控上下文长度实验。100 次 think 和 block_size=4096 不证明长上下文性能。

精确出处：

- 源码、图、设备与主计时：`728d19a470d16d392d5dd6a2452161d1a3e9959e:docs/evidence/lh-a10-attention.md:12`。
- 参数、全程吞吐及不同窗口：`728d19a470d16d392d5dd6a2452161d1a3e9959e:docs/evidence/lh-a10-attention.md:40`。
- 原 Think timer 的打印/初始化边界：`728d19a470d16d392d5dd6a2452161d1a3e9959e:docs/lh-original-test.md:55`。

## B512 归一化与可引用范围

若 B 个样本共计 T 步、计时总量为 M 毫秒，则：

- 均摊 ms/sample-token = M / (B × T)。
- 总吞吐 sample-token/s = 1000 × B × T / M。
- 平均秒/batch-step = M / (1000 × T)，等于均摊 ms/sample-token × B / 1000。

因此 B512 的 3.3 ms/sample-token 表示总吞吐约 303/s，平均同步 batch-step 为
1.6896 s；它不是每条自回归序列 3.3 ms 的增量延迟，也不是首 token 时延。
归一化原契约：
`728d19a470d16d392d5dd6a2452161d1a3e9959e:docs/lh-scale-benchmark.md:140`。

口述可作为“当时用户报告”的参考，后续实测可按上述来源、模式和边界引用。
原 LH Add、原 LH Attention 与当前固定数学
`4f5b9f20a4030d68ee66ae2d26e34659a5e37234` 下的 FC/HB 是不同模型；
参数量相近不建立同函数、相同活跃工作、相同历史窗口或相同计时边界。
这些记录不能直接当作当前 Tide 逐序列 decode、首 token 时延、同数学比较或退化倍数，
也不提供旧大模型完整训练或跨实现数值等价资格。
这里的大模型历史点固定使用 B512，没有给出相同配置下的 batch 容量或吞吐搜索；
B512 本身也不能被认定为当时的最大可行或吞吐最优 batch。
