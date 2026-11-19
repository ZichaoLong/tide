# 执行候选性能、存储与有限扩批量观察 · 2026-11-18

四组同卡固定形状训练比较完成五个候选、共100个正常计时样本；每组另有 baseline/recompute/combined 三个独立内存账本进程。
另两组是统一32 GiB预算的 Tide-only 扩批量观察，没有 Transformer 调优或 A–D 容量比较。
本页只读既有终态结果。样本、工作量、padding、活动变化、分阶段存储与原始身份分别保存在 [JSON](capacity-execution-20261118.json.gz)。

固定形状源码/设施为 `36c42a7a3ef765fe6257c42c7c1404d28857b77c`；扩批量为 `8a22530dae69d9f1fa5ead4a01d1798fd1dfe0ee`；数学为 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`。
数值资格见 [有限舍入调查](capacity-rounding-20261118.md) 及下列 v3/v4 原始记录。完成、工作计数相等或初始哈希相等均不构成轨迹等价资格。

## 同卡五候选正常计时

候选明确固定三个执行选项：baseline=(默认局部 Attention, region_batch1, 无重计算)；local_math 仅改局部 Attention 为 math；
batch4 仅把区域批量设为4；recompute 仅开启区域重计算；combined 同时启用 math/batch4/重计算。数学配置不变。
每组各自固定同一物理卡、输入和初始参数哈希，候选按上述顺序串行独立进程运行；Settle B8/L256，Timed B4/L128，FP32/BF16 分组。
每候选两次预热更新、五个正常训练样本，每步状态从零开始；训练包括必要 Timed flush、loss、反向、完整 AdamW foreach 与同步。
初态/后更新语义及 runtime 审计在计时外，内存账本全部另起进程。正常峰值在预热后重置、后审计前读取。
设备 allocated 预算48 GiB，子进程树采样RSS72 GiB。比值为候选/baseline，小于1表示对应耗时或峰值较小。

| 形态 / 精度 | 候选 | 中位秒 | 正常峰值 GiB | 时间比 | 峰值比 |
|---|---|---:|---:|---:|---:|
| Settle / BF16 | baseline | 2.019 | 13.533 | 1.000 | 1.000 |
| Settle / BF16 | local_math | 2.016 | 13.519 | 0.999 | 0.999 |
| Settle / BF16 | batch4 | 1.546 | 13.659 | 0.766 | 1.009 |
| Settle / BF16 | recompute | 3.632 | 2.183 | 1.799 | 0.161 |
| Settle / BF16 | combined | 2.698 | 2.885 | 1.336 | 0.213 |
| Settle / FP32 | baseline | 1.908 | 15.404 | 1.000 | 1.000 |
| Settle / FP32 | local_math | 1.895 | 15.130 | 0.993 | 0.982 |
| Settle / FP32 | batch4 | 1.510 | 15.909 | 0.792 | 1.033 |
| Settle / FP32 | recompute | 3.459 | 2.283 | 1.813 | 0.148 |
| Settle / FP32 | combined | 2.583 | 3.323 | 1.354 | 0.216 |
| Timed / BF16 | baseline | 4.196 | 13.539 | 1.000 | 1.000 |
| Timed / BF16 | local_math | 4.077 | 13.629 | 0.972 | 1.007 |
| Timed / BF16 | batch4 | 2.950 | 13.715 | 0.703 | 1.013 |
| Timed / BF16 | recompute | 7.405 | 2.697 | 1.765 | 0.199 |
| Timed / BF16 | combined | 5.184 | 3.534 | 1.235 | 0.261 |
| Timed / FP32 | baseline | 3.955 | 15.269 | 1.000 | 1.000 |
| Timed / FP32 | local_math | 3.900 | 15.226 | 0.986 | 0.997 |
| Timed / FP32 | batch4 | 2.748 | 15.848 | 0.695 | 1.038 |
| Timed / FP32 | recompute | 6.873 | 3.090 | 1.738 | 0.202 |
| Timed / FP32 | combined | 4.934 | 4.141 | 1.248 | 0.271 |

本次 batch4 的完整训练时间比为0.695–0.792，峰值比为1.009–1.038；recompute 的峰值降至0.148–0.202，但耗时增至1.738–1.813。
combined 的峰值为0.213–0.271、耗时为1.235–1.354。这显示当前固定形状下，区域批量与重计算分别偏向吞吐和内存，组合没有消除重计算成本。
local_math 的时间差较小，不能凭一次固定顺序的五样本结果宣称稳定通用收益。全部观察的样本range/median低于25%，这仅是有限噪声描述，不是置信区间或随机交叉重复实验。

## 实际工作与活动漂移

初态参数/输入哈希均相同。语义审计的初态和两次预热加五次计时更新后的状态分列，后审计也从零私有状态执行同一输入。
表内活动差是候选减同组baseline的有效触及均值；K padding是executed_rows−selected_rows。
完整正消息、候选更新、Full、区域/KV有效对数、S packing和矩阵行数，以及各项差值均保留在JSON。

| 形态 / 精度 | 候选 | 初态触及差 | 更新后触及差 | K padding 初态 / 更新后 |
|---|---|---:|---:|---|
| Settle / BF16 | local_math | -43.439 | 1305.610 | 49049 / 65413 |
| Settle / BF16 | batch4 | -198.000 | 1063.610 | 55525 / 69339 |
| Settle / BF16 | recompute | 0.000 | 0.000 | 49363 / 66104 |
| Settle / BF16 | combined | 539.561 | -1094.927 | 55460 / 69982 |
| Settle / FP32 | local_math | 0.000 | 435.488 | 49069 / 64433 |
| Settle / FP32 | batch4 | 0.000 | 57.537 | 55149 / 70037 |
| Settle / FP32 | recompute | 0.000 | 90.537 | 49069 / 65490 |
| Settle / FP32 | combined | 0.000 | -609.512 | 55149 / 69648 |
| Timed / BF16 | local_math | -21.662 | -310.801 | 37382 / 40710 |
| Timed / BF16 | batch4 | -8.333 | -3689.954 | 45926 / 47362 |
| Timed / BF16 | recompute | 0.000 | 0.000 | 37316 / 40753 |
| Timed / BF16 | combined | -173.292 | -1988.395 | 46036 / 46041 |
| Timed / FP32 | local_math | 0.000 | -369.110 | 37563 / 39618 |
| Timed / FP32 | batch4 | 0.000 | -422.400 | 46427 / 46362 |
| Timed / FP32 | recompute | 0.000 | -91.641 | 37563 / 39733 |
| Timed / FP32 | combined | 0.000 | -209.387 | 46427 / 46981 |

例如Settle FP32初态batch4的K调用从32降至22，selected rows仍为150,803，executed rows从199,872升至205,952；
它减少调用但增加padding，这与本次更短时间、略高峰值同时出现。计数不提供关键路径时间分摊，不能将全部收益归因到某一个算子。
FP32初态汇总工作相同，七次更新后已存在工作差异；BF16的local_math/batch4/combined连初态汇总就有差异。
BF16 recompute在本组初态及后更新汇总相同，也仍不能据此证明support集合、输出、梯度和参数轨迹相等。有效触及量不是FLOPs或物理读取字节。

## 三候选独立内存账本

账本使用同初始参数/输入、两轮预热，再执行一个诊断更新；正常计时是五个更新，后审计不能与账本终态混同。
`actual_work=null` 明确表示账本不做语义WorkAudit。RuntimeAudit与saved_tensors_hooks属于侵入式诊断，其峰值不能替换正常峰值。
持久参数/梯度/优化器列为释放输出和图后的唯一存储；私有状态列是同阶段保留的detached状态。
saved列是外层hooks持有的同时存活唯一storage峰值。诊断峰值排除boundary，后续各阶段allocator peak从同一次reset累计。

| 形态 / 精度 | 候选 | saved唯一峰值 GiB | 参数+梯度+优化器 GiB | 私有状态 GiB | 诊断峰值 GiB |
|---|---|---:|---:|---:|---:|
| Settle / BF16 | baseline | 12.655 | 0.234 | 0.098 | 13.473 |
| Settle / BF16 | recompute | 1.375 | 0.234 | 0.098 | 2.511 |
| Settle / BF16 | combined | 1.375 | 0.234 | 0.098 | 3.913 |
| Settle / FP32 | baseline | 14.543 | 0.234 | 0.160 | 15.307 |
| Settle / FP32 | recompute | 1.381 | 0.234 | 0.160 | 2.685 |
| Settle / FP32 | combined | 1.381 | 0.234 | 0.160 | 4.435 |
| Timed / BF16 | baseline | 12.605 | 0.234 | 0.164 | 13.549 |
| Timed / BF16 | recompute | 1.396 | 0.234 | 0.164 | 2.698 |
| Timed / BF16 | combined | 1.396 | 0.234 | 0.164 | 3.532 |
| Timed / FP32 | baseline | 14.136 | 0.234 | 0.289 | 15.276 |
| Timed / FP32 | recompute | 1.402 | 0.234 | 0.289 | 3.088 |
| Timed / FP32 | combined | 1.402 | 0.234 | 0.289 | 4.139 |

这些配置实际持有15,694,224参数。参数、梯度与AdamW存储在两精度中均为FP32，合计约0.234 GiB；正常十余GiB峰值不能归结为参数持有量。
重计算的外层saved storage明显降低，正常峰值也降低；账本不保证看见checkpoint内嵌hooks或后端不透明分配。
各类别可能共享storage，saved峰值、私有状态、allocator峰值及foreach规划allowance不能相加。JSON同时保留exclusive归属、可见union、
逐阶段live allocated减device-visible union的余量；该余量包含未追踪张量和allocator/后端因素，不是测得的算子scratch或硬件传输量。

原36c42账本有两处标签/边界需按实际执行解读：内嵌runtime.instrumentation固定写“ordinary forward”，但诊断context实际覆盖forward/readout/loss/backward/optimizer；
boundary的allocator peak在reset之前，含此前预热历史；reset紧随boundary，其他阶段是本诊断段累计峰值，非单阶段增量。
旧raw保持不变。后续仅标签修复 `3543825aa61618b174efe915c3a36a8246acc6fe` 明确了这两点，没有重跑或更改本次数据身份。

## 32 GiB预算的选定Tide扩批量观察

以下均为BF16 Settle、L256、两预热/五采样：baseline预设B8/16，combined预设B16/32/64；各自独立卡运行。
没有调优Transformer，也没有把这些观察添加为A–D容量表。不同B的吞吐用于比较选定生产负载，不是同输入量加速比。

| 候选 | B | 中位秒 | 输入位置/s | 正常峰值 GiB | 稳定/准入 |
|---|---:|---:|---:|---:|---|
| baseline | 8 | 2.034 | 1007.12 | 13.517 | True/True |
| baseline | 16 | 2.781 | 1473.02 | 26.763 | True/True |
| combined | 16 | 3.930 | 1042.35 | 5.498 | True/True |
| combined | 32 | 6.430 | 1274.10 | 10.830 | True/True |
| combined | 64 | 11.878 | 1379.34 | 21.486 | True/True |

baseline的停止记录为 `candidate_limit`；最佳已测稳定点为B16，1473.02输入位置/s。

combined的停止记录为 `candidate_limit`；最佳已测稳定点为B64，1379.34输入位置/s。

按各自最佳已测点，combined吞吐为baseline的0.936倍；这是不同B、不同物理卡下的预算观察。本矩阵中更大的已完成B没有转化为更高的最佳吞吐。
这是预设候选内的有限观察，未证明全局最佳配置或最大可行B；尤其baseline没有尝试B32，不能把未试候选写成OOM失败。
32GiB是同步检查的allocated预算，子进程树采样RSS88GiB；原失败/停止状态和实际活动变化保留。

## 资格链接、资源与复现

数值比较与有限舍入诊断由 [capacity-rounding-20261118](capacity-rounding-20261118.md) 单列。既存v3/v4原始入口如下：

- execution-candidates-fp64-v3（原路径 `../../artifacts/execution-candidates-fp64-v3/results/suite.json`，已退役）：原suite状态 `completed`，源码 `efca7198570e8ee7a21ddccd81165eb36bdb269a`。
- execution-candidates-npu-v3（原路径 `../../artifacts/execution-candidates-npu-v3/results/suite.json`，已退役）：原suite状态 `cancelled`，源码 `efca7198570e8ee7a21ddccd81165eb36bdb269a`。
- execution-batched-npu-v4（原路径 `../../artifacts/execution-batched-npu-v4/results/suite.json`，已退役）：原suite状态 `failed`，源码 `231f56ebba3239230afacf467a80c2d27b292d12`。
- execution-timed-math-npu-v4（原路径 `../../artifacts/execution-timed-math-npu-v4/results/suite.json`，已退役）：原suite状态 `failed`，源码 `231f56ebba3239230afacf467a80c2d27b292d12`。
- execution-timed-recompute-npu-v4（原路径 `../../artifacts/execution-timed-recompute-npu-v4/results/suite.json`，已退役）：原suite状态 `completed`，源码 `231f56ebba3239230afacf467a80c2d27b292d12`。

v3 NPU候选因batch4/combined的BF16 bias epilogue实现问题中止，superseded记录和旧失败保留；v4部分suite仍为failed。
本页不依据后续性能完成把这些失败改判通过，也不把CPU FP64诊断升级为NPU FP32/BF16数值资格。

每进程四线程，固定形状每组候选共用其队列准入的物理卡；实际身份均为协作队列独占、外部任务不受控制。
候选前后host load、初始参数/输入hash、软件版本、算子dtype审计和全部命令留在JSON。同卡固定顺序仍有主机争用和时间漂移。

| 运行身份 | 物理NPU | CPU affinity | 外层时限s / RSS GiB |
|---|---:|---|---|
| execution-perf-settle-bf16-v1 | 14 | 104-107 | 2400 / 80.0 |
| execution-perf-settle-fp32-v1 | 4 | 100-103 | 2400 / 80.0 |
| execution-perf-timed-bf16-v1 | 7 | 112-115 | 2400 / 80.0 |
| execution-perf-timed-fp32-v1 | 2 | 108-111 | 2400 / 80.0 |
| execution-capacity-settle-baseline-v1 | 4 | 100-103 | 2400 / 96.0 |
| execution-capacity-settle-combined-v1 | 14 | 104-107 | 2400 / 96.0 |

原始 artifacts 已退役，旧逐文件核验依赖不再在线。现在可校验并还原保留于 Git 的审阅证据：

```sh
python scripts/materialize_record.py \
  --index docs/evidence/archive-index.json \
  --record docs/evidence/capacity-execution-20261118.json \
  --output /tmp/capacity-execution-20261118-review.json
```

重新执行时使用JSON中的冻结源码及`runs[].launch_arguments`/`children`，选择新输出身份并按当前资源重新准入。
正常性能、侵入式存储诊断、选定扩批量和独立数值资格各自保留边界。
