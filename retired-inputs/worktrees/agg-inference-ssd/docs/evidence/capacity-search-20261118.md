# 有界吞吐与规模搜索 · 2026-11-18

六组既有 v2 搜索共完成 19 个初态审计和 95 个训练计时子进程，保留 285 个原始样本。
下表覆盖较长输入窗口和 FC-K 独立批量搜索；所有搜索都因预设候选耗尽停止，未测得显存或全局吞吐极限。
本页由 `benchmarks.search_report` 核验原始引用并生成基础双表，再逐项复核计时、匹配与资源口径；没有重跑实验。
完整样本、构造器、活动变化、执行命令及哈希分列于 [JSON 证据](capacity-search-20261118.json)。

源码与测量设施均为 `f486a56ed7b6f68e2aff2e7100673bf2a2fe20be`；数学固定 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`。
资格仅为这些训练负载有限完成，不扩展为数值等价、持续训练、恢复或模型质量资格。

## 口径与停止边界

- 每个候选使用一个独立进程、一轮预热更新、三次正常训练采样；`train` 包括 zero_grad、准备与 packing、前向/readout、必要的 Timed flush、loss、反向、完整 AdamW foreach、设备同步及标量 loss 读取。每步从零状态 prefill，参数累计更新；后审计位于共四次更新之后。
- 模型构造、输入传输、初态/后更新审计、checkpoint、JSON 写入和样本后内存探针不计入上述秒数。峰值为预热后清零、后审计前的累计 allocated peak，包含当时存活的参数、梯度、优化器状态和执行临时量。
- 吞吐为实际 B×L 输入位置数除以完整秒数。Transformer 历史窗口固定 128；Tide 既定逻辑 KV 窗口 128×period 不变。扩大 L 是扩大输入流及反向图范围，没有修改历史语义。
- 两表均为同精度比较。BF16 是 autocast 矩阵计算；实际参数、梯度、AdamW 一二阶矩和主权重仍为 FP32，没有另设一份主权重。原始独立 runtime audit 记录了 BF16 矩阵输出；FP32 的 matmul/conv HF32 均关闭。
- 每卡 allocated 预算 48 GiB，单子进程树采样 RSS 预算 112 GiB；外层服务 128 GiB/5400 s，队列等待最多 600 s。搜索总限时 4800 s、单子进程 900 s。预算是同步点检查及采样监督，不是硬显存隔离。
- 候选按单轴倍增；稳定条件为三样本 `(max−min)/median ≤ 25%`。预设平坦规则是连续两个稳定候选相对最佳吞吐增益不超过 3%；本次所有线都先到 `candidate_limit`，没有 OOM、预算、超时或平坦停止。

## 同 B/L 的四类联合匹配

A/B 固定接近 Tide 节点数的 Transformer block 数，分别匹配总参数/初态实际有效触及量；C/D 同理固定层数/深度。
每个 B/L 单独做初态匹配，训练后不重新匹配。所有 block 数目标精确满足，参数相对误差最大 0.730%；构造器、目标、实际值及偏差均在 JSON。
有效触及量统计真实有效 owner/group/source 切片内的唯一 Parameter 标量：FC/Settle 按样本及输入步跨层去重，Timed 按样本及逻辑 tick 跨区域去重。
Timed 的均值涵盖输入和必要 flush 的逻辑时刻，重复 tick 的工作累计，另列每实际输入位置的工作；不推断输出因果归属。该口径不等同于非零梯度、物理读取量或 FLOPs。
参数数量独立按 `blocks×(4d²+3d×FFN+2d)+d` 复核，并枚举合法宽度检查最近点。
表内 A–D 是 **Tide/Transformer 时间比**，大于 1 表示 Tide 耗时更多。`†` 表示该 Transformer 观察超出波动阈值，保留数值但不作为稳定比较。

| 实例 / 精度 | B/L | Tide 秒 | 输入位置/s | Tide 峰值 GiB | A | B | C | D |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FC-K / FP32 | 8/128 | 0.131 | 7803.21 | 1.129 | 1.272 | 1.192 | 9.331 | 7.815 |
| FC-K / FP32 | 8/256 | 0.168 | 12214.85 | 2.106 | 1.223 | 1.237 | 8.147† | 9.280† |
| FC-K / FP32 | 8/512 | 0.250 | 16352.99 | 4.067 | 1.304 | 1.319 | 6.590 | 8.933 |
| FC-K / FP32 | 8/1024 | 0.445 | 18400.58 | 7.976 | 1.526 | 1.539 | 6.614 | 9.776 |
| FC-S / BF16 | 8/128 | 0.263 | 3888.42 | 1.367 | 2.243 | 2.227 | 16.647 | 13.669 |
| FC-S / BF16 | 8/256 | 0.435 | 4704.55 | 2.637 | 3.105 | 3.182 | 20.700 | 18.010 |
| FC-S / BF16 | 8/512 | 0.780 | 5250.95 | 5.161 | 3.945 | 3.813 | 23.650 | 29.318† |
| HB 1B / FP32 | 1/32 | 1.516 | 21.11 | 19.461 | 0.534 | 0.548 | 13.454 | 38.746 |
| HB 1B / FP32 | 1/64 | 1.566 | 40.88 | 21.338 | 0.551 | 0.508 | 13.418 | 38.544 |
| HB 1B / FP32 | 1/128 | 1.763 | 72.59 | 24.501 | 0.639 | 0.646 | 13.926 | 40.017 |
| HB Settle / FP32 | 8/128 | 1.326 | 772.00 | 7.799 | 2.176 | 2.148 | 38.163† | 37.864 |
| HB Settle / FP32 | 8/256 | 1.923 | 1064.73 | 15.322 | 2.522 | 2.390 | 47.292 | 49.808 |
| HB Settle / FP32 | 8/512 | 3.256 | 1257.82 | 30.515 | 3.033 | 2.979 | 48.054 | 52.305† |
| HB Timed / BF16 | 4/128 | 4.109 | 124.62 | 13.543 | 6.195 | 5.973 | 127.472 | 122.681† |
| HB Timed / BF16 | 4/256 | 6.673 | 153.46 | 27.386 | 7.806 | 7.798 | 150.829 | 145.187 |

HB 1B 的实际唯一参数为 976,768,720，节点数 848、区域层数 5。B1/L128 时，Tide 的 1.763 s 比 A/B 的 848-block 构造耗时少，
耗时是 C/D 的 5-block 构造的 13.93/40.02 倍；这两个比较维度不能互相替代。
同点初态有效触及均值 277,431,539，四次更新后为 254,190,352；活动均值没有替代参数持有量，也不表示同等 FLOPs。

FC-K B8 的 L128→1024 吞吐从 7,803 增至 18,401 输入位置/s，峰值从 1.129 增至 7.976 GiB。
HB Settle B8/L512 已测至 30.515 GiB；HB Timed B4/L256 为 27.386 GiB、153.46 输入位置/s。
Timed 在该点的 A/B 时间比约 7.8、C/D 约 145–151，仍有明显差距。单凭这些计时不能把差距分配给算子、调度或反向图；
JSON 保留初态/后更新的工作计数、query packing 和矩阵行数，具体热点与执行变体证据另列。

19 个 Tide 观察均满足有限稳定条件；其余六个超差观察为 FC-K FP32 B8/L256 的 C/D、FC-S BF16 B8/L512 的 D、
HB Settle FP32 B8/L128 的 C 和 B8/L512 的 D、HB Timed BF16 B4/L128 的 D。原始秒数与波动比保留在 JSON，未筛掉或重测。

## 固定构造器的预算内独立批量搜索

本表为 FC-K BF16 训练，L=128，B 候选为 32、64、128、256。A–D 构造器全部冻结于共同校准点 B32/L128，
初态活动目标为 4,194,736；较大 B 的活动变化单独记录。表中是每个模型在已测稳定候选内的最佳吞吐，
本次恰好均为 B256。该搜索没有在每个 B 重新匹配，不能替代上表的逐形状联合匹配。

| 模型 | 最佳已测 B | 秒 | 输入位置/s | 峰值 GiB | 停止 |
|---|---:|---:|---:|---:|---|
| tide | 256 | 0.737 | 44446.54 | 23.738 | candidate_limit |
| transformer_A | 256 | 0.384 | 85225.15 | 8.518 | candidate_limit |
| transformer_B | 256 | 0.420 | 78010.44 | 5.940 | candidate_limit |
| transformer_C | 256 | 0.182 | 180171.28 | 3.530 | candidate_limit |
| transformer_D | 256 | 0.106 | 308147.82 | 2.473 | candidate_limit |

Tide B32→B256 的吞吐从 24,438.54 增至 44,446.54 输入位置/s，峰值从 3.106 增至 23.738 GiB。
B128 已达 44,156.77 输入位置/s，B256 只再增 0.656%；仅一个平坦增量不足以触发两点停止规则，下一候选不在该冻结搜索矩阵中。
B256 的初态 Tide 有效触及均值 4,203,444，比校准点多 8,708（约 0.208%）；四次更新后为 4,339,062。
预算尚有余量不等于更大 B 一定适用，当前证据没有确定最大可行批量。

## 身份、资源与复现

这些作业在 UTC 07:23–07:31 期间重叠运行，各自使用一张协作队列独占准入的 NPU，CPU affinity 互不重叠；
记录身份为 `exclusive_cooperative_queue; external jobs not controlled`。这是队列内独占身份，不代表整机无其他任务；
原始记录保留每个候选前后 host load，短样本仍受主机争用、串行顺序和时间漂移影响。

运行软件为 PyTorch `2.12.0+cpu` + `torch_npu 2.12.0.post3`，模块 `cann/9.2.0:ascend/dev-workspace-9.2.0`；
`+cpu` 是 Torch 包版本标签，实际矩阵输出/设备由运行审计确认是 NPU。私有解释器、模块来源、服务终态、队列映射及精确命令均保留。

| 搜索身份 | 物理 NPU | CPU affinity | 候选范围 |
|---|---:|---|---|
| capacity-fck-budget-bf16-v2 | 14 | 120–123 | batch 32,64,128,256 |
| capacity-fck-length-fp32-v2 | 1 | 100–103 | length 128,256,512,1024 |
| capacity-fcs-length-bf16-v2 | 7 | 104–107 | length 128,256,512 |
| capacity-hb1b-length-fp32-v2 | 4 | 116–119 | length 32,64,128 |
| capacity-settle-length-fp32-v2 | 5 | 108–111 | length 128,256,512 |
| capacity-timed-length-bf16-v2 | 0 | 112–115 | length 128,256 |

只读复现汇总，无需设备或 Torch 导入：

```sh
python -m benchmarks.search_report \
  --inputs artifacts/capacity-fck-budget-bf16-v2/results/search.json \
  artifacts/capacity-fck-length-fp32-v2/results/search.json \
  artifacts/capacity-fcs-length-bf16-v2/results/search.json \
  artifacts/capacity-hb1b-length-fp32-v2/results/search.json \
  artifacts/capacity-settle-length-fp32-v2/results/search.json \
  artifacts/capacity-timed-length-bf16-v2/results/search.json \
  --output artifacts/NEW_CAPACITY_REPORT.md
```

该入口核验配置、审计、校准和计时文件的 SHA256/字节数后生成基础双表，报告稳定候选数量；逐项波动状态以原始 JSON 的 `stable`/`both_stable` 为准。
本页另外独立复核 95 个观察的中位数、吞吐、波动、峰值及全部 76 个 Transformer 构造的目标/公式/最近合法点，
并核对容量表的固定构造与最佳候选。所有 120 个引用配置/结果文件身份一致；六份搜索文件身份如下。

- [capacity-fck-budget-bf16-v2](../../artifacts/capacity-fck-budget-bf16-v2/results/search.json)：`bc15c8a00304f2fbd37769048729130db91143fa8d8befeae8c9274f45e7aa82`。
- [capacity-fck-length-fp32-v2](../../artifacts/capacity-fck-length-fp32-v2/results/search.json)：`6190371bbd7d0b2522142eb483f1d7f2282ba867e85f0efe20088e6afc865507`。
- [capacity-fcs-length-bf16-v2](../../artifacts/capacity-fcs-length-bf16-v2/results/search.json)：`0628e30fdfb4067aece9aca588d361499efee0d5a6b5902ff7b485f39ea07003`。
- [capacity-hb1b-length-fp32-v2](../../artifacts/capacity-hb1b-length-fp32-v2/results/search.json)：`f67d0150b71abd66f06740c4c030f0c243ad909fcb2e0f71a1a89cd618e67330`。
- [capacity-settle-length-fp32-v2](../../artifacts/capacity-settle-length-fp32-v2/results/search.json)：`83112b7c7f64fcc0993c5113280edb9947a166a92e8e115aa14fab8cd98887c3`。
- [capacity-timed-length-bf16-v2](../../artifacts/capacity-timed-length-bf16-v2/results/search.json)：`7273f3fe4a4556ed81c14960f75beb52036208bf73f3c86c93d300e7a253c777`。

若重新执行实验，使用 JSON 中的冻结源码与 `runs[].launch_arguments`；精确子进程命令留在对应搜索文件的 `children`。重新核对当前资源并选择新输出身份；
不能把已有路径当新运行目的地。原始 v2 汇总的 packing 字段存在旧键名遗漏，本 JSON 从未改动的 raw `query_packing` 取回；
原始文件及其空字段保留原样。FP32/BF16 的具体数值超差、持续状态训练、恢复后更新和后续优化资格仍以各自证据为准。
