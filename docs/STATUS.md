# 当前交接：容量、吞吐与区域执行已交付

已完成用户确认的六项阶段工作，承接此前五类完善工作。当前分支
`graph-execution-foundation`；持续 FC/HB 实例路线，通用基座未启动。
数学固定为 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`，108 文件核验通过。
产品边界见 [project-scope](project-scope.md)，后续可选工作见 [ROADMAP](ROADMAP.md)。
本阶段作业全部终态，没有遗留计算任务；失败、超差、停止原因和原阈值均保留。

## 实现与验证身份

- [配置支持](optional-bindings.md)：主菜单 41 空间配置/55 时序绑定、475 几何设置、
  225,400 参数配置及 676,200 分配行通过元数据核对。另实现固定通过 Group Hub、
  显式 ClockConfig 和可选 terminal Hubs。枚举不是所有组合的执行、数值或性能资格。
- [执行选项](instances.md)：同层相容区域合批、选择性区域重算、小 Attention 后端选择、
  最后消费者释放和紧凑 Timed 尾部，保留完整 BPTT。默认 `region_batch_size=1`、不开启重算；候选显式选择。
  BF16 合批的 FP32 偏置提升错误已在 `231f56eb` 修复。
- [完整区域放置](placement.md)：单一参数归属、运输去重、统一 AdamW/checkpoint；
  默认串行，可选 `region_concurrency="per_device"`，每层每设备一个任务、层间 join。
  区域放置目前要求 `region_batch_size=1`。
- 完整模型仍为 PyTorch；独立 [C++17/LibTorch 区域评分切片](native-regional.md)
  有串行/2/4 worker 对照、梯度和恢复检查，不是第二套完整 Tide 或自动择优后端。
- [集成验证](evidence/capacity-validation-20261118.md)：冻结 `8a22530d` 的完整 CPU 套件
  **544 项通过**；`5ae060db` 集成后的安装分类/合批定向检查 **17 项通过**。
  两者之间 `src/tide` 仅改一处说明注释，21 个 Python 文件 AST 一致；不是字节相同。
  新 wheel 的资格仍固定在 `8a22530d`。每组硬件/性能结果保留各自实际冻结提交。

## 内存、吞吐和四类比较

[有界 B/L 搜索](evidence/capacity-search-20261118.md)完成 19 个审计、95 个训练计时子进程，
285 个样本；76 个 Transformer 构造独立核对，block 目标精确、最大参数匹配误差 0.7304%。
所有方向先到 `candidate_limit`，没有测出硬件最大 B/L；原先小 B/L 不能解释为都已占满显存。
FC-K FP32 B8 的 L128→1024 吞吐为 7,803→18,401 输入位置/s，峰值 1.129→7.976 GiB。
FC-K BF16 L128 的 B32→256 为 24,439→44,447/s，峰值 3.106→23.738 GiB；
B128 已有 44,157/s，继续翻倍仅增加约 0.66%。

以下是各共同 B/L 点的完整训练 **Tide/Transformer 用时比**，小于 1 表示 Tide 较快。
A/B 同时匹配近 block 数与总/实际活跃参数；C/D 同时匹配近深度与总/实际活跃参数。
不是同宽度单轴比较，也不说明模型质量；固定校准模型的独立容量搜索另表列示。

| 实例 / 精度 | B/L | A | B | C | D |
|---|---|---:|---:|---:|---:|
| FC-K / FP32 | 8/1024 | 1.526 | 1.539 | 6.614 | 9.776 |
| FC-S / BF16 | 8/512 | 3.945 | 3.813 | 23.650 | 29.318† |
| HB 1B / FP32 | 1/128 | 0.639 | 0.646 | 13.926 | 40.017 |
| HB Settle / FP32 | 8/512 | 3.033 | 2.979 | 48.054 | 52.305† |
| HB Timed / BF16 | 4/256 | 7.806 | 7.798 | 150.829 | 145.187 |

† 原 Transformer 样本波动超过该搜索 25% 标记线，原样保留。全部图、稀疏压力、
有用组合和较小尺寸见[代表矩阵](evidence/roadmap-representative-20261118.md)、
[补充矩阵](evidence/roadmap-supplementary-20261118.md)及新搜索表；不外推到未测组合。

[执行变体与存储账本](evidence/capacity-execution-20261118.md)覆盖 Settle/Timed、两精度，
20 行正常计时/100 样本，另有 12 个侵入式账本：

| 候选 / 基线 | 同 B/L 时间比 | 峰值 allocated 比 | 结论 |
|---|---:|---:|---|
| batch4 | 0.695–0.792 | 1.009–1.038 | 有较短用时，内存略增 |
| 重算 | 1.738–1.813 | 0.148–0.202 | 容量选项，增加计算成本 |
| batch4 + 重算 | 1.235–1.354 | 0.213–0.271 | 折中选项 |

32 GiB 界限下，combined 已测 B64 为 1,379/s、21.486 GiB；基线最佳已测 B16 为
1,473/s、26.763 GiB，比值 **0.9364**。节省内存扩大了已测可行 B，未提高最佳已测吞吐。
基线 B32 未测，两方向均有限候选，卡/B 不同；不是全局最优或四类 Transformer 容量结论。
这些执行候选的活动轨迹可能变化，不能称严格同轨迹加速。账本分类与阶段峰值不能相加。

已有 Attention 使用 native fusion/SDPA 和保留零先验导数的在线分块；本阶段补重算与张量
生命周期优化。没有把所有分配都称为 Flash Attention，也没有证明所有形状均应分块。
[三类 Attention 证据](evidence/roadmap-attention-20261118.md)及
[性能经验](performance-lessons.md)分别记录算子与适用条件。

## 连续运行、恢复与区域多卡

[0.5B/1B 连续证据](evidence/capacity-continuous-20261118.md)覆盖两精度、三模式、
12 子任务/36 连续段：B1、prefill128/chunk64/decode8，时钟 136→272→408。
四份 checkpoint 的成对真实下一更新均在原阈值内；另四个新进程完成加载和真实更新。
原成对检查用 64 输入至 472，新计时用 136 输入至 544，二者不是同一续接轨迹。
新进程第一次更新分别为 0.5B FP32/BF16 27.654/32.043s、1B 14.258/15.177s；
每点仅一次，无预热，不代表稳态。此前小模型的 Timed flush 与三模式矩阵继续保留各自资格。

[区域多卡证据](evidence/region-capacity-20261118.md)：串行/并发共 8 行小型 NPU 资格，
值、梯度、三更新、checkpoint 和真实恢复更新完成；原有限超差保留。
12 行性能共 72 更新，同初参/输入/逻辑分片的 per_device/双卡串行观察时间比
0.816–0.865。后续工作计数不同，Timed FP32 单卡另有计时内 host 间隙，未扣除；
不能把全部时间差归为计算加速。逐卡峰值随放置明显降低，通信和全设备同步均计入。

| 实际容量 | 精度 | 两次完整更新 s | 全部更新最高单卡 allocated GiB |
|---|---|---|---:|
| 9B，四 NPU | FP32 / BF16 | 3.860/2.058；3.946/2.223 | 38.489 / 36.628 |
| 17B，六 NPU | FP32 / BF16 | 5.762/2.508；6.024/2.447 | 48.594 / 46.475 |

实际参数为 8,778,358,224 / 16,707,220,176；均 B1/L8、区域串行、非 foreach AdamW。
全量 finite 扫描通过，optimizer step 1→2、状态 0→8→16。两步容量执行不等于长期稳定
训练资格；大模型长序列、checkpoint/恢复、独立梯度前向和 A–D 训练仍未测。

## 数值、后端与剩余限制

[舍入调查](evidence/capacity-rounding-20261118.md)的八个 CPU FP64 执行候选在原阈值内。
CPU Linear/baddbmm 的两次/一次舍入、NPU addmm/baddbmm 与 mm/bmm 的临界值见证，
以及独立 Decimal 分配解释了选定首处分歧；不抹去 BF16 支持/KV 时刻及后续训练超差。
已有 FP32 梯度/更新超差仍保持原记录，CPU FP64 不作正式产品精度或性能对标。

[隔离 wheel](evidence/capacity-installed-20261118.md)的 21 个文件/模块核验通过，四例
保存数据和下一前向精确。batched 两精度恢复第四次更新精确，placed FP32 在原阈值内，
placed BF16 的梯度/参数/slot 有限超差保留。同 owner 重复并发反向已经可变；另一恢复
owner 的更新精确复现原进程，四个标量 FP64 AdamW 见证解释所选更新差。没有宣称
所有差异源于 checkpoint 或已全部解释。原 harness 的有限值误标结构失败已修复；原 exit2 不改写。

[LibTorch 切片](evidence/native-regional-performance-20261118.md)完成 CPU/NPU、三模式、
串行/2/4 worker 和 CPU inner4 对照：小 NPU 形状有收益，大形状并发未普遍改善。
120 行/600 样本中 27 行波动较大；84 个末态比较 80 项通过，四项 BF16 超差保留，
原 NPU 四 worker 恢复梯度失败也保留。pool stream 八例通过不撤销同 stream 失败。
它不提供整个模型自动切换、设备 allocator 峰值或逐算子放置 trace。

CPU/NPU 已有实际执行。跨 NPU 默认 host staging；direct 后向事件失败仍在原报告。
CUDA 真机、peer DMA/HCCL、resident/编译训练无资格；TorchAir 原探测停于前端 guard。
区域合批与多设备区域归属尚未组合；完整区域/共享参数组仍不能拆开。所有性能结论
只覆盖声明配置、模式和资源，未开展模型质量实验。

## 资源与接续

本阶段 NPU 均为协作队列独占准入；共享主机及队列外任务仍可能争用，原记录保留资源身份。
本线程共享 NPU 授权未使用，不向新线程继承。长任务使用冻结输入、有界 background.slice
服务和采样进程树 RSS；这些界限不是 cgroup 硬内存限制。
四份原用户未跟踪文档 hash 未变；参考、公共环境与下游未修改。工作树、checkpoint、
wheel 及未决见证保留；本次收口未清理产物。交付与下游升级/验收仍按[协作契约](downstream-contract.md)区分。

复现从[开发与安装](development.md)、[容量搜索](capacity-search.md)、[区域放置](placement.md)、
[数值验证](verification.md)和[证据索引](evidence/README.md)进入。按各报告的冻结提交与命令
使用新输出路径及当次资源准入；无需为接续新任务重跑整个已交付矩阵。
