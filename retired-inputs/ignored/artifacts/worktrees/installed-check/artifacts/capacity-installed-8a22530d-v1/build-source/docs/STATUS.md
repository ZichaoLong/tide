# 当前交接：容量、吞吐与区域执行阶段实施中

用户已确认执行 [ROADMAP](ROADMAP.md) 的六项计划，阶段汇报后继续。
授权分支 `graph-execution-foundation`；固定数学仍为
`4f5b9f20a4030d68ee66ae2d26e34659a5e37234`，108 文件核验通过。
原四份用户未跟踪文档保留。下方上一阶段证据保持各自源身份和原数值超差。

## 本阶段已集成

- 可选固定通过 Hub、显式本地时钟、terminal Hubs；支持目录独立核对
  41 空间/55 时间配置、475 几何设置、225,400 参数配置及 676,200 分配行。
  枚举属于元数据覆盖，不是全部组合的执行或性能资格。见 [可选绑定](optional-bindings.md)。
- 相容区域批量、可选区域重算、小 Attention 后端选择、最后消费者释放和紧凑 Timed 尾部；
  BF16 合批偏置舍入错误已在 `231f56eb` 修复。NPU 独立 epilogue 见证确认实际 BF16 输出。
- 四类联合比较的共同 B/L 与独立批量吞吐搜索、隔离存储账本、新进程恢复计时；
  见 [容量设施](capacity-search.md)。`181a522d` 修正自定义时钟及紧凑尾部容量记账。
- 完整区域多设备归属与 checkpoint；默认串行调度，按设备并发仍在独立工作树开发。
  独立 [LibTorch 区域切片](native-regional.md) 已加入；不是完整 Tide 的第二套实现。

## 当前固定检查与活动作业

`062a8ab6` 冻结 CPU 套件 521 项通过（215.48s）。支持目录及区域容量预算另以
`7b77457d` 冻结；当前最新实现增量为 `36c42a7a` 的有限执行变体驱动。
实现提交推送进度以实际 Git 为准；证据收束后单独提交。

- `execution-candidates-fp64-v3` 八个 Settle/Timed 诊断均在原阈值内。
  NPU 重算前向精确；batch4/local_math/combined 仍有有限 FP32/BF16 超差，
  BF16 Timed 在后续更新出现支持/KV 时间差异，正在做首处分歧的有限调查。
  原 `execution-candidates-npu-v3` 因实现错误停下，保留 superseded 与全部失败。
- `capacity-continuous-hb05b-v3` 与 `capacity-continuous-hb1b-v3` 均完成：
  两精度、三模式、B1/prefill128/chunk64/decode8、三段/更新及真实恢复更新。
  四个 `capacity-restore-*-v1` 新进程加载和更新均有限完成；与原 owner 的数值检查另记。
- `region-placement-npu-serial-v1` 双 NPU 两精度 Exchange/Timed、梯度、多步与恢复完成，
  有限超差保留。9B 四卡区域训练 `capacity-region-9b-npu-v1` 在执行中；17B 仍只有预算。
- native CPU 固定29案例和5公式通过；NPU同stream28/29通过，BF16四worker恢复后的
  `v_weight` 梯度最大差 `7.62939453125e-6`，原阈值失败保留。
  CPU/NPU 区域切片正常性能实验已提交，不能由检查计时推出性能结论。
- 四个 `execution-perf-{settle,timed}-{fp32,bf16}-v1` 在执行或排队：
  同卡同 B/L 五候选正常计时，随后 baseline/recompute/combined 独立存储账本。
  所有当前设备作业仍使用独占协作队列；本线程共享授权尚未使用。

运行记录在 `artifacts/任务名/{launch,status,queue}.json` 及各子结果中。
大任务前再次检查账户配额、活动任务、主机及设备；最近配额约75 GiB / 300 GiB软限，
主机约1.8 TiB可用。以上是当次观察，不是后续永久准入。
参考、公共工具链与下游未修改；当前无通用图基座任务。

## 已完成的五类完善工作（固定证据）

分支 `graph-execution-foundation`。数学固定为
`4f5b9f20a4030d68ee66ae2d26e34659a5e37234`，108 个文件校验通过。
当前 FC/HB 主实例路线持续有效；产品边界在 [project-scope](project-scope.md)。
[ROADMAP](ROADMAP.md) 已改为本轮交付与可选择的后续待办；通用基座未启动。

本轮五类工作都有实际结果或经过有限调查的明确边界。全部本轮作业已有终态，
没有遗留本轮计算任务。执行完成、数值容差通过和性能收益分别记录；原有数值超差、
新发现的超差和执行失败均保留，阈值未放宽。

## 实现与身份

- FP32/BF16 明确计算策略、模式一致的 BF16 Attention、NPU KV 压紧和兼容
  head-query 融合；CPU 保留实测更合适的原路径。见[精度](precision.md)和[架构](architecture.md)。
- A–D 四类联合匹配、独立核验的实际有效触及统计、真正增量 KV 基线、完整训练计时、
  Timed flush、续接和恢复后更新；入口见[比较设施](benchmarking.md)。
- 可选 HB 分层放置、统一 AdamW 参数归属和 checkpoint；默认跨 NPU 使用 CPU
  host staging。见[放置契约](placement.md)。
- 模型与打包源码自 `0992b467` 未变，`src/tide` tree 为
  `1ef1183be8b131a9bbfd806ce4d9240ef25ea502`。不同实验仍保留各自冻结提交，
  不把较早矩阵追认为最新设施实测。`ca6199d4` 另修复 launcher 覆盖 CANN/TBE
  Python 路径的问题，实际环境与 launcher SHA 随新作业记录，公共环境未改动。

## 功能、数值和连续使用

[CPU 套件及隔离安装](evidence/roadmap-installed-20261118.md)：`0992b467` 的
424 个 CPU 测试通过；固定 wheel 的 17 个包文件及 15 个已加载 Tide 模块核验通过。
隔离安装从 `/`、无 `PYTHONPATH`、Python `-I` 运行 FP32/BF16 推理、三步训练、
checkpoint 和真实恢复后更新。原 FP32 安装样例的两个 regional-bias 更新超差仍保留
（最大 `4.99934e-5`）。[清理后的独立安装复核](evidence/roadmap-retirement-20261118.md)
再次完成机制与隔离检查，FP32/BF16 各记录一个参数更新超差（最大 `1.38134e-5` /
`1.66446e-5`），BF16 另有 moment 差异。相同输入身份和环境没有保证位级更新轨迹；
这些新观察与原安装结果分别保留。

[初始数值调查](evidence/roadmap-numerics-initial-20261118.md)和
[后续调查](evidence/roadmap-numerics-followup-20261118.md)记录以下边界：

- CPU/NPU 七个 BF16 用例的推理与含梯度前向精确一致。四个 HB 跨精度比较仍出现
  支持/KV 时间差异；独立 Decimal 舍入输入见证复现了实际边界跨越。
- [原 FP32 四项 NPU 超差](evidence/instances-fp32-20261118.md)未重标；新的 FP32/BF16
  梯度和恢复更新有限超差也保留。同 owner 的 FC-S 重复反向可变，独立标量 FP64
  AdamW 解释了选定的实测更新差异；这不构成全部超差的统一解释。
- 小型 whole/chunk 完整反向见证通过 CPU/NPU FP32 和 CPU FP64 原阈值。
  BF16 FC-K/Exchange/Timed 仍有有限值/状态差异，包括强制 tiled；这些见证未发现
  支持、逻辑元数据或梯度连接差异。FP64 只用于有限诊断。
- BF16 参数、输入、梯度、master 和 AdamW 为 FP32；真实 BF16 投影、BF16 KV
  payload、FP32 敏感控制/SSD 状态和 int64 时间分别声明。math/tiled Attention
  将 Q/K/V 乘积升为 FP32，CPU SDPA 审计也观察到 FP32 bmm；不以 payload 名称推断 kernel。

[连续矩阵](evidence/roadmap-continuous-20261118.md)共 480 个子任务、1,440 段连续运行，
覆盖 CPU/NPU、B1/B4、两精度、三模式、Tide+A–D、prefill128/chunk64/decode8，
无该矩阵执行失败或超时。CPU d64 与 NPU d128 不是同形状设备速度比较。
32 个 checkpoint 案例中，CPU16 个所有检查精确一致；NPU11 个在原阈值内，5 个有有限超差。
保存数据和下一次前向/状态精确不等于后续 optimizer 轨迹均精确。
训练包含必要准备、反向、AdamW、detach 和同步；恢复后的第四次更新真实执行但未单独计时。
首训练段含首次反向/slot 建立，后续段状态与参数不同，不能当作独立同分布样本。

## 比较、热点与规模

| 固定证据 | 已完成范围 | 主要结论与边界 |
|---|---|---|
| [代表主矩阵](evidence/roadmap-representative-20261118.md) | CPU/NPU 七配置、两精度、三模式；432 行 / 2,160 样本 | Tide 与 A–D 同时匹配 block/深度及总/实际活跃参数；FC-S 另有 SSD。NPU 训练 Tide/A–B 时间比 1.17–3.80，Tide/C–D 4.12–78.88；CPU 分别 0.66–15.08、1.94–104.05。 |
| [稀疏、形状、组合](evidence/roadmap-supplementary-20261118.md) | 360 行 / 1,080 样本 | 合法人工稀疏压力与常规初始化分列，B/L/节点和有用组合各有实际工作统计。 |
| [0.5B/1B](evidence/roadmap-scale-small-20261118.md) | 实际 FC/HB；120 行 / 360 样本 | NPU 训练 Tide/A–B 0.452–0.695，Tide/C–D 4.673–68.358。 |
| [热点](evidence/roadmap-hotspots-20261118.md)与[Attention](evidence/roadmap-attention-20261118.md) | 有限候选；CPU/NPU 各60 Attention 行 / 300 正常样本 | NPU 压紧观测降时3.62–4.10%，query融合额外3.35–6.40%；部分实际工作变化，不能称严格同轨迹加速。Full 分桶回退、额外 gate/up 未采用；CPU Attention12/60 行明显波动，不给稳健后端排名。 |
| [9B/17B](evidence/roadmap-scale-large-20261118.md) | 5 个完成作业；20 比较行、5 容量行 | 9B 单 NPU 和17B CPU 两精度 Tide+A–D 推理；17B 双卡推理、9B 四卡两精度各两步训练、17B CPU BF16 两步训练。 |

时间比小于1表示 Tide 用时较短。A/B 与 C/D 的宽度、深度和数学工作不同，比例不说明
模型质量。主矩阵 NPU BF16 训练慢0.42–11.89%，峰值 allocated 降7.54–19.35%；
CPU BF16 用时变化 −21.92% 至 +11.50%。精度切换可能改变路由及活跃匹配，收益限于实测配置。
报告分开数学必要工作、padding/重复工作、调度/计算图成本及未解释差距。

大容量实际参数为9B档 `8,778,358,224`、17B档 `16,707,220,176`，均 B1/L8。
17B 双卡推理中位数 FP32/BF16 为1.284/1.478秒；9B 四卡两步完整训练为
FP32 4.875/2.926秒、BF16 5.231/2.987秒，最高单卡峰值52.178/49.638 GiB。
17B CPU BF16 两步143.393/125.446秒，采样进程树 RSS 峰值328.451 GiB；
参数/梯度/moments/状态/输出有限且状态0→8→16。两步工作和 slot 建立不同，
不据此推导稳定吞吐或优化收益。大模型独立梯度前向、checkpoint/恢复、长窗、
A–D 训练未测；17B CPU FP32 训练未测。17B 最宽层的 AdamW 持久预算约83.795 GiB，
当前按层放置不能在60 GiB单卡训练该层，17B NPU训练未运行。

## 后端、资源和交接

[两卡小型资格](evidence/roadmap-placement-20261118.md)覆盖 FP32/BF16 Exchange/Timed
推理、梯度、三步续接与实际恢复更新，有限超差照录。修复启动环境后 auto/BF16 host_stage
完成；synchronized/FP32 在 checkpoint 后原模型反向、direct/FP32 在首次放置更新反向
仍报无效事件句柄，外层作业保留 failed。未声明 peer DMA/HCCL、复制训练或完整轨迹等价。
[TorchAir 探测](evidence/roadmap-resident-20261118.md)在静态 role guard 处停于前端，
backend 回调0次；没有 resident 资格，也不是不可实现的证明。本机无 CUDA GPU，
CUDA 真机未验；当前证据未要求额外 LibTorch 构建。

所有 NPU 作业使用本机协作队列独占准入；共享 NPU 授权本轮未使用，且不向新线程继承。
主机为共享资源，各作业记录 CPU affinity、线程校准、NUMA、争用与实际内存边界。
RSS 界限是采样进程树界限，非 cgroup 硬限制。计算前被替换的作业保留原 stop 状态
及 superseded 原因；不计作硬件失败或通过。公开工具链、参考与下游安装未修改。

[产物收束](evidence/roadmap-retirement-20261118.md)保留小结果、完整命令/输入身份、
生命周期/失败记录、固定 wheel 和六个未决数值见证；释放可由 Git/命令重建的归档、
大张量和一次性环境。四份原用户未跟踪文档保持原 hash，主私有环境和两个工作树保留。
交付与下游升级/验收按[协作契约](downstream-contract.md)分别记录。

复现从[开发/隔离安装](development.md)、[比较设施](benchmarking.md)、
[数值验证](verification.md)和[证据索引](evidence/README.md)进入。
每篇 JSON 保留源码、配置、命令、样本、实际精度/算子/设备、计时和内存口径；
按记录提交重建输入、使用新输出路径和当次资源准入，长任务遵守[执行契约](execution-flows.md)。
历史绝对路径是身份，不保证被清理的归档/张量仍在线。后续无需重跑本轮即可选择新工作。
