# 完整区域放置、同层并发与容量，2026-11-18

本篇记录当前 FC/HB 实例路线中的完整区域归属与有限同层并发，
不改变固定数学 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`。
五个作业均已终态：8 行小型资格完成，9B 四卡和 17B 六卡两精度各两次更新完成，
12 行完整训练性能观察完成。有限数值超差及一组计时内 host 间隙异常保留原记录。
[机器记录](region-capacity-20261118.json.gz)保留各作业独立的源码、参数/输入 hash、
命令、准入、原始样本、数值失败和结果文件 SHA256；实现契约见[放置](../placement.md)。
旧按层放置结果保留原身份，本篇不会把它们改标为区域或并发实测。

## 固定身份和资源

| 作业（`artifacts/`） | 源码 | 物理 NPU | CPU affinity / threads | 范围 |
|---|---|---|---|---|
| region-placement-npu-serial-v1 | `062a8ab6` | 5,7 | 124–127 / 4 | 两精度 Exchange/Timed，串行资格 |
| region-placement-npu-concurrent-v1 | `8a22530d` | 0,4 | 124–127 / 4 | 两精度 Exchange/Timed，per_device 资格 |
| capacity-region-9b-npu-v1 | `36c42a7a` | 0,1,5,7 | 128–135 / 8 | 9B 四卡，两精度各两步 |
| capacity-region-17b-npu-v1 | `8a22530d` | 0,1,2,4,5,7 | 128–135 / 8 | 17B 六卡，两精度各两步 |
| region-placement-perf-npu-v1 | `8a22530d` | 0,5 | 124–127 / 4 | 两精度、Settle/Timed、三种设备映射/调度 |

这些作业使用整组协作独占队列；队列外活动及共享主机争用未被排除。
源码快照逐字节核对 `src/tide`、执行入口和四份相关配置，共 26 或 27 个文件；
每份数学快照的 108 个文件按固定清单核验。实际 Tide 导入来自冻结目录，
TBE 来自所选 `ascend/dev-workspace-9.2.0` 的 CANN 路径；公开环境没有修改。
具体执行器、环境变量、launcher 和记录身份见 JSON。资格作业的计时不用于性能排名。

## 归属与并发的验证边界

每个数学区域保持完整，共享参数不得跨 owner 分片。输入投影由一个 owner 持有；
按目标分片收集并去重的前驱节点银行在调用线程运输，值和强度一起传递。
Timed 私有状态保留 int64 时刻和可微的紧凑最后一 tick；末端不参与输出的 H 区域
不会因拼接再切片而产生虚构零梯度。`placement_adamw` 保留原 Parameter 的
None/零梯度、weight decay、moment 和 step 语义。

默认仍为 `serial`。`per_device` 每层每真实设备一个任务，同设备区域串行、层间 join；
池限于本层最大参与设备数，生命周期为一次 forward/flush。工作线程继承 grad、
inference、autocast、设备和调用者当前 stream，不新建跨设备流或事件协议。
反向由原 Torch autograd engine 执行。公开 TorchNPU 初始化/stream 接口的静态审查
支持实施该有限候选；本篇的实际 NPU 运行是另行取得的硬件证据。

`4f7ea32a` 的 64 项定向 CPU 检查通过，包含实际双线程 barrier 重叠、设备/stream
上下文模拟、推理/无梯度/含梯度及 BF16 TLS、FP64 值/VJP/三次更新、断连梯度、
Timed 尾部、失败后线程归集、检查点重映射和 FP32/BF16 恢复后真实更新。
CPU 索引只是逻辑并发模拟。FP64 是有限诊断，CPU 机制检查没有替代 NPU 数值资格。

内置 `record_work` 在请求的调度路径计算并在调用线程归集。RuntimeAudit 的 operator
见证明确使用 `serial_region_audit`，记录 requested `per_device`、effective `serial`；
它不是并发调度剖析。未知 caller dispatch/function mode 不隐式传播；saved-tensor hooks
和 profiler 必须显式串行。WorkAudit 的共享 hooks 和 dense inventory 不支持 placed shards。

## 小型 NPU 数值、续接与恢复

两个资格作业各覆盖 FP32/BF16 × Exchange/Timed，B2/L5、d64、seed 807、三次继续更新，
普通单卡参考与 placed 模型使用同初参和输入。Exchange-X 为 `incoming_light_64`、
`independent`，实际 1,610,791 参数；Timed Multiscale-X 为 `compact-5-dd4`、
`source_residual`、`region_shared`，1,150,527 参数。两种区域调度使用相同逻辑区域分区。
跨 NPU 运输均为 `host_stage`，没有额外阶段同步。

八行都执行完成，所有 retained comparisons 有限，未发现 metadata 或 support mismatch。
初始 whole result/必要 flush 精确；相同 `2+3` 切分下，普通单卡与 placed 的读出和状态精确。
这些观察不等于此后所有更新轨迹精确。

| 调度 | 精度 / 配置 | 初始参数梯度 max abs | 三更新参数 max abs | 恢复更新参数 max abs | 原结果 |
|---|---|---:|---:|---:|---|
| serial | FP32 Exchange | 4.1910e-8，内 | 9.9778e-5，超 | 2.9104e-11，内 | finite_deviation |
| serial | FP32 Timed | 2.0489e-8，内 | 3.5092e-6，内 | 3.1292e-6，内 | within_threshold |
| serial | BF16 Exchange | 9.7656e-4，超 | 2.5936e-4，超 | 0 | finite_deviation |
| serial | BF16 Timed | 3.7253e-9，内 | 1.3150e-4，超 | 7.2760e-12，内 | finite_deviation |
| per_device | FP32 Exchange | 4.1910e-8，内 | 9.6515e-5，超 | 2.9104e-11，内 | finite_deviation |
| per_device | FP32 Timed | 1.7928e-8，内 | 6.8024e-6，内 | 5.2899e-7，内 | within_threshold |
| per_device | BF16 Exchange | 9.7656e-4，超 | 2.5936e-4，超 | 0 | finite_deviation |
| per_device | BF16 Timed | 5.5879e-9，内 | 2.8670e-4，超 | 1.3858e-6，内 | finite_deviation |

“内/超”按原 `atol=1e-6, rtol=2e-3` 判定；输入梯度使用 `rtol=1e-2`。
最大绝对差不单独决定相对容差结果。BF16 Exchange 输入梯度 max `1.8310546875e-4`
也超差。两种调度的 BF16 whole/split 读出差都为 Exchange `0.03765618801`、
Timed `0.03686928749`，在普通参考中同样出现；状态差分别为 `0.02392578125` /
`0.05607414246`，Timed flush max `0.05133605003`。这些失败没有因并发完成而消失，
也不统一归因为一个偏置或跨设备运输。其完整失败路径与阈值在 JSON。

八行的检查点参数、private state、optimizer 数据重新加载后精确，下一次前向也精确。
原 owner 和同进程新建 restored owner 随后都真实执行一次更新；更新后状态精确，
参数与 optimizer slots 均在原阈值内，部分不位级精确。这不是新进程启动/加载延迟测量。

## 9B / 17B 实际容量

容量入口是一份 placed 模型，避免为大模型同时构造普通单卡参考和恢复副本。
9B/17B 分别是 8,778,358,224 / 16,707,220,176 参数，d768、HB Multiscale-X Settle、
`source_residual/quad/independent`、B1/L8、`--no-foreach`。FP32 参数、梯度和 AdamW
双矩仍是持久主体，BF16 算术不会将它们减半；没有另分配重复 master 权重。
两种规模都使用区域 `serial`，小型 per_device 资格不外推为这些大模型的并发容量。

| 规模 / 卡数 / 精度 | 两次完整更新 s | 两次各自最高单卡 allocated GiB | 期间最高单卡 reserved GiB | 状态 |
|---|---|---|---:|---|
| 9B / 4 / FP32 | 3.85956 / 2.05834 | 36.183 / 38.489 | 42.480 | completed |
| 9B / 4 / BF16 | 3.94632 / 2.22286 | 36.164 / 36.628 | 39.422 | completed |
| 17B / 6 / FP32 | 5.76177 / 2.50813 | 46.475 / 48.594 | 54.379 | completed |
| 17B / 6 / BF16 | 6.02444 / 2.44676 | 46.463 / 46.475 | 51.277 | completed |

9B 和 17B 参数/梯度/slots/buffers/private state/输出/损失的逐步全量 finite 扫描全部通过，
状态位置均为 `0→8→16`，各自 598 / 698 个 optimizer step counter 分别为 1 和 2；每步记录了真实参数
更新的有限采样见证。FP32/BF16 工作量不同，第二步和第一步也不同。两步只证明该
负载的容量与有限更新完成，不构成稳态吞吐、跨精度质量或普通单卡轨迹等价证明。

上述 allocated/reserved 为每步重置后的 allocator 峰值；字段虽简写为 allocated_bytes，
实现读取 `max_memory_allocated`。峰值取全部更新和全部卡，不能只引用第一步约 36.2GiB。
reserved 可包含审计缓存。9B 监督器采样进程树峰值 34.661GiB，界限 160GiB。
CPU 构造各约 83s、初始参数 hash 各约 20.8s、放置约 3.6s 都在训练时间之外。
17B 监督器采样进程树峰值 64.305GiB，界限 256GiB；CPU 构造各约 158.5s、
参数 hash 各约 38.3s、放置约 7.75s，同样独立记录。六卡的实际完成解除这份配置
在整层归属下的最宽层容量障碍，但没有切开完整区域或复制共享权重。

## 相同初始负载的三种放置性能观察

该组冻结 `8a22530d`，12 个独立子进程在同一双卡准入服务内顺序执行。
使用现有 d128/n8/h4/compact-9/region_shared 配置：Settle B8/L256，Timed B4/L128，
FP32/BF16 各自对照。三种映射都保留同样两个逻辑参数分片：`single` 将二者都映射
到 `npu:0`，`dual_serial` 与 `dual_per_device` 映射到 `npu:0/1`。因此 single 是
单物理卡的固定两分片区域路径，不冒称普通未放置模型；第二张已准入卡在该行空闲。
本组统一使用 `placement_adamw(lr=1e-4, foreach=True)`；大容量组是 foreach=False，
两者不合并比较。共同读出测试损失用于覆盖更新路径，不是下游任务质量实验。

每行执行六次真实更新，包含同一初参、输入和 detached private-state 续接。第 0 次冷更新
单列，第 1–5 次作为五个继续更新样本；参数和状态持续变化，不是五个独立同分布重复。
这里的“冷更新”已做计时外前向审计，包含首次反向/optimizer slot 建立，并非冷进程启动。
JSON 保留所有样本、逐步工作计数及与 single/dual_serial 的比较。后续实际工作差异
阻止严格同轨迹提速结论；计数相等也不能证明活跃节点身份或数值轨迹相等。

12 行均完成六次 finite 更新，共 72 次；60 个继续更新样本作为主要观察。
每组三行的初参 hash、输入 hash 和区域布局相同。下面的 allocated 取全部六步的最高
单卡峰值，不把各卡独立峰值相加冒称同时发生的全组峰值。

| 配置 / 精度 | single 中位 s | dual_serial 中位 s | per_device 中位 s | per_device / dual_serial | 最高单卡 allocated GiB：single / dual_serial / per_device |
|---|---:|---:|---:|---:|---|
| Settle / FP32 | 1.98932 | 2.14867 | 1.75875 | 0.8185 | 15.951 / 8.253 / 8.247 |
| Settle / BF16 | 2.15828 | 2.16386 | 1.87096 | 0.8646 | 14.316 / 7.448 / 7.438 |
| Timed / FP32 | 4.98512† | 4.73802 | 3.92653 | 0.8287 | 17.430 / 9.152 / 9.075 |
| Timed / BF16 | 4.66517 | 5.09545 | 4.15820 | 0.8161 | 15.442 / 8.092 / 8.030 |

四组 per_device 对双卡串行的观察时间比为 `0.816–0.865`。FP32 两种双卡路径与
单卡的计数从 index 3 开始有差异；BF16 从 index 1 开始有差异。两个双卡调度之间也
存在后续计数差异。比如 Settle FP32 的 index 4 Full 数为 single 277,263、
dual_serial 277,274、per_device 277,272。以上是这些已记录轨迹的完整训练时间，
不是严格同轨迹提速；没有根据图大小或设备数推导一般加速比。

输入投影/运输仍由调用线程串行准备、每层有依赖 join、反向仍由普通 autograd 执行。
这些是实现中仍存在的成本边界；没有用 host phase 或单侧算子时间把全部差距归因于某项。
本组监督器采样进程树峰值 4.037GiB，界限 96GiB，每个子进程另限 80GiB。

† Timed/FP32 single 的第 1–3 个继续更新存在较大的计时内 host 间隙：
`0.8665 / 0.8043 / 0.7103s`，随后两步为约 `0.017s`。该字段包含阶段持久化 I/O
和调用线程停顿；没有额外测量将其全部归因于某一组件。原端到端样本原样保留，
没有扣除这些间隙，因而该行与两卡总时间之差不能归为多卡计算收益。

计时包含 zero_grad、准备/packing、forward/readout、必要 Timed flush/loss、反向、host-stage
运输、AdamW、detach、所有参与设备同步及两处中间阶段 JSON 持久化 I/O。CPU 构造、
初参 hash、放置、输入/optimizer 准备、逐步工作 replay、算子审计和 finite/storage 扫描在计时外。
`phase_sync=false`，单个 host phase 不等于设备延迟；阶段 I/O 间隙不从完整时间中扣除。
每步峰值在审计图释放后重置，保留缓存可影响 reserved。吞吐以真实 B×L 输入位置除以完整秒数，
Timed 内部 ticks 和 flush 不增加输入位置数。

## 复现与剩余边界

在新输出路径、固定提交和当次整组设备准入下运行：

```sh
python -m scripts.check_placement --partition-kind region --devices npu:0 npu:1 \
  --region-concurrency per_device --transport host_stage --precision bf16 --output NEW-qualification.json
python -m scripts.check_placement --config configs/scale/hb-9b.json \
  --partition-kind region --devices npu:0 npu:1 npu:2 npu:3 --mode train \
  --precision bf16 --batch 1 --length 8 --steps 2 --no-foreach --threads 8 --output NEW-capacity.json
python -m scripts.check_placement --config configs/hb-multiscale-x-settle.json \
  --partition-kind region --devices npu:0 npu:1 --region-concurrency per_device \
  --transport host_stage --mode train --precision fp32 --batch 8 --length 256 \
  --steps 6 --threads 4 --output NEW-performance.json
```

机器记录包含全部实际命令与输出身份。完整区域或不可分共享组仍是最小放置单元，
不能跨卡切开；`region_batch_size>1` 与区域放置未组合。这里没有 peer DMA、HCCL/复制
训练、resident scheduler 或 CUDA 真机资格。大容量行未验证长窗、checkpoint/恢复、
独立梯度前向和 A–D 训练；小型恢复资格不能外推为大模型恢复容量。新性能组只测完整训练，
没有替代已有推理、含梯度前向或四类联合 Transformer 比较。
