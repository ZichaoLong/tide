# 有界连续推理与大模型比较入口

`python -m benchmarks.inference` 只测主干推理。它保留真实 KV、SSD 和 Timed
pending 状态，在同一请求内依次执行 prefill、单位置 decode、可选分块流，最后只做
一次必要的 Timed drain。已有 `benchmarks.continuous` 的训练/续接入口保持原契约。

每次 cold、诊断、warmup 和正常样本都从独立零态开始；请求内部不重置状态、不在
prefill 后 flush。返回 readout 在所属设备上完成，所有参与设备同步后释放该窗口。
两边都不保留整条输出、不拼接、不做任务头、采样或 D2H 输出传输。输入构造、初次
传输、参数哈希、模型构造、诊断、warmup 均不计入正常时间。

## 时间与分母

| 字段 | 含义 |
|---|---|
| `cold` | 本进程中该 subject 首次真实调用，先于它的诊断和 warmup；包含整条请求 |
| `first_any_backbone_output_seconds` | 首个出现正输出强度的调用完成；由独立同负载诊断定位，再映射正常计时边界 |
| `prompt_ingested_seconds` | 最后一个 prompt 输入窗口完成，不保证 Timed 已输出其最深路径 |
| `prompt_output_ready_seconds` | 最后 prompt 注入时刻加完整图深度延迟已被返回窗口覆盖的首个完成边界 |
| `first_incremental_call_seconds` | prefill 后第一次单位置增量调用的时延 |
| `stable_decode` | 排除第一次增量调用后的真实 KV/SSD/pending 续接；不足两次则无稳定样本 |
| `chunked_stream` | decode 后可选分块续接的输入位置吞吐 |
| `drain_seconds` | 唯一末尾排空，输入分母为零；其 clock 前进仍记录在 state 中 |
| `complete_seconds` | 请求首调用至最终同步，含 readout、调度、输出释放和必要 drain |

时延在完整调用边界采样；不能据此宣称逐 tick 的更细首输出时间。
`prompt_output_ready_seconds` 是保守的终端路径就绪边界，没有把后续记忆影响归因到
某个输入；原始 int64 返回时间在独立诊断中逐调用核对。无任务头，以上都不是语言
模型服务 TTFT。所有吞吐以实际 `B × 输入位置` 为分母，逻辑 ticks、padding 和
drain 返回槽均不算新输入。

多个 subjects 在同一进程顺序运行时，后面的 `cold` 可能复用此前编译/进程缓存。
正式比较首次调用应分别启动只含一个 subject 的新进程；记录列明了此前 subjects。

## 有限入口

以下 CPU 小例可验证配置与结果结构。NPU 需另按本机准入和冻结作业规范启动；默认
设备为 CPU，入口不自动申请硬件。每次使用新的输出路径。

```sh
python -m benchmarks.inference \
  --config configs/continuous/hb-settle-small.json \
  --config-overrides '{"aggregation":"incoming_light_64","region_batch_size":2}' \
  --prefill 16 --prefill-chunk 8 --decode 4 --stream-positions 16 --stream-chunk 8 \
  --samples 2 --warmup 1 --threads 4 \
  --output NEW-tide.json --reference-output NEW-reference.json
```

`--config-overrides` 是顶层 JSON 对象；最终配置完整进入结果与 reference 身份。
初筛可用 `--audit none --subjects tide`，但该结果不能生成正式 A–D 匹配。
`--audit-only` 只运行全负载诊断/审计并冻结 reference。大模型可先用 decode 16、
samples 2；最终代表点使用 decode 256 或另行声明的有限长度。

正式比较既可 `--subjects tide A B C D` 顺序运行，也可使用第一步保存的 reference，
分别在新进程运行 `--subjects A`、`B`、`C`、`D`。后者无须重建 Tide：

```sh
python -m benchmarks.inference \
  --config configs/continuous/hb-settle-small.json \
  --config-overrides '{"aggregation":"incoming_light_64","region_batch_size":2}' \
  --subjects A --reference NEW-reference.json \
  --prefill 16 --prefill-chunk 8 --decode 4 --stream-positions 16 --stream-chunk 8 \
  --samples 2 --warmup 1 --threads 4 --transformer-chunk 256 \
  --output NEW-transformer-A.json
```

A/B 固定节点总数附近的 block 数，C/D 固定图深度；A/C 用实际唯一总参数，B/D 用
该完整实际负载的有效触及均值。Timed 均值按 sample/tick 去重，包含唯一最终 drain，
另有每输入位置累计量。构造器在审计后冻结，不在正常计时中重新选择。配置、调度、
B、精度、seed 和实际 Tide 源码哈希不匹配时拒绝复用。输入在共同 feature 前缀上
逐元素配对，记录实际完整输入和前缀哈希。两种主干数学不同，不认证任务质量等价。

## 大模型放置与审计内存

固定其他层数的补充对照使用 `python -m benchmarks.transformer_depth`，例如
`--parameters 8778358224 --layers 32 --devices npu:0`。它沿用同一 Transformer、
近参数搜索、真实缓存请求、诊断和计时函数，明确标记为辅助比较，不替代 A–D。
`--target-reference` 可引用原 Tide 的 reference，核对总参数目标、完整请求调度、
batch/seed/精度及共同输入前缀；不会重建或重新认证旧 Tide。每个作业只构造一个
Transformer，记录实际宽度、FFN、头宽、参数误差、设备与样本。默认两次热身、
五次正常样本，不测 cold；正式 NPU 使用仍需冻结与准入。

`--devices npu:0,npu:1 --placement layer` 使用 Tide 连续层放置；`--placement region`
按实际参数字节平衡完整区域，可加 `--region-concurrency per_device`。实际 admitted
设备由启动者提供。Transformer 使用连续 block 所有权，每层 KV 留在该层设备，
仅传输新 hidden rows；C/D 层数少于设备数时只使用足够的设备并列出其余设备。
`--transport auto` 延续现有多 NPU host-stage 策略，不声明 peer DMA 或 HCCL。

每次只构造一个主干，释放权重和状态后再构造下一个。参数身份按 8 MiB 块计算哈希
和有限性，审计在 placement 前冻结无 Tensor 引用的语义清单，避免旧 dense 参数
被清单挂住。审计使用每参数切片一个 bit 的事件集合，每次实际调用后立即汇总并
释放；保留流总量、极值和逐时刻计数摘要哈希，不保留 `[sample,time,参数标量]` 数组。
它仍随单次窗口和语义切片数量增长，不是与任何配置无关的常量内存。

并发区域的普通 Python hooks 在实际工作线程执行，共享计数器加锁，单次 joined
forward 内逻辑时间不变。它不依赖 caller-thread dispatch mode，也不强制串行。
结果单独声明 requested/effective 并发策略。FP32/BF16 的算术策略、返回 dtype、
状态 dtype、有限输出/状态、实际设备以及 source/config/input/parameter 哈希分别记录。
这些观测不是对未运行硬件算子或任意训练轨迹的资格声明。

每张参与设备分别记录 allocated/reserved 当前值和峰值；
诊断后先释放空闲 allocator 块，再独立 warmup，避免把仅诊断需要的缓存计入正常
reserved；warmup 的常规缓存则保留供正常样本复用。
`--allocated-budget-gib`、`--reserved-budget-gib` 是请求完成后的有限停止条件，
不是硬 allocator 上限。首次、诊断、样本和失败阶段以 fsync/atomic replacement
持久化；长作业仍需外层 service 时间、RSS 和设备准入边界。

## 独立缓存时间与淘汰诊断

性能入口的输出时间、有限性和状态字节记录，不单独证明每个 KV bank 已填满或
正确淘汰。`python -m scripts.check_inference_cache` 在独立、完全不计时的请求中
观察真实正接收写入，以每个 owner/sample 的有限时间戳队列作为参照，再抽查实际
KV、SSD/Conv 和 pending。它不复制 K/V payload，也不改正常生产或计时路径。

```sh
python -m scripts.check_inference_cache \
  --config configs/continuous/fc-k-small.json \
  --prefill 128 --prefill-chunk 32 --decode 256 --sample-every 64 \
  --require-kv-eviction --threads 4 --output NEW-cache-diagnostic.json
```

每个抽查边界记录 KV 容量、正条目数、原始 int64 首末时间、最后正写入时间、窗口
年龄及完整保留时间戳集合的摘要，并逐项与独立写入队列比较。淘汰证据要求同一个
owner/sample 的最早保留时间前移，旧条目已超出该 owner 的窗口，且两端保留集合
均通过核验。空 bank 和不再活跃的 owner 合法；窗口相对于其最后正写入，而不是
当前全局时钟，因此不要求所有 owner 都填满。`--require-kv-eviction` 要求至少一个
owner 出现这样的证据，不代表所有 owner 均已跨窗。

原始输出 int64 时间轴逐调用检查；SSD 原始时间与最后正接收一致，Conv 保留集合
按其有限 horizon 检查。Timed 只在最后 drain，结束时有出边节点必须无待发送强度；
period=1 的终端节点可保留合法输出尾部。每次抽查原子保存进度，失败不会被过滤。
这些观察验证被运行配置的续接元数据契约，不能替代独立数学、梯度或长训练检查。
