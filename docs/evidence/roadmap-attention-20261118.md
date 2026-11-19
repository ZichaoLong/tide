# ROADMAP 三类 Attention：投影后同函数边界，2026-11-18

CPU 的同六类见证也完成 60 行、300 个正常样本，使用独立固定源码与 NUMA 身份；
CPU/NPU 的计时分别列出，NPU 另 60 个冷样本仅作算子补证，全部排除于性能比较。

NPU 六个有限形状/先验见证的 60 行、300 个正常计时样本已完成，独立审计均返回有限输出和
所需梯度。单位先验 KV 的原生 fusion 在此点较快；8 边/8 候选的小型接收与区域 Attention
的 math 路径较快且通常峰值更低。零先验 KV 显示 tiled 与 materialized math 的时间/内存
取舍。这些结果支持按实际形状选择候选，不支持所有 Attention 统一选同一后端。

[完整小记录](roadmap-attention-20261118.json.gz)保留源文件 hash、原始样本、输入/输出 hash、
实际 dispatch、native 形状/dtype、梯度审计、内存和精确队列身份。NPU 主计时源码/设施固定为
`e2a4d39104a006feec4525e534eadae4a94cde0d`，数学为
`4f5b9f20a4030d68ee66ae2d26e34659a5e37234`。

独立算子补证使用 `5059ff8f751a1aeb0e65d6d1346c6a2d461676c0` 的审计过滤修正；
其 60 行均只采一个冷样本，完全排除于本篇性能与峰值比较。两份 NPU 源身份分别保存。

## NPU 测量身份与语义范围

`attention-sites-npu-v7` 经协作队列独占使用物理 NPU 9 → `npu:0`，
四线程、CPU affinity 80–83，1800 秒、48 GiB 采样进程树 RSS 界限。任务在
`2026-11-18T04:16:59.683872+00:00` 结束，exit 0；服务 inactive/dead、MainPID=0、Result=success，
进程树峰值 2.426 GiB。background.slice 启动记录、队列和文件 hash 已保存。
源码与数学 108 文件已核验。环境为 CANN 9.2.0、Torch 2.12.0+cpu、TorchNPU 2.12.0.post3，
HF32 matmul/conv 关闭；队列外任务和宿主负载不受控制，无共享卡借用身份。

算子补证 `attention-operators-npu-v13` 使用物理 9 → npu:0、affinity 172–175、
四线程、1200 秒/48 GiB 界限，在 `2026-11-18T04:35:00.305267+00:00` 成功结束；
进程树峰值 2.426 GiB，终态同为 inactive/dead、MainPID=0、exit 0。
它不补充可用于性能的 warmup/sample；只修正 observer 对 matmul/linear 名称的覆盖。

这是**投影后的同函数 primitive**：q/k/v、先验、mask、反向 cotangent 已驻留设备，
比较 weighted_attention 的 auto/math/tiled，不计投影、接收后 FFN、packing、模型准备、
优化器或 checkpoint。`forward_backward` 包括清叶梯度、FP32 cotangent 均值收缩、
backward 和同步，不能称为完整训练。每行预热 2 次、采样 5 次，普通计时没有审计 wrapper；
计时后另调用一次记录 dispatch/dtype/输出/梯度。峰值在预热后 reset，计时结束且审计开始前读取。

同 case 的 CPU FP32 canonical fixture 固定；同精度各 backend/模式实际 payload hash 完全一致。
不同 case 使用不同 seed，不能将 positive/zero 两个 case 的差值当作仅切换先验的隔离成本。
吞吐分母是 Attention query-head 行，而非输入 token 或服务 TTFT。

| 形状族 | q 形状 | k/v 形状 | 先验与可见性 |
| --- | --- | --- | --- |
| KV Full tile | 8×8×4×128×32 | 8×8×4×255×32 | 127 个旧 key + 128 个写入，原始时间因果 window=128 |
| incoming reception | 8×8×128×2×8×32 | 同 q | 每节点/样本/位置的 8 边自注意力，宽 64 |
| regional | 8×128×2×8×32 | 同 q | 每样本/位置的 8 候选自注意力，编码宽 d/2=64 |

KV 的 unit 为省略先验；zero 每四 key 置零且保留可见正历史，没有人为设置空 KV bank。
reception/region positive 的先验在 [0.25,1)；zero 另外每七行置全零，覆盖空同刻行。
KV zero 的零项为 4,096/16,320；reception zero 为 23,410/65,536；region zero 为
2,930/8,192。显式先验始终 FP32，primitive 的零点右导数职责与调用方 no-write mask 分开。

## NPU 正常计时与峰值

每格为推理 / forward+backward，中位数 ms；“未测”保留原来的有限候选集合。

| case | payload | auto | math | tiled |
| --- | --- | ---: | ---: | ---: |
| kv-unit | fp32 | 0.534 / 0.984 | 未测 | 1.479 / 3.188 |
| kv-unit | bf16 | 0.519 / 0.892 | 未测 | 1.567 / 3.346 |
| kv-zero | fp32 | 0.915 / 4.022 | 1.342 / 2.359 | 1.647 / 4.030 |
| kv-zero | bf16 | 1.868 / 4.229 | 1.392 / 2.539 | 1.757 / 4.203 |
| reception-positive | fp32 | 1.957 / 7.080 | 1.038 / 2.220 | 未测 |
| reception-positive | bf16 | 2.246 / 4.542 | 1.126 / 2.513 | 未测 |
| reception-zero | fp32 | 2.012 / 2.857 | 1.043 / 2.225 | 1.235 / 2.820 |
| reception-zero | bf16 | 1.427 / 3.024 | 1.102 / 2.616 | 1.310 / 2.996 |
| region-positive | fp32 | 0.961 / 2.038 | 0.778 / 1.471 | 未测 |
| region-positive | bf16 | 1.111 / 1.943 | 0.842 / 1.663 | 未测 |
| region-zero | fp32 | 1.008 / 2.328 | 0.790 / 1.475 | 1.099 / 2.291 |
| region-zero | bf16 | 1.267 / 2.584 | 0.846 / 1.769 | 1.162 / 2.523 |

相应 peak allocated MiB，包含驻留输入/梯度和临时张量；不是增量 scratch 或 npu-smi 整卡占用。
baseline allocated、resident bytes、KV bank bytes、peak reserved 与累计 CPU RSS 在 JSON。

| case | payload | auto | math | tiled |
| --- | --- | ---: | ---: | ---: |
| kv-unit | fp32 | 74.03 / 148.00 | 未测 | 112.13 / 167.76 |
| kv-unit | bf16 | 53.53 / 109.97 | 未测 | 134.09 / 194.01 |
| kv-zero | fp32 | 76.38 / 199.93 | 219.54 / 259.63 | 112.26 / 199.88 |
| kv-zero | bf16 | 122.34 / 213.97 | 229.63 / 259.57 | 122.34 / 213.97 |
| reception-positive | fp32 | 105.44 / 402.58 | 80.32 / 144.88 | 未测 |
| reception-positive | bf16 | 109.20 / 236.61 | 104.26 / 160.89 | 未测 |
| reception-zero | fp32 | 105.95 / 197.64 | 80.32 / 144.88 | 126.76 / 197.64 |
| reception-zero | bf16 | 150.76 / 237.64 | 104.26 / 160.89 | 150.76 / 237.64 |
| region-positive | fp32 | 38.04 / 95.95 | 38.04 / 40.04 | 未测 |
| region-positive | bf16 | 35.04 / 46.14 | 35.04 / 37.62 | 未测 |
| region-zero | fp32 | 38.04 / 50.15 | 38.04 / 40.04 | 41.24 / 50.15 |
| region-zero | bf16 | 44.24 / 56.15 | 35.04 / 37.62 | 44.24 / 56.15 |

KV unit 的 fusion 前向反向用时为 tiled 的 FP32 0.309×、BF16 0.267×；这个长矩形
见证支持保留 native。KV zero 的 math 前向反向比 auto/tiled 快，但 FP32/BF16 peak
为约 259.6 MiB，auto/tiled 为约 199.9/214.0 MiB；不能仅选最快时间而隐藏存储代价。

positive reception 的 auto/math 前向反向时间比为 FP32 3.19×、BF16 1.81×；positive
region 为 1.39×/1.17×。zero reception/region 的 math 也较短。这里只测 E=N=8、
head width=32 的投影后边界；在整模型采用按小键/查询数选择 math 前，需保留该源身份、
检查数学/梯度与路由，并对实际组合计时，不能把 local 比率当整网收益。本报告不修改 dispatch。

## NPU 实际后端与精度

| auto 情况 | FP32 推理 / 反向前向 | BF16 推理 / 反向前向 |
| --- | --- | --- |
| KV unit | native / native，QKV head=32 | native / native，QKV head=32 |
| 显式正先验 reception/region | native PSE head=32 / rank-one head=48 | rank-one head=48 / rank-one head=48 |
| 可见零先验 KV/reception/region | native PSE / tiled | tiled / tiled |

native 的实际 API 是 `torch_npu.npu_fusion_attention_v2`；反向审计记录相应 grad v2。
FP32 native 的 QKV 为 FP32，BF16 native 的 QKV 为 BF16。显式正先验的 rank-one 编码
将本例宽 32 扩成 padding 后的 48，并保持原 scale=32^(-1/2)，不是偷偷按 48 重标度。
BF16 forward/backward 输出和 q/k/v 叶梯度是 BF16，显式 prior 梯度是 FP32。

v7 的 45 个 math/tiled 行原本 `ArithmeticAudit.operators=[]`，因为观察器未匹配 NPU
保留的 `aten.matmul.default` 名称。该缺口保留原样，由 v13 独立补证：新增 matmul/linear
过滤后 60 行均有实际算子，math/tiled 在 FP32 和 BF16 payload 下都观察到 FP32 matmul
输入/输出；native 仍观察到对应 payload dtype 的 fusion/grad。这与显式 FP32 稳定子计算一致。

两份 source 的 `attention.py`、`attention_sites.py` 和 fingerprint 所在文件逐字节相同；
相关 precision/statistics diff 仅扩大审计算子名称过滤。六 case 的 60 个逐行 canonical
input、actual payload、output hash、shape、dispatch 和梯度 metadata 全部对应一致。
metadata 不包含梯度数值，所以没有借这组对照宣称梯度逐位相等。原 300 个正常 samples
与 v13 冷样本严格分开，未替换性能或内存数值。

这些是 dispatch-visible tensor dtype 见证，仍不是全部算子或硬件 accumulator 审计。
不能将 BF16 storage 标签说成每个 Attention 子运算都用 BF16 kernel。

BF16 的同 backend 推理与反向模式 output hash 在六 case 全部相同。FP32 positive
reception/region auto 的 hash 不同，记录对应 PSE→rank-one 的原模式相关表示；
其他已测 mode pair hash 相同。hash 相同是该 dtype 输出的逐位事实，不替代梯度或误差界。

## NPU 历史梯度、正确性界限与复现

所有 forward_backward 审计的 q/k/v 梯度均存在且有限，显式 weights 的 FP32 梯度亦然。
unit KV 的旧 127 key 的 k/v 梯度各有 1,040,384 个非零元素，zero KV 各有 778,240 个，
两精度/所有已测 backend 都保留这条反传。它证明该有限原始时间历史支路可达，没有把
历史 detach；不等于多窗口状态续接、恢复后更新或对零点右导数值的独立误差验证。

本入口没有保存跨 backend 输出/VJP 的误差张量、误差范数或阈值判定，只有 hash、有限值、
连通性与旧历史非零梯度。因此“同函数比较”描述数学接口和配对输入，不表示已获数值等价
资格；独立公式/精度资格另行记录，原超差不被此处的 passed 标签覆盖。

正常样本最大极差/中位数为 0.296，原始样本和 host load 均保留。
无失败行；未测项包括 KV unit 的 math、正先验 reception/region 的 tiled、其他键数/宽度、
整节点投影/FFN/packing、CUDA、优化器与真实续接。v13 只补算子见证，不形成重测性能结论。

复现环境按 [development](../development.md)，使用新身份并取得当次设备准入：

```sh
python scripts/launch.py --name new-attention-sites \
  --revision e2a4d39104a006feec4525e534eadae4a94cde0d \
  --npu-count 1 --threads 4 --affinity 80-83 --seconds 1800 --rss-gib 48 \
  -- python -m benchmarks.attention_sites --device npu:0 \
  --cases kv-unit kv-zero reception-positive reception-zero region-positive region-zero \
  --threads 4 --warmup 2 --steps 5 --output '{run}/result.json'
```

复现独立 observer 用 JSON 的 operator_supplement.launch：改为其 source、warmup=0、
steps=1 和新输出 identity，所得冷样本不能加入普通性能比较。fixture/时间可见性入口为
`tests/test_attention_sites.py`；本报告不新增其通过次数声明。


## CPU 正常样本：NUMA7、32 线程

`attention-sites-cpu-v16` 固定 `0992b467584c1f603314ed04d8efe8596a14f576`，
等待 `continuous-cpu-v16` 释放资源后才启动计算；等待记录和 NUMA 策略完整保留。
线程与实际 CPU affinity 为 32 / 280–311，`numactl --membind=7 --physcpubind=280-311`；
宿主其他任务不受控制。这是固定测量身份，不声明 32 线程对每个 Attention 形状都最优。
任务在 `2026-11-18T05:22:41.388599+00:00` 终态 exit 0；服务 inactive/dead、MainPID=0、Result=success。
外层持续 1822.240 秒，包含前置资源等待；采样进程树 RSS 峰值 0.797 GiB。
前置等待 1808.615 秒；从 dependency 释放到服务终态约 13.579 秒，仍包括载入、构造和审计。
源码与固定数学逐文件 hash 已核验；Torch 版本、线程、实际 affinity、CPU load 和 NUMA 信息见 JSON。
预热 2 次、正常采样 5 次；后续独立审计不进入这些样本。它们不是 NPU v13 的冷一次补证。

每格为推理 / forward+backward 的中位数 ms；范围仍是上述投影后 primitive。

| case | payload | auto | math | tiled |
| --- | --- | ---: | ---: | ---: |
| kv-unit | fp32 | 3.071 / 8.680 | 未测 | 21.130 / 47.788 |
| kv-unit | bf16 | 3.474 / 22.805 | 未测 | 6.844 / 21.718 |
| kv-zero | fp32 | 10.035 / 32.472 | 8.082 / 17.624 | 6.529 / 25.034 |
| kv-zero | bf16 | 24.031 / 57.079 | 8.535 / 18.102 | 6.681 / 25.480 |
| reception-positive | fp32 | 2.377 / 22.337 | 2.729 / 8.593 | 未测 |
| reception-positive | bf16 | 13.595 / 8.957 | 3.364 / 31.078 | 未测 |
| reception-zero | fp32 | 10.621 / 36.163 | 2.858 / 8.469 | 3.458 / 10.853 |
| reception-zero | bf16 | 4.398 / 13.583 | 16.153 / 9.862 | 4.249 / 44.545 |
| region-positive | fp32 | 1.005 / 2.363 | 1.582 / 3.101 | 未测 |
| region-positive | bf16 | 6.150 / 2.790 | 7.977 / 3.569 | 未测 |
| region-zero | fp32 | 1.070 / 22.569 | 1.650 / 3.497 | 7.310 / 4.385 |
| region-zero | bf16 | 1.909 / 4.718 | 1.861 / 3.635 | 8.979 / 4.713 |

CPU auto 的实际调用与两个模式如下；math/tiled 为显式请求。

| case | payload | 推理 effective | forward+backward effective | 同 backend auto 输出 hash 相同 |
| --- | --- | --- | --- | --- |
| kv-unit | fp32 | sdpa | sdpa | 是 |
| kv-unit | bf16 | sdpa | sdpa | 是 |
| kv-zero | fp32 | sdpa | tiled | 否 |
| kv-zero | bf16 | tiled | tiled | 是 |
| reception-positive | fp32 | sdpa | sdpa | 是 |
| reception-positive | bf16 | sdpa | sdpa | 是 |
| reception-zero | fp32 | sdpa | tiled | 否 |
| reception-zero | bf16 | tiled | tiled | 是 |
| region-positive | fp32 | sdpa | sdpa | 是 |
| region-positive | bf16 | sdpa | sdpa | 是 |
| region-zero | fp32 | sdpa | tiled | 否 |
| region-zero | bf16 | tiled | tiled | 是 |

实际审计到的算子/dtype 集合（合并本矩阵中该精度/effective 的调用，不推断硬件累加器）：

| payload / effective | dispatch-visible 算子和 dtype |
| --- | --- |
| bf16 / math | `aten.bmm.default [cpu:float32 → cpu:float32]` |
| bf16 / sdpa | `aten.bmm.default [cpu:float32 → cpu:float32]` |
| bf16 / tiled | `aten.bmm.default [cpu:float32 → cpu:float32]` |
| fp32 / math | `aten.bmm.default [cpu:float32 → cpu:float32]` |
| fp32 / sdpa | `aten.bmm.default [cpu:float32 → cpu:float32]` |
| fp32 / tiled | `aten.bmm.default [cpu:float32 → cpu:float32]` |

所有 60 行独立审计均有实际算子、有限输出与所需的有限梯度；KV 旧历史 k/v 反传均非零。
此次 CPU 的全部矩阵算子见证（含 auto→SDPA）均为 FP32 bmm 输入/输出。q/k/v、输出与
q/k/v 叶梯度遵守所请求的 FP32/BF16 payload dtype；显式 prior 及其梯度为 FP32。
math/tiled 同样采用 FP32 子计算；原始 BF16 payload 与
返回/叶梯度 dtype 不等于每个内部子运算都采用 BF16 kernel。SDPA 是实际 API/dispatch 名称，
不能仅据此称 CPU 执行 Flash；其实际算子在上表与逐行审计中保留。
推理/反向模式输出 hash 的不同项：kv-zero/fp32/auto, reception-zero/fp32/auto, region-zero/fp32/auto。
与 NPU v7 的 60 对记录中，canonical hash 一致 60/60，
实际 payload hash 一致 60/60、形状一致 60/60；
CPU/NPU 输出 hash 相同 0/60；不同项均保留，
没有跨设备误差范数或梯度数值阈值判定，不能把 hash 不同直接解释为超出数值阈值。

CPU 没有逐行设备 allocator 峰值：三个 device-peak 字段保持 null。JSON 中的 CPU RSS 是
该进程跨前序 case/backend 的累计高水位，不能冒充每个 backend 的峰值或增量 scratch。
resident input 和 KV bank bytes 是确定的张量负载计数；NPU 峰值表只属于 NPU。
CPU 正常样本最大极差/中位数为 2.571；全部原始样本均保留。
60 行中 12 行的极差超过各自中位数的一半。
例如 BF16 KV-zero 的 auto 与显式 tiled 均实际调用 tiled，payload/output hash 相同；反向中位数却为 57.079 / 25.480 ms。
BF16 reception-zero 的同类对照则为 13.583 / 44.545 ms。
这种同路径、同负载下的不稳定次序说明当前五样本不足以作稳健后端排名；噪声来源没有隔离。
本报告不筛除样本，也不据这些 CPU 中位数宣称固定加速比。
CPU 的这些局部时间不替代整模型、参数更新、连续训练或实际恢复的测量；不修改现有 dispatch。

CPU 复现仍使用新服务身份，并由当前资源调度确定可用 NUMA 节点和核；本次实际命令为：

```sh
python scripts/launch.py --name new-attention-sites-cpu \
  --revision 0992b467584c1f603314ed04d8efe8596a14f576 \
  --threads 32 --affinity 280-311 --seconds 18000 --rss-gib 48 \
  -- numactl --membind=7 --physcpubind=280-311 python -m benchmarks.attention_sites \
  --device cpu --cases kv-unit kv-zero reception-positive reception-zero region-positive region-zero \
  --threads 32 --warmup 2 --steps 5 --output '{run}/result.json'
```

实际 v16 还带前置 `--after-unit tide-continuous-cpu-v16.service` 和 NUMA 审计包装；完整 launch
与 dependency/NUMA 记录在 JSON。被替代的 v14 保留独立终态，不进入本表。
