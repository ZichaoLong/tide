# KV128 与固定32层 Transformer：训练与推理成本拆解

本次同卡 FP32 对照中，中尺度继续训练 Transformer 吞吐为 Tide 的 **5.138倍**；
9B图同 B8 推理的 prefill / 稳定 decode 分别为 **5.852倍 / 7.523倍**。
窄KV保留了大部分独立FFN参数、区域分配、消息处理、私有历史整理和对应反向。
设备观察显示，节点权重索引、Full矩形padding及非矩阵反向是具体成本，不能把差距全部归于Attention。

完整正常样本、诊断聚合、形状、资格及生命周期见[机器可读证据](kv128-transformer-breakdown-20261119.json.gz)。
不同模型函数只作工程参考，不要求数值相等，不比较任务质量；共同32层参考不替代严格A–D。
本次没有改变生产模型、数学、阈值或训练默认设置，也没有新增9B训练资格。

## 身份、形状与测量边界

测量工具先固定并推送 `7196fcabd7ab3b401c1ebdb6107442450a99996c`，四点均执行其冻结源码。
生产 `src/tide` tree 仍为 `5dba5b15c0c71ce729ff82c054e9468916acd7e5`；
数学 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234` 的108文件原字节核验通过。

| 用途 | Tide | Transformer32 |
|---|---|---|
| 推理 | HB Multiscale Settle 5-dd4，768 D/80 H/44区域，d768/摘要192/KV128/FFN3072，7,268,408,784参数 | d4352/FFN15208，68×64头，窗口128，8,778,371,328参数 |
| 训练 | 同拓扑profile的中尺度，192 D/40 H/44区域，d192/摘要48/KV32/FFN768，129,025,464参数 | d496/FFN2048，31×16头，窗口128，129,039,856参数 |

训练配置文件名 `medium-kv128` 表示KV内部宽为内容宽的1/6，**实际为KV32，不是KV128或9B训练**。
推理保留用户指定的共同32层大参考，总参数并不相等；训练另选32层近总参数形状，差+0.01115%。
Tide为独立节点参数、region_batch_size4、source_residual、gather、SSD prefix/doubling；
训练启用Reception重算，推理关闭。KV共同收窄Q/K/V总内部宽，不是GQA。

本机 Ascend910 64GiB，空闲设备队列独占准入。推理pair顺序使用物理卡9、CPU affinity36–39；
训练pair顺序使用物理卡11、CPU affinity40–43；每进程4个Torch线程，最多两卡同时工作。
只读公共 `ascend/dev-workspace-9.2.0` 与私有 `.venv-instances`，Python3.12.13、
Torch2.12.0+cpu、TorchNPU2.12.0.post3、CANN9.2.0；实际矩阵输出观察到FP32/npu:0。
HF32在两个驱动中均显式关闭，文件后缀`tf32`指Transformer32层，绝非TensorFloat-32。
主机仍共享，外部主机争用未被控制；本次没有使用共享占用NPU。

三种观察严格分开：

1. **正常时间**：驻留参数/输入，2次warmup、5次样本；包括主干/readout/调度/同步，
   训练还包括loss、反向、AdamW。构造、hash、CPU锚点、输出检查和JSON写入不计时。
2. **模块同步诊断**：每个嵌套模块入场同步完成后起钟，完成同步在计时内。
   只加exclusive切分，inclusive子树不可与父级重复相加。父级剩余包括未包裹工作、等待与观察开销。
3. **逐算子同步诊断**：另一独立重放，按dispatcher调用记录形状、dtype、次数及完成时间。
   不含入场等待、描述符记录和Python间隙；一次dispatch可能含多个原生kernel或没有kernel。
   算子秒数不是纯设备kernel时间，也不能除以正常wall time得占比。

推理每个样本从空态开始，真实执行prompt128/chunk32/decode16；稳定decode只取末15步。
训练为B16/L128/chunk32，两个真实warmup更新之后测5个继续更新，每次更新2,048输入位置；
4个chunk内保留图，更新之间detach私有状态。目标是主干输出平方均值，不含任务头、数据管线或质量评价。
首输出是第一个有可用输出的主干窗口完成时延，不是完整语言模型TTFT。

## 正常训练吞吐及整步时间

以下阶段取同5个更新的**均值**，因此可相加；吞吐为10,240实际输入位置/5次完整更新时间之和。
前向窗口包含zero_grad、chunk调度及readout。旧`run_stream`的backward区间可能带有待完成拼接/loss；独立loss的同步值另列。

| 正常每次更新 | Tide d192/KV32 | Transformer32 | Tide/Transformer耗时 |
|---|---:|---:|---:|
| 前向窗口 | 1,228.347 ms | 183.183 ms | 6.706× |
| 反向区间（含loss尾部） | 873.269 ms | 216.879 ms | 4.027× |
| AdamW更新 | 17.713 ms | 12.301 ms | 1.440× |
| 其余整步时间 | 0.661 ms | 0.265 ms | — |
| 完整更新 | **2,119.989 ms** | **412.627 ms** | **5.138×** |
| 输入位置/s | **966.043** | **4,963.322** | — |
| allocated / reserved峰值 | **22.739 / 23.777 GiB** | **6.644 / 7.670 GiB** | — |

两者每步相差约1.707s，其中前向差1.045s（61.2%），反向差0.656s（38.4%）；
优化器差约5.4ms。完整更新时间中位数分别2.100965s/0.412245s，与上面的均值口径分别保留。
这说明当前主要机会在前向组织和反向图，不在AdamW。

模块同步诊断分别从同一非空CPU锚点重放首个正常更新，阶段如下：

| 同步阶段，每更新 | Tide | Transformer32 |
|---|---:|---:|
| 前向 | 1,472.619 ms | 331.399 ms |
| loss | 0.113 ms | 0.108 ms |
| 反向 | 987.755 ms | 217.312 ms |
| AdamW | 17.915 ms | 12.334 ms |

Tide同步前向1,472.619ms的互斥拆分如下。FFN已从Full父项扣除：

| Tide同步前向模块 | 每更新ms |
| --- | ---: |
| 消息收集和接收 | 203.683 |
| 描述及区域评分 | 181.441 |
| 分配求解 | 382.442 |
| KV准备/历史整理 | 137.068 |
| SSD准备 | 126.751 |
| Full中的FFN（含归一化） | 67.421 |
| Full其余（扣除FFN） | 223.400 |
| 状态join/split | 38.641 |
| 输入、分组及未细分余项 | 111.773 |
| 合计 | 1,472.619 |

Transformer同步前向331.399ms中：RoPE93.644ms、Attention64.279ms、FFN43.187ms、
QKV/输出投影19.483ms、归一化32.931ms，其余77.876ms。
Tide反向987.755ms中Reception重算可单独观察到72.538ms；其余自动微分不能按前向标签直接归属于FFN等模块。

逐算子同步的**另一趟**反向总和，Tide为2,705.609ms/44,524次dispatch，
Transformer为712.791ms/13,462次。分类是互斥的，`pointwise_or_other`不能都解释成纯逐元素kernel：

| 反向算子分类 | Tide ms | Transformer32 ms |
|---|---:|---:|
| 索引/写回/gather与其反向 | 726.085 | 67.212 |
| 矩阵类 | 201.786 | 59.855 |
| 原生Attention反向 | 42.037 | 42.249 |
| reduction | 114.259 | 33.309 |
| tensor创建 | 219.178 | 7.412 |
| view/metadata | 346.039 | 88.915 |
| pointwise或其他 | 999.773 | 370.439 |

未列的拼接、copy和重算参数索引仍计入总和。Tide具体热点包括
`index_put`336.821ms/1,660次、`npu_gather_backward`226.609ms/176次、
`add`359.866ms/5,737次、`mul`287.937ms/4,401次、`bmm`153.180ms/1,824次。
这些是定位证据，不能把它们相加解释为正常反向873.269ms的组成，也不能全归于某一个前向模块。

## 正常9B图推理与同步模块细拆

同B8、5次样本中位数；所有吞吐按独立样本×输入位置，非节点更新数。

| 正常B8推理 | Tide KV128 | Transformer32 |
| --- | ---: | ---: |
| 首主干输出 ms | 749.359 | 129.075 |
| prefill128总计 ms | 3,051.378 | 521.462 |
| prefill输入位置/s | 335.586 | 1,963.709 |
| 稳定decode每步平均 ms | 438.813 | 58.329 |
| 稳定decode输入位置/s | 18.231 | 137.153 |
| 完整请求 s | 10.075 | 1.455 |
| 独立完整序列/s | 0.794 | 5.499 |

完整请求包含128位置prefill与16步decode。稳定decode的时延/吞吐只用15步；
下面模块诊断则覆盖全部16步，decode列为16步累计除以16，不能与正常15步混同。
正常峰值分配/保留显存为Tide33.491/35.854GiB，Transformer34.866/37.639GiB。
缓存最终大小不等于上述峰值。训练allocator采样含run_stream末尾有限性检查，
检查不计入complete时间，但其临时分配可能计入峰值。训练峰值包含autograd保留等工作；本次未新增同轨迹activation/scratch ledger，
不通过allocator减权重反推纯激活或临时工作区。此前memory-candidate ledger保留其独立seed/B8身份。

| 同步模块，单位ms | Tide prefill总计 | TF prefill总计 | Tide decode每步 | TF decode每步 |
| --- | ---: | ---: | ---: | ---: |
| FFN，含自身归一化 | 893.647 | 352.951 | 169.156 | 42.983 |
| KV准备和历史整理 | 265.170 | — | 71.870 | — |
| SSD准备 | 197.594 | — | 34.956 | — |
| KV/SSD Full扣除FFN | 336.845 | — | 63.331 | — |
| 区域分配求解 | 383.835 | — | 55.136 | — |
| 描述与区域评分 | 314.116 | — | 57.868 | — |
| 消息收集与接收 | 728.645 | — | 38.770 | — |
| 状态join/split | 34.686 | — | 11.358 | — |
| Transformer QKV/输出投影 | — | 85.147 | — | 13.396 |
| Transformer Attention | — | 65.297 | — | 14.598 |
| Transformer RoPE | — | 110.066 | — | 9.909 |
| 其他计算、归一化及未细分调度 | 266.433 | 89.381 | 24.815 | 21.628 |
| 合计 | 3,420.972 | 702.842 | 527.259 | 102.515 |

Tide的FFN已从KV/SSD Full中剥离，其余Full仍含query投影、Attention、打包/发布、输出等。
另一种不拆FFN的父级视图：KV prepare/full在decode为71.870/175.038ms每步，
SSD prepare/full为34.956/57.449ms每步，不能再加上表FFN。
KV历史compaction/gather合计49.343ms/步，包含在KV准备项；208次FFN、128次KV compaction、
640次KV字段gather均为16步累计。SSD准备发生在Full选择之前，包含卷积、递推和摘要，不由Full稀疏率直接缩小。
同步诊断本身改变调度：Tide每步527.259ms、Transformer102.515ms，
不同于正常稳定decode约438.8ms/58.3ms；诊断比例不能替换正常7.523倍。

## 为什么窄KV仍慢：矩阵、索引与padding的证据

固定FFN的9B图中，独立FFN参数为
`848 × 3 × 768 × 3072 = 6,002,049,024`，占KV128总参数的82.58%。
KV768→128只减少记忆投影和历史payload；内容d768、摘要192、FFN3072、原始int64窗口操作、
消息收集、区域分配与SSD Hub准备都继续存在。参数更少不保证实际工作或调度更少。

当前独立节点投影和反向可写成：

```text
counts[i,b] = 本组节点i、样本b的选中query数
qmax = max(counts)
Xpack = pack(selected_X, shape=[owners, B, qmax, d])
Xflat = Xpack.reshape(owners, B*qmax, d)
Wselected = W[owner_ids]                  # 当前gather路径，产生索引结果
Zflat = bmm(Xflat, transpose(Wselected))
Y = publish(Zflat.reshape(owners, B, qmax, out_width), selected_positions)
executed_rows = owners * B * qmax

grad_W = index_add(owner_ids, bmm(transpose(grad_Zflat), Xflat))
grad_X = gather/scatter(grad_Xpack, selected_positions)
```

FFN gate/up/down重复此归属路径。节点矩阵各自独立，不能仅把B和节点展平就得到共享权重的一个GEMM。
Transformer按层直接用相应权重作较规整的投影，没有对应的节点参数池gather。
本次逐算子同步诊断的FFN项为：

| FFN内部，另一独立算子诊断 | Tide prefill总计 | Tide decode每步 | Transformer prefill总计 | Transformer decode每步 |
|---|---:|---:|---:|---:|
| 浮点节点权重索引 | 677.233 ms | **149.615 ms** | 无对应项 | 无对应项 |
| 矩阵类 | 164.437 ms | **15.880 ms** | 310.814 ms | **35.528 ms** |

Tide矩阵类时间在这个局部观察中较少，却多出大量取权重成本；不能把同步FFN169.156ms/步全部叫做GEMM慢。
这一成本具有源码与实际NPU完成调用证据，但没有消除gather后的整体加速实验，不能把149.615ms全视为可回收收益。

实际prefill FFN示例：Tide为`[128,160,768] × [128,768,3072]`的独立节点BMM，
Transformer为`[8,32,4352] × [4352,30416]`的二维gate/up权重投影。
decode示例：Tide SSD FFN为`[16,8,768] × [16,768,3072]`，
Transformer为`[8,1,4352] × [4352,30416]`，以及`[8,1,15208] × [15208,4352]`。
这只是代表操作数，不是所有算子唯一形状，也没有从dispatcher形状推断原生kernel的具体tiling。

窄KV仍保留摘要宽192；decode有`[128,8,1,129] × [128,8,129,192]`的历史摘要计算。
16步中640次历史字段gather产生21.141GiB重复逻辑输出，1,200次state.join的cat产生21.817GiB；
所有参数索引作用域产生261.385GiB重复逻辑输出，其中FFN为193.641GiB。
这些是把实际输出张量逻辑字节按调用数相加，**不是总线流量、峰值显存或带宽测量**。

从实际`Queries.publish`的三个索引长度推导selected rows，从对应FFN down BMM矩形推导executed rows，
两种调用次数逐phase/记忆种类一一对账；这些是本次算子诊断的实际工作，不是正常计时轨迹的活动证书：

| Tide负载/记忆 | 选中行 | FFN执行行 | 执行/选中 |
|---|---:|---:|---:|
| 9B prefill KV | 77,528 | 467,336 | 6.028× |
| 9B decode16 KV | 9,749 | 48,512 | 4.976× |
| 9B prefill SSD | 81,920 | 81,920 | 1× |
| 9B decode16 SSD | 10,240 | 10,240 | 1× |
| 中尺度训练KV | 118,515 | 335,104 | 2.828× |
| 中尺度训练SSD | 81,920 | 81,920 | 1× |

padding是本次实际张量矩形的额外行；1×不排除后端内部tile补齐。选择预算k约束强度质量，不是硬选中节点数。
结合数据，可优先评估独立权重的索引/布局、Full按query数分组、历史bank整理及反向索引汇总；
中尺度前向另有分配求解382ms的明确热点。需要在新有限候选下验证梯度/状态/恢复并重新正常计时，
本次不宣称任何未实现的加速收益，也不因解释差距而替换默认模型。

## 数值、恢复、资源与保留限制

新增观察器CPU定向检查34 passed/1已有profiler warning，覆盖hook恢复、起钟边界、
独立诊断、预算拒绝及四chunk继续/恢复后更新。开发期新增driver的ConvState NamedTuple复制错误
被测试发现并在固定工具身份前修复；不是生产模型修改。未因文档整理重复全CPU矩阵。

训练每个诊断都从参数、梯度、AdamW、RNG及非空私有状态CPU锚点恢复，真实更新位置256→384。
与首个正常更新逐张量比较，保持`atol=1e-6, rtol=2e-3`：

| 候选，两诊断均如此 | 参数max_abs | 梯度max_abs | AdamW max_abs | 输出/完整私有状态 |
|---|---:|---:|---:|---|
| 中尺度KV32 | 2.910383e-11 | 5.820766e-11 | 7.275958e-12 | 精确一致 |
| Transformer32 | 0 | 0 | 0 | 精确一致 |

四个恢复更新20个字段比较均在原阈值内，未观察到新增离散或非有限失败；KV的六个非exact字段保留，
没有定位其因果，不能自动称为舍入。这里是**进程内CPU锚点恢复，不是新的磁盘checkpoint往返**。
仅首个normal更新保留五类完整SHA；CPU张量在进程内比较后未持久保存，输入不要求梯度。
既有五候选独立公式/VJP、checkpoint、新解释器及FP32/BF16超差继续保留原资格边界。

两推理候选各20次调用，两诊断完整readout SHA一致；Tide的raw values/strengths也一致，
记录的支持、输出原始int64时间与私有状态元数据一致。正常推理未保存输出SHA或逐值payload，
不能把诊断的一致性宣称为正常5次输出逐值通过。Tide64个int64私有张量中，44个命名times有完整SHA；
20个卷积时钟仅有shape/stride/dtype/device记录，因此未证明所有私有时钟值一致，更没有比较全部浮点私有state。
训练的完整递归private_state比较包含这些卷积时钟，范围不同。

本次与各自旧KV128 B8、Transformer32 B8两样本screen、中尺度KV32 B16记录的初参数SHA、输入SHA、
模型spec一致；源码/物理卡/采样顺序仍各自记录。旧screen不升级成5次正式点，旧长训练也不被本次5更新替代。
双方输入宽度不同，配对的是共同feature前缀，未声称不同模型参数/完整输入SHA相同。

四个后台`background.slice`服务均passed，26个已记录后代PID/start身份及cgroup均终态，
各冻结源461个常规文件与7196fcab一致；34份原始观察/日志文件保留SHA。
采样预算每进程RSS128GiB、NPU allocated48/reserved56GiB；服务2400/3600s、worker1800s、queue600s，
产物约0.41GiB/8GiB预算。没有预算拒绝、OOM或共享占用卡准入。
四份用户未跟踪文档原字节保留，参考/公共环境/下游未修改。

## 复现入口

完整实际提交命令、环境及CPU/设备选择在
`artifacts/breakdown-{infer,train}-{kv128,tf32}-v1/launch.json`；
原始结果为各目录`breakdown.json`。汇总与审阅脚本在
`artifacts/kv128-transformer-breakdown-20261119`，运行它们只读现有结果，不重启实验。
机器可读证据保留这些文件及脚本的SHA，并内嵌正常样本、关键诊断与终态审计。

冻结工具的worker入口如下；应经`scripts/launch.py --revision 7196fcab`、空闲NPU队列及
有限`background.slice`服务执行，输出必须换成新run目录。不能在活动源码上直接重测或覆盖原记录。

```sh
python -m benchmarks.inference_breakdown --subject tide \
  --config configs/memory-candidates/9b-kv128.json --device npu:0 \
  --batch 8 --prefill 128 --prefill-chunk 32 --decode 16 \
  --warmup 2 --samples 5 --threads 4 --output NEW-INFER-TIDE.json
python -m benchmarks.inference_breakdown --subject transformer --device npu:0 \
  --batch 8 --prefill 128 --prefill-chunk 32 --decode 16 \
  --warmup 2 --samples 5 --threads 4 --output NEW-INFER-TRANSFORMER.json
python -m benchmarks.training_breakdown --subject tide \
  --config configs/memory-candidates/medium-kv128.json --device npu:0 \
  --batch 16 --length 128 --chunk 32 --warmup 2 --samples 5 --threads 4 \
  --output NEW-TRAIN-TIDE.json
python -m benchmarks.training_breakdown --subject transformer \
  --config configs/memory-candidates/medium-kv128.json --device npu:0 \
  --batch 16 --length 128 --chunk 32 --warmup 2 --samples 5 --threads 4 \
  --output NEW-TRAIN-TRANSFORMER.json
```

本轮追加四点有限诊断完成。后续性能候选由[ROADMAP](../ROADMAP.md)记录，
9B训练、多NPU、其他精度/backend、质量验证与新batch搜索均不在这次新增证据范围。
