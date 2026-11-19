# Reception、SSD 与完整训练成本 · 2026-11-18

FP32 是本轮主力。SSD doubling 在 compact-9 完整训练将吞吐提高约19.25%、allocated峰提高1.47%；
Light64 六候选中融合QKV+结构mask+doubling观察到最高吞吐。接收微测FP32 48/48在原阈值内，
BF16保留21个有限超差。0.5B在本B/L下快于极深窄A、慢于宽浅C，具体计算、packing和实际设备事件如下。
原始样本、资源/源码/输入身份及设备摘要见[精简JSON](reception-performance-20261118.json.gz)。

快照：2026-11-18T12:45:17.332519+00:00。固定数学`4f5b9f20a4030d68ee66ae2d26e34659a5e37234`。
本页是测量证据，不修改[精度](../precision.md)、[验证](../verification.md)或
[A–D联合匹配](../benchmark-contract.md)契约。测量提交保留各自身份，不能用整理HEAD替代。

## SSD 基元：直接递推与块乘积

单头矩阵状态与读出为

\[
M_t=a_t M_{t-1}+u_t B_t^\top,\qquad r_t=M_t C_t.
\]

在一个块内，令\(P_{ij}=\prod_{k=j+1}^i a_k\)、\(P_i=\prod_{k=0}^i a_k\)，则

\[
r_i=P_i M_{\rm in} C_i+\sum_{j\le i}P_{ij}u_j(B_j^\top C_i).
\]

这是带真实矩阵状态的SSD递推，与接收边self-attention和历史KV Attention是不同计算。
旧block对32×32局部传播乘积使用cumprod；doubling用有界乘法树重排同一乘积，不除以decay，
保留零decay的导数。L=1直接更新矩阵并读出，省去pairwise tile；incremental只改变L=1，
L>1回到block，doubling则同时提供单步直接式和长块乘积树。块边界仍保留矩阵状态和完整BPTT。

N8/B8/H4、value/state width32、chunk32，FP32 resident冻结输入与cotangent，3次热身/9次计时。
`backward`计入前向和五个输入根的VJP，不含optimizer；返回图/结果销毁与同步计入。

| L / 模式 | block ms | incremental ms | doubling ms | block→doubling allocated 峰 MiB |
|---|---:|---:|---:|---:|
| 1 / inference | 0.5998 | 0.1880 | 0.1784 | 5.220→5.087 |
| 1 / grad_forward | 0.7310 | 0.2397 | 0.2355 | 5.256→5.087 |
| 1 / backward | 1.5536 | 0.7746 | 0.7611 | 6.229→6.218 |
| 256 / inference | 32.3159 | 32.4103 | 5.0262 | 53.305→54.306 |
| 256 / grad_forward | 33.1077 | 33.3118 | 6.2666 | 118.537→143.548 |
| 256 / backward | 85.4902 | 85.2736 | 17.3848 | 140.729→144.790 |

L1在NPU7，L256在NPU1，两组不作跨卡速度比。L1两候选是同一个直接式，微小时间差不代表
不同算法。L256 doubling相对block：inference约6.43倍、含图前向5.28倍、前向+VJP4.92倍；
含图前向allocated峰多约25MiB，前向+VJP峰多约4.06MiB。局部倍数不能直接当完整训练倍数。
原阈值atol=rtol=3e-5下，全尺寸同fixture输出/末态/五根梯度以及有界CPU literal FP64参照均通过；
FP64参照长度为L1和L65的小形状，不冒充全尺寸FP64实测。

primitive记录源码commit为`57eb12c3`，且保留当时dirty名单，不能声称完整clean提交身份。
冻结`src/tide/ssd.py` SHA256为`42e0e9c2de11f7c266af93c7bf43bf3e53e63efd6ed1eb5b86ef18aa1c982af3`；
JSON保留实现文件hash、输入和cotangent hash及实测FP32算子dtype。此处不推断芯片内部累加精度。

## 1. 小规模完整训练：收益与代价

共同负载为 compact-9 HB Settle、n8/hubs4、d128、X memory、region_shared，B8/L256，
FP32、region batch4、Reception checkpoint、SSD prefix，2 次热身后 5 次完整 AdamW 更新。
每次更新从零状态 prefill；这是顺序参数更新，不是带缓存历史的连续训练。普通计时包括
packing/forward/loss/backward/AdamW/sync，排除构造、输入传输、审计、anchor IO 和 checkpoint IO。

| 对象 | 参数数 | 中位秒/更新 | 输入位置/秒 | allocated / reserved 峰 GiB | range/median |
|---|---:|---:|---:|---:|---:|
| source | 15,694,224 | 1.368260 | 1496.79 | 12.376 / 14.344 | 2.10% |
| doubling | 15,694,224 | 1.147342 | 1785.00 | 12.558 / 14.436 | 0.97% |
| light64 | 17,880,464 | 1.844363 | 1110.41 | 12.225 / 14.184 | 5.96% |

source 与 doubling 在同一物理 NPU1，源码分别 `95465188`、`b4cf1514`；Light64 在 NPU7，
源码 `95465188`。准入身份均为 `exclusive_cooperative_queue; external jobs not controlled`。
不能称整机独占。源码/环境/设备完整值保留在 JSON；HF32 关闭。

doubling 普通时间减少 **16.15%**、吞吐增加 **19.25%**；allocated 峰增加
194,885,120 B，即 185.86 MiB / 1.47%。初始参数、输入、初始工作量都相同；各自热身后
normal-start 的参数、optimizer、gradient hash 已不同，RNG hash 相同。第一步诊断工作摘要
相同；末端正消息差 -32、Full 动作差 -5、KV 有效 pair 差 -1180。没有把这些端点摘要称为
全步骤等算量、全支持相同或数值资格。收益不是通过放宽阈值得来。

### 侵入式阶段和设备 trace

下表来自各自第一步 anchor 的单次同步阶段诊断。包含逐 scope 同步开销；
不能代替上表正常吞吐。父/子 inclusive 时间不可相加。

| 对象 | 整次更新 s | 前向 s | 反向 s | AdamW s |
|---|---:|---:|---:|---:|
| source | 1.450307 | 0.847381 | 0.578968 | 0.013347 |
| doubling | 1.269409 | 0.635662 | 0.609266 | 0.013322 |
| light64 | 1.929523 | 0.943211 | 0.960404 | 0.013948 |

source→doubling 的前向 exclusive：SSD matrix **252.224→52.275 ms**，占这次侵入式
前向的 **29.77%→8.22%**；prefix 已在两边启用，为 10.200→10.093 ms。
分配 104.097→93.339 ms，gather 70.087→70.150 ms，query packing 57.523→55.830 ms，
所有 attention site 合计 69.272→69.519 ms。反向 578.968→609.266 ms，确有增加。

完整 runtime 审计中，旧 SSD 的 forward/backward 各有 128 次 `aten.cumprod`，doubling 为 0；
乘法计数 forward 1839→2059、backward 2567→2951。外层保存 hook 的峰 unique storage
12,331,957,617→12,520,701,297 B，与实际 allocator 峰是不同量。hook 不保证覆盖嵌套 hook
及后端 opaque 保存，也不把重复 saved payload 当 unique memory。

独立 profiler 记录的 raw `aclnnCumprod_Cumprod_Cumprod` 旧路径 128 次，累计 inclusive
duration 233.062 ms；新 trace 的 CPU_op 207,880→223,364，enqueue 37,943→40,177。
trace 整体 15.929→18.073 s，远慢于正常计时；这不推翻普通计时收益，也不能用其总时长
计算正常占比。`IndexBackward0` 两边各 1153 次、136.448/136.376 ms；FlashAttentionScoreV2
38 次、11.589/11.762 ms，GradV2 38 次、35.539/35.791 ms。完整trace的process_name与Task Type复核见下文；top50缺席本身不是零调用的证明。

Light64 是另一数学实例，参数 15.694M→17.880M，支持/动作及设备也变了，不能把总时间
差称为同函数的实现开销。其 Reception 前向 inclusive 145.882 ms（自身 exclusive
71.809 ms，attention 子项 74.074 ms），原 source 为 33.614 ms；反向 Reception 重算
347.803 ms，原 source 为 122.635 ms。这描述新算子的成本，不表示该语义应被取消。

## 2. 与 Transformer 的具体差别

以下 A/C 只来自 source 成本任务的相同 B8/L256 FP32 正常训练；该成本任务没有测 B/D，
不能替代完整 A–D 联合匹配主表。A 188 blocks/d80/FFN240/head-dim16，15.672M；
C 9 blocks/d368/FFN1088/head-dim16，15.693M。窗口都为128。

| 对象 | 参数数 | 中位秒/更新 | 输入位置/秒 | allocated / reserved 峰 GiB | range/median |
|---|---:|---:|---:|---:|---:|
| transformer_A/standard | 15,671,760 | 0.759582 | 2696.22 | 3.192 / 3.373 | 2.95% |
| transformer_A/chunk256 | 15,671,760 | 0.584350 | 3504.75 | 3.148 / 3.357 | 0.83% |
| transformer_C/standard | 15,692,624 | 0.040217 | 50924.33 | 0.868 / 0.982 | 21.52% |
| transformer_C/chunk256 | 15,692,624 | 0.037005 | 55343.96 | 0.882 / 0.982 | 0.54% |

source 的第一步 anchor 实际 NodeLinear 前向含 padding/adapter 的乘加对为 **123.319G**；
A 线性层为 **32.034G**。Tide 该计数不含 attention、normalization、非 NodeLinear 运算，
Transformer 该计数也不含 attention、norm/RoPE/activation，不能宣称完整 FLOPs 对齐。
它至少说明参数接近并不保证物理线性工作接近。source 的 K query 选择147,804行，执行
205,312行，执行/选择约1.389；消息正条目1,949,957，而物理 message slots2,850,816。

Tide 前向有 1606 次 `aten.index`、641 次 bmm，反向有1153次 `index_put`；
A 没有对应的动态节点索引链，前向为752次 matmul（188块各四类矩阵投影）及376次
融合 attention，chunk256 则 attention 减为188次。其 Q/K/V 形状主要是
[8,5,128,16]、[8,5,255,16]；chunk256 为[8,5,256,16]。Tide 的区域/节点分组、可变
query packing、独立参数 bmm、SSD 32×32 block 和接收/分配有各自的小算子与索引链。
两者功能和图不同，不能由某个 bmm/attention 个数直接推断总效率。

A chunk256 的 attention operand rectangle 反而从368.660M增至492.831M pair，
普通更新却从0.759582降至0.584350s：减少调用与改善形状可以比减少矩形 pair 更有利。
C 的宽矩阵只串行9层，runtime dispatched calls 2170（chunk256为1774），
A为44593（chunk256为36321），Tide source为57321、doubling为56867。
这些是对应审计观察的 dispatch 数，不等同设备 kernel 数，也不应与 profiler 事件数混加。

## 3. Light64 执行候选保留价值

独立 serial 子进程、相同 NPU7、相同初始参数/输入/工作摘要，源码 `b4cf1514`。
固定候选顺序且每项仅5次，不消除时间/主机争用影响；后续活动与工作摘要略有不同。
这一组 baseline 与上一成本任务 Light64 是不同计时记录，不能混为同一次测量。

| Light64 执行候选 | 中位 s | 输入位置/s | allocated / reserved GiB | 吞吐相对 baseline |
|---|---:|---:|---:|---:|
| baseline | 1.905039 | 1075.04 | 12.184 / 13.838 | +0.00% |
| qkv | 1.869320 | 1095.59 | 12.171 / 13.457 | +1.91% |
| mask | 1.863466 | 1099.03 | 12.184 / 13.838 | +2.23% |
| qkv_mask | 1.833259 | 1117.14 | 12.171 / 13.457 | +3.92% |
| qkv_mask_nocheckpoint | 1.709407 | 1198.08 | 22.244 / 23.080 | +11.44% |
| qkv_mask_scan | 1.683336 | 1216.63 | 12.347 / 13.902 | +13.17% |

qkv_mask_scan（融合QKV+结构mask+doubling）是本组最快观察值，中位时间下降11.64%，
allocated峰仅增1.34%。取消 Reception checkpoint 也有收益，但显存峰增82.57%；与
qkv_mask 本身比较，取消checkpoint时间下降6.76%、峰增约82.77%。因此重算候选与
不重算候选都值得保留，是否选默认要同时满足吞吐、内存余量和数值资格；不能一概关闭。

## 4. 接收 self-attention：局部真实算子与性能

N8/B8/L128/E8/d128，shared，warmup2+steps5，无Reception重算。普通训练计时含AdamW；
输入和 routed mass resident。吞吐单位为 receiver-node×sample×position/s，不能当主干token/s。
算子审计是另一次 post-timing 前向/反向/更新，并不覆盖所有 timed step。

| FP32 profile / case / mode | separate ms | fused ms | masked ms | fused_masked ms |
|---|---:|---:|---:|---:|
| light_64/padded/inference | 4.129 | 4.108 | 4.159 | 4.155 |
| light_64/padded/training | 11.412 | 11.151 | 13.214 | 12.757 |
| light_64/zero/inference | 4.061 | 4.098 | 4.116 | 4.165 |
| light_64/zero/training | 11.399 | 11.125 | 11.599 | 11.310 |
| light_128/padded/inference | 4.826 | 5.067 | 4.844 | 5.065 |
| light_128/padded/training | 15.621 | 15.610 | 19.440 | 19.107 |
| light_128/zero/inference | 4.831 | 5.047 | 4.841 | 5.078 |
| light_128/zero/training | 15.579 | 15.595 | 15.700 | 15.581 |
| wide/padded/inference | 4.968 | 5.183 | 4.991 | 5.179 |
| wide/padded/training | 16.017 | 15.932 | 19.682 | 19.589 |
| wide/zero/inference | 4.966 | 5.248 | 5.124 | 5.225 |
| wide/zero/training | 16.030 | 15.921 | 16.283 | 16.019 |

FP32 在NPU1、源码`d635a982`，**48/48 within_threshold**，原阈值atol1e-6/rtol.002。
padded training 的separate/fused实际走tiled；masked/fused_masked能走NPU融合Attention，
却更慢。light64 separate→masked为11.412→13.214ms、峰371.37→435.69MiB；
light128为15.621→19.440ms、510.03→708.35MiB；wide为16.017→19.682ms、
510.70→709.01MiB。融合路径不是普适收益。light64的native QKV在推理为
[8192,2,8,32]、FP32、带PSE；结构mask训练为[8192,2,8,48]、FP32、无PSE。
真实零质量case训练保留tiled。结构padding与真实零根没有被混为同一个删除规则。

QKV合并的局部算子证据：light64训练bmm36→24、mm18→12；推理bmm12→8、mm6→4。
FP32 light64训练有约2.3–2.4%的观察收益，light128近乎持平，wide约0.5–0.7%；
宽profile推理反而下降。端到端合并收益与局部时间不能线性相加。

BF16在NPU4、源码`f3e6661c`，完整48行，**27 within / 21 finite_deviation**；
exit1来自脚本对非within行的返回条件，并非任务异常中断。21个失败全部是
`root.result.1`（Agg输出）；全有限，metadata/support mismatch均0，其余被比较的梯度、
参数、optimizer条目没有超差。原阈值atol.002/rtol.05保留，最大scaled error8.9582。
每个profile失败模式一致：padded inference masked/fused_masked；padded training
fused/masked/fused_masked；zero training fused/fused_masked。不能把这些结果列为已合格
默认候选，也不能仅由其有限性断言全是正常舍入。BF16个别融合候选有较大速度/显存收益，
但本页不以未通过数值项作取优依据；FP32/BF16卡与源码不同也不作跨dtype速度排名。

| BF16 padded training | separate ms / 峰MiB | fused_masked ms / 峰MiB | 数值状态 |
|---|---:|---:|---|
| incoming_light_64 | 16.009 / 414.24 | 11.531 / 353.88 | finite_deviation |
| incoming_light_128 | 21.587 / 654.04 | 14.216 / 467.38 | finite_deviation |
| incoming_wide | 22.329 / 654.70 | 14.792 / 498.04 | finite_deviation |

上述保留了潜在收益候选，但超差结论不变；完整96行数值与时间在JSON中。

## 5. 0.5B 成本：终态通过

状态快照：`{"state": "passed", "exit_code": 0, "elapsed": 609.1864217109978, "finished": "2026-11-18T12:43:05.122453+00:00", "peak_rss_bytes": 92554670080}`。源码`f3e6661c1b126bdddc66167d4a79c6471acee285`，NPU7，
CPU128–131，FP32 B1/L128，n32/profile9/independent、region batch4、Reception checkpoint、
prefix+doubling。A联合匹配为1712×d128/FFN584，496.562M；C为9×d2032/FFN6296，
494.107M。B/D构造也在记录中，但此成本任务不测B/D。三对象正常计时、阶段/runtime/work/trace均终态completed_finite。

| 对象 | 参数数 | 中位秒/更新 | 输入位置/秒 | allocated / reserved 峰 GiB | range/median |
|---|---:|---:|---:|---:|---:|
| tide | 494,110,704 | 1.343725 | 95.26 | 14.564 / 17.736 | 7.66% |
| transformer_A | 496,562,304 | 12.421117 | 10.31 | 9.274 / 11.324 | 20.57% |
| transformer_C | 494,107,216 | 0.093909 | 1363.03 | 9.257 / 11.361 | 6.29% |

独立阶段诊断（intrusive）：

| 对象 | 整次更新 s | 前向 s | 反向 s | AdamW s |
|---|---:|---:|---:|---:|
| tide | 1.548469 | 0.777330 | 0.712564 | 0.045185 |
| transformer_A | 11.938711 | 5.887328 | 5.458480 | 0.512410 |
| transformer_C | 0.107388 | 0.040327 | 0.030680 | 0.035126 |

Tide初始总参数494.111M，实际触及均值145.753M；第一步anchor为141.547M，
不是用k/n静态代理代替实际活跃参数。anchor K packing选择18,322行、执行96,160行，
约5.248倍；正消息328,220，物理槽1,714,176。Tide前向exclusive allocation215.266ms
（该侵入式前向27.69%），gather69.760ms、packing64.506ms、attention64.014ms、
SSD matrix36.554ms。SSD已优化后，分配和稀疏到矩形的packing成为更明确的后续方向。

该anchor Tide NodeLinear前向45.875G乘加对；A线性层63.504G，且A需要1712层串行。
A审计前向6848次matmul、1712次融合Attention，Q/K/V为[1,4,128,32]；
Tide前向448次bmm、13次native Attention且另有tiled，1839次index，反向1126次index_put。
Tide/A吞吐比约9.24，C/Tide约14.31。这次大模型Tide胜A、小模型Tide负A并不矛盾：物理工作、层数和batch形状都变了。
A普通5样本range/median20.57%，保留抖动，不将单次比值泛化为稳健硬件上限。

## 实际设备轨迹：双边证据

按完整trace的`process_name=Ascend Hardware`识别硬件，再按`Task Type`以`KERNEL`开头识别kernel。
逐文件SHA256已核对原成本记录；没有把全部无cat事件（含Overlap Analysis）当设备kernel。

| 实际trace | CPU_op次数 / inclusive总s | 硬件任务 / union s | KERNEL任务 / union s | 硬件首末跨度s |
|---|---:|---:|---:|---:|
| small source | 207,880 / 3.900288 | 54,803 / 1.157663 | 54,256 / 1.156837 | 1.721982 |
| small doubling | 223,364 / 3.284410 | 58,951 / 0.931571 | 58,404 / 0.930629 | 1.542175 |
| small A | 207,174 / 2.344904 | 38,820 / 0.415785 | 37,316 / 0.414145 | 1.174978 |
| small A chunk256 | 159,046 / 1.902069 | 31,301 / 0.361887 | 30,548 / 0.360870 | 0.945177 |
| 0.5B Tide | 269,223 / 4.714708 | 74,688 / 0.898239 | 74,234 / 0.897864 | 2.192582 |
| 0.5B A | 1,450,816 / 32.556537 | 279,718 / 1.786943 | 267,729 / 1.778240 | 14.960335 |
| 0.5B C | 7,744 / 0.202788 | 1,494 / 0.075734 | 1,458 / 0.075692 | 0.123488 |

union是这次带profiler观察的时间戳区间并集，不是正常运行利用率；CPU事件嵌套、设备事件可重叠，
各列不相加，也不除以普通更新中位数制造“正常耗时占比”。trace外层elapsed还含导出/分析。

small source的Cumprod确为128次`KERNEL_AICPU`、duration总233.062ms；这是Ascend设备任务，
不是host CPU fallback。doubling完整硬件任务表中Cumprod为0；kernel union约1.157→0.931s，
同时kernel次数54256→58404。更多小任务仍可以换来更低昂贵算子成本；正常训练收益以上表为准。

0.5B Tide设备任务可见Add 5654次/112.702ms、Mul 5087/61.248ms、Index 2022/51.317ms，
BatchMatMul 1376/25.256ms。A则MatMul 20544/118.773ms，另有StridedSlice 23968/262.042ms、
Mul 59924/249.302ms、Transpose 15408/185.180ms，AttentionGrad 1712/66.899ms。
C只有108次MatMul/18.541ms，且optimizer设备任务成为显眼部分。
这些duration是各任务名的和，仍非可加的完整墙钟分摊。

0.5B A的硬件首末跨度14.960s而kernel union1.778s，结合1712层和1.45M个host CPU_op，
支持它在本B1形状有显著串行launch/调度间隙，不能把其12.421s普通时间全部解释为Attention算术。
这不是对空隙逐段归因的证明，也不是Transformer全局最佳实现声明。C宽浅结构以更大矩阵、更少层
避开这一重复调度，仍是更高吞吐参照。Tide更少层/相容区域合批的收益和自身分配/索引成本同时存在。

## 对剩余成本的解释与有限后续方向

接收的gather把不同源状态按边/接收者排成矩形，保留真实质量和结构support；对共享源的反向需要
索引梯度归并。packing把实际Full时刻按owner/相容组装成矩形，组内短行补齐，因此选择行数和
执行行数不同。0.5B约5.248倍的K padding比小模型约1.389倍更值得检查：缩小或按有效行数分桶
可降低padding，但会增加矩阵/launch数。本页没有未测候选的加速承诺。

quad allocation对收到质量给定上下界与总质量约束，求分段线性的
\(c_i=\operatorname{clip}(s_i-\lambda,\ell_i,u_i)\)。当前实现排序断点、前缀sum、判定所处面，
必要时使用受保护的二分回退，并为边界导数保存面信息；不是一层普通softmax。0.5B阶段里的
215ms allocation是整站点同步观察，尚没有逐次记录fallback是否发生，不能全部归给sort或fallback。
可以有界研究减轻分配小算子/同步、减少矩形padding及合并索引梯度；这些改动需要保留质量约束、
零质量导数、原时刻KV可见性和私有状态。增加区域并发本身不能消除这些数学/布局工作。

## 6. 可直接引用的边界

本证据支持“有限候选实测、完整训练边界、对应anchor成本与形状分析”，不支持训练全程
等轨迹、普适最优、全尺度资格或任务质量。正常计时、同步阶段、runtime hook、saved hook、
profiler各自单独测量，不相加。大模型首token/持续inference、正式多步延续和恢复的结论
应引用各自专门记录；这些成本任务每步重置模型状态，不能替代那些记录。

## 复现入口与原记录

冻结源码和完整命令由各作业的`launch.json`、`run.sh`、`status.json`保存；不要以当前HEAD替换冻结源码复测。
示例均需按[执行契约](../execution-flows.md)先完成配额/任务/设备准入；下面列的是入口，不请求自动启动。

- `python -m benchmarks.ssd_scan_candidates`：`agg-ssd-single-fp32-v1/scan.json`与`agg-ssd-scan-fp32-v1/scan.json`。
- `python -m benchmarks.costs`：`agg-cost-source-fp32-v1`、`agg-cost-doubling-fp32-v1`、`agg-cost-light64-fp32-v1`、`agg-cost-half-fp32-v1`的`costs.json`。
- `python -m benchmarks.variants`：`agg-light64-variants-fp32-v1/variants/variants.json`及六份独立normal记录。
- `python -m benchmarks.reception`：`agg-reception-sites-fp32-v1/reception.json`和`agg-reception-sites-bf16-v1/reception.json`。

这些路径均位于仓库`artifacts/`；相对路径、字节数和SHA256完整列于配套JSON。
本页数据归纳脚本保留在`artifacts/worktrees/agg-inference-ssd/artifacts/`，包括
`agg-cost-reception-analysis.py`、`agg-trace-readonly.py`和`build-reception-performance-evidence.py`；
其行为仅只读解析既有记录。全部实际trace保留，未清理复现依赖。
