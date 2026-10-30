# core-5 自然维度与三客户端有限对照

数学固定0770593，测量源码`e50b6d8e90fd71879e860250baa81d294b10ee95`。
[机器记录](core5-natural-study-20261029.json)保存25个配置的身份、参数、尺寸、
原始逐窗时间/实际工作量/批统计、严格失败、恢复、内存及记录hash。另有一次独立profile。
25项中7项完整通过；流程正常结束不意味着各配置通过。设施资格见
[三个消费者](core5-study-clients-20261029.md)，后续checkpoint预算修复单列。

## 实际设置

所有本报告实测模型使用core-5 baseline KV：sum、Read⁺、local cat、region ind、
regional_total/context/cat、quad；epsilon=1e-5，发送温度1、gamma_max=2。
FFN=4d，summary/adapter/source/head隐层宽32，KV每头宽64；实际窗口按区域配置列在下表，
无RoPE/距离偏置，短衰减半寿命为一个Settle位置步长，按原始逻辑时间计算。
SrcClass按来源节点标记（内部边为source+1，外部输入为0）；SourceBinding=region另行规定目标区域内共享A/B，embedding按(region,class)保留。完整描述/状态与真实共享角色由workload/inventory给出。
各节点私有状态不共享；无任务头、填充参数或真实轨迹预跑。

<!-- core5-models:start -->
| case | nodes/regions/edges | d/heads/FFN | parameters | B/P | KV windows (logical time) | source classes / sharing |
| --- | --- | --- | --- | --- | --- | --- |
| fc-1b-kv | 64/8/448 | 1024/16/4096 | 969962344 | 1/1 | 0,5120 | source-node; external=0 / region |
| fc-medium-few-kv | 4/2/4 | 256/4/1024 | 4197862 | 2/2 | 2048 | source-node; external=0 / region |
| fc-medium-many-kv | 16/2/64 | 128/2/512 | 4283442 | 2/2 | 2048 | source-node; external=0 / region |
| hb-1b-kv | 176/22/448 | 640/10/2560 | 1051330022 | 1/1 | 0,4608 | source-node; external=0 / region |
| hb-medium-kv | 16/4/32 | 64/1/256 | 1304116 | 4/4 | 0,2560 | source-node; external=0 / region |
| lh-1b-kv | 176/22/1058 | 640/10/2560 | 1051335206 | 1/1 | 0,4608 | source-node; external=0 / region |
<!-- core5-models:end -->

表中的nodes/regions/edges是主体计数，B/P是样本数/每窗位置数；KV窗口采用原始逻辑时间，0表示不裁剪。
非中心区域的窗口为512×位置步长；FC 1B步长10、HB/LH 1B步长9，并非统一2048。
历史清单另列broadcast/sum编码的2个adapter节点计数；本次实际材料使用ports绑定的主体图。
本表由完整recipe解析生成并回归核对，源码仍为e50；2026-10-30仅更正窗口和来源绑定说明，不改原始测量或状态。
HB depth=3、branches=2，扩散/收拢区域数1,2,4,8,4,2,1；相邻偏移{0,1}。
LH-inspired增加局部偏移、跨区域与跳层连接，仍使用core-5局部函数与完整区域分配。
两个medium FC的参数量接近，节点数/维度/连接/实际工作量不同；不能跨模型称函数等价。
请求稀疏预算均为1/8，小区域取整后k=1；这不是实际活动比例或硬Top-k。

纯PyTorch、Python→native、standalone均为prefill、packing=auto→packed，ATen/vendor算子，
无专门稀疏矩阵算子。真实BMM/区域批见下文，CSR邻接本身不算数值packing。
CPU FP32或显式FP64，ATen32线程；NPU FP32/FP32 scoring，ATen8线程；interop/workers均1。
Python栈Torch/Torch-NPU2.12/CANN9.1；standalone栈Torch2.10/CANN9.0，无Python调度/运行时依赖。
NPU各mixed分工见[执行契约](../execution-flows.md)，resident由设备决定窗口内就绪和推进。

每配置一个独立进程、1个warmup训练窗、2个连续测量训练窗。SGD lr=.001、momentum=.3、
weight_decay=.01；每窗detach。时间包括输入、上传、packing、调度、完整CPU返回、loss、
反向、finite检查、SGD、通信/清理与同步；构造、warmup、独立核对和checkpoint另计。
这是短窗机制/成本测试，不是模型效果、长序列或大batch吞吐测试。

## 通过完整核对的结果

单位秒/窗，保留两个连续窗口，不能当作两个独立重复样本。

| 模型、CPU FP64 | 纯PyTorch | Python→native | standalone LibTorch | 每窗candidate / Full |
| --- | --- | --- | --- | --- |
| FC medium few | .118965 / .120979 | .085881 / .086433 | .089176 / .089635 | 16 / 8 |
| FC medium many | .373593 / .287906 | .176687 / .178245 | .190613 / .191778 | 64 / 8 |
| LH-inspired 1.051B | 未测 | 1.439185 / 1.192557 | 未测 | 115/27，114/22 |

七项均通过完整返回/支持/状态、最后参数/梯度/SGD槽、独立调度核对与磁盘换owner精确下一窗恢复。
在这两个medium模型内部，native调度均较快；不能由此断言任意节点数或大矩阵下总是更快。
更多节点保留了相近参数量，却把候选更新从16增到64，所以“参数相同”不能解释全部成本。
LH 1B实际消息155/133、Full只占候选的27/115与22/114；候选更新仍须计算。
该B1/P1原始窗最大region/state批为1，私有投影批最大8，不能称为大batch吞吐。

## 严格失败与停止判定

以下时间仅为失败候选已经发生的成本，**不是合格吞吐或等价加速**。

| 配置 | 两窗秒数或失败阶段 | 严格结果 |
| --- | --- | --- |
| FC few FP32：Python / native / C++ | .127/.128；.093/.093；.095/.095 | 完整KV keys失败 |
| FC many FP32：Python / native / C++ | .413/.334；.223/.225；.227/.230 | 完整KV keys失败 |
| HB medium CPU Python / native | .796/.794；.548/.758 | 完整KV keys失败 |
| HB medium NPU native mixed-a / b / c | 1.298/1.270；1.793/1.755；1.884/1.735 | 完整KV keys失败 |
| HB medium NPU Python mixed-c | 2.131/1.958 | 完整KV values失败 |
| FC 1B CPU FP64 | .764963/.803085 | SGD动量2/4,194,304坐标失败，最大绝对差2.963247e-10 |
| HB 1B CPU FP64 | .808811/1.075661 | 最后梯度59/1,638,400坐标失败，最大绝对差3.525837e-10 |
| FC 1B standalone NPU mixed-c | 1.572019/1.493245 | 验证准备时native checkpoint超过旧4GiB固定上限 |
| HB medium resident，两客户端 | 构造1.357/1.416s，warmup反向提交失败 | CANN507002：stream task buffer full |
| FC 1B standalone resident | 构造拒绝，无测量窗 | KV Full张量分项预算拒绝 |

FP32保持atol1e-6/rtol1e-5，FP64保持atol1e-10/rtol1e-8；没有改变输入、路由或容差。
FC/HB 1B FP64的完整记录分别通过但训练尾部失败，自身恢复精确；微小差异也必须保留严格失败。
原D64/D2048/反馈数值候选的停止判定仍有效，只有覆盖整个算术与伴随累加的新方案才重启。

HB resident两次CANN原始短日志已复制到项目artifacts并记录hash：EE1019/pending tasks超过限制，
发生在local service wait/reset提交；不是此前207005/EE1023的流数量限制，也没有显存OOM证据。
失败前构造peak allocated约420.5MB。需有界服务提交/复用或更粗的节点批，先验嵌套通知、
workspace、独立VJP根和恢复；不通过主机预选轨迹或缩短原输入冒充修复。

FC resident显式tensor/retained/backward/optimizer各12GiB、operator4GiB、total48GiB，
queue1024、stages64、cache32。源码KV Full前置公式在B1/d1024/cache32需要约3.499GB的
当前剩余额度，之后还检查实际保留量；失败节点的剩余额度未导出。
这是保守分项准入，不能推断总HBM耗尽或实际Full峰值就是3.499GB。
下一步须改善/核对局部预算与workspace生命周期，不能盲目提高总预算反复OOM搜索。

## 区域合批与独立profile

HB medium native mixed-c每窗160候选、64 Full、128消息；64个完整region frame通过16次调用，
最大region batch=4，state batch=4，Full batch=16。并未因接口叫prefill而假设所有节点都能合批。

单独profile同输入/同实现，事后仍严格失败；仅用于成本诊断：

| 观测 | 值与解释 |
| --- | --- |
| 真实厂商矩阵算子 | 1438个MatMul、384个BatchMatMul事件 |
| 所有设备kernel | 64,295；设备时间区间并集186.281ms，不是端到端关键路径归因 |
| 标量/传输事件 | 2680个`_local_scalar_dense`、5041个MEMCPY_ASYNC |
| inclusive host阶段 | forward1098.08ms；backward+finite1115.61ms；detach/return/sync246.28ms；SGD6.03ms |
| 相关反向范围 | BatchedQueryVjp约696.38ms；AllocationBatchVjp约21.08ms，均inclusive且可重叠 |

这证明区域分配可以批量运行，并不证明整个模型受分配求解主导。此次更明显的候选方向是
query反向的索引/装配、小算子与同步；需新的独立实现/根测试后才能主张优化收益。
region score/context、quad求解和反向均有实际设备算子；mixed仍允许host标量控制，不能叫resident。
profile采集额外成本不进入上述吞吐表；inclusive事件不能相加来声称独占占比。

## 较大规模与复现

静态计划生成HB 8.118B（d1792）、16.528B（d2560），均176节点/22区域；没有运行吞吐。
仅FP32参数加旧兼容固定buffer下界约34.741GB/70.735GB，16B下界已大于单64GiB卡。
梯度、master、槽、原始状态和反向另加；计划memory_floor中的AdamW是容量估算口径，
不能当作本次SGD测得峰值。rich混合约1.091B；1/32稀疏、704节点的128步长最近方案
实际0.690B（偏离目标约−31%），不隐瞒维度粒度造成的偏差。
1B数值/服务资源尚未取得同配置完整通过，停止继续放大相同失败候选；不把静态计划当运行资格。

运行目录：`artifacts/jobs/core5-study-v2-performance`及`core5-study-v2-supplement`。
各自独立background.slice，32核预算，512/128GiB，7200/1200秒；测量串行，构建/资格在前。
队列选择空闲单卡；其他用户占用设备及共享CPU/内存争用未排除，没有停止其他任务。
新checkout不依赖这些目录：[目录](../../benchmarks/core5-study-v1.json)、公共构造器和
[入口说明](../core5-study.md)可重新生成相同模型与输入。

```sh
git checkout e50b6d8e90fd71879e860250baa81d294b10ee95
# 按development和三个消费者资格报告构建匹配库；长任务使用本机持久服务。
python scripts/run_core5_study.py --case fc-medium-many-kv --execution native-cpu64 \
  --native-library build/cpu --max-materialize-bytes 34359738368 --output-dir NEW_RUN
python scripts/run_core5_study.py --case lh-1b-kv --execution native-cpu64 \
  --native-library build/cpu --max-materialize-bytes 34359738368 --output-dir NEW_LH_RUN
python scripts/index_core5_study.py RUN_PARENT --format tsv
python -S scripts/prepare_core5_study.py --topology hb --width 64 --region-width 4 \
  --depth 1 --samples 4 --positions 4 --output-dir NEW_PROFILE_PLAN
python scripts/benchmark_weighted.py --workload NEW_PROFILE_PLAN/workload.json \
  --device npu:0 --implementation native --native-library NPU_CORE --placement mixed-c \
  --dtype float32 --scoring-dtype float32 --intraop-threads 8 --profile --mode training --output-dir NEW_PROFILE
```

profile的目录必须单独创建；不能用profile时间替代正式测量。新记录仍必须经过完整核对才能标为通过。
