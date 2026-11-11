# KV 查询块：默认接入、三层复测与可重复诊断（2026-11-11）

隐藏 KV 前缀现在以共享写入 bank、B×L 查询列和块伴随计算，已进入 Python/native 默认能力分派。
FC16 完整训练相对同轮显式 packed-v2：CPU 11.601→5.835 秒，NPU 15.124→10.928 秒。
**有明确收益，但仍未接近直接 FC：同配置分别需要 1.390/0.675 秒。**
这次采用查询块接口，不把整个 FC 性能问题称为解决，也不继续无限重复同一候选。

[可查询配对、profile、数值检查及任务终态](kv-query-blocks-20261111.json)与
[资格及复现身份](kv-query-blocks-qualification-20261111.json)保存小型审阅材料。
前置调查保持[原身份和失败](fc-three-layer-comparison-20261111.md)。

## 源码、职责与资源

| 范围 | 身份 |
| --- | --- |
| 数学 | `0770593838a37c790a568226930567ed60f99b83`；19 份 core-5 快照通过离线核对 |
| 主实现、CPU/NPU/安装/SDK 资格 | `5ff3ce2bfe72b4416814b750d5260f5c90038eca` |
| 数值不变的范围标记及诊断修正、计时/profile | `598b4fb5f487befe008f50afb1fe1e378504c417`；前序 `26ab27d` 宏重名编译失败保留 |
| 后备路径先借用 State 预检，避免无用字段复制；通用诊断收集 | `206bc9b33be15b891598997bf12a8cec5e6f337b`；新 native/SDK 构建及定向补验 |
| 诊断收集器差异标记 | `19af25eb9c3f1698813b609eaa48775390cb312b`；单独展示physical visibility/affinity/placement等配对差异，3项定向检查通过；无生产计算变动 |
| 固定下游材料 | `tide-learning-lab@0439efd2af31cf10b07ce9e6423b0195f7449968`；原 FC16 配置及 task/head/loss/AdamW，未写入下游 |
| 计时客户端 | Tide 主表：Python 调用 C++/LibTorch；直接 FC、Transformer：纯 PyTorch。另测纯 PyTorch Tide 训练 |
| 放置/精度 | CPU FP32；NPU FP32 mixed-c，host 推进。无 autocast，matmul HF32 关闭；没有 resident/GPU/多卡性能结论 |
| 环境 | 本机 aarch64；Torch2.12.0+cpu / TorchNPU2.12.0.post3 / CANN9.2；项目私有环境与独立 SDK 入口 |
| 资源 | 构建全部 320 可用 CPU，512GiB/1800秒以内；主计时 CPU80–111/ATen1，NPU host0–15/ATen2、空闲 physical11→npu:0；各128GiB/1800秒以内 |

最终 NPU 补验的实际映射以资格 JSON 的队列记录为准。CPU32 对照使用160–191池、ATen32。
计时期间项目内无重型构建；CPU/NPU 使用不同 CPU 池。外部 CPU/主存争用未排除。
没有共享 NPU 准入；没有磁盘阻塞或因配额拒绝并行。账户约153GiB/软300/硬800，文件系统约944GiB可用、shm约398GiB可用。
源码/profile 暂存项目独占 shm，小型命令、结果、终态持久保存；shm 仍计入主存且不作持久证据。
所有长任务在独立 background.slice、固定源码、有限时长中执行，原编译/启动失败与取消均保留。

## 实现了什么，为什么不是删掉数学职责

[块接口契约](../weighted-query-blocks.md)及[总对应](../semantic-map.md)是维护入口。
每个 owner 的原始左态和实际写入只拼接一次；query 携带 int64 原时间、prefix end 和可见范围。
Read 的短/长通道、Full 的 Attention/FFN、Describe 的评分/证据保持原公式。
块反向直接累计到 bank/write/query 列，不再逐查询构造每个历史前缀的 K/V 伴随，再交由序列 VJP 收集。
Attention 反向用矩阵乘法实现 `dV=Pᵀ·dY`、`dK=dlogitᵀ·Q/√d` 等，避免展开 `[B,H,Q,K,D]`；
可求导消息质量、衰减、attenuation 与其梯度不被 bool mask 或 detach 替代。

适用条件由实际能力判断：隐藏 prepared KV 前缀、FP32/FP64、无位置项/未过期、strict 支持、至少两条查询，
32MiB Attention 工作区估计。sum/qry、Read+/Read-、不同 owner 和 ragged 查询均有实现。
Read0 使用绑定初态；公开末态、显式 trace、过期/位置/异构根/超工作区等继续原合法路径。
这些是块优化的适用条件，不是删去通用图能力。PDG/TimedDAG/SettleGraph 的调度域保持。

内部 provenance 不进入 checkpoint；移动/重绑实际 slot tensor 后失效。公共末态和中间独立根不合并，
未来写入梯度保持 None，连接的零仍为零，参数别名、版本守卫及一阶限制保留。
图v13/checkpoint v5不变，执行策略身份明确含 `kv-block-v1+packed-v2+describe-batch-v2`。
老安装消费者仍需显式升级；源码交付不等于下游验收。

## 有限资格与数值边界

5ff3ce2 固定 CPU：块/根66通过；状态/训练83通过、1旧xfail；查询/诊断221通过、12旧xfail；
通用图173通过、4旧xfail、2个 standalone 可执行路径误指失败。
后两项用实际 SDK 二进制的 FP64/FP32 Settle 检查单独通过；原2失败未改写。
合计543通过/17旧xfail/2启动失败，仅是这四个定向组，不是全库资格。
NPU 流程32通过/1旧xfail，包含设备测试及 CPU 辅助，不宣称33项全部运行在NPU。

Python/native 安装消费者在 CPU FP64、NPU FP32 分别独立三步优化、同进程和新进程精确恢复通过；
独立 SDK 的 complete/outputs 两种返回、CPU/NPU 训练与新进程恢复通过。
598b4fb 的范围标记/块检查12项通过。206bc9b 新构建的 CPU 定向162项通过，
standalone SDK Settle FP32/FP64通过；最终NPU流程11项通过（6项默认图在NPU，5项CPU辅助），新native安装消费者三步/新进程恢复通过，physical1→npu:0。
这些组有重叠，不累加为更大的独立覆盖。

独立证据分工：literal prefix/formula 检查局部数值及一阶导数；独立 reference 调度检查依赖；
单独字段、参数共享、冻结根、别名、零伴随、原版本变更、极大int64时间、保留cut、末状态根与未来None检查连接；
实际安装消费者/SDK 的多步更新与精确下一窗检查训练和恢复。

| 指定任务投影 | 值/末状态 | 参数和输入梯度 |
| --- | --- | --- |
| CPU FC4 B2/L8 FP64，native及Python | 原阈值通过，最大约6.04e-14 | 通过，最大约1.14e-13 |
| NPU FC4 B2/L8 FP32，native及Python | 严格失败，最大1.48e-5 | 严格失败，最大4.20e-5 |
| CPU FC16 B8/L128 FP32，native | 严格失败，最大0.02388 | 严格失败，最大0.43535 |
| NPU FC16 B8/L128 FP32，native | 严格失败，最大0.02622 | 严格失败，最大0.06640 |
| 同一 FC16 FP32 初始材料提升 CPU FP64，native及Python | 原阈值通过，最大≤7.77e-11 | 原阈值通过，最大≤1.80e-10 |

None/present 和非有限差异均为0。FP32 检查未放宽 `atol=1e-6,rtol=1e-5`；FP64 为 `1e-10,1e-8`。
该提升保持原初始化材料，支持本例的浮点放大判断；不证明任意模型或学习轨迹安全。
默认没有改成FP64。大尺度检查是普通任务输出/终态/梯度投影，没有声称完整中间根、完整控制轨迹或多步optimizer等价。
这些原严格失败与性能观察分列，有限诊断后结束，不无限阻塞后续独立工程。

## 三层计时

共同 B8/L128=1024 个输入位置，d128/H2/FFN512，summary/adapter/score/region hidden宽32；
KV qry、Read+、cat/ind/context/cat、regional_total/quad、温度0.25，无位置项/本窗无过期。
FC4 是2区域×2节点、k1、1,711,590参数；FC16 是4×4、k2、5,092,788参数（含共享embedding/head）。
Transformer4/16 为4/16个串行blocks，参数1,576,960/4,722,688，数学不同；不能把差额全当Tide冗余。

各单元 **完整秒数 / 输入位置每秒**，吞吐=1024/完整秒数。一次warmup后重置原参数/optimizer，单次测量；
没有统计稳定或硬件最优声明。推理包括prefill、最后位置head和真实首token返回；
含grad前向是完整序列head/loss及构图；完整训练另含真实反向、finite/clip、AdamW、清理及同步。
不把含grad前向当作训练，不将输入吞吐当后续decode吞吐。

| FC4 | 推理 | 含grad前向 | 完整训练 |
| --- | ---: | ---: | ---: |
| CPU packed-v2 | 0.7080 / 1446 | 1.1290 / 907 | 2.9381 / 349 |
| CPU 默认块 | 0.5223 / 1961 | 0.8268 / 1238 | 1.5681 / 653 |
| CPU 直接FC | 0.1031 / 9933 | 0.1422 / 7199 | 0.3816 / 2683 |
| CPU Transformer4 | 0.0430 / 23833 | 0.0673 / 15222 | 0.2463 / 4158 |
| NPU packed-v2 | 1.0077 / 1016 | 1.8928 / 541 | 3.5468 / 289 |
| NPU 默认块 | 0.8747 / 1171 | 1.4443 / 709 | 2.4938 / 411 |
| NPU 直接FC | 0.0693 / 14769 | 0.0793 / 12914 | 0.1829 / 5599 |
| NPU Transformer4 | 0.00298 / 343350 | 0.00370 / 277129 | 0.01274 / 80369 |

| FC16 | 推理 | 含grad前向 | 完整训练 |
| --- | ---: | ---: | ---: |
| CPU packed-v2 | 2.6916 / 380 | 4.5109 / 227 | 11.6012 / 88 |
| CPU 默认块 | 1.9936 / 514 | 3.0391 / 337 | 5.8351 / 175 |
| CPU 直接FC | 0.4185 / 2447 | 0.4927 / 2078 | 1.3902 / 737 |
| CPU Transformer16 | 0.1697 / 6034 | 0.1948 / 5255 | 0.7473 / 1370 |
| NPU packed-v2 | 3.7863 / 270 | 7.0446 / 145 | 15.1241 / 68 |
| NPU 默认块 | 3.2376 / 316 | 4.9915 / 205 | 10.9279 / 94 |
| NPU 直接FC | 0.2466 / 4153 | 0.2880 / 3556 | 0.6748 / 1518 |
| NPU Transformer16 | 0.00875 / 117048 | 0.01050 / 97527 | 0.04461 / 22955 |

FC16 批首token时延：CPU默认/直接/Transformer16 为1.99220/0.41844/0.16969秒，
NPU为3.23419/0.24640/0.00869秒。这是实测首返回间隔，不用训练时间除B或L推算。
默认/直接训练成本比 FC16 CPU4.20、NPU16.20；FC4 CPU4.11、NPU13.63。
默认路径仍然NPU慢于CPU，而直接FC/Transformer已经NPU更快；不能归结为硬件本身慢。

本轮纯PyTorch Tide FC4/FC16训练：CPU4.117/15.288秒，NPU5.131/21.147秒。
CPU32线程FC16默认/packed-v2/直接FC/Transformer16为6.620/10.884/1.385/0.605秒；
默认小图增加线程没有改善，不能推广为宽模型的最优线程。各自只测一次，CPU池也不同。
FC16 NPU进程峰值allocated默认/packed-v2/直接/Transformer16约562/1268/403/343MiB，含warmup，非单步增量。

FC16候选均16384；CPU Full8344/边消息25060，NPU Full8345/边消息25064。
各设备内部默认、packed-v2、直接FC同计数；仍不能用计数相同认证逐事件支持一致。
约51%执行Full，所有候选仍有Upd/Read/Describe，不能把Full比例当全部计算稀疏比。

## 机制、剩余成本与架构影响

在B8/L128主尺度，块Read16256/16384；块Full CPU8278/8344、NPU8279/8345，均约99.2%。
剩余是公开末态等保留独立根的后备，不是大部分query没有命中优化。
快照layout调用822→62，处理行约49456→388；query根分组涉及的行CPU57496→447。
原KV前缀gather/重复投影已大幅减少。它们是实际计数，非由配置标签推算。

B2/L32独立profile（不可按比例外推大尺度耗时）：

| 实现 | CPU ATen记录 | CPU含私有图唯一节点 | NPU kernel记录 | NPU Computing/Free毫秒 |
| --- | ---: | ---: | ---: | ---: |
| packed-v2 | 619090 | 18909 | 82351 | 307.13 / 2568.28 |
| 默认块 | 462865 | 10163 | 61813 | 221.79 / 1739.46 |
| 直接FC | 68338 | 5338 | 16224 | 63.86 / 378.15 |
| Transformer16 | 11225 | 730 | 2298 | 13.27 / 50.54 |

CPU反向已注册节点执行12564→10040；外层节点却717→1417，外层统计单独看会得出相反结论。
总连接边49673→44333，减少幅度较小；显式保存的唯一storage约27.10→26.78MB，
逻辑引用字节170.49→237.15MB。不能把“少了私有节点”说成所有对象和引用都同比减少。

CPU观测到的mm/bmm/addmm FLOPs：旧2.003G、新1.450G、直接1.603G、Transformer1.812G；
只覆盖这些算子，不含融合Attention和其他数学，不能当完整模型FLOPs。
NPU Transformer16实际16个FlashAttention前向及16个反向主kernel；直接FC和默认块没有同等融合。
前轮3D/4D SDPA负结果仍适用其原范围，不能因调用SDPA就假定已融合可求导质量bias。

NPU profiler仍有停止于RECORD可能解析不完整及未采集AICore指标的警告；导出含真实backward/clip/AdamW。
只报告已导出记录；Free为厂商时间线分类，不能认定某一操作独占该时间；host算子duration不作设备占时。
主表无插桩计时与profile分开。新math范围独立于bind，避免移动函数后把算术时间误记成接口成本。

阶段同步对照见下表；额外同步使它不能与主表逐毫秒互减。内部细分范围是调用线程经过时间，含等待。

| 设备/实现 | 核心前向秒 | backward秒 | 完整区间秒 |
| --- | ---: | ---: | ---: |
| cpu/默认块 | 2.9454 | 2.7461 | 5.8639 |
| cpu/直接FC | 0.5055 | 0.8063 | 1.3908 |
| cpu/Transformer16 | 0.1831 | 0.5184 | 0.7490 |
| npu-0/默认块 | 4.9020 | 5.4984 | 10.5550 |
| npu-0/直接FC | 0.2833 | 0.3533 | 0.7020 |
| npu-0/Transformer16 | 0.0091 | 0.0157 | 0.0424 |

cpu默认减直接完整差额4.4731秒：前向2.4399、反向1.9399、其余0.0933。 npu-0默认减直接完整差额9.8530秒：前向4.6188、反向5.1451、其余0.0891。
这只是对应区间的差值，不是内部机制的因果归因。CPU使用同一CPU池；补充NPU直接基线在空闲physical1、主默认在physical11，均同机同型卡，设备差异与单次测量噪声不能排除，不能把该阶段差额当严格设备归因。

| 位置/职责 | 实证与实现定位 | 本轮结论 |
| --- | --- | --- |
| query的KV查询、Describe/Full及伴随 | `weighted_kv_block_{plan,math,vjp}.cpp`、Python同名层；CPU query nested-grad范围4.384→0.665秒，NPU2.179→0.287秒 | 采用bank块接口；保留所有公开根。剩余query列装配/终态后备有成本，未宣称已融合 |
| 区域优化及反向 | `weighted_region_batch.cpp/weighted_region_vjp.cpp`；主尺度NPU allocation范围约0.178秒，region数值约0.170秒，region nested-grad约2.108秒 | 小型quad求解已经批量；大头之一是行根、输出掩码/依赖分组、重复嵌套grad与回填。没有依据把整体慢归给求解器 |
| 调度、State、Aggregate、消息/返回 | `component_window.cpp`、`block_prepare.cpp`、`weighted_private_rows.h`及相关VJP；仍4个外层阶段，但保留逐事件Tensor/字段/支持装配 | 外部窗口调度能够承载新块；下一边界应让列继续穿过区域/消息/发布，不需要为了这个差距再次更换通用图算法 |
| 直接FC与Transformer | `fc_tensor_baseline.py`；额外Read/Describe/发送强度和区域分配是真实数学；Full按B循环和多小投影是实现选择 | 直接FC不是最优下界，也不是Transformer等价物；可继续做B×L矩阵及有质量梯度的Attention算子，但本轮未实施 |

这里的“发布”是把计算列包装成公共控制/输出/状态字段及其梯度连接，不是网络通信。
下一次有限问题建议先比较紧凑区域列VJP→公共区域接口→实际FC整链，并检查支持回读和结果列贯穿；
同时保留直接FC自身批量化的独立成本锚点。该建议和未完成范围只由ROADMAP跟踪，不在此自动启动新轮次。

## 可重复诊断与维护

有限入口 `run_fc_comparison.py --suite blocks` 已能一次获得 auto/packed-v2/直接FC/Transformer 的三模式，
另有 profiles/checks/axes/boundaries 有限套件。可明确设置B/L、设备、线程、native构建和输出；
这不是必须全部运行的笛卡尔矩阵，也不将基线移入默认factory后继续自称独立参考。

`cost_comparison.py` 现在通过显式manifest配对成本，并用 `execution_diagnostics.py` 附加不同尺度的
profile/census、独立数值检查、JUnit及终态job。没有固定日期/旧作业名依赖；不启动模型即可重新审阅。
TTFT使用真实首返回值，吞吐使用实际输入位置，工作量差异和未验/失败分别保留。
只对同步阶段计算对应差额；私有图与外层图去重方式明确，不将二者简单相加。
FC和KV/SSD层链已有成本适配器；后续LH/其他执行路径复用各自runner，补小型schema适配器即可接入对照，
仍需独立公式和明确返回/根域，不能由收集器自动认证新场景。

完整使用方法和修改责任在[诊断契约](../fc16-cost-attribution.md#repeating-the-investigation)、
[架构图](../architecture.md)、[复现导航](../reproduction.md)、[性能经验](../performance-lessons-prefill.md#隐藏前缀bank与块伴随)。
本轮没有新LH大规模、resident、mixed-a/b、多卡或GPU吞吐结论，也没有下游依赖升级/学习验收。

## 从固定源码重建

在独立206bc9b checkout按[开发入口](../development.md)配置环境，先构建与客户端相符的native/SDK；
若复现本次计时身份，使用598b4fb及其对应构建。下游repo只需能读取固定0439efd，脚本从Git导出指定材料，
不使用可变下游HEAD。不需要本次旧运行目录。

```bash
python scripts/run_fc_comparison.py --suite blocks --device cpu --threads 1 \
  --native-library "$BUILD" --downstream-repo "$TASK_REPO" --output-dir "$NEW_RUN"
python scripts/benchmark_dense_query.py --downstream-repo "$TASK_REPO" \
  --native-library "$BUILD" --device npu:0 --client native --fc-shape fc16 \
  --baseline graph --mode training --trace-records minimal --batch 2 --length 32 \
  --threads 2 --profile npu --profile-shapes --autograd-census \
  --profile-output "$SCRATCH_PROFILE" --output-dir "$NEW_PROFILE"
python scripts/check_fc_comparison.py --downstream-repo "$TASK_REPO" \
  --native-library "$BUILD" --device cpu --fc-shape fc16 --batch 8 --length 128 \
  --dtype float64 --initial-dtype float32 --variants native python --output-dir "$NEW_CHECK"
python scripts/cost_comparison.py --manifest "$MANIFEST" --output "$NEW_REVIEW"
```

这些命令放在固定源码、资源有界的background.slice服务；NPU先通过空闲准入。
独立根/训练恢复使用 `test_kv_block_query/test_kv_sequence_vjp/test_kv_prefix_roots`、
`examples/consumer_core5/dense_query.py --no-trace` 和 `scripts/check_query_sdk.py`。
全部精确argv、构建manifest、安装结果hash、原失败和终态见资格JSON；它们是可重建说明，不承诺保留全部原始profile字节。
