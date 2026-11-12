# LH、FC 与1B：共同原点后继成本和结构诊断

固定数值实现`fb83ee91cc34ba49bf00465916d6e7e7a7eacdec`；42个无插桩成本点逐格完成，
结果hash与原矩阵一致。**FC16默认路径已接近这份直接FC实现；LH176和LH1B仍有明显差距。**
这里给的是实际达到的成本参照，不是直接实现的最优性证明，也不是完整数学等价资格。
严格数值/安装/恢复结果见[配套资格](rope-qualification-20261112.md)。
[机器记录](rope-performance-20261112.json)包含全部42点、材料身份、实际工作、命令、终态、packing及profile摘要。

## 计时、资源与配置

每配置一个新进程、一次恢复初态的预热、一次恢复初态后的测量，没有置信区间或重复扫描。
吞吐定义为**实际输入位置数÷完整同步区间秒数**，不是生成token速度。
LH176分子61；LH1B为481；FC16为1024。下表每格格式均为 **秒 / 输入位置每秒**。
无梯度推理、含grad前向、完整训练分别实测，不能相互代替。

成本任务运行于冻结源码/独立background.slice。CPU作业affinity0–31，ATen线程1或32；
NPU作业affinity32–39、ATen主机线程2，interop1。矩阵math线程与主机可用CPU池分别记录，
不是所有后续构建/测试的线程上限。CPU/NPU任务使用独立CPU池；性能期间未并行自身重构建。

实际NPU为Ascend910_9392物理9，逻辑npu0，CANN9.2/Torch2.12/TorchNPU2.12.0.post3，
`TASK_QUEUE_ENABLE=1/PER_STREAM_QUEUE=0`、未设launch blocking。它是**授权共享卡观察**：
共同驻留PID1711169；两次准入及结束样本均约5654MiB占用、AICore0%。
这些离散样本不证明整个测量区间独占或无争用。同主机另有独立卡1的resident资格任务，
不把本报告解释为机器独占峰值。准入身份、时限、实际affinity均留在JSON。

| 用例 | 拓扑与维度 | 局部/区域设置 |
| --- | --- | --- |
| LH176 | 176节点/22区域/1058主体边；区域层宽1,2,4,8,4,2,1；B4/L16、61实际位置；d64/H1/FFN256/summary32/adapter32；14.903M可训练参数 | textbook异构：扩散56节点qry，Read旋转32维、Full旋转64维；收拢56节点sum、Full旋转32维；中间64节点sum、无旋转/window0。其余window4608。14区域att/content_stats、8中间区域ind；共同Read+/cat/context/cat、regional_total/quad、T1、k1 |
| FC16 | 4区域×4节点，全连接层间；B8/L128、1024位置；d128/H2/FFN512/summary32/adapter32；core4.564M，含embedding/head5.093M | KV qry、无RoPE、window3072；Read+/cat/ind/context/cat、regional_total/quad、T0.25、k2 |
| LH1B | 同176节点/22区域/1058边拓扑；B32/L16、481实际位置；d640/H10/FFN2560/summary32/adapter32；1,051,335,206可训练参数 | baseline全KV sum/无RoPE；中间window0，其余4608；区域全部ind，其余Read+/cat/context/cat、regional_total/quad、T1、k1 |

LH输入为staggered-v1原始逻辑时间，stride9。其textbook配置不是所有节点都qry/att；
首节点配置不能代表完整拓扑。JSON按实际相同配置分组保留所有节点/区域索引及参数。

LH的完整训练包括输入生成/上传、packing、前向、逐端口`1e-6*sum((cG)^2)`、有限检查、
反向、SGD(lr1e-5,momentum0.3)、清理与同步；构造、初态hash和恢复在计时外。
FC沿固定下游`0439efd2`的合成语言任务，包含embedding、core、readout、head/loss、检查、
反向、clip和AdamW；推理仅做最后位置head/greedy首token返回。Transformer16是串行pre-RMS/SDPA/SwiGLU，
同d128/H2/FFN512与任务接口、实际4.723M参数，是不同函数。没有本轮Transformer或直接FC的独立推理/含grad前向点。

LH使用IEEE FP32，NPU matmul/conv HF32均false；FC为vendor-default FP32，matmul HF32=false、conv HF32=true。
默认native指Python调用独立C++列执行器及LibTorch算子，默认PyTorch保持其自身调度。
NPU均为主机推进的mixed-c，不是resident；本轮没有standalone SDK吞吐点。

## LH176 三层与三模式

| 设备/线程 | 路径 | 无梯度推理 | 含grad前向 | 完整训练 |
| --- | --- | ---: | ---: | ---: |
| CPU1 | 直接公式/PyTorch | 0.2032 / 300.2 | 0.2341 / 260.5 | 0.7516 / 81.2 |
| CPU1 | 公共列/native | 0.8498 / 71.8 | 0.9831 / 62.0 | 2.3505 / 26.0 |
| CPU1 | 默认/native | 0.9651 / 63.2 | 1.0831 / 56.3 | 2.4899 / 24.5 |
| CPU1 | 默认/PyTorch | 2.0698 / 29.5 | 2.3413 / 26.1 | 4.0621 / 15.0 |
| NPU/host2 | 直接公式/PyTorch | 0.1776 / 343.5 | 0.2024 / 301.3 | 0.6192 / 98.5 |
| NPU/host2 | 公共列/native | 2.0627 / 29.6 | 2.3464 / 26.0 | 5.2207 / 11.7 |
| NPU/host2 | 默认/native | 2.2282 / 27.4 | 2.4427 / 25.0 | 5.3286 / 11.4 |
| NPU/host2 | 默认/PyTorch | 3.6747 / 16.6 | 4.1031 / 14.9 | 7.1539 / 8.5 |

额外CPU32完整训练：直接公式 **0.7917 / 77.0**，默认native **3.0323 / 20.1**。
这两个点未比CPU1更快，只说明这个d64/B4/L16负载未从该线程配置获益；不外推宽模型或所有多线程情况。

所有LH176点候选8368、selected Full2572、访问边15340，默认/公共另记录区域列94次、状态列176次。
直接公式实际计算11264个dense Full行后mask，公共/默认计算2572行。k是发送强度约束，
k/区域宽=1/8不是实际活动比例；本例selected/candidate约30.7%。相同计数也不单独认证逐事件支持等价。

CPU默认native/直接训练成本比约3.31，NPU约8.61。将公共接口换成默认入口，只增加约0.1394秒/0.1079秒；
主要差距已经存在于公共列计算中，不能笼统归给最外层默认分派。
主机调用区间的训练差额为：

| default−direct | 完整区间 | forward区间 | backward区间 | 其余差额 |
| --- | ---: | ---: | ---: | ---: |
| CPU1 | 1.7384s | 0.9062s | 0.7474s | 0.0847s |
| NPU/host2 | 4.7094s | 2.3995s | 2.2056s | 0.1043s |

forward/backward是calling-thread区间，NPU工作可能跨区间完成；只有完整区间最终同步。
该相减定位差距在哪一段，不证明差额全是可删除的调度或autograd开销。

## FC16 默认、直接FC与Transformer16

| 路径/模式 | CPU1 | NPU/host2 |
| --- | ---: | ---: |
| 默认/native，无梯度推理 | 0.4336 / 2361.8 | 0.2095 / 4887.8 |
| 默认/native，含grad前向 | 0.4725 / 2167.3 | 0.2190 / 4676.0 |
| 默认/native，完整训练 | 1.2410 / 825.1 | 0.6050 / 1692.7 |
| 公共列/native，完整训练 | 1.2419 / 824.5 | 0.5998 / 1707.3 |
| 直接FC/PyTorch，完整训练 | 1.4011 / 730.8 | 0.6835 / 1498.2 |
| Transformer16/PyTorch，完整训练 | 0.7472 / 1370.4 | 0.0423 / 24210.4 |

默认路径实际首token时延：CPU **0.4326s**、NPU **0.2087s**。这是模型已载入、一次本地B8 prompt请求，
包括输入构造/上传、prefill、末位置head、greedy选择、有限检查与返回主机；不含排队、tokenizer或后续decode。
不能用完整训练秒数除B/L得到TTFT。

默认训练与公共列几乎相同；相对直接FC，默认完整训练在这次观察中CPU少0.1601s、NPU少0.0785s。
因此当前FC16默认入口已能承载这份直接公式达到的性能。Transformer16仍明显更快，尤其NPU，
但它省去了Tide特有的区域描述、评分/证据、分配、消息图与记忆职责，不能作为Tide数学等价oracle，
也不能将全部差额直接命名为可移除的架构成本。

FC16是全连接**拓扑**，本次分配的实际selected约50.9%：CPU8344/NPU8345，共16384候选；
访问边分别25060/25064。每个设备内部的默认/公共/直接FC工作计数匹配，跨设备却有一项活动差异。
因此CPU/NPU成本各自有效，但不是完全同路由工作量的严格速度比。完整尺度数值资格本轮未新增。

## LH约1B：仅推理

| 路径 | CPU32 | NPU/host2 |
| --- | ---: | ---: |
| 直接公式 | 1.2547 / 383.3 | 0.1832 / 2625.6 |
| 默认/native | 2.9835 / 161.2 | 3.4587 / 139.1 |

四点使用相同参数/输入hash，候选均54507。但直接公式selected11970、消息72161，
默认selected11971、消息72166；实际Full行分别90112与11971。差异没有被隐藏或强制对齐，
这些成本不能形成严格同函数、同路由的提速声明。NPU峰值allocated约5.46GB/5.26GB（十进制，直接/默认）。
没有新1B训练/精确恢复或9B结果，也不将历史公共CLI的不同计时边界与本表直接相除。

## Packing实际做了什么

另四次CPU/NPU×Python/native独立前向+任务反向观察，不计optimizer、不作吞吐样本。
实际全部选择`settle-columns-v1`与`bounded-common-origin-v1`。同一后端的CPU/NPU计数一致；
Python/native除`weighted_mm`外的逻辑计数一致，差异按原记录列出。

| 观察边界 | calls | rows（按该边界含义） |
| --- | ---: | ---: |
| KV Read相容节点组 | 90 | 176个节点请求 |
| KV Full相容节点组 | 159 | 171个节点请求 |
| 共同原点旋转 | 165 | 165个查询请求 |
| 私有矩阵BMM | 374 | 984个矩阵块 |
| 共享矩阵块GEMM | 26 | 62个矩阵块 |
| 其他weighted GEMM，native | 138 | 6512行 |
| 其他weighted GEMM，Python | 183 | 8668行 |

`weighted_mm`只统计各自`packed.project`包装器的调用，未覆盖所有矩阵乘；因此这45次/2156行差异
不能单独证明Python执行了更多数学工作或解释耗时差。原始metrics及分桶分布均保留。
这些不是GPU/NPU kernel数，也不能跨行直接相加。Full为171个请求形成159组，说明这一实际布局
多数Full组仍很小；Read176→90的合批也未接近直接公式按区域8节点一起算的形状。
共同原点路径确实生效，却不足以把所有来源变换、ragged装配和局部投影自然合成大算子。

## 独立profile与autograd结构

诊断脚本固定`f475020450eb8084301d9add7b611c6c85f2dd95`，数值native库固定fb83；
这段提交范围内Python/C++数值文件未变，新增诊断独立记录来源。CPU/NPU分别direct与default各一次，
相同LH176材料，ATen线程1；NPU仍共享物理9。包括前向、反向、有限检查及图普查，**不含optimizer**。
Census会增加hook和遍历成本，下面不是无插桩计时，不按profile秒数分摊完整训练秒数。

| 设备/路径 | Python可见节点 | native外层+显式私有图并集 | Python可见已执行节点 | Python可见保存的独立storage字节 |
| --- | ---: | ---: | ---: | ---: |
| CPU direct | 11192 | — | 11056 | 204899144 |
| CPU default | 48365 | 52646 | 44894 | 89696568 |
| NPU direct | 10513 | — | 10377 | 211354440 |
| NPU default | 46417 | 50698 | 42946 | 89742936 |

默认native还登记4281个区域私有图节点；其显式保存storage另为5,546,132字节。
Python/native观察空间重叠，不能相加；native显式保存并不是全部ATen保存张量，storage也不是allocator峰值。
默认路径在更少可见保存字节下仍有更多图对象/执行次数，表明应分别处理图粒度、算子提交与存储，
不能只用显存或一个外层Function计数解释性能。

NPU原始捕获中：

| 观察项（前向+反向+诊断） | direct | default |
| --- | ---: | ---: |
| kernel_details行数 | 20466 | 236714 |
| task_time行数（含copy等） | 20959 | 243852 |
| AICPU任务 | 22 | 188 |
| host operator条目 | 131275 | 1245118 |
| aten::matmul | 241 | 4991 |
| aten::bmm | 859 | 1715 |
| index_select / index_copy | 255 / 22 | 13289 / 5904 |
| cat / stack | 330 / 655 | 6089 / 3316 |
| item / _local_scalar_dense（对应两层，不相加） | 850 / 850 | 13288 / 13288 |

直接公式常见BMM形状为`[8,64,64]×[8,64,32]`；默认常见matmul为`[1..8,64]×[64,32]`
及其转置伴随。profile还显示大量copy、index_add/index_copy、张量创建与标量提取。
这支持继续检查相容组划分、来源/私有投影与消息装配边界；并不证明每一个观测算子都可删除，
或通用core-5语义必然要求这些次数。完整阶段的新增实现取舍应由独立局部证明和后继成本验证。

两份NPU日志均有两类原始警告：Level0无法采AIC metrics（自动设None），以及停止时仍处于RECORD、
解析可能不完整。复查实际trace JSON与四份CSV后，direct/default分别有361693/3897153个事件，
prepare、forward、finite-forward、backward、finite-gradients五个task区间都以完整X事件存在，
完整事件duration均合法，设备任务非空，解析日志完成。故可保留上述**实际捕获计数**，
但不宣称无丢失捕获、AICore利用率、完整算力归因或无扰动壁钟分解；警告没有因外层exit0而消失。

## 身份、重现与结论范围

LH176参数hash前缀`939fbb10`、输入/支持`6fb02760`；LH1B为`c08fb9b5`/`3e535fa8`，
所有对应层级、模式、设备一致，完整hash在JSON。FC默认/公共/直接初始与测量参数hash均`7f41057f…`，
Transformer为自己的`5d2db4a0…`。FC原记录没有实际输入字节hash；可追溯固定下游归档与
`synthetic_batch(seed=992)`生成器，但本报告不补造运行时未记录的hash。

42点分为LH176的24点、CPU32训练2点、FC16的12点、LH1B推理4点。
机器记录同时保留原计划`planned_not_run`字节身份与每项实际终态，不能用计划标签代替运行记录。
按[开发/资格规范](../development.md)建立新冻结源码及匹配库，LH计划由
[LH基线配方](../lh-tensor-baseline.md)生成；`benchmark_lh_tensor.py`分别传level/client/mode，
FC用`benchmark_dense_query.py --baseline window|columns|fc-tensor|transformer-16 --warmup-state reset`。
每点完整argv与实际输出hash在JSON，不需要旧shm目录；固定下游Git对象仍是FC任务来源。
诊断用`profile_lh_tensor.py --profile cpu|npu --census`，NPU另给新的trace-dir并重新准入。

本次采用的结论是：FC默认张量路径已经能承载现有直接FC成本；LH默认相容合批与共同原点生效，
但接口/局部计算仍有可定位的碎片边界。下一实现由主任务沿这些证据选择；本报告不把局部合批成功
或42项exit0扩大成全部P1—P6完成、严格等价、大规模训练、resident或GPU性能结论。
