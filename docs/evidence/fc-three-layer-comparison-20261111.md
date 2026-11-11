# FC4/FC16：默认 Tide、直接 FC 与 Transformer 三层对照

本轮补齐了同配置的三种模式、接口省略实验、CPU/NPU 实际算子、阶段差额及独立数值核对。
**默认路径存在显著的表示和执行粒度成本，不能主要归因于区域优化问题；直接 FC 也尚未充分优化。**
这轮交付诊断设施和架构判断，没有修改生产 Python/C++ 计算路径，没有宣称差距已修复。
[机器可读结果](fc-three-layer-comparison-20261111.json)保存103个成本记录、8组投影检查、12个独立profile、
命令、配置、原文件hash和严格失败摘要。profile前的未插桩计时也在103项内，不是103个独立问题。
原始图节点统计仍引用[前轮证据](additive-interface-performance-20261110.md)，本轮ATen/kernel数量不冒充autograd节点数。

## 身份、资源与边界

| 项目 | 固定身份/实际范围 |
| --- | --- |
| 数学 | `0770593838a37c790a568226930567ed60f99b83`；19份core-5快照离线核验通过 |
| 主实验/设施 | `15d391f367c450effd25996b264c223bb8b075db`；主要改动`bf940f1` |
| 同FP32材料提升精度、实际边界计数 | `b0edf96a8b4d86d461f83329559b14cd604ab3e2` |
| SDPA4D实验/最终13项检查 | `fd243f41417b3f2fd8b49156a577335de0588010` |
| 终态证据收集器 | `96500b0`；JSON另记录收集脚本SHA256，只解析已完成数据 |
| 生产计算/复用构建 | `e6f7bd5f570b6227d7664e3be45c4f27eb5b611a`；至上述提交C++内容无差异；Release/ABI1，原构建320 workers |
| 下游任务 | `tide-learning-lab@0439efd2af31cf10b07ce9e6423b0195f7449968`的`configs/studies/language/v6/scale-fc16-kv-quad.json`及固定Git源码；只读 |
| 环境 | aarch64，Python3.12.13，Torch2.12.0+cpu，TorchNPU2.12.0.post3，CANN9.2.0；项目私有venv |
| 主CPU | affinity80–111，ATen intra/inter-op=1/1；另测intra-op32；不是CPU峰值扫描 |
| 主NPU | 空闲physical11→`npu:0`，host affinity0–7，ATen intra/inter-op=2/1；mixed-c，host推进，非resident |
| 客户端 | 主Tide为Python调用C++/LibTorch；直接FC和Transformer为纯PyTorch；边界省略试验用独立Python图调度 |
| 精度 | FP32；CPU matmul=`highest`；NPU matmul HF32关闭、conv HF32开启（本例无卷积）；无autocast |

4项独立`background.slice`服务`gef-fccompare-{cpu,npu,triage,sdpa}`均退出0、终态passed，
MainPID=0/inactive；这只表示流程完成，数值资格见后文。主作业各5400秒，triage1200秒，
SDPA1800秒；每项子进程600秒，记录MemoryMax=128GiB。配置这些属性不等于证明资源独占或控制器强制生效。
CPU/NPU主作业使用不同CPU池并行，没有重型构建重叠，没有使用共享NPU；外部主存/CPU争用未排除。

启动账户292GiB/软300GiB/硬800GiB，文件系统955GiB可用；收尾约900GiB可用，账户仍292GiB。
主机约1.8TiB可用内存，shm400GiB；收尾本轮临时源码/profile约1.3GiB。
复用环境/构建，将临时源码和大型profile放项目独占shm目录，小型日志/配置/结果持久保存。
**没有因磁盘限制减少并行或缩减本轮范围，也没有清理外部仓库。** shm不是持久证据或独立于RAM的容量。

## 实际计算与计时口径

共同输入B8/L128，即8条各128位置的序列，共1024个输入位置；d128，2 heads，FFN512，
summary/adapter/score/region hidden宽32。KV `qry`、Read+、cat/ind/context/cat、
`regional_total/quad`、温度0.25，无旋转/距离项。每个样本从空左态开始，没有截断本窗口KV。

| 配置 | 区域×节点 | 发送总量参数k | 总可训练参数 | 逻辑时间 |
| --- | ---: | ---: | ---: | --- |
| FC4 | 2×2 | 1 | 1,711,590 | stride4，区域rank1/2；KV窗口按原逻辑时间 |
| FC16 | 4×4 | 2 | 5,092,788（core4,564,404） | stride6，rank1–4；配置窗口3072或0（无限） |
| Transformer2/4/16 | 串行2/4/16 blocks | 不适用 | 1,052,672 / 1,576,960 / 4,722,688 | 普通因果sequence位置 |

FC相邻区域全连接，独立输出端口；不是4/16个串行Transformer blocks。FC16实际训练：
CPU候选16,384、Full8,344、边消息25,060；NPU候选16,384、Full8,345、边消息25,064。
同设备默认和直接FC的这些计数相同。FC4两设备均候选4096、Full2096、边消息2100。
约51%的候选执行Full；所有候选仍有状态更新/Read/Describe，不能将Full比例当作全部计算稀疏比。
完整控制身份的独立核对范围另列，计数相同不证明逐事件相同。

每配置独立进程，一次warmup后恢复初始参数/optimizer，再测一次；记录恢复前后参数hash相同。
主表使用`--phase-fences step`，完整区间含输入生成/上传、计算、必要检查、清理、同步。
构造、warmup、profile均在区间外。模型/输入seed分别11/992；共享4096词表、绑定embedding/head，
训练使用原任务AdamW/clip/lr流程。下游JSON内旧foundation版本不代表实际运行版本，以上源码/构建身份优先。

- 推理：关闭grad，完整prefill，最后位置head、greedy首token和host返回；不是后续decode。
- 含grad前向：完整序列head/loss并保留构图，不执行backward或更新。
- 训练：完整序列head/loss、真实backward、finite/clip、AdamW更新和同步。
- `nograd`另测同训练形状的完整head/loss，用于区分头部工作与构图；不能直接用推理差额代表构图成本。

## 三层实测

各单元为 **完整秒数 / 输入位置每秒**；吞吐=1024/秒数。单次观察，不声称统计稳定或最优值。

| FC4对照 | 推理 | 含grad前向 | 完整训练 |
| --- | ---: | ---: | ---: |
| CPU 默认Tide | 0.7068 / 1449 | 1.1955 / 857 | 2.9791 / 344 |
| CPU 直接FC | 0.1038 / 9861 | 0.1434 / 7142 | 0.3889 / 2633 |
| CPU Transformer4 | 0.0441 / 23238 | 0.0694 / 14752 | 0.2497 / 4100 |
| CPU Transformer2 | 0.0227 / 45161 | 0.0477 / 21454 | 0.1684 / 6082 |
| NPU 默认Tide | 1.0552 / 970 | 2.1241 / 482 | 3.6754 / 279 |
| NPU 直接FC | 0.0790 / 12963 | 0.0857 / 11952 | 0.1898 / 5395 |
| NPU Transformer4 | 0.00344 / 297874 | 0.00397 / 257878 | 0.01421 / 72046 |
| NPU Transformer2 | 0.00216 / 473280 | 0.00272 / 375856 | 0.00818 / 125108 |

| FC16对照 | 推理 | 含grad前向 | 完整训练 |
| --- | ---: | ---: | ---: |
| CPU 默认Tide | 2.7557 / 372 | 4.5987 / 223 | 12.4143 / 82.5 |
| CPU 直接FC | 0.4285 / 2390 | 0.4929 / 2077 | 1.4144 / 724 |
| CPU Transformer16 | 0.1683 / 6085 | 0.1964 / 5213 | 0.7660 / 1337 |
| CPU Transformer4 | 0.0433 / 23644 | 0.0687 / 14896 | 0.2481 / 4127 |
| NPU 默认Tide | 4.1311 / 248 | 7.5571 / 136 | 16.5251 / 62.0 |
| NPU 直接FC | 0.2433 / 4208 | 0.3019 / 3392 | 0.7452 / 1374 |
| NPU Transformer16 | 0.00918 / 111580 | 0.01394 / 73431 | 0.04267 / 24000 |
| NPU Transformer4 | 0.00390 / 262861 | 0.00398 / 257336 | 0.01505 / 68019 |

FC16批首token时延：CPU默认/直接/Transformer16为2.75429/0.42839/0.16826秒；
NPU为4.12686/0.24312/0.00911秒。FC4对应CPU0.70616/0.10381/0.04404秒，
NPU1.05363/0.07887/0.00336秒。不能用输入吞吐或训练时间除以B/L代替TTFT。

FC16默认/直接训练成本比CPU8.78倍、NPU22.18倍；FC4为7.66/19.37倍。
直接FC/同节点数Transformer训练成本比FC16为1.85/17.47倍，FC4为1.56/13.35倍。
后一个比例包含不同数学和不同接口职责，不能视作全部可消除的冗余。
默认Tide中NPU更慢，而两种直接计算中NPU更快；这与大量小提交/host等待的观察一致。

FC16 NPU训练进程观察到的peak allocated默认/直接/Transformer16为1267.88/402.94/342.61MiB；
这是包含warmup在内的进程峰值，不是经重置后独立测量的单步增量。
FC16同全序列head/loss的`nograd`：CPU默认/直接/Transformer16为2.8978/0.4636/0.1984秒，
NPU4.0103/0.2786/0.01032秒。CPU1→32的训练则分别从12.414/1.414/0.766变为
23.111/3.997/1.322秒；本d128小算子场景增加线程没有帮助，不能推广为宽模型的最优线程结论。

## 把差额落到计算与接口

独立的阶段同步诊断（B8/L128，秒）如下。它有额外同步，不能与主表逐毫秒相减；
自然`step`计时里的phase字段仅是host调用/等待间隔，不能当NPU阶段设备时间。

| 设备/实现 | 核心前向 | backward | 完整区间 |
| --- | ---: | ---: | ---: |
| CPU 默认 / 直接 / Transformer16 | 4.5265 / 0.5254 / 0.1843 | 7.5594 / 0.8470 / 0.5213 | 12.2736 / 1.4608 / 0.7547 |
| NPU 默认 / 直接 / Transformer16 | 7.2641 / 0.2964 / 0.0100 | 8.5144 / 0.3710 / 0.0162 | 15.9554 / 0.7408 / 0.0465 |

默认减直接：CPU总差10.8129秒，其中前向4.0012、反向6.7124、其余0.0993；
NPU总差15.2146秒，其中前向6.9676、反向8.1434、其余0.1035。
这是对应阶段的差额，尚不是每项内部机制的因果归因。大差额不在任务head或optimizer。

**有界省略实验**保留Python图调度和实际数学函数，只去掉指定私有独立根包装：

| FC16完整训练，秒 | public | 省query包装 | 省region包装 | 两者均省 |
| --- | ---: | ---: | ---: | ---: |
| CPU | 24.7101 | 21.0601 | 22.4125 | 17.6653 |
| NPU | 30.0350 | 25.2045 | 26.4289 | 20.1286 |

两者均省减少28.5%/33.0%，仍远慢于直接公式；相互作用使单项收益不可简单相加。
它们不保留所有公共中间独立根，因此不是可交付优化。不能把Python的差额直接从native时间扣除。
本尺度public投影对照：query省略前向/梯度一致，region/both前向一致，梯度CPU最大5.36e-7、
NPU1.34e-7，均通过原任务投影容差。后补真实调用计数为Read16/16384行、region4/4096行、
Full16/8345行，验证省略确实被执行；后补计数来自FP64提升检查，不改写早期计时源码身份。

| 层/具名代码 | 必须计算的数学 | 当前额外工作与判断 |
| --- | --- | --- |
| [独立FC](../../scripts/fc_tensor_baseline.py)的`aggregate/prepare/full/run` | `m=Σc`，`u=Σcφ(G)/m`；写入KV；短/长两路`min(1,Σw)·softmax(QKᵀ/√d+log w)V`；Describe与选中Full/FFN；区域/端口输出 | 独立按区域推进；Read覆盖B×L，Full仍按B循环；保存实际终态。它自身还重复部分Aggregate、执行多种小投影/检查，不是最优下界 |
| [查询布局](../../cpp/src/weighted_query_layout.cpp) `snapshot_chunks/SnapshotLayout`、[packed query](../../cpp/src/weighted_query_packed.cpp) `packed_memory/packed_full` | 各逻辑时刻的可见KV、短/长窗口、独立Read/Full快照 | 前缀视图先按长度桶变成补齐快照，重复构造visibility/age/index、gather和字段转置。共享bank已减少数值复制，接口仍按快照消费 |
| [query VJP](../../cpp/src/weighted_query_vjp.cpp) `bind_group/QueryVjp::apply` | 独立输入/状态/参数根的一阶导数与版本检查 | 每行私有叶、返回字段、依赖分组及nested grad；包装省略只去掉其中一层，数值layout与索引仍存在 |
| [bank Attention](../../cpp/src/weighted_bank_attention.cpp) `bank_attention/AttentionVjp::apply` | 同一bank上的QK/PV和正确共享梯度汇总 | 前向已用bank矩阵；反向展开`bank×head×query×history×width`并返回逐快照K/V伴随，之后再汇总，未实现端到端bank伴随 |
| [序列VJP](../../cpp/src/weighted_kv_sequence_vjp.cpp) `reduce/SequenceVjp::apply` | 各快照对实际写入的导数相加，保留未来未连接None与过期零 | 收集/cat每个前缀伴随，构造各跨度索引，再gather/sum到写入；共享前向存储并未消除反向跨度物化 |
| [区域VJP](../../cpp/src/weighted_region_vjp.cpp) `bind_group/RegionVjp::apply` | 区域总量与精确分配映射的VJP，再传播到score/evidence等依赖 | 数值分配已按帧批处理；私有列→逐行根→按实际输出掩码/依赖分组grad→scatter仍有成本 |

上表审阅限于这些具名边界；Python对应实现由`weighted/{query_vjp,query_layout,query_packed,
snapshot_bank,region_vjp,kv_sequence_vjp}.py`导航。本轮不宣称全仓逐函数审阅完成。
既有通用能力不是这些具体对象形式的理由；迁移接口可行，但不能删除独立根、None/零、
int64时间、逻辑窗口、状态/续接、稳定选择或从参考补写缺失记录。

区域本例解的是`max s·c−τ‖c‖²/2`，`0≤c≤μ, Σc=T`；
`T=low+sigmoid(context)·(high−low)`，`low=min(1,Σw)`，`high=min(k,Σμ)`。
解为`cᵢ=clip((sᵢ−λ)/τ,0,μᵢ)`并求水位λ。直接基线枚举仿射区间，公共实现见
[精确分配契约](../weighted-allocation.md)；反向是解映射的导数，不对有限求解迭代直接求导。
这是一批N=2/4的小问题，不是每个token单独跑大型优化器。
同尺度native插桩记录：CPU/NPU allocation仅0.054/0.212秒、region数值主体0.088/0.201秒，
region nested grad为0.647/2.469秒；query nested grad为4.688/2.444秒。
这些是互斥调用线程范围中的经过时间，含等待，不是设备归因或可直接相加的多线程wall占比。

## 硬件与算子证据

以下均为独立 **B2/L32** 完整训练profile，不能线性外推到主表B8/L128。

| 实现 | CPU ATen记录数 | 观察到的mm/bmm/addmm FLOPs | NPU kernel记录数 | NPU Computing / Free毫秒 |
| --- | ---: | ---: | ---: | ---: |
| 默认Tide | 619,090 | 2.003G | 82,351 | 308.882 / 3467.548 |
| 直接FC | 68,338 | 1.603G | 16,224 | 63.159 / 452.414 |
| Transformer16 | 11,225 | 1.812G | 2,298 | 12.969 / 58.583 |
| Transformer4 | 3,029 | 0.604G | 615 | 3.581 / 27.193 |

ATen记录含嵌套调用，不等于kernel；FLOPs只算所列矩阵算子，未计融合Attention和非矩阵工作。
矩阵工作处于相近量级，但默认/直接的ATen记录相差9.06倍、NPU kernel相差5.08倍。
默认NPU host记录copy6371次、标量取值4609次、empty_tensor62154次、as_strided77993次；
任务记录包括73167个AIVEC、7440个MIX_AIV、1608个AICORE，另2393个异步copy和136个AICPU。
小算子/对象/索引与提交等待是实证；不把每次标量调用都认定为D2H，也不把copy次数认定为巨量传输字节。
Free是厂商时间线分类，包含提交/等待空隙，不证明单一原因；没有采集AICore利用率。
NPU operator CSV的device duration列全零，不能据此作各语义阶段的设备占时归因。

profile有“停止时仍为RECORD，可能解析不完整”警告；最终同步已执行，导出中实际含backward、clip、
optimizer与AdamW事件，解析报告成功。上述数字限定为已导出记录，不把警告抹掉或当作完整性证明。
它们与完整同步区间及源代码共同支持粒度判断，不用profile wall替代正式吞吐。

同尺度B8/L128中，虽只有4个外层阶段、16个Read和16个Full批，插桩仍观察到822次snapshot layout、
1380次unique gather、416次unique projection、131次bank Attention子调用。
因此“已有packed/少数外层批”不代表内部已经保持B×L块。
同为128输入位置时，B1/L128→B8/L16的直接FC训练CPU0.305→0.538秒、NPU0.331→0.697秒，
尽管Attention历史工作变小，按B循环仍使提交增多。该对照工作量不同，只用于检验粒度假设。

**SDPA负结果**：仅将直接FC的Full Attention替换成3D/4D SDPA，FP32原形状训练CPU
1.414→1.425/1.403秒，NPU0.745→0.748/0.719秒；没有数量级改变。
4D profile中CPU实际32次调用为8次Flash/24次数学展开，NPU32次全部数学展开、无Flash kernel；
Transformer16则有16个FlashAttentionScoreGrad kernel。NPU4D仍有16,344个kernel记录。
实际带权浮点bias及其梯度、ragged布局和后端能力须共同处理，不能简单把`log w`改成bool mask或detach。
本轮未单独隔离触发NPU回退的每个条件，也未实现支持权重梯度的融合Attention。

## 数值与资格结论

设施的固定15d391f检查16通过；fd243f4最终诊断检查13通过（扩展组，与前组重叠，不相加）。
小FC4/B2/L8 CPU FP64全部10个比较通过，含控制/支持、终态、输入和参数导数；
NPU FP32 native数值最大约2.91e-5、梯度3.05e-5，保留严格失败。

完整FC16/B8/L128、minimal返回，默认对直接公式：CPU值最大0.07988、梯度0.89559；
NPU0.07817/0.87883。无非有限，不能宣称本尺度严格等价加速。
追加CPU完整控制路径：453个浮点control字段超差，但**没有bool支持/int64字段差异**；
该路径值最大0.09235、梯度1.03571。第一组区域已有control超差，例如b1/node0/pos2为2.50e-6；
下一区域的终态开始出现更大误差。它不是minimal路径的同一次运算，二者数字不可混写。

将**同一份FP32初始参数**提升FP64，保持完整FC16输入和独立调度：
native值最大1.46e-10，所有值/控制/支持/状态检查通过；梯度最大4.29e-10，
features和embedding.weight各1个坐标仍超过原`atol1e-10/rtol1e-8`。Python有相同两字段的严格残差。
这有力支持浮点算术差异被传播放大；没有抬高容差、换初态或将原失败改为通过。
本轮未定位每一级误差的条件数，未证明任意参数/尺度均安全，也未认证完整训练轨迹等价。

SDPA4D完整投影另有严格失败：CPU值/梯度0.02872/0.14504、NPU0.03148/0.11830；
小型公式/梯度检查通过，不代表完整尺度通过。主成本表允许这类有限数值探索。
本轮没有新增all-root资格、多个optimizer步等价、新进程恢复或学习效果结论；它们的既有证据保持原范围。

## 架构决定与复现

本轮采用诊断设施，保留公共/直接/Transformer三类锚点；不采用包装省略或SDPA试验为生产默认。
证据支持下一次以 **状态bank—Read/Full块—块伴随汇总** 为一个完整边界设计：

1. 状态以bank owner、int64时间/可见范围、查询块传递，B与L同时保留；数值列跨模块传递，避免反复拆行再装回。
2. 普通任务在块内累加K/V/write梯度；中间观测/独立根按明确请求保留因果连接，不能无条件物化全部快照伴随。
3. 区域Describe/分配/控制也保留列接口；选中Full按实际支持执行，通用PDG/TimedDAG/SettleGraph能力后备继续存在。
4. 用同数学直接公式→公共模块→默认入口逐层接入；每层检查必须能力与额外成本，再决定融合或新算子。
5. 默认路径先接近一个经过验证的紧凑FC实现；直接FC与Transformer的剩余差距继续区分新增数学与算子能力。

这不是一套FC专属调度器，也不要求先迁就旧逐行API。设计/实施待后续对齐，LH/GPU/多卡未扩展。
尚未证明最高效切分、最小可达时间或所有差额都可去除；当前证据已足以选择接口边界，停止同候选的重复测量。

从干净固定源码按[development](../development.md)建立私有环境/匹配native库；任务依赖为上述固定下游Git对象，
无需旧artifacts或shm内容。长实验在有界服务运行；NPU使用现行空闲卡准入。以新目录执行：

```bash
python -m pytest -q tests/test_fc_comparison_diagnostics.py tests/test_fc_tensor_baseline.py --dtype both
python scripts/run_fc_comparison.py --suite main --device cpu --threads 1 \
  --native-library "$FC_BUILD" --downstream-repo "$FC_DOWNSTREAM" --output-dir "$FC_NEW/main"
python scripts/run_fc_comparison.py --suite profiles --device cpu --threads 1 \
  --native-library "$FC_BUILD" --downstream-repo "$FC_DOWNSTREAM" --output-dir "$FC_NEW/profile"
python scripts/check_fc_comparison.py --downstream-repo "$FC_DOWNSTREAM" --native-library "$FC_BUILD" \
  --device cpu --fc-shape fc16 --batch 8 --length 128 --trace --dtype float64 --initial-dtype float32 \
  --variants native python both-direct formula-sdpa --output-dir "$FC_NEW/promoted"
```

`boundaries/axes/threads/checks`为同driver的独立有限suite；NPU改device、threads并提供`--profile-scratch`。
SDPA单项用`benchmark_dense_query.py --baseline fc-tensor --client python --formula-attention sdpa4d`，
其余shape/mode/计时边界显式保持。完整原argv与launcher在JSON，绝对路径仅为原身份，不要求恢复旧目录。
`collect_fc_comparison.py --run-root RUN --jobs-root JOBS --output NEW.json`仅收集本套终态布局，
校验吞吐算术，不重跑模型；它不把原严格失败提升为资格通过。
