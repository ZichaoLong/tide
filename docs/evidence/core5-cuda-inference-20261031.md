# RTX3090 规模推理与两项有限优化 — 2026-10-31

[逐配置记录、实际工作、原始窗口标量、构建与回归](core5-cuda-inference-20261031.json)。
数学固定 `0770593838a37c790a568226930567ed60f99b83`。推理与训练、独立正确性与自身恢复、
源码与构建、profile与正式测量分别记录；原严格断言、容差及历史失败保持。
本轮50项正式测量中47项通过完整独立推理及磁盘精确续接，3项原FP32严格失败单列成本。

## 来源、环境和验收

| 固定来源 | 实现和实际资格 |
| --- | --- |
| eee68c4 | 推理不构造optimizer/master；完整前向及磁盘新owner续接；staggered-v1真实不齐输入。A的21正式项18通过、3严格失败；CPU相关23、resident推理/训练两dtype四项通过 |
| 1c903ae | 桶、矩阵行数、活动/缓存观测。core新构建通过；standalone因`work`命名歧义编译失败，原失败保留；接续B的两正式项及四profile通过 |
| 5f9bff1 / 4960b6c | 已成组的detach bool批量取回；首差诊断、显式CUDA eager FP64和standalone命名修复。新core/standalone构建、相关66项、三客户端选定训练9项通过；D的12正式项通过 |
| 9fce486 | 静态规模预算及离线同步API区间并集；GPU literal与两客户端零支持训练通过 |
| 2f17c24 / fd3c72b | 显式完整返回packing及恢复峰值。F目标环境回归33项通过，其中9项直接CUDA；原Half检查材料构造失败。H只修该材料，C++三dtype复制、三客户端小型训练/恢复、10正式项通过 |
| 9acc7aa / f738692 | 三客户端1B共同点及单独d5888近界目录；来源只改变配置，公共core/消费者源码hash匹配的构建身份仍独立保留 |
| 909a322 | 查询内嵌configuration facts并校验模型/工作负载/参数身份；固定源码5项及模型/完整索引查询通过，仅修离线证据入口 |

GPU为Ubuntu22.04/x86_64、RTX3090/24GiB/SM86、驱动595.71.05、Torch2.12.1+cu130、
Toolkit13.0.88、GCC11.4，项目私有环境。构建56线程（affinity112/配额56），容器主存240GiB、数据盘50GiB。
测量固定GPU0，ATen/BLAS/interop/native workers均1；FP32 IEEE、TF32关闭。
大尺度明确使用FP64 payload/scoring，来自同一FP32种子材料提升，原FP64容差atol=1e-10/rtol=1e-8。
它是另一个执行配置，不解除FP32失败。FP16仅做相关设备复制/resident机制验收，无新增FP16规模吞吐。

一次流程含1个warmup、3个连续测量窗、磁盘新owner的完整下一窗精确比较，再独立CPU逐窗核对。
计时包括必要输入准备/传输、在线调度、packing、计算、原完整返回口径中的诊断loss、完整CPU返回与同步；构造、材料生成、独立参考、
checkpoint验收在计时外并单列。大规模仅推理，无optimizer/训练资格。连续状态窗口不是独立重复试验。
本项目重型构建与正式计时串行；profile另进程，外部CPU负载未隔离，不能据单次中位数宣称稳定倍率。
纯PyTorch调度、Python/native、standalone分别实际运行；resident调用公共设备库，不称纯PyTorch常驻调度。

## 实际材料与批次

| 拓扑/局部组合 | 主体节点/物理边 | 去重可训练参数 | batch / 每窗输入位置 / 原逻辑跨度 |
| --- | ---: | ---: | --- |
| FC，KV baseline | 4 / 4 | d8:8,190；d256:4,197,862；d1024:63,868,390 | B1/8/32，P4，span16 |
| FC，KV大尺度 | 4 / 4 | d4096:1,010,339,302；d5888:2,085,424,614 | B1/8及近界B1，P4，span16 |
| HB扩散/收拢，KV baseline / rich | 16 / 32 | 31,388 / 34,236 | B8，P4，span20 |
| LH启发连接，core-5 KV局部函数 | 16 / 68 | 31,420 | B8，P4，span20 |
| FC，GDN / SSD代表 | 4 / 4 | 8,778 / 8,746 | B8，P4，span16 |

FC另有12个冻结标量，HB另有72个、LH另有144个；参数量按canonical owner去重，角色共享与私有状态分开。
完整配置/自然维度/共享绑定由[目录](../../benchmarks/core5-cuda-inference-v1.json)和JSON的models定义。
rich包含qry、attention及位置配置；不遍历候选组合，也不以无用参数凑规模。
staggered-v1保持每端口连续occurrence与原始逻辑时间，实际样本存在不同输入、活动、消息和缓存长度。
prefill在线形成批次，不由CPU预跑轨迹。FC/HB/LH的State实际最大时间批为4；同FC streaming为1，
Read/Full API调用各由prefill的4次变为16次，工作仍89候选/89 Full/78消息。
FC d8/B8的最大Full行25包含样本和时间展平，不能称时间批25；本轮未收集每owner Read/Full的完整T直方图。

FC B8每样本独立输入根为[4,3,2,4,3,2,4,3]，物理输入原子[6,5,3,6,5,4,6,4]；
H实测每样本候选7–14、消息6–12，缓存长度2/4/6/8。B32候选6–14，缓存仍真实异长。
HB rich每样本候选30–60、消息56–112；连续窗最大存储KV长度8→12→16。
发送比例请求1/8经自然取整为k=1，FC区域2节点、HB/LH区域4节点；这是发送强度预算，
实际Full比例从FC d8的100%到HB baseline约93.6%、HB rich99.47%、LH约95.7–97.9%，不能混称活动稀疏度。
A未采集逐样本边界缓存，不能用H的不同配置补造A数据。

## 小中型完整窗口成本

单位ms，三个连续窗口中位。A保留原scalar返回；D为元数据修改后scalar返回，H为显式packed返回。
不同来源逐项核对材料/工作量，表中各通过项均有完整独立记录及精确恢复；不是同一二进制的交错AB试验。

| 材料与执行 | 纯PyTorch | Python/native | standalone |
| --- | ---: | ---: | ---: |
| FC d8/B8，A mixed-c | 306.045 | 246.718 | 310.612 |
| 同材料A resident | — | 145.293 | 141.988 |
| 同材料A CPU / mixed-a / mixed-b | — | 67.494 / 242.459 / 213.991 | — |
| 同材料A streaming | — | 366.369 | — |
| 同材料D mixed-c scalar返回 | 311.968 | 240.310 | 249.257 |
| 同材料H mixed-c packed返回 | 273.837 | 225.788 | 206.459 |
| FC d8/B32，H mixed-c / A resident | — | 756.372 / 562.031 | — |
| HB baseline B8，A mixed-c / resident | — | 949.176 | 633.727（resident） |
| LH启发 B8，A mixed-c | — | 1015.202 | — |
| HB rich B8，D / H mixed-c | — | 982.121 / 905.717 | — |
| FC GDN / SSD B8，A mixed-c | — | 243.170 / 236.304 | — |
| FC d256/B8 FP64，D mixed-c | 316.363 | 269.781 | 293.477 |
| 同材料D CPU | — | 89.712 | — |

小图单线程CPU仍快；resident在所测B8/B32图上比mixed-c便宜，但仍有构造和完整解码成本。
mixed-a/b/c的差异限定同材料；本轮没有建立全图族最优控制放置规则。
同H源码、返回都packed时，局部scalar/packed为400.147/225.788ms；局部算子策略与返回策略是两个独立维度。
4.2M FP64 native B1/B8/B32的H成本为84.246/260.898/787.432ms；D相应83.759/269.781/1197.397ms。
63.87M/B1为107.846→102.260ms。B1小矩阵返回packing无收益，不能把最大观察下降外推。

同模型增batch的明确对照如下。候选吞吐按三个窗口总工作/总耗时计算，显存为测量阶段
allocator峰值，不含驱动；A未单独采恢复峰值，不能用此表作恢复容量保证。

| FC d8 / native mixed-c | B | 中位ms | 候选/s | 测量peak allocated / reserved MiB |
| --- | ---: | ---: | ---: | ---: |
| A scalar返回 | 1 | 65.171 | 209.18 | 8.30 / 22 |
| A scalar返回 | 8 | 246.718 | 361.49 | 8.80 / 22 |
| A scalar返回 | 32 | 852.974 | 404.96 | 10.55 / 24 |
| H packed返回 | 8 | 225.788 | 383.88 | 8.80 / 22 |
| H packed返回 | 32 | 756.372 | 455.07 | 10.55 / 24 |

该小图增batch提高吞吐，但延迟上升、吞吐增幅递减。D的独立profile中，B8→B32的显式CUDA同步
union为14.473→52.707ms，占捕获跨度2.63%→2.81%；同步调用5336→19598。
这只回答已捕获显式API的占比，未测全部主机等待，不能推断大尺度的同步比例。1B显存变化另列如下。

## 1B、近界与客户端资源

9acc7aa的1B native FP64 B1/B8均完整通过。B1三窗303.140/223.885/223.883ms，
B8为838.058/741.435/771.993ms；候选更新吞吐按三窗总工作/总耗时为55.93→113.55/s。
B1每窗14候选/8 Full/8消息，B8为89/50/50，实际State T4；batch增长改变了真实工作量，不能按8倍复制样例解释。
两者材料生成约23.1/23.3s、设备owner构造约1.72/1.69s，均在完整窗口计时之外。

轻量观察中，同权重`weighted_mm`的B1/B8分别40次/64行、49次/394行，平均1.60→8.04行，最大3→20。
私有BMM分别18/117次，全部2行；独立投影8MiB预算各退回1组、2/11行，保留原数学计算。
B1的28/40次所记录MM仅1行；参数规模本身没有形成大M矩阵批。这个计数覆盖公共packing投影，不是模型全部GEMM。
d4096有64个头，Full的8/17个可见桶产生512/1088次逐head attention桶调用；源码
`weighted_full_batch.cpp`的head循环是后续批量ATen候选，但没有大尺度profile占时证据，本轮不追加第三项优化。

native B1/B8构造peak allocated均8.727GiB，测量peak allocated为8.054/8.103GiB；
全程最高reserved约8.986/9.029GiB。恢复阶段记录保留已有峰值、不单独reset，因此不能称恢复独占峰值；
进程lifetime RSS峰值约10.55/11.03GiB，也不是每阶段主存用量。
参数+旧零buffer bank约8.03GiB，独立推理不会分配训练master/梯度/optimizer。

同一1B/B1材料的PyTorch/native/standalone中位分别239.231/223.885/330.295ms，三客户端均完整通过。
PyTorch最高reserved约8.074GiB；standalone测量reserved约8.984GiB，磁盘恢复最高reserved增至17.014GiB、
lifetime RSS约26.11GiB。其owner构造约28.98s含二进制张量读取/转换；Python材料生成与owner构造分别记录，
不能只把Python的约1.5–1.7s构造与C++这一段直接算启动倍率。该点standalone没有速度优势，也有更高恢复容量需求。
这些消费者均完整返回，具体对象构造/解码范围按各自记录保留；没有测最小返回模式。

f738692的d5888近界点亦完整通过：2,085,424,614个去重可训练参数，85个owner、98个角色，
native mixed-c FP64、B1/P4/span16。三窗504.040/384.317/375.775ms，中位384.317ms；
每窗14候选/8 Full/8消息，合计候选吞吐33.22/s，第4窗磁盘新owner精确续接通过。
材料生成52.75s、owner构造3.91s；构造peak allocated18.003GiB，测量16.600GiB，
全程最高reserved18.541GiB；恢复后记录的lifetime RSS峰值20.417GiB，峰值重置范围同前。
轻量观察的MM为36次/64行（均1.78、最大4），私有BMM18次全2行；独立投影预算退回1组/2行。
92个头×8个Full可见桶产生736次attention桶调用，State实际T4。更大参数量仍未产生大M矩阵批；
没有该尺度的完整profile，不能由调用数认定attention为占时瓶颈。

近界准入使用已通过B1最高reserved×新旧真实bank字节比，再加0.5GiB未观测运行时余量，
与现场free memory比较并保留4GiB。d6144预计约20.70GiB（含余量），超过可用预算，明确not-run。
d5888约19.05GiB（含余量），作为单独模型/工作负载准入；没有试探OOM或缩短缓存。
standalone事务恢复保留两份bank：1B约16.06GiB，d5888约33.14GiB，因此近界standalone静态拒绝。
resident目前不支持FP64；即便另选FP32，d4096的KV Full守卫下界仅`256×13×d²`就约52GiB/行，
尚未加cache项，当前24GiB卡不能准入。不是本轮实际OOM，也不盲目提高分区预算。
目录保留的d8192 FP32点因已有同一路径d256严格失败未解除而未运行；不能把另配FP64的耗时外推为FP32/TF32/FP16吞吐。

## 两项优化与六项处置

完整源码导航、采用/后置理由及重启条件见[performance-lessons](../performance-lessons.md#gpu-推理批次与完整返回)。
本轮仅实施已定位的两个有界候选，没有新通信协议、专用核、跨节点调度或反向模板工程。

| 六类问题 | 本轮结论与处置 |
| --- | --- |
| 1 相同长度分桶 | 小型FC B8→B32的Read/Full各21→22桶、89→341行，平均4.24→15.50行，单行桶消失；保留分桶。更多padding须有实际碎裂主导证据 |
| 2 元数据逐条回传 | detach bool按device/dtype合并。B8 native标量取回1380→1114；正式242.668→240.310ms，PyTorch308.299→311.968ms，未证明稳定端到端收益；停止继续扩成统一元数据层 |
| 3 重复打包/小复制 | 新增显式8MiB完整返回packing，逐字段CPU副本仍独立。B8三个客户端中位成本下降约12.2%/6.0%/17.2%，保留默认scalar和显式packed；内部重复summary gather/stack后置 |
| 4 兼容节点合批 | 同权重MM及私有BMM已有真实批量ATen，但私有BMM通常是不同(Wᵢ,xᵢ)对。大宽度实际预算回退另列，不提高预算冒充充分合批；更广State/Read/Full融合后置 |
| 5 resident填充/布局 | FC B8物理160/实际89、B32物理640/实际341，约44–47%缺席行；HB物理768/候选375。比例不等于耗时浪费；大宽度Full资源守卫保留，设备活动行/多时间布局需新方案 |
| 6 反向构图/模板 | 后置；journal、generation、根、参数地址与版本不可只按shape缓存。小型完整根、共享、梯度、optimizer与恢复回归分别通过，不宣称模板复用或大尺度训练通过 |

D→H的B8两Python入口，cudaMemcpyAsync与显式sync各少2788次，返回阶段sync由2792降到4。
独立profile显式CUDA同步区间并集占捕获跨度约2–3%；这不包含全部隐式等待/主机开销，也不是关键路径比例。
native返回阶段union反由106.411增至136.058ms，Python为181.778→58.639ms；不能由调用减少解释全部正式收益。
profile阶段存在嵌套重叠事件，采用区间并集，不累加inclusive时间。

## 原失败与回归边界

A的d256 FP32 B1/B8/B32全部严格失败：首个窗口KV keys 3/768坐标越界，最大绝对差2.0265579e-6，
原atol=1e-6/rtol=1e-5不变。自身恢复通过，67.688/267.957/843.971ms只作失败成本。
D两客户端首差诊断均为`strict_mismatch`，不能被外层passed覆盖。PyTorch最早在time1/batch0/node1新KV keys；
native同一最早事件在Full/emitted，首个严格KV写在time2/node2且上游fiber已不同。
CPU FP64局部锚点不能证明单改KV写即可修复整图；本轮采用独立全FP64配置，原候选停止扩大。

F原Half错误为检查材料`at::tensor`构造报`tensor_cpu not implemented for Half`，尚未进入返回复制。
H先构造double字面再显式转storage，三dtype检查通过；F/H生产benchmark二进制hash相同，checker另有构建身份。
GPU目标环境33项中9项直接CUDA，其余为CPU/静态消费者检查；不把总数称为33个GPU机制。
D三客户端选定训练9项覆盖独立raw/descriptor/control根、共享参数、四次更新及精确恢复。
H三客户端另验小型反馈PDG/KV FP64，SGD原lr/momentum/weight_decay、1warmup+2更新及下一窗精确恢复；
完整记录逐窗比较，最终梯度/参数/槽比较，不称每窗所有梯度保存或完整AdamW矩阵。
旧CPU/NPU strict xfail、standalone SSD/affine训练、workers2、双卡GDN精确恢复边界均未解除。
当前12方向P2P仍false，既有原生跨卡原子不支持事实保持；普通复制不认证多卡resident。

本机AArch64/Ascend当前Torch2.12/CANN9.1私有环境，在同一物理9上完成纯PyTorch相关对照。
F五个NPU字节/根检查和FP32小型HB训练/恢复通过；同FC d8/B8/P4推理的A/D/F中位为
559.661/527.660/576.500ms，D→F返回packing成本上升约9.26%。这是可迁移正确性和性能负结果，
不认证本机C++/standalone NPU返回策略，也不复制GPU矩阵。三个本机服务终态success/exit0且ControlGroup为空。

## 从干净检出复现

只clone授权新分支，匹配CUDA/Torch私有环境，按[development](../development.md)重建。
长命令使用[固定源码持久作业](../verification.md)；所有NEW路径须新建，先核对实际设备/主存/磁盘预算。

```sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TORCH_CUDA_ARCH_LIST=8.6 NVIDIA_TF32_OVERRIDE=0
python scripts/build.py --backend cuda --build-dir NEW_CORE --jobs 56
python scripts/build_weighted_benchmark.py --core-build NEW_CORE --output-dir NEW_BENCH --jobs 56
python scripts/run_core5_study.py --study benchmarks/core5-cuda-inference-v1.json \
  --case fc-small-b8-kv --execution native-cuda-return-packed \
  --native-library NEW_CORE --output-dir NEW_SMALL
python scripts/run_core5_study.py --study benchmarks/core5-cuda-inference-v1.json \
  --case fc-1b-b1-kv --execution native-cuda-fp64-return-packed \
  --native-library NEW_CORE --max-materialize-bytes 12884901888 --output-dir NEW_1B
python scripts/run_core5_study.py --study benchmarks/core5-cuda-inference-v1.json \
  --case fc-1b-b1-kv --execution cpp-cuda-fp64-return-packed \
  --standalone-binary NEW_BENCH/client/weighted-benchmark --export-library NEW_CORE \
  --checkpoint-max-bytes 9663676416 --max-materialize-bytes 12884901888 --output-dir NEW_CPP
python -S scripts/index_core5_study.py docs/evidence/core5-cuda-inference-20261031.json --format tsv
```

小型单卡resident入口（使用公共设备库，默认完整返回scalar）：

```sh
python scripts/build_device_control.py --core-build NEW_CORE --build-dir NEW_RESIDENT_PY \
  --runtime python --python-component weighted --cuda-architectures 86 --jobs 56
python scripts/run_core5_study.py --study benchmarks/core5-cuda-inference-v1.json \
  --case fc-small-b8-kv --execution native-resident --native-library NEW_CORE \
  --resident-library NEW_RESIDENT_PY --output-dir NEW_RESIDENT_RUN
```

standalone resident还需[独立设备库构建](core5-cuda-resident-20261031.md#干净检出复现)，
向新的benchmark构建传`--resident-build NEW_RESIDENT_CPP`，然后选择`cpp-resident`。
CPU/mixed-a/mixed-b/streaming同材料选择`native-cpu`/`native-mixed-a`/`native-mixed-b`/`native-streaming`。
PyTorch选择`python-cuda-fp64-return-packed`；B8选择`fc-1b-b8-kv`。近界选择
`fc-safe-boundary-fp64-b1-kv`，显式材料预算21474836480；这只是命令，不自动准入其他机器。
预算由`planning.inference_budget.fc_inference_budget`重算，再用实际1B峰值校准，保留4GiB设备/8GiB磁盘余量。
`scripts/benchmark_weighted.py --observe-work`另起一次轻量观察，`--profile`只用于有预算的小型代表，不能算合格吞吐。
这两个选项属于该benchmark入口，不可追加到上面的`run_core5_study.py`命令；具体展开命令保留在JSON的observations。
原FP32失败可选`fc-matrix-b8-kv/native-cuda`重建，随后用`diagnose_core5_inference.py`定位首差。
大checkpoint/fixture按已审阅hash和dry-run退役；Git保留配置、结果和生成入口，旧目录不作运行依赖。
本轮共退役59,095,122,182字节生成材料；最终目标机35份现存作业记录全部终态、无存活监督器/工作后代，原失败及停止探针保持原状态。
原始字节哈希只能标识历史材料，不能在退役后恢复文件或认证重建结果；新的构建/运行仍独立验收。
