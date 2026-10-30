# 带权实现的有限选型测量

[固定源码有限对照](evidence/core5-weighted-performance-20261028.md)记录24配置22通过、两项FP32反馈严格失败，
分别给出CPU、Python/native NPU与standalone结果、profile及测量缺口。它不认证所有模型/规模或CUDA真机。

[50a1bd3启动/内存补充](evidence/core5-weighted-observation-20261028.md)另验四个新进程和SDK allocator，不改写原24项测量。

[参数匹配及经验迁移](evidence/core5-performance-transfer-20261028.md)补五族预算、实际D2048材料、
共享存储/复制改动、同材料前后对照和新严格失败。参数数量相近不代表函数或实际工作量相同。

本页规定性能消费者的范围；运行结果从[证据索引](evidence/README.md)进入。
[历史与新性能经验](performance-lessons.md)分列LH、fractal、core-3和core-5的实际证据及适用条件；
[参数匹配扩展](weighted-workloads.md)提供五族、可扩展拓扑和二进制材料；下面的small/medium仍指原固定用例。
测量不扩展数学资格，也不代表模型效果或所有规模的性能。遵守
[执行契约 §6](execution-flows.md)，每配置默认一个独立进程，项目重型计时串行。

`scripts/benchmark_weighted.py`消费公共Python、Python调用native或resident接口。
`weighted_benchmark_cases.py`只构造固定输入、参数和拓扑，不执行参考轨迹。
配置包括非对齐TimedDAG、同秩HB SettleGraph与反馈PDG；KV/sum首组合或完整SSD，
默认FP32、SGD动量、原始时间窗口与短卷积。CPU eager另支持显式`--dtype float64`；
Python、调用native与standalone均从同一FP32种子/哈希材料在执行前提升，不重新随机初始化。
请求与生效dtype、输入精度及容差分别记录；FP64使用既有atol1e-10/rtol1e-8，不能替代原FP32资格。
此benchmark的FP64设备/resident请求提前拒绝，不静默替换。HB保留真实同秩区域，重复边逐条发送。
small为内容宽4、1样本、每窗2位置；medium为宽32、4样本、每窗4位置。
这两个有限尺寸用于判断调度与接口成本，不代表大型模型。

比较问题是：在这些窗口上，streaming/prefill、CPU/eager mixed/resident的完整调用成本如何；
规模变化是否足以抵消打包、设备提交与完整返回的成本。PDG只需native性能，Python作为独立正确性参考。
常驻训练、Python适配与standalone LibTorch必须分别记录；本脚本不能冒充standalone进程。

每次进程只做一个warmup窗口和默认三个连续测量窗口；逻辑时间和状态连续推进。
训练在每个optimizer边界显式detach，参数不重置。loss仅为实际输出数值投影cG的平方和乘.001，
用于验证完整训练机制，不是下游训练配方。记录候选更新、活动Full、消息、输出、pending与原始调度统计。
发送稀疏不能直接解释为所有局部计算变少，resident统计同时给出物理局部/Full行数。

计时包含输入构造/上传、在线调度、packing、前向、完整CPU返回、loss、反向、有限性检查、
SGD更新、切断续接梯度和最终同步。不同接口把完整返回变为Python对象的成本会保留在测量里。
fixture、模型/owner/optimizer构造、warmup、资格对照及最后参数导出分列或排除。
不把调用裸kernel或只计forward当完整训练。记录实际配置、源码与输入hash、栈、分配器峰值、
进程生命周期RSS峰值、负载均值和已知争用；不据单次流程声称统计稳定或普遍最快。

测量结束后才执行独立CPU时间递归，核对全部窗口记录、loss和最终参数值。
CPU eager/native另导出实际最终参数梯度与SGD动量槽，保留None/连接零，分别对照。
Python还比较完整optimizer state_dict；standalone的本入口只有固定SGD组，比较所有owner的动量槽。
最后导出在计时之外；它认证最后窗口梯度/更新槽，不声称逐窗全部梯度都已保存。
当前resident入口解码实际参数梯度、连接标志、SGD槽和FP32 master，并逐项独立核对；
Python及standalone均可显式检查磁盘解码后的新owner下一窗口。其完整训练门禁仍由
[训练资格](resident-weighted-training.md)给出。旧报告同名optimizer标志
实际只核对了更新后的参数，不能据此推断当时已验证动量槽；历史报告保留原始记录。
它不向候选提供选定序列、批次或容量。严格失败保留为失败，不能在该配置上宣称等价加速。
此有限对照不代替局部手算VJP、独立根、共享参数、checkpoint及完整设备资格。
验证失败时也保留已完成的`metrics.json`；d1787e8起同时保留在最终`result.json`。
这些时间是失败候选的成本观察，不能用来宣称等价加速。

示例（干净固定源码，匹配本机环境与持久任务规范）：

```sh
python scripts/benchmark_weighted.py --device cpu --implementation python \
  --topology hb --family kv --size small --mode training --output-dir NEW_RUN
python scripts/benchmark_weighted.py --device cpu --implementation native --native-library CPU_CORE \
  --topology dag --family kv --size medium --schedule prefill --output-dir NEW_RUN
python scripts/benchmark_weighted.py --device npu:0 --implementation resident \
  --native-library PYTHON_CORE --resident-library PYTHON_RESIDENT \
  --topology hb --family ssd --size small --mode training --output-dir NEW_RUN
```

`--profile`必须在另一个独立进程/目录运行，先做独立warmup，再显式完成单个采集周期，只记录一个profile窗口；导出的CPU/CANN/CUDA trace
用于判断实际算子、copy和host scalar成本，其带profiling开销的执行不作为吞吐样本。
profile进程另标记prepare、forward、loss、decode/roots、backward/finite、optimizer和完整返回/同步阶段。
报告包含阶段墙钟、kernel时间覆盖及copy/scalar耗时；嵌套host事件不相加作为独占时间，
kernel覆盖也不等于已证明的关键路径瓶颈。非profile测量不启用这些标记。
`--allow-dirty`只供入口开发检查，输出保留完整dirty清单，不作固定源码资格或选型证据。
源码身份在设备初始化前核对；运行工作目录切到独立输出目录，CANN初始化产生的
临时`kernel_meta/`及profile scratch也归于artifacts。相对库路径先解析为绝对路径。
没有CUDA真机时只记录待验，不运行CPU代替CUDA结果。

独立LibTorch入口在`tools/weighted_benchmark`。静态fixture由
`export_weighted_benchmark.py`导出实际参数TensorImpl别名、完整图/端口、配置与输入；
它不执行图，不包含路由或作用序列。C++解码重新核对完整图identity，安装公共数学函数，
候选使用自己的Streaming/Greedy或resident程序；运行时不链接Python。
同一fixture hash用于Python/standalone对照，工具链版本与接口成本仍需分别解释。
这一有限benchmark数据格式不代替通用图序列化或checkpoint。

```sh
python scripts/export_weighted_benchmark.py --native-library CPU_CORE \
  --topology dag --family kv --size small --output NEW_FIXTURE_JSON
python scripts/build_weighted_benchmark.py --core-build CORE --output-dir NEW_BUILD
NEW_BUILD/client/weighted-benchmark --fixture NEW_FIXTURE_JSON --device cpu \
  --implementation native --schedule prefill --mode training --output-dir NEW_RUN
```

常驻standalone构建另给`--resident-build STANDALONE_RESIDENT`，运行显式指定
`--device npu:0 --implementation resident`。构建器校验core/backend与源码、binary hash、
无Python依赖的loader和help；这些检查本身没有真机或性能结论。
C++计时后使用另一个独立scalar调度核对全部记录、loss、最终SGD参数及上述最后梯度/动量，局部函数锚点沿语义索引追溯。
训练计时包含新建独立输入、完整CPU返回、loss、VJP、finite checks、SGD和同步；
结果文件采用fsynced staging及原子替换，存在的运行目录直接拒绝。

## 启动与内存观测

`scripts/measure_weighted_process.py --output-dir NEW_OBSERVATION --timeout-seconds 180 -- COMMAND...`
在新进程外部观测启动到`TIDE_WEIGHTED_RUNTIME_READY_V1`的时间。两个消费者在校验/加载
所请求库、完成首个1×1设备矩阵乘法及同步之后、fixture构造之前发出该标记。
该值包含exec、动态loader/Python导入、参数/源码检查和管道送达成本；是新进程启动，
没有清空OS文件缓存，不声称隔离测出了loader耗时。标记之后还必须正常结束，缺失、
重复标记、超时或非零退出均失败。外部总进程时间包含事后独立正确性对照，不是吞吐时间。
`runtime_initialization_seconds`仅计程序内显式运行时/库校验及首个算子，
不含main之前的C++动态链接或Python最初导入。两种语言的初始化分界不同，需分别解释。
fixture准备、owner构造与warmup各自单列；新版本将独立参考的静态绑定延至测量之后，
避免候选测量期间保留第二份大模型。旧证据仍按其固定版本的构造/内存边界解释。

`weighted_benchmark_observation.py::Memory`与C++`observation.cpp::capture_memory`
记录runtime_ready、construction、warmup、measurement四个边界，前三个边界读取后重置
设备峰值。measurement峰值包含整个测量窗组，排除最后参数导出及事后参考；每个阶段
峰值仍包含此前保留的活动分配。采样在单步计时之外，当前/峰值allocated/reserved都保留。
CPU仅记录进程生命周期RSS峰值，不能从差值恢复每阶段真实峰值。设备分配器不包括所有
厂商/驱动分配，也不替代驱动空闲内存或容量准入。观测失效明确失败，不把缺失指标写为零。
`test_weighted_process_observation.py`验证真实子进程成功、缺失/重复标记、非零退出及超时记录；
设备计数器与实际完整训练分别用固定源码的有限补充测量验证。

standalone消费者在fixture读取、模型构造、warmup、每个measurement窗口、验证准备和
独立核对前持久化`phase`/`window`，成功后为`phase=complete`。阶段写入使用已有fsync/原子替换，
在单步同步/计时之前执行，不放进完整训练计时。进程被强制终止时，最后结果可以仍为running；
外层job的退出码决定终态，不能仅凭部分metrics认定通过。`test_weighted_benchmark_material.py`
用阻塞FIFO与真实子进程中断验证最后阶段可读，另验五族完整训练及损坏材料的失败阶段。

[单行VJP/字段packing/常驻因子KV对照](evidence/core5-boundary-performance-20261029.md)
保留五项通过、D64两项严格数值失败和一项构造容量拒绝；自身精确恢复与独立正确性分别判定。

自然参数规模、可配置局部候选及统一执行/结果目录从[core5-study](core5-study.md)进入。
[878a5a4自然规模对照](evidence/core5-natural-study-20261029.md)给出实际拓扑、维度、稀疏预算、
客户端/CPU/mixed/resident、真实packing及严格结果；[9bfabbb大文件复验](evidence/core5-checkpoint-budget-20261029.md)
另验5.25GB保存/加载和下一窗数值失败。两者均不扩张未测配置或8B/16B的吞吐资格。
三个client均保留显式FP16、FP32 scoring/optimizer master、loss scale=1；半精度使用既有
atol=1e-3/rtol=2e-2，整数/路由/连接关系与同配置恢复仍严格核对。
请求和生效packing、线程、scoring、放置与peer分别记录。standalone内存表包含每个peer，
不是只报coordinator；运行时允许half的声明与实际设备资格仍分开。
CPU FP16先从相同FP32材料转换，materialize准入保守保留FP32初始化的存储下界。

resident FP16核对使用`qualification_weighted_precision.reference_model`的独立CPU时间递归：
FP32算术/梯度，实际raw/payload边界显式half store，SGD独立FP32 master更新后再发布half。
它复用原设备资格的精度参考，检查公开存储dtype后无损展开比较全部字段；不能把中间字段
强制转half来掩盖算术政策差异。standalone使用已有`weighted_precision_reference`及独立C++调度。
报告`reference_precision`区分该契约与eager同dtype参考；这不是放宽容差或修复原FP16 GDN/AdamW边界。
resident恢复检查先保存实际CPU checkpoint及下一窗口结果，关闭旧owner后再建新owner，
避免仅为验收同时保留两套设备程序/参数bank；磁盘raw解码和完整下一窗核对仍精确，且在计时外。
