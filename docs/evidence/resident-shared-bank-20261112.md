# Resident共享矩阵、局部容量与服务停止

数学仍固定core-5 `0770593`。共享矩阵实现为`3e926edec3eda220ffb77aeb475688b9b56aceb2`，
容量见证修正为`72c1dc02fe62e722efa05015ce445fcfaf70f21a`、`d068fbdf3ba6aae87149232b7e8620558cb20d99`；
父子服务停止实现为`e12aec0c8acb8553de0bc021dab8a1f59db92710`。
[机器记录](resident-shared-bank-20261112.json)保留固定构建、逐项结果、完整配置、失败和现场日志摘要。
[21文件审阅](resident-shared-bank-20261112-reviews.json)绑定实际内容hash，与过去审阅重叠；
自动清单2157文件/5060提取符号不等于全部已审，C++符号解析未运行。

## 数值与存储改动

同一参数bank的R个输入现在计算`Y=XWᵀ+b`，前向只保留一份实际矩阵及其转置；
反向直接计算`barX=barY W`、`barW=barYᵀ X`和`barb=sum(barY)`。
这减少了逐行权重复制和逐行外积，并未合并不同输入根、节点私有状态或不同参数bank。
独立bool根、未使用的非有限参数、连接零与None、保存版本、FP16存储/FP32算术继续区分。
KV query、Full、Residual的组合预留相应分离共享矩阵项，并在构造末尾检查实际storage union增量；
owner总预算、未分配journal与vendor workspace没有放宽。保存tape只枚举借用量，不用参考重算。

| 实际见证 | 前向新增保留字节 | 反向新增保留字节 | 局部额度 |
| --- | ---: | ---: | ---: |
| 单bank投影，R64/d32/o32，FP32 | 34,432 | 28,993 | 65,536 |
| KV query，R4/cache49/d128/summary32，FP32 | 939,552 | 794,555 | 16,777,216 |
| 同一KV query，FP16 raw存储/FP32运算 | 1,168,192 | 794,555 | 16,777,216 |

第一行旧实现仅两份逐行矩阵就需524,288字节；这不是旧完整模型的实测HBM。
KV的原保守前向预留为213,600,256字节，新FP32预留11,225,088字节；表中是另行测得的实际新增量。
KV见证与原FC16容量失败具有相同局部尺寸，使用单条可见历史的字面公式，分别检查两通道返回、
raw值/权重梯度、参数梯度及结构连接，不冒充原FC16全模型等价。

固定3e926ede的CPU检查为32 passed、6未指定设备skip；后六项由实际NPU安装检查补验。
独立SDK的16个组件/dtype检查有14 passed、2 failed。失败是新容量见证未初始化map字段；
72c1dc02补字段后，又发现其`at::tensor(0.25)`期望值为[1]而实际总量为标量[]。
d068fbdf使用`scalar_tensor`后，KV query两dtype全部通过，生产计算和原容差未改变。
两个原作业仍failed，不把补验回填成原始通过。

其余通过包括Projection、KV Full/Update、四循环、Describe及完整图/训练。
flow每dtype21个小图组合覆盖五记忆、异构、FC/HB、非对齐TimedDAG和反馈PDG、三Read快照及完整raw恢复；
training每dtype5配置/20次更新，包含SGD/AdamW、共享、重复端口、空窗和恢复。
安装后独立cwd另有10项NPU检查通过：六个raw字段/dtype独立公式根，以及KV-FC/SSD-PDG两dtype
三步训练与磁盘换owner恢复。它们不能替代原规模的独立完整数值检查。

## 原预算复验与定位

两个代表均为FP32、B4/L16、d128、staggered-v1、实际61输入位置，python-native resident、
主机ATen2/interop1、SGD、输出平方loss尺度1e-6，一次预热一次测量及精确下一窗。
FC为4层×4节点、KV qry；SSD为4层×1节点、state读和原始时间卷积、固定总量。
详细k、区域配置、参数量、四分区和总预算在机器记录中。
这与主FC16 B8/L128三层比较的工作负载不同，不能混用吞吐。

72c1dc02固定安装下，FC16原18,522,046,464字节总预算成功构造，构造后allocated为1,182,845,952字节，
但warmup报`await local service / 507002`，没有有效训练吞吐或恢复结论。
厂商同PID日志明确`EE1019 / stream task buffer full`，不是显存耗尽或流总数上限的证据。
SSD原6,241,124,352字节总预算完成训练与换owner精确下一窗；单次完整训练13.345731秒，
61/秒数=4.5708输入位置/s，独立规模数值为not_checked。保存与恢复在计时外。
最后记录的反向统计中，构造0.877046秒、执行9.776194秒；这些分项不是一次完整步的排他profile，
不把13.35秒全部归给接口或构造，也不声称resident性能目标已经完成。

c9775479的通用阶段诊断调用同一安装/配置的真实公共step：construct、advance和cotangents完成，
backward失败，close完成。前向实际915候选/276 Full、824消息、19个resident阶段；100个服务，
这不是完整B×L张量窗口。额外metrics读取和捕获trace使该进程仅作诊断，不作吞吐。
捕获流522最后记录task2045，随后NOTIFY_WAIT报EE1019；该编号不是通用硬件上限。
`DeviceProgram::kernel`保存回调，在`finish`才提交；结合调用链，错误位于反向程序构造提交，
尚未执行其VJP或optimizer。

## 父子服务停止修正

原coordinator在退出时为所有后代逐一提交copy/request/wait，深层服务也占用顶层流命令。
e12aec0c改为父服务在空闲入口收到stop后依次关闭自己的直接子树，最后确认完成；
coordinator只关闭直接子服务。正常数值调用、设备在线推进、workspace先后关系及容量不变，
没有主机逐事件flush，也没有减少模型输入、改成mixed-c或增加预算。
这不减少服务流总数，不承诺解除其他图的流数量或任务容量限制。

开发真机检查已通过双根、兄弟、三层嵌套与3/0/5次循环的重复窗口和独立数值递推。
固定e12aec0c后分别重建Python-owned resident及独立SDK；复用core保留原构建身份，
与冻结源码的完整C++ hash一致。SDK六项通过：services、flow两dtype、window FP32、training两dtype。
两dtype训练各包含5配置/20次更新、SGD/AdamW与恢复；安装后另10项公共根/三步训练/恢复通过。

原预算FC16再次成功构造，allocated仍为1,182,845,952字节，warmup仍报`507002`。
新进程PID3123702的厂商日志为`EE1019 / stream task buffer full`，失败操作为流26的NOTIFY_WAIT。
这次完整成本入口只记录到warmup边界，没有另跑trace，不能把旧诊断的流522/task2045套到新运行。
父子停止修正因此只能宣称局部服务契约通过，**没有解除原FC16规模限制**。
同配置停止继续重试，没有增加预算、减少输入或启用launch blocking。

SSD同预算再次完成训练及磁盘换owner精确下一窗：单次完整步13.328988秒，
61个实际输入位置，折合4.57649位置/s；独立规模数值仍为not_checked。
其244候选/244 Full、183消息、19个设备阶段、28个服务与此前记录一致。
两次成本13.345731/13.328988秒不能证明性能改善；最后反向统计的构造0.884983秒、
执行9.767101秒亦不是该完整步的排他分解。

`gef-resident-stop-qualified`最终为failed，原因是FC16失败；SDK、安装和SSD各自通过并列保留。
服务终态MainPID=0、ControlGroup为空，无遗留作业。此阶段闭合了共享矩阵/借用计费的小型资格及
原预算失败边界，未完成resident整段B×L列调度、原FC16吞吐、多卡异步或全仓审阅。

## 重现与维护

按[开发入口](../development.md)分别准备匹配Python-owned core/resident和独立SDK，不能混用wheel/SDK库。
在固定源码和依法准入的设备环境中：

```sh
python scripts/build_device_control.py --core-build SDK_CORE --build-dir NEW_BUILD --ascendc-soc Ascend910_9392 --jobs all --checks services weighted-projection weighted-kv-query weighted-kv-full weighted-flow weighted-window weighted-training
python scripts/verify_device_control.py --build-dir NEW_BUILD --output-dir NEW_GATE --device npu:0 --checks services weighted-projection weighted-kv-query weighted-kv-full weighted-flow weighted-window weighted-training --keep-going
TIDE_RESIDENT_DEVICE=npu:0 TIDE_BUILD_DIR=PYTHON_CORE TIDE_RESIDENT_LIBRARY=PYTHON_RESIDENT python -m pytest -q tests/test_resident_weighted_storage_roots.py tests/test_resident_weighted_training.py -k 'public_raw or kv-fc or ssd-pdg' --dtype float32
```

安装测试自身参数化两种storage dtype；全局`--dtype float32`没有删去它们的half用例。
将机器记录`representatives`中所需行的`configuration`写入新JSON，用当前私有site指定库，再运行
`python -m tidegraph run CONFIG --site SITE --output-dir NEW`。
同名配置的旧/新结果由`job`区分；机器记录还保存展开后的局部/区域配置、参数清单总量和全部预算。
这些是失败复现材料，不授权自动重跑已停止的候选。
阶段诊断用`python scripts/trace_resident_step.py CONFIG --site SITE --output-dir NEW`；
显式`TIDE_CANN_TRACE_OPS=1`保存捕获流信息。原失败与新实现分目录，保留各自身份。

长任务使用固定源码、background.slice、明确资源和时限；构建使用320 workers，早先client构建
使用304以避开同时进行的设备检查。性能阶段无项目内重构建，实际host affinity144–159，
空闲NPU物理11映射npu:0；共享卡未使用，外部CPU争用未排除。无存储阻塞。
CUDA通用数值源码、CPU生成语义核检查、NPU构建/运行分别记录；本轮没有GPU真机或多卡resident资格。
未解除项和下一行动只在[ROADMAP](../ROADMAP.md)，本报告不成为另一份待办表。
