# FC16 同尺寸归因、受控成本与当前候选

数学固定0770593838a37c790a568226930567ed60f99b83。
[审阅数据](fc16-cost-attribution-20261110.json)保留每个实际来源、构建hash、请求、阶段、工作量、
测试/作业及失败身份。此报告区分同函数投影、不同函数基线、相关正确性资格和完整规模成本。

## 材料与口径

原任务只读取tide-learning-lab提交`0439efd2af31cf10b07ce9e6423b0195f7449968`的
`configs/studies/language/v6/scale-fc16-kv-quad.json`与任务源码，不修改下游checkout/依赖锁。
固定16节点、4个串行区域×每区4节点，相邻区域全连接；B8/L128即8条序列、每条128个输入位置。
实际每步1024输入位置，16384候选作用；本轮CPU Full8344/消息25060，NPU Full8345/消息25064。
静态全连接不等于全部候选执行Full；k=2约束发送强度，不是硬Top2，不能按k推断活动数。

FP32、d128、2头、FFN512、summary/adapter/score/region hidden为32，qry/Read+/cat/ind/context/cat、
regional_total/quad、temperature=.25；初态来自绑定，窗口3072（末区无界），本次L128未截窗。
完整任务5,092,788可训练参数（core4,564,404），同embedding/head/交叉熵/AdamW/clip1。
合成输入由固定seed生成；一次预热后重置参数及optimizer，再测独立一步，不是连续学习实验。
真实语言数据/学习效果、完整原任务数值与离散独立认证、下游验收未包含。

图入口成本是Python任务调用C++核心/LibTorch，CPU1线程或单NPU mixed-c/主机2线程，
默认component-window、query auto→packed-v2、minimal返回。它不是resident，也不是独立C++任务耗时。
独立C++ SDK另测小型正确性/恢复。直接公式与Transformer使用纯PyTorch设备张量。
主机aarch64、Torch/TorchNPU2.12、CANN9.2；本轮不提供CUDA/x86、多卡或其他mixed位置的新结论。
全部可用320CPU仅用于构建，不能把CPU1线程成本当作320核吞吐。

完整训练计时含输入生成/上传、前向、loss、反向、有限值/梯度检查、clip、optimizer、清理和同步；
每个阶段边界显式同步。构造/编译、预热、单独diagnostic profile以及checkpoint磁盘检查分列。
这是有同步观测点的单进程任务延迟，不是异步服务/生产吞吐最优实现。
吞吐=1024/完整秒数，单位输入位置/秒，不是自回归生成token/秒。
TTFT另测warm-loaded prompt→最后位置head→argmax→有限检查→host返回；是整个B8的首token延迟，
不能除以8或128，也不能从训练前向推导。排队、tokenizer和模型加载不计入此局部TTFT。

## 同尺寸独立基线

仪表实现`09e027765873ce7164b1973e505f5b83c593f25f`；基线脚本与独立公式固定
`df80d043654664a611600bfdef457f0c7585e292`，图推理沿用09e同C++ hash构建。

| 实现 | 参数 | CPU完整训练秒 | NPU完整训练秒 | CPU / NPU 首token秒 |
| --- | ---: | ---: | ---: | ---: |
| 原默认FC16图入口 | 5,092,788 | 18.229637 | 22.998073 | 3.881163 / 4.966304 |
| 独立FC直接张量公式 | 5,092,788 | 2.146496 | 0.315203 | 0.560668 / 0.126653 |
| 16层Transformer | 4,722,688 | 0.776387 | 0.046361 | 0.168450 / 0.009837 |
| 4层Transformer | 1,576,960 | 0.262757 | 0.016948 | 未测 |

Transformer是串行pre-RMS、causal SDPA、无bias SwiGLU；16层匹配节点数，4层匹配串行深度。
它们和FC不是同函数，也未补虚假参数。FC还有Read描述/评分、短长记忆证据与控制、选定Full等职责，
不能理解成Transformer仅多一个分配求解。表中的差距不等价于可直接获得的生产加速比。
直接FC拥有独立区域循环、显式批量公式和独立二次分配区间求解，只覆盖其声明的零左状态/邻区FC域；
返回输出、控制/支持、实际终态KV投影，不提供通用trace、None根或checkpoint解码。
小型CPU5/NPU3检查核对独立标量调度、输出/控制/终态、数值输入/参数梯度；分配另有独立二分与FP64差分。
完整FC16同设备loss和计数接近/相同不认证完整数值等价；严格结论仍not_checked。

## 定位与候选取舍

可选原生阶段计时是调用线程的排他经过时间，含实际数值算子、提交与偶发等待，不是纯设备kernel时间。
不能把多worker计时直接相加为wall。正式成本不启用profile；另一次同材料profile用于归因。

| 原始09e阶段 | CPU秒 | NPU秒 |
| --- | ---: | ---: |
| 查询内部梯度 | 6.751 | 5.917 |
| 区域内部梯度 | 2.712 | 3.426 |
| 查询结果发布 | 0.241 | 1.744 |
| 区域前向分配数值与行包装 | 0.061 | 0.213 |
| 上层调度自身 | 0.911 | 1.200 |

前向分配求解不是当前最大的耗时。查询/区域反向中仍有大量私有根和行级装配；
这些阶段也含真实数学计算，计时本身不能把每一秒都归为纯管理开销。

`9fd82c3`把所有查询/区域私有字段打成列，CPU相关350项/安装恢复通过，却把完整步变为134.121秒，
反向126.856秒。原因是缓存长度桶横跨owner区间，逐view转置反复归零/累加整个大缓冲；
作业在诊断过程中被取消，外层cancelled/143、内层running快照保留，NPU未运行。
`c20aaa4`用一次index gather及其转置消除上述放大，CPU/NPU完整步恢复到19.462801/23.676150秒，
仍未优于原方案。CPU query-gradient反升8.400秒，region-gradient降至1.312秒；
NPU query-gradient5.249、region-gradient2.096秒，前向绑定复制成本增加抵消收益。
因此当前候选撤回查询列打包，只保留区域列。没有靠不同FC/HB/LH名称或节点阈值切换算法。

当前实现`1f3f62d69f3d89b1cc7661a5a509c6e382e9b51d`还把query/region输出逐条clone改为
foreach批量copy到独立分配。输出之间存储和version仍独立，resize/mutation不波及其他行；
原始输入版本、独立None/零根、共享参数和一阶VJP保持。它不是clone整批后返回共享view。

## 当前候选固定资格与成本

固定1f3f62d的独立Python/native和standalone SDK新构建成功；build workers=320。
`gef-fc16-attribution-final`终态passed/0、MainPID=0、control group为空，所有预期内层结果单独核对。
CPU358通过/2原数值xfail；NPU配置流程95通过，其中不读取设备环境变量的辅助检查仍在CPU。
覆盖查询独立字段/状态/参数、区域候选、重复输入/别名/None根、五记忆选定边界、FC/HB/PDG序列作用及恢复。
原调度未修改；非对齐TimedDAG、大尺度HB/LH、多卡或resident不能由该集合外推通过。

纯PyTorch和Python/native各在CPU FP32、NPU FP32完成小型独立三步训练及新进程精确恢复，共4项。
SDK客户端3项CTest和CPU FP64/FP32自定义kernel检查通过；SDK另外CPU FP64/NPU FP32×
complete/outputs共4项三步及新进程恢复通过。complete模式独立检查完整记录和各梯度根，
outputs模式认证实际选定路径的自身精确续接，不冒称未执行的独立数值检查。
不同query策略的恢复身份拒绝仍通过。大尺度FC16性能与这些小型资格分别成立。

| 当前默认入口 | CPU | NPU mixed-c |
| --- | ---: | ---: |
| 完整训练步秒 | 17.496945 | 21.064699 |
| 输入位置/秒（1024/完整步） | 58.52 | 48.61 |
| 相对本轮起点完整步下降 | 4.02% | 8.41% |
| 前向 / 反向秒 | 6.617 / 10.773 | 8.392 / 12.525 |
| warm-loaded 首token秒 | 4.140645 | 4.273320 |
| 推理完整请求秒 / 输入位置每秒 | 4.142771 / 247.18 | 4.277980 / 239.37 |

每点仅一次独立测量，不声称统计稳定。CPU首token较原3.881秒反升，不能声称推理普遍改善。
原NPU训练/阶段在物理11，锚点/原图推理在物理1，当前在物理3共享；队列两次采样、进程身份、
HBM余量与AICore预算及快照已保存。已授权共享时没有绕过锁或改变公共工具链，
但设备/宿主机争用未排除，百分比仅描述这次观察，不构成独占受控硬件结论。

单独阶段pass的CPU区域梯度2.712→1.348秒；NPU区域梯度3.426→2.123、
查询发布1.744→0.683秒。NPU查询梯度仍5.764秒，整体反向仍12.525秒。
36次foreach前端提交处理90265个独立输出（CPU90264），并非36个设备kernel或消除逐对象分配。
NPU训练峰值allocated约4198.45MiB（原4188.97MiB），gather拒绝候选6044.12MiB；
不能只按连续存储或调用次数宣称内存/整链性能改善。
当前通用图训练仍比直接FC公式慢很多；本轮没有达到“接近原生Transformer”的目标。

## 保留的失败与界限

- `098c91a`首次独立区间锚点在完整尺度拒绝可行支持；`df80d04`以中心坐标和最小可行距离修正，
  原failed/1不改写。独立小求解/差分和原有限成本重新完成，不把该基线作为通用模块替代。
- 新mutation测试最初同时断言另一种标量/批量FP32数学比较，出现约1.387e-6的evidence_weight差异，
  在未改复制的旧native同样存在。隔离测试改成同计算修改前后的精确VJP；数值门禁仍独立保持。
  复现`diagnose_dense_rounding.py --query-execution packed-v2 --witness publication`；当前独立query叶实现
  同FP32材料的scalar/packed最大差1.734e-7，两者对FP64最大误差6.248e-7；同材料FP64公式差约2.583e-15。
  这是指定局部见证，不解释或解除其他严格失败。
- NPU复制探针原先以设备signbit(-0)断言原输入而失败；改为host读取符号位后3项通过。
  没有更改数学输入或放宽复制比较。空闲队列30秒超时未执行，后继共享准入成功，两事实分列。
- 较早原始FC16失配、GPU、异步双NPU、resident工作区/生命周期及大尺度独立数值边界不由本轮解除。
  全部当前待办/重启条件仍仅见[ROADMAP](../ROADMAP.md)。用户要求本轮后先对齐，后续实施须新授权。

## 复现与维护

依[开发入口](../development.md)分别构建安装Python/native和standalone SDK，保持固定提交与工具链隔离。
在有限background.slice作业和有效设备准入内，原FC16命令如下；路径替换为当前新安装/输出，不依赖旧运行目录：

```sh
python scripts/benchmark_dense_query.py --downstream-repo DOWNSTREAM_GIT \
  --native-library MATCHING_BUILD --trace-records minimal --device cpu \
  --baseline graph --mode training --profile stages --output-dir NEW_OUTPUT
```

设备可选`npu:0`；`--baseline fc-tensor|transformer-16|transformer-4`加`--profile none`测独立基线；
`--mode inference`获得真实首token。`--batch`和`--length`仅在声明的B<=8/L<=128范围内用于有限诊断。
阶段profile不进入正式吞吐，整套参数/构建身份及实际工作保存在结果；不同返回契约不混算。

测试入口为`test_fc_tensor_baseline`、`test_stage_profile`、`test_private_rows`、`test_independent_copies`，
再加query/region/layout/root、local_batches、selected_boundary、kv_sequence_vjp相关门禁。
设备环境变量分别为`TIDE_FC_DEVICE`、`TIDE_DENSE_DEVICE`、`TIDE_WINDOW_DEVICE`；不读这些变量的辅助测试仍在CPU。
安装消费用`examples/consumer_core5/dense_query.py --no-trace --client python|native`；SDK用
`check_query_sdk.py --return-mode complete|outputs`，静态输入fixture生成见[此前完整命令](prefill-default-20261110.md)。
规范、实现、测试导航由[FC16契约](../fc16-cost-attribution.md)、[架构](../architecture.md)、
[复现索引](../reproduction.md)、[性能经验](../performance-lessons-prefill.md)共同连接。
