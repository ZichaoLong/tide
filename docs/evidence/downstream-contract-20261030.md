# 多实验仓库协作规范与安装消费者资格

实现固定 `36113ce70553d847d0a255dedb182e906645816c`；数学固定
`0770593838a37c790a568226930567ed60f99b83`。本次交付[协作规范](../downstream-contract.md)、
依赖锁/请求/交付/验收/AGENTS模板，以及[可复制的core-5消费者](../../examples/consumer_core5/README.md)。
没有修改基座数学函数、执行器、图/checkpoint版本或容差，也没有新增性能实验。
[机器证据](downstream-contract-20261030.json)保留完整配置、安装/构建身份、各检查结果和失败记录。

## 固定源码结果

只读固定checkout，aarch64、Python3.12.13、Torch2.12.0+cpu：**4项单元检查、8组仓库外安装检查、4项CLI拒绝均通过**。
外层作业 `downstream-qualified` 为passed/0，2026-10-30 04:04:03–04:06:16 UTC，约132.38秒；这是检查耗时，不是性能数据。
服务 `downstream-qualified-20261030.service` 启动时核对PID2538045和独立background.slice/control group。
提交资源为Nice10、CPUQuota3200%、MemoryMax32GiB、900秒，不表示独占整机。
结束后瞬态unit已回收；以持久化作业终态、全部检查产物及hash核对为结果依据，没有保留journal记录。

| 实际用例 | 客户端与精度 | 调度/算子 | 组数 |
| --- | --- | --- | --- |
| FC两层、KV；主体4节点/2区域/4边 | PyTorch、Python→native，各FP32/FP64 | packed prefill，ATen | 4 |
| HB；主体8节点/4区域/16边；实际KV/GDN/SSD | PyTorch、Python→native，FP64 | packed prefill，ATen | 2 |
| 同一FC/KV | PyTorch、Python→native，FP64 | scalar streaming，ATen | 2 |

共同设置：CPU载荷/评分、worker1、ATen intraop/interop及BLAS线程1、autograd多线程true；
宽8、FFN宽32、区域宽2、B2/P2、seed7、region参数共享。
FC图可训练参数8,190，HB图16,976，应用线性head另有9个参数。
区域设置为read_mode=+、local_score=cat、region_score=ind、regional_total/context/cat/quad；
完整节点/区域默认值、窗口和逻辑时间尺度见机器记录的models，不能由宽度推断全部局部配置。

每组先核对真实边及实际执行的记忆族，再做两步独立时间主序参考对照：完整执行记录、状态/history/pending、
控制/消息/端口、输入与参数梯度含None/零、共享参数、SGD槽和StepLR。
应用使用随机数据、线性head/dropout、MSE、SGD(lr=.0001,momentum=.3,weight_decay=.01)，每窗反向后detach再更新。
FP64对照atol=1e-10/rtol=1e-8，FP32为1e-6/1e-5；整数和离散身份严格。
独立调度仍共用局部函数，不代替[语义索引](../semantic-map.md)中的独立公式/求解/梯度根证据。

每组另启动三个新进程：连续3步、前2步保存、重新构造owner加载后第3步。
完整最后观察值及解码后的应用checkpoint按atol=rtol=0逐值比较，均通过。
应用bundle包括图checkpoint实际字节、head、optimizer、scheduler、数据/目标/dropout RNG和进度；
不保存已消费的autograd图或上一步grad缓存，下一步从zero_grad开始。
三个进程也实际调用no-grad推理并检查有限值/shape；这不是单独的独立推理资格。

单元检查覆盖安装内容/环境拒绝、native binary/manifest/环境身份、原子不覆盖与写失败、错误应用身份/进度/时钟拒绝。
四项实际CLI拒绝为坏checkpoint、缺少native库、Python streaming+packed、错误安装内容锁；
均exit1，错误类型已核对，未留下最终输出目录。

## 安装与来源

资格从固定源码副本构建wheel，安装到新私有venv，再把示例和native adapter复制到源码目录外。
每组实际导入路径均落在新venv；不使用基座tests/scripts/PYTHONPATH作为应用依赖。
锁中的source、数学、协议、wheel/安装内容与native身份均保留；应用另记自己的工作负载和执行身份。
安装文件一致证明材料匹配，Git来源仍由此次固定源码构建流程和归档证明。

复用未改的CPU构建，来源 `b255e270eab38faa9c4348e19ed435ce02bbc4b4`；
C++源SHA256为 `d6640489b1351a9a4970f19bd57eed78a851afa8bc60cbc488d785126f8ffb42`，
adapter为 `95fbda055690bc78e43e64f4d4adf2a536f3a11dabd6ad8df754875a602251c8`。
门禁前后核对当前C++源和manifest全部二进制库存；没有重建或增加standalone资格。
源码洁净状态、全文件摘要、源码归档、wheel、安装内容、lock、日志、JUnit及检查产物hash均见机器证据。

## 保留的失败与范围

两个开发冻结树downstream-dev、downstream-dev-v2均failed/1；第三轮downstream-dev-v3单worker开发检查通过。
三者均为ad684b7加各自固定补丁，归档/hash分别记录，不能作为本次干净提交的资格。
前两轮FC/KV FP32 native workers2在独立两步对照通过后，新进程精确恢复失败：
最后观察中45个张量字段不等，涉及输入梯度、参数梯度和SGD槽；输入梯度11/32元素不等，最大绝对差4.656612873077393e-10。
第二轮仅关闭autograd多线程仍失败；数值很小也不满足逐值精确恢复标准。
原源归档、输入生成配置、连续/恢复观察、checkpoint及日志保留于各作业目录，未改失败状态或容差。

当前示例显式worker1是一项不同执行设置，**没有修复workers2边界**。具体根因尚未定位；
重启须有针对并行构图/反向累加顺序的具体诊断，再验证原材料、独立根和新进程恢复，见[ROADMAP](../ROADMAP.md)。
新进程比同进程新owner增加了运行时构图/初始化条件，旧同进程恢复证据不自动覆盖它。

本资格仅属于小型CPU安装示例。NPU/CUDA、mixed/resident、standalone、大模型、分片checkpoint和性能均未在本轮验收。
原大尺度数值/异步/资源限制保持。没有实际实验仓库提供，故没有宣称任何下游项目已验收；
上游交付、每个实验自己的验收和显式依赖升级继续分别记录。

## 从干净源码复现

采用上述实现提交，按[开发起步](../development.md)配置配套Torch、pytest和构建依赖。
不需要旧运行目录；已有构建可以复用，但门禁必须核对源码和binary身份。workload命令为：

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/build.py --backend cpu --build-dir build/cpu --jobs all
python scripts/check_downstream_consumer.py --build-dir build/cpu --output-dir artifacts/downstream-new-check
```

输出路径必须新建。长构建/检查仍在固定checkout、独立有界background.slice服务中运行，遵守[资格流程](../verification.md)。
新实验复制consumer_core5目录、安装固定wheel、生成依赖锁后运行check.py；用--recipe选择其小型代表配置。
调查workers2时可显式添加`--client python-native --native-library MATCHING_BUILD --dtype float32 --workers 2`；
保留原seed/recipe/严格恢复检查与失败产物，不能因某次通过推断并发恢复总是精确。
