# 约1B训练容量与首个非有限字段（2026-10-31）

本报告包括CUDA及NPU原约1B单/双卡训练，后继字段定位单独记录。固定数学与原loss、初始化、输入、SGD不变。
原实现bd2ff2e；失败定位实现31a3b32只在非有限拒绝后检查字段，正常接受条件不变。
固定31a3b32相关40项开发检查通过；本组两个小训练均在真机严格通过完整记录、原loss参数梯度、参数、optimizer及磁盘精确续接。
真实core/resident库来源de999c2，未冒称按新Python提交重新构建。

RTX3090×4、Torch2.12.1+cu130、FP32 payload/scoring、TF32关闭；native mixed-c、packed/scalar完整返回，ATen/interop/worker均1。
容量模型为FC16/8区域、KV/d2048、1,014,118,392个去重可训练参数、8个不可拆共享组，B1/P2/span20。
原loss为weighted-output-square×0.001；SGD lr0.001/momentum0.3/weight_decay0.01。
每配置warmup1窗+连续2窗，1800秒上限，结果没有自动重试或更换配方。

| 配置/来源 | 执行结果 | 已完成测量窗成本 | 已观察逐卡最大allocator bytes | 独立验收/恢复 |
| --- | --- | ---: | --- | --- |
| 单卡/bd2ff2e | 静态拒绝：预计33,024,605,104 bytes，现场可用20,725,170,176 bytes | — | 未分配 | 未检查 |
| 双卡/bd2ff2e | 第三窗非有限，原未定位错误保留 | 1.167410s | 6,144,512,000 / 7,558,612,992 | 未完成；只保存前两窗checkpoint |
| 双卡/31a3b32诊断 | 原失败重现，路径runtime.forward_and_loss[loss] | 1.295268s | 6,144,512,000 / 7,558,612,992 | 未完成；只保存前两窗checkpoint |
| 四卡/31a3b32 | 同一loss字段第三窗非有限 | 1.316339s | 3,434,234,880 / 4,141,319,680 / 2,963,416,064 / 3,434,332,672 | 未完成；只保存前两窗checkpoint |

表中峰值是已完成阶段的已观察值，失败阶段未采样，不声称完整训练峰值或容量资格。
前两个loss为6,330,525及749,825,591,083,008；第三窗所有已检查前向记录有限，但loss非有限，尚未进入该窗反向。
这定位了第一处被拒绝的公开字段，未证明每一个上游算子都无放大。不能称GPU通信失败，也不能以推理通过解除训练失败。
两次诊断的已完成窗耗时是失败成本，不是合格训练吞吐；checkpoint保存不证明后续精确续接。
原先单卡预算拒绝属于保守静态准入，不是实际OOM；双/四卡数值失败使本轮继续缩减该训练预算没有容量选型收益，停止同配方扩大。
重启须有具体数值或训练配方提案；新loss/初始化/lr是新工作负载，不称原失败修复。

两个独立小训练为FC4/d8/B2/P3，finite-check scalar/packed各自完整严格通过，成本0.338520/0.277204秒。
它们仅回归失败定位实现不改变正常训练路径，不能替代上述1B资格。
三组持久任务分别于06:27:30Z、06:32:58Z、06:34:58Z以exit1结束，独立SSH检查无活跃后代。
没有项目重型构建重叠，外部争用未排除。原始失败日志、配置、来源、结果保留；可再生checkpoint经审阅后按产物规则退役。

```sh
tidegraph plan configs/matrices/cuda-capacity.json --matrix --site site.json
# 从矩阵选择相应case，使用新目录；不要把整个矩阵外层结束视为资格。
tidegraph matrix configs/matrices/cuda-capacity.json --site site.json --output-dir jobs/new-training-capacity --detach
tidegraph results docs/evidence/core5-platform-training-capacity-20261031.json.gz --format table
```

## NPU原流程的独立失败

NPU Python环境为Torch2.12.0+cpu/TorchNPU2.12.0.post2、CANN9.1；core源eca070f，Python安装bd2ff2e。
相同FC16/d2048/1.014B、B1/P2/span20、FP32、HF32关闭、mixed-c、packed/scalar完整返回及原loss/SGD，ATen/interop/worker1。
单卡和双卡都通过现场资源准入，第三窗被NonFiniteMismatch拒绝；原错误只定位到runtime。

| NPU卡数 | 已完成计时窗秒 | 已观察allocated峰值bytes | 前两窗loss | 精确恢复/独立参考 |
| ---: | ---: | --- | --- | --- |
| 1 | 1.397913 | 13,674,444,288 | 6,330,527 / 749,830,020,268,032 | 均未执行 |
| 2 | 1.999131 | 6,127,611,392 / 7,541,734,912 | 6,330,527 / 749,830,221,594,624 | 均未执行 |

两项只完成两个窗口的checkpoint发布（next_window=2），每代7,956,480,794 bytes。发布不代表精确恢复通过。
结果及各自supervisor终态failed、无剩余后代已核对；父任务仍执行其他独立case不改变这两个失败。
不将训练失败归因于通信或容量，也不由相同设备上的推理通过解除；后继仅一次31a3b32字段诊断，原失败保留。
原配方保持，停止同配方的大规模训练扩展；它们不提供合格训练吞吐或完整训练峰值。

## NPU失败字段定位与小型回归

31a3b32小型原loss训练两窗0.494196/0.447357秒（中位0.470776），完整独立数值、离散、参数/梯度/optimizer与精确恢复通过。
同源码双卡1.014B原配方第三窗重现非有限，准确拒绝字段为`runtime.forward_and_loss[loss]`；该窗反向尚未进入。
唯一已完成计时窗2.114907秒属于失败成本；独立参考和精确恢复未执行。此定位与CUDA同字段，但不将平台之间结果互相认证。
不改loss、初始化、SGD或阈值；停止该原配方进一步训练规模扩展，推理结果独立保留。
两子任务终态、无剩余后代。固定输入来自`npu-finite.json`的小训练及`npu-capacity.json`的双卡原训练；31a3b32只在拒绝后定位字段。

## 明确不同的新配方：CUDA双卡1.014B

配置`configs/capacity-training/cuda-scaled-loss.json`固定于1a6b08f，实际Python包36baeb7；
模型/workload/execution身份与原配方相同，task/config身份不同。只将loss系数0.001改为1e-9，
原SGD、初始化、输入、共享组、B1/P2/span20、FP32/TF32关闭、ordinary、packed/scalar完整返回及线程1保持。
这是独立工作负载，不是修复原第三窗非有限；本轮只试这一系数，不继续搜索。

先行小型双卡`check --scope full`通过完整推理、十组独立参数/外部根、三步训练及精确恢复；
结果SHA256 `1df6ddb5cd6b8c84f46016d326273aefbfe2752dd1b4474fc456360d1971b8fc`。
其有限范围与完整check记录在[同一检查归档](core5-platform-multicard-checks-20261031.json)的明确后继，不能认证1B独立根的全部组合。

| 新配方实际结果 | 数值 |
| --- | --- |
| 执行 / 独立数值 / 离散 / 磁盘精确下一窗 | 完成 / 超差探索 / 通过 / 通过 |
| warmup及两测量窗loss | 6.330525 / 6.210245 / 6.059679 |
| 两测量窗 / 中位 | 1.260121 / 1.397670 s；1.328896 s |
| 每测量窗candidate / Full / 消息 | 31 / 16 / 28 |
| 每卡全流程已采allocated峰值 | 6.146766 / 7.558613 GB |
| 每卡reserved峰值 | 6.188696 / 7.602176 GB |
| RSS高水位 | 40.264876 GB |
| 构造 / 精确恢复 / 独立参考 / 整流程 | 10.512 / 90.164 / 127.379 / 330.288 s |
| 每窗保存 / 最后单代checkpoint | 31.031 / 29.233 / 29.180 s；7,721,323,530 bytes |

三窗完整记录均有原阈值下数值差异，第一处仍为window0/sample0/node0 KV keys（35/2048坐标超差）；
三个loss及最终355个参数张量的独立比较通过。最终梯度/optimizer整体检查有97个数值差异、0离散差异，
第一见证为`training[gradients][nodes.0.kernel.base]`，最大绝对差4.84893e-6；已截断差异清单不能单独认证所有optimizer槽。
完整离散/None结构检查通过、任意数值超差未改为通过，不提供严格合格训练吞吐。
自身exact-next逐值核对完整解码、下一窗、参数、梯度、optimizer和任务状态，不能替代独立数值检查。

原生通信两测量窗各14组/458976 bytes，包含已插桩前后向transport；Python边界另4次/32784 bytes。
与前向静态7组不同，不能混作同一口径；ordinary请求/生效相符，参数/私有状态/消息目标和计算符合两卡映射。
原主存静态估算35.261GB低于本次恢复RSS40.265GB（尚不含全部文件页），说明训练保留的参数/梯度/槽CPU快照须补预算；
现场可用主存约249GB，因此本次未触发容量失败，但旧估算不能当近界保证。后续预算修正不会改写本次来源。

外层于08:55:33Z exit0，两个子监督exit0，重连检查所有后代均为空。没有重型构建与计时重叠。
原始结果SHA256 `01c4f8dfa963725236d2abb2a96a53b444c00cc99e9caf0227fdd36c3d306523`。
复现：先`plan configs/capacity-training/cuda-scaled-loss.json --matrix --site site.json`，
再用已安装`tidegraph matrix`与新输出目录；来源/安装身份和数学阈值必须显式保留。

归档SHA256：`be7223569aa1b77e843739f862e129bc9145610d40d39d20ee719e24929b1f7a`。
