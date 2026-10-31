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

归档SHA256：`9067192c12bab6888cc5b94b27ee115c763163ef4d6694b4cccd84c99050a730`。
