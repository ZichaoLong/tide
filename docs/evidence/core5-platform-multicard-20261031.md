# 同模型多卡：CUDA/NPU实际放置、通信与训练（2026-10-31）

固定Python实现`bd2ff2ed6786c57f492b6ac8b83b4b3a4b0d821e`；core/resident构建源de999c2，standalone源eca070f，本组仅实际运行Python/native eager。旧库来源/hash与安装内容分开保存在归档，不声称重新构建。RTX3090×4、Torch2.12.1+cu130，FP32/FP32 scoring、TF32关闭、ATen/interop/worker均1、mixed-c/packed/scalar完整返回。每配置warmup1窗、连续2窗、独立参考与磁盘精确续接。一次有限流程，不宣称统计稳定性。

CUDA六项全部执行完成、离散/完整记录检查及精确续接通过。三个1B推理点保留数值超差探索结果；三个小型训练严格通过。原严格阈值、旧gated-HB多卡恢复失败及异步边界不因此解除。

## 同一模型的推理放置

FC16节点/8区域、KV、d2048、FFN4d，1,014,118,392个去重可训练参数，8个不可拆共享组，B1/P2/span20。各卡数的model_id/workload_id相同；参数不复制成独立变量。按共享组确定性平衡放置。三次每窗均31 candidate、16 Full、28消息；State最大样例批1、时间长2。

| 卡数 | 实际节点分配 | 窗口中位秒 | 全程逐卡allocator峰值bytes | 数值/离散/恢复 |
| ---: | --- | ---: | --- | --- |
| 2 | 8/8 | 0.416659 | 2288798208 / 2288798720 | 超差 / 通过 / 精确 |
| 3 | 6/6/4 | 0.416816 | 1748088320 / 1748088832 / 1207379968 | 超差 / 通过 / 精确 |
| 4 | 4/4/4/4 | 0.370152 | 1207378432 / 1207379968 / 1207379968 / 1207378944 | 超差 / 通过 / 精确 |

相同单卡模型在规模组中位0.437248秒；多卡2/3/4分别0.416659/0.416816/0.370152秒，仅为有限成本观察，未严格等价，不称可推广加速。逐节点参数owner、实际计算回调统计、state slots与逐目标消息位置都记录并审阅，与请求映射相符；空pending明确为空，不据空集合推断其他窗口。节点分配2卡为8+8，3卡6+6+4，4卡4+4+4+4。原loss下共享参数梯度贡献另由小训练验收；任意独立VJP根由单独公共full检查验收，不能从loss反向推断。

三个推理点每测量窗的native跨卡传输均7个copy group/28行/229488 logical bytes，普通复制记录同量tensor-leg bytes；Python边界另2次/16392 bytes。该计数不包含厂商隐藏复制或返回packing全部流量，也不是完整传输带宽。当前12方向直接P2P及原生跨卡原子仍不可用，ordinary不认证P2P/NCCL/capture或resident协议。

三点首差均为window0、sample0/node0 KV keys（原atol1e-6/rtol1e-5），不能用恢复成功替代独立正确性。完整差异保留在归档。

## 三种明确通信的小型训练

同一FC8节点/4区域KV模型、16,388参数、B2/P3/span18，2卡各4节点，原weighted-output-square×0.001、SGD lr0.001/momentum0.3/weight_decay0.01。三个通信配置的model/workload/task身份一致。每窗39 candidate、39 Full、58消息；独立记录、原loss参数梯度、参数、optimizer槽及精确续接分别验收，8份比较报告各自零差异。

| 通信 | 中位秒 | native逻辑/物理传输bytes | native端点同步 | Python边界复制/同步 |
| --- | ---: | --- | ---: | --- |
| blocking | 0.592957 | 4176 / 4176 | 24 | 10 / 20 |
| cpu_bridge | 0.668933 | 4176 / 8352 | 24 | 10 / 20 |
| ordinary | 0.583081 | 4176 / 4176 | 0 | 10 / 0 |

Native三种请求均为6组跨卡复制；cpu_bridge的6组明确经主机、物理字节翻倍。Python边界360 bytes是设备到主机输出/目标准备，因此其cpu_bridge_copies=0不代表native未中转。计数含声明路径的前后向同步边界；不能外推为所有ATen/driver流量。所有卡共享一个optimizer窗口边界，未将共享参数拆成副本绕开约束。

## NPU同模型与三通信实际点

固定Python实现仍bd2ff2e，NPU公共core/resident构建eca070f；Torch2.12.0+cpu、TorchNPU2.12.0.post2/CANN9.1，HF32关闭。
FP32 payload/scoring、ATen/interop/worker1、mixed-c/packed/scalar完整返回。配置b495f10冻结；不是重跑GPU或旧NPU矩阵。
相同FC16/d2048模型的2/3卡运行已完成，离散及精确恢复通过、独立FP32超差；原atol1e-6/rtol1e-5保持。

| NPU卡数 | 节点分配 | 中位秒 | 逐卡全程allocated峰值GB | 恢复检查秒 |
| ---: | --- | ---: | --- | ---: |
| 2 | 8/8 | 0.762563 | 2.213210 / 2.213211 | 36.492 |
| 3 | 6/6/4 | 0.887916 | 1.672490 / 1.672491 / 1.131772 | 29.783 |
| 8 | 未分配 | 未测 | 未测 | 未测 |

两实际推理点每窗31 candidate/16 Full/28消息，native普通复制7组/229488 bytes。
参数owner、逐节点state及消息目标与有效映射核对一致；不能由普通复制推断resident多卡资格。
8卡于07:13:54Z等待607.440秒后超时（队列exit3），只有7卡可用，物理9为其他项目所用；没有执行模型、不称硬件不支持。
明确七卡新增配置2562263保留与八卡的不同请求；实际完成结果见下节，原八卡超时不改写。

同一小型FC8/d8双卡训练的三种通信均完成三个更新窗，随后分别接受独立检查与恢复：

| NPU通信 | 中位成本秒 | 完整独立训练 / 恢复 | native复制bytes / 同步 |
| --- | ---: | --- | --- |
| ordinary | 0.786530 | 严格通过 / 精确 | 4176 / 0 |
| blocking | 0.907716 | 未执行独立检查 / 恢复中SIGSEGV | 4176 / 24 |
| cpu_bridge | 0.835494 | 严格通过 / 精确 | 8352物理（4176逻辑）/ 24 |

ordinary/cpu_bridge各8份独立比较零差异；两者不能解除其他旧异步图失败。
blocking于07:14:41Z进程exit -11，无存活后代；最后实验快照phase=exact-resume，checkpoint已保存但exact检查未完成。
该成本仅说明已完成窗，不是合格训练吞吐；不能把未执行的CPU参考或恢复写成通过。保留其小checkpoint/log/config以定位。

此实际崩溃还暴露报告遗漏监督器退出码：b3dc53c按同一config_id关联子任务status，保留原result字节/hash、
已完成窗口与未检查字段，明确显示process-failure和最后阶段。正常严格数值失败仍显示strict-failure，外层passed不能提升子项。
开发与冻结源码12项报告/HTML/归档检查通过；用冻结公开results命令重新投影这5条终态，blocking实际显示-11/exact-resume，
归档无需旧目录仍可读取。该修正不改变运行、梯度或活跃安装，不冒充修复了SIGSEGV。

## 来源、生命周期与复现

任务`jobs/platform-cuda-multicard-bd2ff2e`于06:17:00Z exit0，SSH独立检查supervisor/workload/remaining后代全部为空；外层passed不覆盖三项strict-failure。没有项目重型构建重叠，其他系统负载未隔离。allocator范围不含驱动/厂商隐藏内存；详细构造/计时/恢复峰值和RSS高水位在归档。

```sh
tidegraph plan configs/matrices/cuda-multicard.json --matrix --site site.json
tidegraph matrix configs/matrices/cuda-multicard.json --site site.json --output-dir jobs/new-multicard --detach
tidegraph results docs/evidence/core5-platform-multicard-20261031.json.gz --format table
```

安装/实际site见[公共手册](../human-workflow.md)。约1B三卡并不代表靠近多卡总容量；后续容量准入分别核对逐卡峰值、host和checkpoint磁盘，不直接加总空闲显存。

归档SHA256：`716c706b796d677b4ee2f50f7de5e7c861462677c6d49a0dda3543d01e831912`。

独立根补查入口：从同一三通信小配置运行`tidegraph check selected.json --site site.json --scope full --output-dir checks/new`。该检查包含输出/connected-zero/控制/描述量/物理消息以及非位置切点的私有状态根，随后完整三步训练与恢复。它是独立资格而非本表性能复测；尚未运行的根不填写通过。

## CUDA三通信独立根补查

bd2ff2e固定安装分别对ordinary、blocking和cpu_bridge的小型双卡训练配置执行公共`check --scope full`。
三项均通过：完整独立推理+精确恢复、全部/首输出/connected-zero/首控制/四类descriptor/首物理消息，
以及非位置切面私有raw-state独立根，含参数及外部根；最后原任务三步训练/共享梯度/optimizer与精确恢复通过。
不能由原loss梯度推断任意根；此次是独立的实际补查，也不外推大规模或跨卡resident。
07:50:49Z外层exit0，三子进程终态且无剩余后代；主动断开SSH后重新读取状态确认。
逐项输入、实现/hash和结果见[独立检查记录](core5-platform-multicard-checks-20261031.json)。
复现：从`cuda-multicard.json`取相应case作为case.json，再执行
`tidegraph check case.json --site site.json --scope full --output-dir new-check-dir`。


## 同一1.014B模型的七卡实际点

固定包bd2ff2e、配置2562263，其他环境/精度/输入/完整返回与上表同模型相同。
七卡4/2/2/2/2/2/2节点，8个共享组完整放置；队列等待241.254秒后选择物理4/5/6/7/8/10/11，
映射逻辑npu:0–6。模型、workload、task身份与单/双/三卡相同，不复制共享参数成独立owner。

执行、完整离散与磁盘精确下一窗通过；FP32独立数值仍超原阈值，明确探索，不是合格吞吐。
两测量窗0.809713/0.798220秒，中位0.803966秒；每窗candidate31/Full16/消息28，
ordinary原生复制7组/229488 bytes。请求/实际参数、私有状态、消息目标及计算位置与七卡映射相符。
全流程allocated峰值为逻辑卡0约1.131770GB、其余每卡0.591053GB；reserved分别1.210057/0.641729GB；
RSS20.939211GB，构造10.599秒、恢复29.843秒、独立参考13.323秒、整流程140.059秒。
最后checkpoint4,607,424,708 bytes。更多卡降低逐卡存储，不能由此称计算加速或认证多卡resident。

子监督09:54:56Z exit0，外层exit0，控制组和存活后代为空。
原结果SHA256 `6f01b95c199d54f1c371fc35a7fd07c886fb3a2ee5a85746f40d0519d28609fb`。
干净复现选择`configs/matrices/npu-multicard.json`的`npu-model-1b-cards-7`，
按人类手册plan后运行；这证明本接口可实际使用非二次幂卡数，不认证未来7×T4。


## 原blocking崩溃的一次有界诊断

固定相同bd2ff2e安装、原配置19f83bbaf7ef9ba9d949e4e11210891c027851e6badf25853c3b1aa48a866980，
只开启PYTHONFAULTHANDLER；未修改数值实现、输入或精度。本次三个更新、八份独立比较及
磁盘精确恢复全部通过，09:55:32Z内外监督exit0且无后代，因此没有取得新的崩溃栈。
两窗0.917730/0.910114秒只是诊断过程成本，归档标diagnostic，不重复计入正式吞吐。
原07:14:41Z的SIGSEGV/-11记录、checkpoint和未执行检查全部保留；这次未复现不称修复，
也不能证明其发生概率或归因。停止相同重试；重启须新的可重现触发、native栈或流/owner生命周期证据。
本次原结果SHA256 `c1dfdabaa11314bd795ea7e0140bbe22eff541fc9c85c3f6235b24cfd973e0a1`。
