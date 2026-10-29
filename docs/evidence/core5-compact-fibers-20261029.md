# 常驻局部纤维压缩：完整资格、实际收益与流资源边界

固定实现`04b52153f9785e1489e31c7dd09a16849791d331`（含8c48f49）；
上游`0770593838a37c790a568226930567ed60f99b83`不变。
[审阅JSON](core5-compact-fibers-20261029.json)保存两套新构建、资格、计时、实际状态核对、
取消记录和厂商失败摘录。此前基线为[15c3e65报告](core5-boundary-performance-20261029.md)。

## 改动与数学依据

旧常驻实现为每个节点预留`[samples,global_queue,payload_width]`的局部纤维，
Aggregate还按该长度展开来源矩阵。新实现使用
`min(queue,max(1,物理入边数+输入端口数))`作为节点槽数。
[PDG §1.3–1.4](../upstream/tide-core-5/positive-delay-graph-finite-cut-learning-note.md)的
严格递增端口时间、固定边延迟、每次Full对每条边最多一条消息给出这个上界；
TimedDAG及Settle嵌入继承它。平行边分别计数，SrcClass不参与合并；SourceDomain冲突仍拒绝。

`weighted_node_fiber_capacity`只消费静态拓扑，`append_weighted_node_input`仍在设备上
按实际完整纤维求长度、时间和顺序。indices及零行sentinel仍指向全局ready数组；
超出局部槽数在写入前拒绝，不截断。前向、窗口VJP重放和peer请求采用同一布局；
逆向路由继续按真实全局原子身份找祖先，None/连接零、int64时间及原始状态保持。
没有改变队列容量、在线就绪/选择/推进、参数、数学配置、dtype或容差。
对应入口见[完整图契约](../resident-weighted-flow.md)和[Aggregate](../resident-weighted-aggregate.md)。

固定D64计划480节点/2208边/32输入端口：32个节点入度1、32个入度4、416个入度5。
每样本局部槽总数从1,966,080变为2,240。这是静态布局计数；吞吐及allocator峰值另测。
逐节点局部程序、物理缺席行和全局journal仍保留，尚未完成跨节点ragged数值批量化。

## 干净固定源码资格

两套resident均从干净04b5215、新目录完整构建。未复用旧CANN合并对象。
core未改变，按完整C++源码、binary和依赖hash核验复用；其原bdd7f6c dirty-origin仍保留。
Python-owned为Torch/Torch-NPU2.12/CANN9.1；standalone为LibTorch-NPU2.10/CANN9.0。
均为aarch64 Ascend910_9392。新源码CUDA构建/真机和x86_64均未验，GPU依用户延后。

| 检查 | 实际通过范围 |
| --- | --- |
| CPU元数据 | 两套环境各10 passed/0 skipped；全局索引、局部槽、同类原子、SourceDomain、origin顺序、巨大int64时间、缺席、越界写入前拒绝、已有错误保持 |
| 公共Python/NPU | 56 passed/0 skipped；完整图20项、异构配置4项、一般Settle端口2项、失败/版本/loader边界、五记忆单卡训练16项、双卡训练8项及真实benchmark梯度/SGD/恢复消费者 |
| standalone局部组合 | weighted-stage FP32，24组合；独立字段/原始状态根、全部canonical参数、None/零和非有限根拒绝 |
| standalone完整图 | weighted-flow FP32/FP16各21案例；五族、异构、HB、非对齐/反馈、快照、streaming/prefill、逐边/端口/pending与磁盘续接、容量拒绝 |
| standalone单/双卡训练 | weighted-training与weighted-node-training两dtype，各5组合/20更新/3窗（含空窗），SGD/AdamW、冻结、重复端口、迁移恢复和共同拒绝 |

SDK合计7个组件/dtype检查，全部passed。恢复逐值比较全部实际逻辑字段、原始raw、参数、
梯度连接和optimizer槽；未使用容量padding没有指定值，日志单列其差异。
解码不使用参考递归补写结果。旧CPU eager 517 passed/2原FP16 strict xfail没有在本提交重跑。

8c48f49新增counts界限检查经审查发现会覆盖已有错误，随后加守卫和独立poison反例。
最初两套构建主动取消，outer status为cancelled/143，内层记录未完成；它们未进行设备资格，
没有改记为通过。04b5215的新两套服务均passed/0。

资格采用background.slice、两个build线程/任务、单ATen线程、32GiB配置预算；
构建上限1800秒，设备排队180秒、执行1800秒，服务4000秒。
新构建实际约715.5/840.5秒，设备阶段含排队约316.7/659.2秒。
两套资格使用物理1/9和3/13；性能测量等全部构建/资格终态后单独运行。

## 同材料中型完整训练

两项新性能测量使用Python-owned Torch2.12/CANN9.1公共owner；SDK未做本规模性能测量。

HB/KV FP32，102,604可训练参数；同输入hash
`e6e7b7d9f3f12040bb34638de8cd4ffc0d152c5fb452d41f16dae3bd0e836f42`，
请求/生效容量相同，候选/Full/消息/输出/pending均128/128/192/32/0。
每配置一次独立进程、一个warmup窗和一个完整SGD测量窗；物理9映射逻辑0。
完整记录、最终参数/梯度/SGD槽和计时外磁盘换owner下一窗精确恢复均通过。

| 观测 | 15c3e65 | 04b5215 |
| --- | ---: | ---: |
| 完整训练秒/窗 | 2.793867 | 2.469730 |
| forward owner张量预算计数/字节 | 141,287,567 | 87,457,999 |
| adjoint张量预算计数/字节 | 275,214,029 | 194,327,925 |
| 实测peak allocated/字节 | 514,803,200 | 376,404,992 |
| 实测peak reserved/字节 | 532,676,608 | 419,430,400 |
| 反向构造/us | 927,621 | 1,057,073 |
| 反向执行及等待/us | 1,094,390 | 700,072 |

本例实测allocated峰值减少约26.9%，完整窗单次减少约11.6%；不宣称稳定或普遍加速。
构造成本没有观察到下降，前向owner构造0.884→0.960秒。
retained与辅助预算分别仍230,885,776和311,144,416字节；物理local/Full行仍192/192、6阶段。
这项改进消除了局部填充浪费，未减少逻辑工作、物理节点行或跨窗口程序构造。
计时包含准备、上传、调度、packing、前后向、finite检查、optimizer、返回和同步；
构造、warmup、独立核对及恢复在计时外。外部争用未隔离，预算计数不是allocator实测。

## 原D64的新停止边界

原计划hash `6c8cc36ae97287133d40e67ccc8e137574b80e2df420feff6adfef3395397bb3`，
9,076,975可训练参数，对core-3 Add主体9,076,928多47。相同seed7 FP32材料，
queue4096/stages32/cache4/optimizer_bytes1GiB，总预算5GiB、其余子预算不变。
没有CPU真实轨迹预跑。此次未进入warmup，进程97.5秒后exit1；整个study因该项failed/1。

新的首个错误为`create persistent control stream failed with CANN status 207005`。
本机CANN头将207005定义为`ACL_ERROR_RT_RESOURCE_ALLOC_FAIL`，实际厂商日志明确报告
`Too many streams are created`（EE1023），SQ/CQ观测数量均1984，失败stream_id1797。
JSON保留原文摘录及hash；这些数值是本次观测，不能当所有硬件/进程的固定许可上限。

源码中`TaskServices::begin`每次数值服务创建一个`DeviceProgram`，后者各创建persistent和
launch流；服务数量随节点与数值阶段增长。当前拒绝发生在构造，区域分配求解尚未执行。
局部槽压缩没有减少程序/流的数量；张量与workspace预算也不控制此类厂商执行资源。
本例没有完整前向、反向、optimizer、恢复或合格吞吐结果，也未证明余下构造一定满足张量预算。

按预先声明的运行时失败停止条件结束，不重跑相同候选或放大HBM预算。
重启需要具体的有界服务分派/复用或跨节点数值合批实现，并独立验证嵌套调用、在线推进、
通知与共享workspace生命周期、完整梯度、peer顺序和精确恢复；不能以主机预选序列替代常驻。
完整adjoint仍绑定journal/根/参数地址，不能按shape复用旧对象。

## 可复现入口

按[开发](../development.md)与[设备构建](../device-control.md)选匹配环境、冻结源码、新输出目录，
设置CORE、RESIDENT及显式设备。公共NPU进程在首次autograd前初始化设备。

```sh
python -m pytest -q tests/test_resident_weighted_flow_packing.py
python scripts/verify_device_control.py --build-dir RESIDENT --device npu:0 \
  --checks weighted-stage weighted-flow weighted-training weighted-node-training --output-dir NEW_GATE
python scripts/benchmark_weighted.py --device npu:0 --implementation resident --native-library CORE \
  --resident-library RESIDENT --size medium --steps 1 --checkpoint-check --output-dir NEW_MEDIUM
```

Python-owned测试设`TIDE_BUILD_DIR`、`TIDE_RESIDENT_LIBRARY`、`TIDE_RESIDENT_DEVICE=npu:0`，
在独立进程运行`test_resident_weighted_flow.py`、`test_resident_weighted_training.py`、
`test_resident_weighted_node_training.py`与`test_weighted_benchmark_resident.py`；双卡检查需两张逻辑卡。
D64静态计划生成与原命令见[前一报告](core5-boundary-performance-20261029.md)，容量JSON沿上表。
新候选D64进程预算900秒在运行前固定；medium300秒、queue180秒、服务1560秒、32GiB、单线程。
本次没有超时，也没有根据失败延期。原生NPU D64数值失败及其自身恢复记录保持原结论。
