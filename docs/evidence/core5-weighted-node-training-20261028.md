# Core-5 常驻节点放置窗口反向与公共训练资格 — 2026-10-28

固定运行库 `0155b0f679caab089cbabccf7986e479fe5493a5`，纠正检查器
`220052abb592bc9a9aab6f27e4e042a55415edeb`，固定上游
`0770593838a37c790a568226930567ed60f99b83`。
[同名JSON](core5-weighted-node-training-20261028.json)保留构建、二进制、loader、命令、
输入源码、日志/XML摘要、设备映射及每次任务的独立终态。
数学、放置与训练边界分别从[语义对应](../semantic-map.md)、
[节点放置](../resident-weighted-placement.md)和[公共训练](../resident-weighted-training.md)进入。

| 范围 | 实际结果 |
| --- | --- |
| CPU静态/入口，0155b0f | 10项通过；早期f84a007的8项与37922eb的10项另列，不合并成新增覆盖数 |
| standalone双NPU窗口VJP，FP32 | 四组合各12种独立根；两个三窗桥（含空窗）通过 |
| standalone双NPU公共训练，FP32/FP16各自 | 五配置、20次更新；每次三窗，累计、拒绝、跨放置恢复通过 |
| 两个样本各自跨节点放置，FP32/FP16各自 | 两配置、6次共同更新；完整私有根、canonical梯度和、共同optimizer及拒绝/恢复通过 |
| 单设备窗口回归 | HB/proposal、反馈PDG/old通过 |
| Python-owned双NPU | 37922eb原训练/样本副本/新放置共30项通过；0155b0f再验新增8项放置训练通过 |
| 安装后的公共C++消费者 | 单设备和全节点放远端两模式，手算恒等VJP、None optimizer及换放置恢复通过 |
| CUDA | sm80/90/100运行库、窗口/训练/副本检查器、安装消费者及修正检查器构建/loader/help通过；没有GPU运行证据 |

运行库NPU使用独立LibTorch2.10/CANN9.0，Python-owned使用Torch2.12/CANN9.1，
CUDA使用Torch2.10/cu128、Toolkit12.8.1。不同工具链的结果不合并为同一二进制证据。
源、构建和真实设备范围分列；本报告不新增性能或全部配置的笛卡尔资格。

`weighted_window_peer_check.cpp::run`保留同秩HB、全部节点在远端的非对齐TimedDAG、
反馈PDG和完整区域Attention/来源变换；混合五记忆、Read初态/旧态/候选态、重复边、
零与缺失消息、共享/冻结参数、前向关闭后反向，以及真实raw/Full/参数bank物理设备。
12根模式分别覆盖none、all、content、control、Full、Describe、old、proposal、final、
messages、fibers和pending。独立CPU Greedy/autograd核对全部输入/左切面/共享参数贡献；
候选自己产生设备工作序列，参考没有向候选提供实际轨迹。

`weighted_training_node_check.cpp`与`weighted_training_trajectory.cpp::trajectory`
核对FC/KV/SGD、HB异构/AdamW、非对齐delta/SGD、反馈异构/AdamW、全冻结FC/affine。
四次更新分别使用output、连接零、全部字段和output根；每次含一个空窗。
独立NamedOptimizer核对FP32 master、实际存储参数、generation及所有optimizer槽。
中途交换节点放置，核对完整窗口、参数伴随和精确checkpoint；Python另有新进程磁盘续接。
实际重复端口分别返回，原始int64时钟/时间戳及pending不压缩为有效状态投影。

修正检查器用独立FP64更新算术先确认目标存储溢出，再断言数值拒绝码20、master/槽/
参数/generation完全不变，并从拒绝后的切面继续一窗，核对每个owner仍使用旧参数bank。
`weighted_training_peer_check.cpp`再组合节点放置和样本副本；按全部使用者贡献求和，
不按卡数平均。通信预算与同代槽不一致拒绝单独核对。
安装消费者的简单手算根是公共接口锚点，不能代替上述完整图、共享参数与发布检查。

梯度/master沿用rtol=3e-5、atol=3e-6，named参数1e-5/1e-6；
FP16完整记录采用.02/.001并对照独立FP32算术、显式FP16存储参考。
离散身份、存在/连接位、int64及None/零区别严格比较；恢复和拒绝的不变部分要求精确一致。
局部手算、独立公式/方向导数等数学锚点沿语义索引进入；调度对照不单独证明全部局部公式。

原失败没有改写：

- f84a007窗口资格在全远端nonaligned构造反向时触发CANN361001。厂商日志为
  stream task buffer full；原binary的GDB诊断重现于DeviceProgram::finish→WeightedFlowPeers::finish。
  诊断job exit0只表示抓栈成功，inferior exit2，不是设备通过。0155b0f把每节点VJP及其raw/
  参数归约放入独立嵌套服务，仍由设备按原序推进并共享workspace；原反例未改动并已通过。
- 37922eb及0155b0f的原SDK整组均在五个FP32训练轨迹之后被检查器误拒：数值溢出应为
  optimizer_finite_error=20，却硬编码为int64步数溢出的21。原任务仍failed；220052a只改检查器
  预期及诊断信息，生产optimizer未变。后续补验独立覆盖两dtype及原来尚未执行的副本/窗口/消费者。
- 首次修正检查器CUDA任务因辅助脚本漏装weighted-training组件而在CMake配置阶段失败；
  同版SDK任务尚在等待依赖时主动取消。新的v2任务补齐组件后独立通过，没有覆盖旧终态。

修正检查器链接已核验的0155b0f安装库；对tools/scripts/cpp/python比较只有检查器不同。
逐一核对原构建manifest、binary和loader摘要，在私有库副本上重现CMake RPATH_REMOVE，
结果与安装库逐字节相同，并核对ldd实际装载路径。检查器源码和运行库源码分别记录。

实际运行使用只读冻结worktrees、独立background.slice/cgroup、Nice10、两build workers、
ATen/BLAS1。原构建1500秒，修正检查器构建240秒、依赖等待至多1500秒；两卡queue180秒。
修正后的设备补验1800秒，整job3900秒/unit3960秒；CUDA补验job1900秒/unit1960秒。
任务名、物理→逻辑0/1映射和时间均在JSON；所有任务已终态，原失败/取消保持原值。
已回收unit的默认属性不作为退出证据，以持久退出码和实际输出共同判断。

从干净220052a源码、匹配core和全新输出目录复现，无需旧运行目录：

```sh
python scripts/build_device_control.py --core-build CORE --build-dir NEW_BUILD \
  --runtime standalone --ascendc-soc SOC --jobs 2 \
  --checks weighted-node-window weighted-window weighted-node-training weighted-node-replicas
python scripts/verify_device_control.py --build-dir NEW_BUILD --device npu:0 \
  --output-dir NEW_CHECKS --checks weighted-node-window weighted-node-training weighted-node-replicas
NEW_BUILD/tide-weighted-window-check --device=npu:0 --dtype=float32 --case=hb/proposal
NEW_BUILD/tide-weighted-window-check --device=npu:0 --dtype=float32 --case=feedback/old
python scripts/build_resident_consumer.py --core-build CORE --resident-build NEW_BUILD \
  --output-dir NEW_CLIENT --component weighted-training
NEW_CLIENT/consumer/weighted-training-consumer --device=npu:0 --dtype=float32
NEW_CLIENT/consumer/weighted-training-consumer --device=npu:0 --dtype=float32 --node-placed
```

进程外映射两设备到逻辑0/1。Python使用独立匹配core与`--runtime python --python-component weighted`
构建，运行`test_resident_weighted_node_training.py`；原回归另含`test_resident_weighted_training.py`
和`test_resident_weighted_peers.py`。环境/持久服务见[开发规范](../development.md)。
CUDA换匹配core及`--cuda-architectures '80;90;100'`；真机前向/反向/optimizer/跨卡/容量/性能仍待验。
既有eager GDN FP16、默认异步双NPU、SDK affine及FP32 PDG严格见证继续保留，未由本报告解除。
