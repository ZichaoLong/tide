# 当前交接 — 三平台矩阵、多卡与人类入口（2026-10-31）

授权分支`graph-execution-foundation`，当前已推送`616fe0e`，工作区启动时干净。数学固定`0770593838a37c790a568226930567ed60f99b83`，离线校验通过。Python实现bd2ff2e；活动任务分别使用冻结的1b3e39f/bd2ff2e安装，未修改。远端脏dev保持。旧账户指南已不存在，依用户确认与实际模块/合作队列继续。

## 已完成与审阅证据

- [首组矩阵](evidence/core5-platform-baseline-20261031.md)：98正式流程+4观察，43严格通过、38数值探索、17严格失败。三平台中等模型B128/B512、A/B/C、真实图层级及三客户端均有实际记录；全部执行/离散/精确续接通过。
- [CUDA规模](evidence/core5-platform-scale-20261031.md)：1.010B FP32/native mixed-c B128/B512完成独立检查与精确恢复；数值超差探索接纳，不能称严格合格吞吐。
- [有限值检查packing](evidence/core5-finite-packing-20261031.md)：三平台18条实际记录及完整checkpoint/输出精确同材料比较。B128有限收益、小训练GPU略反增；默认未改。固定bd2ff2e相关78检查通过。显式同步次数未减少。
- GPU finite+FC16/HB20/LH20约1B共9项已终态passed，无后代；三拓扑规模执行/离散/恢复通过，数值探索。原archive在`artifacts/platform-matrix/cuda-topology-scale-bd2ff2e.json.gz`，规模三项已并入规模报告（共5项GPU）。
- GPU `jobs/platform-cuda-multicard-bd2ff2e`六项于06:17Z终态passed，无supervisor/workload/remaining后代。外层完成不代表内层资格；[六项已逐项审阅](evidence/core5-platform-multicard-20261031.md)：三个1B推理数值探索，三通信小训练严格通过；逐卡实际owner和状态/消息位置核对通过。
- GPU人类演练de93246已终态无后代。doctor/lock/consumer/训练推理新进程恢复通过；base related独立根原严格超差保留。stop cancelled/143后无后代，resume到window8；后者未做独立数学检查。已取回human-cuda-reviewed/lifecycle gzip，待正式提交。CPU/NPU人类全流程待做。

## 活动作业与接续

| 位置/任务 | 固定输入、状态与资源 |
| --- | --- |
| CPU `gef-platform-cpu-scale-1b3e39f.service` | background.slice，固定1b3e39f，ATen4/interop1/worker1；7项各1800秒。前5个B1规模点已完成（数值探索）；B128独立检查中，随后B512。结果`artifacts/platform-matrix/cpu-scale-run-1b3e39f`；status/log在`artifacts/jobs/platform-cpu-scale-1b3e39f`。仍运行，未终态不计通过。 |
| NPU `gef-platform-npu-scale-1b3e39f.service` | 有意等待CPU scale终态，随后合作队列1卡、物理候选4–11→逻辑npu:0、priority0。固定1b3e39f/7项各1800秒；外层36000秒含等待。结果`npu-scale-run-1b3e39f`、queue `npu-scale-queue-1b3e39f.json`；status/log在对应artifacts/jobs目录。 |
| GPU下一组 `jobs/platform-cuda-resident-observe-bd2ff2e` | 准备启动：默认resident4项、有界resident2项、1B/B128独立profile1项。冻结bd2ff2e安装；配置来自已提交cb4af7f。每case有限超时，正式计时与profile分开进程。已于06:20Z提交并确认SSH返回后仍运行；无最终结果。 |

```sh
systemctl --user show gef-platform-cpu-scale-1b3e39f.service gef-platform-npu-scale-1b3e39f.service -p ActiveState -p SubState -p ExecMainStatus -p ControlGroup
artifacts/cuda3090/ssh.sh '/root/autodl-tmp/gef-cuda3090/install-platform-bd2ff2e/venv/bin/tidegraph job status /root/autodl-tmp/gef-cuda3090/jobs/platform-cuda-resident-observe-bd2ff2e'
# 本机停止用systemctl --user stop准确unit；GPU用同CLI job stop，随后确认无后代。
```

本机后继准备`gef-platform-next-stages-bd2ff2e.service`，有意等待NPU scale终态：CPU/NPU各一个1B/B128独立profile，NPU同模型2/3/8卡及三通信小训练、1B训练单/2/8卡、resident默认/有界共17个已跟踪case。每项独立公共CLI进程；NPU逐项申请实际所需卡数并释放，priority0/物理候选4–11/等待600秒。配置来自b495f10、安装bd2ff2e；输入`artifacts/platform-matrix/inputs-b495f10-next`，编排`next-stages-bd2ff2e.py/.sh`；外层status/log在`artifacts/jobs/platform-next-stages-bd2ff2e`，内部progress/log在`artifacts/platform-matrix/next-stages-bd2ff2e`。外层总时限43200秒含前序等待；尚未启动测量。

本机人类流程后继`gef-human-local-de93246.service`有意等待next-stages终态，之后CPU全流程、NPU合作队列1卡全流程。固定安装de93246、已准备的public CLI编排，不改活动安装。外层时限57600秒含前序等待，状态/log在`artifacts/jobs/human-local-de93246`；输出`artifacts/platform-matrix/human-{cpu,npu}-de93246`。目前仅等待，未验收。

GPU后继准备`jobs/platform-cuda-training-capacity-first-bd2ff2e`，resident/profile完全终态后运行已提交的1B训练单卡/双卡两个配置；各1800秒，逐卡/主存/磁盘live准入。单卡预计静态拒绝仍实际记录。四卡训练待这两项审阅及磁盘余量核验后接续，避免将checkpoint累积误称硬件限制。

## 环境、来源与资源边界

本机aarch64/320 affinity CPU、约1.8TiB可用主存/2.3TiB空盘；16×64GiB NPU，选择物理4–11前重新经合作队列检查。Torch/TorchNPU2.12/CANN9.1；独立LibTorch2.10/CANN9.0另ABI。CPU core de999c2/standalone eca070f；NPU core/resident/standalone eca070f；GPU core/resident de999c2、standalone eca070f。三平台standalone各3 CTest通过；本阶段不改C++，不重复构建，旧库来源/hash保持。

GPU四张3090/sm86，Torch2.12.1+cu130/Toolkit13；CPU配额56、内存240GiB、数据盘50GiB。GPU新组启动前四卡各空闲24126MiB、0%利用；随后resident/profile正在用GPU0。校准4.561B点需要两代checkpoint42.482GB，必须先审阅/退役终态可再生checkpoint再准入；不因旧产物占盘误称模型硬件不支持。已退役并入库的三拓扑/多卡六个checkpoint27.280GB，凭据`checkpoint-retirement-reviewed-bd2ff2e.json`，清理后free41.735GB（profile仍活动）；此前已退役pilot和1B大batch四项14.576GB，原结果/输入/log保留，凭据`checkpoint-retirement-reviewed-1b3e39f.json`。12方向无直接P2P/原生跨卡原子边界未解除。

安装：本机`artifacts/platform-matrix/install-{1b3e39f,de93246,bd2ff2e}`；远端`install-platform-*`。bd2ff2e包hash`afc35286eae8dca7744134dcc966cde82ddf5b30e07720cd0ef491ee30783ff8`，wheel、site、锁与构建manifest各自保存。活动安装/源码保持不可变。公共CLI/配置入口见[人类手册](human-workflow.md)、[180个有限选点](platform-matrix.md)。

## 下一动作

1. GPU三拓扑规模和多卡6项已审阅；继续resident/1B独立profile，随后1B训练容量与4.561B单/2/3/4卡推理。每个大点之间审阅再退役checkpoint。
2. CPU scale终态后NPU scale自动接续；继续NPU同模型2/3/8卡、三通信训练、训练容量及resident默认/有界；依据自身1B峰值校准64GiB近界点。
3. 在本机计时链结束后完成CPU/NPU安装consumer人类流程。GPU已取回记录单独入库，基座严格失败与下游验收分列。
4. 三平台1B/B128 profile为单独观察，validation=none/checkpoint_check=false，不重复资格、不作吞吐；补真实矩阵形状、桶与活动计数。
5. 综合新增规模/多卡/resident证据处置六项问题，更新唯一ROADMAP、索引、performance-lessons与最终可查询矩阵；阶段提交后普通推送并继续。

从干净仓库按development/human-workflow安装，使用configs/matrices和实测site；审阅gzip可以独立查询，不依赖旧运行目录。历史提交用`resolve_history_commit.py`定位等价源码，不视为重新验证。STATUS唯一交接，ROADMAP唯一待办。

新增六个同材料resident-match配置已开发核验：三平台d16 FC4、B128/B512/P2 eager对应有界resident原模型/输入/task；补齐此前d128不能直接比较的空缺。原174项身份逐项保持。正式运行尚未提交，不计通过。
