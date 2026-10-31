# 当前交接 — 三平台矩阵、多卡与人类入口（2026-10-31）

授权分支`graph-execution-foundation`；本次接续审计实际HEAD为8550e88，工作区干净，未回退源码。
数学固定`0770593838a37c790a568226930567ed60f99b83`，离线校验通过。当前代码/提交用`git status`及`scripts/status.py`核对。
旧账户指南确实不存在，依用户确认、项目site和当前Ascend模块文档继续。远端脏dev保留，不作资格源码。
最新实现为c59688c观测修正、2562263七卡配置、b3dc53c异常退出报告；截至8550e88已普通推送；本段新增NPU原训练失败和b3dc53c已安装报告证据。

## 已审阅结果

- [首组矩阵](evidence/core5-platform-baseline-20261031.md)：98正式流程+4观察，43严格通过、38数值探索、17严格失败；三平台中等模型B128/B512、A/B/C、三层级/拓扑、三客户端小训练。
- [规模阶段](evidence/core5-platform-scale-20261031.md)：19正式流程，CPU7/GPU5/NPU7，含约1B/B128/B512及FC16/HB20/LH20。18项执行/离散/精确恢复通过、数值探索；NPU1B/B512下一窗KV keys精确恢复失败，独立CPU对照未执行。
- [有限值packing](evidence/core5-finite-packing-20261031.md)：三平台18记录；B128成本CPU1.018→0.853/NPU4.044→3.375/GPU2.317→2.067秒，原数值失败保持。六小训练严格通过，两策略checkpoint/输出精确相同，默认不改。
- [实际profile](evidence/core5-platform-profile-20261031.md)：CPU/NPU/GPU约1B/B128均形成168行FFN矩阵，同时864次两行私有BMM、288次共享base GEMV。c59688c补CANN matmul形状漏计，13项冻结解析检查通过；离线重读不算重新验收。
- [同模型多卡](evidence/core5-platform-multicard-20261031.md)：CUDA1B的2/3/4卡、NPU2/3卡推理执行/离散/恢复通过、数值探索；CUDA三通信小训练严格通过，NPUordinary/cpu_bridge严格通过，blocking恢复阶段SIGSEGV/-11、独立检查未执行。原小checkpoint保留。
- [CUDA resident](evidence/core5-platform-resident-20261031.md)：默认B128/B512布局拒绝；结构界定queue=12B/stages32后实际完成、数值探索。d16同材料eager成本2.154/8.680秒对resident2.324/9.473秒，当前未见优势。NPU待当前链。
- [训练容量](evidence/core5-platform-training-capacity-20261031.md)：CUDA1B单卡静态拒绝，双/四卡第三窗原平方loss非有限；31a3b32仅补拒绝后字段定位，小训练回归通过。NPU1/2卡原流程第三窗非有限已入库，独立参考/恢复未执行；待条件定位，不改loss/input/初始化。
- [CUDA4.561B容量](evidence/core5-platform-capacity-20261031.md)：单/双卡执行/离散/精确恢复通过、数值探索；单卡allocated20.058GB，双卡10.299GB/卡，中位0.675/0.755秒，恢复约315/321秒。三卡也已入库：7.859/7.859/5.419GB、中位0.779秒、恢复316.800秒，数值探索；四卡待下一任务。
- [人类流程](evidence/core5-human-workflow-20261031.md)：GPU安装/doctor/lock/consumer/推理训练/新进程恢复/stop/resume/报告完成；原base related独立根失败保留。1b46dc5报告10项及双机新安装、浏览器实际筛选通过。CPU/NPU完整演练待当前链。
- b3dc53c报告补同config_id子监督状态关联；冻结12项通过，真实NPUblocking记录已显示process-failure、-11、exact-resume。原实验快照、已完成窗和未检查状态保留，未声称修复崩溃。

## 活动任务与固定输入

本机服务均在`background.slice`，固定源码/安装/输入不编辑。外层running/终态不覆盖内层结论。

| unit | 当前工作与依赖 | 产物及上限 |
| --- | --- | --- |
| `gef-platform-next-stages-bd2ff2e.service` | profile、NPU2/3卡、三通信、1/2卡训练已终态；8卡训练607.503秒等待超时、未执行；当前resident默认/有界 | `next-stages-bd2ff2e`、`inputs-b495f10-next`；安装bd2ff2e；外层43200秒 |
| `gef-human-local-de93246.service` | 等next-stages终态，再CPU/NPU完整安装消费者流程 | `human-{cpu,npu}-de93246`；安装de93246；57600秒 |
| `gef-platform-final-1e33b4a.service` | 等human-local，再CPU/NPU同材料eager、31a3b32小训练回归、有条件一次NPU非有限定位、NPU12.802B单/2/3/8卡 | `inputs-1e33b4a-final`、`capacity-human-followup-1e33b4a`；性能bd2ff2e/诊断31a3b32；86400秒 |
| `gef-platform-sharded-3e4f6b2.service` | 等final，NPU38.406B单/双卡现场拒绝；三卡仅在12.802B单卡完成有限推理及精确恢复后准入 | `inputs-3e4f6b2-sharded`、`sharded-capacity-3e4f6b2`；安装bd2ff2e；86400秒 |
| `gef-platform-seven-2562263.service` | 等sharded，仅1B七卡推理；保留原八卡等待失败，不静默改卡数 | `inputs-2562263-seven`、`npu-model-1b-cards-7-bd2ff2e`；配置2562263/安装bd2ff2e；86400秒 |
| `gef-blocking-diagnostic-bd2ff2e.service` | 等seven，原blocking小训练一次faulthandler定位；不计性能曲线，不重复重试 | 原`inputs-b495f10-next/npu-train-two-blocking.json`；`npu-blocking-diagnostic-bd2ff2e`；86400秒 |

本机状态/log在`artifacts/jobs/`下去掉`gef-`和`.service`的同名目录；next/final/sharded内部有progress.json及逐项log。
NPU合作队列priority0、候选物理4–11、等待600秒；当前7卡空闲，第8张物理9为其他项目占用，不干预。
原1B八卡推理等待607.440秒、训练607.503秒均退出3，未执行。每case小型900秒、scale/capacity1800秒；外层上限包含依赖等待。
七卡配置开发/冻结各10项通过，只是配置证据，未执行；新38.406B配置同样10项冻结检查通过。

GPU：`jobs/platform-cuda-capacity4561m-c3-bd2ff2e`于07:14:08Z启动，07:23:53Z终态exit0且无后代，执行/离散/精确恢复通过、数值探索。
固定`inputs-ad93848-capacity/cuda-capacity-4561m-cards-3.json`及`install-platform-bd2ff2e`，1800秒、GPU0/1/2、ATen/interop/worker1。
C1/C2已终态且无后代，证据528025c/94754ac审阅后退役各20.699GB checkpoint；凭据`checkpoint-retirement-capacity4561m-c{1,2}-*.json`，结果/config/log保留。
C3启动前free43.951GB满足两代checkpoint42.482GB+1GiB；C3已审阅并按07cb7f0退役checkpoint，接着核对资源启动`jobs/platform-cuda-capacity4561m-c4-bd2ff2e`，同目录cards-4、GPU0–3、1800秒及相同线程；四卡07:24:52Z已启动，前向窗完成、正在exact-resume，尚无最终验收结果。
CUDA9.123B双卡新配置仅待现场准入：两代checkpoint83.896GB超过整块50GiB盘，不分配、不改协议绕过。

```sh
python scripts/status.py
systemctl --user show gef-platform-next-stages-bd2ff2e.service gef-human-local-de93246.service gef-platform-final-1e33b4a.service gef-platform-sharded-3e4f6b2.service gef-platform-seven-2562263.service gef-blocking-diagnostic-bd2ff2e.service -p Id -p ActiveState -p SubState -p ExecMainStatus -p ControlGroup
artifacts/cuda3090/ssh.sh 'cd /root/autodl-tmp/gef-cuda3090 && install-platform-bd2ff2e/venv/bin/tidegraph job status jobs/platform-cuda-capacity4561m-c4-bd2ff2e'
```

停止只针对准确unit或对应`job stop`，核对无后代；不停止其他项目任务。SSH入口是项目私有配置，不输出凭据或修改全局配置。

## 资源和来源

本机aarch64/320 affinity CPU，约1.8TiB可用主存/2.2TiB空盘，16张64GiB级NPU；Python Torch/TorchNPU2.12/CANN9.1，独立SDK LibTorch2.10/CANN9.0另ABI。
GPU四张3090/sm86、Torch2.12.1+cu130/Toolkit13，56核配额、240GiB内存、50GiB数据盘。
CPU core de999c2/standalone eca070f；NPU core/resident/standalone eca070f；GPU core/resident de999c2、standalone eca070f。
实际库hash与安装来源分别保留，本阶段未改C++，不冒称最新HEAD重建。三平台standalone原3 CTest不重复累计。
bd2ff2e安装内容hash`afc35286eae8dca7744134dcc966cde82ddf5b30e07720cd0ef491ee30783ff8`；活动安装在`artifacts/platform-matrix/install-*`及GPU同名根。

NPU12.802B/FC16/d7296由1B实测校准，预算58.565GB/卡、另留6GiB、host241.331GB、两代checkpoint117.232GB。
38.406B/FC48/d7296三卡每卡16节点/约58.566GB，单/双卡预期拒绝，host766.974GB、disk349.567GB；不是扫描8B/16B或OOM找边界。
CUDA直接P2P与原生跨卡原子限制未解除；普通复制与多卡resident资格分开。

## 下一动作

1. C3已入库和退役checkpoint；继续审阅GPU C4终态，入库后退役checkpoint，再记录9.123B现场磁盘拒绝并执行三通信独立根检查。
2. 逐项收集本机链的真实结果，更新已有多卡/训练/resident/容量/人类证据。保留NPU B512精确恢复、blocking崩溃及原数值失败。
3. 完成三平台六项packing处置、统一查询矩阵/未测与拒绝、下游流程/T4待验说明；全部终态后更新唯一STATUS/ROADMAP并普通推送。

从干净仓库按[development](development.md)/[人类手册](human-workflow.md)安装；[189有限配置](platform-matrix.md)可重建材料，审阅gzip直接查询，不依赖旧运行目录。
T4具体操作说明和六项新证据处置已提交；当前新增三卡证据。阶段完成后继续，不暂停。

验收补齐：两平台各三种小型双卡通信追加公共`check --scope full`，单独验任意输出/控制/物理消息/私有state根；原run已比较loss参数梯度，不冒称所有独立根。GPU待C4及9.123B准入后；NPU后继等待blocking诊断终态，每项300秒、一次、两卡；不重复性能矩阵。

独立根后继`gef-multicard-roots-844eb9b.service`等待blocking诊断终态，原bd2ff2e安装，固定`inputs-844eb9b-roots/npu.json`；三通信每项300秒、统一两卡队列600秒、外层86400秒，状态/log在`artifacts/jobs/multicard-roots-844eb9b`，结果在`artifacts/platform-matrix/npu-multicard-roots-bd2ff2e`。
