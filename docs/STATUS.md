# 当前交接 — 已获用户授权继续（2026-10-31）

用户“同意，请你继续”解除87655fc提交后的暂停。继续收齐既有任务、独立根和恢复诊断；
不扩盘、不重启GPU。先用4.561B做有界checkpoint成本拆分，tmpfs须与主存联合准入。

授权分支`graph-execution-foundation`；本次接续审计HEAD87655fc、工作区干净。
数学固定`0770593838a37c790a568226930567ed60f99b83`，离线校验通过。
实际当前提交用`git status --short --branch`和`python scripts/status.py`核对；不回退后续源码。
旧账户指南已不存在，依用户确认、项目site和当前Ascend模块文档继续；远端脏dev保留。

## 已审阅与本次提交

- [首组矩阵](evidence/core5-platform-baseline-20261031.md)：98正式+4观察，三平台中等B1/8/32/128/512、A/B/C、三图层级/拓扑、三客户端小训练；原严格失败保留。
- [规模](evidence/core5-platform-scale-20261031.md)：19正式，CPU7/GPU5/NPU7，约1B大batch及FC16/HB20/LH20。18项完整离散/恢复通过、数值探索；NPU1B/B512下一窗KV keys精确恢复失败，独立CPU未检查，原checkpoint保留。
- [有限诊断packing](evidence/core5-finite-packing-20261031.md)：三平台18记录、六小训练严格通过；B128成本降低但数值失败保持，默认scalar不改。相关独立根回归与各运行loss梯度范围分列。
- [profile](evidence/core5-platform-profile-20261031.md)：三平台1B/B128形成168行FFN，也有864次两行私有BMM、288次共享base GEMV。c59688c修正CANN matmul解析，13项冻结检查通过，不算设备重验。
- [多卡](evidence/core5-platform-multicard-20261031.md)：CUDA1B的2/3/4卡、NPU2/3卡推理完整离散/恢复通过、数值探索。CUDA三通信小训练、NPUordinary/cpu_bridge原loss梯度/optimizer/恢复严格通过；NPUblocking在exact-resume SIGSEGV/-11，独立检查未执行。任意独立根补查另列为后继作业。
- [resident](evidence/core5-platform-resident-20261031.md)：CUDA有界B128/B512完成、数值探索。同材料eager更便宜。新增NPU六项：B1和小训练严格通过，有界B128中位1.653417秒、离散/精确恢复通过、数值超差；有界B512通过HBM准入后构造EE1019/507002流任务缓冲满，未进入计时。原默认预算拒绝保留，短厂商日志/hash已审阅。
- [训练容量](evidence/core5-platform-training-capacity-20261031.md)：CUDA单卡1B静态拒绝、双/四卡第三窗原平方loss非有限；NPU单/双卡原第三窗非有限已入库，参考/恢复未执行。31a3b32只补失败字段定位，不改配方或接受条件。NPU31a3b32小训练严格通过；双卡原训练准确拒绝字段runtime.forward_and_loss[loss]，恢复和独立参考未执行。
- [4.561B容量](evidence/core5-platform-capacity-20261031.md)：CUDA同模型1/2/3/4卡执行/离散/精确恢复通过、FP32数值探索；中位0.675/0.755/0.779/0.784秒，逐卡最大allocated20.058/10.299/7.859/5.419GB。不是多卡总容量极限。四卡07:34:04Z终态exit0、SSH检查无后代；20.699GB checkpoint仍保留，未启动下一GPU任务。
- [人类流程](evidence/core5-human-workflow-20261031.md)：CPU/NPU/GPU各自公开安装锁/配置/检查/训练推理/新进程恢复/日志/停止/恢复/报告完成；本机各26命令。CPU base通过，NPU/GPU base独立根严格失败与下游小任务通过分列。四个本机停止/恢复unit均控制组空。按需返回生命周期演示保持validation=none，不冒称数学资格。
- b3dc53c新私有安装与冻结12项报告检查通过；原blocking崩溃显示process-failure/-11/exact-resume，原快照不改。新报告只查询，不重新认证设备。两轮报告来源/hash在[报告资格](evidence/core5-human-report-20261031.json)。

## 既有本机任务与新增GPU工作

观测07:49Z：next-stages和human-local已终态且控制组空；final已完成CPU/NPU同材料B128/B512、31a3b32小训练与原loss定位，当前NPU12.802B单卡构造。其后依赖链保持。
所有服务在`background.slice`，固定源码/安装/输入不编辑；外层running/终态不覆盖内层资格。

| unit | 已提交工作与依赖 | 输入/产物 |
| --- | --- | --- |
| `gef-platform-final-1e33b4a.service` | CPU/NPU同材料eager、31a3b32小训练回归、有条件一次NPU非有限定位、NPU12.802B单/2/3/8卡 | `inputs-1e33b4a-final`、`capacity-human-followup-1e33b4a`；性能安装bd2ff2e/诊断31a3b32 |
| `gef-platform-sharded-3e4f6b2.service` | 等final；NPU38.406B单/双卡现场准入；三卡仅在12.802B单卡有限执行及精确恢复通过后准入 | `inputs-3e4f6b2-sharded`、`sharded-capacity-3e4f6b2`；bd2ff2e |
| `gef-platform-seven-2562263.service` | 等sharded；同一1B/16节点/8共享组明确七卡推理，4/2/2/2/2/2/2节点 | `inputs-2562263-seven`、`npu-model-1b-cards-7-bd2ff2e` |
| `gef-blocking-diagnostic-bd2ff2e.service` | 等seven；原blocking小训练仅一次faulthandler诊断，不计新性能、不重试抹旧失败 | 原`inputs-b495f10-next/npu-train-two-blocking.json`、`npu-blocking-diagnostic-bd2ff2e` |
| `gef-multicard-roots-844eb9b.service` | 等blocking诊断；两NPU三通信各一次公共`check --scope full`，独立根及三步训练/恢复 | `inputs-844eb9b-roots/npu.json`、`npu-multicard-roots-bd2ff2e`；bd2ff2e |

外层上限86400秒包括依赖等待；各case小型900秒、容量1800秒、独立根300秒。
NPU合作队列priority0、候选物理4–11、每次等待600秒。原八卡推理607.440秒、训练607.503秒等待超时退出3，未执行；第八张物理9为其他项目，不干预。
作业状态/log在`artifacts/jobs/`下去掉`gef-`和`.service`的同名目录；final/sharded及根任务各有progress.json。

```sh
python scripts/status.py
systemctl --user show gef-platform-final-1e33b4a.service gef-platform-sharded-3e4f6b2.service gef-platform-seven-2562263.service gef-blocking-diagnostic-bd2ff2e.service gef-multicard-roots-844eb9b.service -p Id -p ActiveState -p SubState -p ExecMainStatus -p ControlGroup
artifacts/cuda3090/ssh.sh 'cd /root/autodl-tmp/gef-cuda3090 && install-platform-bd2ff2e/venv/bin/tidegraph job status jobs/platform-cuda-capacity4561m-c4-bd2ff2e'
```

如用户要求停止后台作业，只停止准确unit并核对控制组/后代；当前没有收到该停止指令。
GPU9.123B现场磁盘准入已拒绝、无后代；三通信full独立根已持久启动。
作业名`platform-cuda-admission9123m-87655fc`、`platform-cuda-roots-87655fc`；未见终态前不称通过。
83.896GB磁盘预算已含1GiB reserve；现场拒绝独立记录，不冒称构造或设备执行。

## 固定来源与资源

本机aarch64/320 affinity CPU、约1.8TiB可用主存/2.2TiB空盘，16张64GiB级NPU；Python Torch/TorchNPU2.12/CANN9.1，独立SDK LibTorch2.10/CANN9.0另ABI。
GPU四张3090/sm86、Torch2.12.1+cu130/Toolkit13，56核配额、240GiB主存、50GiB数据盘。
CPU core de999c2/standalone eca070f；NPU core/resident/standalone eca070f；GPU core/resident de999c2、standalone eca070f。
旧二进制保留实际hash与来源，不冒称在最新Python提交重建；三平台旧3 CTest不重复累计。
bd2ff2e安装内容hash`afc35286eae8dca7744134dcc966cde82ddf5b30e07720cd0ef491ee30783ff8`；活动安装均冻结。

NPU12.802B预算58.565GB/卡另留6GiB、host241.331GB、两代checkpoint117.232GB。
38.406B三卡预算每卡58.566GB另留6GiB、host766.974GB、disk349.567GB；其实际运行取决于校准，不由预算认证。
CUDA C1/C2/C3 checkpoint已在各自证据审阅后退役，凭据位于远端根；C4保留，结果/config/log始终保留。
CUDA直接P2P/原生跨卡原子限制未解除，普通复制不认证多卡resident。

## 当前接续

1. 先重核上述既有任务的实际终态，审阅内层结果，再补多卡/训练/resident/容量证据；不重跑已完成矩阵。
2. 保留GPU C4 checkpoint用于成本拆分；先执行9.123B现场准入和三通信独立根full检查。远端独立根脚本hash `2f06b44ef74d25e62f0523fbd8a916825f29306fde90e10a00ac7aa257e1142c`、输入hash `89f191eb1e0d892e803f86da5111e90d9cce8a36ddf19edea8ae23bbe24a6dd8`，安装bd2ff2e；外层1200秒/每case300秒。
3. 补tmpfs与主存联合准入、分阶段checkpoint观测的最小实现；相关测试、提交、固定安装后做一次磁盘/tmpfs有界对照。当前盘空闲23.248GB、shm128.849GB空，容器内存上限257.698GB；不把shm当额外物理内存。
4. 收齐有限矩阵与拒绝/未测项，更新最终选型、六项处置和唯一ROADMAP；全部终态与无后代后交付。当前总体任务尚未完成。

从干净仓库按[development](development.md)/[人类手册](human-workflow.md)安装；[189有限配置](platform-matrix.md)可重建材料。
审阅gzip用`tidegraph results docs/evidence/core5-platform-*-20261031.json.gz docs/evidence/core5-finite-packing-20261031.json.gz --format table`查询，不依赖旧运行目录。
下游模板/人类流程及T4迁移清单已交付；T4真机资格未获得。
