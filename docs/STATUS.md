# 当前交接 — 三平台计算收尾（2026-10-31）

用户已授权后台继续，最新要求一次安排完后不必在线等待，之后由用户唤醒审阅；GPU可释放。
本机优先计算与必要正确性/恢复，额外大文件成本实验已取消。
不扩盘、不重启GPU、不重跑已审阅矩阵；tmpfs仅在联合主存准入后周转。活动输入和源码不编辑。
授权分支`graph-execution-foundation`，本次重入59c36a2干净且已同步远端；实际HEAD以`git status --short --branch`为准。
实现4485697及既有证据/交接均已普通推送；本次终态审阅和配额退役已提交8e180b0，待本阶段普通推送。
数学固定`0770593838a37c790a568226930567ed60f99b83`，离线校验通过。参考仓库只读。
旧账户指南已不存在，按用户确认、项目site及`/opt/software/ascend-locks/dev-workspace-9.1.0/README.md`操作。

## GPU已完成，可退掉

释放审阅09:53Z确认四卡无计算进程、项目监督全部终态且无存活后代；结果、配置、构建身份和必要小记录已回收。
后续本机任务不依赖SSH或GPU目录，未启动新的GPU工作。见[释放审阅](evidence/core5-cuda-release-20261031.json)。
原远端dev未修改，16项跟踪修改+4项未跟踪源码已备份并逐文件校验：
`artifacts/cuda3090/retirement-dev-20261031.tar.gz`，5,452,397 bytes，
SHA256 `6b89b411611378a619149d69870c35e3f587db0baf1ed70fd2a21a005e2fe31e`。
这是未提交修改的保全备份，不是可随意删除的实验产物。可重建大checkpoint已审阅退役，不跨机搬运。

## 已审阅范围

八份platform/finite-packing压缩归档现含201条唯一run（包括观察、拒绝、严格失败），不是197项性能通过。
完整配置、实际来源/hash、工作量、逐窗及峰值均在归档；旧C++库没有冒称随较新Python提交重建。

| 范围 | 事实与证据 |
| --- | --- |
| 三平台/大batch | [首组](evidence/core5-platform-baseline-20261031.md)：98正式+4观察；三平台B1/8/32/128/512及B128 A/B/C、三层级/拓扑/客户端、小训练；43正式严格通过 |
| 约1B/更多节点 | [规模](evidence/core5-platform-scale-20261031.md)：CPU7/GPU5/NPU7正式点，FC/HB/LH；18完整离散/精确恢复通过、数值探索；NPU1B/B512原下一窗逐位失败；同进程新checkpoint原边界精确、下一窗仍失败，独立CPU未检查 |
| 六项packing/优化 | [finite](evidence/core5-finite-packing-20261031.md)：18记录、6小训练严格通过；[profile](evidence/core5-platform-profile-20261031.md)及[六项决定](performance-lessons.md#本轮三平台六项打包处置)保留收益、无收益与停止条件 |
| 同模型多卡 | [多卡](evidence/core5-platform-multicard-20261031.md)：CUDA1B 2/3/4卡、NPU2/3/7卡推理离散/恢复通过、数值探索；7卡是真实4/2/2/2/2/2/2节点。8卡原队列超时未运行；用户现授权低占用卡共用，后继准备中 |
| resident | [resident](evidence/core5-platform-resident-20261031.md)：两设备小点/小训练严格通过；CUDA B128/B512、NPU B128完成/恢复通过、数值探索；NPU B512构造EE1019/507002。多卡resident门禁不变 |
| 原1B训练 | [训练](evidence/core5-platform-training-capacity-20261031.md)：原loss单/多卡第三窗首先在loss非有限；新loss 1e-9为新工作负载，CUDA双卡及NPU单/双卡执行/离散/精确恢复通过、独立数值超差 |
| 容量 | [容量](evidence/core5-platform-capacity-20261031.md)：CUDA4.561B单/2/3/4卡、9.123B单拒绝/双卡实际，NPU12.802B单/双/三卡均完成/离散/恢复通过、数值探索。38.406B三卡有限计算窗5.442s，第二次保存1800s超时，完整恢复/独立参考未验；原I/O进程已退出、控制组空 |
| checkpoint | [成本](evidence/core5-checkpoint-cost-20261031.md)：36baeb7实际减少重复快照/CPU精确正例比较，CUDA同材料恢复289.587→223.215s；interval2少保存是耐久取舍，不称算子加速。固定CPU110项及CUDA返回6项/双卡full通过 |
| 人类/下游 | [演练](evidence/core5-human-workflow-20261031.md)：CPU/NPU/GPU安装、配置、检查、运行、停止/恢复及CSV/HTML完成；base失败与下游小任务通过分列。[手册](human-workflow.md)/[下游](downstream-contract.md)/T4迁移配置已交付；T4真机待验 |

6095c6a补齐训练host快照预算，1.014B为52.585GB，固定45项通过；6c6eaa2保存阶段可见性固定30项通过；
122a1ed按进程启动身份区分终态监督与I/O后代，固定20项通过。后台实际终态仍须核对退出码/后代，不能从外层推断内部通过。

## 本次独立根与恢复诊断

- [独立检查](evidence/core5-platform-multicard-checks-20261031.json)：CUDA三通信full已实际通过；NPU原包bd2ff2e三通信完整推理/恢复先通过，随后独立根均EE1017。旧blocking恢复SIGSEGV一次诊断未复现，不称修复。
- 4485697只增加check的根/参考VJP/候选VJP/比较阶段记录，CPU固定27项通过。原未执行的返回测试补齐私有pytest后在NPU实际6项通过（4项明确deselected）。
- 4485697 ordinary双NPU在第一个all根的候选autograd发生SIGSEGV/-11；faulthandler栈已保留。单NPUperiodic在all根比较有输入梯度超差，最大2.83122e-6，原阈值不改。原36baeb7环境缺pytest/单卡数值失败不抹除。
- 显式ASCEND_LAUNCH_BLOCKING=1一次ordinary双卡诊断越过候选VJP，在all根参数梯度比较仍失败（1/64坐标、1.31130e-6）。不是默认异步修复、不是合格吞吐；不继续重复相同事件方案。
- [B512诊断](evidence/core5-platform-resume-diagnostics-20261031.json)：原包1b3e39f重算三窗与旧checkpoint在raw边界即不同，首三个样例state及所有参数布局相同。同进程后继已结束：新checkpoint raw边界精确，下一完整窗keys有3766/32768坐标逐位不同，最大2.09808e-5。采样布局仍相同；不归因磁盘损坏，不再重复相同候选。

## 当前终态与资源

上一条后台链全部结束；三个unit均MainPID0/ControlGroup空：

| unit（加`.service`） | 终态事实 | 结果 |
| --- | --- | --- |
| `gef-npu-compute-followup-4485697` | exit1来自显式同步all根诊断；四个loss训练子任务均exit0，无后代 | `artifacts/platform-matrix/npu-compute-followup-4485697/progress.json` |
| `gef-npu-resume-immediate-61571c4` | exit1，raw解码精确、下一完整窗严格逐位失败，无后代 | `artifacts/platform-matrix/npu-resume-immediate-61571c4/result.json` |
| `gef-closure-4485697` | exit0收集201条，review_required；收集不等于资格 | `artifacts/platform-matrix/closure-4485697/status.json` |

四项训练：小型单/双NPU严格通过；新loss1e-9的1.014B单/双NPU执行、离散、完整精确续接通过，独立数值失败，
中位1.614337/1.867778s，只作探索成本。任意根运行时/数值失败不解除。包4485697、NPU库eca070f分别记录。
原1B/B512失败checkpoint仍在`artifacts/platform-matrix/npu-scale-run-1b3e39f/npu-scale-fc-few-1b-b512/run/checkpoints`；
同进程诊断临时checkpoint在`/dev/shm/gef-npu-resume-immediate-61571c4`，结果/log已经持久保存。

用户新提醒本机硬配额800GiB。实际`quota -vs`为400GiB软限、800GiB硬限；原占用636GiB/宽限6天，
不能把整盘df余量视作用户额度。按既有授权审阅并退役已终止38.406B的首代checkpoint和第二次保存暂存，
共348,423,359,268 bytes；当前约312GiB、已低于软限。配置、结果、元数据和发布时hash保留，
详见现有[容量证据](evidence/core5-platform-capacity-20261031.md)。未动B512复现、环境、活动输入或原GPU dev保全包。
后续大输出优先tmpfs，联合主存准入；持久盘仅保存小记录，不恢复已取消的12.8B/38.4B存储成本实验。

## 下一步与八卡后继

用户已授权七空闲卡加一低占用卡共用，不再机械等待八张无进程卡。当前观察物理4/5/6/7/8/10/11空闲，
物理9有其他账户进程、HBM约5656MiB/65536、AICore0%。不停止或更改该任务。
准备项目站点适配器：保留原队列priority/FIFO、两次采样及八锁整组获取，只对显式PID/启动身份、
利用率和HBM预算许可共用；记录争用，不称独占性能。共享账户helper/全局配置保持只读。
站点适配器、`scripts/run_npu_eight_followup.py`、三项配置及相关测试本阶段提交；尚未启动八卡作业。
驱动将大checkpoint留tmpfs、小结果写持久盘，并在启动前要求低于软配额且距硬限150GiB余量。

有限后继：原目录约1B八卡推理；FC16/d8八卡小训练；仅小训练严格通过才进入新loss1e-9约1B八卡训练。
保留原三个八卡排队超时，不重跑原0.001已知非有限配方。额外12.8B八卡存储流程不作收尾门槛。
先提交站点适配/配置，再对固定源码做队列与相关检查，串接持久任务，一次交接后无需在线等待。

审阅189目录：此前184个名称有结果，两个2B预选点被更贴近容量的实际点替代，三个8卡队列超时。
后继须分别计数；替代不认证原case、诊断不计吞吐。六项经验与最终可查询选型表随八卡终态收尾。
GPU不再访问；T4真机待验。普通推送可用现有项目私有Git SSH配置，不改全局网络。

```sh
python scripts/status.py
quota -vs
systemctl --user show gef-npu-compute-followup-4485697.service gef-npu-resume-immediate-61571c4.service gef-closure-4485697.service -p Id -p ActiveState -p ExecMainStatus -p MainPID -p ControlGroup
```

本机aarch64、320 affinity CPU、约1.8TiB可用内存；16张64GiB级NPU，Python Torch/TorchNPU2.12/CANN9.1，独立SDK LibTorch2.10/CANN9.0另ABI。
以上是近期观察，新任务仍现场准入；主存快照不按卡数相除，tmpfs与主存同一资源。
干净接续从[development](development.md)、[人类手册](human-workflow.md)、[有限配置](platform-matrix.md)进入。
查询：`tidegraph results docs/evidence/core5-platform-*-20261031.json.gz docs/evidence/core5-finite-packing-20261031.json.gz --format table`。
旧源码ID用`python scripts/resolve_history_commit.py OLD_SHA`定位；映射不等于重新验证。
