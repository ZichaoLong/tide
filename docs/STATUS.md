# 当前交接 — 三平台计算收尾（2026-10-31）

用户已授权后台继续，最新要求一次安排完后不必在线等待，之后由用户唤醒审阅；GPU可释放。
本机优先计算与必要正确性/恢复，额外大文件成本实验已取消。
不扩盘、不重启GPU、不重跑已审阅矩阵；tmpfs仅在联合主存准入后周转。活动输入和源码不编辑。
授权分支`graph-execution-foundation`，本次重入03292c3干净；实际HEAD以`git status --short --branch`为准。
实现4485697及其首批资格697cafe已普通推送；827cb2c为后续恢复诊断证据，最终还需推送。
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

八份platform/finite-packing压缩归档现含197条唯一run（包括观察、拒绝、严格失败），不是197项性能通过。
完整配置、实际来源/hash、工作量、逐窗及峰值均在归档；旧C++库没有冒称随较新Python提交重建。

| 范围 | 事实与证据 |
| --- | --- |
| 三平台/大batch | [首组](evidence/core5-platform-baseline-20261031.md)：98正式+4观察；三平台B1/8/32/128/512及B128 A/B/C、三层级/拓扑/客户端、小训练；43正式严格通过 |
| 约1B/更多节点 | [规模](evidence/core5-platform-scale-20261031.md)：CPU7/GPU5/NPU7正式点，FC/HB/LH；18完整离散/精确恢复通过、数值探索；NPU1B/B512原下一窗逐位失败、独立CPU未检查 |
| 六项packing/优化 | [finite](evidence/core5-finite-packing-20261031.md)：18记录、6小训练严格通过；[profile](evidence/core5-platform-profile-20261031.md)及[六项决定](performance-lessons.md#本轮三平台六项打包处置)保留收益、无收益与停止条件 |
| 同模型多卡 | [多卡](evidence/core5-platform-multicard-20261031.md)：CUDA1B 2/3/4卡、NPU2/3/7卡推理离散/恢复通过、数值探索；7卡是真实4/2/2/2/2/2/2节点。8卡原队列超时未运行 |
| resident | [resident](evidence/core5-platform-resident-20261031.md)：两设备小点/小训练严格通过；CUDA B128/B512、NPU B128完成/恢复通过、数值探索；NPU B512构造EE1019/507002。多卡resident门禁不变 |
| 原1B训练 | [训练](evidence/core5-platform-training-capacity-20261031.md)：原loss单/多卡第三窗首先在loss非有限；新loss 1e-9是新工作负载，CUDA双卡执行/离散/精确恢复通过、独立数值超差 |
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
- [B512诊断](evidence/core5-platform-resume-diagnostics-20261031.json)：原包1b3e39f重算三窗与旧checkpoint在raw边界即不同，首三个样例state及所有参数布局相同。不能归因序列化；同次新建checkpoint后继用于进一步区分。

## 当前持久任务

均在background.slice，冻结源码/安装；第二项等待第一项终态且控制组清空，不编辑其输入。
原resume-layout、checkpoint-check-compute、scaled-training-compute均终态/exit1且无后代；原新loss的1B单/双卡未执行。
新的实际loss训练与任意独立根资格分开：仅对应卡数的小训练执行/严格数值/离散/槽/精确恢复通过后进入1B。

| unit（加`.service`） | 当前与界限 | 输出 |
| --- | --- | --- |
| `gef-npu-compute-followup-4485697` | 已启动：同步all根诊断失败；新loss单卡小训练严格通过，单卡1B执行/离散/精确恢复完成、数值探索；双卡小训练严格通过，双卡1B三个更新已完成，当前恢复/独立检查尚待终态。小项300秒、大项1800秒，队列600秒，总10800秒 | `artifacts/platform-matrix/npu-compute-followup-4485697/progress.json` |
| `gef-npu-resume-immediate-61571c4` | 等前项；原包1b3e39f/脚本61571c4，3原窗+同次新checkpoint+下一窗，900秒/queue600秒；总14400秒含依赖 | `artifacts/platform-matrix/npu-resume-immediate-61571c4`；易失输入输出`/dev/shm/gef-npu-resume-immediate-61571c4` |

第一项包4485697、原NPU库eca070f；输入`artifacts/platform-matrix/inputs-npu-compute-4485697`，包和脚本只读。
第二项保持原1B/B512安装内容hash `6af9b3981aa512982618b31b02ad69db5efd20c31438b6cb273038d877c68d29`；
原失败checkpoint仍在`artifacts/platform-matrix/npu-scale-run-1b3e39f/npu-scale-fc-few-1b-b512/run/checkpoints`，只读。
临时文件先联合host61.446GB+未来tmpfs10.850GB与现场MemAvailable准入，并独立检查shm；不是额外存储成本测量。
合作队列priority0，候选物理4–11；不占用/停止其他项目。日志在`artifacts/jobs/`对应去掉`gef-`的目录。

```sh
python scripts/status.py
systemctl --user show gef-npu-compute-followup-4485697.service gef-npu-resume-immediate-61571c4.service -p Id -p ActiveState -p SubState -p ExecMainStatus -p MainPID -p ControlGroup
```

没有停止作业指令。若用户要求停止，停止准确unit并核对原控制组/后代；不能只改状态文件。
GPU后续不再需要访问。普通Git直连22/443超时，已用项目私有proxy helper和任务专用SSH配置普通推送，未改全局网络/SSH配置、未输出凭据。

## 用户唤醒后的审阅工作

1. 后台已自行串接到终态收集，不需要在线agent逐项批准或启动。用户唤醒后先读`artifacts/platform-matrix/closure-4485697/status.json`及三个准确unit；按实际执行、独立数值/离散、恢复分别归档。未终态不重新排队；若有新分歧，只在声明界限内处理。
2. 不重试原NPU事件方案、不放宽阈值、不重复大模型原loss。已取消12.802B tmpfs成本及38.406B interval3重跑保持取消。
3. 审阅189目录：当前184个名称有结果，两个2B预选点由更贴近容量的4.561B/12.802B选点替代，三个8卡项队列超时未运行。最终须列清替代不等于原case通过、目录外后继和诊断不计吞吐。
4. 更新现有训练/规模证据、六项经验、ROADMAP与可查询CSV/HTML；全部任务终态后做最终选型汇总、普通推送。不得将running或监督成功算资格。

本机aarch64、320 affinity CPU、约1.8TiB可用内存；16张64GiB级NPU，Python Torch/TorchNPU2.12/CANN9.1，独立SDK LibTorch2.10/CANN9.0另ABI。
以上是近期观察，新任务仍现场准入；主存快照不按卡数相除，tmpfs与主存同一资源。
干净接续从[development](development.md)、[人类手册](human-workflow.md)、[有限配置](platform-matrix.md)进入。
查询：`tidegraph results docs/evidence/core5-platform-*-20261031.json.gz docs/evidence/core5-finite-packing-20261031.json.gz --format table`。
旧源码ID用`python scripts/resolve_history_commit.py OLD_SHA`定位；映射不等于重新验证。


终态汇总已准备一次性服务`gef-closure-4485697.service`，等待上面两个unit终态且控制组为空后，
核对每个子监督无后代，只调用已有公共report投影到`artifacts/platform-matrix/closure-4485697/results.{json.gz,csv,html}`。
其`status.json`明确`review_required=true`，收集成功不等于数值资格；中断的running快照单列，不自动变成通过。
源4485697冻结证据197条+本次实际训练结果；不写仓库证据、不自动提交、不启动新实验。
后台最长18000秒含依赖等待，background.slice/KillMode=control-group；日志在`artifacts/jobs/closure-4485697/task.log`。
用户稍后唤醒时继续人工审阅/更新证据、最终覆盖表与正常推送。在线跟进按用户要求结束，后台任务继续。
停止整条剩余链（仅在用户要求时）：
`systemctl --user stop gef-closure-4485697.service gef-npu-resume-immediate-61571c4.service gef-npu-compute-followup-4485697.service`，
随后核对三个ControlGroup为空及原workload无存活后代；不要仅停止汇总器而声称计算已停止。
