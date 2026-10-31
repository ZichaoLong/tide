# 当前交接 — 三平台矩阵收尾（2026-10-31）

用户已授权继续，不再执行旧“提交后暂停”。当前策略：不扩盘、不重启GPU；
依最新用户指令优先计算，使用固定源码收齐训练与必要恢复诊断；额外大文件存储测量不作为门槛，不重跑已审阅矩阵。
本机/远端tmpfs仅作经主存联合准入的易失周转，活动作业的路径与输入不改变。

授权分支`graph-execution-foundation`；本次重入HEAD为03292c3、工作区干净且已推送。
数学固定`0770593838a37c790a568226930567ed60f99b83`，离线校验通过。
实际当前提交用`git status --short --branch`和`python scripts/status.py`核对，不回退后续源码。
旧账户指南已不存在，依用户确认、项目site和`/opt/software/ascend-locks/dev-workspace-9.1.0/README.md`继续；
远端脏dev保持原样，不作资格源码。参考仓库只读。

## 已审阅结果

截至本次重入，八份platform/finite-packing归档含197条唯一run，包含观察、拒绝与失败，不是194项性能通过。
完整配置、来源、逐窗结果、峰值与原严格诊断在各gzip；旧二进制不冒称按最新Python提交重建。

| 范围 | 当前事实与证据 |
| --- | --- |
| 三平台有限矩阵与大batch | [首组](evidence/core5-platform-baseline-20261031.md)：98正式+4观察，三平台中等B1/8/32/128/512、B128 A/B/C、三图层级/拓扑/客户端、小训练；43正式严格通过，原数值失败保留 |
| 约1B与FC/HB/LH | [规模](evidence/core5-platform-scale-20261031.md)：19正式，CPU7/GPU5/NPU7；18完整离散/精确恢复通过、数值探索。NPU1B/B512下一窗KV keys精确恢复失败、独立CPU未检查，原checkpoint保留 |
| 有限优化与六项packing | [诊断packing](evidence/core5-finite-packing-20261031.md)：三平台18记录，6小训练严格通过；B128成本下降但数值失败不解除。[profile](evidence/core5-platform-profile-20261031.md)及[六项处置](performance-lessons.md#本轮三平台六项打包处置)分别记录大MM、小BMM、分桶、同步与停止条件 |
| 同模型多卡 | [多卡](evidence/core5-platform-multicard-20261031.md)：CUDA1B两/三/四卡、NPU两/三卡推理离散/恢复通过、数值探索。小训练CUDA三通信及NPUordinary/cpu_bridge严格通过；NPUblocking在exact-resume SIGSEGV。CUDA三通信full独立根另已通过 |
| 单卡resident | [resident](evidence/core5-platform-resident-20261031.md)：两平台B1/小训练严格通过；有界CUDA B128/B512、NPU B128完成且恢复通过、数值探索。同材料eager较便宜；NPU B512构造EE1019/507002，未进入计时；原默认布局拒绝保留 |
| 原1B训练 | [训练容量](evidence/core5-platform-training-capacity-20261031.md)：CUDA单卡静态拒绝；CUDA双/四卡与NPU单/双卡原配方第三窗loss非有限。31a3b32准确定位`runtime.forward_and_loss[loss]`，原失败和未执行的检查/恢复保持 |
| 实际容量 | [容量](evidence/core5-platform-capacity-20261031.md)：CUDA4.561B单/2/3/4卡、NPU12.802B单/双/三卡均完整执行/离散/精确恢复通过、FP32数值探索。CUDA9.123B单卡bank超容量拒绝、双卡实际allocated20.058GB/卡、完整离散/恢复通过；不是多卡resident资格 |
| checkpoint成本 | [成本](evidence/core5-checkpoint-cost-20261031.md)：4.561B磁盘/tmpfs整流程453.560/427.310s；36baeb7磁盘interval2为324.107s，恢复289.587→223.215s、RSS74.880→59.716GB。少保存一次是耐久性取舍，不称算子加速；三条均exact-next通过、独立数学未检查 |
| 人类与下游入口 | [实际演练](evidence/core5-human-workflow-20261031.md)：CPU/NPU/GPU公开安装/配置/检查/推理训练/停止恢复/CSV HTML完成；各自base严格失败与下游小任务通过分列。[手册](human-workflow.md)、[下游模板](downstream-contract.md)、T4配置/迁移步骤已交付；T4真机未验 |

36baeb7 checkpoint优化已通过固定CPU110项（1 standalone明确deselected）、CUDA返回存储6项及双卡full推理/独立根/三步训练/optimizer/恢复。
NPU同实现回归仍在下列队列，不由CUDA或旧NPU通过认证。旧CPU/NPU/GPU失败不因新安装或外层exit0改为通过。

## GPU工作已结束，可释放实例

09:52Z已向用户确认本轮不再需要gpu3090：四卡无计算进程，全部项目监督终态且无存活后代，
结果、身份、配置和必要小记录均已回收。后续本机任务不依赖SSH或GPU运行目录。
GPU新配方双卡1.014B训练的执行/离散/精确恢复完成，独立数值超差保留；
其7.721GB可重建checkpoint及此前9.123B/4.561B大文件已审阅退役，不跨机搬运。

**必须保留原远端dev修改备份**：`artifacts/cuda3090/retirement-dev-20261031.tar.gz`，
5,452,397 bytes，SHA256 `6b89b411611378a619149d69870c35e3f587db0baf1ed70fd2a21a005e2fe31e`。
基线cc6b45f，16项已跟踪修改+4项未跟踪源码，含基线archive、working/index二进制patch与逐文件hash；
已本机解包读取校验，未修改远端dev，也不把这些原有修改称资格实现。此备份不是可随意删除的实验产物。
释放审阅在[小记录](evidence/core5-cuda-release-20261031.json)，私有完整记录在`artifacts/cuda3090/release-review-20261031.json`。
用户自行释放实例；没有执行宿主环境/驱动变更。

## 本机串行依赖链

所有unit在`background.slice`，表中从上到下依赖，waiting不等于实际占卡执行。
09:52Z：38.406B三卡原1800秒到限，监督failed/124；第一次保存957.250秒、warmup5.442秒、next_window1已发布。监督终态时仍有不可中断I/O进程1908740，09:52Z核对该进程已消失且原控制组已空；原监督残留记录仍保留。七卡任务等待241秒后完成，完整离散/恢复通过、数值探索，中位0.804秒；blocking单次诊断完整通过但未复现原SIGSEGV，不称修复；三通信full独立根09:56:35Z全部终态/exit1：完整独立推理和精确恢复通过，随后均EE1017；首根/具体调用点待定位。B512恢复诊断09:59:40Z终态：重放到旧checkpoint边界已逐位不同，采样布局与参数布局相同；不能归因序列化。后继返回测试因私有环境缺pytest未执行；单卡full独立输入根数值超差；新loss双卡small-full又EE1017，1B单/双卡未运行。三项unit现全部终态且控制组为空。
每项使用冻结安装/输入，不编辑任务脚本；sharded提交记录中只有`M docs/STATUS.md`的冻结交接差异，不能改称clean。
目录前缀均为`artifacts/platform-matrix/`，服务名统一加`gef-`及`.service`。

| unit短名 | 待完成工作/停止条件 | 输出目录或progress |
| --- | --- | --- |
| platform-final-1e33b4a | 已终态；12.802B单/双/三卡已审阅，八卡等待超时未运行 | `capacity-human-followup-1e33b4a/progress.json`；`npu-capacity-12802m-cards-N-final-bd2ff2e` |
| platform-sharded-3e4f6b2 | 单/双卡拒绝；三卡超时124；原不可中断I/O进程已退出、控制组空 | `sharded-capacity-3e4f6b2/progress.json`；`npu-capacity-38406m-cards-N-sharded-bd2ff2e` |
| platform-seven-2562263 | 同一1.014B七卡推理，4/2/2/2/2/2/2节点；原八卡等待超时保持 | `npu-model-1b-cards-7-bd2ff2e` |
| blocking-diagnostic-bd2ff2e | 原两NPU blocking小训练一次faulthandler诊断；不重试抹旧SIGSEGV | `npu-blocking-diagnostic-bd2ff2e` |
| multicard-roots-844eb9b | 已终态exit1；两NPU三通信full均EE1017，根与后续训练未完成，保留原失败 | `npu-multicard-roots-bd2ff2e` |
| resume-layout-61571c4 | 已严格失败/exit1：新进程重放与旧checkpoint在raw边界不同，旧checkpoint保留 | `npu-resume-layout-61571c4` |
| checkpoint-check-compute-6c6eaa2 | 已终态/exit1：返回测试缺pytest未执行，单NPU full独立输入根数值失败 | `npu-checkpoint-check-36baeb7` |
| scaled-training-compute-6c6eaa2 | 已终态/exit1：新loss双卡small-full报EE1017，单/双卡1B依赖未运行 | `npu-scaled-training-36baeb7/progress.json` |

用户调整计算优先级后，未开始的12.802B tmpfs成本对照及38.406B interval3重跑已取消，不作为完成门槛。
为保持串行依赖，原checkpoint-check/scaled-training两个仅等待的启动器亦取消并按上表重排；四个原外层cancelled/143且控制组空，无实际case开始，不算数值失败。新启动器外层14400秒，其余原外层86400秒含依赖等待。合作队列priority0、候选物理4–11、max-wait600秒；最新队列只见6张候选卡空闲，其他项目占用不干预。
原八卡1B推理/训练各等待约607秒退出3，未运行。普通复制通过不解除直接P2P/原生跨卡原子及多卡resident门禁。
本机原NPU1B/B512失败checkpoint在`npu-scale-run-1b3e39f/npu-scale-fc-few-1b-b512/run/checkpoints`，
诊断使用原安装`install-1b3e39f/venv`，包hash`6af9b3981aa512982618b31b02ad69db5efd20c31438b6cb273038d877c68d29`。
原始raw边界若已不等，不能把新进程重放差异直接归咎序列化。

```sh
python scripts/status.py
systemctl --user show gef-platform-final-1e33b4a.service gef-platform-sharded-3e4f6b2.service gef-scaled-training-1a6b08f.service -p Id -p ActiveState -p SubState -p ExecMainStatus -p MainPID -p ControlGroup
```

本机job记录/log在`artifacts/jobs/`去掉`gef-`和`.service`的目录；只有终态、退出码及无活跃后代共同核对后才能归档。
没有收到停止作业指令；如后续收到，停止准确unit并核对控制组，不停止其他项目。

## 固定环境与接续

本机aarch64、320 affinity CPU、约1.8TiB可用主存；16张64GiB级NPU，Python Torch/TorchNPU2.12/CANN9.1；
独立SDK LibTorch2.10/CANN9.0另ABI。GPU四张3090/sm86、Torch2.12.1+cu130/Toolkit13，56核配额、240GiB主存、50GiB盘。
CPU core de999c2/standalone eca070f；NPU core/resident/standalone eca070f；CUDA core/resident de999c2、standalone eca070f。
36baeb7 wheel SHA256 `7b5f71c545568b2389f0eeb97914787b16c4ab5378c220c71826ffea534e12d9`；
GPU安装内容hash`5c99b421a7e0ba7a9c09a0943f82b5f67cc1a5aa874e98a2f19d8167db243d1e`。

NPU12.802B原host241.331GB、两代disk117.232GB；tmpfs联合host357.168GB含未来文件116.151GB。
NPU38.406B三卡各58.566GB并另留6GiB，host766.974GB；原两代disk349.567GB，后继单代175.320GB。
本机shm约1.034TB空、MemAvailable约1.8TB，cgroup v1各层无限额已人工核验；不宣称代码自动支持所有v1限额。
这些是近期观察，下一次新作业仍现场准入；disk.estimated_new_bytes已含reserve，不加两遍。

下一步：6095c6a训练主存预算52.585GB及45项固定定向检查已闭环。公共Python/native入口补充`checkpoint-save`可见阶段，开发30项及固定6c6eaa2的30项均通过（tmpfs临时文件、持久日志）；unit已终态无后代，不修改活动包。真实I/O停止延迟又触发只读状态修正：终态监督仍核对原workload启动身份，历史未完成超时不显示成仍在运行；开发20项及固定122a1ed的20项均通过（无失败/跳过）。沿本机链逐项收终态，
将容量、训练、恢复诊断分别补入既有证据，依据首个见证决定有界修复或明确停止。
最终核对189原目录的运行/拒绝/替代项和目录外后继，更新ROADMAP/六项经验/可查询结果；全部作业终态后交付。
不重跑旧矩阵，不自动延长原超时。当前总体任务尚未完成；大规模完整落盘/恢复的额外成本实验按最新用户指令有界停止，必要小型恢复及训练检查继续。

干净仓库从[development](development.md)、[人类手册](human-workflow.md)、[有限配置](platform-matrix.md)进入。
查询：`tidegraph results docs/evidence/core5-platform-*-20261031.json.gz docs/evidence/core5-finite-packing-20261031.json.gz --format table`。
旧源码ID用`python scripts/resolve_history_commit.py OLD_SHA`定位，映射不等于重新验证。

当前开发增量：公共check新增可持久化独立根阶段（不改变算术、容差或调度），CPU定向27项通过。
提交后在固定源码复验同27项；有界NPU定位仅取原ordinary双卡full一次，若需同步栈再显式
ASCEND_LAUNCH_BLOCKING一次诊断，不作性能。补齐私有pytest后重做原未执行的6项返回测试。
单卡新loss的计算不由双卡根错误自动阻塞：先按实际执行/独立记录/恢复的小配置准入，
再决定原已声明1B单/双卡探索任务，保留完整独立根失败，不把它们写成合格训练。
B512下一方案须区分同次运行即时解码与跨进程重放；必要checkpoint可用联合预算后的tmpfs，
不开展已取消的大文件成本实验。
