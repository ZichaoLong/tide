# 当前交接 — 三平台矩阵收尾（2026-10-31）

用户已授权继续，不再执行旧“提交后暂停”。当前策略：不扩盘、不重启GPU；
使用已提交的固定源码串行任务收齐容量、训练及恢复诊断，不重跑已审阅矩阵。
本机/远端tmpfs仅作经主存联合准入的易失周转，活动作业的路径与输入不改变。

授权分支`graph-execution-foundation`；本次重入HEAD为b061404、工作区干净且已推送。
数学固定`0770593838a37c790a568226930567ed60f99b83`，离线校验通过。
实际当前提交用`git status --short --branch`和`python scripts/status.py`核对，不回退后续源码。
旧账户指南已不存在，依用户确认、项目site和`/opt/software/ascend-locks/dev-workspace-9.1.0/README.md`继续；
远端脏dev保持原样，不作资格源码。参考仓库只读。

## 已审阅结果

截至本次重入，八份platform/finite-packing归档含194条唯一run，包含观察、拒绝与失败，不是194项性能通过。
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

## 当前GPU作业

目标`gpu3090`，私有入口`artifacts/cuda3090/ssh.sh`，根`/root/autodl-tmp/gef-cuda3090`。
`jobs/platform-scaled-training-1a6b08f`已于08:55:33Z终态exit0，两个子监督及外层均无后代。
小型两卡full通过；1.014B双卡新loss流程完整执行、离散/精确恢复通过、独立数值仍超差探索。
两测量窗1.260/1.398s，三个loss约6.33/6.21/6.06；最终参数通过，梯度/optimizer整体超差，见训练容量证据。
配置来自1a6b08f，实际包36baeb7；只将loss系数0.001→1e-9，是新任务，不是原失败修复，不继续搜索。
主存恢复RSS40.265GB超过原静态估算35.261GB，6095c6a已补齐CPU训练快照预算并通过固定45项检查；现场余量充分，原结果不改写。
产物`cuda-scaled-training-36baeb7/progress.json`与各`*-supervisor/status.json`保留；GPU现无本项目活动作业。

9.123B原两代磁盘83.896GB拒绝保留；新interval3仅最终保存，单代41.401GB，双卡08:45:38Z前已完成。
外层因单卡拒绝failed/1、双卡子项exit0，均无后代。经b061404审阅/dry-run后退役其可重建checkpoint，
凭据`retirement-capacity-periodic-36baeb7.json`；先前dba2a5e及36baeb7的4.561B checkpoint亦已审阅退役。
全部小结果/config/log保留，不跨机搬运这些大文件。

```sh
artifacts/cuda3090/ssh.sh 'cd /root/autodl-tmp/gef-cuda3090 && install-checkpoint-36baeb7/venv/bin/tidegraph job status jobs/platform-scaled-training-1a6b08f'
```

## 本机串行依赖链

所有unit在`background.slice`，表中从上到下依赖，waiting不等于实际占卡执行。
09:18Z：final已终态exit1、控制组为空；12.802B单/双/三卡均已审阅，八卡等待607.801秒退出3、未运行。38.406B单/双卡bank超容量现场拒绝；三卡通过live准入，正在构造。其后依赖尚未取得结果。
每项使用冻结安装/输入，不编辑任务脚本；sharded提交记录中只有`M docs/STATUS.md`的冻结交接差异，不能改称clean。
目录前缀均为`artifacts/platform-matrix/`，服务名统一加`gef-`及`.service`。

| unit短名 | 待完成工作/停止条件 | 输出目录或progress |
| --- | --- | --- |
| platform-final-1e33b4a | 已终态；12.802B单/双/三卡已审阅，八卡等待超时未运行 | `capacity-human-followup-1e33b4a/progress.json`；`npu-capacity-12802m-cards-N-final-bd2ff2e` |
| platform-sharded-3e4f6b2 | 单/双卡已拒绝；三卡已通过校准条件及live准入并构造，原1800秒 | `sharded-capacity-3e4f6b2/progress.json`；`npu-capacity-38406m-cards-N-sharded-bd2ff2e` |
| platform-seven-2562263 | 同一1.014B七卡推理，4/2/2/2/2/2/2节点；原八卡等待超时保持 | `npu-model-1b-cards-7-bd2ff2e` |
| blocking-diagnostic-bd2ff2e | 原两NPU blocking小训练一次faulthandler诊断；不重试抹旧SIGSEGV | `npu-blocking-diagnostic-bd2ff2e` |
| multicard-roots-844eb9b | 两NPU ordinary/blocking/cpu_bridge各一次full独立根/训练/恢复，300秒/case | `npu-multicard-roots-bd2ff2e` |
| resume-layout-61571c4 | 原1B/B512 raw边界/下一窗及布局诊断，900秒；原1b3e39f安装和checkpoint只读，不改布局或算术 | `npu-resume-layout-61571c4` |
| checkpoint-npu-0f9254a | dba2a5e小训练先验，再有条件12.802B单卡tmpfs成本观察1200秒；不认证36baeb7 | `npu-checkpoint-cost-dba2a5e/progress.json`；大输出`/dev/shm/gef-checkpoint-npu-dba2a5e/run` |
| checkpoint-check-36baeb7 | 新安装返回存储6项及单NPU小full、interval2；180/300秒 | `npu-checkpoint-check-36baeb7` |
| capacity-periodic-beaf34e | 仅原38.406B三卡已完成有限窗且在保存/恢复被监督超时124、新回归通过时，跑interval3一次1800秒；其他情况明确not-run | `npu-capacity-periodic-36baeb7/gate.json` |
| scaled-training-1a6b08f | 新loss先两卡小full，通过后1.014B单/双NPU各一次1800秒；包36baeb7、配置1a6b08f | `npu-scaled-training-36baeb7/progress.json` |

外层上限86400秒含依赖等待。合作队列priority0、候选物理4–11、max-wait600秒；最新队列只见6张候选卡空闲，其他项目占用不干预。
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

下一步：6095c6a训练主存预算52.585GB及45项固定定向检查已闭环；沿本机链逐项收终态，
将容量、训练、恢复诊断分别补入既有证据，依据首个见证决定有界修复或明确停止。
最终核对189原目录的运行/拒绝/替代项和目录外后继，更新ROADMAP/六项经验/可查询结果；全部作业终态后交付。
不重跑旧矩阵，不自动延长原超时。当前总体任务尚未完成。

干净仓库从[development](development.md)、[人类手册](human-workflow.md)、[有限配置](platform-matrix.md)进入。
查询：`tidegraph results docs/evidence/core5-platform-*-20261031.json.gz docs/evidence/core5-finite-packing-20261031.json.gz --format table`。
旧源码ID用`python scripts/resolve_history_commit.py OLD_SHA`定位，映射不等于重新验证。
