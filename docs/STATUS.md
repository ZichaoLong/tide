# 当前交接 — 三平台收尾、产物清理完成（2026-10-31）

本轮CPU/NPU/GPU有限矩阵、同模型多卡、人类CLI及下游流程已完成可闭环的授权范围。
没有活动项目作业，不需继续等待或重新排队；GPU已释放审阅，后续不依赖gpu3090。
原严格失败、探索成本、硬件/运行时限制继续保留；后续问题及重启条件只维护在[ROADMAP](ROADMAP.md)。
用户最新优先计算，未启动的大文件成本对照取消，不扩盘、不重启、不重跑已审阅矩阵。

授权分支`graph-execution-foundation`；本次清理重入49cf37a干净且已同步远端。
固定launcher/配置d3de18f，实际Python安装4485697、NPU库eca070f分别记账；本次只清理已终态产物并保全小记录，
未重新构建或运行实验。数学固定`0770593838a37c790a568226930567ed60f99b83`，离线校验通过。
参考仓库只读。实际当前HEAD及工作区以`git status --short --branch`为准。

## 当前清理终态

已按用户授权收束为`.git`与2,838个Git跟踪文件。固定清理来源`0b07e72`；
26个干净祖先worktree、10个无额外差异的源码快照、环境/构建/原始checkpoint及6个项目tmpfs目录均已退役。
仓库allocated占用279.594GiB→76.410MiB，释放磁盘279.519GiB、tmpfs16.040GiB；
这是删除后、最终证据提交前的实测，后续Git元数据有少量增长。
[清理证据](evidence/artifact-cleanup-tracked-only-20261031.md)保留精确字节、清单和核验范围。

`gef-artifact-cleanup-49cf37a-20261031.service`在`background.slice`于15:58:25Z完成，
持久status为passed/exit0；原PID和控制组已消失，瞬态unit已卸载。
跟踪字节/权限未变，`git clean -ndx`为空，只剩主工作树。
固定数学校验通过；从清理后源码直接查询9份归档得到原204条唯一run，无需Torch或旧安装。
[41份外层任务终态](evidence/retired-jobs-20261031.json.gz)及5份补充记录保留，不将内部失败改为通过。

没有下一项默认计算或清理作业。远端未提交源码及两份早期patch已在仓库外保全，按用户要求稍后分析。
操作状态/日志位于`~/.local/state/graph-execution-foundation/cleanup-20261031/`；
新clone查看结果与重建开发链不依赖此目录。公共软件与参考仓库未修改。

## 最后八卡任务已审阅

`gef-npu-eight-d3de18f.service`于15:07:41Z exit0。MainPID0、ControlGroup空；三个内层监督
与workload均exit0，按启动身份检查无存活后代。[八卡证据](evidence/core5-platform-eight-followup-20261031.md)
及同名压缩归档包含完整配置、实际来源、逐窗/峰值与原结果hash。

| 配置 | 实际结果 | 窗口中位秒 |
| --- | --- | ---: |
| 同一1.014B八卡推理，B1/P2 | 执行/完整离散/精确续接通过；独立FP32超差，探索成本 | 0.861879 |
| FC16/d8八卡小训练，B2/P3 | 8份独立记录/梯度/参数/SGD槽比较及精确续接严格通过 | 1.861679 |
| 1.014B八卡新loss1e-9训练，B1/P2 | 小训练门禁通过后执行；离散/精确续接通过、独立FP32超差 | 1.820309 |

物理4–11中物理9按明确PID/启动身份、HBM/利用率预算共用；不称独占性能。
每卡2节点，8个共享组完整放置。大推理每卡allocated峰值约0.591GB；大训练各卡约1.356–2.063GB，
主存峰值45.140GB。多卡降低设备存储并不将host快照按卡数等分，也不证明达到八卡容量上限。
当时大checkpoint位于已准入tmpfs；实际文件解码/新owner恢复通过，不认证机器重启耐久性。
这些终态原始材料现已按授权退役，原身份/结果保留。
原八卡三项排队超时保留，新后继不是原失败改写；任意根和原loss边界不解除。

## 已交付与实际证据

九份platform/finite-packing归档含204条唯一run：CPU48、NPU78、CUDA78；包括观察、拒绝和失败。
查询字段为69 strict-checks-passed、113 strict-failure、20 not-established、2 process-failure，
不是204项或69项性能全通过。推理161、训练43；profile/诊断与正式吞吐必须按kind/资格分开。

| 范围 | 事实与证据 |
| --- | --- |
| 三平台/大batch | [首组](evidence/core5-platform-baseline-20261031.md)：B1/8/32/128/512、B128同材料A/B/C、三层级/拓扑/客户端和小训练；98正式+4观察中43正式严格通过 |
| 约1B/更多节点 | [规模](evidence/core5-platform-scale-20261031.md)：CPU7/GPU5/NPU7正式点、FC/HB/LH；18完整离散/精确恢复通过、数值探索；NPU1B/B512下一窗逐位失败，独立CPU未检查 |
| 六项packing/优化 | [finite](evidence/core5-finite-packing-20261031.md)：18记录、6小训练严格通过；[profile](evidence/core5-platform-profile-20261031.md)及[六项决定](performance-lessons.md#本轮三平台六项打包处置)保留源码、收益、无收益及停止条件 |
| 同模型多卡 | [多卡](evidence/core5-platform-multicard-20261031.md)：CUDA1B 2/3/4卡、NPU2/3/7卡及新八卡推理离散/恢复通过、数值探索；7卡与8卡分别实际放置，不能认证T4 |
| resident | [resident](evidence/core5-platform-resident-20261031.md)：两设备小点/小训练严格通过；CUDA B128/B512、NPU B128完成/恢复通过、数值探索；NPU B512构造EE1019/507002。多卡resident门禁不变 |
| 训练 | [训练](evidence/core5-platform-training-capacity-20261031.md)：原loss单/多卡第三窗首先loss非有限；新loss1e-9的CUDA双卡及NPU单/双/八卡执行/离散/精确恢复通过、独立数值超差。新配方不称原失败修复 |
| 容量 | [容量](evidence/core5-platform-capacity-20261031.md)：CUDA4.561B单/2/3/4卡、9.123B单拒绝/双卡实际；NPU12.802B单/双/三卡均完成/离散/恢复通过、数值探索；38.406B三卡有限计算窗5.442s，第二次保存1800s超时，完整恢复/独立参考未验 |
| checkpoint | [成本](evidence/core5-checkpoint-cost-20261031.md)：36baeb7减少重复快照/CPU精确正例比较，CUDA恢复289.587→223.215s；interval2少保存是耐久取舍。固定CPU110项、CUDA返回6项/双卡full通过；NPU返回6项通过、4项显式deselected |
| 人类/下游 | [演练](evidence/core5-human-workflow-20261031.md)：三平台安装、配置、基座检查、运行、停止/恢复及CSV/HTML；base严格失败与下游小任务通过分列。[手册](human-workflow.md)/[下游](downstream-contract.md)/T4迁移配置已交付，T4真机待验 |

6095c6a补齐1B训练host预算52.585GB，固定45项通过；6c6eaa2保存阶段可见性固定30项通过；
122a1ed按启动身份区分终态监督与I/O后代，固定20项通过；d3de18f固定17项及真实八卡锁/映射探针通过。
这些设施检查不代替计算资格，也不更新旧二进制来源。

## 必须保留的失败与材料

- [独立检查](evidence/core5-platform-multicard-checks-20261031.json)：CUDA三通信full通过；NPU三通信完整推理/恢复先通过，任意根均EE1017。4485697定位ordinary双卡all根候选autograd SIGSEGV；单卡为输入梯度超差。显式ASCEND_LAUNCH_BLOCKING=1双卡越过崩溃，参数梯度仍1/64坐标超差1.31130e-6；不称默认异步修复。
- [B512诊断](evidence/core5-platform-resume-diagnostics-20261031.json)：旧边界重放已逐位不同；同进程新checkpoint raw边界精确，下一完整窗keys仍3766/32768坐标逐位不同、最大2.09808e-5。采样布局相同，不归因磁盘损坏，不重复相同候选。
- 原blocking小训练恢复SIGSEGV的一次诊断未复现且完整通过，旧失败不抹除；旧CPU/NPU/CUDA严格失败和strict xfail继续按ROADMAP限定。
- 原B512失败checkpoint、即时诊断及八卡tmpfs原始材料已删除。原配置、失败字段和文件hash保留；以后按[复现入口](reproduction.md)生成新见证，不承诺恢复原字节。
- GPU原dirty-dev包已移动到`~/.local/state/graph-execution-foundation/preserved-source/20261031-cuda3090/retirement-dev-20261031.tar.gz`，5,452,397 bytes，SHA256 `6b89b411611378a619149d69870c35e3f587db0baf1ed70fd2a21a005e2fe31e`；保留16项跟踪修改+4项未跟踪源码。两份早期开发patch另存同目录`additional-source-deltas.tar.gz`。目录权限0700，均待后续分析，不是已合并或已验证源码。

## 资源与接续

本机实际配额400GiB软限、800GiB硬限；本次清理后账户占用32422MiB（约31.7GiB），远低于硬限。
以`quota -vs`核对，不用整盘df替代。tmpfs仍与主存、快照和恢复峰值联合准入。
GPU释放事实见[审阅](evidence/core5-cuda-release-20261031.json)，没有新GPU依赖或作业。
旧账户指南已退役，环境以项目site和公共`/opt/software/ascend-locks/dev-workspace-9.1.0/README.md`为准；
Python Torch/TorchNPU2.12/CANN9.1与独立SDK LibTorch2.10/CANN9.0分别激活，公共环境只读。

从干净仓库按[development](development.md)安装，按[人类手册](human-workflow.md)配置site，
[有限矩阵](platform-matrix.md)中配置可重新生成材料；不需要旧运行目录才能查看结论或启动新实验。
没有默认应重启的历史任务，后续仅按ROADMAP具体新方案或新授权推进。

```sh
git status --short --branch
python -B scripts/status.py
python -B scripts/check_upstream.py
quota -vs
# 清理后无需重建库即可查看原结果；使用Python>=3.10，本机公共3.12路径如下：
PYTHONPATH=python /opt/miniconda/envs/ascend910-dev-workspace-torch-npu-2.12.0-py312-r1/bin/python -B -m tidegraph results docs/evidence/core5-platform-*-20261031.json.gz docs/evidence/core5-finite-packing-20261031.json.gz --format table
```

原始任务终态及八卡progress/review已收入[小型归档](evidence/retired-jobs-20261031.json.gz)，
原绝对路径仅为历史来源；其他三个监督`gef-npu-compute-followup-4485697`、
`gef-npu-resume-immediate-61571c4`、`gef-closure-4485697`均已终态且无后代，原失败不改为通过。
历史源码ID用`python scripts/resolve_history_commit.py OLD_SHA`定位；映射不等于重新验证。
