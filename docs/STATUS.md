# 当前交接 — CPU确认完成，NPU探索已启动（2026-11-08）

首批实际实施继续；未回退启动HEAD 942c329，未修改公共基座、公共环境或其他项目。数学固定0770593、基座85d161d，实际wheel/环境/下游身份见dependencies。已启动任务始终读取各自只读快照，普通工作区的后续提交不改变它们。

## 当前作业与源码

| 作业 | 已核实状态 | 内容、配置和预算 |
| --- | --- | --- |
| confirm-a-s11 / s22 / s33 | 三者completed、退出0、无存活后代 | CPU64/B32各400步，query+FC；旧源码0d7ec5b、`.local/source-first-a`；实际各约42min |
| a1-npu-explore | background.slice实际running，物理1→逻辑npu:0 | 固定0494d20、`.local/source-a-npu`；configs/studies/a1-npu-stage.json；1NPU/CPU4线程/RAM预算24GiB，设备≤3h、等待≤30min、服务≤3.5h |
| pilot-b | completed、退出0、设备与后代已释放 | 原模型/TIDE/MLP各40步；本轮不启动B正式重复 |
| a2-hb-prepare | completed、退出0、后代已释放 | 固定aff03ee、`.local/source-hb-pilot`；原批次HB的CPU FP32接入与恢复已完成；准备≤20min/服务≤22min |
| a2-hb-dense-pilot / a2-hb-sparse-pilot | 依赖已满足，两个CPU训练实际running | 固定训练源码aff03ee及等待驱动e712831；共同B32，准备成功、后代释放与哈希核验后并行各80步；每个CPU1线程/RAM预算8GiB，等待≤20min/服务≤70min |

完整状态、日志、停止与恢复方法见[RUNNING](RUNNING.md)。NPU阶段输出`runs/a1-npu-explore/stage.json`分别保存执行、严格数值、离散语义、机制、恢复和学习；尚未完成的学习不当作通过。CPU确认原源码、配置与环境全过程未修改。

## 审阅结论及后续判断

[TL-A1确认结论](studies/TL-A1.md)：三seed独立test为97.27%/99.22%/99.61%，query-only约13%–14%；移除历史validation降至8.59%–14.45%。只支持当前FC/KV、8类短程检索、CPU64配方的学习与历史利用。三seed的validation均达到原50%且高于query-only20个百分点的门槛，因此准备原计划的小HB准入/先导；不自动启动正式重复，不推导NPU、HB或稀疏收益。旧80步先导仅24.22%的事实保留。

[TL-B1先导结论](studies/TL-B1.md)：原模型/TIDE/MLP验证accuracy为12.50%/17.97%/12.50%，TIDE关闭分支accuracy不变、CE仅变化1.97e-5。接入和梯度成立，实际预测收益未证实；未达25%重复门槛，本轮停止于先导。复用只读缓存Qwen2.5-0.5B-Instruct，没有下载权重或扩大替换。

已重读更新的run-ml-experiments与measurement.md。[研究表](studies/first-batch.md)明确A后续不必限CPU64，也不以消除全部浮点差异为学习前置门槛。新NPU实际宽16/8类/B2两步检查：执行completed、严格数值failed、离散24920项与机制passed，未知原因保留；完整AdamW/model/步号/RNG新进程恢复逐值passed。[原始分轴证据](evidence/a1-npu-admission.json)保留原atol1e-6/rtol1e-5、首次观测分歧及梯度/更新尺度，不将有限误差假定为无害舍入。

NPU batch32完整计时step约19.118秒、1.674例/秒；batch128约81.073秒、1.579例/秒。实际max_full_batch为224/896，并非没有扩大算子batch。128未改善吞吐，80步加验证预计超过2h，因此选32、跳过256；保留完整校准记录。一次有界profile超过170秒执行窗口，后代已清理，记为观察失败；不追加profile、不声称已找到耗时算子。query对照已完成，FC已实际训练；同batch各最多80步，学习仍待审阅。[校准和选择依据](evidence/a1-npu-calibration.json)。CPU与NPU精度/观测不同，不作等价速度声明。

原基座NPU根、历史FP32 AdamW及原Qwen CPU cache严格失败仍见[启动审计](startup-audit.md)和[需求](requests/TL-REQ-001.md)；没有改基座核心、阈值或创建外部issue。CPU64、NPU Qwen+TIDE和新NPU FC各自的恢复证据按原范围解释。

## 资源与查看

账号约165GiB/软300/硬800GiB；首批新增上限20GiB，写入余量充分。本机cgroup v1的CPU/内存控制器未按服务委派，CPUQuota/MemoryMax仅为记录的请求，不能称已生效的每服务硬限制。服务位于background.slice的实际name=systemd层级，运行时限有效；应用线程与实际RSS/HBM另记录。NPU校准B32 allocated峰值约96MiB、B128约229MiB，不包含驱动完整占用；阶段进程RSS抽样约3GiB。HB准备与两个等待服务的请求峰值为48GiB/CPU线程7；当前NPU与两个CPU先导为40GiB/6线程，低于首批总边界。当前工作区状态查看器已修正为读取真实cgroup v1的systemd层级，三个活动训练服务及此前准备服务均核对MainPID与实际后代；没有改活动快照、环境或公共控制器。

Lens使用现有认证实例的`tide-learning-lab`项目，依赖独立记录器，项目JSON/JSONL为权威。六个CPU确认run和B先导均verified；每个run的projection.json给出Lens身份。NPU先记录batch校准，后续学习生成后再出现对应run。私有查看入口/转发方法仅在`.local/LENS.md`，没有部署服务或改变访问策略。

## 接续

先核对NPU的stage、各服务生命周期和Lens投影。HB两个候选的离散/机制/完整AdamW恢复均通过，dense的B2严格检查passed，sparse有限严格失败保留，根因未知；[证据](evidence/a2-hb-admission.json)。共同batch校准已完成，选B32，两先导已从等待转为实际训练。[选择依据](evidence/a2-hb-calibration.json)：B32两者计时step约12.72/12.69秒；B128合计吞吐仅提高约1%，且不满足各40分钟的固定80步预算。当前各80步单seed，不自动正式重复。按原研究表审阅有限HB先导后，才决定本批正式稀疏比较；资源与时限按剩余原预算。首批终态交付每组一页结论、依赖/恢复、Lens映射、研究图和后续可选申请，交用户验收，不自动扩大下一批规模、任务或候选。
