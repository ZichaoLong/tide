# 当前交接 — 2026-10-28

授权分支`graph-execution-foundation`；数值实现`474f9af32d35ca7d8bc7bf6359dbddb3ec91dfb3`（含`56631eb`），
阶段记录设施`b62f2db7e9bd72b652cdcb97b11fa6e1317b396a`。数学固定
`0770593838a37c790a568226930567ed60f99b83`；19/5份新/旧快照已同上游Git对象核对。
参考仓库和公共环境保持只读，不使用子代理。用户授权实现、有限实验、阶段提交及普通推送；不强推、不恢复旧队列。

## 本轮结果

[数值/低秩与设施证据](evidence/core5-local-allocation-factorized-kv-20261028.md)及同名JSON是本轮完整入口；
[语义对应](semantic-map.md)、[参数匹配材料](weighted-workloads.md)、[性能经验](performance-lessons.md)可独立导航和复现。

- 两份原D2048实际分配失败行已提交，独立120位Decimal核对有解；cap质量分位局部区间与约束共同阈值消去修复局部相消，保留正支持/残差/精确解VJP。
- 全部eager KV标量、独立packed与共享行保留低秩式；独立dense公式/Jacobian、参数/状态根、共享、optimizer/恢复分别核验。resident组合矩阵/tape未由本项改变。
- 固定474f9af：CPU687 passed/2 strict xfail，补充59项通过，共746 passed；NPU九项组件/完整训练/单双卡恢复及SDK/Python eager KV各四次更新通过；CUDA构建/loader/CPU检查通过，无GPU真机。
- CPU medium单次366.831→363.974ms，NPU mixed累计2189.634→2145.125ms，完整对照均通过；不足以证明稳定收益。profile独立执行，外部争用未排除。
- D64训练窗和约92.62亿主体参数D2048推理窗均已执行，事后keys严格核对失败（2.26498e-6/4.48227e-5）。大模型推理窗130.351s、测量后生命周期RSS约47.19GiB，不计等价吞吐。
- D64两步FP32返回/梯度/SGD仍失败；30组实际分配局部误差≤8.33111e-8，评分进入求解器前已不同。两条显式FP64两步诊断通过，仅用于隔离，不解除FP32失败。
- b62f2db在计时外持久化fixture/构造/warmup/measurement/验证阶段；开发和固定提交各6项通过，含五族FP32训练、损坏材料及FIFO中断。其资格与数值资格分列。

## 作业终态与来源

所有本轮任务已终止，没有排队或运行中的项目任务；原失败没有重标通过。

| 服务unit（均后缀`-20261028`） | 终态 | 输出目录（仓库相对路径） |
| --- | --- | --- |
| core5-factorized-qualify-cpu-v2 | passed/0，687+2 strict xfail | artifacts/jobs/core5-factorized-qualify-cpu-v2 |
| core5-factorized-qualify-npu-v3 | passed/0，物理1/13，九项及eager | artifacts/jobs/core5-factorized-qualify-npu-v3 |
| core5-factorized-qualify-cuda-v3 | passed/0，仅构建/loader/CPU | artifacts/jobs/core5-factorized-qualify-cuda-v3 |
| core5-factorized-study-cpu | failed/1，5项通过/2项严格失败 | artifacts/jobs/core5-factorized-study-cpu |
| core5-factorized-study-npu | passed/0，两项，物理13→npu:0 | artifacts/jobs/core5-factorized-study-npu |
| core5-benchmark-progress-dev | passed/0，开发构建+6项 | artifacts/jobs/core5-benchmark-progress-dev |
| core5-benchmark-progress-qualify | passed/0，固定b62f2db构建+6项 | artifacts/jobs/core5-benchmark-progress-qualify |

每个目录含status.json/task.log；study另有runs/study.json，CPU/设施资格有qualification.json和JUnit，NPU有checks/result.json。
补充59项在`artifacts/diagnostics/core5-factorized-extra-tests`。各阶段源码/快照/binary hash和原失败清单已收录证据JSON。
数值固定树`artifacts/worktrees/core5-private-qualified`，设施固定树`artifacts/worktrees/core5-benchmark-progress-qualified`；均保留未修改。
CPU core复用前已核对完整C++内容及每个binary hash，消费者从固定源码重建；开发manifest基提交不能代替资格身份。
长任务均独立background.slice、Nice10、算子线程1/build2；资源/时限、物理映射和终态见报告，不宣称资源独占。
本文件随证据提交；提交/推送身份与是否有未提交修改，以`git status --short --branch`及`git log -3 --oneline`核验。

## 剩余限制与接续

总体大图FP32严格正确性仍未通过；不能以局部修复、FP64诊断或完成计时窗宣布全目标完成。
原eager FP16 GDN默认AdamW、SDK FP32 affine、默认异步双NPU VJP和反馈PDG见证继续保留。
CUDA真机、x86_64、大型设备完整训练/跨卡吞吐未验；有限容量/正支持数值拒绝不静默放宽。
共享组外同形投影、resident组合矩阵、物理缺席行和窗口反向构造仍有性能空间；当前没有低成本优化已穷尽的结论。

当前约定的F1—F4实施、必要资格和有限测量均已终态，证据/经验已入库；唯一后续清单见[ROADMAP](ROADMAP.md)。
接续先运行`git status --short --branch`、`python scripts/status.py`、`python scripts/check_upstream.py`，再读上述报告。
下一项有依据的调查是D64首个真实投影/评分差异的传播与一阶训练边界，先提取小操作数反例；
不要自动重跑本次大型材料或全开关矩阵。现有材料/命令可由公共workload工具重建，不依赖历史绝对运行目录。
