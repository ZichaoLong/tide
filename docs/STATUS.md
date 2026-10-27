# 当前交接 — 2026-10-27

固定上游 `0770593838a37c790a568226930567ed60f99b83`，授权分支 graph-execution-foundation。
持续实施、私有环境、有限实验、提交和推送均已授权；不使用子代理。参考仓库只读。
历史等待授权/提交后暂停已被取代。完整未完成范围以 [ROADMAP](ROADMAP.md) 为准。

## 阶段结果

- 一般 SettleGraph `efb9d62`：247 项固定源码 CPU 定向资格，证据 d0acd3c，已推送。
- 结构化 Read/int64/SrcClass `714d5f9`：212 项固定源码 CPU 定向资格，证据 cbdd3fd，已推送。
- KV 首组合 `6fb467c`：129 项固定源码 CPU 定向资格，证据 572d4ff，均已推送。
- affine/delta、完整 GDN/SSD、固定时间短卷积与异构绑定
  `442e8b2701164eb7e4241da56c8a395a832ebeb0`：**544 passed / 0 failed / 0 skipped**，
  CPU FP64/FP32；[契约](weighted-recurrent.md)、[证据](evidence/core5-recurrent-20261027.md)。
  证据 955912c 与实现已一起推送。首组合和循环阶段均不代表完整总体任务完成。

- Bernoulli/Shannon/固定 alpha ent 分配 `a05d3ffc946985cfeb124a5e3b3074ceeece306d`：
  **226 passed / 0 failed / 0 skipped**，CPU FP64/FP32；[证据](evidence/core5-entropy-20261027.md)。

- 十种发送/节点增益 `4a0858880831cb46ad96191ab9dd58fca5260223`：**320 passed / 0 failed / 0 skipped**，
  CPU FP64/FP32；[证据](evidence/core5-sending-20261027.md)。实现与证据 76c698e 均已推送。

## 当前任务与下一动作

评分与分配 VJP 增量已完成开发：match/att/grouped、content_stats/content key，
独立参数和显式共享、3/4/5/6 字段描述、五种异构记忆、Read0/-/+、PDG、AdamW/恢复。
分配增加显式共同 score_shift 及其活动总量约束的解析连接零 VJP；保留 caps/bounds 路径。
没有改前向、容差、dtype 或加梯度裁剪；共同 cotangent 中心改善相消。
Encoder/Query 一维输入统一矩阵乘列向量顺序，保留任意 leading batch axes。
相关代码、测试、契约同增量提交；无用户修改。

当前无 live job。最后主工作区定向复核：
`TIDE_BUILD_DIR="$PWD/artifacts/worktrees/core5-bootstrap/build/cpu" python -m pytest -q tests/test_weighted_scoring.py tests/test_weighted_scoring_execution.py --dtype both`
**66 passed / 0 failed / 0 skipped**；完整整图/梯度/优化器/续接的原评分失败已消失。
这是开发检查，不能替代固定源码资格。

下一动作：固定该实现提交，冻结 task-owned worktree `artifacts/worktrees/core5-bootstrap`，
将 worktree HEAD/index 对齐固定提交并核验干净只读；通过 `develop.py --reuse-build` 核验
C++ 源码和二进制 hash，启动新的 scoring-qualify background.slice 服务。
联合检查 allocation_shift/allocation/entropy、scoring/scoring_execution、sending/execution、
weighted KV/recurrent/config 和 standalone 局部公式；精确命令在资格 launcher 与 job 记录。
资格终态后独立提交证据并推送，继续精确联合/packed/批量 VJP、mixed/resident、
standalone 完整训练与有限 profiling。总体任务尚未完成，持续推进。

## 本增量保留的失败与修复

| job / 冻结树 | 终态 | 原因与修复 |
| --- | --- | --- |
| scoring-dev / f1b561974cbeeb8926b4e75c9d91c3b46d50addc | 345 passed / 3 failed，exit 1 | FP32 共同阈值残差被 AdamW 放大；解析 shift VJP 修复。两个 Read 快照根的数值顺序差见下 |
| scoring-vjp / 9b5b17a17c3363857bff997bf8c91e4ec9946bb2 | 246 passed / 40 failed，exit 1 | C++ custom Function 的 undefined shift 占位无 device；改为不求导零标量 |
| scoring-vjp-fixed / 1304ab4b2523095303c7cd888b0142799bceb49e | 301 passed / 1 failed，exit 1 | FP32 PDG Read− key.out_bias 差 2.0266e-6；统一 Encoder/Query 求值顺序后所有 cut 通过，诊断最大差 2.98e-8 |

原日志/status/source 在 `artifacts/jobs/core5-<job>/`，全部失败保留。
最小诊断 `.local/diagnose-core5-scoring.py` 与 `.local/diagnose-scoring-order.py`；
输出分别在 scoring-dev/diagnosis.txt 和 scoring-vjp-fixed/order-diagnosis.txt。
不得把新通过结果覆盖旧失败。

## 环境与保留事项

实际路径 /mi/data2T/zlong/home/llm/graph-execution-foundation；/home/zlong 指向当前 home。
规范 /opt/software/ascend-locks/dev-workspace-9.1.0/README.md 与对应 /etc/modulefiles。
`source /etc/profile.d/modules.sh; module purge; module load ascend/dev-workspace-9.1.0; source .local/venv-core5/bin/activate`。
Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1、aarch64。未修改公共环境。
CUDA 源码/构建/真机分别记录；无 GPU 不阻塞其他工作。

旧 bootstrap 缺 Ninja、缺 TIDE_BUILD_DIR 的 24 项前置失败、structured-dev 的 210/2、
weighted-kv-dev 的 C++ Options 构建失败，以及 recurrent-dev launcher 失败保持原记录。
recurrent-build 的 212/1 是比较目标字段归约顺序问题，修复后新开发和不可变资格分别通过；
原因与修复详见循环证据。原失败日志/源码留在相应 artifacts/jobs 目录。
