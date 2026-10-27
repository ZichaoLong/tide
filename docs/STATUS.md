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

- 完整评分/证据/key 与共同阈值 VJP `cacb645adc64ed06e454f7ac8d8572bf09b98ee2`：
  **593 passed / 0 failed / 0 skipped**，CPU FP64/FP32；[证据](evidence/core5-scoring-20261027.md)。

## 当前任务与下一动作

评分资格 unit `tide-core5-scoring-qualify-20261027` 已终态 inactive/success、exit 0。
只读 worktree `artifacts/worktrees/core5-bootstrap` 干净固定 cacb645；结果在
`artifacts/jobs/core5-scoring-qualify/{status.json,task.log,development.json,source.tar.gz}`。
launcher `.local/core5-scoring-qualify.sh`，全部 16 个带权测试入口加 standalone 局部公式。
实现与证据本次推送；失败详情保留如下。无 live job。

主工作区下一增量未提交：实际 KV ragged 投影与 affine/delta 前缀复合。
新 Python weighted/{packing,kv_packed,recurrent_packed}.py 和独立 C++ weighted_*packed，
接入 packed_sequence、sequence、独立 batch；没有调用逐步 Upd 补写数值。
零接收原样保留 raw tuple、时间差先 int64、完整正前缀可解码；现有语义 VJP 重放仍保留，
不声称完成批量反向。新 Python 独立前缀/非交换矩阵/稀疏时间/独立根 **18 项通过**。
C++ 尚待构建；不是评分 cacb645 的资格。无用户修改。

下一命令：补 standalone packed 手算与 native 入口测试；用 `.local/freeze-core5.py tide-core5-scoring-qualify-20261027`
冻结开发树，新的 background.slice 服务构建并运行 tests/test_weighted_packed.py 和相关执行回归。
之后继续 GDN/SSD 的固定时间卷积联合特征、批量 VJP、mixed/resident、
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
