# 当前交接 — 2026-10-27

固定上游 `0770593838a37c790a568226930567ed60f99b83`；授权分支 graph-execution-foundation。
用户授权持续实施、私有环境、有限实验、提交和推送；不使用子代理，参考仓库只读。
总体尚未完成，不在阶段提交后暂停；完整待办以 [ROADMAP](ROADMAP.md) 为准。

## 已固定的阶段资格

均为 CPU FP64/FP32、具体定向范围，不代表总体/设备矩阵完成。

| 实现提交 | 范围 | 通过数 | 证据 |
| --- | --- | --- | --- |
| efb9d62 | 一般 SettleGraph | 247 | [报告](evidence/core5-settle-20261027.md) |
| 714d5f9 | 结构化 Read/int64/SrcClass | 212 | [报告](evidence/core5-structured-20261027.md) |
| 6fb467c | KV 首组合 | 129 | [报告](evidence/core5-weighted-kv-20261027.md) |
| 442e8b2 | affine/delta/GDN/SSD/短卷积 | 544 | [报告](evidence/core5-recurrent-20261027.md) |
| a05d3ff | 熵分配 | 226 | [报告](evidence/core5-entropy-20261027.md) |
| 4a08588 | 十种发送/增益 | 320 | [报告](evidence/core5-sending-20261027.md) |
| cacb645 | 完整评分/evidence/key | 593 | [报告](evidence/core5-scoring-20261027.md) |
| 26dad7d | KV/循环实际 packed 与固定时间卷积 | 378 | [报告](evidence/core5-packed-20261027.md) |
| 7a017bf | State 批量 VJP | 343 | [报告](evidence/core5-state-vjp-20261027.md) |
| d90cfbc | Aggregate 数值/batched VJP、训练恢复 | 704 | [报告](evidence/core5-aggregate-20261027.md) |

最后资格：tide-core5-aggregate-qualify-20261027，已终态 inactive/success、exit 0；
干净只读 worktree artifacts/worktrees/core5-bootstrap，源码 d90cfbc99bc9af821d24dec11cc27cc59926b02e。
输出 artifacts/jobs/core5-aggregate-qualify；launcher `.local/core5-aggregate-qualify.sh`。
**704 passed / 0 failed / 0 skipped**。build/source/log hashes 已保存到独立证据；
build-manifest 的旧 baseline 字段不代替 job 的干净固定提交。实现与证据 58cae5a 已推送当前分支。

## 当前任务与精确下一动作

Read/Full 数值批量化尚未提交，契约 [weighted-query-batch](weighted-query-batch.md)。
Python/C++ 都已接入 joint_batch；Read 实际批量 Mem，p_v/有限描述头保留逐行运算次序，
Full 实际批量记忆混合/门/RMS/FFN。反向仍是显式语义 replay。
当前没有活动长任务。最近开发 unit `tide-core5-query-heads-dev-20261027` 已终态
failed/exit-code、exit 1，**397 passed / 1 failed**；冻结 tree e5fb6e079865e65e7c11ba1f38737cf6be0af309。
源码/日志/状态在 artifacts/jobs/core5-query-heads-dev；launcher `.local/core5-query-heads-dev.sh`。
失败是新 PDG 三样本 FP32/native case 3 的 evidence 独立根，最大差 1.33e-6。
`.local/diagnose-query-native.py` 确认 native streaming/greedy unpacked/packed 均复现同一差异；
Python GroupedEvidence 的行向量乘转置与 C++ 列向量投影次序不一致。
主工作区已将该标量头对齐教材列向量次序，未改输入、dtype、容差或数学函数；
失败用例、全部 query VJP 原型与评分公式的定向检查 **64 项通过**。

原全头矩阵批开发 unit `tide-core5-query-dev-20261027` 仍为 failed/exit 1、355/1，
源码 tree 4924c94415107900e688516d8e0f18f064e63f2b；其 FP32 evidence_weight 差异和原日志保留。
Python query VJP 原型首次 32/10 的失败源码 tree 20960d1b9477610558ab3ecec64857745c5427c0，
及各投影诊断在 artifacts/diagnostics/core5-query-vjp-projection；headrows 修正检查 42/0，
最近完整开发中的原型检查通过，但尚未接入公共路径。

未提交源码：weighted/{query_batch,read_batch,full_batch}.py，独立 C++ weighted_query_*、
weighted_describe_batch/weighted_full_batch，programs、scoring、CMake、standalone weighted 检查，
test_weighted_read_batch 和语义索引/契约。query_vjp.py/tests 与 C++ query_vjp 草稿是下一增量，
不纳入本次数值提交。全部修改为本任务所作，无用户遗留修改。

数值增量相关开发检查已完成，当前提交固定实现；下一步冻结提交运行 qualification 并单独提交证据。
随后接入 Read/Full 批量 VJP，继续区域批量 VJP、计算共享和正式参数绑定，
mixed/resident 前向/反向/optimizer/跨卡、standalone 完整训练/graph continuation，
有限 profiling 与最终追溯验收。CUDA 源码/构建/真机分别记账；无 GPU 不阻塞其余工作。

## 环境与保留事项

实际路径 /mi/data2T/zlong/home/llm/graph-execution-foundation；/home/zlong 指向当前 home。
规范 /opt/software/ascend-locks/dev-workspace-9.1.0/README.md 与对应 /etc/modulefiles。
`source /etc/profile.d/modules.sh; module purge; module load ascend/dev-workspace-9.1.0; source .local/venv-core5/bin/activate`。
Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1、aarch64；未改公共环境，磁盘约 3.4 TiB 空闲。

开发失败保持原记录；packed 的 363/5 与 State VJP 的 337/6 原因和修复在各自证据中，
没有放宽容差或用新成功覆盖旧失败。更早 bootstrap/structured/KV/recurrent/scoring 的
原失败/最小复现仍在相应 artifacts/jobs；不重新排队或默认复核历史实验。
Aggregate 开发 288/0、696/0 与额外训练 8/0 的身份/范围见新证据。
