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
| 7b8b469 | Read/Full 数值批量、完整快照 | 377 | [报告](evidence/core5-query-20261027.md) |

最后资格：tide-core5-query-qualify-20261027，已终态 inactive/success、exit 0；
干净只读 worktree artifacts/worktrees/core5-bootstrap，源码 7b8b4696a5998ea7caa4fa55dfa9c81945f1ab17。
输出 artifacts/jobs/core5-query-qualify；launcher `.local/core5-query-qualify.sh`。
**377 passed / 0 failed / 0 skipped**。build/source/log hashes 与命令已单独记录；
build manifest 已另存到该 job 目录，后续重建不覆盖旧资格身份。

## 当前任务与精确下一动作

Read/Full 数值增量及固定源码资格已完成，证据 [core5-query](evidence/core5-query-20261027.md)。
当前没有活动长任务。主工作区正在接入 Read/Full 实际批量 VJP：Python/C++ 私有参数叶、
逐行/逐字段根、形状与零分支分组、原始版本与一阶限制；Read 自动采用新能力，
Full 显式 `full_autograd="batched"`，默认 replay 保留。
Python 局部/公共路径 108 项通过；共享参数/重复输入与方向差分新增 10 项通过；
HB/反馈 PDG 三步 AdamW、含 pending checkpoint 换 owner 恢复和保留图切点新增 6 项通过。
C++ 尚未构建；新 standalone 手算 VJP 与 native probe 待运行，不能称为通过。

最近数值开发 unit `tide-core5-query-heads-dev-20261027` 保持 failed/exit 1、**397/1**，
tree e5fb6e079865e65e7c11ba1f38737cf6be0af309。新增三样本 PDG FP32/native case 3 的 evidence
独立根差 1.33e-6；native streaming/greedy unpacked/packed 都复现，诊断见
artifacts/diagnostics/core5-query-native-projection。Python GroupedEvidence 标量行向量投影
现与教材/native 列向量顺序一致；未改输入、dtype、容差或数学函数。
失败用例、query VJP 原型与评分公式的定向检查 **64 项通过**。

原全头矩阵批开发 unit `tide-core5-query-dev-20261027` 仍为 failed/exit 1、355/1，
源码 tree 4924c94415107900e688516d8e0f18f064e63f2b；其 FP32 evidence_weight 差异和原日志保留。
Python query VJP 原型首次 32/10 的失败源码 tree 20960d1b9477610558ab3ecec64857745c5427c0，
及各投影诊断在 artifacts/diagnostics/core5-query-vjp-projection；headrows 修正检查 42/0，
最近完整开发中的原型检查通过，但尚未接入公共路径。

未提交包含 query_vjp Python/C++、公共 Read/Full capability 接口与 policy、
诊断 bindings、standalone query VJP、query 局部/训练测试、数值测试的能力统计更新。
所有修改均为本任务所作。下一步完善契约并冻结草稿，启动有界开发构建/测试。
通过后提交实现，固定源码再资格并单独提交证据；继续区域批量 VJP、正式参数绑定/计算共享，
mixed/resident 前向/反向/optimizer/跨卡、standalone 完整训练/graph continuation，
有限 profiling 与最终追溯。CUDA 源码/构建/真机分别记账，无 GPU 不阻塞其他工作。

## 环境与保留事项

实际路径 /mi/data2T/zlong/home/llm/graph-execution-foundation；/home/zlong 指向当前 home。
规范 /opt/software/ascend-locks/dev-workspace-9.1.0/README.md 与对应 /etc/modulefiles。
`source /etc/profile.d/modules.sh; module purge; module load ascend/dev-workspace-9.1.0; source .local/venv-core5/bin/activate`。
Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1、aarch64；未改公共环境，磁盘约 3.4 TiB 空闲。

开发失败保持原记录；packed 的 363/5 与 State VJP 的 337/6 原因和修复在各自证据中，
没有放宽容差或用新成功覆盖旧失败。更早 bootstrap/structured/KV/recurrent/scoring 的
原失败/最小复现仍在相应 artifacts/jobs；不重新排队或默认复核历史实验。
Aggregate 开发 288/0、696/0 与额外训练 8/0 的身份/范围见新证据。
