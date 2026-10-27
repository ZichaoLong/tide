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

| 67854b3 | Read/Full 批量 VJP、训练恢复 | 647 | [报告](evidence/core5-query-vjp-20261027.md) |

## 当前任务与精确下一动作

Read/Full VJP 固定源码资格已完成：unit tide-core5-query-vjp-qualify-20261027，
inactive/success、exit 0，647 passed / 0 failed / 0 skipped。
源码 67854b306a97a1e39d6c4ae995906963507bdc7b，干净只读 worktree artifacts/worktrees/core5-bootstrap；
输出 artifacts/jobs/core5-query-vjp-qualify，build manifest 已独立保存。
launcher `.local/core5-query-vjp-qualify.sh`，命令/环境/源码与日志 hash 见上述证据。
当前没有活动任务。实现和本次证据待一起推送；远端到 1f9d964。

主工作区下一增量：allocation_batch、allocation_rows、sending_batch、region_batch、region_vjp
及三个局部测试；尚未接入公共区域调度/C++。所有未提交修改均为本任务所作。
分配数值/行 VJP 33 项、sending 17 项通过；区域最新 **46 passed / 2 failed**。
失败是 constrained/match/att/content/mu 的 mixed root：frame.0.node.0.score
标量为 None，批量为连接零。按实际 allocation 分支和 cotangent 模式隔离反向组，
不能根据梯度数值屏蔽。修正前源码/观测在 artifacts/diagnostics/core5-region-vjp-connectivity。

下一命令：使用下方环境运行
`python -m pytest -q tests/test_weighted_region_batch.py --dtype both`；
修复后继续区域 Python/C++ 实际批量、VJP、一般引擎能力门禁与在线调度接入。
之后正式参数绑定/计算共享、mixed/resident 前向/反向/optimizer/跨卡、
standalone 完整训练/graph continuation、有限 profiling 和最终追溯。完整范围见 ROADMAP。
CUDA 源码/构建/真机分开记录，无 GPU 不阻塞其他实施。

## 环境与保留事项

实际路径 /mi/data2T/zlong/home/llm/graph-execution-foundation；/home/zlong 指向当前 home。
规范 /opt/software/ascend-locks/dev-workspace-9.1.0/README.md 与对应 /etc/modulefiles。
`source /etc/profile.d/modules.sh; module purge; module load ascend/dev-workspace-9.1.0; source .local/venv-core5/bin/activate`。
Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1、aarch64；未改公共环境，磁盘约 3.4 TiB 空闲。
短测设置 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0；
TIDE_BUILD_DIR 指向 artifacts/worktrees/core5-bootstrap/build/cpu。

历史失败保持原状态及源码/log；Query 数值 355/1、397/1 与 VJP 编译失败、646/1
已在对应证据解释，不放宽容差或覆盖旧失败。allocation 最初 sort API 的 17/6
在 artifacts/diagnostics/core5-allocation-batch-api。其他开发失败不重新排队。
status 工具已同时扫描 artifacts/NAME 与 artifacts/jobs/NAME，不递归副本，21 项检查通过。
