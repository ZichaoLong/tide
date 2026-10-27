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
| 5391017 | 区域数值/batched VJP、稳定均值反向 | 536 | [报告](evidence/core5-region-20261027.md) |

## 当前任务与精确下一动作

区域数值/VJP 实现 53910174d036e0809be7893527ca1b1c204a460b 的固定源码资格已完成：
unit tide-core5-region-qualify-20261027，inactive/success、exit 0，536 passed / 0 failed / 0 skipped。
干净只读 worktree artifacts/worktrees/core5-bootstrap，tree bfabbb377172fa1c288827923c180717bcae535c；
输出 artifacts/jobs/core5-region-qualify，build manifest 已独立保存。
launcher `.local/core5-region-qualify.sh`，完整命令/环境/源码与日志 hash 见证据。
本次区域实现/证据待一起推送；此前远端到 8fc59ae。当前没有活动任务。

下一增量已经在主工作区起步：SourceBinding 与四种来源参数绑定（region/node/source/adapted），
低秩 A/B 因子为 (node,class) 私有，参数共享不合并物理原子或状态。
草稿见 python/tidegraph/weighted/source_binding.py、binding.py、programs.py、aggregate_batch.py、
__init__.py 及 cpp/src/weighted_source.h、weighted_programs.cpp、weighted_aggregate.cpp。
默认 region 绑定保持原随机初始化次序及 profile hash；23 项旧公式/Aggregate VJP 回归通过。
新增绑定尚无自身完整测试，C++ 新因子路径尚未构建；不属于 5391017，也未被本次资格读取。
所有未提交修改均为本任务所作。

下一步补四种绑定的独立公式、参数别名、因子方向导数、native、优化器/含 pending checkpoint；
再正式固定资格。按教材共享矩阵的宽度相容条件完善配置与拒绝边界。
随后精确计算共享、mixed/resident 前向/反向/optimizer/跨卡、standalone 完整训练/graph continuation、
有限 profiling 和最终追溯。总体尚未完成，完整范围见 ROADMAP。
CUDA 源码/构建/真机分开记录，无 GPU 不阻塞其他实施。

## 环境与保留事项

实际路径 /mi/data2T/zlong/home/llm/graph-execution-foundation；/home/zlong 指向当前 home。
规范 /opt/software/ascend-locks/dev-workspace-9.1.0/README.md 与对应 /etc/modulefiles。
`source /etc/profile.d/modules.sh; module purge; module load ascend/dev-workspace-9.1.0; source .local/venv-core5/bin/activate`。
Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1、aarch64；未改公共环境，磁盘约 3.4 TiB 空闲。
短测设置 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0；
TIDE_BUILD_DIR 指向 artifacts/worktrees/core5-bootstrap/build/cpu。

region4-dev 的 480/2 原失败与诊断在 artifacts/diagnostics/core5-region-pdg-rounding；
独立分支根 46/2 在 artifacts/diagnostics/core5-region-vjp-connectivity。region5-dev 536/0
不覆盖旧失败。前三次启动/编译失败仍留在各 job 状态；证据说明原因。
更早数值/VJP 的失败均留在相应证据/原始记录，不重新排队。status 工具已同时扫描
artifacts/NAME 与 artifacts/jobs/NAME，不递归 worktree/archive 副本。
