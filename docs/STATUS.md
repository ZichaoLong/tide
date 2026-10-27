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
| 503f609 | 来源绑定/类别宽度、稳定 Attention/RMS VJP | 665 | [报告](evidence/core5-source-20261027.md) |
| 52dee1e | Aggregate 精确计算复用 | 543 | [报告](evidence/core5-sharing-20261027.md) |
| 6dc2665 | KV 基座投影复用 | 309 | [报告](evidence/core5-kv-sharing-20261027.md) |
| a7106fc | standalone 训练/完整数值 checkpoint | 20 | [报告](evidence/core5-native-training-20261027.md) |

## 当前任务与精确下一动作

完整原生 checkpoint/standalone 训练 a7106fc 的固定资格已完成：
unit tide-core5-native-training-qualify-20261027，inactive/success、exit 0；20 passed / 0 failed / 0 skipped。
干净 worktree artifacts/worktrees/core5-bootstrap，tree 6efba49f16535366a5975228d041019c913f737f；
输出 artifacts/jobs/core5-native-training-qualify，build manifest 已独立保存；launcher .local/core5-native-training-qualify.sh。
证据将单独提交推送。当前没有活动任务。

主工作区正在实现 weighted placement，尚未构建：结构化 Read 的临时记忆/参数/状态转移，
区域 control/query 与 selection/分配分别放置，保留原参数身份与精确 raw int64；
显式 scoring dtype、原始 VJP 版本检查与发送正支持的转换拒绝。
Python/C++ 实现与 tests/test_weighted_placement.py 已写，仍需完善契约、构建并修正检查。
短定向公式测试曾误把固定非 grad 参数列为梯度根，1 项失败；已修正测试根清单，待重测。
这些草稿不属于 a7106fc 或本固定资格。

下一步完成 placement 实现/契约与定向检查，冻结新的开发 tree，
启动 tide-core5-weighted-placement-dev-20261027，launcher .local/core5-weighted-placement-dev.sh，
输出 artifacts/jobs/core5-weighted-placement-dev；background.slice、2 build workers、单线程 ATen/BLAS、1200 秒上限。
通过后提交实现，再固定资格并独立记录证据。继续真实 mixed/resident 前向/反向/optimizer/跨卡、
有限 profiling 与最终追溯；CUDA 源码/构建/真机分别记录。总体剩余见 ROADMAP。

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

来源两轮真实失败与旧5391017基线复现保留在来源证据及原始诊断；不因665项新通过覆盖旧失败。
