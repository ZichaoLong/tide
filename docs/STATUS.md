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

## 当前任务与精确下一动作

KV 基座投影复用实现 6dc2665c4fbd9053173bbd9d26c7e1b03b62abae 已完成固定资格：
unit tide-core5-kv-sharing-qualify-20261027，inactive/success、exit 0，309 passed / 0 failed / 0 skipped。
worktree artifacts/worktrees/core5-bootstrap 干净，tree 8a2f2705a9021a7ee2bd05d41ea0c74b5b6d7c08；
输出 artifacts/jobs/core5-kv-sharing-qualify，build manifest 已独立保存；launcher .local/core5-kv-sharing-qualify.sh。
证据 ffe6d8d 已推送；主工作区 native checkpoint 草稿不属于本资格。

原生 checkpoint/standalone 训练开发 tree 4c70968719d7235a2459a72baab94a044fb60ebd 已通过：
unit tide-core5-native-training-dev-20261027，inactive/success、exit 0；43 项编译/链接、20 passed / 0 failed / 0 skipped。
输出 artifacts/jobs/core5-native-training-dev，build manifest 已独立保存；launcher .local/core5-native-training-dev.sh。
原生进程每 dtype 覆盖五记忆/异构 × FC/HB/非对齐/反馈共 24 案例，三步 AdamW、保留图切点、
含 pending checkpoint 换 owner 以及恢复后的第四次更新；真实原始 int64/非空 History/别名与加载拒绝通过。
正从已测 tree 提交该增量；主工作区后续 weighted placement 草稿尚未编译，不混入本提交或资格。

下一步冻结该实现提交，启动 tide-core5-native-training-qualify-20261027，
launcher .local/core5-native-training-qualify.sh、输出 artifacts/jobs/core5-native-training-qualify，--reuse-build、
background.slice、单线程 ATen/BLAS、1200 秒上限。
检查 `systemctl --user show tide-core5-native-training-qualify-20261027 -p ActiveState -p Result -p ExecMainStatus`、
`cat artifacts/jobs/core5-native-training-qualify/status.json`、`tail -n 30 artifacts/jobs/core5-native-training-qualify/task.log`。
尚未启动固定资格，不可标为通过。

通过后提交该实现，再冻结实现提交进行同范围资格，单独提交证据并推送；
继续 mixed/resident 前向/反向/optimizer/跨卡、有限 profiling 与最终追溯。
CUDA 源码/构建/真机分别记录；总体剩余见 ROADMAP。

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
