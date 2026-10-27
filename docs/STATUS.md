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

## 当前任务与精确下一动作

Aggregate 复用实现 52dee1e5da119851e97fc302f71a52098aef270d 的固定资格已完成：
unit tide-core5-sharing-qualify-20261027，inactive/success、exit 0，543 passed / 0 failed / 0 skipped。
只读 worktree artifacts/worktrees/core5-bootstrap 干净，tree 8bcfe5d0485b7dae63e556f9e93408f345e69c39；
输出 artifacts/jobs/core5-sharing-qualify，build manifest 已独立保存；launcher .local/core5-sharing-qualify.sh。
本次证据单独提交推送，不混入后续 KV 实现。Aggregate 证据已推送到 82917c2；当前 KV 状态见下文。

KV 基座复用开发已通过：tide-core5-kv-sharing-dev-20261027，inactive/success、exit 0，
309 passed / 0 failed / 0 skipped；源码 tree ac222fb4b72840e69884a795be9009064e2055e3。
177 项编译/链接完成，Python/native 共享基座、私有状态/适配器、全部使用者 VJP、
在线三种调度、三步 AdamW/pending 恢复，以及通用 State/五种记忆回归和 standalone 手算通过。
输出 artifacts/jobs/core5-kv-sharing-dev，build manifest 已保存；launcher .local/core5-kv-sharing-dev.sh。
当前没有活动任务。正在从上述已测 tree 提交 KV 实现，不混入后续 native checkpoint 草稿。

下一步冻结该实现提交，启动 tide-core5-kv-sharing-qualify-20261027；
launcher .local/core5-kv-sharing-qualify.sh，输出 artifacts/jobs/core5-kv-sharing-qualify。
--reuse-build 校验 C++ 源码和全部二进制 hash，测试范围与开发相同；
独立 background.slice、2 build workers、单线程 ATen/BLAS、1200 秒上限。
检查 `systemctl --user show tide-core5-kv-sharing-qualify-20261027 -p ActiveState -p Result -p ExecMainStatus`、
`cat artifacts/jobs/core5-kv-sharing-qualify/status.json`、`tail -n 30 artifacts/jobs/core5-kv-sharing-qualify/task.log`。
尚未启动资格，不可标为通过。通过后单独提交证据并推送，继续下述实现。
一次新续接测试误将中间状态 objective 计入整段目标，2 项失败；改为完整末态和拼接输出后通过，
实现未因此改变。所有未提交修改均为本任务所作。

主工作区开始后续独立原生完整 checkpoint 草稿：graph_checkpoint* 与共享二进制 codec，
已接入 CMake/build manifest，增加独立 C++ 五记忆/异构 × FC/HB/非对齐/反馈的训练恢复检查，
尚未构建测试；契约 docs/native-graph-checkpoint.md。不属于 ac222fb，不能随 KV 实现提交或认证。
cpp/bench/compare.cpp 同时补齐结构化 Read 字段与整数 Tensor 严格比较。
后续 launcher 已写 .local/core5-native-training-dev.sh（未启动），须先完成 KV 实现/固定资格/证据，
再冻结新的全工作树，启动独立新 job 输出 artifacts/jobs/core5-native-training-dev。
随后继续 mixed/resident 前向/反向/optimizer/跨卡、standalone 完整训练/续接、
有限 profiling 与最终追溯，完整范围见 ROADMAP。CUDA 源码/构建/真机分别记录。

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
