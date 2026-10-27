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

## 当前任务与精确下一动作

来源实现 503f6092d35dc57f2052c598e15073fde65e834b 的固定资格已完成：
unit tide-core5-source-qualify-20261027，inactive/success、exit 0，665 passed / 0 failed / 0 skipped。
只读 worktree artifacts/worktrees/core5-bootstrap 干净，tree 3146f4c20c11d5c57edc6b213b956bdd83e91207；
输出 artifacts/jobs/core5-source-qualify，build manifest 已独立保存。
完整命令与日志 hash 见证据，launcher .local/core5-source-qualify.sh。来源资格没有活动进程。
本次证据单独提交并推送；不混入主工作区的下一增量。

Aggregate 计算复用开发已完成：tide-core5-sharing-dev-20261027，inactive/success、exit 0，
543 passed / 0 failed / 0 skipped；源码 tree ae014cf867a308a1aa87c5d9527f4a62c9bdcd91。
输出 artifacts/jobs/core5-sharing-dev，build manifest 已保存；launcher .local/core5-sharing-dev.sh。
包含 Python/native 相同输入/参数复用、物理来源、独立根、AdamW/pending 恢复与 standalone，
以及相关来源、Aggregate、图和调度回归。准备从此已测试 tree 提交实现。

下一步以该实现提交显式冻结，启动 tide-core5-sharing-qualify-20261027；
launcher .local/core5-sharing-qualify.sh，输出 artifacts/jobs/core5-sharing-qualify。
采用 --reuse-build 校验 C++ 源码与全部二进制 hash，范围与开发相同；
独立 background.slice、2 build workers、单线程 ATen/BLAS、1200 秒上限。
检查 `systemctl --user show tide-core5-sharing-qualify-20261027 -p ActiveState -p Result -p ExecMainStatus`、
`cat artifacts/jobs/core5-sharing-qualify/status.json` 与 `tail -n 30 artifacts/jobs/core5-sharing-qualify/task.log`。
当前尚未启动此资格，不可标为通过；通过后另提交证据并推送。

主工作区另有下一增量：KV 基座投影复用，经显式 State preparation 能力接入，
保留节点适配器、summary、私有状态及所有使用者贡献；不属于上述 Aggregate tree。
新 state_preparation/weighted.kv_projection 与 packed/State VJP 已接入 Python/C++；
6 项 Python 手算/完整根/保留图检查通过，C++ 尚未构建；训练/恢复后续开发门禁补验。
一次新续接测试误将中间状态 objective 计入整段目标，2 项失败；改为完整末态和拼接输出后通过，
实现未因此改变。所有未提交修改均为本任务所作。
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
