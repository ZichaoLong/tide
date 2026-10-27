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
Read/Full 实现与证据 8fc59ae 已推送。本次区域实现即将提交，随后运行固定源码资格。

主工作区区域数值/VJP 草稿已接入公共能力接口与 Python/C++ 在线调度。
Python 区域分支隔离修正后 48 项通过；新增能力/版本检查与分配短测 82 项通过。
原始 46/2 的 None/零失败记录保持；中间 branch 字段拼写错误 38/10 已改为 constraint。
C++ 实际批量分配、发送、区域和独立行根 VJP 已构建验证；公共范围见 weighted-region-batch。
所有未提交修改均为本任务所作。

区域开发 unit tide-core5-region4-dev-20261027 已终态 failed/exit 1，tree
5d0e007b295767384c6c87e5795d7915194173da，**480 passed / 2 failed**。
C++ 构建及 standalone literal 检查通过；两项都是新增 FP32 Python/native PDG case 4 的
完整 objective 质量根 input.1，差 7.63e-6。源码/log 位于 artifacts/jobs/core5-region4-dev。
诊断 artifacts/diagnostics/core5-region-pdg-rounding：关闭区域合批仍失败，已隔离 SSD Full
微小前向舍入在 Aggregate 商梯度大项相消中放大。未采用降低合批范围或改投影的尝试。
当前主工作区已将 weighted mean 的一阶质量导数改用等价的 centered quotient VJP，
前向、输入、dtype、容差不变；单原子采用数学上的质量不变性。修正后的独立公式、Aggregate VJP 与 Python 在线/训练短测 39 项通过。

最近开发 unit `tide-core5-region5-dev-20261027` 已完成 **536 passed / 0 failed / 0 skipped**，launcher `.local/core5-region5-dev.sh`。
固定只读 worktree artifacts/worktrees/core5-bootstrap，tree 080ec009e52e469f08e9c7cecc1568c90668f333；输出
artifacts/jobs/core5-region5-dev；两个 build workers、ATen/BLAS 单线程、Nice=10、
background.slice、RuntimeMaxSec=1200。已核验 inactive/success、exit 0；这是开发快照，不冒充固定提交资格。
检查 `systemctl --user show tide-core5-region5-dev-20261027 -p ActiveState -p Result -p ExecMainStatus`；
日志 `tail -60 artifacts/jobs/core5-region5-dev/task.log`；停止同名服务。
本次提交固定区域实现和稳定 mean VJP；下一步用该提交冻结并运行同范围资格。
计划 unit tide-core5-region-qualify-20261027，launcher .local/core5-region-qualify.sh，
输出 artifacts/jobs/core5-region-qualify；develop.py --reuse-build 校验源码/二进制 hash。
资格尚未启动；通过后证据单独提交并推送。随后四种来源参数绑定与共享计算。

区域前三次启动/构建失败分别保留：region-dev 输出目录已存在而拒绝覆盖，构建未开始；
region2-dev tree 40637a5 缺 grad-mode include；region3-dev tree 63020f9 argsort bool/int64
重载歧义。两项编译接口问题已修正，pytest 当时未启动；不重标旧失败。
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
