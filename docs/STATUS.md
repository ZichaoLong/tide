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
build-manifest 的旧 baseline 字段不代替 job 的干净固定提交。实现与证据提交后推送当前分支。

## 当前任务与精确下一动作

主工作区未提交 Read/Full 数值批量化，契约 [weighted-query-batch](weighted-query-batch.md)。
Python weighted/{query_batch,read_batch,full_batch}.py 与独立 C++ weighted_query_*.cpp、
weighted_describe_batch.cpp、weighted_full_batch.cpp 已接入 joint_batch。
保留原始 int64 时间、完整快照/字段、多头 KV 与所有循环 Full；反向仍明确语义重放。
Python 局部/异构 PDG 数值与独立根 **66 项通过**；C++ 尚未构建。
修改还包括 CMakeLists、weighted/programs、C++ weighted_programs、standalone weighted.cpp、
语义索引/契约与 test_weighted_read_batch。所有未提交内容均本任务所作，无用户修改。

下一条：`python .local/freeze-core5.py tide-core5-aggregate-qualify-20261027`，
冻结主工作区到 task-owned artifacts/worktrees/core5-bootstrap，随后启动：
unit `tide-core5-query-dev-20261027`，launcher `.local/core5-query-dev.sh`，
输出 artifacts/jobs/core5-query-dev/{status.json,task.log,development.json,source.tar.gz}。
background.slice、Nice=10、两个 build workers、单线程 ATen/BLAS、15 分钟上限。
运行期间不改其源码/build。该任务尚未启动，结果未知。
检查 `systemctl --user show tide-core5-query-dev-20261027 -p ActiveState -p Result -p ExecMainStatus`；
日志 `tail -40 artifacts/jobs/core5-query-dev/task.log`；停止同名服务。
测试新 Read/Full、五种记忆执行、评分/发送、State VJP、Aggregate 训练与 standalone。

开发通过后提交数值增量并做固定源码资格；继续 Read/Full/区域批量 VJP、
计算共享和正式参数绑定方式、mixed/resident 前向/反向/optimizer/跨卡、
standalone 完整训练/graph continuation、有限 profiling 与最终追溯验收。
CUDA 源码/构建/真机分别记账；无 GPU 不阻塞其余工作。

## 环境与保留事项

实际路径 /mi/data2T/zlong/home/llm/graph-execution-foundation；/home/zlong 指向当前 home。
规范 /opt/software/ascend-locks/dev-workspace-9.1.0/README.md 与对应 /etc/modulefiles。
`source /etc/profile.d/modules.sh; module purge; module load ascend/dev-workspace-9.1.0; source .local/venv-core5/bin/activate`。
Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1、aarch64；未改公共环境，磁盘约 3.4 TiB 空闲。

开发失败保持原记录；packed 的 363/5 与 State VJP 的 337/6 原因和修复在各自证据中，
没有放宽容差或用新成功覆盖旧失败。更早 bootstrap/structured/KV/recurrent/scoring 的
原失败/最小复现仍在相应 artifacts/jobs；不重新排队或默认复核历史实验。
Aggregate 开发 288/0、696/0 与额外训练 8/0 的身份/范围见新证据。
