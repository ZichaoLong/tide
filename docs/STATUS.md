# 当前交接 — 2026-10-27

固定上游 `0770593838a37c790a568226930567ed60f99b83`，授权分支 graph-execution-foundation。
持续实施、私有环境、有限实验、提交和推送均已授权；不使用子代理。参考仓库只读。
历史等待授权/提交后暂停已被取代。完整未完成范围以 [ROADMAP](ROADMAP.md) 为准。

## 阶段结果

- 一般 SettleGraph `efb9d62`：247 项固定源码 CPU 定向资格，证据 d0acd3c，已推送。
- 结构化 Read/int64/SrcClass `714d5f9`：212 项固定源码 CPU 定向资格，证据 cbdd3fd，已推送。
- KV 首组合 `6fb467c`：129 项固定源码 CPU 定向资格，证据 572d4ff，均已推送。
- affine/delta、完整 GDN/SSD、固定时间短卷积与异构绑定
  `442e8b2701164eb7e4241da56c8a395a832ebeb0`：**544 passed / 0 failed / 0 skipped**，
  CPU FP64/FP32；[契约](weighted-recurrent.md)、[证据](evidence/core5-recurrent-20261027.md)。
  证据 955912c 与实现已一起推送。首组合和循环阶段均不代表完整总体任务完成。

- Bernoulli/Shannon/固定 alpha ent 分配 `a05d3ffc946985cfeb124a5e3b3074ceeece306d`：
  **226 passed / 0 failed / 0 skipped**，CPU FP64/FP32；[证据](evidence/core5-entropy-20261027.md)。

- 十种发送/节点增益 `4a0858880831cb46ad96191ab9dd58fca5260223`：**320 passed / 0 failed / 0 skipped**，
  CPU FP64/FP32；[证据](evidence/core5-sending-20261027.md)。实现与本次证据待一起推送。

## 当前任务与下一动作

发送不可变资格 `tide-core5-sending-qualify-20261027` 已终态 inactive/success，退出 0。
冻结 worktree `artifacts/worktrees/core5-bootstrap` 干净只读、源码 4a08588；目前无 live job。
记录 `artifacts/jobs/core5-sending-qualify/{status.json,task.log,development.json}`；launcher `.local/core5-sending-qualify.sh`。
原开发与端点快照日志/源码保留在 sending-dev / sending-corners，计数分别 304 / 70，不合并。

下一增量未提交：match、att、grouped evidence、content_stats/content key，及其适用配置。
Python scoring.py 和 C++ weighted_scoring.{h,cpp} 已编写并接描述/区域/native；
新三/四/五/六字段按函数布局，评分/证据/key 默认参数独立，现无计算共享优化声明。
Python 新手算、独立标量 Encoder/全部根、key 直接依赖、query 权重及 KV/熵回归 **71 项通过**。
C++ 新增评分尚未构建，整图/共享/训练/续接检查待补；不属于发送 4a08588 资格。
无用户修改。接下来完成本增量测试、冻结开发构建，再固定实现与独立资格。
随后联合/packed/批量 VJP、mixed/resident、standalone 完整训练与有限 profiling。
总体任务尚未完成，持续推进。

## 环境与保留事项

实际路径 /mi/data2T/zlong/home/llm/graph-execution-foundation；/home/zlong 指向当前 home。
规范 /opt/software/ascend-locks/dev-workspace-9.1.0/README.md 与对应 /etc/modulefiles。
`source /etc/profile.d/modules.sh; module purge; module load ascend/dev-workspace-9.1.0; source .local/venv-core5/bin/activate`。
Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1、aarch64。未修改公共环境。
CUDA 源码/构建/真机分别记录；无 GPU 不阻塞其他工作。

旧 bootstrap 缺 Ninja、缺 TIDE_BUILD_DIR 的 24 项前置失败、structured-dev 的 210/2、
weighted-kv-dev 的 C++ Options 构建失败，以及 recurrent-dev launcher 失败保持原记录。
recurrent-build 的 212/1 是比较目标字段归约顺序问题，修复后新开发和不可变资格分别通过；
原因与修复详见循环证据。原失败日志/源码留在相应 artifacts/jobs 目录。
