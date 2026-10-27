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

## 当前任务与下一动作

熵分配不可变资格 `tide-core5-entropy-qualify-20261027` 已终态 inactive/success，退出 0。
记录 `artifacts/jobs/core5-entropy-qualify/{status.json,task.log,development.json}`；
launcher `.local/core5-entropy-qualify.sh`；冻结源码 a05d3ff。
worktree `artifacts/worktrees/core5-bootstrap` 干净只读；无 live job，可在确认终态后复用。
开发快照与原日志保留在 entropy-dev；开发和本次资格各 226 项，不合并计数。

下一增量已开始：`python/tidegraph/weighted/sending.py` 与
`cpp/include/tide/weighted_sending.h`、`cpp/src/weighted_sending.cpp` 是未提交的新发送原语。
实现 Attention 正支持/零基权重右导数、比例、固定排序零温填入与 hard 选择；
尚未接 CMake/配置、尚未测试，不在熵分配资格内。无用户修改。
接下来补独立公式/方向导数、全部发送配置与节点增益绑定、native 完整记录/训练/恢复。
随后 match/att/grouped/content key、联合/packed/批量 VJP、mixed/resident、
standalone 完整训练与有限 profiling。阶段成果不代表总体完成，持续推进。

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
