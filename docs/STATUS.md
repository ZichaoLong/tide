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
  证据单独提交后与实现一起推送。首组合和循环阶段均不代表完整总体任务完成。

## 当前任务与下一动作

循环不可变资格 `tide-core5-recurrent-qualify-20261027` 终态 inactive/success、退出 0。
源码 442e8b2；`artifacts/worktrees/core5-bootstrap` 干净、只读。
记录 `artifacts/jobs/core5-recurrent-qualify/{status.json,task.log,development.json}`。
launcher `.local/core5-recurrent-qualify.sh`；background.slice、Nice=10、单线程、15 分钟上限。
当前无 live job，确认终态后可复用该 task-owned worktree/build。

下一增量正在主工作区：Bernoulli、Shannon 和固定 alpha ent 正则的稳定响应/精确映射 VJP。
未提交 `weighted/regularizers.py`、`weighted/entropy_allocation.py`、
`tests/test_weighted_entropy_allocation.py`：35 项 Python 定向检查通过，含独立 80 位 Decimal 求解、
VJP/方向差分、边界/温度和正支持下溢拒绝。C++ weighted_entropy.cpp/weighted_regularizers.h 与
公共头声明已编写，尚未构建；尚未接入 RegionalConfig、native 和公共导出。
这些文件不属于 442e8b2 或其资格。没有用户修改。

接下来完成正则公共/native/区域集成及独立 C++ 检查，再实现其余正式发送函数、match/att、
grouped 证据/content key、参数共享。随后继续联合/packed/批量 VJP、mixed/resident、
standalone 完整训练与有限 profiling。不要把明确拒绝未完成配置当作完成。
短测命令（环境下）：`python -m pytest -q tests/test_weighted_entropy_allocation.py --dtype both`。
长构建/资格继续固定输入、独立 background.slice 服务和有界资源；先更新本交接。

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
