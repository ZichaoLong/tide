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

## 当前任务与下一动作

熵分配开发 `tide-core5-entropy-dev-20261027` 已终止 inactive/success，退出 0，
冻结开发树 `42fdcc30af20d7a101e4fd423c7912bc70ef2c0d`，**226 passed / 0 failed / 0 skipped**。
记录 `artifacts/jobs/core5-entropy-dev/{status.json,task.log,development.json,source.tar.gz}`。
Bernoulli、Shannon 与固定 alpha ent 的响应、精确映射一阶 VJP 已接公共/native RegionalTotal；
默认 quad 图身份保持。独立 Decimal 求解、隐式 Jacobian、支持下溢拒绝、完整训练/恢复检查通过。
本次固定实现后即启动不可变 CPU 定向资格，结果另提交；不把开发快照当完整总体资格。

计划 unit `tide-core5-entropy-qualify-20261027`，launcher `.local/core5-entropy-qualify.sh`；
实现源码为本次提交，task-owned worktree `artifacts/worktrees/core5-bootstrap`，build/cpu。
确认上一服务终态后对齐该提交、核验干净并置源码只读，保留已归档开发源码。
输出 `artifacts/jobs/core5-entropy-qualify/{status.json,task.log,development.json}`。
background.slice、Nice=10、ATen/BLAS 单线程、两个构建线程、15 分钟上限；
develop.py --reuse-build 核验二进制/源码哈希，测试集合沿 entropy-dev 八个入口。
查看 `systemctl --user show tide-core5-entropy-qualify-20261027 -p ActiveState -p Result -p ExecMainStatus`；
日志 `tail -60 artifacts/jobs/core5-entropy-qualify/task.log`；
停止 `systemctl --user stop tide-core5-entropy-qualify-20261027`。
运行期间不得修改该 worktree/build，尚无不可变资格通过结论。

随后继续正式发送函数（Attention 零基权重右导数、零温/硬排序及 k/K 分离）、
match/att/grouped/content key；再推进联合/packed/批量 VJP、mixed/resident、
standalone 完整训练与有限 profiling。首组合与循环/熵增量不代表总体任务完成。
当前增量未包含其他用户修改；实现/契约/测试将在本次一并提交。

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
