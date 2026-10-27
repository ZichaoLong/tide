# 当前交接 — 2026-10-27

固定上游 `0770593838a37c790a568226930567ed60f99b83`，授权分支
`graph-execution-foundation`。持续实施、私有环境、有限实验、提交及推送均已授权；不使用子代理。
历史等待授权/提交后暂停要求已被取代。完整未完成范围以 [ROADMAP](ROADMAP.md) 为准。

## 已完成的阶段

- 一般 SettleGraph `efb9d629438fb77582842bb133158be7f9f70afb`，固定源码 CPU 247 项通过；
  [证据](evidence/core5-settle-20261027.md)，实现与证据 d0acd3c 已推送。
- 具名 Read、int64 状态、typed packing、SrcClass `714d5f963b676d95a0ae763171a7d93806f183d3`，
  固定源码 CPU 212 项通过；[证据](evidence/core5-structured-20261027.md)，实现与证据 cbdd3fd 已推送。
- 带权 KV 首组合 `6fb467c25755a2b3f8938952f1b2ec1011cbcd11`，Python/native、FC/HB 正式绑定，
  独立公式、完整记录/根、多步优化器和恢复，固定源码 CPU FP64/FP32 **129 项通过**。
  [证据](evidence/core5-weighted-kv-20261027.md)单独提交；首组合不代表完整任务已完成。

## 当前任务和下一动作

KV 不可变资格已终止成功，129 项通过，证据提交 572d4ff 和实现 6fb467c 均已推送。
原始记录 `artifacts/jobs/core5-weighted-kv-qualify`，固定源码 checkout 原为 6fb467c。

当前增量已编写 affine/delta、固定逻辑时间短卷积、完整 GDN/SSD 的 Python/独立 C++、
共同描述、异构绑定与 native 适配；契约 [weighted-recurrent](weighted-recurrent.md)。
开发快照 `ea36b7521e8547e2f94666bb466abeeb47a27316` 的
`tide-core5-recurrent-build-20261027` 已构建完成，测试 **212 passed / 1 failed**，终态 exit-code/1。
失败是 SSD FP32 Read0 复合标量目标梯度：前向逐位相同，分组 output/state/pending VJP 逐位相同，
比较器未固定字典字段归约顺序，目标相差 1.9e-6，平方目标的相消梯度差 7.15e-6。
主工作区已修复比较目标的字段/owner排序，未改模型公式/容差；排列不变性和原失败短测 3 项通过。
原失败源码、日志/清单位于 `artifacts/jobs/core5-recurrent-build`，保持失败。
首轮 core5-recurrent-dev 的目录预创建 launcher 失败亦保留，未开始构建/测试。

修复后的冻结开发树 `2fa1fb1793753a17a4c4f7d98f07678d11c5774c` 已完成
`tide-core5-recurrent-fix-20261027`：**536 passed，退出 0，inactive/success**。
原始源码/日志/清单在 `artifacts/jobs/core5-recurrent-fix`。
包含全部 weighted 测试、比较目标排列不变性、structured_records、semantics、
isolated_gradients 和 CLI standalone C++ kernel。另补非对齐 TimedDAG 四族/no_grad 8 项短测通过。
本增量合并范围为 544 项，尚未对该合并源码执行不可变资格。

当前无 live job。此提交固定 affine/delta、GDN/SSD、固定时间短卷积与异构绑定，
并修复验证目标的字段归约顺序；没有其他用户修改。
下一动作：将 `artifacts/worktrees/core5-bootstrap` 更新为本实现提交的干净只读 checkout，
运行 `.local/core5-recurrent-qualify.sh` 调用 job.py/develop.py --reuse-build 的相同定向范围，
服务 `tide-core5-recurrent-qualify-20261027`、background.slice、Nice=10、15 分钟上限。
输出 `artifacts/jobs/core5-recurrent-qualify/{status.json,task.log,development.json}`。
先确认实际终态/退出码/计数和源码干净，再将证据单独提交并推送，继续后续阶段。
查看 `systemctl --user show tide-core5-recurrent-qualify-20261027 -p ActiveState -p Result -p ExecMainStatus`；
停止 `systemctl --user stop tide-core5-recurrent-qualify-20261027`。

仍待完整正式分配/评分/证据/key、联合/packed/批量 VJP、mixed/resident、standalone 完整训练和有限选型。

## 环境与保留事项

实际路径 `/mi/data2T/zlong/home/llm/graph-execution-foundation`；/home/zlong 指向当前 home。
现行规范 `/opt/software/ascend-locks/dev-workspace-9.1.0/README.md` 与对应 /etc/modulefiles。
`source /etc/profile.d/modules.sh; module purge; module load ascend/dev-workspace-9.1.0; source .local/venv-core5/bin/activate`。
Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1、aarch64。未修改公共环境，无用户未保存修改。
ObsidianVault/lh/fractal-latcarf 只读。CUDA 源码/构建/真机分别记录，GPU 真机待验不阻塞其余工作。

旧 bootstrap 缺 Ninja、缺 TIDE_BUILD_DIR 的 24 项前置失败、structured-dev 的 210/2、
weighted-kv-dev 的 C++ Options 构建失败均保留原记录；后续成功不改写旧失败。
