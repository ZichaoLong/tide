# 当前交接 — 2026-10-27

本轮采用固定上游 `0770593838a37c790a568226930567ed60f99b83` 的 tide-core-5 与
带权消息局部定义。用户授权完整阶段目标、私有环境、有限实验、分阶段提交并推送；
不使用子代理。历史等待授权/提交后暂停安排已被本次任务取代。

## 当前状态

- 实现基线 `35d63588f3a967e4b65da9c55be05a47f4d964f8`，授权分支
  `graph-execution-foundation`；启动工作区干净，上游固定提交与工作区已核实且只读。
- 实际路径 `/mi/data2T/zlong/home/llm/graph-execution-foundation`；`/home/zlong` 指向
  `/mi/data2T/zlong/home`。旧 `/var/tmp` 路径不是启动依赖。
- 旧 core-3 快照离线校验通过；core-5 采用、一般 SettleGraph 与公共表示扩展正在实施。
  历史证据仅认证其各自源码，不认证本轮功能。
- 本机旧个人 Ascend 指南已不存在；当前来源为
  `/opt/software/ascend-locks/dev-workspace-9.1.0/README.md` 与
  `/etc/modulefiles/ascend/dev-workspace-9.1.0`。使用该模块及项目
  `.local/venv-core5`，不修改公共工具链。CPU 算子探针与 11 项最小语义检查通过（Torch 2.12.0+cpu，Python 3.12.13）。
- 基线长任务 `tide-core5-bootstrap-tools-20261027` 已通过，93 项 CPU FP64/FP32 检查；
  原生全部构建目标成功。来源 `35d6358`，原始日志/清单保留于
  `artifacts/jobs/core5-bootstrap-tools`。首轮 `core5-bootstrap` 缺 Ninja 失败保持失败。
- 一般端口开发任务 `tide-core5-settle-dev-20261027` 已通过：247 项 CPU FP64/FP32，
  包含 Python、Python/native 和独立 C++ 公式/梯度/续接/共享参数检查。
  固定开发树 `4ded7416941fc279b04fad43242b544ee3e6fc51`；详细源码归档、构建身份和
  终态见 `artifacts/jobs/core5-settle-dev/{status.json,development.json,task.log}`。
- 实现提交 `efb9d629438fb77582842bb133158be7f9f70afb` 的不可变资格已通过：247 项，
  `tide-core5-settle-qualify-20261027` 终态成功、退出 0；证据
  [core5-settle-20261027](evidence/core5-settle-20261027.md)。当前无活动长任务。
- 当前未提交内容：本交接和随后 core-5 采用/表示实现。没有用户未保存修改。

## 下一动作

1. 单独提交上述资格证据并推送授权分支。主工作区保留下一增量：具名 Read 字段、
   int64 状态槽、独立根字段打包和 SrcClass；16 项 Python 定向检查已通过，C++ 尚未编译。
2. 完成 `docs/structured-records.md` 与对应表，运行旧 Read/State/Checkpoint 相关回归；
   冻结当前开发树，用新 `core5-structured-dev` job 在已停止的临时 worktree 增量构建。
   长任务仍用固定源码、两个编译线程、ATen/BLAS 单线程与有限时长，启动前更新本交接。
3. 通过后提交表示增量并资格；继续带权 Agg/KV/Describe/分配/Full 的首个完整组合，
   不暂停等待授权。完整候选、设备、性能与剩余边界见 ROADMAP。

环境：`module purge; module load ascend/dev-workspace-9.1.0; source .local/venv-core5/bin/activate`。
没有数学歧义或用户未保存修改。参考仓库保持只读。GPU 真机独立待验。
