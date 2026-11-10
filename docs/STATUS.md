# 当前交接 — 联合batch/sequence与默认窗口优化实施中（2026-11-10）

分支graph-execution-foundation；数学0770593；起点fb32e22。
用户已批准上一轮完整计划并要求直接实施。允许全部320 CPU、空闲及有余量共享NPU，
以尽快获得结果为原则；数值舍入超差只作有限诊断，不无限阻塞后续。
参考与下游只读，不使用子代理。两份用户未跟踪handoff保持原样。

## 当前进度

启动git/status/upstream审计通过，无继承活动作业。正在实现FC16块内跨B/sequence准备；
随后独立实现静态分量窗口默认调度，补五种记忆、标准层链及代表性CPU/NPU回归。
已有PackedSequence保留样本/原始时间，训练State仍逐时间绑定；新增量须保留独立梯度根。
上一轮固定证据见[evidence/prefill-default-20261110.md](evidence/prefill-default-20261110.md)；
其CPU/NPU成本、严格失败和未覆盖项保持原身份，不算本轮新结果。
执行数值/资源约定见execution-flows；唯一完整待办见ROADMAP。

## 作业与资源

已通过gef-joint-prefill-block-a：固定开发快照artifacts/joint-prefill/source-block-a，
tree dd7432a054912ed898eaf10214589a1a429b1e22bd2b8c04b5e30b24d26c848f；脚本artifacts/joint-prefill/build-block-a.sh；
全320核构建、CPU相关检查，MemoryMax512G、时限3600秒。
作业记录artifacts/jobs/joint-prefill-block-a/status.json，日志同目录task.log。
这是开发快照检查：133构建步骤完成，相关CPU80项通过；服务inactive/success。
正式资格仍在实现提交后进行。正在主工作区独立开发窗口调度，未混入局部准备提交。
可复用本机CANN9.2私有Python环境artifacts/tl-req-004/installed-env；
新C++源码另构建，不将旧build-e冒充新实现。旧证据目录仅作定位，不需恢复旧服务。
当前文件系统约919GiB可用，320可用逻辑CPU；NPU有空闲和低负载共享卡，启动前再采样准入。

## 下一步

1. 完成局部联合准备及独立根/共享/零写入检查；冻结快照后在background.slice构建与测试。
2. 固定实现提交后取得CPU/NPU/standalone证据；继续调度、模块和代表回归，不在阶段提交后停工。
3. 有限舍入差异保留严格失败及幅度、首差、工作量；非有限/实现/控制/恢复错误另行处理。

接续命令：git status --short --branch；python scripts/status.py；python scripts/check_upstream.py。
本轮未提交修改包含已批准对齐记录及正在实施的源码/契约，实际清单以git status为准。
