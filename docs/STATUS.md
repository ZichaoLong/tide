# 当前交接 — 加法基线与默认接口接入（2026-11-10）

用户已授权本轮修订计划；完成本轮后汇报并暂停对齐。分支`graph-execution-foundation`，
起点`a61e028`；数学固定`0770593838a37c790a568226930567ed60f99b83`，启动核验通过。
上一轮资格/成本见[interface-heads](evidence/interface-heads-performance-20261110.md)，不重新启动旧队列。

## 当前实施

已更新仓库三项skills：有限加法/减法方法、设计时的紧凑数学/公共批接口/小组合对照，
以及采用/拒绝/延期条件。账户skills不修改。
已扩充FC16、KV/SSD层链的训练/非零左态/跨窗与投影checkpoint基线。Python/C++
默认packed Full增加真实共享bank的B/H/Q/K矩阵计算与独立快照VJP；通用后备保持。
已修复resident局部首次借用计费，包含RowGraph/Describe/Recurrent/Full/相关VJP。
定向Python13通过/1既有xfail，C++新矩阵单元语法检查通过；尚未正式构建/设备验证。
较广开发检查56通过/2xfail/1 Read-sum FP32严格失败；上一固定554f2f5已复现相同
1.535518e-6梯度超差，FP64通过。本增量不进入该Read路径，原阈值与失败保留。
相关测试后分阶段提交；固定实现资格与证据分别提交。本轮不自动扩展成无限优化。

## 资源与现场

没有活动计算或队列。可用320 CPU；内存约1.7TiB，磁盘可用681GiB，账户284GiB/300GiB软配额
（硬配额800GiB），新构建/日志仍须有界。使用全部可用CPU和空闲NPU；共享卡未获本轮授权。
长作业按execution-flows放固定源码background.slice；构建与正式计时不互扰。
用户未跟踪文档`docs/tide-learning-lab-TL-REQ-003-handoff.md`与`...004...`原样保留，不提交。
当前未提交：仓库skills/STATUS的本轮修订；后续以`git status`为准。

## 下一步

核对紧凑基线与默认KV/SSD计算的差距，先实现可独立验证的批量边界和状态续接，
再固定源码运行相关CPU/NPU/SDK及resident见证；记录原严格超差与实际计时职责。
续接命令：`git status --short --branch`、`python scripts/status.py`、`python scripts/check_upstream.py`。
待办及限制由[ROADMAP](ROADMAP.md)维护；本页仅为当前交接。
