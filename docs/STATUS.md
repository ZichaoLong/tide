# 当前交接 — 完成 P1—P6（2026-11-12）

授权继续[ROADMAP P1—P6](ROADMAP.md#当前计划批量张量贯穿与架构验证已授权实施)，可重写、阶段提交并推送；GPU真机延后。
固定数学0770593838a37c790a568226930567ed60f99b83的19份快照已通过。共享NPU持续授权涵盖正确性、profile和性能；每次仍采样身份/资源并双采样组锁。用户TL-REQ-004/005/006三份未跟踪handoff保持只读、不提交。

## 当前源码与动作

正在提交来源块合批与resident v2公共实现；远端仍51e93e74。用户三份handoff不提交。
- 来源块：Python/C++完整来源MLP合批、同输入/完整参数槽精确复用、稳定伴随。开发CPU81 passed/5原严格数值xfail；新native/SDK待固定构建。无性能结论。
- resident v2：真实Model/Continuation、来源低秩VJP、schema2账本、公开字段根、C++/Python消费者及静态默认分派。定向syntax及CPU metadata/default检查通过；真实公共设备与FC16规模未验。
- 下一作业使用新固定提交及 `/dev/shm/gef-p1p6-20261112/core-source-v2`，分别构建Python-owned与SDK，再从相同源码构建resident；不使用fb83旧core冒充匹配。
- resident_rewrite接续实验默认能力分派（真实Settle证书、tensor输入/loss、版本化容量与记录），独立于冻结构建；主代理统一审阅集成。
- 静态LH报告及fb83的42点成本/资格报告待证据提交。静态审阅覆盖入口/返回、循环和实际张量工作量，与profile对照，不把调用次数直接当性能原因。

## 固定资格与有限成本

- fb83ee91：Python-owned新构建CPU281 passed/9原xfail、wheel安装；独立SDK及benchmark/3CTest通过。
- LH176非空CPU FP64 task/messages/raw数值投影通过；NPU messages/raw有限严格失败，结构/离散通过。None/零连接投影边界保留，不能宣称全部严格连接等价。
- CPU/NPU × Python/native独立安装的位置与LH176三AdamW/新进程精确恢复通过。NPU lifecycle外层failed保留。
- 55a普通双卡队列EE1017保持；显式TASK_QUEUE_ENABLE=1/PER_STREAM_QUEUE=1且unset launch blocking：列整链、SDK五记忆三放置和一般delta-pdg/ssd-dag/kv-pdg-initial六格四步/恢复通过。配置资格不能跨队列外推。
- 85912 resident v1实际小图flow、三AdamW×两retained窗BPTT、fresh-process全部通过。仅raw block session；完整Model/Continuation/public/default是当前v2。
- fb83的42成本点CPU/NPU均exit0：LH176三层三模式、CPU32、FC16默认及direct/Transformer、LH约1B。1B direct/default活动/消息计数有差，不作严格等价加速。待完成审阅报告。
- f475的LH同材料direct/default算子及autograd诊断CPU/NPU均终态passed；无optimizer、不是吞吐。NPU日志含profiler schedule警告，trace完整性单独审阅。四个fb83 packing观察及SDK NPU查询/owner通过。

旧失败（a605计数器编译、SDK JSON include、源码hash拒绝、fixture/launcher边界等）均按原记录保留。
已发布55a/a097资格和边界导航：[query-lifecycle](evidence/query-lifecycle-20261112.md)、[边界审阅](evidence/p1p6-boundary-review-20261112.md)。

## 作业与资源

目前此前作业均终态，先运行 `python scripts/status.py` 核对新后继；完整保留失败用 `--all-jobs`。
固定源码/构建在 `/dev/shm/gef-p1p6-20261112/`；持久计划/矩阵/log索引在 `artifacts/p1p6-20261112/`，job状态在 `artifacts/jobs/p1p6-*/`。
fb83源码core-rope，Python-owned core-rope-native，SDK core-rope-sdk；f475诊断源码lh-profile；85912设备源码session-v1-fixed及session-v1-fixed-device。
当前没有活跃长作业。新构建冻结源码并核对core源码hash，不读共享WIP；使用background.slice、显式CPU/内存/时限及停止条件。
最近quota301GiB/软300/硬800，6天grace，shm300GiB空闲；无写入阻塞。大构建/profile继续有界shm，持久记录原子fsync；不清理用户/参考数据。

下一步：审阅新证据与来源块改造；公共resident v2稳定后提交并构建，独立整图根/训练/完整Continuation/新进程/安装资格，再执行原预算FC16规模。维护语义版本、能力/导航/经验和P1—P6实际范围。阶段提交后继续，不重复请求常规授权。
