# 当前交接 — 批量接口重构对齐已记录，执行计划待审阅（2026-11-11）

用户已确认：固定core-5全局/局部数学为锚点、性能测试牵引、公共基座定位；架构/API/源码
必要时可整体重写。KV是重点候选，区域评分/证据/不同发送与约束求解及反向同样必须张量贯穿，
覆盖控制、活动支持、Full、消息路由与下一阶段聚合。不同记忆保留各自有类型的私有状态，
不强制同一大张量；以当前有限候选设计，不承诺任意未来模块只靠reshape就高效接入。

本次请求是记录对齐并整合新计划。本次仅更新STATUS、ROADMAP和execution-flows，
没有开始新一轮源码重构、构建或性能实验，没有活跃项目作业。
已确认原则见[持续契约](execution-flows.md#批量接口与模块替换)，待审阅实施顺序见
[唯一计划](ROADMAP.md#当前计划批量张量贯穿与架构验证待审阅实施)。

## 从这里接续

1. `git status --short --branch`、`python scripts/status.py`、`python scripts/check_upstream.py`。
2. 先读ROADMAP页首P1—P6及execution-flows新增约束；当前先向用户提交计划，收到实施指令后按该计划推进。
3. 复用[上轮固定证据](evidence/kv-query-blocks-20261111.md)和[可重复诊断入口](fc16-cost-attribution.md#repeating-the-investigation)。
   新版重点是默认整链和各层实际增量，不能只再做一个局部优化后声称差距已闭合。
4. 开始实施时先具体化P1契约/符号/反例和P2有限对照。FC/KV与区域是主线，早期GDN/SSD及异构小例检验接口。
   保留一般PDG反馈、非对齐TimedDAG与一般Settle端口检查；不自动恢复历史大矩阵或GPU作业。

## 源码与证据身份

生产代码基线`63daf0f`；本次只有文档记录，不产生新的数值、后端或性能资格。
数学固定`0770593838a37c790a568226930567ed60f99b83`，19份core-5离线快照校验通过。
用户未跟踪的`docs/tide-learning-lab-TL-REQ-004-handoff.md`和
`docs/tide-learning-lab-TL-REQ-005-handoff.md`原样保留，不纳入本次提交。

上轮KV块实现`5ff3ce2bfe72b4416814b750d5260f5c90038eca`，计时/profile
`598b4fb5f487befe008f50afb1fe1e378504c417`，最后后备预检/诊断
`206bc9b33be15b891598997bf12a8cec5e6f337b`，配对差异标记`19af25e`，证据`63daf0f`。
相关资格原始通过、xfail、启动失败及补验分别见报告和资格JSON，不由本次文档提交重新认证。

FC16 B8/L128/d128默认/直接FC完整训练CPU为5.835/1.390秒，NPU为10.928/0.675秒。
约99.2%的Read/Full已命中块，但跨模块仍拆行重组；剩余代价没有完整归因。
完整尺度FP32任务投影严格失败保留，同一FP32材料提升FP64后值/末状态/任务梯度通过；
不外推完整中间根、所有轨迹、多步训练、任意规模或设备。新计划承担后续验证，非已完成声明。

## 作业、环境与资源

没有活跃作业。上轮原始记录`artifacts/kv-block-20261111/`与
`artifacts/jobs/kvblock-*/status.json`只作原证据；命令/终态/日志见
[evidence/kv-query-blocks-qualification-20261111.json](evidence/kv-query-blocks-qualification-20261111.json)。
最近`gef-kvblock-final-build2`、`gef-kvblock-anchors-cpu`、`gef-kvblock-final-npu`终态passed；
原内部严格失败、早期构建失败和取消未改写。没有恢复这些作业或编辑其固定输入。

上轮开发入口`artifacts/kv-block-env.sh`，构建`artifacts/kv-block-20261111/build-final`
及`build-final-sdk`，复用前核验源码/ABI/环境身份；干净接入以[development](development.md)为准。
本次记录前文件系统约908GiB可用、shm约398GiB可用；这不是未来作业的容量准入。
CPU/内存和空闲NPU可充分使用；共享NPU须当前授权，GPU真机延后。新长任务须固定源码、
有限资源/时长与独立background.slice；磁盘确实阻塞时及时汇报，可用有界shm但不作持久证据。

Historical source IDs: python scripts/resolve_history_commit.py OLD_SHA; see docs/history-migration.md.
