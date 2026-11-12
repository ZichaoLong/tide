# 当前交接 — 完成 P1—P6（2026-11-12）

当前授权：持续完成[ROADMAP P1—P6](ROADMAP.md#当前计划批量张量贯穿与架构验证已授权实施)，允许重写、阶段提交并推送当前分支，GPU真机延后。
数学固定0770593838a37c790a568226930567ed60f99b83；19份快照再次通过。
共享NPU的当前对话持续授权涵盖正确性、profile与性能；每次重新采集身份/资源，经双采样和组锁准入，标明共享条件。
三个用户TL-REQ-004/005/006 handoff只读、不提交；下游、参考仓库与公共环境不改。

## 当前实现与下一动作

已提交bb27e4b0（LH非空左态/独立根）、4e3e4bdb（计划图AdamW/新进程恢复）、55a3074a（跨节点KV查询/Describe合批及standalone列owner）、a09735f1（resident KV sequence）。
当前HEAD a09735f1；本批尚未推送，远端基线2bca34fd。

- 主代理固定共同整数原点RoPE、execution-v6显式迁移、SDK检查与队列请求记录；相关Python CPU检查67通过，native须新构建。之后独立Python-owned/SDK构建和定向资格，再测LH三层/三模式及FC防退化。
- resident_rewrite继续session/owner、跨窗训练和恢复；稳定local/flow/独立公式已待主代理冻结设备验证。完整flow采用设备choose/commit及实际journal反向；组件不代替整图或FC16规模。
- 导航修复与24文件/3文档段落审阅已经完成，789个本地链接/锚点检查通过；尚待统一提交，未宣称全仓逐函数审阅完成。

## 终态与精确范围

固定源码工作树：/dev/shm/gef-p1p6-20261112/core-55a3074a（tracked只读）。
构建：core-55-native、core-55-sdk、core-55-benchmark/client（同父目录）。

- gef-p1p6-core-55-native：passed；CPU254 passed/9原xfail；wheel和独立安装完成。
- gef-p1p6-core-55-sdk：failed；core/query/owner已通过，benchmark因JSON include缺失失败。后继gef-p1p6-benchmark-55显式include构建和3个CTest通过，旧失败保留。
- gef-p1p6-lifecycle-55-cpu：passed；standalone五记忆三步/新进程5项，LH176七独立输出族FP64数值投影，安装Python/native三AdamW和精确恢复。None/zero连接差异仍按原投影范围保留。
- gef-p1p6-lifecycle-55-npu：failed；两族FP32严格数值投影有限超差，离散/结构通过；安装Python/native三AdamW及新进程精确恢复均passed。外层failed不重标。
- gef-p1p6-peer-55：failed；普通队列EE1017保持；PER_STREAM_QUEUE=1与TASK_QUEUE_ENABLE=0各stock12轮VJP通过，全部未设launch blocking。
- gef-p1p6-peer-full-55：passed；TASK_QUEUE_ENABLE=1/PER_STREAM_QUEUE=1，双NPU Python/native×KV/SSD独立根/训练/恢复4项，4安装客户端及SDK五记忆×三放置15项通过。只认证已测配置，普通队列失败独立保持。
- gef-p1p6-sequence-a097及-device：passed；固定a097，B2/L3/N2、非空raw/独立VJP/saved reverse。不是resident整图/owner/optimizer资格。

作业终态/log：artifacts/jobs/p1p6-*/；计划/矩阵/逐项报告：artifacts/p1p6-20261112/。
接续先运行 `python scripts/status.py`，检查后继任务实际终态。当前以上作业均已终止；新的构建命令/预算在提交前后更新本页。

## 资源、证据与继续条件

大构建继续有界shm，持久记录在磁盘；最近quota293GiB/软300/硬800，无grace，shm305GiB空闲。没有存储阻塞；不清理用户/参考数据。
构建可用全部可用CPU/内存；以CPUAffinity协调独立任务，性能计时避开自身重构建。无NPU锁/等待许可阻塞，准入仍需实际采样。

55a整链证据草稿artifacts/p1p6-20261112/query-lifecycle-evidence-draft.json待纳入NPU终态并审阅发布。
已有各轮证据保持原源码/失败身份。剩余限制与有限停止条件只在ROADMAP；每项实现固定后执行相关资格，证据独立提交，阶段提交后继续。
