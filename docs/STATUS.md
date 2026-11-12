# 当前交接 — 直接FC跨B批量化已完成，停下对齐（2026-11-12）

数学固定 `0770593838a37c790a568226930567ed60f99b83`；授权分支 `graph-execution-foundation`。
本轮只实施用户明确授权的直接FC改造及CPU/NPU吞吐比较。其他重写、旧代码/测试清退及skills治理
继续只读对齐，不因历史P1–P6授权自动开工。固定数学是正确性锚点；旧实现和旧测试不构成架构约束。

## 已完成

实现 `cc6ccbf7` 的 `scripts/fc_tensor_baseline.py::full` 消除Python B/L循环：
实际选中Q统一投影，Attention按B×H×Qmax×Dh对每样本唯一KV bank计算，输出/FFN只算选中行。
保留动态索引、Attention padding和节点/来源分派；不称最优实现，也没有修改默认基座执行器。

固定资格CPU31/NPU9项通过。独立逐查询指数商检验Full值及全部输入VJP，CPU另有非空左态、
共享/跨窗、三步SGD和新进程恢复。完整FC16性能点未执行逐张量严格核对，保留有限FP32差异。

20点全部执行完成；FC16 B8/L128/d128/H2/FFN512、FP32、纯PyTorch主机调度：
完整训练CPU1线程1.3646→1.0444秒（750→980输入位置/秒），NPU host2线程.6736→.2913秒
（1520→3516）。同轮Transformer16为CPU.7465/NPU.04209秒；NPU剩余6.92倍差距未完全归因。
完整三模式、配置、工作量、首token、SDPA负结果及复现见
[evidence/fc-batch-full-20261112.md](evidence/fc-batch-full-20261112.md)及同名JSON。

## 作业与资源终态

`gef-fc-batch-full-{cpu,npu}.service`均inactive/dead，MainPID0、空cgroup、exit0；
两个内层matrix各10点均completed/exit0，CPU/NPU JUnit分别31/9通过，无遗留本轮作业。
原始记录在 `artifacts/fc-batch-full-20261112`，status/log在
`artifacts/jobs/fc-batch-full-{cpu,npu}`；正式小记录已进Git，不依赖原目录才能复现。
冻结源码 `/dev/shm/gef-fc-batch-full-20261112/{before,after}` 分别是 `bb23a9fa`/`cc6ccbf7`，均干净。

CPU affinity0–7、NPU host8–15；两服务各128GiB/1800秒，单点240秒；队列准入空闲物理0→npu:0。
外部项目共享主机/其他卡，不宣称独占峰值。复用CANN9.2/TorchNPU2.12私有环境，无C++构建。
全CPU/内存及本对话共享NPU授权有效；共享授权不延续到新对话。磁盘/shm未阻塞本轮。

## 已确认反馈与下一动作

张量贯穿包含Aggregate、KV状态/Read/Describe、区域评分/分配、Full、消息、反向和更新。
现有skills已要求这些边界；直接样板仍需审查自身是否有B/L碎片，不能因名称叫“直接”就当性能上限。
新样板可重建测试、拆分Settle窗口与PDG/TimedDAG就绪批实现；其他架构事项本轮只讨论。
默认基座、LH、resident及历史数值/异步限制不由本次修改解除，唯一待办仍在ROADMAP。

下一动作仅是向用户报告并对齐后续计划；重入先运行 `git status --short --branch`、
`python scripts/status.py`，阅读本页与上述证据，不自动启动旧矩阵或总体重写。
三份用户未跟踪REQ-004/005/006交接文档保持原样，不提交。
