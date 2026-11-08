# TL-S1 — 等卡超时，NPU采样尚未执行

2026-11-08终态审阅。问题原为：同一FC/B32/NPU FP32步骤中，主线程采样及公共工作计数能否缩小耗时解释，观测开关是否保持模型/optimizer状态。

`s2-npu-sampling`进入共享队列但未获得空闲卡；请求等待300秒，队列终态记录306.70秒，完整服务308.20秒。队列理由`insufficient_free_devices`，launcher exit3、supervisor/systemd exit124；实际workload未启动，没有worker-state、模型输出或Lens模型run。当前cgroup与已知后代均为空。[原记录hash及终态](../evidence/second-batch-terminal.json)。

执行状态是排队超时，严格数值、观测开关等价、离散/工作观测、恢复和学习均没有本组新证据。不能把没有性能结果解释为NPU性能差，也不能继承首批profile失败作为本次测量。原严格失败与首批profile超时分别保留。

源码94bf712、配置`configs/studies/second/npu-sampling.json`仍固定；如果后续选择补做，使用新作业ID和新输出，保留一次计算≤9分钟的边界。建议将等待预算单列为≤2小时，保留服务总时限及队列无进程/双采样/锁规则。当前没有自动重试、CPU替代测量或新性能归因，公共执行核心不修改。
