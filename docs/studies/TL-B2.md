# TL-B2 — 冻结骨干比较未启动，收益仍未检查

2026-11-08终态审阅。问题原为：同固定Qwen2.5-0.5B-Instruct骨干冻结后，head-only/TIDE/MLP各40步/B32/SGD .01，TIDE分支是否对共同检索任务作出实际预测贡献。冻结骨干的梯度/optimizer和新进程完整恢复是训练前置检查；原25%学习门槛不变。

`s2-frozen-growth`在队列等待，前段等待TL-S1、随后等待真实空闲卡；请求等待600秒，实际队列记录610.39秒，完整服务611.93秒。最终为`timed_out`/`insufficient_free_devices`，launcher exit3、supervisor exit124，workload没有启动。故冻结准入、恢复和三候选训练均未执行，未创建学习run或Lens投影，当前无后代。[生命周期证据](../evidence/second-batch-terminal.json)。

这不是冻结策略、TIDE分支或模型加载失败。不能从首批未冻结骨干的梯度/恢复检查推导本次冻结路径已经通过；初态、严格数值、离散机制、恢复和学习在此变更下仍按未检查记录。首批B1预测贡献未证实的结论不变。

固定源码94bf712及`configs/studies/second/b-stage.json`所引用的配置保留，原模型缓存和环境未改。建议有设备窗口后再以新作业ID补做原小比较：单NPU/CPU4线程/RAM24GiB、实际阶段≤45分钟；将排队预算单列，建议≤2小时，并设置含清理的服务总上限。仍先做冻结检查和完整恢复，无收益则停止，不增加替换层数。当前未重试，也未自动改成CPU全模型训练。
