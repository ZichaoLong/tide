# 当前实例路线：本轮结果与后续待办

产品范围由 [project-scope](project-scope.md) 维护；当前实现、数值边界与复现入口在
[STATUS](STATUS.md) 和 [固定证据](evidence/README.md)。数学仍固定为
`4f5b9f20a4030d68ee66ae2d26e34659a5e37234`。主实例路线持续有效，通用基座未启动。

## 本轮五类工作

本轮以实际结果或已调查的具体边界收束；执行成功、数值等价和性能收益分别记录。
所有原始超差、运行失败和阈值保留，没有将预算、计划或未运行路径重标为通过。

| 工作 | 已交付范围与证据 | 适用边界 |
|---|---|---|
| 比较设施、配置和规模 | [A–D 设施](benchmarking.md)、[CPU/NPU 七点主矩阵](evidence/roadmap-representative-20261118.md)、[稀疏/形状/组合](evidence/roadmap-supplementary-20261118.md)、[0.5B/1B](evidence/roadmap-scale-small-20261118.md)、[9B/17B](evidence/roadmap-scale-large-20261118.md) | 四类同时匹配两个约束；活跃触及、教材代理和重复工作分开。大规模只资格化实测的模式、B/L 和设备。 |
| 热点和差距解释 | [有限候选](evidence/roadmap-hotspots-20261118.md)、[三类 Attention](evidence/roadmap-attention-20261118.md) | 采用 NPU KV 压紧和兼容 head-query 融合；CPU 保留原路径。Full 分桶回退、额外 gate/up 无明确收益。实际工作变化与未解释差距均保留。 |
| 连续运行和完整计时 | [连续矩阵](evidence/roadmap-continuous-20261118.md)、[whole/chunk VJP](evidence/roadmap-numerics-followup-20261118.md)、[放置恢复](evidence/roadmap-placement-20261118.md) | 真增量 KV、B1/B4、跨窗、Timed flush、三段续接及恢复后实际更新；恢复一致性与性能计时各有边界。 |
| FP32/BF16 数值 | [明确精度策略](precision.md)、[初始调查](evidence/roadmap-numerics-initial-20261118.md)、[后续调查](evidence/roadmap-numerics-followup-20261118.md)、[安装和 CPU 套件](evidence/roadmap-installed-20261118.md) | 修复 BF16 模式依赖的 Attention 表示；保留跨精度支持差异、有限分块/梯度/AdamW 超差。CPU FP64 是诊断。 |
| 执行路径和硬件 | [host-stage 与事件失败](evidence/roadmap-placement-20261118.md)、[实际大容量](evidence/roadmap-scale-large-20261118.md)、[TorchAir 有限探测](evidence/roadmap-resident-20261118.md) | CPU/单 NPU 与必要多 NPU 已实测；CANN 路径覆盖错误已修复。直接/同步 NPU 传输仍有事件失败；没有 resident 或 CUDA 真机资格。 |

## 后续工作及重启依据

以下是可按实际需求选择的待办，不自动启动新实验或继承本线程的共享 NPU 授权。
实施前重新核对 HEAD、活动任务、配额和资源；已有有效证据沿用原身份。

| 候选工作 | 已知依据 | 何时继续及必要检查 |
|---|---|---|
| BF16 支持稳定性及重复反向/恢复更新 | 真正的舍入边界跨越与同 owner 反向变动已有独立见证 | 需要更严格轨迹一致性或下游复现时，针对评分链/归约/贡献汇总做有限候选；保持原阈值和支持语义。 |
| 小接收/区域 Attention 的普通 math 路径 | NPU 已测小型形状的 post-projection 局部见证快于 native/tiled；尚未形成整网采用证据，CPU 无稳健排名 | 先覆盖零/微小正 prior、VJP、共享梯度及两精度模式一致性，再测完整训练/续接。 |
| 高 padding Full 的更低调用成本布局 | 旧分桶减少部分 padding 却增加调度和反向图，整网回退 | 出现更高 padding 的实际负载或可批量化索引方案时再比较，避免仅按行数预测收益。 |
| 更大规模、长窗口与层内切分 | 9B 四卡容量已实测；17B 最宽层的 FP32-master AdamW 持久预算约 83.8 GiB | 如需 17B NPU 训练或更大 B/L，先预算层内分片/通信/临时张量，并验证统一更新和真实恢复；不能用总卡数消除单层瓶颈。 |
| 原生传输、resident 与 CUDA | host_stage 可用；原生事件生命周期仍失败；TorchAir 在静态 role guard 前端停止；本机无 CUDA GPU | 有必要吞吐或硬件条件时启动独立有限试验。先去除可静态处理的 guard，再观察实际 backend 调用和动态选择；编译成功不等于 resident。 |
| 更精确的短时性能比较 | CPU 短基线及不同 NPU/时段有可见波动；同构基线也会漂移 | 只有结论依赖细小时间差时，设置交错/随机顺序的配对样本、记录频率与争用；不为微小排名无界复测。 |

数据、任务头、loss、训练配方和模型质量归下游实验。本轮交付不代替任何下游依赖升级或验收；
按 [协作契约](downstream-contract.md) 分别记录。后续更换上游数学或启动通用基座需要明确的新任务。
