# TL-A0 — 接入机制成立，严格数值资格按范围保留

问题是实验自有embedding/读出/loss与公共执行核心、共享参数、optimizer及应用恢复能否可靠组合。数学固定0770593，基座85d161d；只消费安装的公开API，没有复制执行核心。全部检查、原始失败与hash见[启动审计](../startup-audit.md)、[NPU实际尺寸](../evidence/a1-npu-admission.json)和[HB实际尺寸](../evidence/a2-hb-admission.json)。

CPU64微型FC完整独立前向/状态/路由、输入/参数梯度、共享求和、AdamW、因果和streaming/prefill通过。实际任务尺寸的NPU FP32 FC及CPU FP32 HB分别完成离散支持、梯度连接、参数共享和因果检查。它们不是每个正式训练步的CPU参考认证。

严格数值失败没有消失：原基座NPU根、历史FP32 FC AdamW、后续NPU FC及CPU HB quad均保留原阈值下的有限超差与未知原因。实际HB dense的B2严格通过不覆盖quad或其他配置。已知机制/离散/非有限/恢复错误仍阻塞受影响路径；本次有限数值探索不改阈值，也不把小误差推定为无害舍入。

新进程恢复在CPU FC、NPU FC、两种CPU HB和NPU Qwen+TIDE各自的小型声明范围通过，比较模型、optimizer、数据步、CPU/Python/设备RNG及episode边界；跨后端、不同物理映射和完整长轨迹逐值恢复未检查。四个新增80步终态checkpoint另已核对step和hash，这不等于再次执行恢复。

结论：相应机制已足以支持本批明确标记的学习探索；严格实现等价仍不成立于上述失败范围。现无活动作业。后续数值需求保留在[TL-REQ-001](../requests/TL-REQ-001.md)，不阻塞独立的学习问题，也不因学习成功自动关闭。
