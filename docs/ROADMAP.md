# 训练默认、路径核对与性能归因：本轮完成与后续

用户已确认的六项执行工作已完成，当前事实在[STATUS](STATUS.md)，
持续范围在[project-scope](project-scope.md)。数学仍固定
`4f5b9f20a4030d68ee66ae2d26e34659a5e37234`；通用基座未启动。
完成有界工作不等于消除全部数值偏差或证明全局性能最优。

## 本轮已完成

| 计划项 | 交付与核验 |
|---|---|
| 代表负载与预算 | FC-K、Settle-X、Timed-X及0.5B；补齐batch-only，分清纯合批重算，记录allocated/reserved及有限停止原因：[搜索](evidence/training-search-20261118.md) |
| 跨实现数值与路径 | 非空state/AdamW共同起点、同owner重复、真实更新、新进程恢复；31候选与有限FP64/Attention见证，原失败保留：[路径](evidence/training-paths-20261118.md) |
| 双边性能归因 | 正常样本与侵入式诊断分开，实际硬件trace、数学工作/padding、保存内存与Transformer两种有限调优：[成本](evidence/training-costs-20261118.md) |
| 训练优化与默认 | Reception局部重算、摘要prefix、命名预设；六组20更新和六次独立进程恢复更新：[连续](evidence/training-continuous-20261118.md) |
| 其他模式 | 推理、含图前向、完整训练A–D主表，首个主干输出、缓存decode、Timed输入/flush；修复末尾拼接同步并重测相关项：[模式](evidence/training-modes-20261118.md) |
| 可发现入口 | CPU安全默认、显式NPU预设、覆盖/dry-run/真实resume、模式建议、新wheel隔离安装：[建议](execution-recommendations.md)、[验证](evidence/training-validation-20261118.md) |

四类Transformer比较仍按[联合契约](benchmark-contract.md)。参数/block匹配不等于
算术量或关键路径匹配；不禁用Tide合法合批，也不把Transformer称为已证明的最优实现。
完整模型仍为PyTorch，既有LibTorch切片保留原范围。

## 有限后续机会

这些是当前证据提出的后续候选，不是隐含的新矩阵或本轮未提交的运行任务：

1. **矩阵SSD扫描。** 实际trace中Cumprod的AICPU任务有明显成本。可以对衰减乘积与
   矩阵块传播作独立消融；先保持零衰减、完整梯度、int64时钟、状态续接，再比较
   时间和内存。向量摘要prefix已优化，不把旧合并scan时间全部归给它。
2. **索引、padding与allocation。** 根据实际selected/executed行和kernel工作寻找
   packing/专用反向/融合机会。当前证据不建立完整host-to-device关键路径，也不能
   从某个kernel累计duration直接预测端到端收益。
3. **未解释的训练数值差异。** Timed BF16的支持/时间分歧、单/多设备反向累加及
   严格恢复超差仍保留。进一步定位优先复用已有anchor和首次支持见证；不再仅凭
   提高全量捕获上限重复大轨迹，也不改变阈值来取得通过标签。
4. **真实下游负载校准。** 当前选择基于合成主干训练。加入数据/任务头/实际loss后
   重核对reserved与设备外占用，按模式选择B/L和候选。更长时间、更多尺度、CUDA、
   peer通信或编译训练需要相应新任务与硬件证据；没有从本轮自动继承资格。

执行继续遵循[精度](precision.md)、[验证](verification.md)和
[执行契约](execution-flows.md)。实现先冻结，资格另行记录；长期作业有界且可查终态。
共享NPU授权仅属于给出授权的线程，参考/公共环境/下游权限不变，不查退役历史。
