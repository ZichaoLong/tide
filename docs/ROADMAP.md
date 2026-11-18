# 当前实例路线：已交付阶段与后续待办

容量、吞吐与区域执行的六项计划已实施并收束；当前事实与源身份见 [STATUS](STATUS.md)，
完整结果见[证据索引](evidence/README.md)。数学保持
`4f5b9f20a4030d68ee66ae2d26e34659a5e37234`。持续实例路线按
[project-scope](project-scope.md)推进，通用基座未启动。

## 本阶段交付

| 阶段 | 已交付 | 验收范围与退出 |
|---|---|---|
| 1 配置与验证 | [支持目录、固定通过 Hub、显式时钟、terminal Hubs](optional-bindings.md) | 全菜单元数据核对；有限公式/VJP/多步/恢复和隔离安装检查。可表达、执行、数值、性能分列，未穷举执行全部组合。 |
| 2 容量与吞吐 | [B/L 单轴与固定模型预算搜索](evidence/capacity-search-20261118.md)、[独立存储账本](evidence/capacity-execution-20261118.md) | 六组搜索、19 审计/95 训练子进程；到 candidate_limit，未声称硬件极限。参数/状态/保存张量/storage 与 allocator 分开。 |
| 3 内存与算子 | 紧凑 Timed 尾部、最后消费者释放、区域重算、局部 Attention 后端选择 | [正常计时与诊断](evidence/capacity-execution-20261118.md)分开；重算作为容量选项。扩大 B 后最佳已测吞吐未超过基线，不转为默认。 |
| 4 区域批量与 C++ | 相容区域合批；独立 LibTorch 评分切片、串行及 2/4 worker | [合批](evidence/capacity-rounding-20261118.md)修复 BF16 偏置错误并保留舍入边界；[native](evidence/native-regional-performance-20261118.md)有限精度/性能对照完成，只保留可选子路径，不建设第二套完整模型。 |
| 5 同层多设备 | 完整区域归属、运输去重、统一更新/checkpoint、serial/per_device | [双卡资格与性能、9B 四卡和17B 六卡两精度更新](evidence/region-capacity-20261118.md)完成。完整区域不可拆、batch1限制及有限数值差异明确保留。 |
| 6 集成比较与续接 | 共同 B/L 的 A–D、固定校准模型吞吐搜索；0.5B/1B 三模式连续运行与新进程恢复 | [12 子任务/36 段及四份恢复](evidence/capacity-continuous-20261118.md)完成；[544 项 CPU 与17项定向](evidence/capacity-validation-20261118.md)、[隔离 wheel](evidence/capacity-installed-20261118.md)收口。数值失败未改阈值或重标。 |

本阶段采用的常规改进是张量生命周期修正和语义/记账错误修复。合批、重算、小 Attention
math、区域 per_device 都是显式候选，选择依据是具体工作量、精度和设备。既有在线分块和
native Attention 保留各自适用条件。性能观察不能给所有图、活动比例和候选组合自动排名。

## 后续可选工作

没有本阶段遗留计算作业。以下根据后续实际负载选择，不自动启动新矩阵或扩大产品范围：

1. **减少有限精度轨迹差异。** 从已定位的 BF16 Linear/矩阵归约和共享梯度累加见证选择
   一个问题，评估稳定布局/累加的精度与成本；保留原失败及阈值。单次前向精确不认证
   后续 optimizer 轨迹；不以无界舍入调查阻止其它工作。
2. **组合设备归属与区域合批。** 当前 whole-region placement 要求 batch1。先验证同卡
   相容区域合批的参数/私有状态所有权、None梯度、完整BPTT和恢复，再衡量运输、调度、
   实际工作量与逐卡峰值；不能用已有分开的检查代替组合资格。
3. **按目标负载扩大容量或窗口。** 当前 B/L 搜索都有限停止，基线 Settle B32等点未测。
   当下游需要时固定窗口/卡数/预算并给予 Tide 和 A–D 同等调优机会，找到可复核边界。
   输入 L、真实历史窗口、chunk、每次更新跨度分别声明；长输入不等于扩大历史窗。
4. **大模型连续与恢复。** 9B/17B 当前只新增 B1/L8 两步区域训练。长序列、更多更新、
   大 checkpoint/恢复和 A–D 训练须另定主机/设备预算；先估计恢复副本和 optimizer峰值。
5. **有证据后再扩展后端。** native 是区域切片；当前没有将其接入完整模型的普遍收益证据。
   CUDA 真机、peer DMA/HCCL、resident/编译训练仍无资格。只有具体收益/消费者需求时
   才启动对应有限实现及硬件验证，现有前端拒绝或事件错误不自动等于永久不可行。

## 后续工作的固定口径

Transformer比较遵守 [benchmark-contract](benchmark-contract.md)：A/B 匹配 block数与
总/实际活跃参数，C/D 匹配深度与总/实际活跃参数。共同 B/L 每点建立匹配；预算搜索
冻结校准模型并记录活动变化，若声称最优点仍匹配活跃量则重新匹配复测。
同负载时间、独立容量吞吐、数学工作、padding/搬运/调度和未解释差距分别记录。

FP32/BF16按 [precision](precision.md) 和 [verification](verification.md) 执行；CPU FP64
只作有限诊断。完整训练包括准备、反向、AdamW、detach和全设备同步；包含的探针/I/O
不能事后从端到端时间扣除。续接保留声明的梯度与detach边界，并真实执行恢复后的更新。

长作业遵守[执行契约](execution-flows.md)：当次配额/活动任务/资源核对、冻结输入、独立
输出、时间/内存界限、background.slice及持久记录。按收益并行并统一准入；实现提交后
做固定资格，证据另提交，按当次授权正常推送。共享 NPU 权限不从本阶段继承。
参考、公共环境与下游保持各自边界；本库交付不代表下游已经升级或接受。
