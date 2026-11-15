# MoE补充批：执行与验收口径

本页落实[已授权研究表](moe-controls-proposal.md)，不扩大候选。共同墙钟从
2026-11-15 09:05:08至11月17日09:05:08（北京时间）；已有v10截止和源码独立。
六种骨干先测完整global64更新的micro16/32/64，再按实际吞吐、显存和头部余量选择。
至多8个新增单卡worker，共用旧12slot和设备生命期锁；CPU4线程/服务、RSS64GiB、服务内存72GiB。
新增盘目标64GiB/停止96GiB，账号硬余量至少200GiB；不设累计算时上限，不自动延伸256Mi。

|阶段|问题/改变量|候选与控制|曝光与指标|晋级/停止条件|
|---|---|---|---|---|
|实现准入|dropless top2能否正确训练|分桶batched GEMM vs独立逐专家CPU公式|小型前向、输入/参数/aux VJP、支持、因果、别名、optimizer|非有限/丢token/非法路由/确切恢复失败阻塞对应路径；有限严格超差保留|
|批量校准|global64的有效物理batch|六骨干各micro16/32/64，5完整更新，末3计时|含packing/辅助预遍历/VJP/Adam的targets/s、峰值HBM|有限且有余量，选实测最快；不改global64|
|配方筛选|MoE与稠密均有合理学习率/平衡配方|24MoE+6dense，seed33|8Mi validation131072、NLL与aux分离、各层负载/冷专家/熵|每形状前2有限完整结果到32Mi；失败保留，不跨形状挤掉E64|
|配方固定|更长先导是否改变选择|至多12个32Mi前缀|同validation；从起点声明256Mi LR日程|每形状32Mi最低NLL，接近时确定性ID排序；不得查看confirmation选择|
|主比较|共同曝光和近活跃容量下语言质量|四MoE形状+两匹配dense，各seed11/22|从零128Mi，8/32/128Mi检查点，独立262K确认|无自动256Mi；缺失不算通过；不按质量排除合格确认端点|

标准因果attention与现有Transformer共用融合路径。专家GELU两矩阵、独立初始化；top2选择FP32
logits，再对选中logits做softmax（与全softmax选择后归一化实数等价，避免选中和下溢）。
没有容量截断、共享专家或路由噪声。专家选择10%/3.125%与整体活跃参数比例单列。

当前栈的`npu_grouped_matmul`前向可用但没有自动反向；`npu_grouped_matmul_add`不支持FP32。
原失败保存在`runs/moe-kernel-v1`，不靠隐式降精度训练。正式候选用普通Torch分桶bmm：
按每专家分配数的2次幂分桶，同桶专家一次batched GEMM，所有选中行保留，填充行少于真实分配数的2倍。
每层一次小计数下载，整数packing元数据在host构造，数值/反向在NPU；无逐token/逐专家Python矩阵乘。
CPU逐专家路径仅用于小型独立参考，不当作NPU吞吐基线。

辅助目标按层均值：`f_i=count_i/(2*Nvalid)` stop-gradient，
`balance=E*sum(f_i*mean(p_i))`，`z=mean(logsumexp(logits)^2)`；padding不计统计。
物理拆分时先做确定性无梯度遍历收集全global64频率，逐micro的概率和/z项以同一个Nvalid归一化。
额外前向行数/耗时独立记录；CE、balance、z与合并objective分列。专家bias虽存成二维，明确不做weight decay。

同配置新进程恢复用三次B4/L32真实数据的完整任务+aux更新，比较模型/Adam/梯度/RNG/游标逐位一致。
这不是长轨迹逐位重放证明。每个正式配置独立准入和恢复，不使用退出码代替内层验收。
只为本批校准保留8/32Mi科学权重；所有后继已结束后可按哈希回执退休校准的冗余Adam与小型恢复临时载荷。
正式128Mi优化器/权重及全部配置、检查、失败、原始指标保留，不触碰历史v9/v10。

Lens项目`tide-learning-lab`：`TL-MOE-CAL`用于实现/批量/配方，`TL-MOE-CONTROL`用于正式MoE及dense，
`TL-MOE-CONFIRM`用于独立确认。私有入口只在`.local/LENS.md`；项目JSON/JSONL是权威。
研究图仅登记运行关联，学习结论等待审阅。
