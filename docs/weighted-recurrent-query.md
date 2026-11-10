# 循环 Read / Full 按用途求值

固定`0770593`的[循环记忆](weighted-recurrent.md)、GDN/SSD与原始逻辑时间短卷积定义不变。
本页说明[批量查询](weighted-query-batch.md)内部只计算实际消费的数值分量；
它不改变原始状态、checkpoint解码或局部函数的返回结构。

| 用途 | 实际堆叠及衰减字段 | 特征 |
| --- | --- | --- |
| 四记忆的sum Read | Z、W | 无矩阵查询/卷积 |
| affine/delta的state Read | H、W | 原Read query投影 |
| GDN/SSD的state Read | H、Z、W，各一次 | 当前query；短期summary仍由Z/W得到 |
| 四记忆的Full | H | GDN query及原输出门；SSD query、value、gate |

`query_batch.effective(..., full=...)`和C++ `query::effective`按此表返回内部结果；
未消费的内部槽位是None/undefined，不能用它替代完整raw state。
所有模式都先验证查询时间，再用int64相减计算`max(0,t-s)`，之后才转换浮点衰减。
负查询时间拒绝，超过`2**54`的非对齐逻辑时钟不压紧成接收计数。

GDN/SSD的`features(..., full=...)`保留完整输入投影、完整卷积raw向量以及现有
逐缓存/查询校验。只删掉未消费的GDN key归一化、SSD key扩展及delta softplus。
SSD Full仍消费卷积value与投影gate；GDN Full的输出门是原有独立参数/归一化分支。
SSD切片前检查完整projection输出宽度`heads*value_width+conv_width+heads`，并检查
`delta_bias`存在且形状为`(heads,)`；这些检查只读取元数据，保持该未消费参数的None梯度。
矩阵更新继续调用独立scalar/packed更新函数，未采用查询的裁剪特征。

固定时间卷积仍按原始时间查找tap。空缓存不连接历史值；非空缓存没有任何匹配tap时，
历史值的梯度是已连接的零。未使用的key通道中的非有限raw/cache值也必须经过原校验，
不能因为其后不再归一化或扩展就放过非法输入。

Read和Full只读各自绑定快照，不合并它们的归一化分支，不改参数共享与私有状态。
删除unused计算后，未消费的raw字段仍保持None梯度；完整参数/输入所有权的版本守卫继续生效。
裁剪只依赖用途和静态配置，数值零或零混合系数不会触发额外删除，已连接的零梯度仍保留。
公共字段、逐槽Full载荷、部分/重复根与共享参数贡献仍按[查询VJP](weighted-query-vjp.md)验证。

实现位置：Python `weighted/query_batch.py`、`weighted/full_batch.py`；
C++ `weighted_query_recurrent.cpp`、`weighted_full_batch.cpp`及内部`weighted_query_batch.h`。
`test_weighted_recurrent_query_scope.py`检查完整字段、独立状态/输入/参数根、跨Read/Full参数别名、
None/连接零、负时间、空/无匹配卷积、int64大时间、unused通道守卫与版本修改反例。
它还观察每个实际raw字段的访问次数、完整projection尺寸、query归一化/扩展次数，
确认删除的是实际计算。独立公式仍见`test_weighted_recurrent_formulas.py`，scalar实现保持独立。

这些局部检查不构成吞吐结论，也不把CPU检查算作native/设备资格。
[554f2f5固定资格](evidence/interface-heads-qualification-20261110.md)分别记录CPU/NPU、
安装消费者、独立LibTorch两种新增组合的完整训练与新进程精确恢复。
