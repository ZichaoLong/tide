# 来源函数参数绑定

固定 [architecture §2.2](upstream/tide-core-5/local-semantics/weighted-message/architecture-learning-note.md#22-来源函数与参数共享)
式 (4) 定义 `F(v,a)(z)=B(v,a) SiLU(A(v,a) N(z)+e(v,a))`。
输入和输出均为内容空间 R^d；归一化 epsilon 沿用节点记忆配置。
本模块无状态，不读取时间、候选状态或区域选择。Aggregate 对每个物理原子调用该函数，
之后按质量归约为 `(r,u)`；同类原子不能合并为一个原子。

公共入口：`bind(..., source_binding=SourceBinding(mode, rank), source_hidden_width=...)`。
默认 `SourceBinding()` 为 region，保持此前默认参数初始化顺序和 graph/profile 身份。
其他模式、rank 和类别宽度进入 aggregation profile；checkpoint 继续用 v5 并校验参数别名图。

| mode | A/B | embedding | 额外因子 |
| --- | --- | --- | --- |
| region | 每区域一份 | 每 `(region,class)` 一份 | 无 |
| node | 每节点一份 | 每 `(node,class)` 一份 | 无 |
| source | 每 `(region,class)` 一份 | 同左 | 无 |
| adapted | 每区域一份 base | 每 `(node,class)` 一份 | 每 `(node,class)` 私有 LA/RA/LB/RB |

adapted 的实际矩阵为 `A=baseA+LA@RA`、`B=baseB+LB@RB`，明确指定正整数 rank r；
四因子形状为 `(p_a,r),(r,d),(d,r),(r,p_a)`。其他模式拒绝 rank。
各参数有限，独立新参数沿用标准差 0.15 的显式 seed 初始化；初始因子无需为零。
`source_hidden_width` 接受统一正整数或覆盖全部实际类别的完整 `{class:p_a}` 映射，
在绑定时复制、排序并冻结。共享 A/B 的同一组必须同宽；不同组可以异宽。
不凭类别不同假定计算不同，也不凭参数共享假定输入相同。

`SourceFactory` 负责构造这些参数别名。调用者仍可明确绑定其他同形参数，
这作为模型的实际别名图保存与检查；加载必须重建相同别名。低秩因子、
KV 适配/摘要及节点状态互相独立。参数共享不改变私有状态、活动集合或消息存在性。

| 定义与依赖 | 代码及实际返回 | 独立检查 |
| --- | --- | --- |
| §2.2 四种绑定/宽度 | [SourceBinding/SourceFactory](../python/tidegraph/weighted/source_binding.py)、[bind/validate_binding](../python/tidegraph/weighted/binding.py)；实际 Parameter 别名 | [参数与宽度检查](../tests/test_weighted_source_binding.py)：同/异组身份、私有状态、不相容宽度/rank/形状拒绝、默认 profile |
| 式 (4) 及低秩矩阵 | `SourceTransform.matrices/forward`；[C++ source_matrices](../cpp/src/weighted_source.h) | 同上：一维独立手算及 A/B/e、四因子、输入的解析导数；[standalone](../cpp/test/weighted_source.cpp) 独立手算及形状拒绝 |
| §2 式 (3) 每原子归约 | [scalar](../python/tidegraph/weighted/programs.py)、[batch](../python/tidegraph/weighted/aggregate_batch.py)、[native](../cpp/src/weighted_aggregate.cpp)；真实类别宽度分桶，直接返回实际 `(r,u)` | [执行测试](../tests/test_weighted_source_execution.py)：双样本 HB/反馈 PDG、完整记录、独立根、None/连接零、保留图续接 |
| 共享参数反向/训练/恢复 | [Aggregate VJP](weighted-aggregate-batch.md) 捕获完整参数树、实际分组私有图；checkpoint 保存共享参数及原始状态/pending | [训练](../tests/test_weighted_source_training.py)：三步 AdamW、优化器矩、含 pending 换 owner 恢复与下一更新、异绑定/异别名拒绝 |

局部手算不依赖两种调度互比；图测试核验调度与共享根汇总，不能单独认证数学公式。
standalone 在本增量认证局部公式/梯度，完整图训练矩阵继续单独补齐。
CPU FP64/FP32 固定实现 503f609 的 [665 项资格](evidence/core5-source-20261027.md) 已通过；
本项没有声称已经复用相同参数/输入的计算，或认证 mixed/resident/设备运行。
