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
| 式 (4) 跨来源列与精确复用 | [Python source列](../python/tidegraph/weighted/column_sources.py)、[native分组](../cpp/src/weighted_column_sources.cpp)/[显式VJP](../cpp/src/weighted_column_source_ops.cpp)；[Python区域聚合](../python/tidegraph/weighted/column_aggregate.py)、[native区域聚合](../cpp/src/weighted_column_aggregate.cpp) | [独立公式/别名/版本](../tests/test_column_sources.py)、[非空异宽来源及独立调度](../tests/test_column_source_schedule.py)、[standalone source组](../cpp/test/weighted_sequence_columns.cpp)；实际资格按固定实现另记 |
| 共享参数反向/训练/恢复 | [Aggregate VJP](weighted-aggregate-batch.md) 捕获完整参数树、实际分组私有图；checkpoint 保存共享参数及原始状态/pending | [训练](../tests/test_weighted_source_training.py)：三步 AdamW、优化器矩、含 pending 换 owner 恢复与下一更新、异绑定/异别名拒绝 |

局部手算不依赖两种调度互比；图测试核验调度与共享根汇总，不能单独认证数学公式。
standalone 在本增量认证局部公式/梯度，完整图训练矩阵继续单独补齐。
CPU FP64/FP32 固定实现 a5b2f42 的 [665 项资格](evidence/core5-source-20261027.md) 已通过；
该历史资格不覆盖后继列缓存、mixed/resident或新设备运行。

## 张量列、精确共享和一阶转置

默认 Settle 列路径在每个区域收集实际来源列，按行数、内容/隐藏/低秩尺寸、device、dtype、
alpha、epsilon 分组。一次批量归一化、SiLU 和两次 shared GEMM/private BMM 计算
`z+alpha F(z)`；浮点打包/拆包不进入 autograd 图。每个独立来源输出有自己的伴随槽，
反向只处理有伴随的输出，未观察的根为 None，观察到的零伴随仍连接。
实际原始输入、A/B/embedding和四个低秩因子的版本均保存，使用者修改后明确拒绝。
低秩转置为 `dLA=dA RAᵀ`、`dRA=LAᵀ dA` 等；真实参数别名汇总全部数学槽贡献。

计算复用只认输入Tensor身份及完整有序参数槽身份相同，不能用值相等、数据指针、
SrcClass相同或去重后的`Module.parameters()`列表代替。后者可能遗漏低秩槽的不同别名排列。
整Aggregate复用还要求有序物理来源的strength/content根和int64行域相同；来源出现次数
保留在键中，每条边/端口仍贡献自己的质量。两种缓存都只存活于一次区域调用，不跨窗保存。
共享输出通过稳定使用顺序的只读视图归并伴随，避免跨owner的到达次序选择累加顺序。

`max_workspace_bytes`约束附加批量暂存与伴随组的保守估算，不是模型/整窗保留图的总内存预算。
相容组按预算切开，单列超出附加打包预算时使用原独立公式，不截断输入或静默更换dtype。
缓存不进入checkpoint，后继窗口从实际raw状态、相同参数别名和新输入重建。一般记录执行器
保持独立路径；旧内部`aggregate_columns(..., cache=...)`消费者应改用一次区域的
`aggregate_columns_many`，不携带跨窗口外部缓存。
