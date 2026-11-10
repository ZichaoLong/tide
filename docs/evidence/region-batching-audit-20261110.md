# 区域分配、评分与批次接口审阅（2026-11-10）

审阅源码固定`06b242eef2a203a950e30ced9b50d5c6d1cb8a73`，数学固定`0770593`。
这是源码/接口审阅及小型CPU算子计数，不是全模型、设备或吞吐资格。
实际文件与符号声明见[hash绑定manifest](region-batching-audit-20261110-reviews.json)。
当前待办仍由[ROADMAP](../ROADMAP.md)管理；此处不建立第二个任务清单。

## review-scope

完整读取区域batch/VJP/元数据、批量分配求解及精确映射VJP、正则响应、发送族、
标量/批量评分与Describe分派、私有列packing及相关局部测试。`programs.py`只声明
实际审阅的Describe和RegionalTotal方法；C++函数为人工具名定位，未运行Clang枚举，
不把读过文件换算为全部C++函数数量。native标量分配、resident算子、设备调度和跨卡
不在本次完整审阅范围。沿用三项[仓库技能](../../.agents/skills/maintain-tide-foundation/SKILL.md)。

## 已有机制和数学边界

设R为当前就绪的B×L区域帧数，N为每帧完整真实候选数。`region_batch.evaluate` /
`weighted::batch::regions`按N/dtype/device分组，Query和context head在R上做矩阵批，
每一行仍对应同sample/region/time的完整候选。没有把候选拆成分别归一化的子问题。
`PrivateRows`把同一候选字段按R合并为私有叶，外层按cotangent定义模式和KKT分支
回收参与帧，保留caps分支无score根、正常分支连接零、未参与帧None及共享参数全部贡献。

`allocation_batch.solve`求解固定教材§6.1的一维乘子，实际控制是
`c_v=g(t_v-lambda,mu_v;tau)`。Python quad排序断点并校正实际响应；native quad使用
至多80轮有界二分和仿射校正。Python bern/ent使用受保护Newton，native独立二分。
所有迭代同时处理R行，均在no-grad数值域。`allocation_rows._Rows.backward` /
`AllocationVjp::apply`按§6.2实际响应解析计算VJP，未对有限迭代轨迹求导。
实际居中响应、共同阈值连接零、端点约定、严格正支持和失败快照是优化必须保留的边界。

`sending_batch._Sorted` / `Sorted`中的零温循环沿N推进，对R同时计算；这不是逐B×L执行。
顺序减remaining影响浮点边界的消息存在性，不能直接换成cumsum后称同函数完整等价。
稳定平票使用int64节点ID，fractional `regional_total_zero`不等于硬Top-k。

## 具体发现与局部修复

| 固定源码位置 | 观察与影响 | 后继修复/验证边界 |
| --- | --- | --- |
| Python `region_batch.evaluate`、native `batch::regions` | 所有发送族都创建nodes；native每帧先创建一个设备Tensor再stack，即使quad完全不用节点ID | 只在排序族构造；native一次建立整组int64数组。稳定ID排序、重复/非法ID拒绝保持 |
| Python `_Region.forward`与外层`region_vjp.evaluate` | 数值层已经读取active；私有VJP丢弃该集合，发布后再次逐控制读支持。native原本复用数值结果 | 将同一实际active随VJP发布携带。payload转换后的支持核对另有职责，继续保留 |
| native `batch::regions`→`allocate_batch` | `scores[j]/caps[j]/low[j]/high[j]/shift[j]`每行建立SelectBackward；反向每次分配完整原字段并填零 | 用各字段的一次unbind保留行次序和独立VJP输入；None/零及分支选择不改。组内配置temperature共用一个常量 |
| Python/native Region VJP结果发布 | 历史接口为每控制创建独立可变存储，即使只是只读中间结果 | 按本轮统一软件契约迁移为只读detach view；独立根、原refs版本检查保留。调用者需修改时显式clone；迁移不是固定数学变更 |

新回归[`test_region_batch_publication`](../../tests/test_region_batch_publication.py)直接观察ATen
int64数组构造、host scalar读取和真实SelectBackward，不只统计包装函数调用。
2/17帧包括超过2**54的节点ID、稳定平票、固定总量、proportional、硬发送及分数填充；
与独立scalar局部求值比较部分/零control根、未使用帧及完整损失梯度。

Python工作树开发检查：新回归加原`test_weighted_region_batch`，FP64/FP32共60 passed。
这是开发结果，后继固定实现的native/设备/安装消费者资格另记，不能由本报告提前认证。

## 固定native小探针

使用已有匹配构建，未重建、未启用设备。其manifest源码为上述06b242e；
Python-owned NPU构建实际运行CPU张量，Torch2.12.0+cpu/aarch64，ATen线程1。
`_tide_native.so` SHA256为
`9d8de4701a36b28f5d7c49789168704c68a8a56d37ef970cef6d90e63f1ba40d`。
使用新测试fixture的regional_total/quad、N=3、FP64；全部控制平方加权损失真实反向。

| R | node-ID int64数组构造 | 完整[R,3] SelectBackward | 完整[R] SelectBackward | 独立scalar根核对 |
| ---: | ---: | ---: | ---: | --- |
| 2 | 2 | 4 | 2 | passed |
| 17 | 17 | 34 | 17 | passed |

这里的SelectBackward是实际反向ATen调用；每次都以完整字段尺寸为目标。
它解释可消除的装配与提交，不提供wall-time收益。新测试对修复实现要求排序ID数组构造
每组一次、非排序零次，且不再出现这些完整字段的逐行SelectBackward。
复现新门禁：在匹配构建/环境中运行
`TIDE_BUILD_DIR=MATCHING_BUILD python -m pytest -q tests/test_region_batch_publication.py tests/test_weighted_region_batch.py --dtype both`。
旧实现可用同一fixture和`MetadataCalls`单独观察上述计数；不能把旧失败门禁重标为新实现通过。

## 仍有价值的独立候选

- `allocation_batch.solve`及native同函数的输入有限性/定义域检查已经按R合批，但仍按字段多次
  host读bool。可在同一依赖阶段合并detached检查标志并保持首个错误类别、失败快照和全组拒绝；
  它们随组而非R增长，不应误报成逐token求解。
- Python `entropy`每次迭代读`live.any()`；native每8轮读收敛标志。Python可保持每行live冻结、
  有界分段读取收敛结果，减少设备等待；须先对实际响应、残差、最小正支持及端点作逐值核对。
- `allocation_rows`在内层公开行接口上又stack输入、逐行clone输出，随后Region VJP再次发布。
  内部数值批与公开独立行接口可分离，避免强迫通用内部路径承担公开可变对象的成本；
  零/caps/normal行、可训练bounds/temperature/shift、部分根和原版本守卫必须独立验收。
- 固定基线的`read_batch.evaluate` / `weighted_describe_batch::evaluate`仅对
  cat/ind/regional_total/context/cat启用整批Describe，其他合法组合仍逐请求投影。
  Python Encoder/MatchScore/GroupedEvidence/Query本身已有leading-axis公式；可扩展描述批接口，
  不必枚举整个模型组合。需要修正key/evidence中的固定轴索引，明确FP32舍入与完整字段根。
- score/evidence/key显式共享同一Encoder时，当前仍重复求值。可用调用内缓存复用相同完整
  参数绑定、epsilon和输入身份的编码；同值不同根、只共享部分参数、不同Read快照不能合并。
  所有使用者cotangent须汇总，不能从参数共享直接推定计算共享。

成熟经验的适用顺序是先修边界重复工作、保持实际批次，再比较新的求解实现或融合。
历史[FC16归因](fc16-cost-attribution-20261110.md)中NPU region gradient约3.43秒、
前向分配约0.21秒，不能因求解算法复杂而忽略伴随装配。新吞吐结论仍须完整训练计时。
