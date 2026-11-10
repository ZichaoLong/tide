# 带权 Read / Full 的批量数值求值

固定 `0770593` 的 [共同架构 §3–4](upstream/tide-core-5/local-semantics/weighted-message/architecture-learning-note.md)、
[KV §2–4](upstream/tide-core-5/local-semantics/weighted-message/kv-memory-learning-note.md)及
[循环记忆](weighted-recurrent.md)。[评分契约](weighted-scoring.md)给出描述字段与配置适用域。
这是同一局部函数的批量数值候选；不修改活动集合、状态采用、消息存在性或 checkpoint 版本。

## 输入、计算与解码

输入是同一节点程序的一组已经就绪的真实请求。Read 各自持有模式对应的快照：
Read⁰ 显式生成绑定初态，Read⁻ 读旧状态，Read⁺ 读 proposal；Full 只处理已选中候选的
comparison（本支线即 proposal），即使 Read⁰/⁻ 采用其他状态也不替换 Full 的快照。
每行保留内容、原始逻辑时间和私有状态；不存在真实轨迹预跑或按接收次数重编码时间。

| 数学函数 | Python | 独立 C++ | 批量域/布局 |
| --- | --- | --- | --- |
| KV Mem | `weighted/query_batch.py::kv`、`query_packed.py` | `weighted_query_kv.cpp::memory`、`weighted_query_packed.cpp` | legacy按可见长度分组，默认能力路径使用原始槽/gather；每行完整可见K/V/summary/weight，int64 age先减后转浮点 |
| 循环 Mem | `query_batch.py::memory_read,effective` | `weighted_query_recurrent.cpp::memory,effective` | 按实际用途堆叠H/Z/W，原始int64时钟差控制短期/矩阵衰减；见[循环查询](weighted-recurrent-query.md) |
| GDN/SSD 查询与短卷积 | `query_batch.py::features,convolve` | `weighted_query_recurrent.cpp::features,convolve` | 按实际 cache 长度分组，绝对时间槽匹配；查询当前 tap 与带权历史分别处理 |
| Describe | `read_batch.evaluate`、`dense_describe.columns` | `query::describe`、`query::describe_columns` | 默认能力路径合批p_v及完整有限头，leading B/L轴与独立行/字段根；显式legacy保留逐行头 |
| Full | `weighted/full_batch.py::evaluate` | `weighted_full_batch.cpp::full` | KV 多头旋转/距离项，四种循环读出、门/RMS 与 FFN；返回未重复缩放的 `(c,G)` |

KV 查询保留每行全局归一化，按实际正权重项数合批 softmax；无正权重时返回声明的零。
浮点衰减下溢为零的项照实处理，不人为裁剪，不从数值零猜测是否存在原始消息。
无穷窗口仍读全部合法可见项，过期 raw 项仍留在原状态中，查询不写状态。
固定时间卷积空缓存分支不虚构历史；非空缓存缺失 tap 通过精确匹配得到零项。
循环Read/Full只计算各自消费的字段及query特征，保留完整投影与卷积/cache校验；
原始状态始终完整，内部未消费字段不能用作checkpoint解码。具体用途和独立根检查见[循环查询契约](weighted-recurrent-query.md)。

Full 各头可有不同旋转频率和距离斜率，头间循环不改变每头按行合批的事实。
当前输入 u、记忆混合、FFN 的作用顺序不变；GDN 为逐头 gated RMS，SSD 为门后全通道 RMS。
同一行所有逐边/端口输出复用实际 `(c,G)` Tensor，保留槽身份；这里只广播载荷，
不提前投影成 `cG`。所有返回都来自本次候选实际计算，不调用 scalar 函数补写数值。

## 能力与反向边界

Read的Mem与Describe具有独立能力：默认weighted `eager_runner`的packed prefill按
[查询策略](weighted-dense-query.md)选择KV packed-v2及五记忆的完整Describe列计算。
匹配条件下p_v、cat/match、evidence、key与gain共享矩阵按实际就绪行合批，发布边界再拆行；
显式legacy和能力不适用路径保持原逐行描述头。有限归一化梯度舍入见证保留原严格阈值及
具名失败，不用旧逐行运算次序限制当前默认路径。参考scalar调度/局部函数仍独立。
Full的记忆混合/门/RMS/FFN实际按行合批；调用次数、内部图及设备kernel数仍须分别观察。

Read/Full 的 `joint_batch` 声明数值合批，记录中 `read_scalar_batch_steps` 与
`full_scalar_fallback_steps` 为零。形状分桶、索引准备仍在 eager 主机侧，
不把一次 API 调用计作一个设备核，也不认证常驻就绪/选择/推进。

数值批量与反向能力分别验收。[批量 VJP](weighted-query-vjp.md)定义当前 Read 自动采用的
实际私有批量图及`full_autograd="batched"`。weighted `eager_runner`和实验CLI在packed路径
设置batched；底层`Native`/`ExecutionOptions`及直接greedy接口的默认仍是replay，需显式配置。
实际replay统计为`semantic_full_replays`，不能混称这些入口的默认行为。逐字段/状态槽/Full 槽根的 None/零契约不变。
参数共享身份和私有状态保留；复用参数不意味着不同输入/快照可以复用计算。
数值资格 0925539 使用当时的 Read/Full replay，不能冒用为后续批量反向资格。

## 验证与资格

[test_weighted_read_batch](../tests/test_weighted_read_batch.py)覆盖五种记忆、全部适用 Mem 读取、
cat/match 与 grouped/attention 字段，空/旧/过期/初始快照，超过 2**54 的不齐时钟与短卷积。
局部用例禁止数值 batch 调用 scalar Read/Full/Mem/features；异构反馈 PDG 比较完整记录、
每字段独立根及整目标 VJP，并检查实际能力统计。既有 FC/HB/非对齐 DAG、优化器、
checkpoint 与快照分离测试继续回归。

[standalone C++](../cpp/test/weighted.cpp)用显式 SiLU(4/sqrt(2)) 区分 populated/expired/Read⁰，
用 `(0.2,1+3/sqrt(2))` 与 `(0.2,1)` 核对 Full 的有效窗口、载荷和广播别名。
KV/循环的独立手算与 dense recurrence 仍在原公式测试中；这些检查与调度等价各有职责。
开发/不可变资格的实际范围在 STATUS/evidence，源码实现不等于设备真机通过。

## 单行反向边界

一个实际分组只有一行时，Python/C++直接在原参数与该行输入上构造相同数值批图；
`singleton_vjp.boundary` / `SingletonVjp`保留所有原值版本检查、独立字段None/连接零、
只读发布与一阶限制。没有其它行需要隔离，因此不再detach/rebind参数并嵌套调用autograd.grad。
多行仍使用原私有图隔离。`test_weighted_query_vjp`及`test_weighted_aggregate_vjp`
覆盖singleton/多行版本、逐字段根、重复输入、参数别名和独立方向导数；固定资格另列。

默认auto与显式legacy/dense-v1/packed-v2的能力、身份和迁移见[weighted-dense-query](weighted-dense-query.md)。
