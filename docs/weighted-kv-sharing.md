# KV 基座投影计算复用

固定 [KV §1.2](upstream/tide-core-5/local-semantics/weighted-message/kv-memory-learning-note.md#12-写入投影与当前特征)
式 (2) 和 [FC §6.2](upstream/tide-core-5/instances/fully-connected-layered-weighted-message.md#62-共享计算与宽度成本)
允许同一基座、相同输入的投影共同求值。来源参数别名、[Aggregate 复用](weighted-compute-sharing.md)
和本项是三个独立层次；私有状态始终保留。

当前合法 packed 块完成 Aggregate 后，State 程序可显式声明 `preparation_key` 与
`prepare_shared`。默认无能力。KV 的充分判据为正接收分支、完整 Content Tensor 身份、
实际 base Parameter 身份和 epsilon 相同；不比较浮点值，不跨块缓存，不读取参考轨迹。
至少两个请求满足判据时计算一次 `x=N(u)` 和 `b=base@x`，形成临时具名张量 `{normal,base}`。
逐节点继续计算 `b+(adapter_u@adapter_v)@x` 和 `summary@x`，按实际逻辑时间写入自己的缓存。
无复用的行保留 `(base+adapter_u@adapter_v)@N(u)` 求值；两式是同一数学函数的不同算术分组，
FP32/FP64 仍使用既定容差与严格离散路由检查。没有为复用修改 dtype、输入或平票规则。

准备结果不是新的数学 Content 字段、持久状态或 checkpoint 记录。
它们由当前事件私有地携带，经 `PackedSequence.prepared` 传入精确前缀与 State VJP；
解码只使用实际投影，不调用参考递归。退出块前移除事件中的准备映射，
autograd 保留反向所需引用；checkpoint 仍只编码原始条目、时间、权重及 pending。
零接收保持完整旧状态，包括已过期条目；不计算新投影。窗口和 int64 时间不变。

State 批量 VJP 把准备张量作为显式独立操作数，保存原始版本，按实际准备字段结构分组。
局部适配器/summary 贡献由各自私有图返回；共享投影图累加全部使用者的 cotangent。
未使用的准备字段返回 None，连接零不被数值屏蔽；独立同值输入与独立同值参数不合并。
共享基座、适配器或输入的实际别名由 autograd 汇总；不通过再次计算完整基座来补写反向。
本接口延续明确的一阶限制，不能对准备图求高阶导数来绕过 State VJP 边界。

| 定义/入口 | 实现与独立检查 |
| --- | --- |
| 相同函数/输入与作用域 | [Python plan](../python/tidegraph/state_preparation.py)、[C++ plan](../cpp/src/state_preparation.cpp)；[Python KV](../python/tidegraph/weighted/kv_projection.py)、[C++ KV](../cpp/src/weighted_kv.cpp)；独立基座/输入拒绝复用 |
| 式 (2) 和逐节点私有返回 | [Python projection](../python/tidegraph/weighted/kv_projection.py)、[C++ projection](../cpp/src/weighted_kv_projection.cpp)；[手算及完整根](../tests/test_weighted_kv_sharing.py)、[standalone 手算 Jacobian](../cpp/test/weighted_kv_sharing.cpp) |
| State VJP 操作数/原位版本 | [Python](../python/tidegraph/state_vjp.py)、[C++](../cpp/src/state_vjp.cpp)；独立 keys/values/summary/weight 根、None/连接零、所有使用者贡献 |
| 在线与续接 | 上述测试覆盖 streaming/frontier/greedy、推理和保留图切点；[训练与 checkpoint](../tests/test_weighted_compute_sharing.py)覆盖三步 AdamW、含 pending 换 owner 恢复及下一次更新 |

`state_preparation_calls` 数实际共同准备调用，`state_preparation_events` 数其覆盖的事件，
`state_preparation_reused_events` 数省去的重复准备；后两者之差等于前者。
准备按独立键分别计算，节点内适配/写入仍批量执行；没有声称跨键合并了所有投影调用。
候选写入、私有 KV 长度、选择、Full 和消息数量均不因本项减少。
本项是 eager packed 路径，mixed/resident 与真实设备资格单独验收；当前运行证据见 [STATUS](STATUS.md)。
