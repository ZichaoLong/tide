# KV 基座投影计算复用

固定 [KV §1.2](upstream/tide-core-5/local-semantics/weighted-message/kv-memory-learning-note.md#12-写入投影与当前特征)
式 (2) 和 [FC §6.2](upstream/tide-core-5/instances/fully-connected-layered-weighted-message.md#62-共享计算与宽度成本)
允许同一基座、相同输入的投影共同求值。来源参数别名、[Aggregate 复用](weighted-compute-sharing.md)
和本项是三个独立层次；私有状态始终保留。

当前合法 packed 块完成 Aggregate 后，State 程序可显式声明 `preparation_key` 与
`prepare_shared`。默认无能力。KV 的充分判据为正接收分支、完整 Content Tensor 身份、
实际 base Parameter 身份和 epsilon 相同；不比较浮点值，不跨块缓存，不读取参考轨迹。
至少两个请求满足判据时计算一次 `x=N(u)` 和 `b=base@x`，形成临时具名张量 `{normal,base}`。
在同一共享键内按私有矩阵形状/dtype/device 再分组，两个以上同形行使用真正 BMM 计算
`adapter_v@x`、`adapter_u@low` 和 `summary@x`，再加共同 `b`。单行或不同形状组沿用已有路径。
各节点按实际逻辑时间写入自己的缓存。
所有eager行（标量、未共享packed、共享组）都保留低秩形式`base@x+adapter_u@(adapter_v@x)`，
避免反复构造大小`2d×d`的稠密adapter临时矩阵。未共享packed行的归一化结果也供summary使用。
独立调度仍分别执行；[手算完整Jacobian](../tests/test_weighted_kv_sharing.py)、
[多维dense矩阵/字段根](../tests/test_weighted_kv_formulas.py)与显式FP64诊断保留作公式锚点，不能用两条相同错误的执行自证。
这是同一数学函数的算术分组选择，FP32/FP64仍使用既定容差和严格离散检查，不改dtype、输入或平票。
resident目前仍使用稠密组合矩阵/tape，另按实际设备与完整记录验收；本次eager优化不声称已经优化该路径。

准备结果不是新的数学 Content 字段、持久状态或 checkpoint 记录。
它们由当前事件私有地携带，经 `PackedSequence.prepared` 传入精确前缀与 State VJP；
解码只使用实际投影，不调用参考递归。退出块前移除事件中的准备映射，
autograd 保留反向所需引用；checkpoint 仍只编码原始条目、时间、权重及 pending。
零接收保持完整旧状态，包括已过期条目；不计算新投影。窗口和 int64 时间不变。

State 批量 VJP 把准备张量作为显式独立操作数，保存原始版本，按实际准备字段结构分组。
局部适配器/summary 贡献由独立行的一阶 VJP 返回；共享投影图累加全部使用者的 cotangent。
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
`state_private_projection_calls` 数实际私有矩阵批调用，`state_private_projection_rows` 数这些调用的总行数，
`max_state_private_projection_batch` 取跨执行块最大值（不是累计）。State 回调数继续单独报告；
实际合批的是私有线性映射，私有缓存写入并未合并。没有声称跨键合并了所有投影调用。
候选写入、私有 KV 长度、选择、Full 和消息数量均不因本项减少。
本项是 eager packed 路径，mixed/resident 与真实设备资格单独验收；
[07e56ca资格和有限对照](evidence/core5-private-projection-20261028.md)给出CPU、mixed-c与独立后端证据，
包括未改善的性能和严格数值失败；当前交接见 [STATUS](STATUS.md)。

## 独立私有行的一阶边界

`StateKernel::prepare_private_batch` 是默认原样传递的可选能力；Python 同名方法亦可缺省。
它只处理已确定共同函数/输入的准备组，不改变一般引擎的状态或拓扑表达域。
KV 的 `linear_rows` 接受各行向量及各自矩阵，实际前向为 BMM；反向仅合批有根的行：
`dx=Wᵀg`、`dW=gxᵀ`，没有根的行保持 None，连接零保持零。重复参数/输入身份继续由
外层 autograd 汇总，数值相等的独立根仍分开。原输入/参数全部保存版本，拒绝高阶。
不能直接用裸 stack/BMM 的自动反向替代这个边界，否则独立未用行会变成连接零。

| 定义对应 | 实现/独立检查 |
| --- | --- |
| 私有线性映射与分组 | [Python](../python/tidegraph/weighted/kv_projection_batch.py)、[C++](../cpp/src/weighted_kv_preparation.cpp) |
| 实际 BMM 与独立 VJP | [Python](../python/tidegraph/weighted/row_linear.py)、[C++](../cpp/src/weighted_row_linear.cpp) |
| 手算解/Jacobian、别名、未用行、零根、版本、不同 rank 分组 | [Python tests](../tests/test_weighted_private_projection.py)、[standalone](../cpp/test/weighted_sharing.cpp) |

准备结果 `{values,summaries}` 只携带候选实际计算结果，现有 State VJP 把它们作为独立操作数；
没有参考重算或状态补写。无共享键的私有 base/输入仍走原路径，这一边界不宣称全节点异构合批。
对应性能与后端资格必须由固定源码运行单独给出，单元测试不构成提速证据。
