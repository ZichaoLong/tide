# 带权 Aggregate 的精确计算复用

固定 [architecture §2.2](upstream/tide-core-5/local-semantics/weighted-message/architecture-learning-note.md#22-来源函数与参数共享)
与 [FC §6.2](upstream/tide-core-5/instances/fully-connected-layered-weighted-message.md#62-共享计算与宽度成本)。
参数别名由 [SourceBinding](weighted-source-binding.md) 构造；本项复用实际相同函数和输入的计算。
各节点自己的 Upd、Read、状态、选择、Full 和逐边/端口记录仍独立存在。

packed 调度在当前已合法就绪的块内比较完整纯函数键：alpha/epsilon、按实际原子次序的
类别、载荷 Tensor 身份，以及这些类别实际使用的 A/B/embedding/适配因子 Parameter 身份。
只在键完全相同的请求间计算一次 Aggregate，其他请求引用候选实际算出的 `(r,u)`。
它不比较浮点值、不调用参考递归、不跨执行块缓存，不预跑真实轨迹。
数值相等但独立的 Tensor 根不能复用；未使用类别的参数不成为虚构依赖。
当前判据是充分条件，原子排列不同或独立构造的同值输入可以保留重复计算。

`AggregateProgram.sharing_key` / `AggregateKernel::sharing_key` 是显式能力；普通程序默认无能力。
带权 Aggregate 无状态、时间无关且不产生 per-source contributions，满足上述判据。
别的程序若声明此能力，键必须保证完整返回和求导依赖等同；一般引擎不推断其函数独立性。
unpacked 标量路径保持逐请求计算，reference 调度也保持独立执行。

`layered_inputs` 在同一样本/位置的各入口使用一个真实广播载荷 Tensor；它仍输出各个物理端口记录。
带权绑定已经验证不可训练的单位 transport scale，候选发送保留同一 Full 载荷的广播身份。
跨设备实际复制产生新的身份时，判据自然保守；没有依据 CPU 轨迹产生设备缓存。
pending/checkpoint 继续编码实际原子与原始状态，不保存此执行期缓存；恢复可重新建立复用关系。

反向图只有真正共享的计算结点，每个使用者的下游边分别贡献 cotangent，autograd 汇总所有贡献。
物理 SourceInput/atom/port 身份按本请求解码，不能借用代表请求的来源槽。
共享 Aggregate 结果不合并节点的记忆写入或参数适配，更不合并私有 checkpoint 状态。
这与“两个独立 Tensor 数值恰好相等而合并它们的梯度根”不同。

| 入口 | 实现与检查 |
| --- | --- |
| 函数键 | [Python](../python/tidegraph/weighted/sharing.py)、[C++](../cpp/src/weighted_sharing.h) |
| 当前合法块的代表请求/解码 | [Python](../python/tidegraph/aggregate_sharing.py)、[C++](../cpp/src/aggregate_sharing.cpp)；独立接入 Python blocks 与 C++ block/stream |
| 完整记录与根 | [tests](../tests/test_weighted_compute_sharing.py)：四种参数绑定、双样本/多位置 FC、独立根和连接零、同值独立输入拒绝复用、私有状态 |
| 训练与恢复 | 同上：三步 AdamW、共享使用者参数梯度、含 pending checkpoint、换 owner 后下一更新 |
| 独立原生使用者 | [standalone](../cpp/test/weighted_sharing.cpp)：实际一个代表请求、两份物理来源、解析双使用者导数、独立同值根无复用 |

`aggregate_computed_events` 数实际求值的代表纤维，`aggregate_reused_events` 数复用的纤维，
两者相加仍等于候选数。`aggregate_calls` 数实际 batch/scalar 调用；既有 batched-autograd
事件计数描述能力覆盖的逻辑事件数。source rows、候选写入、Full、消息数量不因此减少。
本项不是跨节点共享 KV base 投影，也不认证 mixed/resident；这些继续单独实施。
当前实现/资格状态见 [STATUS](STATUS.md)，未构建路径不宣称通过。
