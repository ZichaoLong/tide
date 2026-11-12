# Resident KV Read、选择与消息列组合

本边界采用固定 [0770593](upstream.json) 的带权共同架构 §3、§4.2 与 KV §3–4，
接续 [resident 块调度与 Read-selection](resident-block-execution.md)。它是内部数值组合，
不是已接入默认入口的完整图执行器；构建、设备执行和性能资格须分别记录。

## 输入、输出与快照

[`BlockKVMessageInput`](../tools/device_online/block_kv_messages.h) 明确包含两个 typed raw operand：

- `read.read` 是真实绑定的 Read^0、Read^- 或 Read^+ 快照，进入 KV sum/qry Memory 与 Describe。
- `proposal` 是本次 Upd 产生、供 Full 读取的状态。它可以有不同 cache capacity，不能由
  Read 的有效投影或参考递归补造。Read^+ 绑定可以将两者指向同一实际状态；反向仍先返回两条
  独立贡献，由接入 Upd 的 owner 按这个绑定汇总。

两阶段共用同一 `read.content[R,N,d]`，其中 `R=B×L`。Full 使用 selection 保存的原始 int64
时钟和逐节点 active，不复用区域 any 支持。`full_owners[R,N]` 可单独绑定 Full 参数 bank；
省略时沿用 Read 的 node owners。参数共享与每行私有状态、消息支持保持独立。

`append_block_kv_messages` 顺序组合 Read/Describe、完整区域 SelStep、selected KV Full 和
消息列绑定。Whole-window 张量直接交接，没有逐事件 Tensor 输出。它复用既有 vendor 批量
算子与有界元数据算子，没有新增专用设备 kernel，也没有按实际 B×L 轨迹建立服务树。

返回 `BlockMessageColumns` 的 `strength` 是 c，`content` 是未缩放 G，`present` 是实际活动支持，
`times` 是原始逻辑时钟。零 G 仍可是一条存在的消息。缺席行的浮点占位为零，解释必须同时读取
presence。只有显式指定 `numerical_projection=true` 才另算 cG；它不替换消息的 content。
这些是每个源节点的发送列，尚未展开成具有物理边/输出端口身份的记录；重复边广播由后续路由边界处理。

## 一阶伴随与参数身份

`BlockKVMessageRoots` 提供五类独立根：candidate controls、selected Full G、message c、
message G，以及显式的 cG 投影。每个根有自己的 cotangent 和 bool 连接位。
两者同时未定义表示没有该根；缺席或未连接的数值缓冲可以含 NaN。

direct control 根按候选 presence 解释，因此 c=0 的候选不会因未执行 Full 而失去合法的
control 依赖。Full 与消息根按 actual active 解释。多个根使用同一字段时累加其全部贡献；
连接零与 None 分开。cG 的乘法只出现在声明的投影和对应 VJP，避免发送强度重复相乘。

[`append_block_kv_messages_vjp`](../tools/device_online/block_kv_messages_vjp.cpp) 返回独立 Read raw、
proposal raw 和参数角色伴随，以及实际共同 content 输入的合计伴随。保存值来自这一次真实
forward 的 tapes；独立 reverse 程序不重读后来被覆盖的原始 raw、时钟、内容或参数。

[`append_block_kv_message_parameter_vjp`](../tools/device_online/block_kv_message_parameters.cpp)
接受显式的 role→bank-owner 名称表与 `ParameterRegistry`，把 Read、Describe、region head
和 Full 的贡献按实际 TensorImpl alias 身份汇总。可以选择 trainable 子集；已选择的角色必须给全
bank 名称，并满足 owner 形状。若按物理参数选择 trainable 子集，映射须包含这些 owner 的全部
使用角色；仅映射共享参数的一个别名会有意省略其他角色贡献。数值相等的独立参数不会被合并。绑定和更新 bank 的责任属于外层
owner，本接口不从打包后的数值猜测参数身份，也不自动执行 optimizer。
KV sum 没有 Read query/key/value 参数角色或对应连接位；参数发布/反向映射仅枚举真实角色。
Full 的 query 与其他 Attention 参数不因此消失。Mem/total 独立根的值与连接由 typed
Memory tape 分派，sum 不借虚假 query 参数或当前内容建立无关计算图。

## 检查、成本与剩余接口

[`block_kv_messages_check.cpp`](../tools/device_online/block_kv_messages_check.cpp) 使用独立公式：
Read 端沿用两候选 quad 闭式解；Full 端独立计算 proposal Attention、位置/距离项、残差与 SwiGLU。
材料为 R4/N3 的独立快照，Read cache=2、proposal cache=3，包含当前时刻的新条目、不同快照内容、
同秩区域、缺席/零质量、存在且 G=0 的消息、七组独立根、全部 raw/输入/参数梯度，以及原缓冲
覆盖后的 saved reverse。Read.query/Full.query 和 Full.gate/up 真实共享；Full.out 数值相等但
参数身份独立。该检查不调用 candidate 来补写公式结果，不把 Transformer 对照当作等价证明。
R4 可按 B2×L2 解释布局，但这里逐行供应独立 raw 快照，未执行跨时间 Upd；它不证明跨序列
状态隔离、状态 scan 或完整 prefill 正确性。另含 Full/Read 不同合法 owner、默认省略 Full owner、
默认不计算投影及拒绝其根、完全未定义根、半定义根拒绝、共享参数 trainable 子集、坏名称/形状、
三处小预算拒绝与消息时间 int64 精确比较。message c 与 message G 分别作为独立根检查。

当前组合的数值检查源只声明 FP32、KV/qry/cat/ind/context/cat/quad。局部函数既有其他配置
资格不会自动成为本组合的新资格。实际设备结果要关联冻结源码；源码或 CPU 公式检查通过
不表示候选 CANN/CUDA 反向通过。

Read/proposal raw 仍按帧提供，尚未共用 KV 前缀 bank。Full 的活动遮罩决定有效语义、参数使用
和根连接，但当前 vendor 运算以固定容量行执行，仍有 padding 的算术/存储成本；不能只看 active
计数就宣称这些算术全部省去。应同时记录实际活动行和物理行/缓存容量。

完整图还需实际 Aggregate/Upd 与 Next、设备块调度循环集成、逐边/端口路由、独立中间观察根、
跨窗 tape/状态所有权、optimizer、精确 checkpoint 和多卡验收。当前组合的可用性与这些完整
生命周期能力分别报告，不替代 ROADMAP 的未完成项。
