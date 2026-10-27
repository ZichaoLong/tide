# Weighted 实际窗口反向

本页描述正在集成的内部后端接口 `WeightedWindowAdjoint`，实现/运行状态以STATUS为准。
公共retained token、optimizer发布和训练checkpoint属于后续owner层，不由本页推断完成。
数学固定到[局部架构](upstream/tide-core-5/local-semantics/weighted-message/architecture-learning-note.md)
及[PDG有限切面](upstream/tide-core-5/positive-delay-graph-finite-cut-learning-note.md)。

前向journal保存实际fiber rows/offsets、原始int64元数据、所有局部值及窗口左侧pending身份和
私有状态存在位。`weighted_flow_budget.cpp`在分配前计入新增journal和边界缓冲区。
`WeightedWindowJournal::load`以设备stage索引读取；不解码为CPU参考递归。

`weighted_window_vjp.cpp`构造一条设备逆序循环：从实际stage count减一推进，加载原始旧态与完整
纤维、以同一参数快照重算局部tape，组合[阶段VJP](resident-weighted-stage-vjp.md)，再按
[消息祖先](resident-weighted-reverse-route.md)传回生产者或独立窗口边界。
raw状态梯度沿每节点/样本的阶段前驱传递；缺席行保持identity。Read^0绑定常量及原本不存在的
左侧初态不导出为用户checkpoint叶子。参数角色每阶段归约并按canonical owner累加；全部角色先校验名称/形状，再按实际trainable集合选择，
冻结部分或全部参数仍保留输入和私有状态的VJP。

内部根是候选实际journal布局中的独立FP32值/bool连接位：Content、control、未缩放G、每个
Describe字段、old/proposal/final raw字段、物理fiber/message及右侧pending。根存储与形状固定，
调用者仅填入实际存在记录的cotangent，ragged padding置零；记录/端口别名的公共根绑定需另层汇总。
所有窗口内加载、反向推进及祖先路由都留在设备；主机仅构造、边界种根、提交和检查结果。

反向默认沿用forward的总tensor预算，也可显式传入另一个正字节数。该预算合计仍存活的forward、
反向根、原始状态、参数bank与程序持有的设备张量；workspace另受operator预算约束。
初始化时预检确定的根/边界/状态/参数开销，构造各数值阶段时按去重storage继续核算，
包括未被当前Read模式使用但owner仍持有的绑定初态。容量拒绝不执行反向，也不消耗成功的前向窗口。

空窗口是已存在raw状态和pending的恒等映射；同一个成功窗口允许不同根的独立VJP。
后续advance改变generation，旧反向owner明确拒绝；错误使整个反向owner失效。
这还不允许参数更新、跨窗口retained token或完整训练checkpoint。

表示与数学对象的对应如下；S为journal容量、B为样本数、Q为物理队列容量，W为含强度的payload宽度。

| 对象 | 实际表示和反向入口 | 适用条件 |
| --- | --- | --- |
| 作用次序与完整纤维 | `WeightedWindowJournal`的counts、fiber.rows/offsets、[S,Q,6]精确坐标和present | 只用候选实际完成的stage前缀；没有主机提供的事件次序 |
| Agg、SelStep、Full的公开浮点量 | `WeightedWindowNodeRoots::{content,control,full,describe}`，值[S,B,D]及独立[S,B]bool连接位 | full是未缩放G；control独立表示c；未活动Full不能种根 |
| Upd的旧态、候选态与右切面 | `old/proposal/final`按`WeightedRawLayout`浮点字段打包，连接位按字段保存 | int64时钟/长度/时间戳无梯度；有效状态投影不能替代raw字段 |
| 逐边消息、收到的原子和右pending | `messages/fibers/pending`按原物理行保留W维值和bool根 | 重复物理边各计一次；未连接数值区允许NaN，不能由零值推断缺席 |
| 左边界与参数 | `states()/boundary()/parameters()` | 左侧不存在状态属于绑定初态常量；参数以真实alias汇总，冻结参数不导出根 |

`weighted_window_case.cpp`使用候选先行的完整图、其实际checkpoint及独立CPU Greedy调度，比较
完整前向与所有参数/输入/左raw/pending的连接位和梯度。`weighted_window_compare.cpp`将实际journal
根按六元物理坐标绑定到独立参考的返回；参考不参与候选解码或执行。局部数学独立公式沿
[阶段VJP](resident-weighted-stage-vjp.md)、[消息祖先](resident-weighted-reverse-route.md)及其数值组件追溯。

`weighted_window_check.cpp`定义12个异构图/Read组合：FC/HB、非对齐TimedDAG、PDG反馈/重复边、
五记忆、三Read、全部或部分参数冻结。`weighted_window_config_check.cpp`另定义15个正式KV qry组合，
覆盖十种发送、固定/上下文控制、cat/match、ind/att、grouped evidence、两key、三正则和来源变换。
每配置检查none/full/content/control/describe/old/proposal/final/messages/fibers/pending/all这12种根模式，
再检查空窗口恒等及旧generation拒绝；局部字段逐一隔离的锚点由阶段组件提供。
离散支持、int64和None严格相等，窗口VJP浮点容差rtol=3e-5、atol=3e-6，与阶段VJP相同。
检查程序的`--case=TOPOLOGY/READ`只过滤原样用例，单例通过不能替代完整顺序/配置矩阵。
这些测试定义不是已运行资格，正式证据单独提交；已知运行失败保留于STATUS及证据。

NPU层级设备服务将节点/区域组合分别录制，父流通知后等待子流，所有层级共享一个高水位workspace。
嵌套严格LIFO构造，根边界按父→子发送终止通知；每个子服务在正常调用返回后均处于idle。
CUDA继续在同一条件图中表达这些静态层级。此拆分解决CANN单流任务容量，不改局部数学或阶段顺序。

NPU的`weighted_copy`对已校验、不重叠的静态bool目标视图使用连续片段的厂商copy。
布局仅由shape/stride/offset生成，不读取运行时值或压紧事件；公共dtype仍是bool。
这是对连续创建窗口后raw字段连接位ScatterUpdate失败的定向处理，未推断所有厂商scatter都有缺陷。
动态journal索引和数值scatter保持各自实现，CUDA继续用原有索引写入。
`weighted_copy_check.cpp`以独立CPU索引验证18个owner、每个3次提交的布局复制，
包括大于2^54的int64、非连续源/目标、false覆盖true和浮点字段；完整窗口矩阵仍需单独通过。

固定2622e60的NPU/CPU/CUDA构建资格与保留失败见[证据](evidence/core5-resident-weighted-window-20261028.md)；后续保留训练不由此推断完成。
