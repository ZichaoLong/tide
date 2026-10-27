# 带权常驻的样本副本与共同更新

本页定义样本副本契约；[固定NPU/CUDA资格](evidence/core5-resident-weighted-peers-20261028.md)
分别记录Python-owned、standalone两dtype与CUDA构建。CUDA真机待验。
数学采用[PDG有限切面](upstream/tide-core-5/positive-delay-graph-finite-cut-learning-note.md)
与[局部架构](upstream/tide-core-5/local-semantics/weighted-message/architecture-learning-note.md)。
每个样本的状态、消息、历史和调度独立；只有模型参数共享。

`WeightedTrainingSession::step_replicas` / Python `WeightedResidentTraining.step_replicas`
消费2至16个不同逻辑设备上的公共owner。每个owner运行完整图，局部sample ID属于该owner；
下游负责将独立样本分组和保存这一静态映射。每张卡的窗口内就绪、选择、合批及推进仍在设备上完成。
同一样本内不切分Attention、候选区域或原始时间窗口。这是样本并行接口；不宣称已提供单样本的跨卡节点切分。

调用次序为各owner的advance、完整backward或显式accumulate，然后一次step_replicas。
参数梯度按传入owner顺序求和，连接位为逻辑或；不除以卡数，不改变用户loss的归约。
参数/别名/optimizer配置与generation必须一致，私有状态和时间切面可不同。
参数的所有使用者贡献汇入同一个更新；各卡随后保留相同master、槽和实际量化参数。
FP16存储与FP32算术/伴随/master继承[公共训练契约](resident-weighted-training.md)。

共同更新程序使用既有PeerExchange设备通知与设备到设备复制。完整FP32 master、动量、
二阶槽、maximum、bias correction和int64 steps先逐字节核对；不一致为error25。
全部卡先计算proposal，经设备一致的error决策后才能commit，有限性/half可表示性拒绝仍为20/21。
拒绝时所有master/槽/bank与generation不变，原梯度仍可由调用者显式detach。
运行异常使参与owner全部失效，须从先前切面创建新owner；不返回部分提交的成功结果。

`communication_bytes`在创建复制缓冲区前准入，额外保留检查覆盖程序实际引用；各owner原有
forward/retained/backward/optimizer预算继续有效。timeout显式限制在1至60000毫秒。
每个设备程序全部提交后才等待，peer包释放在程序完成之后。没有CPU梯度求和或主机逐事件控制。
返回每卡实际总梯度与连接位、共同step结果、传输字节和保守张量预留；调用者长期保留返回张量由调用者预算。

checkpoint仍为每个owner的精确公共切面，包含全部私有raw/pending/ledger和共同optimizer状态。
分别恢复后，下一次共同更新重新核验optimizer状态；恢复不依赖旧进程的peer句柄。
仅映射输出或有效状态不能代替这些精确切面。该接口不自动定义数据游标、RNG或任务级分布式checkpoint事务。

独立验收要求：各卡完整记录/私有根对独立CPU调度，公共总梯度对所有样本的梯度和；
SGD与AdamW多步、FP32/FP16、原始状态续接、恢复、None/零与不一致/容量拒绝。
`test_resident_weighted_peer_identity.py`单独锚定int64大值、signed zero、细小FP32与half次正规差异。
通信组件资格不能代替公共多卡训练资格；NPU真机、Python-owned、standalone与CUDA构建/真机分别记录。
