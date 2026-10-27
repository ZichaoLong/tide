# 单样本的常驻节点放置

采用固定[core-5与带权定义](semantics.md)。`WeightedFlow`的可选`node_devices`按节点编号
指定一个同后端、显式逻辑设备；空C++列表或Python的None保持全部节点在coordinator。
设备可重复、可以没有节点放在coordinator；区域、队列、history与完整返回journal仍由
coordinator拥有。这是同一图/样本内的物理放置，和[独立样本副本](resident-weighted-peers.md)不同。
当前新增前向/精确续接实现等待资格；放置后的retained VJP与公共训练接入仍在实施中。

`weighted_flow_peers.cpp::WeightedFlowPeers`为每个远端设备建立一个服务组。
节点实际参数bank、完整原始状态、Upd/Read/Full及局部数值tape在指定owner上。
`weighted_flow_views.h::WeightedFlowView`仅保存候选实际算出的区域输入和返回字段；
本地字段使用原计算句柄，远端字段通过设备packet复制。它不执行参考递归、不重算状态。
为保持完整可观测返回，目前每次作用的old/proposal字段都传回coordinator的journal；
此实现不宣称最少通信或最小HBM占用。

窗口内协议为：

1. 设备服务准备来源参数，并发送窗口左切面的真实int64元数据与错误状态。
2. coordinator从真实pending队列决定在线就绪与节点批，发送每个节点的完整输入纤维。
3. owner完成Aggregate、Upd、对应初态/旧态/候选态的Read，返回完整描述与raw快照。
4. coordinator对每个完整区域执行一次SelStep；Attention可见性、归一化与稳定平票
   保持全区域，随后向每个owner发送该节点的control与active。
5. owner在候选快照上计算Full，返回未重复乘强度的实际`(c,G)`；coordinator保持物理边与端口
   身份路由，并完成队列容量预检。
6. 所有数值/容量检查均通过后，同一错误字决定队列、所有节点状态和history的提交。
   未激活候选仍采用更新；出现拒绝时不采用该阶段，整个owner标记failed。
   空窗或错误退出都发送stop；所有远端服务与coordinator先提交再等待。

所有阶段的选择、分支、P2P请求/响应均由device programs推进；主机只在窗口边界
提交/等待/导出，没有逐事件标量轮询。固定packet和队列容量限制同时在途数据；
逻辑时间与逐条时间戳仍为int64。公共checkpoint解码实际私有owner状态、history、pending
和输入账本，可在另一份合法节点放置下恢复，物理放置不进入数学graph identity。
参数共享仍按CPU模型的canonical别名记录；每个owner的物理bank副本不是新的数学参数。
跨owner反向贡献归约与共同optimizer发布必须另行接通和验收，不从前向结果外推。

预算统计跨程序唯一tensor storage、完整journal、peer镜像与packet、各owner数值workspace。
显式设备、长度/后端/索引、无法访问的peer及容量不满足均拒绝；没有CPU后备。
边界输入校验失败可重试；执行错误必须从早先checkpoint构造新owner。

入口：`include/tide/resident_weighted.h::WeightedFlow`（六参数构造），
`tidegraph.weighted.WeightedResident(..., node_devices=[...])`；manifest分别记录请求与生效列表。
实现：`weighted_flow.cpp`、`weighted_flow_peers.cpp`、`weighted_flow_region.cpp`及
`weighted_flow_records.cpp`。数学对应沿[完整常驻图](resident-weighted-flow.md)到各局部函数。

独立验证：`weighted_flow_peer_check.cpp`使用独立CPU调度核对五记忆、同秩HB、非对齐DAG、
反馈PDG、完整区域Attention/来源变换、零接收、重复边/端口、全部记录、跨放置磁盘恢复及空窗。
容量反例还直接检查每个物理节点的raw状态和presence，证明拒绝阶段没有部分采用。
`test_weighted_resident_placement.py`只验证静态owner定义域，不作设备执行证据。
各dtype、Python适配、standalone、CUDA构建与真机结果仍分别记录。
