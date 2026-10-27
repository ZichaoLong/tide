# 单样本的常驻节点放置

采用固定[core-5与带权定义](semantics.md)。`WeightedFlow`的可选`node_devices`按节点编号
指定一个同后端、显式逻辑设备；空C++列表或Python的None保持全部节点在coordinator。
设备可重复、可以没有节点放在coordinator；区域、队列、history与完整返回journal仍由
coordinator拥有。这是同一图/样本内的物理放置，和[独立样本副本](resident-weighted-peers.md)不同。
前向/精确续接已有[固定源码资格](evidence/core5-weighted-node-placement-20261028.md)。
放置后的retained VJP与公共训练/optimizer发布已实现，独立固定源码资格状态见[STATUS](STATUS.md)。

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
跨owner反向贡献归约与共同optimizer发布分别验收，不从前向结果外推。

预算统计跨程序唯一tensor storage、完整journal、peer镜像与packet、各owner数值workspace。
显式设备、长度/后端/索引、无法访问的peer及容量不满足均拒绝；没有CPU后备。
边界输入校验失败可重试；执行错误必须从早先checkpoint构造新owner。

入口：`include/tide/resident_weighted.h::WeightedFlow`（六参数构造），
`tidegraph.weighted.WeightedResident(..., node_devices=[...])`；manifest分别记录请求与生效列表。
实现：`weighted_flow.cpp`、`weighted_flow_peers.cpp`、`weighted_flow_region.cpp`及
`weighted_flow_records.cpp`。数学对应沿[完整常驻图](resident-weighted-flow.md)到各局部函数。

独立验证：`weighted_flow_peer_check.cpp`使用独立CPU调度核对五记忆、同秩HB、非对齐DAG、
反馈PDG、完整区域Attention/来源变换、零接收、重复边/端口、全部记录、跨放置磁盘恢复及空窗。
容量反例先用独立CPU调度确认首区域两个跨卡节点同时作用、第一阶段的4条消息超出容量2，
再检查每个物理节点的raw状态/presence及区域history，证明拒绝阶段没有部分采用。
同时断言raw、Full与每个实际参数bank的物理设备，不能仅根据manifest请求列表认证放置。
`test_weighted_resident_placement.py`只验证静态owner定义域，不作设备执行证据。
各dtype、Python适配、standalone、CUDA构建与真机结果仍分别记录。

## 放置后的窗口反向

`WeightedWindowSnapshot`保存每节点的物理owner；在窗口边界把实际右切面raw字段与存在位
复制到coordinator，供公共字段根和跨窗桥使用。原owner的raw存储、参数bank、服务程序和
这些镜像均按storage身份计入预算；复制前检查容量。保留快照克隆全部实际操作数，
关闭原前向后仍可求导；不重新执行参考、不用有效状态补写raw内容。

`WeightedWindowAdjoint`以coordinator上的实际journal逆序推进，装配物理完整纤维并把该阶段
实际old快照传回原owner。各owner重建自身Aggregate/Upd/Read/Full数值tape，完整区域选择
及其VJP仍由coordinator一次完成。`WeightedFlowView::bank`保留每行存在/参数绑定，
`append_weighted_region_views_vjp`不依赖远端未传回的局部tape。

`WeightedFlowPeers::append_reverse`将独立Content、G、proposal与每个Describe根发给owner，
在那里执行局部VJP；返回实际payload/old/initial及具名参数贡献。coordinator保持每节点
滚动raw伴随、逐物理边祖先路由和canonical归约，按稳定节点顺序汇总所有共享使用者。
反向重放不采用私有状态，不改变前向continuation。空窗发送stop；全部设备先提交再等待。
None/零连接位、int64元数据、冻结参数和梯度根身份沿用[窗口VJP](resident-weighted-window-vjp.md)。
每个远端节点的局部VJP及raw/参数归约放入独立的嵌套设备服务，主peer程序只按原顺序
请求并等待；所有阶段仍串行共享workspace。该分段避免把多个节点的完整归约累积到同一
CANN任务缓冲区，不改变区域可见性、梯度归约或窗口边界；CUDA继续使用同一条件图。

新`weighted-node-window`检查定义FP32四组合×12独立根，以及两个三窗桥（含空窗）；
覆盖五记忆、同秩HB、非对齐图、反馈/重复边、Read初态/旧态/候选态、全区域Attention、
来源变换、跨owner共享参数、全部节点远端放置和原前向关闭后的反向。检查真实反向节点的
raw/Full/参数bank设备。独立CPU Greedy/autograd提供完整字段和梯度对照，不能替代局部公式锚点。
此检查定义不表示已运行通过；FP16反向、公共训练、optimizer与跨放置训练checkpoint
由[公共训练放置契约](resident-weighted-training.md#单样本节点放置与发布)独立验收。
