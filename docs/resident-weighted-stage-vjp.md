# Weighted resident 数值阶段反向组合

固定教材 [架构 §2–4](upstream/tide-core-5/local-semantics/weighted-message/architecture-learning-note.md)
的 Aggregate→Upd→Mem→Describe→SelStep→Full 链，使用
[完整图前向](resident-weighted-flow.md)的实际局部 tape 与参数角色。
这是完整图训练的组成部分；跨阶段/跨窗口消息与状态前驱、optimizer publication
和训练 checkpoint 仍需由独立训练 owner 管理，不能由本组件资格推断完成。

`weighted_flow_node_vjp.cpp::append_weighted_node_vjp` 接收完整 Content、未缩放 G、
候选 raw fields 与命名 Describe 的独立 cotangent。Full 读候选态，Mem 根据绑定
选择初态/旧态/候选态；三个快照的梯度不会混合。Upd 的所有候选更新均参与链式
反向，零接收保留原始字段；逻辑时间和 cache stamps 没有浮点 cotangent。

`weighted_flow_region_vjp.cpp::append_weighted_flow_region_vjp` 按实际设备 pack/order
还原每个候选的描述根，汇总完整区域 control 根。它使用发送函数声明的精确解或
分支 VJP，不使用旧 HST。一个 `(c,G)` 载荷根需同时提供 c 与 G 的连接位，数值零
不表示缺失。公开 Content 是一个结构化 Tensor 根；raw state 与 Describe 的字段
保持独立根。填充行没有数学作用。

`weighted_flow_adjoint.cpp` 按连接位汇总，允许未连接 cotangent 中有不可用数值；
已连接的非有限值明确拒绝。来源的实际物理原子保持独立，低秩因子经已保存的
适配链求导。`WeightedBanks::roles()` 将每个实际角色/切片梯度恢复到注册参数的
原尺寸，随后交给 canonical 参数归约，包含同一参数的跨节点和跨函数全部使用者。

| 固定定义 | 代码与返回 | 独立检查 |
| --- | --- | --- |
| 三 Read 快照、Full 候选快照 | node VJP 的 old/initial，内部 proposal 汇总 | 五记忆 × 三读取模式；逐个 raw/Describe/Content/control/Full 根 |
| 完整区域分配、物理纤维 | region VJP order/inverse；payload 与连接位 | CPU Greedy 独立调度+标量函数，重复类别、缺席、零接收 |
| 共享参数与私有状态 | role→原 shape/offset→ParameterRegistry | canonical owner 梯度及 None/零；每样本/节点 raw 字段独立 |

`weighted_stage_check.cpp` 和拆分的 fixture/compare 源只验证组合；局部公式的数学
独立锚点沿各 resident-weighted-* 契约追溯。实际检查包括五记忆×三快照的15个基础配置，以及qry、区域Attention/match/grouped、节点增益、异宽来源低秩链的9个组合；已连接非有限根明确拒绝。候选先自行执行；参考输出不进入候选。
候选解码只读取真实布局。构建/运行入口：

```sh
python scripts/build_device_control.py --core-build CORE --build-dir NEW_BUILD \
  --checks weighted-stage weighted-flow --ascendc-soc Ascend910_9392 --jobs 2
python scripts/verify_device_control.py --build-dir NEW_BUILD --output-dir NEW_RESULT \
  --checks weighted-stage weighted-flow --device npu:0
```

源码、实机验证与独立的[完整训练边界](resident-weighted-training.md)从 STATUS/证据索引进入；本页不把测试定义
当作已运行证据。当前数值阶段为 FP32，CUDA 构建与真机也分别记录。

固定源码阶段资格见[2026-10-28报告](evidence/core5-resident-weighted-stage-20261028.md)；完整训练范围独立记录。
