# Timed 跨层恒零边的路由梯度修复已交付

基座实现 `3150ae64` 在求和、加权平均、注意力等消费者读取边质量之前，对已证明恒零的质量应用零导数。Multiscale Timed 按时钟标记无外部注入的首层发送量，再按每个节点的真实入边、发送上界和一拍延迟逐层传递，覆盖区域批处理、设备放置和带梯度的跨窗口状态。独立零先验、弱正质量及可独立求导的 pending 保留原导数，临时证明只存在于当前计算图，旧检查点仍可读取。[路由语义](semantics.md)

冻结版通过 146 项 CPU 与 67 项 Ascend 定向检查，下游通过 92 项迁移回归。Timed/K/singleton7 的 A2 第 2,727 和 3,252 步原批次在 CPU、Ascend 上均通过，logits 和下一私有状态与各自原前向逐位一致，123 份参数梯度均通过 FP64 对照。隔离 Ascend 从第 3,221 步运行至 3,257 步，另一进程精确加载第 3,256 步并复算下一步。私有状态逐位一致，参数、AdamW 矩、裁剪后梯度最大绝对差分别为 1.86 × 10⁻⁹、4.66 × 10⁻¹⁰、1.86 × 10⁻⁹。[下游发布验收](../../tide-learning-lab/docs/evidence/pilot-r3-r14-foundation-route-acceptance.json)

北京时间 2026-11-21 20:39:04，下游正式 r15 已固定本实现并登记上一批全部 333 个未完成任务。原故障 A2 在物理卡 9 上通过第 3,252 步，正式第 3,263 步检查点已通过文件、状态和任务契约校验。B 关闭。[下游正式接续](../../tide-learning-lab/docs/evidence/pilot-r3-main-r15-handoff.json)

当前源码还包含既有的 quad 常数下界、KV 未写入槽位、Shannon 正支持和 Attention 单支持修复，资格按设备、精度及实测负载记录。[quad 验收](../../tide-learning-lab/docs/evidence/pilot-r3-r11-foundation-quad-reception-acceptance.json) · [KV 验收](../../tide-learning-lab/docs/evidence/pilot-r3-r10-foundation-kv-acceptance.json) · [Shannon 支持策略](shannon-positive-floor.md) · [Attention 单支持修复](attention-native-support-repair.md)
