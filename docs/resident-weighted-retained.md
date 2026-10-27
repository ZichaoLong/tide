# Weighted 保留窗口与有限切面反向

本页对应固定[PDG有限切面](upstream/tide-core-5/positive-delay-graph-finite-cut-learning-note.md)
和[共同局部架构](upstream/tide-core-5/local-semantics/weighted-message/architecture-learning-note.md)。
内部保留层与[公共训练owner](resident-weighted-training.md)分别验收；后者管理token、optimizer发布和训练checkpoint。
实现/运行状态以STATUS为准，本页的检查定义不表示已经通过。

`WeightedWindowSnapshot::retain`在窗口边界复制候选实际journal、完整raw字段、精确int64元数据、
存在位、左右pending及外部输入身份，并保留真实参数值及alias分区。它不调用参考递归、不预跑
未来轨迹、不用有效状态投影代替raw解码。保存前按实际张量尺寸检查独立字节预算。
快照另保存节点物理放置，边界raw镜像只复制实际计算结果；局部反向回到原owner，
公开根与跨窗桥保留coordinator布局。具体见[节点放置](resident-weighted-placement.md)。
GPU张量与CPU参数快照有版本检查，原地改写即使数值相同也使记录失效。

`WeightedWindowAdjoint(saved)`不依赖原前向owner存活；复用已声明的实际journal逆序算法。
借用当前窗口的旧入口仍检查前向generation，不能用它冒充保留窗口。
保留对象和构造的反向共同计入反向总tensor预算；训练owner还须合计其它窗口及活着的前向。

`append_weighted_window_bridge`只接受相同图/设备/样本布局和相邻逻辑切面。
后窗左raw字段VJP加到前窗final根；后窗左pending按完整六元物理坐标匹配前窗右pending。
匹配、None/连接零合并和逐边贡献都在设备，外部输入保持各自独立叶子。
没有连接的毒值不被读取，有限性或祖先错误使整个结果无效。中间空窗口保持恒等。

每个保存窗口有私有参数快照。`bind_weighted_window_parameters`先检查完整canonical名称、
alias、形状和dtype，再将其贡献绑定到同一训练generation的参数owner后求和；不能因数值相等
而合并独立参数，也不能把私有状态当共享参数。参数更新与retained token失效由下一层管理。

验证入口：

- `weighted_window_check --saved-window`：原12异构图/Read组合、每组12根模式；保存后推进并关闭
  原前向，再比较全部VJP；保存预算拒绝和原地metadata修改拒绝单列。
- `weighted_retained_check`：FC/HB/非对齐TimedDAG/PDG反馈×old/proposal，三个连续窗口含空窗。
  候选先运行；独立CPU Greedy保持跨窗口autograd连接。none/full/final/pending/all五种根模式
  比较全部参数、各窗外部输入、最左pending和每个raw字段，None/零及整数严格一致。
- 完整单窗口及局部独立公式沿[窗口VJP](resident-weighted-window-vjp.md)追溯。

浮点VJP继续使用rtol=3e-5、atol=3e-6；不放宽路由、连接位或checkpoint元数据。
本层不定义loss、不更新参数，不承诺磁盘上恢复retained autograd。
