# v6执行瓶颈定位

用户2026-11-09要求定位为什么当前TIDE NPU训练比CPU慢，必要时提出公共基座修正需求。
研究ID为`TL-EXEC-v6-PROFILE`，沿用v6固定数学0770593、基座74e7bbd、私有安装和native构建。
现有训练、冻结源码及基座均不修改；这里只新增实验侧公共接口诊断。

问题是完整训练步的主要成本与执行缺陷，不要求先兑现理想稀疏加速比。
先验怀疑是逐事件host判断、碎片化算子和反向根处理；源码阅读不当作耗时归因。

| 诊断 | 固定条件和改变项 | 观察与停止 |
| --- | --- | --- |
| native CPU/NPU对应 | FC16 KV quad，width128，B2/L8；一预热、两步测量、一完整CPU算子profile | 分离core前向、任务readout/head、backward、clip和AdamW；分析调用次数及形状 |
| Python/NPU调用栈 | 同上，公共Python packed，记录CPU/NPU trace和cProfile | 把热点定位至具体公共实现；不把Python定位数据冒充native占比 |
| 原形状复核 | 同一FC16，B8/L128，CPU/NPU各一预热和一步测量，不加算子profile | 验证小形状归因在原工作量上的阶段分布；共享负载单列 |
| 条件性隔离 | 最多六个具名小探针：优先mixed-b/a或width256/512；B8/L16 | 只验证前述观测支持的假说，不遍历执行矩阵，不进入正式学习 |

本定位新增最多1 NPU卡时、3 CPU核时、4GiB产物，最多同时2个NPU及2个CPU单线程/双线程worker，
单项≤20分钟，共同≤2小时；计入v6共同预算，不影响原截止。实际诊断worker使用`v6-diag-`身份，
由现有budget监督计入总账。达到预算保留未完成原因，不自动无限扩展。

使用`scripts/diagnose_campaign_execution.py`调用公共模型；分阶段同步用于定位，不冒充无插桩的原吞吐。
预热、分段测量、cProfile、CPU/NPU profiler分别保存；trace不参与学习或参考生成。
任务侧只增加观测范围，不替换数学或执行函数。新配置有独立固定源码与输出目录。
项目JSON为权威，Lens项目`tide-learning-lab`，诊断前缀`tl-v6-diag-`。

执行、严格数值、离散机制、恢复、学习独立记录。本诊断不新增学习或恢复资格；
公共placement变更的数值/工作差异保留，不能宣称等价加速。建议修正应附原阈值、独立梯度/共享/
optimizer及恢复要求，不能为提速删除必要检查或改变None/连接零、支持与时间语义。
