# v10特化校准：已能承载171M比较，质量结论待训练

源码89217ec3的完整更新校准、6c927b32的融合梯度探针、f263f0cc及456a79c6的真实数据校准，
汇总见[固定证据及hash](../evidence/direct-v10-optimization.json)与[选择依据](../evidence/direct-v10-selection.json)。
正式模型冻结为0ea05d61；上游基座没有改动或接入。

## 修改与实测

特例限HB/KV/qry、区域共享、单头、固定局部邻接、pool/quad。
Full与两次Read采用NPU融合注意力，接收权重经Q/K附加坐标保留反向；
节点交接改成unbind，聚合采用静态邻接gather/sum，删除未消费的摘要。
短期mass共享L×L衰减矩阵，长期mass使用cumsum；没有为每节点保存浮点L×L mass矩阵。
仍有布尔因果/有效key mask，融合调用没有把全部浮点注意力分数和概率交给Python autograd保存。
这不是所有中间量都线性化，也不是实际跳过不活动Full。

|固定形状，FP32/L256|旧B4：位置/s、峰值MiB|特化B4：位置/s、峰值MiB|所选较大batch校准|
|---|---|---|---|
|R20 d64|2439 / 6760|4351 / 2870|B32约18050 / 21486|
|R64 d64|1081 / 21514|3909 / 9088|B16约5941 / 34684|
|R20 d128|2370 / 8905|4854 / 5136|B16约10689 / 18245|
|R64 d128，171.46M|1027 / 28380|3059 / 16080|B8约3470 / 30089|

每格五次完整微batch更新，最后三次中位数；包含前向、loss、反向、AdamW及同步，
输入准备/上传在计时外。表中数值策略及更新后路由有差别，不能称严格等价加速。
真实语料global64校准：R28 d64约12027、d128约7867；R64 d64约6010、d128约3650有效目标/s。
最大模型128M纯更新估计约10.2小时，另计检查、验证、checkpoint和共享干扰；不是完整作业耗时承诺。

FFN重算在最大形状B8把峰值约30089降至20094MiB，但吞吐约3470降至3145；
B16重算约3299、38160MiB，仍未改善周转，故主配置关闭。foreach未在代表大模型稳定获益，也关闭。
所有候选避免以反复OOM寻找上限；较大未尝试batch保留为not_attempted。

## 数值、机制和恢复

融合采用实数公式`softmax(q·k/√d + log(w_j) + rate·stride·j)`，
省略的`−rate·stride·i`为softmax行常数。原normal-tiny概率/衰减floor不在新attention路径中，
零key遮罩、空历史零输出和零边界梯度保持；`log(w)`输入仍以dtype tiny保护，不能称任意极小权重精确实现。
原FP32阈值atol1e-6/rtol1e-5保留；探针有两项有限严格失败，Q/K/V/接收权重梯度有限且机制检查通过。
CPU FP64完整小HB pool/quad的参数角色、前向、VJP及活动与旧独立参照一致；因果与FFN重算梯度检查通过。
原短程三步新进程恢复逐位通过；主表每项有自己的准入/恢复，长轨迹独立精确重放未认证。

## Trace与后续瓶颈

固定R64/d128/B8 trace导出了30次`npu_fusion_attention`和30次对应反向，
kernel文件含30次FlashAttentionScore和30次FlashAttentionScoreGrad，确认实际路径。
节点传递20次UnbindBackward，静态邻接仍有29次IndexSelectBackward；逐节点select反向已消除。
cumsum、index-add反向、batch matmul和逐元素算子仍有成本，是后续特化线索，当前不扩大优化。

原profile服务在导出后调用不支持的`key_averages`而失败，后代已清空；原失败与CSV原字节保留。
CSV含停止时RECORD警告，未采level1 AICore指标；仅作已观察算子/形状证据，
不把累计kernel时长当独占墙钟，也不把这里的零AICore字段当矩阵核未执行。
设备级AICore/AIVector/AICPU与host观测将在正式运行另记，不归因到共享卡上的单个进程。

本轮性能工作已足够支持[20条正式比较](language-capacity-activity-v10.md)。
稠密物理执行、低逻辑活动与学习质量分别报告；后续真实跳过计算和公共基座迁移仍需新证据。

## 正式启动后的准入口径补证

0ea05d61的8项quad准入出现大幅CPU/NPU差异，pool前向则通过。
定位发现原检查在NPU执行一次`update(... tokens=0)`后启用了warmup（k=R/2、tau=1），
CPU参照只加载参数，未调用`set_progress`，仍用最终k=2和选定温度。
因此该对比不是同一活动配置，不能把约0.8–0.9相对差解释为浮点误差或训练模型失效。
原检查及失败保持原字节；活动源码和训练日程未改。

新检查在加载state后对齐`set_progress(model.progress)`。新增CPU回归覆盖有warmup的实际HB，
相同设置的前向及全部参数VJP通过；5项v10规则测试完成。另提交四个正式quad形状的
对齐后NPU复核，使用新源码/输出身份并服从同一12worker资源锁，不中断正式训练。
复核未完成时仍记pending，不用CPU回归代替新NPU数值结果。
