# v10 pool 非有限梯度：独立有界定位

2026-11-15，用户要求定位失败原因并建议解决方案。原主轨迹与活动源码均不改变，不自动恢复正式训练。
原三项失败仍为 failed，诊断退出成功不改变其状态。模型/update 导入原冻结 0ea05d61，
诊断工具单独提交、冻结；原配置、安装、数据、optimizer/RNG/cursor 身份逐项核对。

|问题|对象/对照|操作|预算/停止|验收|
|---|---|---|---|---|
|参数梯度坏值还是范数归约溢出？|d128 R64 pool seed11/22、R28 pool seed22|最近完整检查点回放至原失败下一步；保持 global64、原 microbatch 与 L256|最多3卡并行，各3300s计算/3600s服务，4 CPU线程/72GiB内存，合计12GiB新增上限；首次异常即停止更新|逐参数 finite、CPU FP64稳定范数；保留失败前权重及原错误|
|哪一个反向算子首次产生坏值？|真实失败步的同权重/同batch|独立重算，捕获除法/log/exp/rsqrt/fused attention反向边界及其上游梯度|仅失败步，包含在上限内|有限上游梯度变为非有限下游梯度的算子见证；原前向调用栈|
|修复是数学等价还是配方改变？|抽取的失败算子、CPU FP64、原FP32/NPU实现|有界局部公式重排/稳定VJP验证|最多少量秒级局部探针，不扩大模型候选|前向/反向、支持和边界分别报告；不以关闭finite检查作为修复|

补充快速定位：允许一个额外短探针，把R28 seed22最近检查点权重送入原失败batch；
最多300s，不做optimizer更新。它与严格回放区分，不能声称为原失败权重；
输出checkpoint带 `probe=True/diagnostic_not_formal_resume`，正式训练入口会拒绝其作为训练谱系。
这样在3项完整回放进行时利用额外空闲卡定位；最多4项诊断同时占卡，仍服从全局12slot与12GiB诊断总盘限。
首次定位后，每个实际失败batch至多追加一个300s算子/全模型VJP候选检查，包含在每项总1小时诊断窗口内。
候选完整batch仅检查前向身份、梯度有限、支持、单次optimizer更新；不启动新学习轨迹。

共享低负载选卡、12slot总并发上限、background.slice、原共同截止及>=200GiB硬余量继续有效。
不改 microbatch 去声称精确回放；跳过只读validation，使用历史共同step的loss/梯度范数和token位置核对，
不把仅checkpoint可加载称为全轨迹精确恢复。旧权重+原失败batch若单独测试，明确标记非原轨迹。

入口 `scripts/diagnose_v10_gradients.py`；各诊断 `v10-diag-<原配置ID>`，新输出 `runs/v10-diag-<ID>/`。
记录原主run为 `tl-v10-main-<ID>`；诊断是故障证据，不添加正式学习曝光、不更改Lens原失败状态。
使用 `python scripts/job.py status artifacts/jobs/v10-diag-<ID>` 检查；
`systemctl --user stop tide-lab-v10-diag-<ID>.service` 停止，并核对所有后代。

待运行/科学审阅：极小正质量归一化的VJP是候选机制，不能在真实回放前宣布已定位。
