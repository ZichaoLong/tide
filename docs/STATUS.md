# 当前交接 — HB与采样已审阅，TIDE启动修复后补做中（2026-11-08 13:09）

原7个第三批作业全部终态：6个completed，B阶段最后TIDE候选启动失败导致failed，均已核对无后代。当前只有`s3-growth-tide-repair1`活动，补齐先前批准但未实际训练的TIDE候选；不是新一批。**共同截止仍为2026-11-08 21:29:11北京时间**。[本批进展与已审阅结论](studies/third-batch-results.md)、[检查、逐例预测和计时证据](evidence/third-batch-progress-review.json)。

| 组 | 状态与结果 | 解释边界 |
| --- | --- | --- |
| HB新增seed22/33四训练 | 均400步completed；复用seed11后六模型完成2048唯一映射评价 | 六个checkpoint hash/config/安装/step核对；不是新做400步完整恢复 |
| HB三seed质量/成本 | dense/quad平均99.805%/99.040%；quad各低0.146–1.758个百分点，Full/消息约少45% | 当前小HB检索学习成立；无质量非劣或固定稀疏额外价值结论 |
| 独立推理计时 | 固定64例、各7轮，中位用时少8.03%–13.54% | 同一CPU/精度/输入/checkpoint的非独占观察，不推广通用硬件加速 |
| B三个控制 | full/head/MLP各200步，val20.313%/19.531%/19.531%，均未达25%门槛 | test按规则未揭示；MLP关闭无类别预测变化；缺TIDE不能判断TIDE收益 |
| TIDE补做 | 获物理NPU1，原3步冻结精确恢复再次通过，正式200步已开始 | 源码/config不变；旧失败保留，按原预算补齐而不重跑其他对照 |
| NPU采样 | completed、Lens核对、设备已释放；详细观测off/on模型/AdamW逐位相同 | 单次19.59/19.13秒不作开销或kernel归因 |

启动失败原因已经确定，是主代理上次交接脚本的路径变量被循环覆盖：将研究图误写进未来TIDE的state.json，预先创建目录，触发批驱动防覆盖。该文件schema是研究图而非训练状态，没有TIDE训练进程或正式模型结果。原目录、日志、hash不删除；新增review写入器只允许图写到research/map.json、证据写到docs/evidence，两个回归检查确认错误路径在mkdir前拒绝。[修复事实及预算](evidence/third-batch-repair1.json)。

补做计算cap7,350秒（恢复≤750、训练/记录≤6,600），加原B阶段485.28秒仍低于原3h；队列等待已缩短以容纳原绝对截止。没有重新获得12h、没有新增候选。状态/日志/停止/恢复见[RUNNING](RUNNING.md)。活动运行只读源码仍`1c6cb44197cb603cdab1e89b1be1a96bf692c789`，树hash `284807226729141d002e77c01d65c687ef6bc5652cb7864a434a2596cce021fa`，本次复核未变；数学0770593、基座85d161d及私有环境保持。

执行、严格数值、离散语义、恢复、学习分别记录：原quad和CPU/NPU严格失败不变，新的学习与重复推理不解除它们；B小型机制和3步/B2完整恢复通过不认证200步长轨迹。原批9个有指标run的Lens投影均verified/readback_matches，补做正在记录，终态待核对。Lens服务active，项目`tide-learning-lab`，私有入口仍`.local/LENS.md`。

当前账号约165/800GiB；仅补做使用1NPU/4CPU线程/24GiB，成功临时大checkpoint按原策略退役。余下工作是TIDE终态与完整B结论，然后交由用户验收；可选下一批及依据已列进展报告，尚未开展。
