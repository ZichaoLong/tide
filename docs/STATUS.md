# 当前交接 — 第二批已启动，CPU运行/NPU排队（2026-11-08）

用户已认可首批建议并授权[第二批有限研究表](studies/second-batch.md)。旧作业全部终态；五个新作业已接受，三个CPU作业实际运行，两个NPU作业按共享队列等待，没有改写首批输入或结果。

记录节奏修复、FC/HB稀疏的观测等价及新进程完整恢复、未记录步骤的非有限保护、配对任务干预检查通过。实际公共基座文件和依赖RECORD与锁一致；原consumer wheel保留，新作业显式使用冻结实验源码。新的Lens小记录已verified/readback_matches。详见[准备证据](evidence/second-batch-preflight.json)。

本批范围：三个既有FC checkpoint的query/延迟评估；一对HB400步；一次NPU轻量采样；冻结骨干的head-only/TIDE/MLP40步对照。研究图新增四组与待审阅结果，执行完成不自动生成科学支持。NPU均有占用，按队列等待；CPU工作可先行。所有后续多seed、替换或任务扩展仍须本批审阅。

首批结论继续有效：FC CPU64三seed独立test97.27%–99.61%；NPU80步val24.22%；HB31.25%/35.16%未达50%，Full/消息减少未证明稳定速度收益；B旁路预测贡献未证实。原严格失败与未知原因保留。[首批总报告](studies/first-batch-results.md)。

账号约165/800GiB；新增磁盘预算20GiB。cgroup v1只确认服务生命周期隔离，CPU/内存配置为请求，应用线程和超时实际约束。固定数学0770593、公共基座85d161d及原私有环境不变。Lens项目`tide-learning-lab`，私有入口见`.local/LENS.md`，没有新部署或访问策略修改。

## 当前后台与固定输入

统一源码`94bf712f8bd2e4007f0f56d1fea4799c99833dcd`，只读`.local/source-second-batch`；实际manifest已核对。`s2-fc-mechanisms`、`s2-hb-dense`、`s2-hb-sparse`在CPU运行；`s2-npu-sampling`队列第1、`s2-frozen-growth`第2，等卡上限分别5/10分钟。状态为最后观察，不是实时监控。

所有服务已核对真实name=systemd cgroup，位于background.slice并与会话隔离；新HB已出现真实optimizer步，低频记录中未观测步骤也计入完整工作量。[提交复核](evidence/second-batch-submission.json)、[固定配置、预算、状态/日志/停止/恢复命令](RUNNING.md)。当前新结果尚未经审阅，图中四份新结果保持待填。交接后无须保持会话，用户后续可唤醒检查；队列超时不会自动重试。
