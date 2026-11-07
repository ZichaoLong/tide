# 当前交接 — 第二批已授权，准备提交（2026-11-08）

用户已认可首批建议并授权[第二批有限研究表](studies/second-batch.md)。旧作业全部终态；当前准备新源码快照和后台提交，没有改写首批输入或结果。

记录节奏修复、FC/HB稀疏的观测等价及新进程完整恢复、未记录步骤的非有限保护、配对任务干预检查通过。实际公共基座文件和依赖RECORD与锁一致；原consumer wheel保留，新作业显式使用冻结实验源码。新的Lens小记录已verified/readback_matches。详见[准备证据](evidence/second-batch-preflight.json)。

本批范围：三个既有FC checkpoint的query/延迟评估；一对HB400步；一次NPU轻量采样；冻结骨干的head-only/TIDE/MLP40步对照。研究图新增四组与待审阅结果，执行完成不自动生成科学支持。NPU均有占用，按队列等待；CPU工作可先行。所有后续多seed、替换或任务扩展仍须本批审阅。

首批结论继续有效：FC CPU64三seed独立test97.27%–99.61%；NPU80步val24.22%；HB31.25%/35.16%未达50%，Full/消息减少未证明稳定速度收益；B旁路预测贡献未证实。原严格失败与未知原因保留。[首批总报告](studies/first-batch-results.md)。

账号约165/800GiB；新增磁盘预算20GiB。cgroup v1只确认服务生命周期隔离，CPU/内存配置为请求，应用线程和超时实际约束。固定数学0770593、公共基座85d161d及原私有环境不变。Lens项目`tide-learning-lab`，私有入口见`.local/LENS.md`，没有新部署或访问策略修改。
