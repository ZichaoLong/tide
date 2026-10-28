# 当前交接 — 2026-10-28

执行用户新授权的三个方面：已知限制处理、LH/fractal及本仓库core-3性能经验迁移、
正确性与参数匹配性能设施。按ROADMAP的A—E持续实施，不在阶段提交后暂停。
授权分支graph-execution-foundation；基线58de04b1bfd64704616f55f4f9ea84a404775ded，
实现基线1d19c5998b96689b3620ecb69c7f712a9dd84ad0，固定数学0770593838a37c790a568226930567ed60f99b83。

## 当前工作与下一步

启动状态干净，无继承活动任务；19份core-5/5份legacy快照校验通过。
当前实施A/B：静态参数清单、参数预算匹配、可扩展拓扑/用例、历史经验索引与过时文档纠偏。
随后处理最小数值/异步复现，依赖独立时并行推进工程工作；不使用子代理。
静态参数/计划及Python共享计算开发测试120项通过，低秩手算/完整根另16项通过。
新增二进制材料、五族standalone配置解码、初始化/共享KV/ready副本优化。
下一步：冻结开发副本构建核心与standalone，跑相关完整根/训练/恢复及损坏材料拒绝测试；通过后固定实现提交。
长任务开始前在此记录冻结源码、unit、日志、有限资源及确切检查命令。

## 活动任务

准备提交core5-performance-dev开发服务：background.slice，Nice10，2构建线程/单线程ATen，CPUQuota200%、MemoryMax64G、2400秒上限。
固定副本artifacts/worktrees/core5-performance-dev，基线58de04b加本轮dirty快照；scripts/develop.py记录source.tar.gz与树hash。
unit=core5-performance-dev-20261028；记录artifacts/jobs/core5-performance-dev/{status.json,task.log,development.json}。
检查：systemctl --user show core5-performance-dev-20261028 -p ActiveState -p Result -p ExecMainStatus；python scripts/status.py。
停止：systemctl --user stop core5-performance-dev-20261028。开发任务不替代固定提交资格；开发服务已核验active/running、Transient=yes及独立ControlGroup；终态待验。
另提交core5-async-baseline-20261028：固定58de04b，artifacts/worktrees/core5-performance-baseline，
默认异步双NPU Python delta-pdg完整VJP，当前9.1工具链；队列等候180秒、运行300秒，
Nice10/单线程/MemoryMax32G，日志artifacts/jobs/core5-async-baseline/task.log和checks/result.json。
已终态failed/exit1：完整all-VJP重现CANN rtStreamWaitEvent invalid event，设备物理1/13。
CPU半精度GDN诊断开发重现step1 nodes.5.kernel.conv.weight[0,2]差.001983642578125；
显式保留原容差及失败，开发记录artifacts/core5-half-numeric-dev，待固定提交复核。
检查/停止同名systemctl unit；该任务独立记录旧限制重现，失败不改记通过。
旧core5-review-fixes各任务保持终态，不恢复旧队列。

## 已有证据与保留边界

[上一实现资格](evidence/core5-review-fixes-20261028.md)：CPU442项，standalone NPU FP32/FP16四组件、
单设备/双卡训练恢复通过；CUDA sm80/90/100构建与CPU31项通过，无GPU真机资格。
CANN任务Python PATH问题已解决，旧失败仍保留；参考仓库和公共工具链保持只读。
默认异步双NPU eager VJP、eager FP16 GDN/AdamW、SDK FP32 affine及FP32反馈PDG严格失败
尚未由当前新增工作解除。CUDA真机/x86_64、大规模容量和跨卡吞吐另验。
现有core-5性能仅覆盖宽4/32、3/8节点；不能外推旧core-3十亿级参数用例。

本轮源修改及证据以git status和ROADMAP为准；本页取代此前已完成审查任务的当前状态。
