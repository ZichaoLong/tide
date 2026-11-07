# 当前交接 — 首批实验接入与先导（2026-11-07）

用户已明确授权本机CPU/NPU首批研究；启动时HEAD 942c329、工作区干净，没有回退或覆盖已有修改。首批范围与停止规则见[研究表](studies/first-batch.md)。基座只读固定85d161d，数学固定0770593。

私有venv继承CANN9.2.0公共基线，安装固定tidegraph wheel与Transformers4.57.1。来源、wheel、实际安装内容和环境身份见dependencies。原基座综合根三项严格失败保留。

已完成CPU64小型FC独立参考/梯度/共享参数/AdamW/因果/续接检查，项目完整checkpoint新进程恢复逐值通过。CPU32稠密发送出现参数更新有限超差，显式eps1e-6配方仍未全部通过；原记录保留，A路线先采用CPU64。NPU仍待本项目检查。具体证据将汇总到启动审计。

B路线选已有只读缓存Qwen2.5-0.5B-Instruct，完整revision与hash见[选择说明](checkpoint-selection.md)，未下载权重。原模型与旁路完整检查尚待执行。

Lens独立记录器已在tide-learning-lab项目完成单事件写入读回（status=verified，readback_matches=true）。训练环境与记录器依赖隔离；私有入口留在.local，项目记录仍为权威。

当前无正式训练服务。下一步冻结本实现，运行CPU/NPU短检查与batch校准，按研究表门槛提交有限先导和比较。活动源码只读冻结归档，作业状态/log/stop/resume入口为scripts/job.py。
