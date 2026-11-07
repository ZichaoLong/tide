# 当前交接 — 首批有界研究正在后台执行（2026-11-07）

首批已获明确授权并实际启动；未回退启动HEAD 942c329，未修改公共基座/环境或其他项目。活动实现固定`0d7ec5bdaa34556df9c93886e95f55fa5c9876ab`，只读归档`.local/source-first-a`。工作区后续提交是配置、检查、证据和图谱，不会改变活动作业。数学0770593、基座85d161d；实际wheel/安装/环境/下游身份见dependencies。

## 当前作业

| 作业 | 已确认状态 | 内容与预算 |
| --- | --- | --- |
| confirm-a-s11 / confirm-a-s22 / confirm-a-s33 | 三个background.slice服务实际running | CPU FP64，各seed query-only+完整FC，B32各400步；每服务CPU1线程、RAM8GiB、硬限80分钟 |
| pilot-b | background.slice实际running，物理1→逻辑npu:0 | Qwen原模型已完成40步，TIDE运行，MLP按顺序等待；各B32/40步，NPU1+CPU4线程、RAM32GiB、硬限70分钟（含≤15分钟等待） |

最多同时1 NPU、CPU配额7线程、RAM上限56GiB。首批新增磁盘上限20GiB；账号启动约165GiB/软300/硬800，当前余量充分。CPU并非独占，不能把并行期间计时解释成独占基准。

完整固定配置、状态/日志/停止/恢复命令见[RUNNING](RUNNING.md)。所有作业终态、原始JSONL与checkpoint均保存在项目目录；缺失结果不是通过。NPU整组锁、双采样与排队发生在服务内；停止方法已用真实含孙进程的CPU服务验证，检查无存活后代。

## 已有结论与正确性边界

[TL-A1先导结论](studies/TL-A1.md)：seed11/80步，验证accuracy 11.72%→24.22%，query-only10.94%，移除历史10.16%。只支持当前小模型开始使用历史；不证明已解决关联检索，也未到50% HB门槛。原任务的三个种子400步确认已启动；HB效果比较仍依赖确认结果，不自动扩组合。

[启动审计](startup-audit.md)/[原始检查汇总](evidence/startup-checks.json)：CPU64完整独立根、共享梯度、optimizer、因果/续接通过；新进程应用恢复在CPU FC与NPU Qwen+TIDE上均逐值通过。已用实际任务尺寸宽16/8键8值再做B2独立检查，结果通过。

原基座NPU综合根2.861e-6、下游FP32 dense AdamW参数更新、原checkpoint CPU cache数值严格失败均保留。[TL-REQ-001](requests/TL-REQ-001.md)只在实验侧记录，未改基座或发布外部issue。CPU cache有限探索继续完成其余检查，状态不是passed；B的FP32研究不声称严格等价CPU根。原Qwen在NPU的初态保持/梯度进入/关闭分支/缓存与精确恢复实际通过，仍不替代学习效果。

选择缓存Qwen2.5-0.5B-Instruct，不下载权重，来源/hash/背景见[选择说明](checkpoint-selection.md)。所有候选训练骨干+共同分类头，TIDE/MLP只加一个末端残差旁路，无多层替换。

Lens已在正确实例的`tide-learning-lab`项目写入并读回，认证网页/API检查通过。训练与Lens SDK依赖隔离，项目记录权威；run目录projection.json保存对应身份，私有访问方式见`.local/LENS.md`。本轮未部署服务或改访问策略。

## 下次唤醒

先查四个作业与runs各batch.json；审阅正式A的三seed/test、B的同预算对照和分支干预，保留失败种子与数值边界。之后决定本批HB及B有限重复是否满足前提；未满足则给出负面或无结论报告。按首批研究表完成每组一页结论、可复现配置/恢复证据、Lens对应及研究图，再交用户验收；不自动扩下一批任务、规模或候选。
