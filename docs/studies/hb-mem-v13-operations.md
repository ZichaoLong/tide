# HB Mem.sum/qry v13：操作与有限执行表

用户2026-11-17已授权执行六组。共享截止2026-11-19 17:44+08，不按每个作业重新计时；本机最多16张共享卡，无累计卡时/核时上限。只完成128Mi，不自动追加候选、种子、深度或曝光。

## 已固定的科学比较

12条sum从零训练；Q10-L、POOL-L各补seed11/22的qry，共4条条件控制。理由：v10旧商VJP/求解器或混合修复谱系不能作为当前数值实现只改Mem的严格控制。新两侧均使用已接受ordered-v1商VJP和当前分配器；旧结果保留作历史背景。其余四组的8个qry终点只在新确认集重评，不重训。

统一W103训练池、词表4096、L256、global64、seed11/22、AdamW、256Mi学习率日程；实际训练终点128Mi。共同8/32/128Mi曲线。sum减少Read Q/K/V参数，Full注意力、图/读出及组内其他函数保持；pool为全活动参考，CONTENT-S保留Read内容直达.1。

|组/Mem|总参数M|物理microbatch|累积次数|校准targets/s|峰值MiB|实现|
|---|---:|---:|---:|---:|---:|---|
|q10-s/sum|20.515|16|4|5653|12225|matrix|
|q3-s/sum|46.392|8|8|2544|14256|matrix|
|q10-l/sum|72.100|8|8|4167|11507|matrix|
|q3-l/sum|163.595|8|8|2098|26241|matrix|
|pool-l/sum|163.595|4|16|2056|14412|matrix|
|content-s/sum|49.013|8|8|2578|14628|matrix|
|q10-l/qry|75.541|8|8|6903|13259|fused qry|
|pool-l/qry|171.459|8|8|3929|30279|fused qry|

以上是共享设备上的synthetic完整更新校准，含VJP/Adam，不含数据/验证/写checkpoint；不是语言质量、等计算或独占性能结论。选择已完成且峰值≤34000MiB的观察，取距最快5%以内的最小batch，以改善共享卡可入场性。更大batch的3次原OOM保留；只采用已完成有限的较小batch记录，正式链另设接入、先导和受训恢复门槛。sum此次没有兑现端到端加速，仍可检验其质量和信息作用。

## 正确性与独立确认

独立CPU FP64历史/VJP及整网测试通过；qry新旧小型CPU/NPU前向/VJP逐值相同。NPU L256接收权重VJP保留原1e-6/1e-5有限失败（最大绝对差约2.74e-5，整体相对L2约2.1e-7）。有界分解也观察到CPU32/NPU32历史累积差异；未来输入梯度严格为零，但原超差的每个原因尚未全部隔离。按既有授权进入明确标记的FP32探索，原2e-5可行性、不许非有限/因果/恢复错误的门槛保持。见[资格审阅](../evidence/hb-mem-v13-admission.json)。

每条新链：小型接入→新进程3步连续/1+2恢复→8Mi有限学习先导→受训global64/L256连续2步与1+1恢复→128Mi→新262K确认→512训练段诊断。先导只要求有限学习信号和资源可行，不按轻微NLL差异淘汰。pool采用全活动验收，不误套低活动门槛。临时恢复checkpoint只在比较通过后按SHA回执退役；原失败不会自动重试。

`runs/v13-data/data-lock.json`固定138个新文章、262144目标，排除旧train/validation/test及v11/v12确认。早期v1/v4/v5标题也已核对在官方文章排除集中。test不打开；新确认只用于终点审阅，不用于配方/温度选择。

## 固定输入与记录

正式快照`.local/source-hb-mem-v13-main`；配置表`configs/studies/language/hb-mem-v13/table.json`和`budget.json`。精确commit、作业ID、实际卡号与Lens映射见`docs/evidence/hb-mem-v13-handoff.json`。已有校准快照和作业保持不变。

Lens项目`tide-learning-lab`：`TL-HB-MEM-SUMQRY`为正式训练与先导，`tl-v13-pilot-<组>-<mem>-s<seed>`及`tl-v13-main-<组>-<mem>-s<seed>`；`TL-HB-MEM-CONFIRM`为新确认，复用权重用`tl-v13-reuse-*`；`TL-HB-MEM-DIAG`为终点诊断；`TL-HB-MEM-CAL/CHECK`是成本/正确性检查。实际显示名和UUID取各projection.json，私有入口在`.local/LENS.md`。

账号硬限取min(700GiB,实时quota)；整盘可用另查。16链同时受训恢复的保守数据峰约74GiB，按80GiB新增规划，终态权重/Adam约34GiB；新worker为并发保存预留32GiB，低水位时在更新边界请求停止，保留原记录。不能保证其他用户任意突发写入不影响磁盘。见[空间核算](../evidence/hb-mem-v13-storage-plan.json)。

## 查看、停止、恢复

`python scripts/review_hb_mem_v13.py`汇总真实作业/阶段/设备及学习进度；单个作业用`python scripts/job.py status artifacts/jobs/v13-main-q3-s-sum-s11`或`logs`。阶段日志在`runs/v13-main-<ID>/<阶段>.log`，训练曲线是`pilot/metrics.jsonl`和`train/metrics.jsonl`。硬件分列AICore/AIVector/AICPU记录于`runs/v13-hardware/observations.jsonl`，含其他共享进程影响。

停止某条：`python scripts/job.py stop artifacts/jobs/v13-main-<ID>`；该工具确认整个cgroup及后代退出。不要仅杀Python子进程或删除活动输入。批量停止仅针对本批显式作业清单。

恢复不是重新运行提交脚本：先核对退出及`train/checkpoint.json`或pilot保存点、SHA、配置/source/data/optimizer身份，再以新作业名、新输出目录、原冻结源码提交`campaign_stage.py train --resume <latest.pt> --config <原配置> --data <原训练池> --admission <原check/result.json> --recovery <适用recovery.json> --until-tokens 134217728 --deadline-epoch 1795081440`（另传output/run-id/recording-group）；通过原job.py/environment/资源策略启动。恢复前保留失败，不跳过必需的受训恢复，也不重置48h截止。已失败科学链不自动续跑。
