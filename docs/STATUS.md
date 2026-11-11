# 当前交接：执行已授权的优先清理与公开模型整理

2026-11-11：用户已授权清理上一轮优先四组共610个文件（46.185GiB），其余候选暂缓。
六个公开原模型全部保留，将核对固定catalog逐文件hash后按repo/revision整理，以旧路径链接保持可用；
后续由用户sudo迁入`/mi/data2T/models`并转移所有权，本线程不执行sudo。
此处是执行前交接；实际删除/整理结果以随后生成的`docs/evidence/artifact-retirement-20261111.json`
及`public-model-staging-20261111.json`为准。科学checkpoint、失败证据、结果/metrics和Lens均保留。

## 上一轮已完成交接（历史只读审计）

# 当前交接：TL-REQ-005已提交，磁盘只读审计完成

2026-11-11（北京时间）：用户已同意正量链路方案并授权提交基座需求；随后仅调查两个实验目录的清理候选。
**[TL-REQ-005](requests/TL-REQ-005.md)已普通推送（69be6d0），基座本地新增`docs/tide-learning-lab-TL-REQ-005-handoff.md`；
待基座回复/交付，尚未下游验收。** [提交凭据](evidence/tl-req-005-submission.json)。
基座原tracked状态和既有003/004交接文件保留，未修改执行源码或其STATUS/ROADMAP，未提交/推送基座分支。
当前安装仍是74e7bbd，没有接入另线程性能工作，没有新训练或数值修复作业。

需求包含稳定Bern响应/完整KKT VJP、稳定Aggregate/KV复合计算、显式Bern/attention正支持与前置cast/正配置检查，
保留合法零、严格失败、连续衰减许可及新模式恢复身份。log-mass/更大范围融合为有条件后继，不阻塞可独立交付项。
三项真实失败与静默小见证的范围见[补证报告](studies/positive-path-audit-v6-results.md)；原d6582a1证据不改写。

[磁盘只读审计](studies/disk-cleanup-audit-20261111.md)及[逐文件清单](evidence/disk-audit-20261111.json)：
本工作目录约38.985GiB，gef-lm-experiments约122.585GiB；账户约291.888/800GiB，硬配额余量约508.112GiB。
优先候选46.185GiB（恢复检查checkpoint、可重建B缓存、旧benchmark checkpoint和GGUF）；
可选另92.570GiB旧公开模型权重。**本轮删除0字节，未清理Lens。**
保留跨目录Qwen05基线（33份本项目作业引用）、三项未解决失败的原输入/诊断、所有最终/里程碑科学checkpoint与数据。
294个相关unit无active/MainPID，2360条已记录PID/start-tick身份无匹配存活；删除前仍需再次核对，清单不是删除授权。

原v6的200配置全部终态：原始184 completed、11 failed、5 budget_truncated；筛选181完成/3真实下溢，
规模3个Transformer达目标/13个TIDE未达目标（8个到限终止的内层running记录保留）。已用878.66 CPU核时、73.78 NPU卡时。
总体科学审阅和密封test仍待办；当前终态不等于严格数值、学习或scaling通过。
TL-POSITIVE-v6四个CPU诊断亦已完成，约0.2395 CPU核时/49MiB；恢复只认证六个可用历史标量观察，不认证全状态长轨迹。
Lens仍为tide-learning-lab / TL-POSITIVE-v6，四项投影verified/readback_matches；私有入口`.local/LENS.md`不入公开记录。

下一步：基座线程按需求接受/交付，下游另用新私有安装验收；清理待用户决定实际范围，不自动执行或追加研究批次。

## 历史快照：2026-11-09执行定位与训练提交

2026-11-09 21:57（北京时间）：用户要求的CPU/NPU定位已完成，14个具名诊断全部终态，10个服务无存活后代；Lens投影均verified/readback_matches。结论是当前eager路径存在大量逐行Describe、KV动态索引及碎片化反向，不能仅用width128解释CPU优势。原形状B8/L128新测CPU/native63.61秒、NPU/native265.00秒；宽度128→512缩小差距但尚未反超。mixed-a约改善37%，已有传输选项约改善3%；均为有限诊断，非严格等价速度或语言质量认证。

[诊断结论](studies/language-execution-diagnosis-v6-results.md) · [证据/运行/Lens映射](evidence/language-execution-diagnosis-v6.json) · [基座需求TL-REQ-004](requests/TL-REQ-004.md)。诊断源码d90e733/b4bc62b、数学/基座/安装与native构建固定；约0.81 CPU核时、0.36 NPU卡时、2.42GiB产物。基座只读，原训练及其固定源码未改。建议优先由基座改善批量前后向以支撑更大数据曝光；不要求先兑现稀疏比例对应的吞吐，也不自动部署尚未交付的优化。

以下仅保留当时的训练提交快照；其中“运行/等待”不是当前状态。当前逐配置状态使用`python scripts/campaign_status.py --brief`。执行和正支持诊断均不代替v6终态学习审阅。

2026-11-09 21:19:32（北京时间）只读核对：**200个配置全部由后台worker接收，111个正在训练、84个等待同worker内的前项、5个完成**。116个本批训练worker均核对属于background.slice；不是200个独占设备。原v5已终态，旧建议已被本次用户授权的较大研究表取代。

[研究表](studies/language-coverage-scale-v6.md) · [操作/停止/恢复](studies/language-coverage-scale-v6-running.md) · [逐配置/运行/Lens对应](evidence/language-v6-handoff.json) · [离线研究图](../web/index.html)。最新状态使用 `python scripts/campaign_status.py --brief`；文档与研究图是观察时点快照。

| 并行线 | 实际提交 | 训练规模及对照 |
| --- | ---: | --- |
| 局部候选并集覆盖 | 184配置 | 91个具名局部配方各在FC16/HB40承载（mixed FC另有明确结构差异），加2个近参数Transformer；L32、65,536有效目标、全局B8、seed11 |
| 实质数据/容量/深度 | 16轨迹 | 14主候选加2个NPU同模型观察；主要TIDE约5.09–25.63M参数，FC16/32/64、HB40/88、KV/SSD/GDN、Shannon控制，3档Transformer；L128、全局B32，262K/1M/4M里程碑 |

数据为固定WikiText103第一训练parquet，完整原LFS hash通过。训练20,971,520 token/4088篇；验证258,990/134篇；密封test294,476/157篇。旧文档hash及文章标题均不进入新确认集，排除分片边界残文；近重复仍未完全排除。沿用旧train-only 4096 BPE词表，不把重复曝光当作独立数据增长。训练和筛选不读取test。

三个规模Transformer已各完成4,194,304有效训练目标；它们的执行完成不代替与TIDE共同曝光的科学比较。

主要TIDE在CPU1/FP32运行：合法候选采用公共native，positive-floor-v1 Shannon按已发布能力采用Python。Transformer在NPU，另有FC16 KV/SSD quad两条Python/NPU观察；均非DDP。规模物理B8、四microbatch累积为全局B32，Transformer同全局B32。FC16 width128/B8/L128实际完整步：CPU native约62.6–63.0秒、Python约89.7–90.0秒；NPU native约232–242秒，Python较小形状也未占优。计时用于执行选型，没有认证全模型严格等价加速。

期限固定为**2026-11-11 20:31:31北京时间**；1536 CPU核时、384 NPU卡时，单筛选≤8h、单规模≤36h；CPU包含NPU的host线程和等待/启动的保守扣账。监督 `v6-budget-v2` 使用实际进程RSS补足本机v1 cgroup资源控制局限，旧时间监督保留；本次观测RSS约356.3GiB、主机可用内存约1473GiB，新增产物约18.6GiB、账号硬配额余量约612GiB。达到期限/预算如实截断，不承诺所有大模型都达到4M，也不回退到旧短曝光来标记成功。

数学0770593、foundation74e7bbd、私有环境`.venv-positive-floor-v1`与250安装文件锁固定。首64覆盖worker源码c458571；其余及修正作业4087310；HB88容量修正源码4b3e88a（96GiB、扣除原已用473秒），数据制作3786de0，native来自相同foundation74e7bbd的独立构建。冻结源码、依赖及活动输入均不修改。普通分支推送，无强推。

保留的启动失败：下载read timeout、HTTP/2取消及一个Range断连；最终复用已有字节并核对原完整hash。首个native调用的trace参数不兼容、18个独立worker输出目录冲突、2个Shannon native能力拒绝均发生在正式学习前，原记录不改写；使用新身份修正。HB88首次大训练达到约48.9GiB，单项内存保护停止、优化器更新0；在总768GiB以内提高该项到96GiB，保持架构/batch/数据/目标，以新作业memory2重新通过检查和恢复并进入训练，原失败保留。严格1e-6/1e-5数值超差保持失败，执行、离散/机制、恢复、学习、资源各轴分开。每个正式候选先通过自身B2/L3独立检查和三个新进程的三步恢复，不能外推完整长轨迹精确重放。

Lens项目`tide-learning-lab`：筛选前缀`tl-v6-cover-`，规模前缀`tl-v6-scale-`，接入检查`tl-v6-harness-`。小型记录及恢复短训投影已verified/readback_matches；正式运行中的投影标为recording，不提前称终态读回通过。私有查看/SSH方法在`.local/LENS.md`，不入公开仓库。项目原始JSONL与检查/状态为权威。

本轮交接允许作业继续后台运行，无自动扩展候选、第二种子、16M目标或B多层替换。终态后按共同token前缀、heldout质量、Full/消息及局部边界审阅，冻结候选再打开test，形成分组一页结论及下一批资源建议；由用户验收后决定下一步。旧v5结论见[原终态报告](studies/language-completion-v5-results.md)，其严格失败及限定结论不升级。
