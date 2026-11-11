# 当前交接：HB成本实测与学习审阅完成

2026-11-11：[性能结论](studies/hb-cost-results-20261111.md)、[学习结论与后继建议](studies/hb-learning-results-20261111.md)已完成。
25个成本case（每格三个模式、每模式三次正式重复）和八个只读权重VJP全部完成；
四个background.slice服务退出，当前MainPID/后代清空，33个Lens投影核对身份、事件数和读回通过。
[成本/作业/Lens证据](evidence/hb-cost-review-20261111.json) · [共同曝光原曲线](evidence/hb-learning-review-20261111.json) ·
[梯度证据](evidence/hb-gradient-review-20261111.json) · [初始化分析](evidence/hb-initialization-20261111.json)。

共同B4/L32/d128下，HB40直接Attention更新CPU1/NPU约0.244/0.047秒，固定TIDE约20.20/64.03秒。
直接图省略TIDE部分数学/共享/控制和公共返回职责，倍率不是纯开销或严格等价加速；
原TIDE严格失败、离散/恢复边界保持，计时不产生学习资格。基座工作树和新优化未接入。

学习审阅：HB88在同32,768目标validation上133K→262K的NLL从11.7535升到12.2515；
HB40后期改善但同曝光仍落后部分对照。FC32也早期恶化后恢复、d32 HB proportional较好，
所以不能升级为HB架构总体失败。当前固定std0.15、缺最终读出归一化与强方向梯度是具体待干预线索。
后继建议先独立校准Full矩阵fan-in和最终RMSNorm，再让同配方HB40/88、FC16及校准Transformer进入1M/4M共同曝光；
用proportional/必要的稠密发送区分路由与拓扑。此建议尚未提交新语言训练，也未打开密封test。

成本固定源码b2e85cb，梯度固定源码8bb7ab9；安装74e7bbd、数学0770593，
`.venv-positive-floor-v1`与`.local/native-build-v6`均复用。实际wall 30.53分钟，
保守CPU 1.567核时（含NPU host）、NPU 0.509卡时。
作业ID：`hb-cost-20261111-cpu-main`、`hb-cost-20261111-cpu-eight`、`hb-cost-20261111-npu-main`、`hb-gradient-20261111`。
读取状态/日志用`python scripts/job.py status artifacts/jobs/<ID>` / `logs`；固定输入未改。
成本复测需要新job/output，不存在本轮可续训checkpoint；审阅可用`python scripts/review_hb_cost.py`只读重建。
Lens项目`tide-learning-lab`：`TL-HB-COST-20261111`（前缀`tl-hb-cost-`）与
`TL-HB-LEARNING-20261111`（前缀`tl-hb-grad-`）；私有入口仅在本机`.local/LENS.md`。

共享模型根目录已按用户操作改为`/mi/data2T/models/huggingface`，
[路径表](../dependencies/public-model-paths.json)与[迁移复核](evidence/public-model-relocation-20261111.json)记录六个模型、102个原文件。
六个旧别名已直接指向新位置；inode/device/size/mtime与原完整hash清单一致，未改模型字节。
不需原集合外壳，旧sudo发布脚本已停用。历史610文件/46.185GiB清理保持已完成，科学checkpoint/失败保留。

v6原200配置均已终态；本次HB限定审阅没有把全组RV6-COVER/RV6-SCALE或密封test结果槽自动填为通过。
TL-REQ-005仍按原请求/待交付记录保留；本轮没有修补公共执行。研究图新增成本/学习两个研究组及其限定结果。

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
