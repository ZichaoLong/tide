# 当前交接：direct-v8 广覆盖已在10张NPU执行

2026-11-11 18:17:27北京时间观察：[有界研究表](studies/direct-coverage-v8.md) · [查看/停止/恢复与Lens](studies/direct-v8-operations.md) · [实际逐配置状态](evidence/direct-v8-observations.json) · [研究图](../web/index.html)。

已实现并提交91个具名局部配方×FC/HB=182格，d128/L128/global B32，各1,048,576有效目标。
本次是独立直接数学模块，包含记忆/描述/分配；物理Full稠密计算后遮罩，不声称真实稀疏执行或公共continuation等价。
两anchor已经完成，另外180格由10个后台worker顺序覆盖；另有一个具名修正排队。当前局部尝试状态（含r2）为{'completed': 72, 'running': 5, 'pending': 95, 'qualifying': 5, 'failed': 6}。
所有活动训练/等待均在background.slice。研究地图保留结果awaiting，不凭退出码生成支持。

Transformer 14格调参＋4格seed22复核均完成1M：[一页结论](studies/direct-v8-transformer-results.md)。
选择GPT2式std0.02、sqrt(2·层数)残差输出缩放、LR0.001、AdamW(.9,.95)/矩阵decay0.1。
TF6/12双seed平均NLL为5.7874/5.8340，seed11旧配方为6.1584/6.1940；改善是配方包效果，不是最优Transformer认证。
FC/HB anchor的seed11最终NLL为5.3621/5.3854，均从约8.4下降；仍不能提前认定广覆盖优胜或普遍scaling。

182格实际CPU小型前向/梯度/因果检查通过，8项独立数学检查通过；13个NPU/12个CPU成本case完成。
HB40完整B32更新NPU约0.296秒、CPU1约9.83秒。B64吞吐更高，但本轮固定global B32保留短程更新与TF对照口径。
首轮正式接入保留CPU/NPU严格1e-6/1e-5有限失败；因果、支持、共享optimizer、三步新进程恢复及学习各轴单列。

已保留的失败：FC gated_delta/state训练出现分配可行性残差3.1233e-5>2e-5；其他任务继续。
[有界排查](evidence/direct-v8-quad-refinement.json)发现大评分偏移的减法消减可造成同量级残差，
活动集相对坐标求根在768个合成行上把最大残差降至约1.19e-7，原阈值不变，九项数学检查通过。
原坏行未落盘，原因未完全确认；只对该候选排队一个r2新身份复跑，未改活动源码。
截至本快照共6项失败，均为FC gated_delta的分配可行性残差：state、conv1/8、heads2、mem-off、ffn-off。
其中heads2/mem-off/ffn-off在初始validation、尚未训练时失败，最大残差0.65224；conv1/8分别完成16,200/594,196目标后失败。
小型接入/三步恢复通过不能认证长序列执行；这些不是统一的轻微超差，未证明与首个失败同因。
全部原失败保留并排除后继种子选择，不能当作已成立的学习负面结论；仅state有一个r2排队，其余没有自动复跑。

TF源码2702d91；局部校准/anchor 778f9e9；180格覆盖及有限确认5db28ae；单项r2 cd85637。
数学0770593、规划/安装74e7bbd、私有`.venv-positive-floor-v1`保持，未接入基座新优化。
截止2026-11-13 17:43:58，240NPU卡时/768CPU核时/128GiB新增磁盘；启动账号168GiB/800GiB。
监督v8-budget；覆盖v8-cover-w0..9；修正v8-repair-w0等待同一10卡资源槽。
v8-confirmation-controller已等待覆盖终态，随后按预声明桶最多20个seed22，不自动打开真实稀疏或规模主线。

Lens项目tide-learning-lab：TL-DIRECT-TF-v8、TL-DIRECT-COVER-v8、TL-DIRECT-CONFIRM-v8；
新小记录verified/readback_matches，正式训练持续投影。前缀tl-v8-；私有入口和SSH方法仅本机`.local/LENS.md`。
只读快览：`python scripts/review_direct_v8.py`。查看不需重跑；各运行的原始失败、checkpoint、配置、安装身份均保留。

## 前一轮已完成的交接

# 当前交接：direct-v7已完成，直接HB出现有限容量收益

2026-11-11 16:51北京时间终态：[总交付](studies/direct-language-v7-results.md) ·
[配方结论](studies/direct-v7-recipe-results.md) · [扩大曝光结论](studies/direct-v7-scale-results.md) ·
[固定证据/配置/运行/Lens](evidence/direct-v7-observations.json) · [学习与容量图](evidence/direct-v7-learning.svg)。

12格262K配方筛选、10条双seed主轨迹均完成4,194,304有效目标；4条主轨迹继承screen，合计18次独立神经初始化。
共同选择fan-in、无最终RMSNorm。4M两seed平均validation NLL：FC16 4.6785、HB40 4.6126、HB88 4.5342；
近参数Transformer12/28为5.1666/5.4658。HB88−HB40两seed各约−0.079，配对文档区间均低于零。
1M时seed11差异区间仍跨零；结果仅限该直接函数/配方/语料/规模，不能称为普遍scaling规律。
Transformer配方仍弱，且两档TF落后固定二元参照；不能据此声称优于可靠Transformer架构。

本轮使用普通Torch稠密图，**不包含TIDE路由、Read/Describe、payload或可学习稀疏发送**。
它给A路线提供条件配方/拓扑线索，不覆盖旧TIDE HB88负面结果，不替代B预训练路线。
残差补救条件未触发，两条控制未运行、效果未检查；test仍密封。

训练固定源码d30559c（`.local/source-v7-main`）；数学0770593/安装74e7bbd、私有环境保持。
19项模型/数据/生命周期测试、等待预算回归和解析参照seed隔离回归通过；18套新配置三步新进程精确恢复通过。
原CPU/NPU严格1e-6/1e-5参数VJP有限失败保留，因果/共享/optimizer/有限性通过；长轨迹独立精确重放未检。
解析参照首次seed22复用缓存的错误以原记录保留，新b463f80/r2身份修正；各神经训练独立进程，不受影响。

15个服务全部终态、无后代；**当前没有本批活动或排队任务**。
全批跨度13.23分钟、0.685 NPU卡时；全部服务1.851 CPU核时，含0.5开发预留共2.351核时，终态新增约14.45GiB。
没有采用基座新优化，没有更改活动源码，也没有自动追加更大批次。

Lens项目`tide-learning-lab`：`TL-DIRECT-RECIPE-v7`的12筛选+2校准、
`TL-DIRECT-SCALE-v7`的10主训练+2修正解析参照均核对读回。原2个解析投影保留，其中seed22样本身份错误单列。
run前缀`tl-v7-screen-`、`tl-v7-main-`、`tl-v7-count-controls-r2-`；私有入口仅本机`.local/LENS.md`。
[查看/日志/停止/恢复](studies/direct-v7-operations.md)，只读快览：`python scripts/review_direct_v7.py`。
原checkpoint、资格记录、父SHA和冻结配置均保留；终态后不自动续到8M或新候选。

下一批建议：选定HB配方扩大同训练池曝光；少量TF LR/残差初始化校准与保留位置的上下文检查；
上游可用后同配方稠密TIDE桥接，再测真实稀疏。建议和单项资源上限在总交付中，待用户验收后再对齐。

## 前一轮已完成的交接

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
