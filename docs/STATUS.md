# 当前交接：v10局部VJP修复已授权，正在冻结及提交

2026-11-15用户批准[修复执行表](studies/direct-v10-repair.md)。生产路径仅新增显式ordered-v1聚合反向；
13项原路径/局部回归、7项新增及提交契约检查、2项状态分类检查通过（含重复局部检查）。
三条pool从已核对SHA的诊断失败前快照迁移；新进程完整batch恢复及64更新先导通过后，自动续训至原128Mi。
新配置仅变更VJP选项，不改原48h截止、global64/physical8或16、LR/数据和种子；不改变原控制器的256Mi决定。
尚未完成新源码NPU资格或提交；后续实际交接会替换本段。原固定作业不修改，原三项failed保留。
只读修复入口`python scripts/review_v10_repair.py`，原20条用`python scripts/review_direct_v10.py`。

以下为历史交接快照。

# 当前交接：三项pool失败已定位，局部VJP候选通过；原训练保持原身份

2026-11-15 04:36北京时间附近核对。[完整定位报告](studies/direct-v10-gradient-results.md) ·
[回放/算子/候选证据](evidence/direct-v10-gradient-diagnosis.json) ·
[当前主作业](evidence/direct-v10-observations-after-diagnosis.json)。

三条d128 pool在原1524/1102/3555步全部复现，61个共同记录点loss/梯度范数/tokens完全一致。
均为聚合`value/received`的DivBackward0在有限上游下产生非有限梯度；不是仅范数归约问题。
每例34个参数梯度张量被污染；候选仅重排局部VJP，三个完整失败batch前向逐位相同、梯度全有限，
一次optimizer更新通过，范数0.55234/0.59418/0.53265。CPU/NPU抽取算子的FP64参考均过原1e-6/1e-5。
正式模型代码、活动源码、原failed、原严格有限超差均未改；没有自动恢复失败训练。
候选新源码恢复/长期稳定/学习质量仍未检查，下一步建议见ROADMAP。

主轨迹当前12完成128M、5运行、3原失败；控制器running，原48h截止不变。
仍运行：hb-d128-r64-q3-s11 70.02M（物理5）；hb-d128-r64-q3-s22 67.77M（物理15）；hb-d64-r64-q3-s11 115.98M（物理0）；hb-d128-r28-pool-s11 98.68M（物理4）；hb-d128-r28-q10-s11 100.00M（物理8）。
8个诊断服务全部终态且已核对无后代；约1.112累计服务小时，新增诊断及固定源码5.420GiB，账号约298/800GiB。
独立confirmation/test状态不因诊断升级，d128原成组延伸前提仍不满足。

固定：原模型0ea05d61；回放工具95bce59a；候选检查a8669103。复查`python scripts/review_v10_gradients.py`，
主作业仍用`python scripts/review_direct_v10.py`。Lens原三个failed run及私有入口`.local/LENS.md`保持；
诊断不计作新学习轨迹，具体配置、checkpoint/batch/算子见证路径在报告。

以下为此前交接快照，不代表当前状态。

# 当前交接：v10中期，11完成128M／6运行／3pool失败

2026-11-15 03:52北京时间核对。20条主轨迹全部启动过，无排队主任务；11条完成128M，6条运行，3条训练失败。
[中期报告及共同曝光表](studies/direct-v10-progress-20261115-am.md) ·
[实际作业状态](evidence/direct-v10-observations-20261115-am.json) ·
[失败/验证/活动与检查证据](evidence/direct-v10-progress-20261115-am.json)。

三项失败：d128 R64 pool seed11/22最后成功23.22M/16.79M，d128 R28 pool seed22为54.21M；
均触发梯度总范数非有限检查，已停止且后代清空。具体梯度张量非有限或范数归约溢出尚未区分，不能称已定位下溢。
外层worker正常退出造成服务completed；只读审阅器现分列服务与实际训练状态，不再将这3项计为完成。
原失败及全部checkpoint保留，没有复跑/恢复或修改活动源码。

仍运行：d64 R64 q3双seed约103.38M/113.60M（物理0/6）；d128 R64 q3约61.85M/59.41M（5/15）；
d128 R28 q10 seed11约82.67M（8）；d128 R28 pool seed11约87.69M（4）。最慢距128M估计仍需5–7小时。
正式固定源码0ea05d61，检查补证6125c751；截止仍11月16日21:51:33。控制器正常，等主阶段终态再冻结延伸/做独立确认。
按原全组完成规则，d128组因3项pool失败不具备256M延伸资格；d64仍待q3双seed完成。没有更改候选或延伸规则。

共同32M：q10 d64/d128 NLL4.2100/3.8487，两seed均改善；q3为4.7357/4.5930，但seed11变差、seed22改善。
已齐128M窄档：q10 3.7705、小稠密3.5952、TF3.1859，暂未显示近活跃预算下更多总容量的收益。
宽档小稠密/TF128M为3.2159/3.0208；q10只完成seed22，NLL3.3000。均为选择用validation，不是独立确认结论。
活动约10%/3%保持；32M宽档实际活跃参数10.354M/9.291M，差11.44%，不再称严格匹配。物理Full仍稠密遮罩。

20项三步新进程恢复均通过；对齐后NPU复核四形状完成，因果/支持通过，前向2严格通过、2有限超差，全部存在VJP严格失败。
原阈值及错误比较记录保留，这与3项非有限训练失败分开。262K独立确认未开始，test仍密封。
Lens：14终态投影verified/readback_matches（含3项失败），6项recording。只读`python scripts/review_direct_v10.py`。
当前新增约21.74GiB、账号292.09/800GiB、硬余量507.91GiB。4次设备采样AICPU均0%，AICore中位12–23%，AIVector63–66.5%。
[查看/停止/恢复](studies/direct-v10-operations.md)与私有Lens入口`.local/LENS.md`保持。

以下为历史交接快照；旧状态不代表当前运行。

# 当前交接：v10已固定并后台执行，12训练/8排队

2026-11-14 22:25北京时间附近核对：20个正式服务已接收，12条进入真实训练、8条排队；
已启动12条均通过各自三步新进程精确恢复，12个Lens投影recording，无正式训练失败。
控制器及各服务均在background.slice。另有1项同日程准入复核排队，遵守相同12slot锁。
[研究表](studies/language-capacity-activity-v10.md) · [实际逐作业状态/卡号](evidence/direct-v10-observations.json) ·
[校准/参数](evidence/direct-v10-selection.json) · [性能与检查](studies/direct-v10-optimization.md) ·
[查看、停止、恢复及Lens](studies/direct-v10-operations.md)。

共同截止 **2026-11-16 21:51:33 +08:00**；无累计CPU/NPU算时上限，最多12单卡worker。
正式源码 **0ea05d610df82df5b55663cbcd2fa4bcc92c3a6d**，`.local/source-v10-execution`；
检查补证另用6125c751，不改任何活动输入。安装74e7bbd及数学0770593保持，不接入上游新优化。
冻结配置`configs/studies/language/direct-v10/table/`，冻结预算/资源在同目录上一级。

|构型（均seed11/22）|当前物理设备/状态|
|---|---|
|d128 R64 pool|1 / 7，训练|
|d128 R64约3% quad|5 / 15，训练|
|d64 R64约3% quad|0 / 6，训练|
|d128 R28 pool|4 / 14，训练|
|d128 R28约10% quad|8 / 2，训练|
|d64 R28约10% quad|9 / 10，训练|
|两档小稠密、两档校准TF，共8条|已接收，等待slot/显存；结束即释放并动态取得下一项|

全部L256/global64，参数3.46–171.46M；四稀疏构型为22.24/50.32/75.54/171.46M，
小稠密为3.756/9.668M。共同8/32/128M检查点；两宽度组依完成/有限性/剩余时间整组延伸256M，
不是按学习排名选择。所有起点即使用256M LR日程；262K独立确认在决策冻结后评价，test密封。
前2M为活动warmup，不能说整段训练都在3%/10%。物理Full仍全节点稠密后遮罩。
目前只有短校准及正在增长的曝光，尚无正式容量或语言质量结论。

NPU特化已测：最大171M真实短程约3650有效目标/s、峰值30833MiB；同B4旧/新约1027/3059位置/s，
但attention floor策略不同，不能称严格等价加速。FFN重算未改善吞吐，不选。
八次设备[采样](evidence/direct-v10-resources.json)：AICPU均0%，各卡AICore中位11–20.5%，
AIVector中位60–65%；仍非矩阵核充分利用，设备共享和host成本分别保留。
trace汇总API失败原样保留，导出的30次FlashAttention前向/反向及算子CSV为有限路径证据。

准入口径补证：原quad检查NPU已进入warmup，CPU只载参数却未同步set_progress，出现大幅无效对比。
保留原failed，不把它解释成普通浮点误差；新CPU回归通过，同日程NPU四形状复核`v10-admission-aligned`
已排队，最晚11月15日01:51:33结束。原fusion两项有限严格失败仍保留；各项执行、严格数值、离散机制、恢复、学习分列。

当前账号270.50/800GiB，硬余量529.50GiB；新批参数checkpoint预计26.55GiB，含全延伸44.25GiB，
加记录/临时峰值后受64GiB目标/96GiB上限、至少200GiB硬余量约束。v9及历史失败没有删除。
只读状态：`python scripts/review_direct_v10.py`；控制器日志`artifacts/jobs/v10-controller/job.log`；
Lens项目tide-learning-lab、TL-CAPACITY-v10/TL-ACTIVE-v10/TL-CONTROL-v10，入口`.local/LENS.md`。
停止/恢复命令见操作页，不改冻结源码。普通推送后的分支文档更新不改变训练身份。

以下是历史交接，旧“无活动任务”等表述只描述当时状态。

# 当前交接：v9训练全部终态；独立确认待完成

2026-11-13 14:16北京时间复核：[总报告](studies/direct-v9-results.md) ·
[规模/曝光](studies/direct-v9-scale-results.md) · [低活动](studies/direct-v9-activity-results.md) ·
[入口树](studies/direct-v9-inlet-results.md) · [检查/成本](studies/direct-v9-controls-results.md) ·
[逐轨迹、父SHA、Lens及文档配对证据](evidence/direct-v9-results.json)。

58/58主轨迹和12/12条件延伸完成，19项batch校准和72项学习筛选完成；目前无运行/排队或存活后代。
控制器11月13日10:21:22完成，原共同截止21:36:26保持。72个服务65完成/3历史失败/4历史停止；
142条完成训练=72筛选+58主+12延伸，另保留4次原budget_truncated。恢复和延伸不计独立seed。
四次中断来自quota软告警误判；原失败不覆盖，没有新增模型执行失败。

双seedvalidation：HB pooled40/88/184在64M为3.4179/3.2596/3.1433，quad为3.4767/3.3518/3.2240；
浅宽约50M HB最好3.0112，近参数TF2.9457。HB88 pooled/quad与TF在256M为3.0541/3.1366/2.6804，
均继续改善但与TF差距未缩小。容量/深度同时变化，不能归因于纯深度或普遍scaling。
R64 q3在64M NLL4.4996，Full3.45%、接收16.61%；同形状pooled4.1092、Bern4.1150。
R32低活动h2→h3质量变差4.6423→4.7773；树两组均减少接收但NLL变差，R64含边界Full5.22%。
物理Full仍全部稠密遮罩，好的质量和最低活动未在同一配置结合。独立262K确认集未测，test仍密封。

训练源码6e7bd49e、恢复控制器1809009a、数学0770593、安装74e7bbd保持；没有采用基座新优化。
70个终点完整validation/有限生成、142条完成训练Lens读回均通过；原严格数值失败/阈值保持。
三步新进程检查和16条实际恢复/延伸父SHA核对通过，不认证完整长轨迹独立精确重放或公共continuation。
本批终态磁盘57.90GiB/64GiB；账号269.61/800GiB、硬余量530.39GiB。历史74.76GiB清理另记。
成本为记录的共享设备租期302.89卡时、服务线程配额1267.76核时，不等于实际忙碌算时。

只读：`python scripts/review_direct_v9_results.py`；原`review_v9_progress.py`仅汇总58主轨迹，不含12延伸。
Lens项目tide-learning-lab，TL-SCALE-v9/TL-ACTIVITY-v9/TL-INLET-v9/TL-CALIBRATION-v9；
延伸run前缀tl-v9-recover-extend-，精确投影见证据，私有入口仅.local/LENS.md。
没有自动下一批。建议先冻结关键checkpoint做独立确认，再将浅宽质量参照与低活动联合；
保留入口树/加深的负面证据。TL-REQ-006待上游定位/交付，性能不作为继续质量研究前置门槛。
以下保留历史快照，不能用旧running字段判断当前任务。

# 历史交接：11月12日存储清理与运行中快照

2026-11-12 20:20存储交接：[清理报告与保留范围](studies/checkpoint-retirement-20261112.md) ·
[逐文件执行凭据](evidence/checkpoint-retirement-execution-20261112.json)。已释放74.7565GiB，
本目录约65.74GiB、GEF约1.618GiB，账户约262.01/800GiB。
整个v9、所有非completed轨迹和失败复现、关键对照、配置/结果/日志保留；历史筛选权重已选择性退役。
原恢复通过记录不承诺退役字节仍可取得，后续加载先查清单。Lens与活动训练源码/输入保持。
清理后5个活动v9服务对应的2份固定源码和18个注册输入hash一致；本次不更新科学结论或启动新训练。

## 上次研究进度汇报：15:54快照

2026-11-12 15:54北京时间附近：[下午汇报](studies/direct-v9-progress-20261112-pm.md) ·
[逐轨迹/配置/恢复谱系](evidence/direct-v9-progress-20261112-pm.json) · [故障与恢复](studies/direct-v9-recovery-20261112.md)。

19项batch校准、72项筛选完成；主58条50完成、8运行、0未启动。规模30条收齐，活动16完成/4运行，入口4完成/4运行。
原控制器15:19因quota软告警返回1失败，停止4条训练。账号约300GiB仍在6天宽限，硬余量约499GiB，
本批约44.4GiB。修正监控后，15:50新控制器与12worker实际启动；4条恢复越过原checkpoint，4条首启TF已完成。
原控制器failed及四条budget_truncated保留，这是调度故障，不是学习失败或原预算耗尽。

新controller 1809009a，实际训练仍6e7bd49e；数学0770593、安装74e7bbd和配置/data/optimizer保持。
活动物理卡1/11为R32h3 quad恢复，7/15为R64 Bern恢复，4/14为R32 quad入口，8/2为R64 Bern入口。
均在background.slice；共同截止仍11月13日21:36:26，约余29h40m。条件延伸尚未启动，控制器按原条件推进。

64M双seed：HB pooled40/88/184 NLL3.4179/3.2596/3.1433；quad3.4767/3.3518/3.2240，
对应TF2.9978/2.9027/2.9457。浅宽50M HB3.0112仍好于深HB，但TF领先。FC16M匹配TF已齐，均优于当前FC。
R64 q3入口树16M NLL4.8274、原入口4.7907；主体接收9.54%对16.48%，主体Full3.95%对3.34%，
含边界全图Full5.22%。减少接收已成立，质量收益未见；仍物理稠密遮罩，独立confirmation/test未开。

原严格数值失败及阈值保持，实际恢复推进不等于长轨迹独立精确重放。四条旧中断Lens已另存收尾证据，
旧projection记录保留；新run前缀tl-v9-recover-main-，组名及私有入口不变。
AICore/AIVector/AICPU分析已交接[TL-REQ-006](requests/TL-REQ-006.md)，基座与训练代码未接入新优化。

只读：`python scripts/review_v9_progress.py`，`python scripts/job.py status artifacts/jobs/v9-recovery-controller`。
旧`review_direct_v9.py`会同时显示原failed控制器与新recovery状态；不能只看原campaign字段。
停止：`systemctl --user stop 'tide-lab-v9-*.service'`，随后核对后代。不得修改两套活动固定源码或配置。

# 2026-11-11启动时交接（历史）

2026-11-11 21:50北京时间附近复核。[固定执行表](studies/language-scale-activity-v9.md) ·
[操作/Lens/停止/恢复](studies/direct-v9-operations.md) · [实际观测](evidence/direct-v9-observations.json) ·
[接入与batch校准](evidence/direct-v9-admission.json)。

共同截止 **2026-11-13 21:36:26 +08:00**；无累计NPU/CPU算时上限，最多12个单卡worker。
训练固定源码 **6e7bd49e**（`.local/source-v9-execution`），数据准备bd666d9a；数学0770593、安装74e7bbd。
只修改工作树文档不影响活动源码。控制器`v9-controller`在background.slice，执行未来阶段已受同一截止约束。

数据完成：128M=134,217,728有效目标池，25,440文档；原validation258,990/test294,476原字节保留；
另有新confirmation262,144/135文档，尚未用于选择。test密封。
数据锁SHA40b7c0ebb9650d11aa0c0080532c884e2ca3846db578155f3df7068b82191d2f，目录runs/v9-data-r3。
原缺tokenizers解释器失败与直连超时分别保留在v9-data/v9-data-r2，r3复用既有数据环境并单次netpath重试。

19项实际形状NPU完整更新校准全部完成，global batch64/L256；按内存和有效吞吐选micro：
HB40/88/184为32/16/8，FC16/32/64为64/64/32，宽区域R32/R64为8/4，树对应8/4；TF各档64。
完整样本与峰值在接入证据中；性能是探索性计时，不是跨后端数值等价认证。

72个有限前缀已交12个worker：12 HB配方、36 TF配方、24活动终点校准。
实际使用物理卡0、1、2、4、5、6、7、8、10、11、14、15（共享状态以queue-state为准）。
首批12个HB配置接入通过，均保留CPU/NPU严格有限数值失败；首批12项三步新进程恢复全部通过，12项均已进入真实训练；长轨迹精确重放仍未检查。
执行、数值、离散、恢复、学习、性能分列；目前不产生语言/规模结论。

之后控制器在有限规则内自动冻结选定配置，执行58条主轨迹：30规模/容量/深度、20活动、8入口树。
原入口复用活动组对照。主体参数约5–55M，主终点16M/64M；条件最多六条中档续至256M、
六条同形状活动对照续至64M。未触发条件保持未检查，不改变问题或扩候选。

Lens项目tide-learning-lab，小型记录已verified/readback_matches，四个组TL-CALIBRATION-v9、TL-SCALE-v9、
TL-ACTIVITY-v9、TL-INLET-v9；私有入口仅.local/LENS.md。
只读`python scripts/review_direct_v9.py`；训练日志在runs/v9-<phase>-<id>/train.log，
停止全批`systemctl --user stop 'tide-lab-v9-*.service'`，再只读核对各服务后代清空。
不要修改活动配置/快照，恢复使用原latest.pt+原配置/检查/恢复证据与新输出。

以下为历史对齐与终态记录，过时的“未授权/未提交”不限制当前明确授权的v9执行。

# 当前交接：v9主体获基本同意；48h无聚合算时限，入口/出口边界对齐

2026-11-11最新用户反馈已写入[研究章程](RESEARCH_CHARTER.md)，README/AGENTS要求重入阅读。
核心目标是同模型语言next-token质量、数据/容量/深度扩展、低活跃与局部化；允许继续用更快的稠密遮罩，
逻辑活动、实际计算和硬件吞吐分开，实际稀疏kernel不是后继质量研究门槛。

[v8结果重审与下一批建议](studies/language-scale-activity-v9-proposal.md)：
共同4M/16M/64M里程碑、中档三方最多256M成组延伸；HB40/88/184、FC16/32/64、浅宽与重新校准TF；
用户接受48h总墙钟，取消NPU卡时和CPU核时上限；64GiB新增磁盘目标保留，当前尚未提交。
[HB低活跃对齐](studies/hb-low-activity-alignment.md)纠正分母：Full/全图、接收/全图、Full/候选分列。
当前HB已支持部分接收；v8 hard-selection分别约36.7%/75.0%/48.9%，物理Full仍100%。
约10%建议R32、约3%建议R64保留约两个发送者及可能的评分梯度；都是待对齐配置，非学习结果。
[本次证据](evidence/hb-activity-alignment.json)新增四个CPU固定路由、KV无接收/未发送记忆和quad局部VJP检查；没有新训练或NPU作业。
本次没有训练、下载或设备分配；静态/小型机制核对不是新规模学习，研究图中的v9三组仍planned、结果awaiting。
后续用户基本同意低活跃方向，并提出1→4→16入口与汇总树。
[边界拓扑说明](studies/hb-boundary-topology-alignment.md)核对当前区域数1→2→4→2→1，
相同单位入口是所选HB实例的边界规定，不是TIDE一般限制。入口树为新候选，接入覆盖/输出/固定开销需显式定义，尚未加入运行表。
[HB三轮解释](studies/hb-evidence-reconciliation.md)明确：旧配方/曝光/性能确有问题，尚无证据把旧HB88劣化归因于基座算错；
v7简化图容量收益不改写旧结果，v8还没有HB88/184训练。

以下保留v8终态身份与原交接；后继优先级以新建议为准，历史批次和固定输入不变。

## v8已完成的交接

# 当前交接：direct-v8 广覆盖与有限复核已终态

2026-11-11 19:24:47北京时间复核：[终态交付](studies/direct-v8-results.md) · [首轮结论](studies/direct-v8-coverage-results.md) · [种子复核](studies/direct-v8-confirmation-results.md) · [Transformer调参](studies/direct-v8-transformer-results.md)。

本批18:49结束，40个服务均终态且无后代，当前没有本批运行或排队任务。
182项首轮中157完成/25失败；19项局部seed22中18完成/1失败；18项TF全部完成；单项r2完成。
共220次训练尝试，194次达到1,048,576目标、260更新并完成32,768目标validation与有限输出推理，26次原失败保留。

双seed平均validation NLL：FC direct-attention 5.2639、HB pooled-attention 5.2090；
各自anchor为5.3665/5.4033，校准TF6/12为5.7874/5.8340。仅限本轮近参数、1M及参与选择的validation，test密封。
FC/HB quadratic温度1均值5.3102/5.2876，逻辑Full/候选约58–62%，是后继稀疏候选；实际Full仍全部稠密计算。
FC mixed seed11为5.2149，但seed22在1,046,992目标失败，不补填最终结果。

26次原失败均为quad分配可行性残差>2e-5（约2.04e-5至0.652），6次在首个optimizer更新前；不能等同于学习负面。
其中FC gated_delta/state相对坐标求根r2完成1M，NLL5.3088；未覆盖其余25个未完成身份的原因/修复。
220项小型同栈三步新进程恢复均通过，220项均保留至少一个CPU/NPU严格数值失败；长轨迹独立重放、公共continuation未认证。

固定源码TF2702d91、anchor778f9e9、覆盖/确认5db28ae、r2 cd85637；数学0770593、规划/安装74e7bbd、私有环境保持。
配置SHA已核对实际identity，19份自动生成confirmation配置已归档。未使用基座新优化，未修改历史活动输入。
全批监督64.7分钟，保守8.204 NPU卡时/20.289 CPU核时，新增54.28GiB，终态配额余量577.57GiB。
220个训练Lens投影均verified/readback_matches；项目tide-learning-lab，三个TL-DIRECT-*-v8组保持，私有入口仅`.local/LENS.md`。

[逐运行状态/Lens](evidence/direct-v8-observations.json) · [质量/工作量/配对统计/失败](evidence/direct-v8-results.json) · [研究图](../web/index.html) · [操作方法](studies/direct-v8-operations.md)。
只读：`python scripts/review_direct_v8.py`；结果重建：`python scripts/review_direct_v8_results.py`，均不重跑模型。
下一步可选数值补齐、真实稀疏、数据/容量阶梯，有限范围与资源建议在总交付；尚未提交，不自动扩大本批。
已完成KV候选无需等待所有失败修复；B预训练路线没有新增作业。

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
