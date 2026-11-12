# 当前交接：v9主训练36完成、8运行、14未启动；硬件利用已留证

2026-11-12上午只读复核：[中期汇报](studies/direct-v9-interim-20261112.md) ·
[共同曝光/活动/逐运行](evidence/direct-v9-interim-20261112.json) ·
[AICPU/AICore原始采样](evidence/direct-v9-resources-20261112.json) · [操作/Lens](studies/direct-v9-operations.md)。

19项batch校准及72项学习筛选全部完成。主58条：36完成、8运行、14在固定worker序列中尚未启动；
没有训练执行失败，无自动重跑。四个worker已释放卡，当前活跃物理0、1、2、4、6、8、10、14。
控制器继续运行，条件延伸尚未触发；截止仍11月13日21:36:26 +08:00，约余36h。
固定训练源码6e7bd49e未改动；数学0770593、安装74e7bbd；上述采样没有新训练、profiler或上游修改。

后续用户授权将设备占用问题交接基座：[TL-REQ-006](requests/TL-REQ-006.md)记录固定v9证据、
AICore/AIVector/AICPU口径、待核查机制及验收建议。基座目录新增同编号handoff；待基座回复和下游验收，
未修改基座执行代码或活动作业。下列运行数量仍对应09:35中期快照，不是此次文档交接的重新采样。

64M、双seed、同131072目标validation：HB40 pooled NLL3.4179，HB184 pooled3.1433；
浅宽HB40/d256/FFN1280约50M参数为3.0112，近参数TF为2.9457，TF约25M为2.9027。
已有本配置族容量/数据收益，但深度不自动更好，TF仍领先；不能声称联合目标已达成。
R64/h2在16M：稠密NLL4.3774，Full10.58%为4.6415，Full3.34%为4.7907，固定轮转3.125%为5.0550。
低活跃两seed达带且优于该固定控制，仍有质量代价；物理Full100%稠密遮罩。入口树还缺16M完整对照。

8轮/37.67秒设备采样：HB88 pooled的AICore中位3%、AICPU30–31%、总NPU94.5%；
R32入口树Bern为AICore2%、AICPU33%、总NPU96.5–100%。不能据总占用宣称算力充分利用。
样本是设备级，物理1另有进程；不同计数器不能相加或充当kernel耗时比例。热点归因须后续短trace。

116条已启动训练均有小型机制/三步新进程恢复通过，也均保留至少一个CPU/NPU严格有限超差。
108条完成Lens投影verified/readback_matches，8条录制中；严格阈值未变，长轨迹精确重放未检。
confirmation/test未开，原两次数据准备失败保留。账号约275GiB/800GiB，新批约39GiB/64GiB。

继续只读检查`python scripts/review_direct_v9.py`。停止全批：
`systemctl --user stop 'tide-lab-v9-*.service'`，再核对MainPID/后代；不要改活动快照/配置。
详情、原始计数、对照、未知原因和后续基座分析线索均在中期报告。

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
