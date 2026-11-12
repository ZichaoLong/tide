# 历史验证与测量证据

[LH/HB同材料三层成本（2026-11-12）](lh-three-layer-20261112.md)：c83a2820的176节点CPU12/NPU10点。
直接公式/公共native/默认native训练CPU0.880/2.964/3.063秒、NPU0.608/7.123/7.496秒；
相同材料与实际候选数核对，dense Full和selected Full工作量及完整观察职责分列，不做原LH等价速度比。

[对齐消息/区域列资格（2026-11-12）](aligned-columns-20261112.md)：129bbbdc固定CPU99通过/9原数值xfail，
NPU half15通过及六安装消费者三步训练/新进程恢复。FC16完整步CPU1.272/NPU0.618秒；
完整尺度严格数值与TTFT未测，不以局部装配修改解除旧resident/异步限制。

[Resident共享矩阵、局部容量与服务停止（2026-11-12）](resident-shared-bank-20261112.md)：3e926ede单bank整块投影/伴随、e12aec0c父子服务停止；CPU32通过/6设备skip，独立SDK及安装后公共根/训练/恢复分列。原容量见证的两次失败保留；原预算SSD训练/精确恢复通过，13.329秒/61输入位置；FC16仍EE1019/507002，未取得吞吐，不能把小图通过称为规模解除。

[FP16张量窗口与跨owner后继（2026-11-11）](column-half-20261111.md)：5816a4c4精度实现、5959bbc3默认实验接入、ce50a2a5 SDK入口；CPU定向188通过/6原xfail，扩大回归112通过/2原xfail/2旧sum失败。单NPU五记忆half及CPU/NPU八安装消费者通过；双NPU普通异步EE1017保留，显式同步全根/安装/SDK另配通过，不作异步或吞吐资格。

[多payload owner张量窗口（2026-11-11）](column-owners-20261111.md)：796baafa实现、efc2776b SDK入口；固定CPU243通过/13原数值xfail，独立SDK CPU通过。两次NPU准入超时均未计算；跨设备资格与性能仍待验。默认FC16两客户端CPU成本单列。

[张量窗口混合放置与有序共享伴随（2026-11-11）](column-placement-20261111.md)：540ccbd8/2ca00458固定CPU、NPU安装训练/恢复及SDK15项；Python KV mixed-a原精确恢复失败由局部共享使用归并修复。9项混合成本与主PyTorch FC16成本单列。

[公共实验张量输入/loss与有限代表（2026-11-11）](experiment-columns-20261111.md)：b993f38固定133项、8安装消费者；18个宽/多节点训练成本、24层链三模式、约1B/9B四推理点。原scalar loss检查器失败、NPU三个精确恢复失败与规模未验范围分列。

[卷积有序转置与精确恢复（2026-11-11）](column-tap-transpose-20261111.md)：8c72ca94固定74项、独立SDK；无全局deterministic的六个NPU组合各两个新进程精确恢复通过，四个安装消费者另列原deterministic=true测试策略。

[五记忆native张量窗口资格（2026-11-11）](recurrent-columns-20261111.md)：b860e6a固定CPU306通过/13原数值xfail，24个CPU/NPU安装消费者三步训练与新进程精确恢复通过；standalone五记忆CPU实跑分列。

[KV位置列资格（2026-11-11）](column-positions-20261111.md)：9fa852c固定CPU、六安装CPU/NPU消费者与独立SDK；相对旋转/距离、int64时间、三步训练与新进程精确恢复，原数值xfail保留。

[张量窗口整链与FC成本（2026-11-11）](tensor-windows-20261111.md)：01cde3b固定CPU232通过/13具名数值xfail，CPU/NPU六安装消费者训练/精确恢复和独立SDK通过；50成本点/8独立profile。FC16 native完整训练CPU1/NPU为1.2863/0.6788秒，直接FC1.3706/0.6978秒；完整尺寸严格资格和其他路径另列。

[区域张量列及结构支持伴随（2026-11-11）](regional-columns-20261111.md)：2d2f8de公共列接口及默认区域adapter，CPU两精度203项、一般图/训练恢复FP64 27项、standalone手算核通过。原FP32 PDG超差保留；单次FC16 CPU1完整训练5.213秒，未声称默认整链贯穿或新NPU资格。

[KV查询块默认接入、三层复测与重复诊断（2026-11-11）](kv-query-blocks-20261111.md)：5ff3ce2块接口，598b4fb固定计时/profile，206bc9b后备预检与诊断附件。FC16 CPU/NPU完整训练11.601→5.835、15.124→10.928秒，直接FC仍1.390/0.675秒；局部根/三客户端训练恢复与大尺度任务投影分列。原FP32严格失败保持，同FP32材料提升FP64通过；提供不绑定作业名的成本配对、profile/census、JUnit/终态复查。

[FC4/FC16三层完整对照与架构调查（2026-11-11）](fc-three-layer-comparison-20261111.md)：15d391f主实验、b0edf96同材料精度诊断、fd243f4算子试验；CPU/NPU三模式、私有根省略、真实kernel与同步阶段差额。完整训练FC16默认/直接CPU12.414/1.414秒、NPU16.525/0.745秒。完整FP32严格失败及FP64两个微小梯度残差保留；包装省略和SDPA未作为生产优化，状态bank/块伴随为后继接口边界。

[FC16/LH模式覆盖与增量成本换算（2026-11-11）](performance-accounting-20261111.md)：仅审阅既有e6f7bd5/0e6916d等记录，无新运行。区分实际输入位置、首token、三种模式缺口、默认/直接公式的阶段差额及小用例单侧诊断；历史LH/1B/9B保持原身份。

[有状态加法基线、默认Full共享bank与借用计费（2026-11-10）](additive-interface-qualification-20261110.md)：e6f7bd5计算实现、0e6916d实际调用观察、57439d7 half拒绝见证；CPU493通过/7xfail/4原失败及后继分列，NPU配置37通过/1xfail/2既有CPU辅助失败。三客户端训练/恢复、投影基线及resident真机范围分别记录，不认证全规模或新resident训练吞吐。

[加法接口三层有限成本（2026-11-10）](additive-interface-performance-20261110.md)：CPU/NPU局部/层链/FC共42个成本点、12个代表训练/精确下一窗及NPU约1B推理/精确下一窗。FC16训练12.283/15.511秒、首token2.737/4.055秒；局部bank前向更快，独立快照反向更慢，NPU完整训练未改善。图统计和阶段诊断独立，原census启动失败及补验分列。

[通用Describe、循环查询裁剪与常驻诊断（2026-11-10）](interface-heads-qualification-20261110.md)：554f2f5的CPU490通过/14保留xfail；NPU配置80通过/2xfail/1原观察器失败，ebfe075修正后的设备单例通过。三客户端小型训练/精确恢复、两种SDK组合、常驻FP32/FP16小图和构造见证分别记录；局部借用计费已定位，容量守卫未修复。

[通用Describe增量有限成本（2026-11-10）](interface-heads-performance-20261110.md)：554f2f5的CPU/NPU共14个宽FC/HB/LH/sum/SSD/三客户端训练与精确下一窗，10个FC16成本点、8个层链成本点、8个独立图诊断。FC16训练12.614/15.497秒、TTFT3.083/3.960秒；层链仍明显慢于限定张量公式。规模独立数值未验，原启动失败保留。

[常驻存储/服务审阅](resident-workspace-audit-20261110.md)：47704c9固定41完整文件及2局部范围；区分owner去重、首次局部借用、journal与workspace，single-bank投影仅设计。新增诊断与设备见证另记实现及实际资格。

[批量发布与能力分派（2026-11-10）](interface-batch-publication-20261110.md)：47704c9的局部CPU501通过/13保留xfail，图CPU308通过/4原严格失败；有界诊断后其余训练/恢复通过。NPU配置122通过/3xfail，三客户端小训练/恢复分列。FC16完整训练CPU1/NPU12.447/17.212秒，NPU推理反升；小图私有union11317、SelectBackward640。

[调度与接口审阅](scheduling-interface-audit-20261110.md)、[区域批量审阅](region-batching-audit-20261110.md)、[循环与Describe审阅](recurrent-interface-audit-20261110.md)：固定06b242e逐文件/具名函数记录；实际容量缺陷、重复扫描/上传/逐行转置与模块能力边界。审阅范围有重叠，开发检查不代替后继实现固定资格。

[唯一KV存储与图统计（2026-11-10）](interface-snapshot-bank-20261110.md)：1528df8的CPU301通过/3保留xfail，NPU配置流程72通过/1保留xfail（含CPU辅助），三客户端小型训练/恢复；FC16 CPU1/NPU训练14.628/20.634秒，NPU推理略升，完整尺度数值未验。小图615外层/13477含私有节点，不能将32个query边界当全部图规模。

[接口审阅与可复现覆盖（2026-11-10）](implementation-audit-20261110.md)：固定4d777bc基线实际审阅25文件/128函数，自动清单其余明确未审；checkpoint重复解码、插件完整记录与漏子包hash三项修复，开发检查与固定资格分列。

[FC16同尺寸归因与当前默认候选（2026-11-10）](fc16-cost-attribution-20261110.md)：独立FC/Transformer公式锚点、真实首token、原生阶段计时；拒绝全查询列打包退化，保留区域列和独立输出批量copy。1f3f62d CPU358通过/2原xfail、NPU流程95通过（含CPU辅助），三客户端小训练/新进程恢复分列；FC16训练CPU18.230→17.497/NPU22.998→21.065秒，CPU首token反升，NPU共享争用和完整尺度数值not_checked明确保留。

[联合prefill部署代表与容量修正（2026-11-10）](prefill-deployment-20261110.md)：宽FC、FP16、异构LH、约1B/9B、CPU线程与双NPU有限成本/恢复；默认异步挂起、resident工作区拒绝、B32中断及后继结果分列。485a3e2选定推理预算/恢复修正46项固定安装通过，规模独立数值not_checked。

[同公式层链基线（2026-11-10）](layer-chain-baseline-20261110.md)：KV/SSD独立张量公式、CPU13/IEEE NPU7项小型检查及12个成本点；默认图入口仍明显较慢。vendor卷积HF32有界见证解释指定差异，原失败保留，不认证标准Transformer/Mamba或完整记录等价。

[局部合批与owner伴随汇合（2026-11-10）](prefill-local-batches-20261110.md)：f15/3d两套固定构建和安装/SDK训练与精确恢复分列；3d CPU281通过/4数值xfail，NPU目标流程68通过（含CPU辅助）。原FC16同minimal返回CPU22.30→18.64秒、NPU31.02→22.90秒；算子回读和小投影减少，规模数值不自动认证。

[联合prefill有限代表矩阵（2026-11-10）](prefill-representatives-20261110.md)：a553bea的CPU16/NPU18项完成运行与精确续接，覆盖宽FC、176节点HB/LH、sum/qry、SSD/GDN层链、三客户端及mixed-a/b/c；独立数值not_checked。auto/greedy差异有限，NPU小规模仍较慢；另有CANN9.2 resident小图6项独立推理/训练/恢复通过。

[Query完整根合并与选定返回边界（2026-11-10）](query-root-boundary-20261110.md)：a553bea固定CPU249通过/2原xfail，NPU73通过/2原xfail/1观测失败；fb406dd补线程观测后CPU24/NPU2通过。安装消费者和独立SDK的CPU/NPU训练/新进程恢复分别通过；原失败保留。CPU原FC16完整/最小返回29.788/22.068秒，数值、不同返回与成本分列。

[默认prefill与跨时间区域批（2026-11-10）](prefill-default-20261110.md)：7fe97f2核心、5f563f6 SDK恢复策略守卫；CPU397通过/2既有xfail、NPU26实际设备+7辅助通过，三客户端小图训练与精确恢复分列。原FC16受控成本CPU62.177→38.953秒、NPU237.119→101.756秒；实际工作、内存取舍和剩余碎片化另列，未认证原任务完整等价或下游验收。

[TL-REQ-004显式密集查询（2026-11-10）](dense-query-tl-req-004-20261110.md)：75a5027实现/新构建/安装，26ca1b6诊断；CPU25通过/2数值xfail，NPU13实际设备及2辅助通过，四个公共消费者三步独立训练/新进程恢复通过。原FC16成本与小profile另列，保留大用例工作量/梯度范数差异、CPU局部严格失败及下游尚未验收事实。

[TL-REQ-003显式正支持数值模式（2026-11-09）](positive-support-tl-req-003-20261109.md)：2c11695实现、2502fa4补测、b4879ff修正资格筛选；CPU新策略30/相关314通过，NPU13项实际设备目标及17项CPU/配置辅助通过。实际HB16行、独立VJP、稳定Aggregate/KV、两区域完整根/AdamW/新进程精确恢复；原严格失败和8个native加载失败保留，不认证原HB16学习训练或其他后端。

[CANN9.2项目迁移（2026-11-06）](cann92-project-migration-20261106.md)：两套真实构建、三客户端CPU/NPU完整推理与声明训练/新进程恢复通过；三处独立综合根严格失败保留。固定52项CPU、22项随机LM、9项设施分列；新私有环境、旧环境拒绝及两处构建/测试驱动修复，不升级历史resident/规模资格。

[Dense/MoE首轮终态审阅（2026-11-03）](llm-terminal-audit-20261103.md)：80项已终态；43执行完成、32下载依赖失败、4超时、1静态预算拒绝，严格检查与6项自身恢复分列；含CPU/NPU大batch实际曲线。
[Dense/MoE后台入口与启动（2026-11-02）](llm-background-execution-20261102.md)：固定6bbc7e2设施81项、15dfabe队列19项；真实下载/CPU/NPU队列，首次FP32独立超差和原启动失败分列。运行中不代表通过。
[Dense/MoE实测准备（2026-11-02）](llm-experiment-preparation-20261102.md)：只读环境审计、9次有界网络样本、MoE容量与待授权有限表；c28完成约1.06MiB/s样本，优化引擎配套差异明确保留。未下载权重、安装引擎或启动模型/后台任务。

[Qwen公开吞吐调查（2026-11-02）](qwen-throughput-survey-20261102.md)：83条外部记录（CPU15/NPU13/CUDA55）、30份来源及JSON/CSV；区分总/输出token、逻辑batch/动态服务、dummy权重、全参训练与条件未明项。固定config的容量下界和待授权实测表已准备；没有下载权重或新增本机模型资格。

[2026-10-31跟踪内容清理](artifact-cleanup-tracked-only-20261031.md)：固定0b07e72执行完成，释放磁盘279.519GiB及tmpfs16.040GiB，仓库降至76.410MiB；仅主worktree和Git跟踪内容，原204条结果仍可查询。41份任务终态已归档，未提交源码在仓库外保全待分析；不新增计算资格。

[八卡共享资源终态](core5-platform-eight-followup-20261031.md)：固定d3de18f配置/launcher、4485697安装、eca070f库；三项全部执行/离散/精确续接通过，小训练严格通过，两项约1B数值探索。无存活后代，不改写原排队超时，不认证任意根、多卡resident或八卡容量上限。

[GPU释放前审阅](core5-cuda-release-20261031.json)：本轮GPU任务全部终态、无计算进程；正式小证据已回收，原远端dev的20项未提交源码已在本机保留并校验。后续本机任务不依赖GPU实例；此审阅不增加数学或性能资格。

[CPU/NPU约1B大batch算子观察](core5-platform-profile-20261031.md)：固定bd2ff2e两项独立profile；168行大MM、864次两行私有BMM与288次共享base GEMV并存。c59688c修正CANN matmul漏计，13项固定解析检查通过；保留原trace hash，不重跑或新增数学资格。

[CUDA4.561B单至四卡安全容量代表](core5-platform-capacity-20261031.md)：单卡allocated20.058GB、reserved20.246GB；执行/离散/精确恢复通过，原FP32数值超差探索。双/三/四卡逐卡最大峰值10.299/7.859/5.419GB，完整离散和精确恢复通过、数值探索；同模型放置不等于多卡容量上限；NPU12.802B单/双/三卡已完成，CUDA9.123B单卡拒绝/双卡实际近界、离散/精确恢复通过，FP32数值探索。

[人类安装消费者、停止/恢复与离线报告](core5-human-workflow-20261031.md)：CPU/NPU/GPU各自公开流程实际演练；原NPU/GPU base独立根严格失败与下游任务通过分列。1b46dc5固定10项与双机新安装报告生成、浏览器筛选通过；CPU/NPU各26命令终态，停止及恢复控制组分别为空，完整独立数值验收与生命周期未检查分列。

[CUDA/NPU约1B训练容量与非有限字段](core5-platform-training-capacity-20261031.md)：单卡静态拒绝、双/四卡第三窗loss非有限；原配方不变，检查/恢复未完成。31a3b32两小训练严格回归通过，失败成本与资格分列；NPU单/双卡原第三窗非有限，独立参考与恢复未执行。新loss系数1e-9的CUDA双卡及NPU单/双卡1.014B完整训练/离散/精确恢复完成，独立数值仍超差；新任务身份不解除原失败。

[CUDA/NPU单卡resident大batch与布局](core5-platform-resident-20261031.md)：默认B128/B512静态拒绝保留；结构界定queue/stages后两项实际完成、离散/恢复通过、数值探索，实际行/填充/资源分列；两平台B1及小型训练严格通过；NPU有界B128完成、数值探索，B512构造EE1019流任务容量失败。

[CUDA/NPU同模型多卡与三通信训练](core5-platform-multicard-20261031.md)：CUDA bd2ff2e固定6项，1.014B推理执行/离散/恢复通过、数值探索；小型双卡三通信严格训练通过。逐卡参数/状态/消息与计算统计、共享组、物理传输及恢复峰值单列，不认证P2P或多卡resident。NPU2/3卡推理探索、ordinary/cpu_bridge小训练严格通过，blocking恢复SIGSEGV；原8卡资源等待超时未执行；明确共享资源后继见本页八卡终态报告。

[三平台有限值诊断packing](core5-finite-packing-20261031.md)：bd2ff2e固定18项，默认/原阈值保留；B128同材料成本降低、6小训练严格通过，两策略既有完整checkpoint/输出精确相同。CUDA减少小kernel但显式同步未减少，收益和停止边界分列。

[CPU/NPU/GPU约1B B128/B512与三拓扑阶段](core5-platform-scale-20261031.md)：原FP32超差探索、完整离散与磁盘续接分别记录；B512恢复峰值5.505GB，CPU七项与GPU五项规模均保留数值超差，NPU六项探索及1B/B512下一窗精确恢复失败；GPU另有独立1B/B128实际MM/BMM/profile，不作吞吐，不认证约1B训练。

[三平台首组矩阵与B128/B512基线](core5-platform-baseline-20261031.md)：98项正式流程、4项独立观察；43严格通过、38数值探索、17严格失败。CPU/NPU/GPU同材料A/B/C、实际矩阵批/峰值/控制放置与完整诊断归档分别记录。

[RTX3090规模推理与有限优化](core5-cuda-inference-20261031.md)：FC/HB/LH启发、真实不齐batch、三客户端、mixed/resident的完整前向与磁盘恢复分列；显式FP64约1B及安全近界、严格FP32失败、实际矩阵批/预算回退和两项packing优化均保留原身份。六项处置与NPU负性能结果分别记录，不认证大尺度训练或多卡resident。

[RTX3090时间线与有限完整训练](core5-cuda-finite-20261031.md)：固定32edb0f，两个完整resident客户端观察到设备就绪/打包/选择/推进；首轮37项18通过/19失败，两个小型FC scalar补充通过。容量452让LH中型精确续接通过，但独立keys仍失败；中型无合格吞吐，小图GPU未见对CPU收益。原严格标准和硬件限制保持。

[RTX3090单卡resident与eager补验](core5-cuda-resident-20261031.md)：32edb0f四阶段新构建/sm86，单卡56组件/dtype+53公共检查零跳过通过；scalar三客户端18项、standalone双卡6项、native半精度6项及相关CPU通过。原FP32/双卡精确恢复失败保持；FP16反馈两例在CPU复现同值，SSD显式scalar通过仍不解除packed失败。时间线/性能分列。

[RTX3090 core-5 eager与跨卡首轮资格](core5-cuda-eager-20261031.md)：x86_64/Torch2.12.1+cu130/sm86；PyTorch/native单卡FP32和FP16各19/19，standalone分别23/24和22/24；完整训练、同栈精确恢复与CPU交接分列。72方向/实现/流传输单元通过，完整双卡native GDN精确恢复失败；原严格失败、流警告和resident硬件限制保留。无性能声明。

[历史迁移与新CPU接入](history-migration-20261030.md)：716提交独立核对、映射与固定源码定向验证；发布及临时材料退役见[原子发布与新检出](history-publication-20261030.md)。

2026-10-30历史清理：本目录历史报告和JSON保留当时实际源码/二进制身份与结果；
旧提交号用`python scripts/resolve_history_commit.py OLD_SHA`定位等价代码。
[迁移规则](../history-migration.md)说明单分支clone与旧归档取回；映射不表示重新运行历史资格。

[仅保留Git内容的清理与重建](artifact-cleanup-tracked-only-20261030.md)：e99733d全新CPU构建/132项定向检查及8组安装消费者通过；删除全部ignored本地产物22.379GiB，仅保留Git内容；520份终态摘要和两份实际失败行承接复现。本轮临时验证环境/工作树亦已删除，目录约126MiB（含Git）。

[第二轮磁盘清理](artifact-cleanup-round2-20261030.md)：12d3e16基线，30份历史源码副本及5个大文件退役65.853GiB；仓库约22.50GiB，仅主工作区。28份未提交差异归档、失败记录和小型反例保持，三组构建独立保留；搬迁后CPU加载/smoke及6项接入检查通过。

[第一轮磁盘清理](artifact-cleanup-20261030.md)：cc89ca8基线，51个干净祖先worktree、18处旧build、14份已记录trace及LH通过checkpoint退役，回收58.876GiB；28份未提交源码、当前构建和失败材料保持。CPU smoke及6项接入检查通过，预检原失败单列。

[多实验仓库协作规范与安装消费者](downstream-contract-20261030.md)：36113ce，4项单元、8组CPU仓库外安装/独立训练/新进程精确恢复、4项错误请求拒绝通过。交付依赖锁/请求/交付/验收模板；native worker1资格与workers2原失败分列，无新增设备/性能资格。

[core-5实际配置覆盖与导航修正](core5-coverage-correction-20261030.md)：0f4506d，CPU相关73项通过，含26个真实FC/HB/LH配置×PyTorch/native的52项两步训练和新owner精确恢复。修正旧FC单层/伪混合覆盖、窗口与来源分类说明；不增加性能或设备资格。

[逐配置机器索引](core5-index.json)集中已审阅的逐配置正确性和性能记录，各报告保留原库存、来源与资格，不能将不同源码的检查数相加。查询例：`python -S scripts/index_core5_study.py docs/evidence/core5-index.json --view coverage --topology fc --kind correctness --format tsv`。仅含已有逐配置记录，其他历史汇总/模块证据继续从下方进入。

[core-5自然规模与三客户端有限性能](core5-natural-study-20261029.md)：e50b6d8，25配置中7项完整通过；中型FC三客户端CPU FP64及LH1.051B完整训练/精确恢复。真实区域批与独立NPU profile、FP32/FP64严格失败、resident在途任务/预算拒绝分列；8B/16B仅静态计划。

[core-5目录与消费者资格](core5-study-clients-20261029.md)：372 CPU365；e50相关CPU43、standalone CPU6/NPU8、Python NPU单卡/half/双卡3项。独立存储精度参考和实际梯度/optimizer/磁盘恢复；计数有重叠，不能相加为一次全矩阵。

[native显式文件预算与原1B复验](core5-checkpoint-budget-20261029.md)：b255，CPU32+6消费者/SDK8消费者；默认4GiB保护与原格式不变。5.25GB文件保存/加载成功，原owner下一窗非有限梯度导致严格失败，不宣称完整恢复通过。

自然规模和checkpoint两份JSON可用`python -S scripts/index_core5_study.py docs/evidence/core5-natural-study-20261029.json docs/evidence/core5-checkpoint-budget-20261029.json --format tsv`直接查询，无需旧运行目录。

[常驻局部纤维压缩资格与有限性能](core5-compact-fibers-20261029.md)：04b5215；CPU两环境各10、公共NPU56、SDK7组件/dtype通过。medium完整核对/恢复通过，单次2.794→2.470s、peak allocated约−26.9%；原D64新的首个构造拒绝为CANN流资源207005/EE1023，保留终态与重启条件。

[单行VJP/字段packing与常驻因子KV有限性能](core5-boundary-performance-20261029.md)：15c3e65对照f03e9be；五项完整训练通过，CPU未见收益，mixed有单次下降；D64单/双NPU自身恢复精确但独立CPU核对失败，resident构造容量拒绝。独立profile、实际工作、分区与停止条件分别记录。

[常驻因子KV/实际训练观测/拒绝快照资格](core5-resident-boundary-20261029.md)：15c3e65；CPU设施10项、公共NPU14项、SDK组件9项及两套eager各6组通过。实际梯度/SGD物理owner/磁盘精确恢复独立核对；CPU反向后晚注册NPU的原失败及early规避探针分列。性能另验，原数值/异步和GPU边界保留。

[单行反向/异形传输/分配观察资格](core5-eager-boundary-20261029.md)：e0d6e27；CPU517 passed/2原strict xfail，原D64及反馈FP32仍严格失败，各调度自身逐值恢复通过；native完整实际分配观察与预算有证据。性能和resident增量分别验收。

[首写入/half反向/完整双卡桥接隔离](core5-boundary-isolation-20261029.md)：bdd7f6c；原D2048实际操作数分离来源与私有投影误差，局部连续FP64有改进但未认证全图；half归一化FP64 VJP仍失败，默认async完整CPU桥接仍EE1017。有限预算及停止/重启条件分别记录。

[独立输入KV矩阵批资格与有限对照](core5-independent-projection-20261029.md)：f03e9be；CPU273 passed/2原FP16 strict xfail，纯Python与Python/native NPU各六组、公共常驻12项、新SDK两dtype六项及eager六组通过。D64显式FP64两步训练/逐值恢复通过；479次写入投影全部覆盖，但CPU中型单次+4.35%、D64+0.16%、NPU中型+1.36%，未见明确加速。独立profile与全部原失败保留；GPU延后，原数值/异步边界不改。

[非GPU已知边界与定向解除](core5-nongpu-boundaries-20261029.md)：aadd3b4设施14项通过；外层FP64汇合不足以解除D64，原反馈FP32失败/显式FP64四窗全通过，各自逐值恢复。SDK原affine mixed-a/c四次更新与恢复通过；FP16 GDN和默认异步双NPU严格失败保留。GPU延后，无新增吞吐结论。

[D64算术边界与跨步误差隔离](core5-arithmetic-isolation-20261029.md)：e75e1f5九项探索均严格失败；d8f7132交付可复现诊断，CPU19项设施检查通过。最佳局部方案首步事件逐值相同，但反向/更新仍有容差内差异，第二步状态/梯度/动量槽失败；两条调度自身精确恢复通过。没有采用局部生产策略或新增吞吐/设备声明。

[FP32 GDN/SSD归一化反向](core5-fp32-normalization-20261028.md)：ad6d1e7；CPU453 passed/2原FP16 strict xfail，原两项FP32 GDN AdamW/恢复通过。独立实数Jacobian及逐值前向、standalone NPU四组/纯Python NPU两组完整训练恢复、CUDA构建/CPU核通过；旧失败记录和其他精度/硬件边界保留。

[稳定共享准备、显式精度与完整训练观测](core5-stable-preparation-precision-20261028.md)：1f1ec57；精确恢复修复、真实梯度/optimizer槽检查。CPU572 passed/2原FP32 GDN失败（旧基线复现），NPU eager KV训练及CUDA构建通过；D64四项FP64训练和匹配92.62亿主体参数的CPU FP64推理完整通过，原FP32失败仍保留。

[Core-5 局部分配坐标与全部eager KV低秩](core5-local-allocation-factorized-kv-20261028.md)：474f9af（含56631eb）；实际失败行/独立Decimal、cap质量分位局部根和共同阈值消去。CPU746 passed/2 strict xfail、NPU九项及eager训练恢复、CUDA构建通过；两个大图已执行测量窗但完整状态仍失败。CPU中型约持平、NPU单次约−2%，不作稳定加速声明。b62f2db阶段持久化另有6项固定资格。

[Core-5 Quad响应修复与私有矩阵批](core5-private-projection-20261028.md)：07e56ca；共同偏移闭式解/VJP、真实私有BMM/独立行根。CPU574 passed/2 strict xfail、NPU单/双卡相关资格及CUDA三架构构建通过；有限CPU近持平、mixed-c稍慢，D64完整训练及D2048残差仍严格失败，没有GPU真机或大型吞吐结论。

[Core-5 参数匹配与性能经验迁移](core5-performance-transfer-20261028.md)：2da3f2c/0503593；五族真实参数清单、core-3主体匹配、实际92.62亿参数材料、共享存储/复制优化；CPU376 passed/2 strict xfail、设施14项、NPU八项及CUDA构建/CPU核验。单次CPU下降、NPU稍慢且allocated小降；新增D64完整训练和quad共同偏移严格失败均保留。

[Core-5 三项审查修复](core5-review-fixes-20261028.md)：1d19c59；ClockedState参数/VJP、quad正支持与参数bank准入。CPU442项、standalone NPU两dtype四组件、CUDA CPU31项/三架构构建通过；NPU首次环境失败保留，无GPU真机。

[Core-5 单样本节点放置反向/训练](core5-weighted-node-training-20261028.md)：0155b0f运行库/220052a检查器；FP32四组合×12根及跨窗桥、两dtype各20训练更新/6次副本共同更新、Python原30项与修复后8项、安装消费者通过。CUDA三架构/loader通过，无真机；原设备与检查器失败保留。

[Core-5 单样本节点放置前向](core5-weighted-node-placement-20261028.md)：d072c17运行库/1289bc9检查器；CPU8、Python-owned50、NPU两dtype各9双卡配置及单设备图/训练回归通过；CUDA三架构与安装消费者构建通过。放置后的VJP/optimizer另验收。

[Core-5 新进程启动与内存](core5-weighted-observation-20261028.md)：f72a6cc；四个补充配置完整对照通过，四阶段allocator/启动边界，CPU5项及CPU/NPU/CUDA消费者构建通过；CUDA无真机。

[Core-5 有限完整调用性能](core5-weighted-performance-20261028.md)：ceae7bf；24配置22通过，CPU反馈两项严格浮点失败保留；CPU/mixed/resident、Python与standalone分列，独立profile及工作量，冷启动/SDK allocator指标仍有限制。

[Core-5 常驻样本副本与master复验](core5-resident-weighted-peers-20261028.md)：7db0b08；standalone两dtype各20训练更新及6次双卡共同更新，Python-owned672e948四项通过；CUDA三架构构建。原检查器失败保留；单样本节点切分未由此实现。

[Core-5 FP16公共全图/训练资格](core5-resident-half-training-20261028.md)：21ed86d；CPU32、Python-owned NPU52、standalone两dtype各21图/15配置/20更新、完整VJP与精确恢复；CUDA三架构构建，peer与性能另验收。

[Core-5 FP16四循环/KV Full资格](core5-resident-half-memory-20261028.md)：bcef2c4；CPU24、NPU两dtype四组合、CUDA三架构构建；完整half图另验收。

[Core-5 FP16原始状态/投影/卷积资格](core5-resident-half-primitives-20261028.md)：2e4d445；CPU32、NPU两dtype六组合、CUDA三架构构建；明确FP32算术/伴随，完整half图另验收。

[Core-5 weighted公共训练资格](core5-resident-weighted-training-20261028.md)：343429d；CPU28、Python-owned NPU28、standalone五配置20更新、保留窗口/累计/精确恢复与两安装消费者，CUDA三架构构建；FP16/peer另验收。

[Core-5 weighted完整窗口VJP资格](core5-resident-weighted-window-20261028.md)：2622e60，NPU27配置/324根模式与36前向回归、CPU17项及CUDA三架构构建；公共retained训练另验收。
- [Core-5 消息祖先反向资格](core5-resident-weighted-route-20261028.md)：ad8e7b8；CPU12/NPU12、精确时间和独立边界根，CUDA三架构构建；完整窗口继续。

- [Core-5 常驻完整数值阶段 VJP](core5-resident-weighted-stage-20261028.md)：ebe5471；NPU24配置/342根模式/完整前向回归，CUDA三架构与安装loader；跨阶段训练继续。

- [Core-5 完整 weighted 常驻 CUDA 构建](core5-weighted-flow-cuda-20261028.md)：8eebcec；显式sm80/90/100、架构8项、DSO及安装loader通过；无GPU真机。

- [Core-5 完整 weighted 常驻推理](core5-resident-weighted-flow-20261028.md)：5efce32；CPU50、standalone NPU21图/15配置、Python-owned23项、安装消费者与精确续接。训练另验收。

- [Core-5 常驻批量head资格](core5-resident-head-20261027.md)：161ccea；CPU8/NPU完整共享VJP与实际版本，CUDA三架构构建。

- [Core-5 常驻发送原语资格](core5-resident-sending-20261027.md)：8e7298d；CPU34/NPU八配置完整VJP，CUDA三架构构建。

- [Core-5 常驻Alloc组件资格](core5-resident-allocation-20261027.md)：2814b87；CPU29项、NPU FP32完整VJP，CUDA三架构构建，未认证完整图。

[Core-5 CUDA链接构建与CPU24项资格](core5-cuda-eager-20261027.md)：670ead3；GPU真机待验。

- [Core-5 standalone master/schema2资格](core5-native-master-20261027.md)：670ead3；CPU24项，NPU FP16显式SGD24图/AdamW2图及设备→CPU精确恢复续接。

- [Core-5 eager FP16 NPU有限资格](core5-half-device-20261027.md)：01689c0；CPU12项、Python/native mixed与显式同步双卡、master恢复及归属profile。

- [Core-5 带权 FP16 CPU 资格](core5-half-20261027.md)：8c8824f；99项通过、2项GDN AdamW strict xfail单列。

- [Core-5 eager NPU 混合与两卡同步资格](core5-device-20261027.md)：470ce06；CPU200项、Python/native有限矩阵通过，standalone严格失败单列。

- [Core-5 带权执行位置 CPU 资格](core5-placement-20261027.md)：9234c13，508 项；typed Read/control/selection、packed copy VJP 与训练恢复。

- [Core-5 standalone 完整训练/checkpoint CPU 资格](core5-native-training-20261027.md)：a7106fc，20 项；每 dtype 24 图案例、三步训练与恢复后第四次更新。

- [Core-5 KV 基座投影复用 CPU 资格](core5-kv-sharing-20261027.md)：6dc2665，309 项；私有状态、完整使用者 VJP、训练与恢复。

- [Core-5 Aggregate 精确计算复用 CPU 资格](core5-sharing-20261027.md)：52dee1e，543 项；物理来源、所有使用者反向、训练和恢复。

- [Core-5 来源绑定与稳定一阶归约 CPU 资格](core5-source-20261027.md)：503f609，665 项；四种别名/宽度、独立公式、AdamW 与恢复。

- [Core-5 区域实际批量/VJP CPU 资格](core5-region-20261027.md)：5391017，536 项；完整候选、精确解反向、独立根、在线训练/续接。

- [Core-5 Read/Full 批量 VJP CPU 资格](core5-query-vjp-20261027.md)：67854b3，647 项；独立字段/行根、共享参数、AdamW 与恢复。

- [Core-5 Read/Full 数值批量 CPU 资格](core5-query-20261027.md)：7b8b469，377 项；实际 Mem/Full、完整快照与独立根。

[core-5 Aggregate 批量计算/VJP](core5-aggregate-20261027.md) — `d90cfbc`，704 项相关 CPU 资格。

[core-5 State 批量 VJP](core5-state-vjp-20261027.md) — `7a017bf`，343 项相关 CPU 资格。

[core-5 精确联合记忆](core5-packed-20261027.md) — `26dad7d`，378 项相关 CPU 资格。

[core-5 完整评分/证据/key](core5-scoring-20261027.md) — `cacb645`，593 项带权支线 CPU 资格。

[core-5 带权循环记忆](core5-recurrent-20261027.md) — `442e8b2`，544 项定向 CPU 资格。

[core-5 带权 KV 首组合](core5-weighted-kv-20261027.md) — `6fb467c`，129 项定向 CPU 资格。

[core-5 结构化表示](core5-structured-20261027.md) — `714d5f9`，212 项定向 CPU 资格。

[core-5 一般 SettleGraph](core5-settle-20261027.md) — `efb9d62`，247 项定向 CPU 资格；不认证尚未完成的带权局部函数或新设备路径。

本目录保留已审阅的实现提交、验证范围、结果、失败和性能限制。
它是历史证据，不是待执行清单；当前任务只由 [STATUS](../STATUS.md) 和
[ROADMAP](../ROADMAP.md) 决定。

2026-10-12 用户确认旧实验默认不再复核，原始构建/运行产物可退役。
报告中的绝对路径、`TASK/...` 和 `artifacts/...` 仅标识当时的来源，
**不承诺文件仍在线可用**。必要记录与退役说明见 [产物保留规则](../artifact-retention.md)。
历史报告保留原文和原失败；原始文件移除不等于原验证未发生，也不构成新源码通过证据。

## 旧基座结论入口

- 基座：[最终资格](foundation-final.md)、[跨图资格](cross-family-qualification.md)、[公共库](library-foundation.md)。
- eager 加速器：[范围与环境](accelerators-20260928.md)。
- 旧批量实现：[初始 CPU/CUDA 资格](batched-vjp-cuda-local-20261010.md)、[Add 后续](add-batch-followup-20261010.md)。
- NPU：[公共模块](npu-module-batching-20261011.md)、[常驻完整组件](resident-composite-20261011.md)、[stream/精度](precision-and-stream-gates-20261011.md)。
- 性能：[旧全尺寸结论](selection-terminal-20261010.md)、[六组件](npu-batched-components-20261011.md)、[旧完整选型](native-selection-20261011.md)。
- 不能忽略的限制：[原始近似平票路由失败](original-add-route-witness-20261004.md)、旧完整选型中的两项容量拒绝。
- 接续与清理：[固定教材/干净 clone 起步验证及历史产物退役](artifact-retirement-20261012.md)。

查看某个结果时先确认源码、环境、模块、图类别、精度、输入和观察量范围。
跨硬件浮点结果、共享服务器争用与历史计时不能承诺原样重放。
默认基于当前源码创建新的独立验证/测量，不重启旧 job 或补写旧结果。

- [Core-5 熵分配 CPU 定向资格](core5-entropy-20261027.md)：a05d3ff，226 项通过。

- [Core-5 十种发送函数 CPU 定向资格](core5-sending-20261027.md)：4a08588，320 项通过。

- [常驻完整带权区域组合](core5-resident-region-20261027.md)：ce63aef，CPU33/NPU14配置完整VJP与保存版本，CUDA三架构构建。

- [常驻完整带权Aggregate](core5-resident-aggregate-20261027.md)：92751c0，CPU15/NPU物理原子及完整VJP/保存版本，CUDA三架构构建。

- [常驻来源低秩参数链](core5-resident-source-20261027.md)：6f709f8，CPU18/NPU完整因子链与保存版本，CUDA三架构构建。

[带权常驻参数归约与局部训练](core5-resident-parameters-20261027.md)：2e335e7，CPU3/NPU八步更新与新owner恢复、CUDA三架构构建。

[常驻原始状态与缓存](core5-resident-raw-20261027.md)：cb68d91，CPU14/五schema三dtype精确codec/NPU完整字段VJP及CUDA三架构构建。

[常驻固定时间短卷积资格](core5-resident-convolution-20261027.md)：f89bdfe，CPU/NPU完整组件及CUDA三架构构建。

[常驻共享投影资格](core5-resident-projection-20261027.md)：2af401d，CPU/NPU完整组件及CUDA三架构构建。

[常驻 KV 接收更新资格](core5-resident-kv-update-20261027.md)：681a7e9，CPU/NPU完整组件及CUDA三架构构建。

[常驻带权摘要与 Attention 归约资格](core5-resident-pool-20261027.md)：afe7b5e，CPU/NPU完整组件及CUDA三架构构建。

[常驻 KV 原始时间窗口与 sum 记忆资格](core5-resident-kv-sum-20261027.md)：fea6fa7，CPU/NPU完整组件及CUDA三架构构建。

[常驻相对旋转与极大逻辑年龄数值转换资格](core5-resident-rotary-20261027.md)：1eb4b72，CPU/NPU完整组件及CUDA三架构构建。

[常驻 KV 内容检索完整 Mem 资格](core5-resident-kv-query-20261027.md)：8b1be80，CPU/NPU完整组件及CUDA三架构构建。

[常驻 KV 完整内容与公共残差资格](core5-resident-kv-full-20261027.md)：386970a，CPU/NPU完整组件及CUDA三架构构建。

[常驻完整 Describe 资格](core5-resident-describe-20261027.md)：cd576d2，CPU/NPU完整组件及CUDA三架构构建。

[常驻四循环 Upd/Mem/Full 固定源码资格](core5-resident-recurrent-20261028.md)：6e2ecb7，CPU/NPU完整组件及CUDA三架构构建。

[已安装带权配置检查接口](core5-weighted-binding-20261028.md)：ce21a83，CPU Python/native 45 项与 C++ 两种 dtype。

[常驻半精度KV读取](core5-resident-half-kv-read-20261028.md)：db85db6，CPU27、NPU sum/qry两dtype及CUDA三架构构建。

[常驻半精度KV更新](core5-resident-half-kv-update-20261028.md)：69bf4f0，CPU26、NPU更新/raw两dtype及CUDA三架构构建。

- [数值/异步失败隔离](core5-numerical-isolation-20261028.md)：固定b0321b9/62073dd；GDN首次梯度分歧与不经过图模块的双NPU反向反例。

- [core-5公共配置/自然规模固定CPU资格](core5-planning-20261029.md)：8f51ba6新CPU构建、336通过；设备及后续benchmark扩展另验。

[三通信CUDA双卡独立根补查](core5-platform-multicard-checks-20261031.json)：固定bd2ff2e三项full检查通过，与原性能结果分列；NPU三通信完整推理/恢复先通过，独立根阶段均EE1017，原严格失败保留。

[大模型checkpoint磁盘/tmpfs成本](core5-checkpoint-cost-20261031.md)：dba2a5e固定67项CPU回归、CUDA小型full检查通过；4.561B两项完整恢复成本观察，独立数学未检查，不计吞吐资格。

[NPU B512恢复诊断](core5-platform-resume-diagnostics-20261031.json)：固定旧包跨进程重放到旧raw边界即逐位不同；同进程新checkpoint raw精确、下一完整窗仍失败。采样布局相同，不归因磁盘，不新增性能或独立数值资格。

八卡共享队列/配额适配固定d3de18f：17项CPU策略检查与真实八卡锁/映射探针通过，记录在[多卡检查](core5-platform-multicard-checks-20261031.json)；计算资格独立待终态，不能由队列探针推断。

- [LM分段恢复与后台启动](llm-ranged-recovery-20261103.md)：固定27项、实际Range及启动失败保全；下载中不是模型通过。

- [LM剩余清单、后台后继与小诊断](llm-continuation-20261103.md)：94/48固定设施、7活跃服务；43后继与14引擎/放置流程，严格失败与待验分列。

- [LM后继57项终态审阅](llm-terminal-audit-20261104.md)：36执行完成、19依赖未跑、超时/预算各1；OLMoE实际结果、4次精确续接与严格失败，当前无活动作业。

- [LM失败重启与完整llama构建](llm-restart-20261104.md)：固定118项设施、21项选择性后继；6服务已运行/等待，模型结果尚待终态审阅。

- [LM重启后21项终态](llm-restart-terminal-20261104.md)：OLMoE CPU/双NPU B512、llama B1及两次精确恢复；2超时、3依赖未跑、10因运行cwd污染触发保护而未提交，当前无活动任务。

- [LM目录隔离与选择性后继](llm-isolation-launch-20261104.md)：694eb60固定125项设施通过；15项后台依赖队列，保留旧严格失败、二进制与断点。

- [LM目录隔离后继15项终态](llm-isolation-terminal-20261105.md)：13项推理成本完成、1项独立数值/token失败、1项双NPU B512容量拒绝；MoE16片下载完成，9服务终态、无中断或遗留进程。

- [显式LM容量试跑入口](llm-capacity-entry-20261105.md)：59e9aca固定安装设施135项通过；observe显式记录预算并尝试分配、单配置持久队列、失败保持。B512尚未提交。

- [联合KV准备、固定检查与FC16成本](joint-kv-preparation-20261110.md)：aab8cb7，CPU80、NPU相关33与两安装客户端训练/精确恢复；单次成本与独立资格分列。

- [默认分量窗口固定资格](component-window-20261110.md)：CPU391、NPU28通过/2独立根超差与33相关检查，独立SDK三种设备/精度训练和精确恢复。

- [KV整序列反向固定资格](kv-sequence-vjp-20261110.md)：CPU143、NPU49及2原数值xfail；安装消费者、有限诊断和原FC16两种返回成本分列。
