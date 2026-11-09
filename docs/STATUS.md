# 当前交接 — v5全部终态，限定结果已审阅（2026-11-09 14:36核对）

**当前无本批运行、排队或自动后继作业。** HB16 sparse于12:36:54完成，B FC8于13:43:33完成；全部相关cgroup和已知后代清空，三个预算监督也已结束。本次没有启动、恢复、迁移训练或改动固定输入。[总报告与逐组一页结论](studies/language-depth-v5-results.md)、[终态统计/身份/Lens证据](evidence/language-depth-terminal.json)。

| 研究 | 终态和已审阅观察 |
| --- | --- |
| A FC16串联 | dense128/test NLL6.864746；同曝光FC4 dense6.842736。context-reset正支持下溢，整链failed，sparse未训练 |
| A HB16分支汇合 | 独立sparse128/test NLL6.927488，Full/candidate57.28%，实际内部消息65610；同曝光FC4 sparse6.838241。原dense86更新后下溢，无完整同拓扑配对 |
| B单槽FC8 | dense/sparse各128及评价完成，NLL4.380228/4.395503。稀疏差+0.015275，单侧95%上界0.027216<0.05；Full减48.51%、内部消息减47.90%。比旧FC4略好，仍弱于简单MLP4.323800和原MLP继续训练3.809893 |
| A容量控制 | 浅宽56更新后失败；独立Transformer2048完成，128/2048 test分别7.133504/6.346539，日程/曝光差别明示 |
| A HB40 | 前驱失败后等待器终止，训练0步，科学结果未检查 |
| 旧A v4 | dense/sparse512及评价完成，NLL6.149196/6.141408，Full减49.88%；条件FC8正式配对跳过。它使用不同确认文档，分数不与v5直接混算 |

当前证据支持B双区域的限定语言质量/实际计算稀疏折衷；A多区域有学习信号，但尚未显示加深带来的质量收益。单seed、L32、小语料与缺失对照限制保留；没有通用语言、长上下文或规模化成功结论。数值严格失败不被学习效果覆盖。

v5累计**22931.15秒=6.3698/24计算worker小时**，含失败、校准、复评及诊断；原共同截止**22:49:13**不变。旧v4另扣**5.0344/8 NPU小时**。剩余额度不自动扩展候选或释放HB40。数学0770593、安装foundation85d161d/tidegraph0.3.0保持，59个固定输入、248个安装文件hash全匹配；9份正式终点checkpoint文件hash匹配，未据此声称新增长期恢复验证。

六轴：执行状态如上；训练后A FC16/HB16 sparse小型core92/31元素严格超差，B完整/cache14569/18154元素严格失败，原阈值和未知原因保留；已测离散/因果/跨区域梯度及三步新进程恢复分别通过，长期独立恢复未查。共享资源时间单列，不要求稀疏比例兑现为吞吐比例。

Lens项目`tide-learning-lab`，本次v5和旧A的57份已有投影身份、事件数与readback一致；旧v4两路36份有重叠，不能相加。主要曲线前缀`tl-lm-a-hb16-sparse-v5-`、`tl-lm-b-fc8-v5-budget2-`及`tl-lm-a-transformer-cpu-v5-`。私有入口`.local/LENS.md`，具体映射与查询方法见[操作页](studies/language-depth-v5-running.md)。

用户正在上游处理TL-REQ-003；实验安装尚未切换，不将上游工作区改动视为已交付/验收。后续优先验收固定基座修复并保持模型配方；独立补原FC16 quad不依赖Shannon修复，HB16 dense和FC16失败干预需新基座验收。共同有界评分配方降为备选，替代下方旧快照的首选建议。本批结论待用户验收，不自动扩HB40或后续规模。

---

以下为历史时间戳快照；当前状态和优先级以上方为准。

# 当前交接 — HB16正支持失败已定位（2026-11-09 12:15）

本次有界重放从原第64步checkpoint继续，在第87步再次捕获HB16 dense的分配失败，第72/80步loss和梯度范数与原记录一致。实际四候选caps全1，节点10的数学分配约3.10588e-46，低于FP32最小subnormal的一半；80位Decimal独立闭式解与FP64单行相符。原残差在内部容差内，触发的是正支持硬检查。单行FP64及提高温度通过不认证整模型修复；合成反例表明FP64也可能下溢。[实际证据](evidence/hb16-positive-support-v5.json)、[三方分析与有限建议](studies/allocation-support-v5.md)、[需求TL-REQ-003](requests/TL-REQ-003.md)。

诊断作业`lm-hb16-underflow-v5`用时571.69秒，已终止且后代清空；只消耗原24计算worker小时预算，原HB16失败状态保留。源码固定`9ebbc63784a29f2c785bb7568ddb9da63b63dd55`，快照`.local/source-language-underflow-v5`，树hash`b57a7dc0eebb2f9edb9ae5ecef3cb255afcf89c68982507c9eff14dee6c86b10`。未来入口已保存公共异常实际输入，活动快照未变。基座/上游未修改，TL-REQ-003尚未获得评审或交付。

| 作业 | 12:15实际状态与范围 |
| --- | --- |
| `lm-a-fc16-v5` | dense128训练、常规test和小型推理完成；context-reset评价发生同类下溢，链失败且后代清空；原sparse尚未训练 |
| `lm-a-hb16-v5` / `lm-a-hb16-sparse-v5` | 原dense86更新后失败；独立quad仍在CPU1训练（最近已记录80/128步）；没有完整dense128配对 |
| `lm-a-controls-v5` / `lm-a-transformer-cpu-v5` | 浅宽56更新后失败保留；独立CPU Transformer2048及评价完成、待科学审阅 |
| `lm-b-fc8-v5-budget2` | 原NPU12 dense仍在训练（最近已记录96/128步），sparse为原后续；不迁移或重复 |
| `lm-a-hb40-v5` | 前驱失败后依赖等待终止，训练0步；不绕过前提 |
| `lm-a-language-v4` | 已完成并确认后代清空；主候选512和评价完成，条件FC8只完成接入/恢复/校准，正式配对因剩余预算或准入条件跳过；待完整科学审阅 |

[实际交接快照](evidence/language-depth-handoff.json)核对59个固定输入和248个基座安装文件全部一致。新总预算监督`lm-depth-budget-diagnosis-v5`包含原全部v5费用及本次诊断，12:15累计约4.55/24小时，共同截止22:49:13；旧监督仍在，未增加额度。严格数值、离散语义、恢复、学习和性能分别记录。

后续建议是一对现有match/ind有界评分子族的Shannon/quad先导，完整表和停止条件已登记，尚未提交新学习配置。研究图记录本次定位、FC16干预失败和旧v4终态，科学结论仍待审；不按诊断退出码宣称架构成功。Lens继续沿用`tide-learning-lab`及原运行ID，私有入口`.local/LENS.md`；本次技术诊断只存项目证据，不增加无学习曲线的投影。查看/停止/恢复沿用[操作页](studies/language-depth-v5-running.md)。

---

以下为此前时间戳快照，当前状态以上方及实际运行记录为准。

# 当前交接 — v5多区域研究已后台实施（2026-11-09）

用户已接受多层研究表并明确授权轻载共享NPU/CPU资源调度与旧数据清理。当前新批共同截止北京时间**22:49:13**，累计**24计算worker小时**（CPU训练替代相应NPU份额），最多6个同时计算的worker含旧v4。全批预算监督已实际启动，等待依赖不算已训练。[作业/源码/实际Lens ID快照](evidence/language-depth-handoff.json)、[查看分类与操作](studies/language-depth-v5-running.md)。

| 研究 | 实际状态 | 固定内容 |
| --- | --- | --- |
| A FC16串联 | CPU1已正式训练dense，sparse为同链后续 | 4区域，width32，两候选各128步、512日程前缀 |
| A HB16分支汇合 | CPU1已正式训练dense，sparse为同链后续 | 4区域/3依赖层级；与FC16近同参数、同曝光 |
| A浅宽/Transformer控制 | 共享NPU已训练浅宽dense，Transformer为同链后续 | FC4 width64/128步；2block width96/FFN512/2048步并保留128前缀 |
| B FC8承接槽内加深 | 新预算尝试已通过接入/微型恢复并启动训练阶段 | 原1.5h尝试在配对预算准入失败，训练0次；新尝试同两候选/B8/128步，≤4.5h，统一计入24h |
| A HB40递归嵌套 | 已提交，等待HB16完成及机制/验证学习条件 | 10区域/5依赖层级，条件64步配对 |
| 旧权重同文档复评 | A/B均已提交，等待并发资源前驱 | 只评价，不算新训练或新seed |

CPU/NPU短校准支持A深层CPU1：FC16/B8约22.21秒/步，CPU4约32.17秒，共享NPU约80.20秒。CPU4/B32约13.21 token/s，CPU1/B8约11.52；保留B8匹配已有采样/更新，后续不改活动配方。B/容量控制在用户授权的轻载共享卡运行，映射与后续争用有记录；NPU大batch限时失败独立保留。[资源决策](evidence/language-depth-resource-decision.json)、[B预算调整](evidence/language-depth-budget-adjustment.json)。

数据小记录`tl-lm0-depth-v5`已进入`tide-learning-lab`并verified/readback_matches。新16文档排除已观察32份，训练/验证保持。Lens按运行名`v5`和配置`study_id`识别；该记录器未填原生Group，不把项目分类误称Lens Group。正式曲线、最终test与inference分开；接入/校准只保留项目记录。私有入口仍为`.local/LENS.md`。

新作业源码分别100b450（数据/校准/容量控制）、aa76da5（CPU深度链）、8626ab2（B预算调整/复评/监督）；快照及47个唯一固定输入hash已核对，基座248个安装文件匹配，数学0770593/foundation85d161d保持。CPU深层新增跨区域梯度、消息和无直接端口绕过检查；三步新进程精确恢复随候选接入检查。 新增有限严格超差已保留：FC16为619/2810、HB16为900/381、B FC8为14/144个元素，最大绝对差4.77e-5；相关支持/因果/梯度/微型恢复分开通过，原因未知，见[检查审阅](evidence/language-depth-checks.json)。严格有限失败、离散、恢复、学习、性能分别保留，尚无v5科学结论；研究图新增5组与5个待审结果。

六个历史大产物已删除，回收**2.0126GiB**；44个旧技术Lens投影已删除，删除时161→117，5个已不存在/身份未确认项未猜删。最终模型、历史学习曲线、全部v4/v5及原始指标/失败/日志保留。[文件退役](evidence/artifact-retirement-v5.json)、[Lens退役](evidence/lens-retirement-v5.json)。

旧v4 A仍按0692756原快照与原截止运行，本次没有中断或修改。11:05最近采样稀疏400/512；其条件FC8继续按原链决定，旧B已终态。各时间戳快照非实时，使用上方操作页查询。

---

以下为早先交接快照；当前授权与作业状态以上方为准。

# 当前交接 — 多区域深度提案，旧A继续运行（2026-11-09）

北京时间10:28只读核对：A仍在物理NPU1、background.slice运行，稀疏最新记录第304/512步；最近验证第256步NLL 6.377497。B已完成且cgroup/已知后代清空。A本轮已记录阶段无执行失败，FC8尚未启动。[本次生命周期/资源快照](evidence/language-depth-planning-status.json)。

10:13左右出现其他用户新负载，包括与A共用物理1；本次共享队列空闲设备0张。A计时采样曾从先前约14秒/步升到53秒/步，最新回到15.4秒/步，存在波动；资源争用及未分解原因单列，此前11:10–11:25完成预测不再可靠。旧A计算截止14:28:03、整批18:47:01和原预算保持；本次没有停止、迁移、恢复或修改活动输入。

已形成[多层/嵌套与清理建议](studies/language-depth-proposal.md)：A FC16串联与近同参数HB16分支汇合，B在原一个MLP替换槽内部加深为FC8；条件HB40检验递归嵌套，辅以浅宽和更合适规模的Transformer控制。新批建议上限12h/24累计NPUh、最多6卡含旧作业，目前仅提案，未创建运行配置或提交队列。旧v4条件FC8不重复安排；研究图仍保留已有事实和v4待审槽。

账号约170/800GiB，runs约4.3GiB，JSON/指标/日志合计约40MiB。清理候选6个旧缓存/微型恢复checkpoint共2.01GiB，路径与hash见[清单](evidence/storage-retention-proposal.json)；最终模型、原始证据、当前v4和活动输入保留。本次未删除任何文件。源码仍0692756、冻结快照及数学/基座身份不变；旧结果和数值失败解释沿用下方已审阅记录。查看、停止和恢复方法见[RUNNING](RUNNING.md)。

---

以下为此前有时间戳的交接快照；当前进度与资源状态以上方为准。

# 当前交接 — A稀疏训练中，B已终态审阅（2026-11-09）

北京时间09:20核对：A继续占用物理NPU1→npu:0，`background.slice`及实际后代存在；B于07:04:35完成并确认cgroup空、已知后代全部终止，NPU3已释放。本次没有提交、恢复、停止或修改任何活动作业输入。[本次快照及逐文档结果](evidence/language-v4-status.json)、[操作说明](RUNNING.md)。

| 研究 | 当前状态及观察 | 尚待完成 |
| --- | --- | --- |
| A FC4语言 | dense512和独立test/推理完成；sparse最近记录第48/512步，最后完整checkpoint第32步；无执行阶段失败 | 稀疏最终确认集评价；质量/稀疏配对结论 |
| A普通LM参照 | Transformer512与额外2048均完成；新确认集NLL分别6.947570/6.194960，unigram6.941797 | 512仍欠训练；2048参照明确多4倍token，不作为等曝光优势证明 |
| B同瓶颈控制 | 新稠密对照128步及全部六模型确认复评完成；[一页审阅](studies/TL-B-LM-v4.md) | 本批B无后继；长期恢复和跨seed/广泛语料仍未检查 |
| FC8双区域 | 未启动 | 待A两TIDE验证学习信号、相关机制/恢复/校准及完整配对预算；不以同比吞吐加速作门槛 |

A dense确认集NLL **6.149196**，与更充分Transformer2048相差−0.045765（文档95%区间[−0.084312,−0.004705]）；单seed/短上下文，不能宣称普遍优势。上下文重置使dense NLL增加0.072681，Transformer2048增加0.653819，说明两者对更早上下文的依赖不同。零core使dense增加2.088091。当前稀疏训练采样Full约50.6%，不是训练后质量/活动结论。

B新确认集：原MLP继续训练3.605958、简单MLP4.098327、TIDE dense4.179921/sparse4.191096、同32维瓶颈4.264717、冻结4.286356。TIDE优于同32维新对照，但仍落后入口106维MLP；信息压缩、模块及优化尚未完全区分。新对照相对冻结点估计略好，但非劣单侧95%上界0.055677＞原0.05，保留未通过事实。TIDE稀疏减稠密+0.011175、上界0.019523，Full减49.13%；这是旧权重新文档确认，不是新seed。

六轴：所有已完成阶段执行成功；A dense训练后小型数值/因果/分块/streaming通过，三候选微型精确恢复通过。B机制与微型恢复通过，45个NPU权重梯度元素有限超差及完整/cache20391元素严格失败保留，argmax一致，原因未定。四个已完成训练终点及dense128前缀共5份checkpoint已CPU解码并核对完整有限状态；未读取活动稀疏checkpoint张量，未做长轨迹独立等价。[解码证据](evidence/language-v4-checkpoint-inspection.json)。

固定执行源码`069275628eefc1011e2b69526a28126e2bc25985`与只读`.local/source-language-confirm-v4`已复核未变，17份作业固定输入及248个安装文件hash匹配；数学0770593、foundation85d161d不变。配置`configs/studies/language/fc-confirm-v4.json`，新16文档/4096确认目标、A512为130706训练目标/B128为32503，B8/L32/seed11，不加seed22。

本批保守累计用约**2.45/8 NPU小时**，B用340.644秒。A自身计算截止北京时间**14:28:03**，整批共同截止仍为**18:47:01**；B未用额度不自动提高A上限。按当前稀疏约13.7秒/步估计，主比较及评价约11:10–11:25完成，受实际运行影响；随后只执行已冻结的条件FC8。配额170GiB/800GiB，本批目录约810MiB，资源护栏未触发。旧批3.3307NPUh单独封存。

Lens项目`tide-learning-lab`：27份已完成NPU投影核对通过，另有已核对的数据小记录；稀疏实时投影状态recording。入口`.local/LENS.md`，run映射见本次JSON。研究图B结论经审阅后更新，A与联合结果仍awaiting，不按退出码升级。工作区文档可更新，活动快照不可改。

---

# 当前交接 — 语言v4后台运行（2026-11-09）

用户批准的新有界批次已实施。北京时间07:02观察，A/B均在`background.slice`实际运行，分别物理NPU1/3→logical npu:0；新数据已完成并核对Lens。权威状态随运行变化，见[操作说明](RUNNING.md)与[本次启动快照](evidence/language-v4-handoff.json)。

| 作业 | 当前已观测 | 自动后续范围 |
| --- | --- | --- |
| `lm-a-language-v4` | Transformer机制/微型精确恢复通过，512和额外2048步均完成并独立评价 | FC4 dense/sparse/Transformer各512；单独标注额外曝光Transformer2048；两TIDE验证学习及预算允许后，FC8匹配128配对 |
| `lm-b-language-v4` | 新对照128步、微型精确恢复、独立评价与完整推理已完成；旧权重确认复评中 | 单一同32维入口稠密对照128步；旧四checkpoint及冻结模型在新文档复评；不重训旧B候选 |

[固定研究表](studies/language-confirmation-v4.md)、[配置](../configs/studies/language/fc-confirm-v4.json)。独立测试预先固定16份未用文档，每路线4096目标；旧训练/验证张量、顺序、词表不变。A B8/L32/512步为130706训练目标，额外Transformer2048为522600；B128为32503。seed11、microbatch8、accumulation1，不自动加seed22。FC8若运行，128步与本次FC4第128步检查点及512步LR日程前缀配对，不能拿FC4最终512作等曝光规模比较。

执行固定`069275628eefc1011e2b69526a28126e2bc25985`，只读`.local/source-language-confirm-v4`，树SHA256 `73a8df2680042cc69fdf1bfa0c8a7c5a2d8dabd3bffc6c7adb91a61cc9a9ae5e`；数学0770593、foundation85d161d及248安装文件保持。CPU新增及相关测试9项、图/审阅隔离16项、离线浏览器检查通过。公共环境及基座只读，不改活动源码、配置、cache和依赖。

新B稠密对照190464参数，对比TIDE190757；独立CPU FP64公式/手算VJP通过。NPU新检查在原1e-6/1e-5下有45个权重梯度元素有限超差（最大2.37e-6），值、输入梯度、梯度进入、未来VJP为零及alpha0精确保持通过；原因未定。新B完整替换full/tail小检查通过；完整/缓存推理严格结果独立保留。相关小检查和精确恢复通过才继续标记训练探索，不能称严格等价。旧B full/cache及dense单元素失败均保留。执行、严格数值、离散语义、恢复、学习和性能分列，当前无v4科学结论。

本批新预算最多2NPU、8累计NPUh、8计算线程/48GiB；A≤7.5h含条件scale（自身≤4h），B≤0.5h，CPU-only≤2h、新磁盘≤8GiB、配额留80GiB。共同截止**2026-11-09 18:47:01北京时间**，不随服务重置；实际CPU数据服务8.26秒。旧批累计3.3307NPUh独立封存。吞吐比例仍为工程观察，不作实际稀疏+语言质量的准入门槛。

Lens项目`tide-learning-lab`，小记录`tl-lm0-confirmation-v4`已verified/readback_matches。训练run前缀`tl-lm-a-language-v4-`/`tl-lm-b-language-v4-`；组TL-A-LM/TL-B-LM，条件规模TL-LM-SCALE，检查TL-LM-EXEC。私有入口在`.local/LENS.md`。研究图保留v3限定结论，并新增RLC-A/B/JOINT待审槽；任何退出码不自动填科学支持。

---

# 当前交接 — 语言v3已终态并审阅（2026-11-09）

A于北京时间00:47、B于00:42完成，七候选均128步并完成独立test/推理，全部cgroup及已知后代已核对终止。**当前无活动、排队或自动后继作业。** [终态总报告](studies/language-v3-results.md)、[逐文档/生命周期/数值证据](evidence/language-v3-terminal.json)、[操作说明](RUNNING.md)。

| 主线 | 独立test NLL | 实际稀疏及范围 |
| --- | --- | --- |
| A从零完整TIDE | dense6.779286、sparse6.802554；Transformer7.022139、unigram6.946099 | 稀疏减稠密+0.023268，单侧95%文档上界0.037871<0.05；Full减49.73%。普通LM对照欠训练，不证明普遍架构优势 |
| B预训练局部承接 | frozen4.361484、原MLP继续训练3.684206、MLP4.147179、dense4.234056、sparse4.236845 | 稀疏减稠密+0.002789，上界0.011343；Full减49.03%。保持冻结基线，但控制更好，无TIDE独特优势 |

**当前单seed、较小曝光下的质量/Full活动折衷成立；语言可用性、规模化与跨区域局部性联合目标尚未完成。** FC4一个区域、零内部边，candidate/Upd/Read未减少；Full不是总FLOPs，硬件吞吐兑现单列，不作为上述筛选门槛。FC8与第二seed均未启动。A/B词表与预训练背景不同，不横比NLL/PPL。

执行、严格数值、离散语义、恢复、学习、性能分列：A训练后小型core/分块/streaming通过；B五模型full/cache在原阈值下均有限失败、argmax一致，B dense core另有1个共享梯度元素超差。所测因果/未来VJP/离散机制通过，未知数值原因保留。学习结果不覆盖严格失败。

七候选B2/L3三步新进程精确恢复来自未改代码的v2；本次7个128步checkpoint均CPU解码、有限张量及完整optimizer/RNG/游标/身份核对通过。[解码证据](evidence/language-v3-checkpoint-inspection.json)。未执行128步独立不中断对照，当前没有中断训练需要恢复。

固定执行源码`6cd4f856df02e9c90645198cf5998a27a877b4ec`与只读`.local/source-language-explore-v3`已复核；数学0770593、foundation85d161d、248安装文件保持不变。配置`configs/studies/language/fc-explore-v3.json`；B8/L32/128/seed11，A32677、B32503训练目标，独立test16文档/4096目标。原65536准入及旧失败保留，原0.05质量和严格数值阈值不改。

v3两服务保守扣8438.933秒；含v1/v2共11990.574秒=3.3307NPU小时，8小时余4.6693小时，原共同截止仍为北京时间11月9日07:28:46。余量不重置截止，不自动追加候选；当前包等待用户验收。配额169GiB/800GiB。

Lens项目`tide-learning-lab`，29个v3投影均verified/readback_matches且事件数一致，完整run→destination映射在终态JSON的lens。私有入口`.local/LENS.md`。六组一页结论由总报告链接；研究图新增RL-A/RL-B/RL-JOINT三份限定审阅，RL-SCALE仍awaiting。下一批选项、依据与分别的资源上限见[联合审阅](studies/TL-JOINT.md)。本次检查没有提交新训练。

## v2终态（历史记录）

最新状态核对：`lm-a-language-v2`于北京时间20:43、`lm-b-language-v2`于20:47自然终止，chain均failed，错误为`no calibrated common case fits minimum updates/tokens`。所有相关cgroup及已知后代均已确认终止；当前无本批训练或排队。原[运行时交接](evidence/language-running-handoff.json)是当时快照，已由[本次终态证据](evidence/language-admission-terminal.json)接替。

**正式语言训练0次，训练后独立test 0次，FC8规模点未启动。** A三/B四候选的小型机制检查和B2/L3三步新进程精确恢复通过；B8/B32、L32的校准已有记录。三步校准不算完成语言先导，不能据此判断效果或宣称联合目标成立。详见[接入终态报告](studies/language-admission-outcome.md)。

停止来自实验编排门槛：B8/L32的128更新至多32768目标，低于我预设65536；B32/L32虽够目标数，整组保守预测超过7200秒训练分项。L128或更大batch有超时保留。代码没有要求“稀疏快于稠密”，但这些固定门槛实际阻碍了学习验证，应在后续配置中重排，不能将它们当作架构判据。

按最新用户澄清，**实际计算稀疏且语言质量可以是主要证据；硬件是否兑现同比例吞吐是后续优化问题**。吞吐仅用于资源规划和工程记录，不是联合质量/结构验收的必要条件。A FC4/B8校准Full事件从1024降到594/600，平均减少41.70%，步时反而15.95→20.65秒；这是有范围的稀疏观察，不是总算术量减少42%或训练后质量结论。Upd/Read未减，FC4无内部消息。

总8NPU小时按四个service墙钟保守扣3551.641秒（0.9866h），尚余7.0134h；共同截止仍为2026-11-08 23:28:46 UTC / 北京时间11月9日07:28:46，不重置预算。CPU-only worker总额2h、最多2NPU/8计算线程/48GiB、新增磁盘8GiB和配额余量要求不变。

固定源码`7d1a95f16a3a390e7bbf60d18f25d2928ca849f0`、只读快照`.local/source-language-routes-v2`与[FC配置](../configs/studies/language/fc-pilot-v2.json)保持原样。数学0770593、foundation85d161d、数据/依赖锁未改；旧HB失败、接入缺陷和校准时限均保留。当前研究表和设计文件只修正解释与后续方向，不重写既有运行输入。

两路v2共25个已生成Lens投影均verified/readback_matches；时限失败阶段没有终态投影时仍保留项目原始metrics/log，不能算投影通过。无正式学习曲线。入口在`.local/LENS.md`，状态/日志方法见[RUNNING](RUNNING.md)。四个新科学结果仍awaiting。

后续优先完成有对照的语言学习探索，复用已校准B8/L32；若采用更小曝光，应单独命名、冻结配置，保留旧门槛未满足事实及原质量/数值判据，先取得质量与实际活动证据。本次状态核对没有提交或恢复新作业。

## 第四批固定终态交接（历史结果保持）

本批于**18:02:05北京时间**结束。原服务保留failed（单次训练时限），完成服务completed；两者实际cgroup空且全部已知后代终止，当前无本批排队、训练或自动后继。所有四候选完成128次逻辑更新、验证与独立测试，两个TIDE时限失败均保留。[终态总报告](studies/fourth-batch-results.md)、[原始终态证据](evidence/fourth-batch-terminal.json)。

主要结论：TIDE在固定局部输入下学会教师函数近似，固定10%非零接入通过小样本任务保持筛选；**尚未证实TIDE独特优势**。同参数量MLP局部拟合更好，均值控制也通过保持筛选。原教师MLP仍计算，没有效率收益或广泛能力保持结论。

| 项目 | 终态观察 |
| --- | --- |
| 原生能力 | 查表val58/64，90.625%；test229/256，89.453%。复制val44/64，68.75%未达准入，test未检查 |
| 局部信息 | 原维/PCA16 ridge验证相对MSE0.014904/0.228786；线性诊断不代表所有模型的上限 |
| 影子拟合 | TIDE s11/s22 test相对MSE0.425328/0.425006；MLP0.377568/0.367180，四者原val门槛均达 |
| 非零接入 | TIDE两种子均231/256，90.234%；均值也231/256，MLP231/256与230/256。新错TIDE0/1例，有限样本不证明总体非劣 |
| 严格数值 | 原生及9个非零模型完整/缓存检查均有限失败，原atol1e-5/rtol1e-4保留，未知原因未解除 |
| 离散/机制 | 上述10个检查例argmax一致；解析MSE梯度、分支/输入梯度及唯一optimizer参数通过，范围有限 |
| 恢复 | 两候选新进程3步B2 tensor/值精确；两TIDE实际从第96步恢复到128，无完整128步独立不中断参考 |

固定模型源码1ec3f73、完成协调器4502d07，两只读快照复核未变；数学0770593、foundation85d161d、安装248个文件hash不变。配置`configs/studies/fourth/b-preserve.json`，B128/128更新/seed11、22/AdamW。实际分配计算3705.94秒（61.77min），保守扣账3714.47秒＜2h；整批墙钟62.06min＜12h。最多1NPU/4CPU线程/请求24GiB，列出的新增任务文件29.35MiB；两次时限失败及未知未发布重算成本已计入，未重置预算。

四份一页报告：[BASE](studies/TL-B4-BASE.md)、[INFO](studies/TL-B4-INFO.md)、[SHADOW](studies/TL-B4-SHADOW.md)、[PRESERVE](studies/TL-B4-PRESERVE.md)。R14–R17由主代理审阅为限定结果，待用户验收；研究图保留失败尝试和未检查范围，不由退出码生成优势结论。

Lens项目`tide-learning-lab`，20个实际run投影已核对verified/readback_matches；私有入口/SSH见`.local/LENS.md`，运行对应在终态JSON的lens。查看、停止和恢复说明见[RUNNING](RUNNING.md)，读取结果不需要重启计算。

A第四批未新增工作，HB三seed原结论保持。总报告中的下一组选项保留为当时建议；优先级已由本页上方语言与结构研究表接替。旧10%混合区分度及固定稀疏对照转为可选诊断，不自动开始后批。

最新共享资源波动：HB16 sparse CPU1新校准38.35秒/步，早先为20.48，未将这些时间当独占benchmark。
