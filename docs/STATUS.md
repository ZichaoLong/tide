# 当前交接 — v3较小曝光语言比较已后台运行（2026-11-08）

北京时间23:34已实际启动两路，当前核对时间UTC 2026-11-08T15:37:14.493383+00:00。[研究表](studies/language-exploration-v3.md)、[启动证据](evidence/language-exploration-handoff.json)、[查看/停止/恢复](RUNNING.md)。

| 作业 | 实际分配 | 当前范围 |
| --- | --- | --- |
| `lm-a-language-v3` | 物理NPU3 → logical npu:0 | 小Transformer128步、独立test与推理已完成；TIDE稠密训练stage已启动，随后稀疏 |
| `lm-b-language-v3` | 物理NPU5 → logical npu:0 | 冻结基线、原MLP与简单MLP的128步训练/独立test/推理已完成；TIDE稠密训练stage已启动，随后稀疏 |

这是执行进度，**尚无完整TIDE语言比较或联合质量/结构结论**。已完成控制只保留原始独立评价，不能提前认证语言可用性。研究图的四个科学结论仍awaiting。后续stage可随后台继续；以只读状态为准。

冻结源码`6cd4f856df02e9c90645198cf5998a27a877b4ec`、快照`.local/source-language-explore-v3`、树SHA256 `2c44b51421077624867ee27fa6fa6c72d8c95e60a28736be2565083260da8e77`；配置`configs/studies/language/fc-explore-v3.json`。数学0770593、foundation85d161d及安装环境不变，248文件核对一致。23个执行/依赖文件未改，复用v2七候选机制、三步精确恢复和B8/B32校准；新增代码只负责核验复用与有界编排。不要编辑快照、环境、活动数据/cache/config或checkpoint。

固定B8/L32/128更新、seed11；A每候选32677有效目标，B32503。原65536目标准入失败保留，新配置明确为较小曝光探索；原质量ΔNLL≤0.05、结构筛选及数值阈值未放宽。当前硬件同比例加速不是质量/结构准入条件。执行、严格数值、离散语义、恢复、学习和性能分列。

旧四次失败保守已扣3551.641秒（0.9866 NPU小时）。新A/B外层计算上限9120/9017秒，各自训练7200秒、评价1800秒且受外层共同约束；最多同时2NPU/8线程/48GiB。原8NPU小时总额与北京时间11月9日07:28:46共同截止不重置；本次不会自动启动FC8或seed22，剩余预算待学习证据审阅。当前配额169GiB/800GiB，新增文件仍受8GiB上限。

小型Lens记录`tl-lm-v3-recording`已verified/readback_matches，控制组已产生新的已核对投影；项目`tide-learning-lab`，TL-A-LM/TL-B-LM，run前缀`tl-lm-a-language-v3-`与`tl-lm-b-language-v3-`。训练实时指标/实际映射见各stage的metrics.jsonl/projection.json；私有查看入口`.local/LENS.md`，不改服务访问策略。

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
