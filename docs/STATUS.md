# 当前交接 — 联合语言研究后台运行（2026-11-08）

已执行用户授权：自然文本next-token质量与规模、稀疏、局部结构共同推进。当前[研究表](studies/language-structure.md)和[FC先导配置](../configs/studies/language/fc-pilot-v2.json)记录有界调整 `LM-ADJUST-001`；原HB16配置和失败保持。

A/B服务 `lm-a-language-v2`、`lm-b-language-v2` 已核对在 `background.slice`，通过共享队列分别取得物理NPU1、3（进程内均为npu:0）。当前进行准入/恢复与校准，正式训练只在各自门槛和预算成立后自动开始。运行观测、实际阶段及Lens对应见[交接证据](evidence/language-running-handoff.json)；实时状态以运行目录和job工具为准。

固定模型/协调器源码 **7d1a95f16a3a390e7bbf60d18f25d2928ca849f0**，只读快照`.local/source-language-routes-v2`，树hash `4917f4f31f0360fc3c775fab47d6a39e394cbc2553e8744af76280da2ab610f4`。数学0770593、基座85d161d和248文件安装锁未变。活动作业不读取后续工作区修改。

共同截止仍为 **2026-11-08 23:28:46 UTC / 11月9日07:28:46北京时间**。总额8NPU小时、2CPU-only worker小时，最多同时2卡/8计算线程/48GiB；新增磁盘≤8GiB，配额留≥80GiB。初始A/B失败按service墙钟保守扣521/198秒，后续计算cap为17449/10572秒，不重置总额；各阶段类别也扣旧耗时。账号最近约166GiB/800GiB。服务CPU/内存属性为请求，本机控制器限制沿用已知范围；应用4线程与22GiB阶段RSS守卫实际启用。

数据作业与三次CPU准入均终止，失败没有覆盖。固定WikiText revision、文档与词表身份见[数据记录](evidence/language-data-ready.json)；624/60/62 train/validation/test原始文章，本包固定8个validation、16个test文章，各前两个128-token块。A/B验证2048、test4096有效token；L32时保留全部目标重分块。预训练污染和近重复仍未知，跨tokenizer不横比PPL。

[CPU准入](evidence/language-cpu-admission.json)保留FP32严格超差；稀疏因果逐位扰动最大5.96e-7，独立参考、未来VJP零与早期支持机制通过。不能称严格数值等价。初始HB16 NPU链因校准时限/协调器重复字段结束，B还发现错误解冻固定传输参数；[初始尝试](evidence/language-initial-attempts.json)记录全部失败和恢复范围。两项代码缺陷在新源码修复，原作业自然终止、无存活后代。

当前FC4候选为1区域/0内部边，A配对小Transformer、B配对原MLP继续训练和简单MLP替换。固定128更新，先据实测选择B8/32/128及L128或已声明L32回退，实际暴露须65536–1m；resolved JSON在训练前只读冻结。A有学习信号且完整预算允许时才推进FC8双区域配对，未运行不能称规模化成立。质量ΔNLL≤0.05、原严格数值阈值不变，四个新科学结果位均awaiting。

Lens项目`tide-learning-lab`已核对数据小记录与实际准入记录；仍用外部JSONL recorder，未部署或改服务。查看入口/SSH在本机`.local/LENS.md`，新run以`tl-lm-a-language-v2-`、`tl-lm-b-language-v2-`开头。来源、状态、日志、停止和新输出恢复方法见[RUNNING](RUNNING.md)。

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
