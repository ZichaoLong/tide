# 当前交接 — 语言与结构联合目标，设计已重排（2026-11-08）

用户最新重点是语言next-token达到接近对照的性能，并在同一模型上推进规模化、稀疏化、局部化。
已更新[研究设计](research-design.md)、[有界研究表](studies/language-structure.md)、[计划JSON](../research/language-structure-plan.json)、[研究图](../web/index.html)与[待办](ROADMAP.md)。
新研究均planned，结果awaiting；计划`launchable:false`，本轮没有新下载、设备分配或训练提交。第四批已终态，无该批自动后继。

- A改为全位置自然文本LM，与小稠密Transformer及同架构稠密TIDE比较；合成检索保留为机制依据。
- B利用原Qwen语言能力，目标是一个固定MLP的全位置完整承接与能力保持；原模型继续训练及简单MLP替换作对照。固定10%混合不再是主完成标准。
- 联合评价绑定同一checkpoint的NLL、容量、实际工作、局部边界、内存与时间，不能跨A/B拼结论。
- [公共静态规划](evidence/language-structure-static-plan.json)确认HB16/40/88节点均可保持region4、入/出度≤4及各4端口。它没有运行/学习/性能资格；新全位置应用路径尚未实现和验收。
- 首选WikiText-2 raw数据尚未下载或锁定；A训练集4096 BPE未拟合。公开基座、数学、原Qwen及安装锁保持不变。
- 拟议执行包共12h墙钟、≤8NPU小时、≤2CPU-only worker小时，最多2张真实空闲卡，合计≤8线程/48GiB，新磁盘≤8GiB、配额余量≥80GiB；分项及停止条件见研究表。预算是设计，尚未提交。
- 下一实施顺序为数据与全位置接入→有效batch/吞吐与预算冻结→A/B有界语言先导→条件规模点/重复→联合审阅。旧FP32/full-cache严格失败保留，有限数值探索、离散/恢复/学习/性能分轴记录。

研究图默认当前主线，可切换历史证据；既有12份限定研究结果及所有失败/未检查记录保留。Lens继续查看历史运行，私有入口见`.local/LENS.md`；新研究尚无run ID。

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
