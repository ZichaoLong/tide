# 当前交接 — 联合语言研究已获授权并进入实施（2026-11-08）

用户已批准[联合研究表](studies/language-structure.md)，并授权有依据的局部模块、参数、扩展、定位、loss和训练策略调整。
实际配置为[language/initial.json](../configs/studies/language/initial.json)，设计JSON仍不可直接提交。
共同截止为2026-11-08 **23:28:46 UTC / 11月9日07:28:46北京时间**，从本轮启动审计开始计时；NPU总额8h、CPU-only worker总额2h，最多2张卡/8线程/48GiB。失败、重算和后续调整不重置预算。

数据作业`lm-data-v1`已完成且无存活后代，源码`492f46d`的只读快照保持。WikiText来源、共享分片SHA、文档/目标及tokenizer锁已形成，[数据记录](evidence/language-data-ready.json)给出身份和范围。
624个训练文档、60个验证文档、62个测试文档；本包固定8个validation文档和16个test文档，各前两段128位置。A/B验证2048、测试4096有效token；训练池分别517581/64304 token，训练曝光按实际更新另计。

A的4096 BPE仅拟合训练文档，B保留原Qwen词表；词表及hidden背景不同，不横比PPL。数据准备不认证模型学习或因果/恢复。5项应用单元检查通过，新的实际core序列/梯度/恢复和吞吐仍在准入阶段。
Lens小记录`tl-lm0-data-v1`已verified/readback_matches，继续沿用外部JSONL记录器；训练环境没有Trackio SDK，不为记录新增SDK或修改服务。私有入口见`.local/LENS.md`。

接下来依次进行CPU小型独立检查、两条NPU路线的机制/恢复/有效batch校准，再锁正式更新和token预算。L128不可负担时允许统一L32后续配置，保留原尝试；不改成单答案任务。
尚未提交正式长训练，图中四个新科学结果位仍awaiting。旧12份限定结论与严格失败不变。后台操作使用`scripts/job.py status/logs/stop`，运行目录仅由作业写入。

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
