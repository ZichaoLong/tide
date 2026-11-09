# v5 多区域研究终态审阅

2026-11-09：最后一个训练链 B FC8 于北京时间13:43:33结束；14:36重新核对全部服务、cgroup和已知后代均已终止，无排队或自动后继。原失败保留。实际累计22931.15秒，即 **6.3698/24计算worker小时**，含准入失败、校准、复评和下溢诊断，等待另列；共同截止仍为22:49:13。旧v4另扣5.0344/8 NPU小时，不合并或清零。

结论是：**B双区域在此次单seed短上下文上保持了可接受的稀疏质量折衷，并实际减少内部消息；A深层已能学习，但尚未显示加深的质量收益。** HB16 dense和浅宽对照的下溢、FC16干预失败使部分比较不完整，不能据此判定深层架构无效，也不能称联合目标全部达成。

## 同一新确认集的结果

v5固定16份新文档；每路线原计划4096目标，实际屏蔽padding后A为3878、B为3876个有效目标。下面只在同词表/同目标内部比较。新候选B8/L32、seed11、microbatch8、accumulation1。A新TIDE各128更新/32677目标，512步LR前缀；B各128更新/32503目标，13 MSE+115 CE。Transformer128前缀来自2048步日程，最终522600训练目标，均需明示曝光和日程差别。

| 研究组 | 结果与边界 | 一页报告 |
| --- | --- | --- |
| TL-A-DEPTH，FC16串联 | dense128 test NLL6.864746；同曝光FC4 dense6.842736；context-reset硬失败，sparse未训练 | [FC16](TL-A-DEPTH-v5.md) |
| TL-A-TOPO，HB16分支汇合 | sparse128 NLL6.927488，Full/candidate57.28%；同曝光FC4 sparse6.838241；HB16 dense86更新后失败 | [HB16](TL-A-TOPO-v5.md) |
| TL-B-DEPTH，原槽内FC8 | dense4.380228、sparse4.395503；稀疏差+0.015275，单侧95%上界0.027216<0.05；Full减少48.51%，消息减少47.90% | [B FC8](TL-B-DEPTH-v5.md) |
| TL-A-CAPACITY，容量控制 | 浅宽dense56更新后失败；近参数Transformer128/2048分别7.133504/6.346539 | [控制](TL-A-CAPACITY-v5.md) |
| TL-A-NEST，HB40 | 前驱失败后等待器终止，训练0步，没有学习结果 | [未检查范围](TL-A-NEST-v5.md) |

16文档配对bootstrap采用20000次重采样，按整文档、有效token加权。区间只覆盖这些文档/权重条件下的描述性不确定性，没有多重比较校正，也不覆盖训练种子或预训练污染。完整数值、逐阶段来源hash与Lens映射见[终态证据](../evidence/language-depth-terminal.json)，可重算：

```sh
python scripts/review_language_depth_results.py
```

旧A FC4 512也已完成：[单独审阅](TL-A-LM-v4.md)。其测试文档与v5不同，不混用NLL；旧权重只有在v5新文档复评后才进入上表的同集比较。

## 六轴验收与复现

| 轴 | 已观察事实与限制 |
| --- | --- |
| 执行 | FC16 dense训练/普通test、小型推理完成，context-reset失败；HB16 sparse、B两候选、独立Transformer完成；dense HB16/浅宽失败；FC16 sparse/HB40未训练 |
| 严格数值 | 原core atol1e-6/rtol1e-5，Qwen1e-5/1e-4保持。训练后FC16/HB16 sparse小型core分别92/31个元素超差；B core终点此例通过，但完整/cache分别14569/18154元素超差。早先准入有限失败不撤销，未知原因保留；下溢是独立的正支持硬失败 |
| 离散语义 | 已完成候选的小型支持/未来VJP/因果/跨区域梯度检查通过，B原MLP调用0；不覆盖原失败轨迹的正支持资格或所有输入 |
| 恢复 | 改变路径的B2/L3三步新进程精确恢复通过；本次9份终点checkpoint文件hash匹配（含旧v4四份），这只是文件完整性，未新增张量解码或128/512步独立不中断等价试验 |
| 学习 | 独立test与控制、上下文干预、训练/验证下降分别记录；B局部加深略改善旧FC4但仍弱于简单MLP；A深层暂无质量收益 |
| 性能 | CPU FC16/HB16 sparse采样约21.07/19.29秒每步；共享NPU B dense/sparse约35.86/27.85秒。争用、后端与规模不同，不能当独占速度比；实际Full/消息减少不要求同比吞吐提升 |

数学0770593838a37c790a568226930567ed60f99b83，安装foundation85d161d5c9148eac2b164f89a0ba4204b4331898，tidegraph0.3.0。依赖身份见[安装锁](../../dependencies/installation.lock.json)；本次59个固定输入和248个安装文件全部hash一致。各作业冻结commit/树hash和配置在[生命周期快照](../evidence/language-depth-handoff.json)及终态证据中；没有修改冻结源码、依赖、数据或原运行记录。

Lens项目`tide-learning-lab`：本次v5与旧A合计57份已有投影的manifest/run身份、verified/readback_matches及事件数核对一致。旧v4两路另有36份核对，包含与57份重叠的旧A，不能相加。主要曲线为`tl-lm-a-hb16-sparse-v5-train-tide-sparse-s11`、`tl-lm-b-fc8-v5-budget2-train-tide-{dense,sparse}-s11`、`tl-lm-a-transformer-cpu-v5-train-transformer-s11`；详见[分类/操作页](language-depth-v5-running.md)，本机私有入口`.local/LENS.md`。

## 后续候选，尚未启动

1. 先对上游正式交付的固定提交建立新的私有安装锁，检查实际失败行、归一化聚合、VJP、数学零与状态恢复；建议≤0.5 CPU worker小时。用户正在上游处理需求，本次只看到未提交改动，不将其视为交付或已验收。优先保持现有模型配方评估基座修复；共同有界评分子族降为备选，不再作为自动首选。
2. 补齐现有表缺口：独立FC16 quad不依赖Shannon修复，可优先CPU；dense HB16及FC16失败干预依赖新基座验收。建议整组≤4计算worker小时，重新命名与冻结，保持旧失败和旧锁，不静默恢复成另一模型。先短校准再决定能否在原22:49:13前完成；预算余量本身不是自动启动新批次的理由。
3. 配对完整后再选一个语言配方问题。HB16从64→128的NLL仍下降0.219397，先考虑验证集上的曝光/优化是否足够；不立即扩HB40、语义组合或多层Qwen替换。若需要下一批训练，另交有限研究表与预算，不能拿已观察确认集反复调参。

本次仅审阅和记录，没有启动、迁移或恢复训练；这批限定结论待用户验收。
