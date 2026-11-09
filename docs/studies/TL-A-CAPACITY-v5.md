# TL-A-CAPACITY：浅宽对照补齐（2026-11-09）

**浅宽FC4完整学习与评价已完成；当前点估计略优于窄FC4，但不足以确认改善。** 原FC4 width64/616741可训练参数，CPU1/FP32、B8/L32、seed11、128更新/32677训练目标、512步日程前缀。从旧NPU最后完整32步checkpoint保留参数/optimizer/RNG/采样进度，迁入74e7bbd显式positive-floor-v1在CPU继续96步；这是新身份的状态转换，不称为跨补丁/后端精确恢复。原56步后失败与重算费用保持。

同v5 16文档/3878目标，浅宽NLL6.824360061，原width32 FC4 dense为6.842736169；差−0.018376108，逐文档95%区间[−0.039897,0.003801]，包含0。它也给出低于FC16/HB16 dense终点的点估计，但参数、宽度和拓扑同时改变，不能据此证明宽度优于深度或否定scaling。

验证NLL从原初始化8.365799到6.651351；本次续训起点7.120655。context-reset6.866836，增加0.042475；原重置定义/位置混杂保持。Full15616、无内部消息；本候选只有dense，没有浅宽稀疏配对或跨区域局部通信结论。

完整训练/test/推理/reset completed且后代清空，终点checkpoint hash通过。补正训练累计10752补值坐标（96更新，含replay），test3379、reset2420；这是数值补正实际发生的计数，不代表唯一活跃节点或补值有效贡献。终点core38元素有限超差，离散/独立因果/分块及三步同栈恢复通过；长期独立恢复未查。校准CPU约5.47秒/步、共享NPU约14.38，因此选择CPU；不作通用后端性能资格。

普通LM参照仍为原2block/width96/FFN512 Transformer，同文档128/2048分别7.133504/6.346539。128日程/预热不同，2048有16倍曝光，均不可用来宣称同预算架构优势。HB40现已完成；下一步选择有限的充分训练/优化对齐；本次不扩宽度网格。

[补齐配置](../../configs/studies/language/completion-v5/wide-dense.json) · [原始状态/统计/检查/Lens](../evidence/language-completion-terminal.json) · [运行与复现](language-completion-v5-running.md)

全批已终态，完整对照、数值失败、资源与下一批选项见[补齐总审阅](language-completion-v5-results.md)。

---

以下保留补齐前的历史结果，当前结论以上方为准。

# TL-A-CAPACITY：浅宽与普通语言模型控制

问题：深层TIDE的观察来自结构、参数量还是训练曝光？浅宽FC4 width64有616741可训练参数；Transformer为2 block、width96、4头、FFN512，670592参数，与FC16接近。均B8/L32/seed11；浅宽目标128步，Transformer2048并保留128前缀，后者采用2048日程。

浅宽dense只完成56次更新，下一次forward出现正支持下溢，原共享NPU作业失败。没有正式终点test，不能据失败说浅宽学不会。原链后续Transformer未训练，改为原候选独立CPU1作业，2048更新及test/推理均完成。

新Transformer同v5文档test128为**7.133504**、2048为**6.346539**；旧更小Transformer2048在这些文档复评为6.245490。新深层TIDE128好于新Transformer128，却弱于额外16倍曝光Transformer2048；128前缀日程不同且仍欠训练，不能拿它证明架构优势，也不能拿2048终点作等token比较。更大Transformer并未自动改善旧参照，规模、日程和优化仍需区分。

新Transformer context-reset增加1.020985 NLL，而HB16 sparse为0.057520；这是不同已训练模型的干预敏感性，不是孤立的上下文能力排名。普通推理/因果与三步微型恢复通过，长期独立恢复未查。CPU训练约96.98秒，记录用于研究预算，不与共享NPU时间拼成后端加速结论。

建议：保留现有128前缀及额外曝光参照；上游验收后补浅宽控制，再决定一个验证集驱动的曝光/优化问题，不新增宽度网格。迁移须匹配token、日程、词表和训练充分程度。

[控制配置](../../configs/studies/language/a-controls-v5.json) · [独立Transformer配置](../../configs/studies/language/a-transformer-cpu-v5.json) · [证据/统计](../evidence/language-depth-terminal.json)
