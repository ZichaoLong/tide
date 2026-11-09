# TL-A-CAPACITY：浅宽与普通语言模型控制

问题：深层TIDE的观察来自结构、参数量还是训练曝光？浅宽FC4 width64有616741可训练参数；Transformer为2 block、width96、4头、FFN512，670592参数，与FC16接近。均B8/L32/seed11；浅宽目标128步，Transformer2048并保留128前缀，后者采用2048日程。

浅宽dense只完成56次更新，下一次forward出现正支持下溢，原共享NPU作业失败。没有正式终点test，不能据失败说浅宽学不会。原链后续Transformer未训练，改为原候选独立CPU1作业，2048更新及test/推理均完成。

新Transformer同v5文档test128为**7.133504**、2048为**6.346539**；旧更小Transformer2048在这些文档复评为6.245490。新深层TIDE128好于新Transformer128，却弱于额外16倍曝光Transformer2048；128前缀日程不同且仍欠训练，不能拿它证明架构优势，也不能拿2048终点作等token比较。更大Transformer并未自动改善旧参照，规模、日程和优化仍需区分。

新Transformer context-reset增加1.020985 NLL，而HB16 sparse为0.057520；这是不同已训练模型的干预敏感性，不是孤立的上下文能力排名。普通推理/因果与三步微型恢复通过，长期独立恢复未查。CPU训练约96.98秒，记录用于研究预算，不与共享NPU时间拼成后端加速结论。

建议：保留现有128前缀及额外曝光参照；上游验收后补浅宽控制，再决定一个验证集驱动的曝光/优化问题，不新增宽度网格。迁移须匹配token、日程、词表和训练充分程度。

[控制配置](../../configs/studies/language/a-controls-v5.json) · [独立Transformer配置](../../configs/studies/language/a-transformer-cpu-v5.json) · [证据/统计](../evidence/language-depth-terminal.json)
