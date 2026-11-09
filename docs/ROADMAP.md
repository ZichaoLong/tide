# 当前研究优先级

用户已批准[语言v4研究表](studies/language-confirmation-v4.md)，配置[fc-confirm-v4.json](../configs/studies/language/fc-confirm-v4.json)。联合重点仍是自然语言next-token质量与实际稀疏、规模和局部化；吞吐比例单列为后续工程优化。状态见STATUS。B同瓶颈对照已终态并形成[限定结论](studies/TL-B-LM-v4.md)；A稀疏及条件FC8继续原固定链，不新增任务。

1. 冻结未观察的16份确认文档，保留训练/验证/词表。
2. A固定FC4 dense/sparse及Transformer，B8/L32/512更新；另用2048更新Transformer提供明确额外曝光的参考。
3. B仅增加同32维入口、近参数量稠密分支，128更新；旧权重在新文档复评以区分瓶颈解释。
4. 若A验证学习与预算前提成立，检查/校准FC8并做128更新配对，比较本次FC4轨迹前128步，相同512步LR日程。
5. 终态按六轴审阅，逐组一页结论及Lens/研究图关联；不按退出码宣布科学支持，不自动加seed22。

新批最多2NPU/8NPUh/12h、8计算线程/48GiB，CPU-only2h，新增8GiB，配额余80GiB。A总7.5h、B0.5h；条件scale≤4h来自A余量，不相加；截止北京时间2026-11-09 18:47:01。旧批3.3307NPUh保留。

下一批再依证据选择独立seed、更长上下文/训练、新容量或公共执行优化。本批不扩层、不做所有语义组合、不申请远端GPU、不开发agent全套基准或公共站点。公共HB全位置成本缺口仍见[TL-REQ-002](requests/TL-REQ-002.md)，未交付且只阻塞依赖它的工作。

历史[v3报告](studies/language-v3-results.md)的单seed小曝光折衷、欠训练Transformer、B简单MLP更好、严格有限失败全部保留；合成检索/旧10%混合结果不升级为语言或规模证明。
