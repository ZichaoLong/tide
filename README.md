# TIDE Learning Lab

这是`ZichaoLong/tide`仓库的独立`tide-learning-lab`分支，负责core-5的训练/推理研究、
实验对照、证据和资源决策。公共实现由固定版本的`graph-execution-foundation`提供。
本分支使用独立文件树；不复制基座核心，也不把基座旧性能结果当作新模型效果。

**核心目标：同一模型的自然语言next-token质量接近合理调参的对照，同时推进更多数据、更大容量、更深依赖、更低活跃比例和有界局部交互。**
“可以学习”是起点；主要投入应走向联合目标的实质实验。稠密计算后遮罩若更快，可以继续用于训练；逻辑活跃比例、实际计算工作和硬件吞吐分别报告。
长期目标、用户反馈及推进原则统一见[研究章程](docs/RESEARCH_CHARTER.md)，重新接续时先读此页。

当前direct-v8的220次尝试已终态：194完整1M结果、26原失败，完成有限种子复核和Transformer调参。
[广覆盖结果](docs/studies/direct-v8-results.md) · [HB结果变化为何不能归因于基座算错](docs/studies/hb-evidence-reconciliation.md) ·
[下一批数据/容量/深度/活跃比例建议](docs/studies/language-scale-activity-v9-proposal.md)。v9已按[固定执行表](docs/studies/language-scale-activity-v9.md)执行：128M独立训练池，72项筛选完成，58条规模/活动/入口树主轨迹正在推进；共同48h，无NPU/CPU聚合算时上限。
[11月12日中期结果与AICPU/AICore观测](docs/studies/direct-v9-interim-20261112.md)保留质量进展、稀疏代价与硬件效率的区别。
[HB部分接收、记忆更新及10%/3%方案对齐](docs/studies/hb-low-activity-alignment.md)区分接收、选择和物理执行。
[设计JSON](research/language-structure-plan.json)仍不可直接启动，实际历史配置在`configs/studies/language/`。

- [打开研究图](web/index.html)：离线HTML，默认当前联合主线，可切换历史证据；支持详情、搜索、分类、邻接聚焦和缩放。
- [图的读法与维护](docs/research-map.md) / [原型验证](docs/verification.md)
- [当前研究设计](docs/research-design.md)
- [当前交接](docs/STATUS.md) / [后台操作与恢复](docs/RUNNING.md) / [待办](docs/ROADMAP.md)
- [基座来源](dependencies/foundation-source.json) / [实际安装锁](dependencies/foundation.lock.json) / [环境身份](dependencies/environment.lock.json)
- [正支持补正验收](docs/studies/positive-support-acceptance-results.md) / [四种数值方案](docs/studies/shannon-numerical-options.md)：当前选择74e7bbd、新私有环境；旧运行与严格失败保留。
- [HB直接图成本实测](docs/studies/hb-cost-results-20261111.md) / [HB学习结果与后继建议](docs/studies/hb-learning-results-20261111.md) / [共享模型当前路径](dependencies/public-model-paths.json)

直接用浏览器打开`web/index.html`，不需要安装Python包、网络/CDN或运行模型。
GitHub文件页只展示源码；可下载本分支后本地打开。重新生成与检查只需Python标准库：

```sh
python scripts/render_map.py
python scripts/render_map.py --check
python -m unittest discover -s tests -p test_research_map.py -v
```

图的数据是`research/map.json`。实验/结果节点的状态由记录显式给出；
“待运行”“待填结果”不会被呈现为通过。已有基座证据、独立接入检查和经审阅的先导结果分别限定范围；严格失败和未覆盖项一同保留。

用户已授权分阶段实验、私有环境和普通推送；具体范围以当前研究表和STATUS为准。
历史v6旧配方HB扩大未改善；v7简化直接图在双seed/4M出现HB40→88容量收益；v8补回局部语义后尚未复跑HB88。
三轮函数、配方和曝光不同，不能合并为一次修复或普遍scaling结论。旧准入失败、累计扣账及固定作业输入均保留。

可选的真实浏览器检查需要本机已有Firefox与支持实验WebSocket的Node20+，不会下载依赖：

```sh
node --experimental-websocket tests/browser_smoke.mjs
```

该检查限时40秒、使用临时私有profile并停止完整浏览器进程组；截图写入忽略的`artifacts/`。
这是图形入口验证，不是任何训练/推理资格。
