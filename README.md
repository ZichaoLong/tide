# TIDE Learning Lab

这是`ZichaoLong/tide`仓库的独立`tide-learning-lab`分支，负责core-5的训练/推理研究、
实验对照、证据和资源决策。公共实现由固定版本的`graph-execution-foundation`提供。
本分支使用独立文件树；不复制基座核心，也不把基座旧性能结果当作新模型效果。

当前重点：同一模型的语言next-token质量与规模化、稀疏化、局部化。v6的200个配置已终态；已完成[HB学习原因审阅与直接实现成本对照](docs/studies/hb-cost-and-learning-20261111.md)，具体状态见STATUS。[设计JSON](research/language-structure-plan.json)仍不可直接启动，实际配置在`configs/studies/language/`。

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

用户已授权分阶段实验、私有环境和普通推送；具体范围以当前研究表和STATUS为准。历史v3七候选的[终态报告](docs/studies/language-v3-results.md)保留小曝光结论；v6增加了数据、参数和结构规模，HB扩大在已测曝光内未显示收益，原因需进一步区分。旧准入失败、累计扣账及固定作业输入均保留。

可选的真实浏览器检查需要本机已有Firefox与支持实验WebSocket的Node20+，不会下载依赖：

```sh
node --experimental-websocket tests/browser_smoke.mjs
```

该检查限时40秒、使用临时私有profile并停止完整浏览器进程组；截图写入忽略的`artifacts/`。
这是图形入口验证，不是任何训练/推理资格。
