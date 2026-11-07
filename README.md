# TIDE Learning Lab

这是`ZichaoLong/tide`仓库的独立`tide-learning-lab`分支，负责core-5的训练/推理研究、
实验对照、证据和资源决策。公共实现由固定版本的`graph-execution-foundation`提供。
本分支使用独立文件树；不复制基座核心，也不把基座旧性能结果当作新模型效果。

当前阶段：研究对齐与可浏览图形原型。新实验均未运行，可靠模型基线尚未选定。

- [打开研究图](web/index.html)：离线HTML，节点/关系点击查看、搜索、分类、邻接聚焦和缩放。
- [图的读法与维护](docs/research-map.md)
- [研究设计对齐稿](docs/research-design.md)
- [当前交接](docs/STATUS.md) / [待办](docs/ROADMAP.md)
- [基座来源选择](dependencies/foundation-source.json)：尚非安装锁。

直接用浏览器打开`web/index.html`，不需要安装Python包、网络/CDN或运行模型。
GitHub文件页只展示源码；可下载本分支后本地打开。重新生成与检查只需Python标准库：

```sh
python scripts/render_map.py
python scripts/render_map.py --check
python -m unittest discover -s tests -v
```

图的数据是`research/map.json`。实验/结果节点的状态由记录显式给出；
“待运行”“待填结果”不会被呈现为通过。图中仅有一项已引用的基座小型证据，
其严格失败和未覆盖范围一同保留，它不认证本分支的训练效果。

研究实施需继续与发起者对齐基线、任务、对照、预算和停止条件，再统一授权。

可选的真实浏览器检查需要本机已有Firefox与支持实验WebSocket的Node20+，不会下载依赖：

```sh
node --experimental-websocket tests/browser_smoke.mjs
```

该检查限时40秒、使用临时私有profile并停止完整浏览器进程组；截图写入忽略的`artifacts/`。
这是图形入口验证，不是任何训练/推理资格。
